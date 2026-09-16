#!/usr/bin/env python3
"""Paired, local-only generation using the trained quantized base, LoRA off/on."""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from core import (ROOT, HERE, MODES, check_inputs, check_adapter, select_cases, digest,
                  sha, text_sha, read_json, rows, write_json, run_paths, expected_keys,
                  check_record, export_results)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_protocol(cfg, record, cases, subset, root=ROOT):
    code = {p.name: sha(p) for p in sorted((Path(root) / 'eval/ab_v1').glob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.sh', '.ps1')
            and p.name not in ('verification.json', 'test_eval.py')}
    return {'release': cfg['release'], 'subset': subset, 'configuration': cfg,
            'model_record': record, 'cases_sha256': digest(cases), 'code_sha256': code,
            'prompt_source': 'training/local8gb/core.py::format_prompt',
            'baseline': 'same pinned NF4 base with PEFT adapter disabled; NOT unquantized base',
            'retrieval_used': False, 'corpus_targets_supplied_to_generation': False}


def require_budget(prompt_ids, maximum_new, context):
    if not prompt_ids or len(prompt_ids) + maximum_new > context:
        raise ValueError(f'输入{len(prompt_ids)}+输出预算{maximum_new}超出上下文{context}；未截断。')


def check_layer_state(layers, disabled):
    if not layers or any(getattr(layer, 'merged_adapters', []) for layer in layers):
        raise ValueError('Adapter layers missing or merged; cannot compare base fairly')
    if any(bool(layer.disable_adapters) != disabled for layer in layers):
        raise ValueError('Adapter disable/enable state does not match requested condition')


@contextlib.contextmanager
def condition(model, layers, mode):
    if mode not in MODES:
        raise ValueError('Unknown generation condition')
    with model.disable_adapter() if mode == 'base' else contextlib.nullcontext():
        check_layer_state(layers, mode == 'base')
        yield
    check_layer_state(layers, False)


def load_runtime(record, adapter):
    # Import order matters for Unsloth patches. No dependency installs here.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from unsloth import FastLanguageModel
    import torch
    from transformers import AutoTokenizer, GenerationConfig, set_seed
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    if platform.system() != 'Linux' or not torch.cuda.is_available():
        raise RuntimeError('需要已训练模型的Linux/WSL2 CUDA环境；本程序不租GPU。')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('请设置CUDA_VISIBLE_DEVICES=0，只使用一张卡。')
    packages = {n: importlib.metadata.version(n) for n in
                ('torch', 'unsloth', 'unsloth_zoo', 'transformers', 'peft', 'accelerate',
                 'bitsandbytes', 'triton', 'tokenizers', 'safetensors', 'huggingface_hub')}
    if packages['unsloth'] != record['runtime']['unsloth'] or packages['transformers'] != '4.57.6':
        raise RuntimeError('请复用训练时的独立环境；不要为评测自动升级依赖。')
    dtype = record['runtime']['dtype']
    if dtype != 'bfloat16' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('本基线固定训练时的bfloat16；不会自动换精度。')
    with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        names = list(f.keys())
        if len(names) != record['adapter']['tensor_count'] or any('lora_' not in n for n in names):
            raise ValueError('Unexpected adapter tensors')
        if any(not torch.isfinite(f.get_tensor(n)).all().item() for n in names):
            raise ValueError('Non-finite adapter weight')
    model_info = record['model']
    try:
        snapshot = snapshot_download(model_info['id'], revision=model_info['revision'], local_files_only=True)
    except Exception as exc:
        raise RuntimeError('未找到训练时固定revision的本地HF缓存。请使用原WSL用户和HF_HOME；本入口不自动下载替代模型。') from exc
    base_cfg = read_json(Path(snapshot) / 'config.json')
    quant = base_cfg.get('quantization_config', {})
    if (base_cfg.get('model_type') != 'qwen3' or quant.get('bnb_4bit_quant_type') != 'nf4'
            or not quant.get('load_in_4bit', quant.get('_load_in_4bit', False))):
        raise ValueError('Cached model is not the pinned NF4 Qwen3 base')
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                        trust_remote_code=False, use_fast=True)
    if tok.eos_token_id is None:
        raise ValueError('Missing EOS token')
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    tokenizer_fingerprint = digest(tok.backend_tokenizer.to_str())
    preflight_path = adapter.parent.parent / 'preflight.json'
    preflight_checked = False
    if preflight_path.is_file():
        preflight = read_json(preflight_path)
        if preflight['tokenizer_sha256'] != tokenizer_fingerprint:
            raise ValueError('Tokenizer differs from training preflight')
        preflight_checked = True
    model, _ = FastLanguageModel.from_pretrained(model_name=snapshot,
        max_seq_length=record['data']['effective_sequence_length'], dtype=torch.bfloat16,
        load_in_4bit=True, fast_inference=False, device_map={'': 0}, trust_remote_code=False,
        local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    check_layer_state(layers, False)
    if len(layers) * 2 != record['adapter']['tensor_count']:
        raise ValueError('LoRA layer count differs from recorded adapter')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    env = {'packages': packages, 'python': platform.python_version(), 'platform': platform.platform(),
           'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'dtype': dtype,
           'tokenizer_sha256': tokenizer_fingerprint, 'training_tokenizer_receipt_checked': preflight_checked,
           'context': record['data']['effective_sequence_length'],
           'reproducibility': 'paired seeds and pinned inputs; bitwise portability is not guaranteed'}
    return model, tok, layers, torch, GenerationConfig, set_seed, env


def generation_config(cfg, tok, cls, smoke=False):
    values = dict(cfg['generation'])
    if smoke:
        values['max_new_tokens'] = cfg['smoke_max_new_tokens']
    return cls(**values, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
               bos_token_id=tok.bos_token_id, use_cache=True, return_dict_in_generate=False,
               output_scores=False, forced_eos_token_id=None)


def probe_switch(model, tok, layers, torch, ids):
    x = torch.tensor([ids[:32]], dtype=torch.long, device='cuda')
    results = []
    with torch.inference_mode():
        for mode in ('lora', 'base', 'lora'):
            with condition(model, layers, mode):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
                last = logits[0, -1].float().cpu()
                if not torch.isfinite(last).all().item():
                    raise ValueError('Non-finite probe logits')
                results.append(last)
                del logits
    delta = (results[0] - results[1]).abs().max().item()
    restored = torch.allclose(results[0], results[2], rtol=1e-4, atol=1e-4)
    if delta <= 1e-7 or not restored:
        raise ValueError('Adapter switch probe failed: no effect or restoration mismatch')
    return {'finite': True, 'enabled_minus_disabled_max_abs': delta,
            'enabled_state_restored': restored, 'checked_layers': len(layers),
            'not_a_literary_quality_test': True}


def generate_one(model, tok, layers, torch, set_seed, ids, gen, mode, seed):
    set_seed(seed)
    x = torch.tensor([ids], dtype=torch.long, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), condition(model, layers, mode):
        output = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gen)
    new_ids = output[0, len(ids):].tolist()
    if not new_ids or len(new_ids) > gen.max_new_tokens:
        raise ValueError('Unexpected generation length')
    reason = 'eos' if new_ids[-1] == tok.eos_token_id else 'length'
    if reason == 'length' and len(new_ids) != gen.max_new_tokens:
        raise ValueError('Unexpected early stop; refusing ambiguous result')
    result = {'text': tok.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
              'new_token_ids': new_ids, 'generated_tokens': len(new_ids), 'finish_reason': reason,
              'elapsed_seconds': time.perf_counter() - started,
              'peak_allocated_vram_gib': torch.cuda.max_memory_allocated() / 2**30}
    del x, output
    return result


def execute(args, cfg, record, cases, training, protocol, folder):
    adapter = (args.adapter or ROOT / record['adapter']['relative_path']).resolve()
    artifact = check_adapter(adapter, record)
    model, tok, layers, torch, GenConfig, set_seed, env = load_runtime(record, adapter)
    prompts = {c['case_id']: tok.encode(training.format_prompt(c['instruction_de']),
                                      add_special_tokens=False) for c in cases}
    for ids in prompts.values():
        require_budget(ids, cfg['generation']['max_new_tokens'], env['context'])
    gen = generation_config(cfg, tok, GenConfig)
    signature = digest({'protocol': protocol, 'artifact': artifact, 'environment': env,
                        'generation_config': gen.to_dict(), 'prompt_ids': prompts})
    path = folder / 'manifest.json'
    old = read_json(path) if path.exists() else None
    if old and old['signature'] != signature:
        raise ValueError('配置、输入或环境已变化。请保留旧结果，使用新的--run-id。')
    if not old and any((folder / 'records').glob('*.json')):
        raise ValueError('Saved records have no owning manifest')
    try:
        commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    manifest = old or {'signature': signature, 'protocol': protocol, 'artifact': artifact,
        'environment': env, 'generation_config': gen.to_dict(), 'git_commit': commit,
        'started_at_utc': now(), 'expected_generations': len(expected_keys(cases, cfg['seeds'])),
        'inference_only': True, 'training_invoked': False, 'network_download_enabled': False}
    manifest.update(status='running', last_started_at_utc=now())
    write_json(path, manifest)
    try:
        print('检查adapter开关、logits及短生成……', flush=True)
        first = cases[0]
        gate = probe_switch(model, tok, layers, torch, prompts[first['case_id']])
        short = generation_config(cfg, tok, GenConfig, smoke=True)
        previews = {}
        for mode in MODES:
            previews[mode] = generate_one(model, tok, layers, torch, set_seed,
                                          prompts[first['case_id']], short, mode, cfg['seeds'][0])
            if not previews[mode]['text'].strip():
                raise ValueError(f'{mode}: smoke generation was empty')
        write_json(folder / 'smoke.json', {'signature': signature, 'status': 'passed',
                                          'probe': gate, 'previews': previews, 'excluded_from_evaluation': True})
        if args.command == 'smoke':
            manifest['status'] = 'smoke_passed'
            write_json(path, manifest)
            print('短生成通过；未执行正式场景评测。', flush=True)
            return
        total = manifest['expected_generations']
        done = 0
        for c in cases:
            ids = prompts[c['case_id']]
            for seed in cfg['seeds']:
                for mode in MODES:
                    key = f'{c["case_id"]}-{seed}-{mode}'
                    out = folder / 'records' / (key + '.json')
                    if out.exists():
                        check_record(read_json(out), key, signature, c, seed, mode, ids)
                    else:
                        print(f'生成 {done + 1}/{total}: {key}', flush=True)
                        value = generate_one(model, tok, layers, torch, set_seed, ids, gen, mode, seed)
                        value.update(key=key, signature=signature, case_id=c['case_id'], seed=seed,
                                     mode=mode, prompt_ids_sha256=digest(ids), prompt_tokens=len(ids),
                                     text_sha256=text_sha(value['text']), created_at_utc=now())
                        write_json(out, value)
                    done += 1
        actual = {p.stem for p in (folder / 'records').glob('*.json')}
        if actual != set(expected_keys(cases, cfg['seeds'])):
            raise ValueError('Incomplete or unexpected candidate set')
        manifest.update(status='completed', completed_at_utc=now(), completed_generations=done)
        write_json(path, manifest)
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error_type=type(exc).__name__, error=str(exc)[:1500], stopped_at_utc=now())
        write_json(path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'smoke', 'run', 'all', 'export'])
    parser.add_argument('--set', choices=['dev', 'holdout'], default='dev')
    parser.add_argument('--allow-holdout', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--data-only', action='store_true', help='check inputs without local adapter (check only)')
    args = parser.parse_args()
    if args.data_only and args.command != 'check':
        parser.error('--data-only is only valid with check')
    cfg, record, all_cases, training, data = check_inputs()
    cases = select_cases(all_cases, args.set, args.allow_holdout)
    protocol = source_protocol(cfg, record, cases, args.set)
    name = args.run_id or ('r8-' + args.set + '-v1')
    folder, public = run_paths(name)
    if args.command == 'check':
        info = {'dataset': data, 'cases': len(cases), 'expected_generations': len(cases)*len(cfg['seeds'])*2,
                'gpu_generation_tested': False}
        info['adapter'] = ('not_checked_data_only' if args.data_only else
                          check_adapter(args.adapter or ROOT / record['adapter']['relative_path'], record))
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder / 'session.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('该评测运行正在进行，不能重复启动。') from exc
        manifest_path = folder / 'manifest.json'
        old = read_json(manifest_path) if manifest_path.exists() else None
        if old and old['protocol'] != protocol:
            raise ValueError('本run-id已绑定其他配置/代码/题目；请指定新的run-id。')
        if args.command != 'export' and not (old and old.get('status') == 'completed'):
            execute(args, cfg, record, cases, training, protocol, folder)
        if args.command in ('all', 'run', 'export'):
            result = export_results(folder, public, cases, cfg,
                                    rows(ROOT / 'Faust_Corpus/data/master_segments.jsonl'))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print('请分享blind_review.zip；不要先上传runs中的解盲与原始生成记录。')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已停止；完整候选已保存，使用同一命令可续跑。', file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        print('评测停止；未重新训练、未截断输入、未更换模型或生成参数。', file=sys.stderr)
        sys.exit(1)
