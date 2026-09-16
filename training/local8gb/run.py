#!/usr/bin/env python3
"""Local single-GPU QLoRA with checked data, no truncation, and a smoke-test gate."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from core import (ROOT, RELEASE, Collator, budget, check_receipt, encode_pair, finite,
                  fingerprint, format_prompt, latest_checkpoint, read_json,
                  safe_run_dir, sha, training_rows, validate_data, write_json)

PACKAGES = ('unsloth', 'unsloth_zoo', 'torch', 'transformers', 'peft', 'accelerate',
            'bitsandbytes', 'triton', 'datasets', 'tokenizers')


def config_for(profile: str) -> dict:
    cfg = read_json(Path(__file__).with_name('config.json'))
    if profile == 'lean':
        cfg.update(rank=8, alpha=16)
    return cfg


def versions() -> dict:
    return {name: metadata.version(name) for name in PACKAGES}


def doctor(cfg: dict) -> dict:
    import torch
    if platform.system() != 'Linux':
        raise RuntimeError('此入口只支持Linux/WSL2。Windows请双击Run_Faust_8GB.cmd。')
    if not torch.cuda.is_available():
        raise RuntimeError('未检测到CUDA。Windows先安装/更新NVIDIA驱动，并确认WSL2中nvidia-smi可用。')
    if int(os.getenv('WORLD_SIZE', '1')) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError('此入口使用一张卡；多卡请设置CUDA_VISIBLE_DEVICES=0，不使用torchrun。')
    prop = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    if prop.major < 7 or total < 7 * 2**30:
        raise RuntimeError('此8B入口要求支持CUDA的约8GB或更大NVIDIA GPU。')
    if free < cfg['minimum_free_vram_gib'] * 2**30:
        raise RuntimeError(f'当前空闲显存只有{free / 2**30:.2f} GiB；关闭游戏、本地模型等再重试。')
    stack = versions()
    if stack['transformers'] != '4.57.6' or stack['unsloth'] != '2026.8.12':
        raise RuntimeError('环境与本入口固定版本不同；请使用start.sh建立独立环境。')
    mem = {}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemTotal:', 'MemAvailable:')):
                mem[line.split(':')[0]] = round(int(line.split()[1]) / 1024**2, 2)
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'gpu': prop.name, 'total_vram_gib': total / 2**30, 'free_vram_gib': free / 2**30,
              'cuda': torch.version.cuda, 'packages': stack, 'ram_gib': mem,
              'dtype': 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if mem.get('MemAvailable', 20) < 10:
        print('提示：Linux可用内存少于10GiB。建议物理内存32GB；WSL默认限额可能需要调整。', flush=True)
    return result


def prepare(cfg: dict, folder: Path) -> tuple:
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    data = validate_data()
    folder.mkdir(parents=True, exist_ok=True)
    pin_path = folder / 'model_pin.json'
    if pin_path.exists():
        pin = read_json(pin_path)
        if pin['model_id'] != cfg['model_id']:
            raise ValueError('输出目录已固定其他模型；请指定新的--run-id。')
    else:
        revision = HfApi().model_info(cfg['model_id'], revision=cfg['model_revision']).sha
        if not revision or not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('无法固定模型commit')
        pin = {'model_id': cfg['model_id'], 'revision': revision}
        write_json(pin_path, pin)
    tok = AutoTokenizer.from_pretrained(pin['model_id'], revision=pin['revision'],
                                       trust_remote_code=False, use_fast=True)
    if tok.eos_token_id is None:
        raise ValueError('分词器缺少EOS')
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    encoded, lengths = {}, {}
    for split in ('train', 'validation'):
        pairs = [encode_pair(tok, r) for r in training_rows(split)]
        encoded[split] = [x[0] for x in pairs]
        lengths[split] = [x[1] for x in pairs]
    effective = budget(lengths['train'] + lengths['validation'], cfg['sequence_limit'])
    prep = {'data': data, 'pin': pin, 'config': cfg, 'effective_sequence_length': effective,
            'lengths': lengths, 'tokenizer_sha256': fingerprint(tok.backend_tokenizer.to_str()),
            'truncated_samples': 0, 'test_used_for_training_or_selection': False}
    write_json(folder / 'preflight.json', prep)
    print(f'实际token检查完成：train={len(encoded["train"])}，validation={len(encoded["validation"])}，'
          f'最长={max(x["total_tokens"] for s in lengths.values() for x in s)}；'
          f'运行上下文={effective}；未截断。', flush=True)
    return tok, encoded, prep


def runtime():
    # Must precede transformers/peft imports in this process.
    from unsloth import FastLanguageModel
    import torch
    from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
    return FastLanguageModel, torch, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed


def load_base(fast, torch, cfg: dict, prep: dict, dtype: str):
    from huggingface_hub import snapshot_download
    pin = prep['pin']
    snapshot = snapshot_download(repo_id=pin['model_id'], revision=pin['revision'],
        allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', '*.tiktoken', '*.txt'])
    model_config = read_json(Path(snapshot) / 'config.json')
    if model_config.get('model_type') != 'qwen3':
        raise ValueError('模型不是预期的Qwen3')
    quant = model_config.get('quantization_config', {})
    if not quant.get('load_in_4bit', quant.get('_load_in_4bit', False)):
        raise ValueError('不是已量化的4bit快照；停止，避免误载16bit模型。')
    if quant.get('bnb_4bit_quant_type') != 'nf4':
        raise ValueError('量化格式不是NF4')
    model, tok = fast.from_pretrained(model_name=snapshot,
        max_seq_length=prep['effective_sequence_length'], dtype=getattr(torch, dtype),
        load_in_4bit=True, fast_inference=False, device_map={'': 0}, trust_remote_code=False)
    model.config.use_cache = False
    return model


def attach_lora(fast, model, cfg: dict):
    model = fast.get_peft_model(model, r=cfg['rank'], lora_alpha=cfg['alpha'],
        target_modules=cfg['target_modules'], lora_dropout=0, bias='none',
        use_gradient_checkpointing='unsloth', random_state=cfg['seed'])
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, p in trainable):
        raise ValueError('不是LoRA-only训练；停止')
    model.print_trainable_parameters()
    return model


def identity(cfg: dict, prep: dict, env: dict) -> str:
    return fingerprint({'config': cfg, 'data': prep['data']['data_fingerprint'],
        'pin': prep['pin'], 'tokenizer': prep['tokenizer_sha256'],
        'length': prep['effective_sequence_length'], 'packages': env['packages'],
        'python': env['python'], 'cuda': env['cuda'], 'gpu': env['gpu'], 'dtype': env['dtype'],
        'code': {f: sha(Path(__file__).with_name(f)) for f in ('run.py', 'core.py')}})


def git_revision() -> str | None:
    try:
        p = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    except FileNotFoundError:
        return None


def portable_adapter(folder: Path, prep: dict) -> None:
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json'):
        if not (folder / name).is_file():
            raise ValueError(f'保存结果不完整：{name}')
    config = read_json(folder / 'adapter_config.json')
    config['base_model_name_or_path'] = prep['pin']['model_id']
    config['revision'] = prep['pin']['revision']
    write_json(folder / 'adapter_config.json', config)


def execute(cfg: dict, folder: Path, smoke: bool) -> None:
    fast, torch, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed = runtime()
    env = doctor(cfg)
    set_seed(cfg['seed'])
    tok, encoded, prep = prepare(cfg, folder)
    signature = identity(cfg, prep, env)
    output = folder / (f'smoke-{time.time_ns()}' if smoke else 'train')
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / 'smoke_latest.json'
    if not smoke:
        if not receipt_path.exists():
            raise ValueError('请先试跑：start.sh smoke 或 start.sh all。')
        check_receipt(read_json(receipt_path), signature)
    checkpoint = None
    manifest_path = output / 'run_manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['signature'] != signature:
            raise ValueError('旧运行与代码/环境/数据不一致，拒绝覆盖。请指定新--run-id。')
        if old['status'] == 'completed':
            print(f'此运行已完成：{output / "adapter"}', flush=True)
            return
        checkpoint = latest_checkpoint(output)
        if checkpoint is None:
            if any(output.glob('checkpoint-*')) or (output / 'adapter').exists():
                raise ValueError('旧运行只有不完整权重；请保留日志并使用新的--run-id。')
            write_json(output / f'previous_attempt-{time.time_ns()}.json', old)
            print('此前没有产生检查点；保留失败记录，从原始底模重新开始。', flush=True)
    elif any(output.iterdir()):
        raise ValueError('输出目录不是空目录，且没有运行manifest；拒绝覆盖。')
    manifest = {'release': RELEASE, 'signature': signature, 'config': cfg, 'environment': env,
        'model_pin': prep['pin'], 'data': prep['data'], 'git_commit': git_revision(),
        'status': 'running', 'smoke_only': smoke, 'started': time.time(),
        'resume_checkpoint': str(checkpoint) if checkpoint else None,
        'loss': 'per-example completion+EOS mean; prompt/pad labels=-100',
        'test_used': False, 'effective_sequence_length': prep['effective_sequence_length']}
    write_json(manifest_path, manifest)
    if smoke:
        write_json(receipt_path, {'status': 'running', 'signature': signature})
    (output / 'environment_freeze.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model = attach_lora(fast, load_base(fast, torch, cfg, prep, env['dtype']), cfg)
        train_set, val_set = encoded['train'], encoded['validation']
        accumulation = cfg['gradient_accumulation_steps']
        if smoke:
            # Both real longest samples are tested; validation NEVER enters backprop.
            ordered = sorted(train_set, key=lambda x: len(x['input_ids']))
            train_set = [ordered[-1]] * accumulation + [ordered[len(ordered) // 2]] * accumulation
            val_set = sorted(val_set, key=lambda x: len(x['input_ids']), reverse=True)[:2]
        from datasets import Dataset
        class CheckedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                outputs = model(**inputs)
                loss = outputs.loss
                if not torch.isfinite(loss.detach()).all().item():
                    raise FloatingPointError('模型loss非有限，停止')
                return (loss, outputs) if return_outputs else loss

            def _get_train_sampler(self, train_dataset=None):
                if smoke:
                    from torch.utils.data import SequentialSampler
                    return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)
                return super()._get_train_sampler(train_dataset)
        class LogGuard(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                for name in ('loss', 'eval_loss', 'grad_norm'):
                    if logs and name in logs:
                        finite(logs[name], name)
                if logs:
                    with (output / 'progress.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps({'step': state.global_step, **logs}) + '\n')
        args = TrainingArguments(output_dir=str(output),
            num_train_epochs=cfg['epochs'], max_steps=2 if smoke else -1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=accumulation, learning_rate=cfg['learning_rate'],
            optim='adamw_8bit', weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type='linear',
            fp16=env['dtype'] == 'float16', bf16=env['dtype'] == 'bfloat16', tf32=False,
            gradient_checkpointing=False, max_grad_norm=1.0, # already enabled by Unsloth smart checkpointing
            eval_strategy='steps', eval_steps=1 if smoke else cfg['save_steps'],
            save_strategy='steps', save_steps=1 if smoke else cfg['save_steps'],
            save_total_limit=2, save_safetensors=True, load_best_model_at_end=True,
            metric_for_best_model='eval_loss', greater_is_better=False,
            prediction_loss_only=True, logging_steps=1, logging_nan_inf_filter=False,
            report_to=[], push_to_hub=False, dataloader_num_workers=0,
            dataloader_pin_memory=False, remove_unused_columns=False, label_names=['labels'],
            seed=cfg['seed'], data_seed=cfg['seed'])
        callbacks = [LogGuard()]
        if not smoke:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg['early_stopping_patience']))
        trainer = CheckedTrainer(model=model, args=args,
            train_dataset=Dataset.from_list(train_set), eval_dataset=Dataset.from_list(val_set),
            data_collator=Collator(tok.pad_token_id), processing_class=tok, callbacks=callbacks)
        trainer.model_accepts_loss_kwargs = False
        if checkpoint is None:
            write_json(output / 'initial_validation.json', trainer.evaluate())
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        metrics = dict(result.metrics)
        metrics.update(trainer.evaluate())
        finite(metrics['eval_loss'], 'final eval_loss')
        adapter = output / 'adapter'
        trainer.save_model(str(adapter))
        tok.save_pretrained(adapter)
        trainer.save_state()
        portable_adapter(adapter, prep)
        from safetensors import safe_open
        with safe_open(str(adapter / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
            keys = list(weights.keys())
            if not keys or any('lora_' not in k for k in keys):
                raise ValueError('adapter权重内容不符合LoRA-only')
            for key in keys:
                if not torch.isfinite(weights.get_tensor(key)).all().item():
                    raise FloatingPointError('保存的adapter含非有限权重')
        metrics.update(elapsed_seconds=time.perf_counter() - started,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_vram_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(output / 'metrics.json', metrics)
        manifest.update(status='completed', ended=time.time(), optimizer_steps=trainer.state.global_step,
            best_checkpoint=trainer.state.best_model_checkpoint, best_metric=trainer.state.best_metric)
        write_json(manifest_path, manifest)
        if smoke:
            if latest_checkpoint(output) is None:
                raise ValueError('试跑未保存完整优化器检查点')
            write_json(receipt_path, {'status': 'passed', 'signature': signature,
                'path': output.name, 'metrics': metrics, 'adapter_reopened': True,
                'longest_train_tokens': max(r['total_tokens'] for r in prep['lengths']['train']),
                'longest_validation_tokens': max(r['total_tokens'] for r in prep['lengths']['validation']),
                'optimizer_state_saved': latest_checkpoint(output) is not None,
                'smoke_adapter_used_for_real_training': False})
            print('最长训练样本、最长验证样本、梯度累积、优化器更新及保存已通过试跑。', flush=True)
        else:
            print(f'正式训练完成，adapter保存在：{adapter}', flush=True)
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()
        manifest.update(status='failed_or_interrupted', error_type=type(exc).__name__,
                        error=str(exc)[:1500], ended=time.time())
        write_json(manifest_path, manifest)
        if smoke:
            write_json(receipt_path, {'status': 'failed', 'signature': signature, 'path': output.name})
        if oom:
            print('显存不足，已停止；没有删样本或截断原文。可关闭占用后重试，或显式选lean（rank8）。'
                  '若仍不足，保留诊断记录，再考虑更小底模或24GB云GPU。', file=sys.stderr)
        raise


def generate(cfg: dict, folder: Path, instruction_path: Path) -> None:
    fast, torch, *_ = runtime()
    manifest = read_json(folder / 'train/run_manifest.json')
    if manifest['status'] != 'completed' or manifest['smoke_only']:
        raise ValueError('没有正式训练完成的adapter')
    env = doctor(cfg)
    tok, _, prep = prepare(cfg, folder)
    if identity(cfg, prep, env) != manifest['signature']:
        raise ValueError('推理环境或配置与本次训练不一致；先核对运行记录。')
    instruction = instruction_path.read_text(encoding='utf-8').strip()
    ids = tok.encode(format_prompt(instruction), add_special_tokens=False)
    available = prep['effective_sequence_length'] - len(ids)
    if available < 32:
        raise ValueError('输入太长；未截断')
    from peft import PeftModel
    model = PeftModel.from_pretrained(load_base(fast, torch, cfg, prep, env['dtype']),
                                      str(folder / 'train/adapter'), is_trainable=False)
    fast.for_inference(model)
    x = torch.tensor([ids], dtype=torch.long, device='cuda')
    with torch.inference_mode():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x),
            max_new_tokens=min(512, available), do_sample=True, temperature=0.85, top_p=0.9,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    result = tok.decode(y[0, len(ids):], skip_special_tokens=True)
    path = folder / f'preview-{time.time_ns()}.txt'
    path.write_text(result + '\n', encoding='utf-8')
    print(result + '\n\n保存至：' + str(path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'doctor', 'prepare', 'smoke', 'train', 'generate'])
    p.add_argument('--profile', choices=['standard', 'lean'], default='standard')
    p.add_argument('--run-id')
    p.add_argument('--instruction', type=Path, default=Path(__file__).with_name('example_instruction_de.txt'))
    args = p.parse_args()
    cfg = config_for(args.profile)
    folder = safe_run_dir(args.run_id or f'local8gb-r{cfg["rank"]}')
    if args.command == 'check':
        print(json.dumps(validate_data(), ensure_ascii=False, indent=2))
    elif args.command == 'doctor':
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'environment.json', doctor(cfg))
    elif args.command == 'prepare':
        prepare(cfg, folder)
    elif args.command == 'generate':
        generate(cfg, folder, args.instruction)
    else:
        execute(cfg, folder, args.command == 'smoke')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已中止；保留已完成的检查点和日志。')
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
