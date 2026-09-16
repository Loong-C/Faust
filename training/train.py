#!/usr/bin/env python3
"""Single-NVIDIA-GPU, completion-only QLoRA. Never starts a paid service."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from common import (ROOT, VERSION, CompletionCollator, digest, encode_split,
                    latest_checkpoint, read_json, records, sha_file, validate_data, write_json)


def environment(require_gpu: bool = True) -> dict:
    import torch
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'packages': {}}
    for package in ('transformers', 'peft', 'accelerate', 'bitsandbytes', 'tokenizers', 'safetensors'):
        result['packages'][package] = importlib.metadata.version(package)
    required = {'transformers': '4.57.6', 'peft': '0.18.1', 'accelerate': '1.12.0', 'bitsandbytes': '0.49.2'}
    for name, version in required.items():
        if result['packages'][name] != version:
            raise ValueError(f'{name} 版本不符；请通过 training/run.sh 安装固定环境。')
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('未发现 CUDA GPU。Colab 请先选择 GPU；本方案不使用 CPU、TPU 或 Mac MPS 训练。')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('本脚本仅支持单 GPU；不要用 torchrun/多卡启动。')
        prop = torch.cuda.get_device_properties(0)
        result.update(gpu=prop.name, vram_gib=round(prop.total_memory / 2**30, 2),
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      compute_dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16')
        if result['vram_gib'] < 14:
            raise RuntimeError('显存小于 14 GiB；此 8B 配置不适用。建议 24GB 以上 NVIDIA GPU。')
        if result['vram_gib'] < 23:
            print('提示：16GB 卡仅作尝试，不保证容纳最长样本。若试跑内存不足，请换 24GB 以上 GPU；不要截断原文。', flush=True)
    return result


def resolve_revision(config: dict, output: Path) -> str:
    existing = output / 'run_manifest.json'
    if existing.exists():
        old = read_json(existing)
        if old['config'] != config:
            raise ValueError('输出目录属于不同配置；请选择新的 --output，不能覆盖已有实验。')
        return old['model_revision']
    from huggingface_hub import HfApi
    revision = HfApi().model_info(config['model_id'], revision=config['model_revision']).sha
    if not revision or not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('无法固定模型版本；停止而不使用未知的 main 版本。')
    return revision


def tokenizer_for(config: dict, revision: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config['model_id'], revision=revision,
                                       use_fast=True, trust_remote_code=False)
    if tok.eos_token_id is None:
        raise ValueError('底模没有 EOS')
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok


def prepare(config: dict, output: Path) -> tuple:
    data = validate_data()
    revision = resolve_revision(config, output)
    tok = tokenizer_for(config, revision)
    train, tr_lengths = encode_split(tok, records('train'), config['max_length'])
    val, va_lengths = encode_split(tok, records('validation'), config['max_length'])
    report = {'data': data, 'model_id': config['model_id'], 'model_revision': revision,
              'max_length_budget': config['max_length'], 'truncated_samples': 0,
              'tokenizer_sha256': digest(tok.backend_tokenizer.to_str()),
              'prompt_format': 'plain Aufgabe/Textauszug; NO chat template; separate field tokenization',
              'train': tr_lengths, 'validation': va_lengths,
              'test_used_for_training_or_model_selection': False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'preflight.json', report)
    print(json.dumps({'counts': data['counts'], 'max_train_tokens': max(x['total_tokens'] for x in tr_lengths),
                      'max_validation_tokens': max(x['total_tokens'] for x in va_lengths),
                      'truncated': 0, 'revision': revision}, ensure_ascii=False), flush=True)
    return tok, train, val, report


def load_model(config: dict, revision: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    dtype = getattr(torch, dtype_name)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=revision,
        quantization_config=quant, torch_dtype=dtype, device_map={'': 0},
        attn_implementation='sdpa', trust_remote_code=False, use_safetensors=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    lora = LoraConfig(r=config['rank'], lora_alpha=config['alpha'], lora_dropout=config['dropout'],
                      target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('可训练参数不符合 LoRA-only 约束')
    model.print_trainable_parameters()
    return model


def git_state() -> dict:
    def call(*args):
        p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'commit': call('rev-parse', 'HEAD'), 'status': call('status', '--porcelain')}


def train(config: dict, output: Path, smoke: bool, resume: str) -> None:
    import torch
    from transformers import (Trainer, TrainingArguments, EarlyStoppingCallback,
                              TrainerCallback, set_seed)
    env = environment()
    set_seed(config['seed'])
    tok, train_data, val_data, prep = prepare(config, output)
    code = {name: sha_file(ROOT / 'training' / name) for name in ('common.py', 'train.py')}
    signature = digest({'config': config, 'data': prep['data']['data_fingerprint'],
                        'revision': prep['model_revision'], 'tokenizer': prep['tokenizer_sha256'],
                        'code': code, 'dtype': env['compute_dtype'], 'smoke': smoke,
                        'packages': env['packages'], 'torch': env['torch']})
    path = output / 'run_manifest.json'
    checkpoint = None
    if path.exists():
        old = read_json(path)
        if old['signature'] != signature:
            raise ValueError('实验签名不一致；请保留旧目录，为新配置另设 --output。')
        if old.get('status') == 'completed':
            print(f'该运行已经完成：{output}。如需重新实验，请换输出目录。')
            return
        if resume == 'auto':
            checkpoint = latest_checkpoint(output)
            if checkpoint is None and any(output.glob('checkpoint-*')):
                raise ValueError('只有不完整检查点；请使用新的输出目录，旧目录不覆盖。')
        elif resume != 'none':
            checkpoint = Path(resume).resolve()
            if checkpoint.parent != output.resolve() or not all((checkpoint / f).is_file() for f in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors', 'adapter_config.json')):
                raise ValueError('只能恢复同一实验目录下的完整检查点')
        else:
            raise ValueError('输出目录已有实验。--resume none 仅允许新目录。')
    elif any(output.glob('checkpoint-*')) or (output / 'adapter').exists():
        raise ValueError('目录有旧权重但没有实验签名，拒绝覆盖')
    manifest = {'release': VERSION, 'status': 'running', 'smoke_test': smoke,
                'signature': signature, 'config': config, 'model_id': config['model_id'],
                'model_revision': prep['model_revision'], 'data': prep['data'],
                'tokenizer_sha256': prep['tokenizer_sha256'], 'code_sha256': code,
                'environment': env, 'git': git_state(), 'started_unix': time.time(),
                'resume_checkpoint': str(checkpoint) if checkpoint else None,
                'loss': 'completion + EOS only; prompt and padding labels = -100',
                'test_used': False}
    write_json(path, manifest)
    (output / 'pip_freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    model = load_model(config, prep['model_revision'], env['compute_dtype'])
    if smoke:
        # Test the longest train/validation examples, not merely the first two.
        train_data = sorted(train_data, key=lambda x: len(x['input_ids']), reverse=True)[:2]
        val_data = sorted(val_data, key=lambda x: len(x['input_ids']), reverse=True)[:2]
    class FiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ('loss', 'eval_loss'):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f'非有限 {key}；停止并保留日志')
    args = TrainingArguments(output_dir=str(output), num_train_epochs=config['epochs'],
        max_steps=2 if smoke else -1, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'], weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type='linear', max_grad_norm=1.0, optim='adamw_torch',
        fp16=env['compute_dtype'] == 'float16', bf16=env['compute_dtype'] == 'bfloat16',
        tf32=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=1 if smoke else config['eval_save_steps'],
        save_strategy='steps', save_steps=1 if smoke else config['eval_save_steps'],
        save_total_limit=2, save_safetensors=True, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, logging_steps=1,
        logging_nan_inf_filter=False, report_to=[], push_to_hub=False,
        dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'],
        seed=config['seed'], data_seed=config['seed'])
    callbacks = [FiniteLoss()]
    if not smoke:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience']))
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                      data_collator=CompletionCollator(tok.pad_token_id), processing_class=tok,
                      callbacks=callbacks)
    if not checkpoint:
        write_json(output / 'base_validation.json', trainer.evaluate())
    try:
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(output / 'adapter'))
        tok.save_pretrained(output / 'adapter')
        trainer.save_state()
        metrics = dict(result.metrics)
        metrics.update(trainer.evaluate())
        metrics['peak_allocated_vram_gib'] = torch.cuda.max_memory_allocated() / 2**30
        write_json(output / 'metrics.json', metrics)
        manifest.update(status='completed', ended_unix=time.time(),
                        best_checkpoint=trainer.state.best_model_checkpoint,
                        best_validation_loss=trainer.state.best_metric,
                        optimizer_steps=trainer.state.global_step)
        write_json(path, manifest)
        print(f'完成：{output / "adapter"}；试跑模型不能当作正式模型。' if smoke else
              f'训练完成：{output / "adapter"}。请运行 compare.py 做原创场景比较。', flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted_or_failed', error_type=type(exc).__name__)
        write_json(path, manifest)
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            print('显存不足：没有截断或删掉样本。请更换 24GB/更大 GPU；新硬件使用新输出目录。', file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'prepare', 'smoke', 'train'])
    parser.add_argument('--config', type=Path, default=ROOT / 'training/config.json')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('FAUST_OUTPUT', ROOT / 'runs/faust-lora-v0.1.0')))
    parser.add_argument('--resume', default='auto')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(validate_data(), ensure_ascii=False, indent=2))
        return
    config = read_json(args.config)
    output = args.output.resolve()
    # Nothing ever writes into source data or plan/ during GPU training.
    if output == ROOT or any(p in output.parents or p == output for p in (ROOT / 'Faust_Corpus', ROOT / 'plan', ROOT / 'training')):
        raise ValueError('输出不能位于语料、计划或代码目录中')
    if args.command == 'prepare':
        prepare(config, output)
    else:
        train(config, output, args.command == 'smoke', args.resume)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'停止：{exc}', file=sys.stderr)
        raise
