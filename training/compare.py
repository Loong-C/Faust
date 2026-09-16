#!/usr/bin/env python3
"""Generate matched base/LoRA samples. Default: 3 new dev scenes, no test targets."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import re
import time
from common import ROOT, format_prompt, pick_reference, read_json, read_rows, records, validate_data, write_json


def grams(text: str, n: int = 8) -> set[tuple]:
    words = re.findall(r"[\wÄÖÜäöüß]+", text.casefold())
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path(os.environ.get('FAUST_OUTPUT', ROOT / 'runs/faust-lora-v0.1.0')))
    p.add_argument('--limit', type=int, default=3, help='0 means every case in the chosen dev/holdout set')
    p.add_argument('--set', choices=['dev', 'holdout'], default='dev')
    p.add_argument('--reference', action='store_true', help='also add base+reference / LoRA+reference conditions')
    p.add_argument('--max-new-tokens', type=int, default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[20260916])
    args = p.parse_args()
    data = validate_data()
    manifest = read_json(args.run / 'run_manifest.json')
    if manifest.get('status') != 'completed' or manifest.get('smoke_test'):
        raise ValueError('请选择完整正式训练，不可使用 smoke 模型')
    if data['data_fingerprint'] != manifest['data']['data_fingerprint']:
        raise ValueError('数据与训练记录不一致')
    config = manifest['config']
    cases = [r for r in read_rows(ROOT / 'eval/original_scenes.jsonl') if r['set'] == args.set]
    if args.limit < 0:
        raise ValueError('limit 不能为负数')
    if args.limit:
        cases = cases[:args.limit]
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel
    from train import environment, tokenizer_for
    env = environment()
    dtype = getattr(torch, env['compute_dtype'])
    tok = tokenizer_for(config, manifest['model_revision'])
    base = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=manifest['model_revision'],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
        use_safetensors=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, args.run / 'adapter')
    model.eval()
    model.config.use_cache = True
    n_new = args.max_new_tokens or config['max_new_tokens']
    if not 1 <= n_new <= 4096:
        raise ValueError('max-new-tokens 必须在 1–4096 之间')
    source_grams = set().union(*(grams(r['target_text']) for r in records('train')))
    out = args.run / 'comparisons' / (time.strftime('%Y%m%d-%H%M%S') + '-' + args.set)
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for case in cases:
        ref = pick_reference(case) if args.reference else None
        for with_ref in ([False, True] if ref else [False]):
            prompt = format_prompt(case['instruction_de'], ref['target_text'] if with_ref else None)
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) + n_new > model.config.max_position_embeddings:
                raise ValueError('生成上下文超长；没有自动裁掉参考文本')
            inputs = {'input_ids': torch.tensor([ids], device=model.device),
                      'attention_mask': torch.ones((1, len(ids)), dtype=torch.long, device=model.device)}
            for seed in args.seeds:
                for condition in ('base', 'lora'):
                    set_seed(seed)
                    context = model.disable_adapter() if condition == 'base' else contextlib.nullcontext()
                    with context, torch.inference_mode():
                        output = model.generate(**inputs, do_sample=True, temperature=config['temperature'],
                            top_p=config['top_p'], repetition_penalty=1.0, max_new_tokens=n_new,
                            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                    generated_ids = output[0, len(ids):].tolist()
                    text = tok.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    g = grams(text)
                    row = {'case_id': case['case_id'], 'set': args.set, 'seed': seed,
                           'condition': condition + ('+reference' if with_ref else ''),
                           'reference_segment_id': ref['segment_id'] if with_ref else None,
                           'prompt': prompt, 'text': text, 'generated_tokens': len(generated_ids),
                           'hit_max_new_tokens': len(generated_ids) == n_new and generated_ids[-1] != tok.eos_token_id,
                           'train_8gram_overlap': len(g & source_grams) / max(1, len(g))}
                    results.append(row)
                    # Flush after each sample so interruption does not erase all outputs.
                    with (out / 'generations.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(case['case_id'], row['condition'], row['generated_tokens'], flush=True)
    blind, key = [], []
    rng = random.Random(61427)
    for case in cases:
        for seed in args.seeds:
            rows = [r for r in results if r['case_id'] == case['case_id'] and r['seed'] == seed]
            rng.shuffle(rows)
            blind.append(f"情景 {case['case_id']}，种子 {seed}\n{case['description_zh']}\n")
            for i, row in enumerate(rows):
                label = chr(65 + i)
                blind.append(f"候选 {label}\n{row['text']}\n")
                key.append({'case_id': case['case_id'], 'seed': seed, 'label': label, 'condition': row['condition']})
            blind.append('记录：更符合情景的是？人物声音是否不同？哪里解释过多或像套话？可并列，不强选。\n' + '='*60)
    (out / 'blind_review.txt').write_text('\n\n'.join(blind), encoding='utf-8')
    write_json(out / 'blind_key.json', key)
    write_json(out / 'generation_manifest.json', {'training_signature': manifest['signature'], 'environment': env,
        'config': config, 'cases': [c['case_id'] for c in cases], 'set': args.set, 'seeds': args.seeds,
        'max_new_tokens': n_new, 'reference': args.reference, 'test_corpus_not_loaded_as_generation_targets': True,
        'warning': '8-gram overlap is only a reuse diagnostic, not a style/quality detector.'})
    print(f'比较文件：{out}。中文情景可读；德文候选可交回本会话解释，不要只凭验证损失认定文学质量。')


if __name__ == '__main__':
    main()
