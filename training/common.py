"""Faust preparation primitives: no model import or network access on import."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'faust-lora-prep-0.1.0'


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{i}: 非法 JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{i}: 必须是对象')
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def validate_data(root: Path = ROOT) -> dict:
    """All splits may be HASHED here. Only train/validation enter the trainer."""
    lock = read_json(root / 'training/data_lock.json')
    for rel, expected in lock['files'].items():
        path = root / rel
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f'数据版本不匹配：{rel}。停止；不要自动修改校验值。')
    pairs = read_rows(root / lock['pairs'])
    master = {r['segment_id']: r for r in read_rows(root / lock['master'])}
    if len(pairs) != 195 or len(master) != 195:
        raise ValueError('应为 195 条管理数据，而不是 195 条训练数据')
    seen, groups, targets = set(), {}, {}
    counts = {}
    for split, count in lock['counts'].items():
        selected = [r for r in pairs if r['split'] == split]
        exported = read_rows(root / f'Faust_Corpus/splits/sft_v1/{split}.jsonl')
        expected = [{'prompt': r['instruction_de'], 'completion': r['target_text']} for r in selected]
        if len(selected) != count or expected != exported:
            raise ValueError(f'{split} 导出或数量不一致')
        for r in selected:
            sid, group, target = r['segment_id'], r['scene_group_id'], r['target_text']
            if sid in seen or not r['instruction_de'].strip() or not target.strip():
                raise ValueError(f'重复或空记录：{sid}')
            seen.add(sid)
            if master[sid]['target_text'] != target or master[sid]['split'] != split:
                raise ValueError(f'原文或划分不一致：{sid}')
            if hashlib.sha256(target.encode()).hexdigest() != r['target_text_sha256']:
                raise ValueError(f'目标哈希不一致：{sid}')
            if groups.setdefault(group, split) != split:
                raise ValueError(f'场景组泄漏：{group}')
            if targets.setdefault(target, split) != split:
                raise ValueError(f'跨集合目标重复：{sid}')
        counts[split] = len(selected)
    if seen != set(master):
        raise ValueError('主数据覆盖不一致')
    return {'ok': True, 'release': lock['data_release'], 'counts': counts,
            'data_fingerprint': digest(lock['files']), 'source_commit': lock['source_commit']}


def records(split: str, root: Path = ROOT) -> list[dict]:
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('未知集合')
    return [r for r in read_rows(root / 'Faust_Corpus/data/sft_v1/faust_sft_pairs.jsonl') if r['split'] == split]


def format_prompt(instruction: str, reference: str | None = None) -> str:
    """This BASE model uses one identical plain-text format in training/generation."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('指令不能为空')
    prefix = ''
    if reference:
        prefix = '### Vergleichstext (nicht abschreiben)\n' + reference + '\n\n'
    return prefix + '### Aufgabe\n' + instruction + '\n\n### Textauszug\n'


def encode_pair(tokenizer: Any, instruction: str, target: str, maximum: int) -> dict:
    if not isinstance(target, str) or not target.strip():
        raise ValueError('目标文本为空')
    if tokenizer.eos_token_id is None:
        raise ValueError('分词器必须定义 EOS')
    # Deliberately concatenate separately encoded fields. Inference uses the
    # SAME prompt encoding, so a BPE merge across the boundary cannot shift masks.
    p = tokenizer.encode(format_prompt(instruction), add_special_tokens=False)
    c = tokenizer.encode(target, add_special_tokens=False)
    ids = p + c + [tokenizer.eos_token_id]
    if len(ids) > maximum:
        raise ValueError(f'完整样本需 {len(ids)} tokens，超过 {maximum}；未截断任何文字。')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(p) + c + [tokenizer.eos_token_id]}


def pad_features(features: list[dict], pad_id: int) -> dict:
    if not features:
        raise ValueError('空批次')
    size = max(len(x['input_ids']) for x in features)
    batch = {k: [] for k in ('input_ids', 'attention_mask', 'labels')}
    for x in features:
        n = len(x['input_ids'])
        if n != len(x['labels']) or n != len(x['attention_mask']):
            raise ValueError('input/mask/label 长度不等')
        batch['input_ids'].append(x['input_ids'] + [pad_id] * (size - n))
        batch['attention_mask'].append(x['attention_mask'] + [0] * (size - n))
        # Do not mask by token identity: PAD and EOS can have the same ID.
        batch['labels'].append(x['labels'] + [-100] * (size - n))
    return batch


class CompletionCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        return {k: torch.tensor(v, dtype=torch.long) for k, v in pad_features(features, self.pad_id).items()}


def encode_split(tokenizer: Any, rows: list[dict], maximum: int) -> tuple[list[dict], list[dict]]:
    encoded, lengths = [], []
    for r in rows:
        try:
            item = encode_pair(tokenizer, r['instruction_de'], r['target_text'], maximum)
        except ValueError as exc:
            raise ValueError(f'{r["segment_id"]}: {exc}') from exc
        target_n = sum(x != -100 for x in item['labels'])
        encoded.append(item)
        lengths.append({'segment_id': r['segment_id'], 'total_tokens': len(item['input_ids']),
                        'completion_tokens_including_eos': target_n,
                        'prompt_tokens': len(item['input_ids']) - target_n})
    return encoded, lengths


def latest_checkpoint(output: Path) -> Path | None:
    valid = []
    for p in output.glob('checkpoint-*'):
        if re.fullmatch(r'checkpoint-\d+', p.name) and all((p / f).is_file() for f in
                ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json', 'adapter_model.safetensors')):
            try:
                step = int(p.name.split('-')[1])
                if read_json(p / 'trainer_state.json')['global_step'] != step:
                    continue
                valid.append((step, p))
            except (KeyError, ValueError):
                continue
    return max(valid, default=(0, None))[1]


def pick_reference(case: dict, root: Path = ROOT) -> dict:
    """Optional ablation: deterministic form matching, TRAIN targets only."""
    candidates = records('train', root)
    matches = [r for r in candidates if r['scene_specification']['form_de'] == case.get('form_de')]
    candidates = matches or candidates
    return min(candidates, key=lambda r: (abs(len(r['target_text']) - 1500), r['segment_id']))
