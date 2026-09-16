"""Offline data/masking/identity helpers. Importing this module never loads a model."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RELEASE = 'faust-local8gb-0.2.0'
COUNTS = {'train': 158, 'validation': 18, 'test': 19}
PAIR_PATH = 'Faust_Corpus/data/sft_v1/faust_sft_pairs.jsonl'
MASTER_PATH = 'Faust_Corpus/data/master_segments.jsonl'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    result = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{i}: expected JSON object')
            result.append(row)
    return result


def validate_data(root: Path = ROOT) -> dict:
    lock = read_json(root / 'training/local8gb/data_lock.json')
    for rel, expected in lock['files'].items():
        if not (root / rel).is_file() or sha(root / rel) != expected:
            raise ValueError(f'数据不一致：{rel}。请停止，不要改校验值绕过检查。')
    rows = read_rows(root / PAIR_PATH)
    master_rows = read_rows(root / MASTER_PATH)
    master = {r['segment_id']: r for r in master_rows}
    if len(master_rows) != 195 or len(master) != 195 or len(rows) != 195:
        raise ValueError('全量管理数据应有195个唯一片段；训练集只有158条。')
    seen, groups, targets = set(), {}, {}
    for split, expected_n in COUNTS.items():
        part = [r for r in rows if r['split'] == split]
        if len(part) != expected_n:
            raise ValueError(f'{split} 数量错误')
        exports = read_rows(root / f'Faust_Corpus/splits/sft_v1/{split}.jsonl')
        if exports != [{'prompt': r['instruction_de'], 'completion': r['target_text']} for r in part]:
            raise ValueError(f'{split} 导出与主表不一致')
        for row in part:
            sid, group, text = row['segment_id'], row['scene_group_id'], row['target_text']
            if sid in seen or sid not in master or not row['instruction_de'].strip() or not text.strip():
                raise ValueError(f'空白、重复或未知片段：{sid}')
            seen.add(sid)
            if master[sid]['target_text'] != text or master[sid]['split'] != split:
                raise ValueError(f'原文或划分变化：{sid}')
            if hashlib.sha256(text.encode('utf-8')).hexdigest() != row['target_text_sha256']:
                raise ValueError(f'目标文本哈希错误：{sid}')
            if groups.setdefault(group, split) != split or targets.setdefault(text, split) != split:
                raise ValueError('跨集合场景或原文重复')
    if seen != set(master):
        raise ValueError('覆盖不完整')
    return {'ok': True, 'counts': COUNTS, 'data_fingerprint': fingerprint(lock['files']),
            'test_read_for_integrity_only': True}


def training_rows(split: str, root: Path = ROOT) -> list[dict]:
    if split not in {'train', 'validation'}:
        raise ValueError('本机训练入口禁止装入test；test只作文件完整性检查。')
    return [r for r in read_rows(root / PAIR_PATH) if r['split'] == split]


def format_prompt(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('说明不能为空')
    return '### Aufgabe\n' + instruction + '\n\n### Textauszug\n'


def encode_pair(tok: Any, row: dict) -> tuple[dict, dict]:
    """Separate-field BPE; same prefix encoding used for inference. No truncation."""
    if tok.eos_token_id is None or not row['target_text'].strip():
        raise ValueError('缺少EOS或目标正文')
    p = tok.encode(format_prompt(row['instruction_de']), add_special_tokens=False)
    c = tok.encode(row['target_text'], add_special_tokens=False) + [tok.eos_token_id]
    data = {'input_ids': p + c, 'attention_mask': [1] * (len(p) + len(c)),
            'labels': [-100] * len(p) + c}
    report = {'segment_id': row['segment_id'], 'total_tokens': len(p) + len(c),
              'prompt_tokens': len(p), 'completion_tokens': len(c)}
    return data, report


def budget(lengths: list[dict], limit: int) -> int:
    if not lengths or limit < 64:
        raise ValueError('非法长度预算')
    maximum = max(r['total_tokens'] for r in lengths)
    if maximum > limit:
        bad = [r['segment_id'] for r in lengths if r['total_tokens'] > limit]
        raise ValueError(f'完整样本最大{maximum} tokens，超过上限{limit}：{bad}。原文未截断。')
    return min(limit, math.ceil(maximum / 64) * 64)


def pad_features(features: list[dict], pad_id: int) -> dict:
    if not features:
        raise ValueError('空批次')
    n = max(len(x['input_ids']) for x in features)
    result = {k: [] for k in ('input_ids', 'attention_mask', 'labels')}
    for x in features:
        size = len(x['input_ids'])
        if size != len(x['labels']) or size != len(x['attention_mask']):
            raise ValueError('mask/label长度不同')
        for key, pad in [('input_ids', pad_id), ('attention_mask', 0), ('labels', -100)]:
            result[key].append(x[key] + [pad] * (n - size))
    return result


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        return {k: torch.tensor(v, dtype=torch.long) for k, v in pad_features(features, self.pad_id).items()}


def latest_checkpoint(folder: Path) -> Path | None:
    candidates = []
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                'adapter_config.json', 'adapter_model.safetensors']
    for p in folder.glob('checkpoint-*'):
        if re.fullmatch(r'checkpoint-\d+', p.name) and all((p / f).is_file() for f in required):
            try:
                step = int(p.name.split('-')[1])
                if read_json(p / 'trainer_state.json')['global_step'] == step:
                    candidates.append((step, p))
            except (KeyError, ValueError):
                continue
    return max(candidates, default=(0, None))[1]


def safe_run_dir(name: str, root: Path = ROOT) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,90}', name):
        raise ValueError('run-id只能由字母数字和._-组成')
    result = (root / 'runs' / name).resolve()
    if (root / 'runs').resolve() not in result.parents or (root / 'runs').is_symlink():
        raise ValueError('拒绝通过符号链接将输出写到仓库以外')
    return result


def check_receipt(receipt: dict, signature: str) -> None:
    if receipt.get('status') != 'passed' or receipt.get('signature') != signature:
        raise ValueError('缺少本配置的成功显存试跑；先执行smoke或完整启动入口。')


def finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f'{label}非有限，停止训练')
    return value
