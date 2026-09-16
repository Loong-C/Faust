"""Offline evaluation helpers. No model imports, network, training or API calls."""
from __future__ import annotations
import collections
import csv
import hashlib
import hmac
import importlib.util
import json
import math
import re
import secrets
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODES = ('base', 'lora')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    out = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{number}: expected an object')
            out.append(value)
    return out


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def git_blob(path):
    # Git text blobs are LF; tolerate Windows checkout CRLF for metadata/code only.
    # Frozen corpus files are still checked by raw SHA-256 in validate_data.
    b = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()


def write_text(path, text):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing symlink: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if tmp.is_symlink():
        raise ValueError(f'Refusing symlink: {tmp}')
    tmp.write_text(text, encoding='utf-8', newline='\n')
    tmp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_rows(path, values):
    write_text(path, ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def safe_folder(root, relative):
    root = Path(root).resolve()
    path = root
    for part in Path(relative).parts:
        if part in ('..', '') or Path(part).is_absolute():
            raise ValueError('Unsafe output path')
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Refusing symlink in output path: {path}')
    if root not in path.resolve().parents:
        raise ValueError('Output must be below repository root')
    return path


def run_paths(name, root=ROOT):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,70}', name):
        raise ValueError('run-id只能使用字母、数字、._-，不能包含路径。')
    return (safe_folder(root, 'runs/eval-' + name),
            safe_folder(root, 'eval/results/' + name))


def load_training_core(root=ROOT):
    path = Path(root) / 'training/local8gb/core.py'
    spec = importlib.util.spec_from_file_location('faust_training_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_cases(cases, cfg):
    seen = set()
    for c in cases:
        if not re.fullmatch(r'NEW_\d{2}', c.get('case_id', '')) or c['case_id'] in seen:
            raise ValueError('Missing or duplicate case ID')
        seen.add(c['case_id'])
        if c.get('set') not in cfg['expected_cases'] or c.get('target_text') is not None:
            raise ValueError('Invalid case set or unexpected target in generation prompt')
        if c.get('fictional') is not True or not c.get('instruction_de', '').strip():
            raise ValueError('Expected fictional case with a nonempty German instruction')
        if not c.get('description_zh', '').strip():
            raise ValueError('Missing Chinese case description')
    if dict(collections.Counter(c['set'] for c in cases)) != cfg['expected_cases']:
        raise ValueError('Original case counts changed')


def check_inputs(root=ROOT):
    root = Path(root)
    cfg = read_json(root / 'eval/ab_v1/config.json')
    for path, expected in cfg['locked_git_blobs'].items():
        if git_blob(root / path) != expected:
            raise ValueError(f'冻结输入发生变化：{path}。停止，不改校验值绕过。')
    training = load_training_core(root)
    data = training.validate_data(root)
    record = read_json(root / cfg['model_record'])
    if record['status'] != 'completed' or data['data_fingerprint'] != record['data']['fingerprint']:
        raise ValueError('Training record or dataset fingerprint mismatch')
    cases = rows(root / cfg['cases_file'])
    check_cases(cases, cfg)
    return cfg, record, cases, training, data


def select_cases(cases, subset, allow_holdout=False):
    if subset not in ('dev', 'holdout'):
        raise ValueError('Unknown subset')
    if subset == 'holdout' and not allow_holdout:
        raise ValueError('保留集需显式指定--allow-holdout；先固定配置再评测。')
    return [c for c in cases if c['set'] == subset]


def check_adapter(path, record):
    path = Path(path).resolve()
    weight = path / 'adapter_model.safetensors'
    if not weight.is_file() or not (path / 'adapter_config.json').is_file():
        raise ValueError(f'未找到正式adapter：{path}。切换分支不会下载本机权重。')
    if weight.stat().st_size != record['adapter']['bytes'] or sha(weight) != record['adapter']['sha256']:
        raise ValueError('adapter大小或SHA-256与已登记的训练结果不符')
    cfg = read_json(path / 'adapter_config.json')
    expected = record['model']
    if (cfg.get('peft_type') != 'LORA' or cfg.get('task_type') != 'CAUSAL_LM'
            or cfg.get('base_model_name_or_path') != expected['id']
            or cfg.get('revision') != expected['revision']
            or cfg.get('r') != expected['rank'] or cfg.get('lora_alpha') != expected['alpha']):
        raise ValueError('adapter类型、底模revision或rank/alpha不符')
    if cfg.get('bias', 'none') != 'none' or cfg.get('modules_to_save'):
        raise ValueError('关闭adapter不能恢复纯底模：存在额外训练参数')
    modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if set(cfg.get('target_modules', [])) != modules or any(cfg.get(k) for k in
            ('rank_pattern', 'alpha_pattern', 'use_dora', 'use_rslora', 'fan_in_fan_out')):
        raise ValueError('Adapter module selection/scaling differs from the registered training recipe')
    return {'weights_sha256': sha(weight), 'config_sha256': sha(path / 'adapter_config.json')}


def expected_keys(cases, seeds):
    return [f'{c["case_id"]}-{seed}-{mode}' for c in cases for seed in seeds for mode in MODES]


def check_record(row, key, signature, case, seed, mode, prompt_ids=None):
    if (row.get('key') != key or row.get('signature') != signature
            or row.get('case_id') != case['case_id'] or row.get('seed') != seed
            or row.get('mode') != mode or not isinstance(row.get('text'), str) or row.get('text_sha256') != text_sha(row['text'])):
        raise ValueError('Existing generation is corrupt or belongs to another run')
    ids = row.get('new_token_ids')
    if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('Invalid saved token IDs')
    if row.get('generated_tokens') != len(ids) or row.get('finish_reason') not in ('eos', 'length'):
        raise ValueError('Invalid completion metadata')
    if prompt_ids is not None and row.get('prompt_ids_sha256') != digest(prompt_ids):
        raise ValueError('Prompt tokenization changed on resume')


def content_words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


class CopyIndex:
    """Exact longest contiguous normalized-word matches, never across scenes.

    Segment boundaries WITHIN the same scene remain searchable. This is an
    overlap diagnostic, not proof of plagiarism or training-set memorization.
    """
    def __init__(self, segments):
        self.words, self.owners = [], []
        self.positions = collections.defaultdict(list)
        previous = None
        for s in segments:
            group = (s['work_id'], s['scene_id'], s['split'])
            if previous != group and self.words:
                self.words.append('\0')
                self.owners.append(None)
            owner = {'segment_id': s['segment_id'], 'scene_id': s['scene_id'], 'split': s['split']}
            for word in content_words(s['target_text']):
                self.positions[word].append(len(self.words))
                self.words.append(word)
                self.owners.append(owner)
            previous = group

    def compare(self, text):
        query = content_words(text)
        previous, best = {}, {}
        hits = {8: 0, 12: 0, 16: 0}
        for qi, word in enumerate(query):
            current, longest = {}, 0
            for pos in self.positions.get(word, []):
                length = previous.get(pos - 1, 0) + 1
                current[pos] = length
                longest = max(longest, length)
                split = self.owners[pos]['split']
                for scope in ('all', split):
                    if length > best.get(scope, {}).get('words', 0):
                        best[scope] = {'words': length, 'generated_word_start': qi - length + 1,
                            'source_start': self.owners[pos - length + 1],
                            'source_end': self.owners[pos],
                            'excerpt_normalized': ' '.join(query[max(0, qi - length + 1):qi + 1][-32:])}
            previous = current
            for n in hits:
                hits[n] += int(longest >= n)
        return {'word_count': len(query), 'longest_match': best,
                'matched_ngram_fraction': {str(n): hits[n] / max(1, len(query) - n + 1) for n in hits}}


def text_diagnostics(text):
    words = content_words(text)
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return {'empty': not text.strip(), 'nonempty_lines': len(lines), 'word_count': len(words),
            'repeated_4gram_fraction': 1 - len(set(grams)) / len(grams) if grams else 0,
            'repeated_line_fraction': 1 - len(set(lines)) / len(lines) if lines else 0,
            'template_marker_in_output': any(x in text for x in ('### Aufgabe', '### Textauszug', '<|im_start|>'))}


def make_blind(generations, cases, seeds, secret):
    by_key = {r['key']: r for r in generations}
    if len(by_key) != len(generations) or set(by_key) != set(expected_keys(cases, seeds)):
        raise ValueError('Cannot export blind review: missing, duplicate or extra generation')
    blind, reveal = [], []
    for c in cases:
        for seed in seeds:
            key = f'{c["case_id"]}-{seed}'
            token = hmac.new(bytes.fromhex(secret), key.encode(), hashlib.sha256).hexdigest()
            order = MODES if int(token[-1], 16) % 2 else MODES[::-1]
            pair_id = 'P_' + token[:16]
            row = {'pair_id': pair_id, 'case_id': c['case_id'], 'description_zh': c['description_zh'],
                   'instruction_de': c['instruction_de']}
            mapping = {'pair_id': pair_id, 'case_id': c['case_id'], 'seed': seed}
            for label, mode in zip(('A', 'B'), order):
                g = by_key[f'{key}-{mode}']
                row[label] = {'text': g['text'], 'finish_reason': g['finish_reason']}
                mapping[label] = mode
            blind.append(row)
            reveal.append(mapping)
    blind.sort(key=lambda r: r['pair_id'])
    return blind, reveal


def export_results(folder, public, cases, cfg, segments):
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('status') != 'completed':
        raise ValueError('Run is not complete; no partial blind export')
    signature = manifest['signature']
    generations = []
    for c in cases:
        for seed in cfg['seeds']:
            for mode in MODES:
                key = f'{c["case_id"]}-{seed}-{mode}'
                row = read_json(folder / 'records' / (key + '.json'))
                check_record(row, key, signature, c, seed, mode)
                generations.append(row)
    secret_path = folder / 'private' / 'blinding_secret.json'
    if not secret_path.exists():
        write_json(secret_path, {'secret': secrets.token_hex(32)})
    secret = read_json(secret_path)['secret']
    blind, reveal = make_blind(generations, cases, cfg['seeds'], secret)
    index = CopyIndex(segments)
    diagnostics = []
    for g in generations:
        match = index.compare(g['text'])
        diagnostics.append({'key': g['key'], 'case_id': g['case_id'], 'mode': g['mode'],
            'finish_reason': g['finish_reason'], **text_diagnostics(g['text']), 'source_overlap': match,
            'overlap_review_flag': match['longest_match'].get('all', {}).get('words', 0) >= cfg['copy_warning_min_words']})
    write_rows(folder / 'raw_generations.jsonl', generations)
    write_rows(folder / 'diagnostics.jsonl', diagnostics)
    write_json(folder / 'private/reveal.json', reveal)
    write_json(folder / 'summary.json', {
        'generation_count': len(generations), 'pair_count': len(blind),
        'finish_reasons': dict(collections.Counter(g['finish_reason'] for g in generations)),
        'overlap_warning_count': sum(d['overlap_review_flag'] for d in diagnostics),
        'quality_winner': None, 'note_zh': '只报告运行与复述诊断，不用自动分数代替文学判断。'})
    allowed = {'blind_review.jsonl', 'blind_review.txt', 'review_scores.csv', 'README_盲评.txt', 'bundle_manifest.json', 'blind_review.zip'}
    public.mkdir(parents=True, exist_ok=True)
    if any(p.name not in allowed for p in public.iterdir()):
        raise ValueError('Public result directory has unknown files; refusing to overwrite')
    lock = public / 'bundle_manifest.json'
    if lock.exists():
        old = read_json(lock)
        if old['run_signature'] != signature:
            raise ValueError('Public directory belongs to another run')
        for name, checksum in old['files'].items():
            p = public / name
            if p.exists() and sha(p) != checksum:
                raise ValueError('评阅文件已被修改，保留你的意见，拒绝覆盖：' + name)
    elif any(public.iterdir()):
        raise ValueError('Nonempty public directory has no ownership manifest')
    write_rows(public / 'blind_review.jsonl', blind)
    preview = '匿名德文原始输出；中文只解释题目，不是译文。A/B在每组内独立打乱。\n'
    for r in blind:
        preview += f'\n{r["pair_id"]} | {r["case_id"]}\n{r["description_zh"]}\n'
        for label in ('A', 'B'):
            preview += f'\n版本 {label}（停止原因：{r[label]["finish_reason"]}）\n{r[label]["text"]}\n'
    write_text(public / 'blind_review.txt', preview)
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['pair_id', 'case_id', 'content', 'voices', 'language', 'overall', 'notes_zh'])
    writer.writeheader()
    writer.writerows({'pair_id': r['pair_id'], 'case_id': r['case_id']} for r in blind)
    write_text(public / 'review_scores.csv', '\ufeff' + stream.getvalue())
    write_text(public / 'README_盲评.txt',
        '这是盲评包，不含模型身份、解盲密钥或训练权重。可直接上传整个ZIP让助手阅读全文。\n'
        '请先比较内容遵循、人物声音、德语表达与整体偏好；引用具体原句说明依据。\n'
        'CSV填写A、B、tie、both_bad或uncertain；不能判断时留空，不强选。\n'
        '同一场景的三个种子是重复测量，不应当作三篇独立题目计算显著性。\n'
        'length表示达到输出上限，不能假称完整收尾；eos也不保证戏剧自然结束。\n'
        '请勿先读取runs目录中的raw_generations、diagnostics、manifest或private/reveal。\n'
        '中文题目说明原样保留；如有歧义，以德语instruction为准。不自动翻译生成文。\n')
    share = ['blind_review.jsonl', 'blind_review.txt', 'review_scores.csv', 'README_盲评.txt']
    write_json(lock, {'run_signature': signature, 'pair_count': len(blind),
                     'files': {name: sha(public / name) for name in share}})
    with zipfile.ZipFile(public / 'blind_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in share:
            z.write(public / name, name)
    return {'pairs': len(blind), 'public_bundle': str(public / 'blind_review.zip')}
