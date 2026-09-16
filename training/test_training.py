"""Offline tests. They do not claim an 8B CUDA run or pretrained tokenizer test."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from common import (ROOT, CompletionCollator, digest, encode_pair, encode_split, format_prompt,
                    latest_checkpoint, pad_features, pick_reference, read_json, read_rows,
                    records, validate_data)

class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(x) + 1 for x in text]

class DataTests(unittest.TestCase):
    def test_frozen_corpus(self):
        self.assertEqual(validate_data()['counts'], {'train':158, 'validation':18, 'test':19})
    def test_test_is_not_training(self):
        train = {r['segment_id'] for r in records('train')}
        self.assertTrue(train.isdisjoint(r['segment_id'] for r in records('test')))
    def test_scene_groups_do_not_cross(self):
        groups = [{r['scene_group_id'] for r in records(s)} for s in ('train','validation','test')]
        self.assertTrue(all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i)))
    def test_tampered_data_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            b=Path(temp); lock=read_json(ROOT/'training/data_lock.json')
            (b/'training').mkdir(); shutil.copy(ROOT/'training/data_lock.json',b/'training/data_lock.json')
            for rel in lock['files']:
                dest=b/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/rel,dest)
            with (b/lock['pairs']).open('a') as f:f.write(' ')
            with self.assertRaisesRegex(ValueError,'版本不匹配'):validate_data(b)
    def test_original_cases(self):
        r=read_rows(ROOT/'eval/original_scenes.jsonl')
        self.assertEqual(len(r),16);self.assertEqual(sum(x['set']=='dev' for x in r),12)
        self.assertEqual(len({x['case_id'] for x in r}),16)
        self.assertTrue(all(x['fictional'] and x['target_text'] is None for x in r))
        originals={x['instruction_de'] for s in ('train','validation','test') for x in records(s)}
        self.assertTrue(all(x['instruction_de'] not in originals for x in r))
    def test_reference_training_only(self):
        for r in read_rows(ROOT/'eval/original_scenes.jsonl'):
            self.assertEqual(pick_reference(r)['split'],'train')
    def test_export_matches_every_pair(self):
        for split in ('train','validation','test'):
            xs=records(split);ys=read_rows(ROOT/f'Faust_Corpus/splits/sft_v1/{split}.jsonl')
            self.assertEqual([(r['instruction_de'],r['target_text']) for r in xs],[(r['prompt'],r['completion']) for r in ys])

class MaskTests(unittest.TestCase):
    def setUp(self):self.tok=CharTokenizer()
    def test_prompt_mask(self):
        row=encode_pair(self.tok,'Aufgabe','Faust.\nHallo!',1000)
        n=len(self.tok.encode(format_prompt('Aufgabe')))
        self.assertEqual(row['labels'][:n],[-100]*n)
        self.assertEqual(row['labels'][n:],self.tok.encode('Faust.\nHallo!')+[0])
    def test_eos_not_masked(self):
        x=encode_pair(self.tok,'x','Y',1000)
        self.assertEqual(x['labels'][-1],0)
    def test_actual_eos_and_pad_same_id(self):
        a=encode_pair(self.tok,'x','Y',1000);b=encode_pair(self.tok,'xx','YYYYY',1000)
        batch=pad_features([a,b],0);n=len(a['input_ids'])
        self.assertEqual(batch['labels'][0][n-1],0)
        self.assertTrue(all(x==-100 for x in batch['labels'][0][n:]))
        self.assertEqual(batch['attention_mask'][0][n:],[0]*(len(b['input_ids'])-n))
    def test_no_truncation(self):
        with self.assertRaisesRegex(ValueError,'未截断'):encode_pair(self.tok,'x','abc',2)
    def test_at_limit(self):
        n=len(format_prompt('x'))+len('abc')+1
        self.assertEqual(len(encode_pair(self.tok,'x','abc',n)['input_ids']),n)
    def test_target_exactly_reconstructs(self):
        for r in records('train'):
            e=encode_pair(self.tok,r['instruction_de'],r['target_text'],10000)
            ids=[x for x in e['labels'] if x!=-100][:-1]
            self.assertEqual(''.join(chr(x-1) for x in ids),r['target_text'])
    def test_empty_prompt(self):
        with self.assertRaises(ValueError):format_prompt(' ')
    def test_empty_target(self):
        with self.assertRaises(ValueError):encode_pair(self.tok,'p',' ',100)
    def test_empty_batch(self):
        with self.assertRaises(ValueError):pad_features([],0)
    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):pad_features([{'input_ids':[1], 'labels':[], 'attention_mask':[1]}],0)
    def test_no_chat_or_think_template(self):
        p=format_prompt('Bitte')
        self.assertNotIn('<think>',p);self.assertNotIn('<|im_start|>',p)
        self.assertTrue(p.endswith('### Textauszug\n'))
    def test_split_length_accounting(self):
        rows=records('validation')
        enc,stats=encode_split(self.tok,rows,10000)
        self.assertEqual(len(enc),18)
        for e,s in zip(enc,stats):self.assertEqual(s['total_tokens'],s['prompt_tokens']+s['completion_tokens_including_eos'])
    def test_tensor_and_backward_cpu(self):
        import torch
        torch.manual_seed(17)
        torch.set_num_threads(1)
        batch=CompletionCollator(0)([encode_pair(self.tok,'x','AB',1000),encode_pair(self.tok,'y','CDE',1000)])
        model=torch.nn.Sequential(torch.nn.Embedding(256,8),torch.nn.Linear(8,256))
        logits=model(batch['input_ids'])[:,:-1,:].contiguous()
        labels=batch['labels'][:,1:].contiguous()
        loss=torch.nn.functional.cross_entropy(logits.view(-1,256),labels.view(-1),ignore_index=-100)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(model[0].weight.grad).all())
        # Masked prompt positions contribute no DIRECT loss term; their context can still influence targets.
        changed=logits.detach().clone();changed[labels==-100]=10000
        other=torch.nn.functional.cross_entropy(changed.view(-1,256),labels.view(-1),ignore_index=-100)
        self.assertAlmostEqual(loss.item(),other.item(),places=6)

class CheckpointTests(unittest.TestCase):
    def test_empty_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as d:self.assertIsNone(latest_checkpoint(Path(d)))
    def test_incomplete_checkpoint_not_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint-9';p.mkdir();(p/'trainer_state.json').write_text('{"global_step":9}')
            self.assertIsNone(latest_checkpoint(Path(d)))
    def test_latest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            for step in (2,11,7):
                p=Path(d)/f'checkpoint-{step}';p.mkdir()
                for f in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors'):(p/f).write_bytes(b'test')
                (p/'trainer_state.json').write_text(json.dumps({'global_step':step}))
            self.assertEqual(latest_checkpoint(Path(d)).name,'checkpoint-11')
    def test_manifest_digest_order_independent(self):
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))

class StaticTests(unittest.TestCase):
    def test_python_compiles(self):
        for p in (ROOT/'training').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_config(self):
        c=read_json(ROOT/'training/config.json')
        self.assertEqual(c['rank'],16);self.assertEqual(c['max_length'],4096)
        self.assertIn('Base',c['model_id']);self.assertEqual(c['epochs'],3)
    def test_train_does_not_select_test(self):
        tree=ast.parse((ROOT/'training/train.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='records']
        self.assertTrue(calls);self.assertTrue(all(n.args[0].value in {'train','validation'} for n in calls))
    def test_notebook_clean_and_parseable(self):
        p=ROOT/'training/Faust_LoRA_Colab.ipynb'
        if not p.exists():self.skipTest('notebook not authored yet')
        n=read_json(p);self.assertEqual(n['nbformat'],4)
        for cell in n['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
                ast.parse(source);self.assertFalse(cell.get('outputs'));self.assertIsNone(cell.get('execution_count'))

if __name__=='__main__':unittest.main(verbosity=2)
