"""Offline unit tests, including real CPU tensor masking/backpropagation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import core
import run


class ToyTokenizer:
    eos_token_id = 0
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [b + 1 for b in text.encode('utf-8')]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()
        self.row = {'segment_id': 'synthetic', 'instruction_de': 'Beschreibe den Raum.',
                    'target_text': 'Faust.\nIch bin hier.'}

    def test_01_real_files_integrity(self):
        self.assertEqual(core.validate_data()['counts'], {'train':158,'validation':18,'test':19})
    def test_02_train_is_not_whole_corpus(self):
        self.assertEqual(len(core.training_rows('train')),158)
    def test_03_validation_count(self):
        self.assertEqual(len(core.training_rows('validation')),18)
    def test_04_test_loading_forbidden(self):
        with self.assertRaises(ValueError): core.training_rows('test')
    def test_05_bad_split_forbidden(self):
        with self.assertRaises(ValueError): core.training_rows('other')
    def test_06_prompt_format(self):
        self.assertEqual(core.format_prompt('x'), '### Aufgabe\nx\n\n### Textauszug\n')
    def test_07_blank_prompt(self):
        with self.assertRaises(ValueError): core.format_prompt('  ')
    def test_08_mask_prompt(self):
        x, n = core.encode_pair(self.tok,self.row)
        self.assertEqual(x['labels'][:n['prompt_tokens']],[-100]*n['prompt_tokens'])
    def test_09_eos_is_supervised(self):
        x,_=core.encode_pair(self.tok,self.row)
        self.assertEqual(x['labels'][-1],0)
    def test_10_target_bytes_roundtrip(self):
        x,n=core.encode_pair(self.tok,self.row)
        self.assertEqual(bytes(i-1 for i in x['input_ids'][n['prompt_tokens']:-1]).decode(),self.row['target_text'])
    def test_11_unicode_roundtrip(self):
        row={**self.row,'target_text':'Öl, Größe — süß!\n啊。'}
        x,n=core.encode_pair(self.tok,row)
        self.assertEqual(bytes(i-1 for i in x['input_ids'][n['prompt_tokens']:-1]).decode(),row['target_text'])
    def test_12_empty_completion(self):
        with self.assertRaises(ValueError): core.encode_pair(self.tok,{**self.row,'target_text':''})
    def test_13_no_eos(self):
        self.tok.eos_token_id=None
        with self.assertRaises(ValueError): core.encode_pair(self.tok,self.row)
    def test_14_pad_dynamic(self):
        a,_=core.encode_pair(self.tok,self.row)
        b,_=core.encode_pair(self.tok,{**self.row,'target_text':'x'})
        x=core.pad_features([a,b],0)
        self.assertEqual(len(x['input_ids'][0]),len(a['input_ids']))
        self.assertEqual(x['labels'][1][len(b['labels']):],[-100]*(len(a['labels'])-len(b['labels'])))
    def test_15_eos_equal_pad_kept(self):
        a,_=core.encode_pair(self.tok,self.row)
        b,_=core.encode_pair(self.tok,{**self.row,'target_text':'x'})
        x=core.pad_features([a,b],0)
        self.assertEqual(x['labels'][1][len(b['labels'])-1],0)
        self.assertEqual(x['attention_mask'][1][-1],0)
    def test_16_empty_batch(self):
        with self.assertRaises(ValueError): core.pad_features([],0)
    def test_17_bad_lengths(self):
        with self.assertRaises(ValueError): core.pad_features([{'input_ids':[1],'labels':[],'attention_mask':[1]}],0)
    def test_18_no_input_mutation(self):
        a,_=core.encode_pair(self.tok,self.row); before=copy.deepcopy(a)
        core.pad_features([a],0);self.assertEqual(a,before)
    def test_19_budget_rounding(self):
        self.assertEqual(core.budget([{'segment_id':'a','total_tokens':2030}],4096),2048)
    def test_20_budget_ceiling(self):
        with self.assertRaisesRegex(ValueError,'未截断'): core.budget([{'segment_id':'long','total_tokens':4097}],4096)
    def test_21_budget_not_fixed_padding(self):
        self.assertEqual(core.budget([{'segment_id':'a','total_tokens':700}],4096),704)
    def test_22_bad_budget(self):
        with self.assertRaises(ValueError): core.budget([],4096)
    def test_23_path_escape(self):
        for path in ['../raw','/tmp/run','x/y','..','']:
            with self.subTest(path=path), self.assertRaises(ValueError): core.safe_run_dir(path)
    def test_24_path_normal(self):
        self.assertEqual(core.safe_run_dir('local8gb-r16').name,'local8gb-r16')
    def test_25_stale_smoke(self):
        with self.assertRaises(ValueError): core.check_receipt({'status':'passed','signature':'old'},'new')
    def test_26_failed_smoke(self):
        with self.assertRaises(ValueError): core.check_receipt({'status':'failed','signature':'x'},'x')
    def test_27_good_smoke(self):
        core.check_receipt({'status':'passed','signature':'x'},'x')
    def test_28_finite_loss(self):
        for x in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(FloatingPointError): core.finite(x,'loss')
    def test_29_valid_loss(self):
        self.assertEqual(core.finite(3.2,'loss'),3.2)
    def test_30_no_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'checkpoint-1';p.mkdir();core.write_json(p/'trainer_state.json',{'global_step':1})
            self.assertIsNone(core.latest_checkpoint(Path(t)))
    def test_31_checkpoint_numeric_order(self):
        with tempfile.TemporaryDirectory() as t:
            for n in (9,10):
                p=Path(t)/f'checkpoint-{n}';p.mkdir()
                for name in ('optimizer.pt','scheduler.pt','rng_state.pth','adapter_config.json','adapter_model.safetensors'): (p/name).touch()
                core.write_json(p/'trainer_state.json',{'global_step':n})
            self.assertEqual(core.latest_checkpoint(Path(t)).name,'checkpoint-10')
    def test_32_lean_does_not_change_text_budget_or_base(self):
        a,b=run.config_for('standard'),run.config_for('lean')
        self.assertEqual({k for k in a if a[k]!=b[k]},{'rank','alpha'})
        self.assertEqual(b['rank'],8)
    def test_33_hash_corruption_rejected(self):
        with patch('core.sha',return_value='incorrect'):
            with self.assertRaisesRegex(ValueError,'数据不一致'): core.validate_data()
    def test_34_signature_dictionary_order(self):
        self.assertEqual(core.fingerprint({'a':1,'b':2}),core.fingerprint({'b':2,'a':1}))
    def test_35_output_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'runs').mkdir();(root/'Faust_Corpus').mkdir()
            (root/'runs'/'bad').symlink_to(root/'Faust_Corpus',target_is_directory=True)
            with self.assertRaises(ValueError): core.safe_run_dir('bad',root)

    def test_36_real_cpu_backprop(self):
        import torch
        from torch import nn
        torch.manual_seed(5)
        a,_=core.encode_pair(self.tok,self.row)
        b,_=core.encode_pair(self.tok,{**self.row,'target_text':'x'})
        batch=core.Collator(0)([a,b])
        model=nn.Sequential(nn.Embedding(257,8),nn.Linear(8,257))
        logits=model(batch['input_ids'])[:,:-1]
        labels=batch['labels'][:,1:]
        loss=nn.functional.cross_entropy(logits.reshape(-1,257),labels.reshape(-1),ignore_index=-100)
        self.assertTrue(torch.isfinite(loss).item());loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        masked=labels==-100; changed=logits.detach().clone();changed[masked]=1000
        second=nn.functional.cross_entropy(changed.reshape(-1,257),labels.reshape(-1),ignore_index=-100)
        self.assertAlmostEqual(loss.item(),second.item(),places=5)


if __name__=='__main__': unittest.main(verbosity=2)
