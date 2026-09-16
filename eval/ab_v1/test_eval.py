"""Offline tests. Mock generation is not a CUDA/Unsloth integration test."""
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import run


def case(i=1, subset='dev'):
    return {'case_id': f'NEW_{i:02d}', 'set': subset, 'instruction_de': 'A fragt. B antwortet.',
            'description_zh': '甲提问，乙回答。', 'fictional': True, 'target_text': None}


def segment(text, sid='s1', scene='c1', split='train'):
    return dict(segment_id=sid, work_id='F1', scene_id=scene, split=split, target_text=text)


def saved(c, seed, mode, sig='S'):
    text = 'Eine Probe. ' + mode
    return dict(key=f'{c["case_id"]}-{seed}-{mode}', signature=sig, case_id=c['case_id'],
                seed=seed, mode=mode, text=text, text_sha256=core.text_sha(text),
                new_token_ids=[2,3], generated_tokens=2, finish_reason='eos',
                prompt_ids_sha256=core.digest([1]))


class Inputs(unittest.TestCase):
    def test_git_blob_crlf_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a';b=Path(d)/'b'
            a.write_bytes(b'first\nsecond\n');b.write_bytes(b'first\r\nsecond\r\n')
            self.assertEqual(core.git_blob(a),core.git_blob(b))
            self.assertNotEqual(core.sha(a),core.sha(b))

    def test_real_frozen_data(self):
        cfg, record, cases, training, result = core.check_inputs()
        self.assertTrue(result['ok'])
        self.assertEqual(result['counts'], {'train':158,'validation':18,'test':19})
        self.assertEqual(len(cases), 16)
        self.assertEqual(record['model']['rank'], 8)

    def test_prompt_exactly_reuses_training(self):
        training = core.load_training_core()
        self.assertEqual(training.format_prompt('abc'), '### Aufgabe\nabc\n\n### Textauszug\n')

    def test_original_cases_no_targets(self):
        cfg, record, cases, training, data = core.check_inputs()
        self.assertTrue(all(c['target_text'] is None for c in cases))
        self.assertEqual(len(core.expected_keys(core.select_cases(cases,'dev'),cfg['seeds'])),72)

    def test_holdout_blocked(self):
        with self.assertRaises(ValueError): core.select_cases([case()], 'holdout')

    def test_holdout_explicit(self):
        self.assertEqual(core.select_cases([case(1),case(2,'holdout')],'holdout',True),[case(2,'holdout')])

    def test_bad_subset(self):
        with self.assertRaises(ValueError): core.select_cases([case()], 'all')

    def test_duplicate_cases(self):
        with self.assertRaises(ValueError): core.check_cases([case(),case()], {'expected_cases':{'dev':2}})

    def test_target_rejected(self):
        c=case();c['target_text']='should not be here'
        with self.assertRaises(ValueError): core.check_cases([c], {'expected_cases':{'dev':1}})

    def test_case_missing_description(self):
        c=case();c['description_zh']=''
        with self.assertRaises(ValueError): core.check_cases([c], {'expected_cases':{'dev':1}})

    def test_budget_no_truncation(self):
        with self.assertRaises(ValueError): run.require_budget([1]*1000,1024,1920)

    def test_budget_exact_fit(self):
        run.require_budget([1]*896,1024,1920)

    def test_empty_prompt(self):
        with self.assertRaises(ValueError): run.require_budget([],20,100)

    def test_paths_reject_traversal(self):
        for name in ('../adapter','/tmp/out','a/b','','.','..'):
            with self.subTest(name=name), self.assertRaises(ValueError): core.run_paths(name)

    def test_paths_reject_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'runs').symlink_to('/tmp')
            with self.assertRaises(ValueError): core.run_paths('x',root)

    def test_atomic_write_no_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.symlink_to(Path(d)/'original')
            with self.assertRaises(ValueError): core.write_json(p, {})

    def test_data_only_cli(self):
        p=subprocess.run([sys.executable,str(core.HERE/'run.py'),'check','--data-only'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertFalse(json.loads(p.stdout)['gpu_generation_tested'])

    def test_start_rejects_training_action(self):
        p=subprocess.run(['bash',str(core.HERE/'start.sh'),'train'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)

    def test_start_does_not_install_missing_environment(self):
        with tempfile.TemporaryDirectory() as d:
            env=dict(os.environ,FAUST_ENV_HOME=d)
            p=subprocess.run(['bash',str(core.HERE/'start.sh'),'all'],env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,3,p.stderr)
            self.assertEqual(list(Path(d).iterdir()),[])


class Adapter(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        (self.p/'adapter_model.safetensors').write_bytes(b'only-test-bytes')
        self.cfg=dict(peft_type='LORA',task_type='CAUSAL_LM',base_model_name_or_path='x/y',
                      revision='abc',r=8,lora_alpha=16,bias='none',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        self.record=dict(model=dict(id='x/y',revision='abc',rank=8,alpha=16),
                         adapter=dict(bytes=15,sha256=core.sha(self.p/'adapter_model.safetensors')))
        core.write_json(self.p/'adapter_config.json',self.cfg)

    def tearDown(self): self.tmp.cleanup()

    def test_metadata_valid(self): self.assertIn('weights_sha256',core.check_adapter(self.p,self.record))

    def test_missing_weights(self):
        (self.p/'adapter_model.safetensors').unlink()
        with self.assertRaises(ValueError): core.check_adapter(self.p,self.record)

    def test_wrong_weight(self):
        (self.p/'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError): core.check_adapter(self.p,self.record)

    def test_wrong_revision(self):
        self.cfg['revision']='bad';core.write_json(self.p/'adapter_config.json',self.cfg)
        with self.assertRaises(ValueError): core.check_adapter(self.p,self.record)

    def test_extra_trainable_bias(self):
        self.cfg['bias']='all';core.write_json(self.p/'adapter_config.json',self.cfg)
        with self.assertRaises(ValueError): core.check_adapter(self.p,self.record)

    def test_modules_to_save(self):
        self.cfg['modules_to_save']=['lm_head'];core.write_json(self.p/'adapter_config.json',self.cfg)
        with self.assertRaises(ValueError): core.check_adapter(self.p,self.record)


class SavedRecords(unittest.TestCase):
    def test_valid(self): core.check_record(saved(case(),1,'base'),'NEW_01-1-base','S',case(),1,'base',[1])

    def test_corrupt_text(self):
        r=saved(case(),1,'base');r['text']='modified'
        with self.assertRaises(ValueError): core.check_record(r,r['key'],'S',case(),1,'base')

    def test_different_signature(self):
        r=saved(case(),1,'base')
        with self.assertRaises(ValueError): core.check_record(r,r['key'],'T',case(),1,'base')

    def test_changed_prompt(self):
        r=saved(case(),1,'base')
        with self.assertRaises(ValueError): core.check_record(r,r['key'],'S',case(),1,'base',[99])

    def test_invalid_tokens(self):
        r=saved(case(),1,'base');r['new_token_ids']=['bad',3]
        with self.assertRaises(ValueError): core.check_record(r,r['key'],'S',case(),1,'base')

    def test_invalid_stop(self):
        r=saved(case(),1,'base');r['finish_reason']='guessed'
        with self.assertRaises(ValueError): core.check_record(r,r['key'],'S',case(),1,'base')


class Overlap(unittest.TestCase):
    def test_exact_match(self):
        r=core.CopyIndex([segment('eins zwei drei vier')]).compare('null eins zwei drei vier ende')
        self.assertEqual(r['longest_match']['all']['words'],4)

    def test_case_punctuation_normalization(self):
        r=core.CopyIndex([segment('Grüße, Straße!')]).compare('GRÜSSE STRASSE')
        self.assertEqual(r['longest_match']['all']['words'],2)

    def test_no_cross_scene_match(self):
        r=core.CopyIndex([segment('eins zwei'),segment('drei vier','s2','c2')]).compare('eins zwei drei vier')
        self.assertEqual(r['longest_match']['all']['words'],2)

    def test_cross_segment_same_scene_detected(self):
        r=core.CopyIndex([segment('eins zwei'),segment('drei vier','s2')]).compare('eins zwei drei vier')
        self.assertEqual(r['longest_match']['all']['words'],4)

    def test_split_attribution(self):
        r=core.CopyIndex([segment('eins zwei'),segment('eins zwei drei','s2','c2','test')]).compare('eins zwei drei')
        self.assertEqual(r['longest_match']['train']['words'],2)
        self.assertEqual(r['longest_match']['test']['words'],3)

    def test_no_match(self):
        r=core.CopyIndex([segment('eins zwei')]).compare('xyz')
        self.assertEqual(r['longest_match'],{})

    def test_empty(self):
        self.assertEqual(core.CopyIndex([]).compare('')['word_count'],0)

    def test_ngram_fraction(self):
        r=core.CopyIndex([segment('a b c d e f g h i')]).compare('a b c d e f g h i')
        self.assertEqual(r['matched_ngram_fraction']['8'],1)

    def test_repetition(self):
        self.assertGreater(core.text_diagnostics('a b c d\na b c d')['repeated_4gram_fraction'],0)

    def test_matches_bruteforce(self):
        rng=random.Random(41)
        for _ in range(35):
            a=[rng.choice('abcde') for _ in range(15)];b=[rng.choice('abcde') for _ in range(10)]
            expected=0
            for i in range(len(a)):
                for j in range(len(b)):
                    n=0
                    while i+n<len(a) and j+n<len(b) and a[i+n]==b[j+n]: n+=1
                    expected=max(expected,n)
            got=core.CopyIndex([segment(' '.join(a))]).compare(' '.join(b))
            self.assertEqual(got['longest_match'].get('all',{}).get('words',0),expected)


class BlindExport(unittest.TestCase):
    def setUp(self):
        self.cases=[case(),case(2)];self.seeds=[1,2,3]
        self.gs=[saved(c,s,m) for c in self.cases for s in self.seeds for m in core.MODES]
        self.secret='ab'*32

    def test_deterministic_mapping(self):
        self.assertEqual(core.make_blind(self.gs,self.cases,self.seeds,self.secret),
                         core.make_blind(self.gs,self.cases,self.seeds,self.secret))

    def test_pair_identification(self):
        blind,reveal=core.make_blind(self.gs,self.cases,self.seeds,self.secret)
        mapping={r['pair_id']:r for r in reveal}
        for row in blind:
            for label in ('A','B'):
                self.assertTrue(row[label]['text'].endswith(mapping[row['pair_id']][label]))
                self.assertNotIn('mode',row[label]);self.assertNotIn('seed',row)

    def test_missing_generation_blocks_export(self):
        with self.assertRaises(ValueError): core.make_blind(self.gs[:-1],self.cases,self.seeds,self.secret)

    def test_duplicate_generation_blocks_export(self):
        with self.assertRaises(ValueError): core.make_blind(self.gs+[self.gs[0]],self.cases,self.seeds,self.secret)

    def test_no_constant_side_assignment(self):
        _,reveal=core.make_blind(self.gs,self.cases,self.seeds,self.secret)
        self.assertEqual({r['A'] for r in reveal},{'base','lora'})

    def test_export_roundtrip_and_no_private_files(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'private_run';public=Path(d)/'public'
            cfg={'seeds':self.seeds,'copy_warning_min_words':16}
            core.write_json(folder/'manifest.json',{'signature':'S','status':'completed'})
            for r in self.gs: core.write_json(folder/'records'/(r['key']+'.json'),r)
            out=core.export_results(folder,public,self.cases,cfg,[segment('eine probe')])
            self.assertEqual(out['pairs'],6)
            with zipfile.ZipFile(public/'blind_review.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(set(z.namelist()),{'blind_review.jsonl','blind_review.txt','review_scores.csv','README_盲评.txt'})
            self.assertEqual(len(core.rows(public/'blind_review.jsonl')),6)
            before=(public/'blind_review.jsonl').read_bytes()
            core.export_results(folder,public,self.cases,cfg,[segment('eine probe')])
            self.assertEqual(before,(public/'blind_review.jsonl').read_bytes())
            with (public/'review_scores.csv').open('a') as f: f.write('my feedback')
            with self.assertRaises(ValueError): core.export_results(folder,public,self.cases,cfg,[])
            self.assertIn('my feedback',(public/'review_scores.csv').read_text())

    def test_partial_run_not_exported(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'p';core.write_json(f/'manifest.json',{'status':'failed'})
            with self.assertRaises(ValueError): core.export_results(f,Path(d)/'pub',[],{},[])


class FakeLayer:
    disable_adapters=False
    merged_adapters=[]


class Switch(unittest.TestCase):
    def test_missing_layers(self):
        with self.assertRaises(ValueError): run.check_layer_state([],False)

    def test_merged_rejected(self):
        l=FakeLayer();l.merged_adapters=['default']
        with self.assertRaises(ValueError): run.check_layer_state([l],False)

    def test_wrong_state_rejected(self):
        with self.assertRaises(ValueError): run.check_layer_state([FakeLayer()],True)

    def test_restored_on_exception(self):
        layer=FakeLayer()
        class Model:
            @contextlib.contextmanager
            def disable_adapter(self):
                layer.disable_adapters=True
                try: yield
                finally: layer.disable_adapters=False
        with self.assertRaises(RuntimeError):
            with run.condition(Model(),[layer],'base'):
                self.assertTrue(layer.disable_adapters)
                raise RuntimeError('interrupt')
        self.assertFalse(layer.disable_adapters)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional CPU torch not installed')
    def test_mock_cpu_generation_and_probe(self):
        import torch
        layer=FakeLayer();seed_log=[]
        shim=types.SimpleNamespace(tensor=lambda x,**kw:torch.tensor(x,dtype=kw.get('dtype')),
            long=torch.long,ones_like=torch.ones_like,inference_mode=torch.inference_mode,
            isfinite=torch.isfinite,allclose=torch.allclose,
            cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0))
        class Model:
            @contextlib.contextmanager
            def disable_adapter(self):
                layer.disable_adapters=True
                try: yield
                finally: layer.disable_adapters=False
            def __call__(self,**kw):
                n=kw['input_ids'].shape[1]
                return types.SimpleNamespace(logits=torch.ones(1,n,4)*(1 if layer.disable_adapters else 2))
            def generate(self,**kw):
                return torch.cat([kw['input_ids'],torch.tensor([[2 if layer.disable_adapters else 3,0]])],dim=1)
        tok=types.SimpleNamespace(eos_token_id=0,decode=lambda ids,**kw:' '.join(map(str,ids)))
        model=Model();gen=types.SimpleNamespace(max_new_tokens=10)
        self.assertTrue(run.probe_switch(model,tok,[layer],shim,[1,2])['enabled_state_restored'])
        a=run.generate_one(model,tok,[layer],shim,seed_log.append,[9],gen,'base',42)
        b=run.generate_one(model,tok,[layer],shim,seed_log.append,[9],gen,'lora',42)
        self.assertEqual(seed_log,[42,42]);self.assertEqual(a['new_token_ids'],[2,0])
        self.assertEqual(b['new_token_ids'],[3,0]);self.assertEqual(a['finish_reason'],'eos')
        self.assertFalse(layer.disable_adapters)


if __name__=='__main__': unittest.main()
