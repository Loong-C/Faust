#!/usr/bin/env python3
"""Offline tests for apply_release.py. Usage:
python plan/test_apply_release.py --package Faust_SFT_v1_Update.zip --root .
Tests use isolated temporary project copies, never modify your real repository.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, shutil, sys, tempfile, zipfile
from pathlib import Path
sys.dont_write_bytecode=True

def run(root:Path,archive:Path)->dict:
    module_path=Path(__file__).with_name('apply_release.py')
    spec=importlib.util.spec_from_file_location('faust_release_apply',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parent=json.loads((root/'Faust_Corpus/verification/sft_v1/parent_snapshot.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    def snapshot(p):return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
    def raises(fn):
        try:fn()
        except (ValueError,OSError,KeyError,zipfile.BadZipFile):return True
        return False
    with tempfile.TemporaryDirectory(prefix='faust-sync-test-') as tmp:
        work=Path(tmp)/'project';work.mkdir()
        for rec in parent['files']:
            dest=work/'Faust_Corpus'/rec['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/'Faust_Corpus'/rec['path'],dest)
        before=snapshot(work)
        r=module.process(work,archive,False)
        check('dry_run_has_no_file_writes',snapshot(work)==before and r['mode']=='dry_run_no_writes')
        r=module.process(work,archive,True)
        check('apply_successful',r['mode']=='applied' and (work/'plan/STATE.json').exists())
        check('apply_preserves_every_parent_file',all(snapshot(work)[name]==value for name,value in before.items()))
        r=module.process(work,archive,True)
        check('repeat_is_idempotent',r['files_to_write']==0)
        target=work/'Faust_Corpus/data/sft_v1/faust_sft_pairs.jsonl';old=target.read_bytes();target.write_bytes(old+b'\n')
        before_conflict=snapshot(work)
        check('changed_local_file_rejected_without_writes',raises(lambda:module.process(work,archive,True)) and snapshot(work)==before_conflict)
        target.write_bytes(old)
        raw=work/'Faust_Corpus'/parent['files'][0]['path'];oldraw=raw.read_bytes();raw.write_bytes(oldraw+b'\n')
        check('wrong_parent_checksum_rejected',raises(lambda:module.process(work,archive,False)))
        raw.write_bytes(oldraw)
        state=work/'plan/STATE.json';oldstate=state.read_bytes();state.write_text('{"release_id":"unrelated-version"}',encoding='utf-8')
        check('wrong_state_version_rejected',raises(lambda:module.process(work,archive,False)))
        state.write_bytes(oldstate)
        with zipfile.ZipFile(archive) as src:
            contents={i.filename:src.read(i.filename) for i in src.infolist() if not i.is_dir()}
        altered=Path(tmp)/'altered.zip'
        with zipfile.ZipFile(altered,'w',zipfile.ZIP_DEFLATED) as z:
            for name,data in contents.items():z.writestr(name,data+b' ' if name=='Faust_Corpus/data/sft_v1/faust_sft_pairs.jsonl' else data)
        check('corrupt_payload_rejected',raises(lambda:module.process(work,altered,False)))
        for extra,label in [('../outside.txt','path_traversal_rejected'),('plan/unlisted.txt','unlisted_member_rejected')]:
            with zipfile.ZipFile(altered,'w',zipfile.ZIP_DEFLATED) as z:
                for name,data in contents.items():z.writestr(name,data)
                z.writestr(extra,b'bad')
            check(label,raises(lambda:module.process(work,altered,False)))
    return {'all_checks_passed':all(c['passed'] for c in checks),'check_count':len(checks),'checks':checks,'tests_touch_real_project':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();result=run(a.root.resolve(),a.package.resolve());s=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(s,encoding='utf-8')
    print(s);raise SystemExit(0 if result['all_checks_passed'] else 1)
