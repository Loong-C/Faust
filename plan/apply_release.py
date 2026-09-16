#!/usr/bin/env python3
"""Apply a versioned Faust delta package; dry-run unless --apply is supplied.
Python 3.10+ standard library. Checks all payload hashes, parent hashes, paths and
conflicts before writing. Full-snapshot ZIPs should instead be extracted into a
new empty directory. No network, shell commands, or deletion of user files.
"""
from __future__ import annotations
import argparse, hashlib, json, os, stat, tempfile, zipfile
from pathlib import Path, PurePosixPath

LIMIT_FILE=32*1024*1024
LIMIT_TOTAL=256*1024*1024

def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def safe_name(name:str)->str:
    if '\\' in name or ':' in name or '\x00' in name:raise ValueError('Unsafe path: '+repr(name))
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ('..','.') for x in name.split('/')) or not name or name.endswith('/'):
        raise ValueError('Unsafe file path: '+repr(name))
    if p.parts[0] not in {'Faust_Corpus','plan'}:raise ValueError('Unexpected top-level folder: '+name)
    return p.as_posix()

def target_path(root:Path,name:str)->Path:
    name=safe_name(name);p=root/name
    current=root
    for part in PurePosixPath(name).parts:
        current=current/part
        if current.is_symlink():raise ValueError('Symlink target is not allowed: '+name)
    if not p.resolve().is_relative_to(root):raise ValueError('Path escapes project: '+name)
    if p.exists() and not p.is_file():raise ValueError('Target is not a file: '+name)
    return p

def replace_bytes(p:Path,data:bytes)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    name=None
    try:
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.faust-update-',delete=False) as f:
            name=f.name;f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,p);name=None
    finally:
        if name and os.path.exists(name):os.unlink(name)

def process(root:Path,package:Path,apply:bool=False)->dict:
    root=root.resolve()
    if not root.is_dir():raise ValueError('Project root does not exist.')
    with zipfile.ZipFile(package) as z:
        members={}
        total=0
        for info in z.infolist():
            if info.is_dir():continue
            name=safe_name(info.filename)
            if name in members:raise ValueError('Duplicate archive path: '+name)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlink in ZIP: '+name)
            if info.file_size>LIMIT_FILE:raise ValueError('Oversized member: '+name)
            total+=info.file_size
            if total>LIMIT_TOTAL:raise ValueError('Archive too large.')
            members[name]=info
        manifests=[n for n in members if n.startswith('plan/releases/') and n.endswith('/manifest.json')]
        if len(manifests)!=1:raise ValueError('Expected one release manifest; use a delta package.')
        manifest_path=manifests[0];manifest_bytes=z.read(manifest_path);m=json.loads(manifest_bytes)
        if m['package_type']!='delta':raise ValueError('This is not a delta manifest.')
        files=m['files']
        names=[safe_name(r['path']) for r in files]
        if len(set(names))!=len(names) or manifest_path in names:raise ValueError('Bad manifest paths.')
        if set(members)!=set(names)|{manifest_path}:
            raise ValueError('Package contains unlisted or missing files. A full snapshot cannot be applied as a delta.')
        payload={}
        for r in files:
            data=z.read(r['path'])
            if len(data)!=r['bytes'] or digest(data)!=r['sha256']:
                raise ValueError('Payload checksum failed: '+r['path'])
            payload[r['path']]=data
        payload[manifest_path]=manifest_bytes
    # Every Stage-1 baseline file must agree. Do not infer a matching base from filenames.
    for r in m['requires']['parent_files']:
        p=target_path(root,r['path'])
        if not p.is_file() or digest(p.read_bytes())!=r['sha256']:
            raise ValueError('Parent snapshot mismatch: '+r['path'])
    state=root/'plan/STATE.json'
    if state.exists():
        state_path=target_path(root,'plan/STATE.json')
        local=json.loads(state_path.read_text(encoding='utf-8'))
        if local.get('release_id') not in {m['parent_release_id'],m['release_id']}:
            raise ValueError('Local release differs from package parent; stop and reconcile versions.')
    records={r['path']:r for r in files}
    records[manifest_path]={'path':manifest_path,'previous_sha256':None}
    changes=[];skipped=[];before={}
    # Preflight all conflicts before any write.
    for name,data in payload.items():
        p=target_path(root,name);old=p.read_bytes() if p.exists() else None
        if old==data:skipped.append(name);continue
        expected=records[name].get('previous_sha256')
        if expected is None:
            if old is not None:raise ValueError('Different local file would be overwritten: '+name)
        elif old is None or digest(old)!=expected:
            raise ValueError('Expected old file hash does not match: '+name)
        before[name]=old;changes.append(name)
    written=[]
    if apply:
        try:
            for name in changes:
                p=target_path(root,name)
                # Recheck for changes made after preflight.
                current=p.read_bytes() if p.exists() else None
                if current!=before[name]:raise ValueError('File changed during apply: '+name)
                replace_bytes(p,payload[name]);written.append(name)
        except Exception:
            for name in reversed(written):
                p=root/name
                if before[name] is None:p.unlink(missing_ok=True)
                else:replace_bytes(p,before[name])
            raise
    return {'release_id':m['release_id'],'parent_release_id':m['parent_release_id'],
            'mode':'applied' if apply else 'dry_run_no_writes',
            'parent_files_verified':len(m['requires']['parent_files']),
            'files_to_write':len(changes),'already_identical':len(skipped),
            'write_paths':changes,'note_zh':'所有校验通过。' if apply else '仅检查；加 --apply 才会写入。'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True,help='Folder containing Faust_Corpus and plan')
    p.add_argument('--package',type=Path,required=True,help='Delta update ZIP')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    try: print(json.dumps(process(a.root,a.package,a.apply),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:p.exit(2,str(exc)+'\n')
