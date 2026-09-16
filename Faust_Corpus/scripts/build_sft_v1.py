#!/usr/bin/env python3
"""Build Stage-2 training pairs from reviewed, individually authored annotations.

Python 3.10+, standard library only. No network, API calls or training.
This script does NOT generate semantic annotations: review/sft_v1/annotations.jsonl
is the authored source. Stage-1 files are read-only. Rebuild writes sft_v1 outputs.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

BASE_SHA256='690a1c4433423b7425ac2bdd5124d2be7da4916c44d1fd4358438ca0dc102708'
RELEASE='faust-sft-0.1.0'
INTRO=('Schreibe einen literarischen Textausschnitt auf Deutsch nach den folgenden '
       'inhaltlichen Vorgaben. Halte die Reihenfolge der Vorgänge ein. '
       'Unterscheide ausgeführte Handlungen von Berichten, Wünschen, Träumen und '
       'Behauptungen der Figuren. Gib nur den Textausschnitt aus, ohne Erläuterung.')

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def read_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def prompt_from(a: dict) -> str:
    # Only allow-listed authoring fields enter the prompt. No German target,
    # following-source context, Chinese, IDs or source locations are exposed.
    return (INTRO+'\n\nForm: '+a['form_de']+'\nAusgangslage: '+a['situation_de']+
            '\nAblauf: '+' '.join(a['beats_de']))

def build(root: Path) -> dict:
    source=root/'data/master_segments.jsonl'
    if sha(source.read_bytes())!=BASE_SHA256:
        raise ValueError('Stage-1 master corpus hash changed; do not silently rebase annotations.')
    master=read_rows(source)
    authored=read_rows(root/'review/sft_v1/annotations.jsonl')
    if len(authored)!=195 or len({a['segment_id'] for a in authored})!=195:
        raise ValueError('Expected 195 unique authored annotations.')
    by_id={a['segment_id']:a for a in authored}
    if set(by_id)!={m['segment_id'] for m in master}:
        raise ValueError('Annotation IDs do not match the parent corpus.')
    pairs=[]; specs=[]
    for m in master:
        a=by_id[m['segment_id']]
        if a['target_text_sha256']!=sha(m['target_text'].encode('utf-8')):
            raise ValueError('Target hash mismatch: '+m['segment_id'])
        for k in ['form_de','situation_de','description_zh']:
            if not isinstance(a[k],str) or not a[k].strip(): raise ValueError('Empty '+k)
        if not 3<=len(a['beats_de'])<=9 or any(not isinstance(x,str) or not x.strip() for x in a['beats_de']):
            raise ValueError('Bad beat list: '+m['segment_id'])
        instruction=prompt_from(a)
        spec={'segment_id':m['segment_id'],'split':m['split'],
              'form_de':a['form_de'],'situation_de':a['situation_de'],
              'beats_de':a['beats_de'],'ending_state_de':a['beats_de'][-1],
              'speakers':m['speakers'],'description_zh':a['description_zh'],
              'scope_zh':'仅描述本片段；必要的定位上下文不扩写为片段内事件。'}
        specs.append(spec)
        pair={'schema_version':'1.0','release_id':RELEASE,
              'segment_id':m['segment_id'],'work_id':m['work_id'],
              'scene_id':m['scene_id'],'scene_group_id':m['scene_group_id'],
              'scene_title':m['scene_title'],'scene_title_zh':m['scene_title_zh'],
              'split':m['split'],'instruction_de':instruction,
              'description_zh':a['description_zh'],'scene_specification':spec,
              'target_text':m['target_text'],'target_text_sha256':m['target_text_sha256'],
              'instruction_sha256':sha(instruction.encode('utf-8')),
              'source':{k:m[k] for k in ['source_sha256','source_xpath_start','source_xpath_end',
                       'event_ids','source_issue_ids','verse_element_index_range',
                       'source_verse_number_status','verse_element_count','prose_element_count']},
              'quality':{'text_basis':'uploaded_GerDraCor_TEI_via_reviewed_v2',
                         'annotation_review':'assistant_self_review_complete',
                         'independent_human_review':False,'external_llm_api_called':False,
                         'original_target_unchanged':True,
                         'lexical_audit_file':'review/sft_v1/lexical_overlap.jsonl',
                         'pretraining_contamination_excluded':False}}
        pairs.append(pair)
    write_rows(root/'data/sft_v1/scene_specs.jsonl',specs)
    write_rows(root/'data/sft_v1/faust_sft_pairs.jsonl',pairs)
    for split in ['train','validation','test']:
        selected=[p for p in pairs if p['split']==split]
        write_rows(root/f'splits/sft_v1/{split}.jsonl',
                   [{'prompt':p['instruction_de'],'completion':p['target_text']} for p in selected])
        write_rows(root/f'splits/sft_v1/{split}_index.jsonl',
                   [{'row_index_zero_based':i,'segment_id':p['segment_id'],
                     'instruction_sha256':p['instruction_sha256'],
                     'target_text_sha256':p['target_text_sha256']} for i,p in enumerate(selected)])
    doc=root/'docs/sft_v1';doc.mkdir(parents=True,exist_ok=True)
    text=['《浮士德》戏剧说明中文对照｜faust-sft-0.1.0',
          '此文件供阅读与核查，不作为训练输入。德语说明见主数据的 instruction_de。',
          '角色的传言、梦境和设想与实际事件分别叙述；不补足原作未说明的情节。\n']
    for p in pairs:
        text.append(f'{p["segment_id"]}　{p["scene_title_zh"]}　{p["split"]}')
        text.append(p['description_zh']+'\n')
    (doc/'剧本说明_中文.txt').write_text('\n'.join(text),encoding='utf-8')
    return {'release_id':RELEASE,'rows':len(pairs),
            'splits':{s:sum(p['split']==s for p in pairs) for s in ['train','validation','test']},
            'master_sha256':BASE_SHA256,'annotation_sha256':sha((root/'review/sft_v1/annotations.jsonl').read_bytes()),
            'network_used':False,'model_training_performed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    try: print(json.dumps(build(args.corpus.resolve()),ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError) as exc: parser.exit(2,str(exc)+'\n')
