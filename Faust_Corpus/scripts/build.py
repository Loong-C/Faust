#!/usr/bin/env python3
"""Rebuild the reviewed Faust corpus from two local TEIs and frozen cut decisions.
Python 3.10+, standard library only. No model or external network call.
"""
from __future__ import annotations
import argparse, collections, copy, datetime, hashlib, json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET
from tei_reader import TEIParser, tag, norm, sha_bytes, sha_text, NS, XML
ROOT=Path(__file__).resolve().parents[1]
FILES={'F1':'ger000243-goethe-faust-eine-tragoedie.tei.xml',
       'F2':'ger000201-goethe-faust-der-tragoedie-zweiter-teil.tei.xml'}
def jwrite(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def jlwrite(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def render(es):
 out=[];last=None
 for e in es:
  if last is not None:
   same=(e['kind']==last['kind']=='verse' and e.get('stanza_ids')==last.get('stanza_ids') and e['source_atom_id']==last['source_atom_id'])
   out.append('\n' if same else '\n\n')
  out.append(e['text']);last=e
 return ''.join(out)
def anchor_index(aa,marker):
 """integer: after atom; N@i: after event i; N!: before last trailing stage;
 N^: after last spoken event (all trailing stage directions go right)."""
 s=str(marker);num=int(re.match(r'\d+',s).group());a=aa[num-1]
 assert a['atom_id'].endswith(f'A{num:04d}')
 if '@' in s:
  ix=int(s.split('@')[1])
 elif s.endswith('!'):
  assert a['events'][-1]['kind']=='stage',(a['atom_id'],s)
  ix=len(a['events'])-2
 elif s.endswith('^'):
  ix=max(i for i,e in enumerate(a['events']) if e['kind'] in ('verse','prose'))
 else:ix=len(a['events'])-1
 assert ix>=0
 return a['events'][ix]['event_id']
def build(base):
 decisions=json.loads((base/'review/decisions.json').read_text('utf-8'))
 frozen=json.loads((base/'review/frozen_split_plan.json').read_text('utf-8'))
 scenes=[]; source_atoms=[]; works=[]; structure=[]; headers=[]; issues=[]; characters=[]; inline_markup=[];raw_hashes={}
 for work,filename in FILES.items():
  raw=(base/'raw'/filename).read_bytes();raw_hashes[work]=sha_bytes(raw)
  parser=TEIParser(raw,work);ss,aa,meta=parser.run()
  assert meta['source_root_xml_id']==('ger000243' if work=='F1' else 'ger000201')
  body=parser.root.find('t:text/t:body',NS); assert body is not None
  meta.update(source_filename=filename,provenance='user_uploaded_TEI_snapshot',repository_commit=None)
  meta['license_statements']=[{'xpath':parser.paths[id(e)],'text':norm(''.join(e.itertext())),'target':e.get('target')} for e in parser.root.findall('.//t:licence',NS)]
  works.append(meta)
  for el in parser.root.findall('.//t:listPerson/t:person',NS)+parser.root.findall('.//t:listPerson/t:personGrp',NS):
   characters.append({'work_id':work,'character_id':el.get('{'+XML+'}id'),'source_kind':tag(el),
    'names':[norm(''.join(ch.itertext())) for ch in el if tag(ch) in ('persName','name')],
    'source_attributes':dict(el.attrib),'source_xpath':parser.paths[id(el)]})
  for el in body.iter():
   if tag(el) in ('emph','hi'):
    inline_markup.append({'work_id':work,'source_xpath':parser.paths[id(el)],'tag':tag(el),
     'text':norm(''.join(el.itertext())),'source_attributes':dict(el.attrib)})
  for el in body.iter():
   if tag(el) in ('div','head'):
    row={'work_id':work,'source_xpath':parser.paths[id(el)],'kind':tag(el),'type':el.get('type'),'text':norm(''.join(el.itertext())) if tag(el)=='head' else None}
    if tag(el)=='div':row['direct_titles']=[norm(''.join(x.itertext())) for x in el.findall('t:head',NS)]
    (headers if tag(el)=='head' else structure).append(row)
  for a in aa:
   for i,e in enumerate(a['events']):
    e.update(event_id=f'{a["atom_id"]}_E{i+1:04d}',source_atom_id=a['atom_id'],scene_id=a['scene_id'],work_id=work,
             source_speakers=a['speakers'][:],effective_speakers=a['speakers'][:],speaker_attribution='source_sp_who' if a['speakers'] else 'not_specified')
   if a['kind']=='speech' and not a['speaker_labels']:
    issues.append({'issue_id':a['atom_id']+'_speaker_label','kind':'speaker_label_not_separate','source_atom_id':a['atom_id'],
     'resolution_zh':'原文以舞台说明而非 speaker 元素标示合说；保留 who 的多说话人信息，正文不新增人物标签。','status':'reviewed_preserved'})
   if a['scene_id']=='F1_S018' and a['kind']=='lg':
    for e in a['events']:e['effective_speakers']=['gretchen'];e['speaker_attribution']='inferred_from_explicit_scene_stage'
  # Clear source-encoding role shifts, determined from actual source text, not rewritten.
  for aid,prefix,newwho,reason in [
    ('F1_S009_A0034','Mephistopheles in obiger Stellung.','mephistopheles','同一 sp 内舞台说明明确改为魔鬼发言；仅更正之后的说话人元数据。'),
    ('F1_S013_A0010','Marthe durchs Vorhängel guckend.','marthe','同一 sp 内舞台说明明确转到玛尔特；仅更正随后诗行的说话人元数据。'),
    ('F2_S027_A0011','Engel schwebend in der höheren Atmosphäre','chor_der_engel','童子 sp 中以舞台说明引入天使；随后天使诗行的说话人用人物表已有的天使合唱标识，保留源 who。')]:
   matches=[a for a in aa if a['atom_id']==aid]
   if not matches:continue
   a=matches[0];active=False;changed=[]
   for e in a['events']:
    if e['kind']=='stage' and e['text'].startswith(prefix):active=True
    elif active and e['kind'] in ('verse','prose'):
     e['effective_speakers']=[newwho];e['speaker_attribution']='reviewed_explicit_inline_stage';changed.append(e['event_id'])
   assert changed,(aid,prefix)
   issues.append({'issue_id':aid+'_inline_role','kind':'inline_role_change_within_source_sp','source_atom_id':aid,'affected_event_ids':changed,
                  'evidence':prefix,'resolution_zh':reason+' 原文、标点和标签文本均不改写。','status':'reviewed_metadata_annotation'})
  for a in aa:
   for e in a['events']:
    if e['text'].startswith("Faust. Die Mütter! Mütter!"):
     e['effective_speakers']=['faust'];e['speaker_attribution']='reviewed_explicit_inline_speaker'
     issues.append({'issue_id':e['event_id']+'_inline_faust','kind':'inline_speaker_in_other_sp','source_atom_id':a['atom_id'],
       'affected_event_ids':[e['event_id']],'source_xpath':e['source_xpath'],'evidence':e['text'],
       'resolution_zh':'原始 who 属于魔鬼，但此诗行明写 Faust.；保留原文，只将有效说话人元数据标为浮士德。','status':'reviewed_metadata_annotation'})
  # Embedded songs/exchanges whose source only identifies their situation, not each line.
  for aid,stageprefix,label,reason in [
    ('F1_S024_A0007','Faust, Mephistopheles, Irrlicht im Wechselgesang.', ['faust','mephistopheles','irrlicht'], '鬼火 sp 中嵌入三人交替歌唱；后续歌唱仅标明共同参与者，不凭空指定每一行由谁演唱。'),
    ('F1_S028_A0001','Er ergreift das Schloß. Es singt inwendig.', [], '浮士德 sp 中嵌入牢房内歌声；这里未显式逐行指定歌者，因此元数据保留为未指名的内场歌声，不误归给浮士德。')]:
   matches=[a for a in aa if a['atom_id']==aid]
   if not matches:continue
   active=False;changed=[]
   for e in matches[0]['events']:
    if e['kind']=='stage' and e['text'].startswith(stageprefix):active=True
    elif active and e['kind']=='verse':
     e['effective_speakers']=label;e['speaker_attribution']='source_inline_ensemble' if label else 'source_inner_song_unspecified';changed.append(e['event_id'])
   assert changed,(aid,stageprefix)
   issues.append({'issue_id':aid+'_embedded_song','kind':'embedded_song','source_atom_id':aid,'affected_event_ids':changed,'evidence':stageprefix,'resolution_zh':reason,'status':'reviewed_preserved_with_annotation'})
  scenes.extend(ss);source_atoms.extend(aa)
 assert raw_hashes==frozen['source_hashes'],'Frozen split source mismatch'
 assert set(decisions)=={s['scene_id'] for s in scenes}
 segments=[];atoms=[];event_rows=[];ledger=[]
 for scene in scenes:
  sid=scene['scene_id']; dec=decisions[sid]; aa=[a for a in source_atoms if a['scene_id']==sid]
  ev=[e for a in aa for e in a['events']]; pos={e['event_id']:i for i,e in enumerate(ev)}
  cut_ids=[anchor_index(aa,c['after']) for c in dec['cuts']]; stops=[pos[x]+1 for x in cut_ids]
  assert stops==sorted(set(stops)) and all(0<x<len(ev) for x in stops),(sid,stops)
  bounds=[0]+stops+[len(ev)]
  split=frozen['group_assignment'][scene['scene_group_id']]
  scene.update(title_zh=dec['title_zh'],split=split,segment_ids=[],source_headings=[h for h in headers if h['work_id']==scene['work_id'] and h['source_xpath'].startswith(scene['source_xpath']+'/t:head[')])
  for k,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
   se=ev[lo:hi];pid=f'{sid}_P{k+1:03d}'; text=render(se); newatoms=[]
   for parent,g in __import__('itertools').groupby(se,key=lambda e:e['source_atom_id']):
    ee=list(g);original=next(a for a in aa if a['atom_id']==parent)
    first=next(i for i,e in enumerate(original['events']) if e['event_id']==ee[0]['event_id']);last=first+len(ee)
    aid=parent if first==0 and last==len(original['events']) else f'{parent}_e{first+1:04d}-{last:04d}'
    a={'atom_id':aid,'source_atom_id':parent,'segment_id':pid,'scene_id':sid,'work_id':scene['work_id'],'split':split,
       'source_xpath':original['source_xpath'],'source_kind':original['kind'],'is_source_atom_slice':len(ee)!=len(original['events']),
       'event_ids':[e['event_id'] for e in ee],'effective_speakers':list(dict.fromkeys(w for e in ee for w in e['effective_speakers'])),
       'text':render(ee)}
    atoms.append(a);newatoms.append(aid)
   spoken=[e for e in se if e['kind'] in ('verse','prose')];verses=[e for e in se if e['kind']=='verse']
   adjacent_before=ev[max(0,lo-18):lo];adjacent_after=ev[hi:min(len(ev),hi+12)]
   internal_end=k<len(bounds)-2
   note=dec['cuts'][k]['reason_zh'] if internal_end else dec['whole_scene_note_zh'] if k==0 else '到达原始场景末尾；不跨越原场景边界。'
   continues=bool(lo and ev[lo-1]['source_atom_id']==se[0]['source_atom_id'] and se[0]['kind']=='verse')
   row={'segment_id':pid,'work_id':scene['work_id'],'scene_id':sid,'scene_group_id':scene['scene_group_id'],
        'scene_title':scene['title'],'scene_title_zh':dec['title_zh'],'hierarchy':scene['hierarchy'],'split':split,
        'target_text':text,'target_text_sha256':sha_text(text),'atom_ids':newatoms,
        'source_atom_ids':list(dict.fromkeys(e['source_atom_id'] for e in se)),'event_ids':[e['event_id'] for e in se],
        'source_sha256':raw_hashes[scene['work_id']],
        'source_xpath_start':se[0]['source_xpath'],'source_xpath_end':se[-1]['source_xpath'],
        'source_verse_numbers':None,'source_verse_number_status':'absent_in_uploaded_TEI',
        'verse_element_index_range':[verses[0]['verse_index_in_work'],verses[-1]['verse_index_in_work']] if verses else None,
        'verse_element_count':len(verses),'prose_element_count':sum(e['kind']=='prose' for e in se),
        'source_speakers':list(dict.fromkeys(w for e in se for w in e['source_speakers'])),
        'speakers':list(dict.fromkeys(w for e in se for w in e['effective_speakers'])),
        'speaker_labels_in_source':[e['text'] for e in se if e['kind']=='speaker'],
        'word_count':len(text.split()),'character_count':len(text),'token_count':None,
        'word_count_method':'Unicode whitespace-delimited tokens, not model subword tokens',
        'continues_source_speech':continues,'recommended_task':'continuation_or_description_with_speaker_context' if continues else 'description_to_original',
        'context_before_text':render(adjacent_before),'context_after_text':render(adjacent_after),
        'context_before_event_ids':[e['event_id'] for e in adjacent_before],
        'context_after_event_ids':[e['event_id'] for e in adjacent_after],
        'context_after_is_reference_only':True,
        'previous_segment_id':f'{sid}_P{k:03d}' if k else None,
        'next_segment_id':f'{sid}_P{k+2:03d}' if internal_end else None,
        'segmentation_method':'TEI_source_anchored_model_reviewed_boundaries',
        'semantic_review_status':'reviewed_by_ChatGPT_in_this_conversation',
        'boundary_after_reason_zh':note,'scene_review_note_zh':dec['whole_scene_note_zh'],
        'source_issue_ids':[issue['issue_id'] for issue in issues if set(issue.get('affected_event_ids',[])) & set(e['event_id'] for e in se) or issue.get('source_atom_id') in list(dict.fromkeys(e['source_atom_id'] for e in se))]}
   assert spoken,pid
   segments.append(row);scene['segment_ids'].append(pid)
   ledger.append({'boundary_id':pid+'_END','segment_id':pid,'scene_id':sid,'split':split,
                  'boundary_type':'internal_reviewed' if internal_end else 'original_scene_end',
                  'cut_after_event_id':se[-1]['event_id'],'next_event_id':ev[hi]['event_id'] if internal_end else None,
                  'left_context':render(se[-12:]),'right_context':render(ev[hi:min(len(ev),hi+12)]),
                  'reason_zh':note,'review_status':'reviewed','reviewer':'ChatGPT','human_expert_certification':False})
  event_rows.extend(ev)
 jwrite(base/'data/works.json',works);jwrite(base/'data/document_structure.json',structure)
 jlwrite(base/'data/source_headings.jsonl',headers);jlwrite(base/'data/source_atoms.jsonl',source_atoms)
 jlwrite(base/'data/characters.jsonl',characters);jlwrite(base/'data/inline_markup.jsonl',inline_markup)
 jlwrite(base/'data/scenes.jsonl',scenes);jlwrite(base/'data/events.jsonl',event_rows)
 jlwrite(base/'data/atoms.jsonl',atoms);jlwrite(base/'data/master_segments.jsonl',segments)
 for sp in ('train','validation','test'):jlwrite(base/f'splits/{sp}.jsonl',[r for r in segments if r['split']==sp])
 jlwrite(base/'review/boundary_review.jsonl',ledger);jlwrite(base/'review/source_annotations.jsonl',issues)
 jwrite(base/'verification/source_manifest.json',{'source_kind':'user_uploaded_TEI_snapshot','source_hashes':raw_hashes,
  'git_commit':None,'independently_downloaded_upstream':False,'external_collation_performed':False,
  'edition_from_TEI_header':'Hamburger Ausgabe, Band III, vierte Auflage, Wegner 1959',
  'explanatory_note_zh':'本包以用户上传的两个文件为完整处理范围；没有伪称从某个 Git 提交下载，也未把保存时间当作原文件下载时间。'})
 preview=[];directory=[]
 for s in scenes:
  ss=[r for r in segments if r['scene_id']==s['scene_id']]
  directory.append(f'{s["scene_id"]}　{s["title_zh"]}　{s["title"]}\n集合：{s["split"]}；片段数：{len(ss)}\n')
  for r in ss:
   directory.append(f'{r["segment_id"]}　{r["word_count"]} 个空白分词；{r["verse_element_count"]} 个诗行元素。边界理由：{r["boundary_after_reason_zh"]}\n')
   preview.append(f'【{r["segment_id"]}】{r["scene_title_zh"]} / {r["scene_title"]}\n集合：{r["split"]}；空白分词数：{r["word_count"]}；边界已由本轮模型复核。\n切分说明：{r["boundary_after_reason_zh"]}\n\n{r["target_text"]}\n\n'+ '═'*45+'\n\n')
 (base/'corpus_preview_de.txt').write_text(''.join(preview),encoding='utf-8')
 (base/'切分目录_中文.txt').write_text('\n'.join(directory),encoding='utf-8')
 jwrite(base/'verification/build_manifest.json',{'version':'2.0','source_hashes':raw_hashes,
  'decision_sha256':sha_bytes((base/'review/decisions.json').read_bytes()),'builder_sha256':sha_bytes(Path(__file__).read_bytes()),
  'external_llm_api_calls':0,'semantic_review':'Performed by ChatGPT in the conversation; frozen decisions reproduced by script',
  'sentence_rewriting':False,'original_verse_numbers_invented':False,'model_token_count_available':False})
 print(json.dumps({'segments':len(segments),'scenes':len(scenes),'events':len(event_rows),'atoms':len(atoms),
  'words':sum(s['word_count'] for s in segments),'min_words':min(s['word_count'] for s in segments),'max_words':max(s['word_count'] for s in segments),
  'source_annotation_records':len(issues),'cuts':len(ledger)-len(scenes)},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT);args=p.parse_args();build(args.base)
