#!/usr/bin/env python3
"""Independently verify the delivered dataset against its unmodified TEI inputs.
Python 3.10+, standard library only. Does not use the extraction class.
Run from any directory: python scripts/audit.py
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
NS = {'t': 'http://www.tei-c.org/ns/1.0'}
XML = 'http://www.w3.org/XML/1998/namespace'
FILES = {
 'F1': ('ger000243-goethe-faust-eine-tragoedie.tei.xml', '24069c0cf849d19a18badfcb7ac0b2e013ac2f9b77111b3a879b71d32ff2aaa4'),
 'F2': ('ger000201-goethe-faust-der-tragoedie-zweiter-teil.tei.xml', '32295a15d07b825b934166a496dfa8321822a9fcff6d162c7025d1483c12ed7c')}
LEAF = {'l':'verse', 'p':'prose', 'speaker':'speaker', 'stage':'stage', 'trailer':'trailer'}
def norm(s): return re.sub(r'[\t\r\n ]+', ' ', s).strip()
def sha(b): return hashlib.sha256(b).hexdigest()
def jl(p): return [json.loads(x) for x in p.read_text('utf-8').splitlines() if x.strip()]
def local(e): return e.tag.rsplit('}',1)[-1]
def paths_of(root):
 result={}
 def visit(e,p):
  result[id(e)]=p; c=collections.Counter()
  for ch in e:
   c[local(ch)]+=1;visit(ch,f'{p}/t:{local(ch)}[{c[local(ch)]}]')
 visit(root,'/t:TEI');return result

def audit(base):
 checks=[]
 def check(name,ok,detail=None):
  row={'check':name,'passed':bool(ok)}
  if detail is not None:row['detail']=detail
  checks.append(row)
 segments=jl(base/'data/master_segments.jsonl');events=jl(base/'data/events.jsonl')
 atoms=jl(base/'data/atoms.jsonl');source_atoms=jl(base/'data/source_atoms.jsonl')
 scenes=jl(base/'data/scenes.jsonl');heads=jl(base/'data/source_headings.jsonl')
 ledger=jl(base/'review/boundary_review.jsonl');issues=jl(base/'review/source_annotations.jsonl')
 decisions=json.loads((base/'review/decisions.json').read_text('utf-8'))
 frozen=json.loads((base/'review/frozen_split_plan.json').read_text('utf-8'))
 manifest=json.loads((base/'verification/source_manifest.json').read_text('utf-8'))
 build_manifest=json.loads((base/'verification/build_manifest.json').read_text('utf-8'))
 lookup={e['event_id']:e for e in events};amap={a['atom_id']:a for a in atoms}; smap={s['segment_id']:s for s in segments}; scene_map={s['scene_id']:s for s in scenes}
 source_stats={}; expected_all=[]; expected_heading=[]; sp_paths={}; raw_hashes={}; source_ids={}; emphasis=[]
 for work,(filename,expected_hash) in FILES.items():
  data=(base/'raw'/filename).read_bytes();raw_hashes[work]=sha(data)
  check(work+'_raw_snapshot_byte_identical',sha(data)==expected_hash)
  root=ET.fromstring(data);paths=paths_of(root);body=root.find('t:text/t:body',NS)
  check(work+'_TEI_identity',root.tag=='{'+NS['t']+'}TEI' and root.get('{'+XML+'}id')==('ger000243' if work=='F1' else 'ger000201') and root.get('{'+XML+'}lang')=='de')
  expected=[];titles=[];visible=[];verse_index=0
  for el in body.iter():
   kind=local(el)
   if kind in LEAF or kind=='head':
    txt=norm(''.join(el.itertext()))
    if txt:
     row={'work_id':work,'source_xpath':paths[id(el)],'kind':LEAF.get(kind,'head'),'text':txt}
     visible.append(row)
     if kind=='head':titles.append(row)
     else:
      if kind=='l':verse_index+=1;row['verse_index_in_work']=verse_index
      expected.append(row)
   if kind=='emph':emphasis.append({'work_id':work,'source_xpath':paths[id(el)],'tag':kind,'text':norm(''.join(el.itertext())),'attributes':el.attrib})
  actual=[e for e in events if e['work_id']==work]
  same=len(expected)==len(actual) and all(all(a.get(k)==v for k,v in x.items()) for x,a in zip(expected,actual))
  check(work+'_all_content_leaves_exact_text_kind_and_order',same,{'expected':len(expected),'actual':len(actual)})
  check(work+'_no_unrepresented_body_text',norm(''.join(body.itertext()))==norm(' '.join(v['text'] for v in visible)))
  actualheads=[h for h in heads if h['work_id']==work]
  check(work+'_all_body_headings_preserved_as_metadata',len(actualheads)==len(titles) and all(h['source_xpath']==v['source_xpath'] and h['text']==v['text'] for h,v in zip(actualheads,titles)))
  raw_sp={paths[id(e)] for e in body.iter() if local(e)=='sp'};sp_paths[work]=raw_sp
  check(work+'_every_original_speech_preserved',raw_sp=={a['source_xpath'] for a in source_atoms if a['work_id']==work and a['kind']=='speech'})
  check(work+'_no_unavailable_standard_verse_number_fabricated',all(not el.get('n') for el in body.iter() if local(el)=='l') and all(s['source_verse_numbers'] is None and s['source_verse_number_status']=='absent_in_uploaded_TEI' for s in segments if s['work_id']==work))
  tc=collections.Counter(local(e) for e in body.iter())
  counts={k:tc[k] for k in ['l','p','sp','speaker','stage','trailer','head']}
  counts['act_nodes']=len(body.findall('.//t:div[@type="act"]',NS));counts['scene_nodes']=len(body.findall('.//t:div[@type="scene"]',NS))
  counts['leaf_content_scenes']=sum(s['work_id']==work for s in scenes)
  counts['bytes']=len(data);counts['source_sha256']=sha(data)
  source_stats[work]=counts
  source_ids[work]={e.get('{'+XML+'}id') for e in root.findall('.//t:person',NS)+root.findall('.//t:personGrp',NS)}
  expected_all+=expected;expected_heading+=titles
 check('unique_segment_identifiers',len(smap)==len(segments))
 check('unique_atom_identifiers',len(amap)==len(atoms))
 check('unique_event_identifiers',len(lookup)==len(events))
 check('unique_source_leaf_addresses',len({(e['work_id'],e['source_xpath']) for e in events})==len(events))
 check('source_atoms_recover_event_stream',[e for a in source_atoms for e in a['events']]==events)
 check('target_segments_partition_events_once_in_source_order',[x for s in segments for x in s['event_ids']]==[e['event_id'] for e in events])
 check('target_segments_partition_atoms_once_in_source_order',[x for s in segments for x in s['atom_ids']]==[a['atom_id'] for a in atoms])
 check('atoms_partition_events_once_in_source_order',[x for a in atoms for x in a['event_ids']]==[e['event_id'] for e in events])
 check('every_target_word_and_punctuation_matches_source',all(norm(s['target_text'])==norm(' '.join(lookup[x]['text'] for x in s['event_ids'])) for s in segments))
 check('target_hashes_and_length_fields_correct',all(sha(s['target_text'].encode())==s['target_text_sha256'] and len(s['target_text'].split())==s['word_count'] and len(s['target_text'])==s['character_count'] for s in segments))
 check('all_segments_contain_spoken_or_poetic_text',all(any(lookup[x]['kind'] in ['verse','prose'] for x in s['event_ids']) for s in segments))
 check('no_target_crosses_scene_boundary',all(all(lookup[x]['scene_id']==s['scene_id'] for x in s['event_ids']) for s in segments))
 check('no_atom_has_wrong_parent_or_order',all(len({lookup[x]['source_atom_id'] for x in a['event_ids']})==1 and lookup[a['event_ids'][0]]['source_atom_id']==a['source_atom_id'] for a in atoms))
 check('source_hashes_consistent',raw_hashes==manifest['source_hashes']==frozen['source_hashes']==build_manifest['source_hashes'] and all(s['source_sha256']==raw_hashes[s['work_id']] for s in segments))
 check('no_false_commit_pinned_download_claim',manifest['source_kind']=='user_uploaded_TEI_snapshot' and manifest['git_commit'] is None and manifest['independently_downloaded_upstream'] is False)
 check('not_claiming_unknown_model_token_counts',all(s['token_count'] is None and 'not model' in s['word_count_method'] for s in segments))
 check('source_speaker_ids_valid',all(all(x in source_ids[e['work_id']] for x in e['source_speakers']) for e in events))
 check('effective_speaker_ids_valid_or_explicitly_unspecified',all(all(x in source_ids[e['work_id']] for x in e['effective_speakers']) for e in events))
 check('annotated_source_cases_have_resolution_and_no_pending_items',all(i['status'].startswith('reviewed') and i['resolution_zh'] for i in issues))
 check('all_scenes_have_frozen_semantic_decisions',set(decisions)==set(scene_map))
 check('every_delivered_boundary_has_review_record',len(ledger)==len(segments) and {r['segment_id'] for r in ledger}==set(smap) and all(r['review_status']=='reviewed' and r['reason_zh'] and r['human_expert_certification'] is False for r in ledger))
 check('frozen_decisions_match_build_manifest',sha((base/'review/decisions.json').read_bytes())==build_manifest['decision_sha256'])
 check('every_review_anchor_is_actual_target_end',all(r['cut_after_event_id']==smap[r['segment_id']]['event_ids'][-1] for r in ledger))
 check('correct_internal_boundary_count',sum(r['boundary_type']=='internal_reviewed' for r in ledger)==len(segments)-len(scenes)==sum(len(d['cuts']) for d in decisions.values()))
 # A continued speech is cut only at an existing stanza change or a stage movement.
 split_speeches=[];invalid_stanza_cuts=[]
 for r in ledger:
  if r['boundary_type']!='internal_reviewed':continue
  left=lookup[r['cut_after_event_id']];right=lookup[r['next_event_id']]
  if left['source_atom_id']==right['source_atom_id']:
   split_speeches.append({'segment_id':r['segment_id'],'source_atom_id':left['source_atom_id'],'left_kind':left['kind'],'right_kind':right['kind']})
   if left['kind']==right['kind']=='verse' and left['stanza_ids']==right['stanza_ids']:invalid_stanza_cuts.append(r['boundary_id'])
 check('no_cut_inside_a_source_poetic_line_or_same_stanza',not invalid_stanza_cuts,invalid_stanza_cuts)
 check('no_speaker_label_detached_at_target_end',all(lookup[s['event_ids'][-1]]['kind']!='speaker' for s in segments))
 groups=collections.defaultdict(set)
 for s in segments:groups[s['scene_group_id']].add(s['split'])
 check('whole_scene_group_split_is_disjoint',all(len(x)==1 for x in groups.values()))
 check('split_assignment_matches_frozen_plan',all(s['split']==frozen['group_assignment'][s['scene_group_id']] for s in segments))
 for split in ['train','validation','test']:
  check(split+'_file_matches_master_subset',jl(base/f'splits/{split}.jsonl')==[s for s in segments if s['split']==split])
 check('context_is_from_same_scene_and_split',all(all(lookup[x]['scene_id']==s['scene_id'] for x in s['context_before_event_ids']+s['context_after_event_ids']) for s in segments))
 check('context_does_not_overlap_its_own_target',all(not set(s['event_ids']) & set(s['context_before_event_ids']+s['context_after_event_ids']) for s in segments))
 check('context_text_matches_its_source_events',all(norm(s['context_before_text'])==norm(' '.join(lookup[x]['text'] for x in s['context_before_event_ids'])) and norm(s['context_after_text'])==norm(' '.join(lookup[x]['text'] for x in s['context_after_event_ids'])) for s in segments))
 check('future_context_clearly_marked_reference_only',all(s['context_after_is_reference_only'] is True for s in segments))
 check('segment_neighbors_match_source_order',all((s['previous_segment_id'] is None or smap[s['previous_segment_id']]['next_segment_id']==s['segment_id']) and (s['next_segment_id'] is None or smap[s['next_segment_id']]['previous_segment_id']==s['segment_id']) for s in segments))
 # Exact duplicates are not confused with repeated refrains at smaller granularity.
 duplicates=collections.defaultdict(list)
 for s in segments:duplicates[s['target_text_sha256']].append(s['segment_id'])
 duplicates=[g for g in duplicates.values() if len(g)>1]
 check('no_exact_duplicate_target_segments',not duplicates,duplicates)
 check('source_poem_and_prose_element_counts_conserved',sum(s['verse_element_count'] for s in segments)==sum(v['l'] for v in source_stats.values()) and sum(s['prose_element_count'] for s in segments)==sum(v['p'] for v in source_stats.values()))
 # Regression cases identified by actual source text.
 def same_segment_for_atom_numbers(scene,ns):
  ids=[f'{scene}_A{n:04d}' for n in ns]
  present=[{s['segment_id'] for s in segments if set(s['source_atom_ids']) & {aid}} for aid in ids]
  return len(set.union(*present))==1
 def segment_for_text(needle):return {s['segment_id'] for s in segments if needle in s['target_text']}
 check('Faust_I_dedication_is_included',sum(s['verse_element_count'] for s in segments if s['scene_id']=='F1_S001')==32)
 check('spinning_room_unlabelled_poetry_is_included',any(s['scene_id']=='F1_S018' and s['speakers']==['gretchen'] for s in segments))
 check('short_open_field_scene_not_lost',len([s for s in segments if s['scene_id']=='F1_S027'])==1 and sum(s['prose_element_count'] for s in segments if s['scene_id']=='F1_S027')==6)
 check('final_chorus_and_Finis_are_preserved',events[-1]['kind']=='trailer' and events[-1]['text']=='Finis.' and 'Das Ewig-Weibliche' in segments[-1]['target_text'])
 check('no_Pfui_direct_rejoinder_split',segment_for_text('Pfui über dich!') & segment_for_text('Ihr habt das Recht, gesittet Pfui zu sagen.'))
 check('rat_song_with_refrains_preserved_together',len(segment_for_text("Es war eine Ratt' im Kellernest,"))==1 and len([s for s in segments if s['scene_id']=='F1_S008' and ("Es war eine Ratt'" in s['target_text'] or 'Als hätte sie Lieb\' im Leibe.' in s['target_text'])])==1)
 check('flea_song_and_chorus_preserved_together',len([s for s in segments if s['scene_id']=='F1_S008' and ('Es war einmal ein König,' in s['target_text'] or 'Wir knicken und ersticken' in s['target_text'])])==1)
 check('inline_Faust_speaker_corrected_in_metadata_only',all(e['effective_speakers']==['faust'] and e['source_speakers']==['mephistopheles'] for e in events if e['text'].startswith('Faust. Die Mütter! Mütter!')))
 check('no_editorial_body_text_unclassified',all(e['kind'] in set(LEAF.values()) for e in events))
 split_stats={}
 for split in ['train','validation','test']:
  rs=[s for s in segments if s['split']==split]
  split_stats[split]={'segments':len(rs),'leaf_scenes':len({s['scene_id'] for s in rs}),'words':sum(s['word_count'] for s in rs),'verse_elements':sum(s['verse_element_count'] for s in rs)}
 lengths=sorted(s['word_count'] for s in segments)
 report={'version':'2.0','all_checks_passed':all(x['passed'] for x in checks),'check_count':len(checks),
  'failed_checks':[c for c in checks if not c['passed']], 'checks':checks,'source_statistics':source_stats,
  'corpus_statistics':{'leaf_content_scenes':len(scenes),'xml_scene_nodes':sum(x['scene_nodes'] for x in source_stats.values()),'source_atoms':len(source_atoms),'partition_atoms':len(atoms),'source_events':len(events),'body_headings':len(heads),'segments':len(segments),'words':sum(lengths),'min_words':min(lengths),'median_words':lengths[len(lengths)//2],'max_words':max(lengths),'internal_boundaries_reviewed':len(segments)-len(scenes),'original_scene_endings_reviewed':len(scenes),'source_annotation_records':len(issues),'splits':split_stats},
  'source_atom_splits':split_speeches,'preserved_inline_emphasis':emphasis,
  'exceptions_resolved':[{'kind':'short_complete_scene','segment_ids':[s['segment_id'] for s in segments if s['word_count']<150],'resolution_zh':'短场完整保留，不增写、不跨场拼接。'},
   {'kind':'long_complete_exchange','segment_ids':[s['segment_id'] for s in segments if s['word_count']>800],'resolution_zh':'森林与岩洞中保留整轮双人交锋，共806个空白分词；不在直接斥责和回应之间切分。这不是800个模型token的限制。'}],
  'semantic_review':{'completed_for_all_delivered_boundaries':True,'reviewer':'ChatGPT in this conversation','external_llm_api_calls':0,'human_literary_expert_review':False,'notes_zh':'判断依据是原文对白、诗节和舞台动作；连续戏剧的片段仍可依赖前文，所有片段附同场上下文。没有遗留给用户的待审核切点。'},
  'limits':{'independent_critical_edition_collation':False,'upstream_git_commit_verified':False,'model_specific_tokenization':False,'pretraining_exposure_ruled_out':False,'synthetic_instructions_generated':False},
  'coverage_scope_zh':'用户上传的两份 TEI 正文全部纳入；章节标题单独保存为元数据，页码不进入训练正文，原始 XML 原样附带。空格与XML排版换行规范化，德文措辞和标点不改写。'}
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT);p.add_argument('--output',type=Path)
 args=p.parse_args();r=audit(args.base)
 output=args.output or args.base/'verification/quality_report.json';output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:r[k] for k in ['all_checks_passed','check_count','failed_checks','corpus_statistics']},ensure_ascii=False,indent=2))
 raise SystemExit(0 if r['all_checks_passed'] else 1)
if __name__=='__main__':main()
