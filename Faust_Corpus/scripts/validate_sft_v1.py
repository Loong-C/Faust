#!/usr/bin/env python3
"""Independently inspect Stage-2 data and frozen Stage-1 inputs. Python 3.10+.
Automated checks do not certify semantic interpretation or absence of prior model
exposure. Semantic review is separately declared as assistant self-review.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path

def jl(p): return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def h(b):return hashlib.sha256(b).hexdigest()
def words(t):return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+",t.casefold(),re.U)
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def audit(base):
    parent=json.loads((base/'verification/sft_v1/parent_snapshot.json').read_text(encoding='utf-8'))
    master=jl(base/'data/master_segments.jsonl'); pairs=jl(base/'data/sft_v1/faust_sft_pairs.jsonl')
    ann=jl(base/'review/sft_v1/annotations.jsonl'); specs=jl(base/'data/sft_v1/scene_specs.jsonl')
    checks=[]
    def check(name,truth,details=None):
        c={'check':name,'passed':bool(truth)}
        if details is not None:c['details']=details
        checks.append(c)
    check('exactly_195_rows',len(master)==len(pairs)==len(ann)==len(specs)==195)
    ids=[m['segment_id'] for m in master]
    for label,data in [('pairs',pairs),('annotations',ann),('specifications',specs)]:
        check(label+'_same_IDs_and_order',[r['segment_id'] for r in data]==ids)
        check(label+'_unique_IDs',len({r['segment_id'] for r in data})==len(data))
    changed=[r['path'] for r in parent['files'] if not (base/r['path']).is_file() or
             h((base/r['path']).read_bytes())!=r['sha256']]
    check('all_parent_files_byte_identical',not changed,{'file_count':len(parent['files']),'changed':changed})
    check('all_targets_identical_to_parent',all(p['target_text']==m['target_text'] for p,m in zip(pairs,master)))
    check('all_target_hashes_valid',all(h(p['target_text'].encode())==p['target_text_sha256']==m['target_text_sha256']==a['target_text_sha256'] for p,m,a in zip(pairs,master,ann)))
    check('all_instruction_hashes_valid',all(h(p['instruction_de'].encode())==p['instruction_sha256'] for p in pairs))
    check('splits_inherited_unchanged',all(p['split']==m['split'] for p,m in zip(pairs,master)))
    check('source_locations_preserved',all(all(p['source'][k]==m[k] for k in p['source']) for p,m in zip(pairs,master)))
    check('structured_fields_match_authored_annotations',all(all(p['scene_specification'][k]==a[k]==s[k] for k in ['form_de','situation_de','beats_de','description_zh']) for p,a,s in zip(pairs,ann,specs)))
    check('end_state_is_final_beat',all(s['ending_state_de']==s['beats_de'][-1] for s in specs))
    check('no_empty_annotation_fields',all(all(a[k].strip() for k in ['form_de','situation_de','description_zh']) and all(b.strip() for b in a['beats_de']) for a in ann))
    check('at_least_three_ordered_beats_each',all(3<=len(a['beats_de'])<=9 for a in ann))
    check('Chinese_only_in_audit_fields',all(not re.search(r'[\u3400-\u9fff]',p['instruction_de']) for p in pairs))
    check('Chinese_mirror_present_each',all(re.search(r'[\u3400-\u9fff]',p['description_zh']) for p in pairs))
    # Reconstruct prompt suffix independently, rather than importing builder helpers.
    check('prompt_uses_only_allowed_annotation_fields',all(p['instruction_de'].split('\n\n',1)[1]==
       'Form: '+a['form_de']+'\nAusgangslage: '+a['situation_de']+'\nAblauf: '+' '.join(a['beats_de']) for p,a in zip(pairs,ann)))
    prefixes={p['instruction_de'].split('\n\n',1)[0] for p in pairs}
    check('generic_prefix_is_constant',len(prefixes)==1)
    check('no_segment_ID_in_prompt',all(p['segment_id'] not in p['instruction_de'] for p in pairs))
    check('no_XML_in_prompt',all('<TEI' not in p['instruction_de'] and '<sp' not in p['instruction_de'] for p in pairs))
    check('no_target_contained_in_own_prompt',all(p['target_text'] not in p['instruction_de'] for p in pairs))
    check('no_style_analysis_request',all(not re.search(r'Goethes? Stil|imitier|Stilanalyse|Knittelvers|Kreuzreim',p['instruction_de'],re.I) for p in pairs))
    all_indexes=[]; split_stats={}
    for label,expected in [('train',158),('validation',18),('test',19)]:
        sub=[p for p in pairs if p['split']==label]
        out=jl(base/f'splits/sft_v1/{label}.jsonl'); index=jl(base/f'splits/sft_v1/{label}_index.jsonl')
        check(label+'_row_count',len(out)==len(index)==len(sub)==expected)
        check(label+'_only_prompt_completion',all(set(o)=={'prompt','completion'} for o in out))
        check(label+'_content_exact',all(o=={'prompt':p['instruction_de'],'completion':p['target_text']} for o,p in zip(out,sub)))
        check(label+'_index_ID_order',[x['segment_id'] for x in index]==[p['segment_id'] for p in sub])
        check(label+'_index_positions',all(x['row_index_zero_based']==i for i,x in enumerate(index)))
        check(label+'_index_hashes',all(x['target_text_sha256']==p['target_text_sha256'] and x['instruction_sha256']==p['instruction_sha256'] for x,p in zip(index,sub)))
        all_indexes += [x['segment_id'] for x in index]
        split_stats[label]={'rows':len(sub),'scene_groups':len({p['scene_group_id'] for p in sub})}
    check('split_export_complete_nonoverlapping',set(all_indexes)==set(ids) and len(all_indexes)==195)
    groups=collections.defaultdict(set)
    for p in pairs:groups[p['scene_group_id']].add(p['split'])
    check('scene_groups_never_cross_splits',all(len(v)==1 for v in groups.values()))
    check('all_instructions_distinct',len({p['instruction_de'] for p in pairs})==195)
    check('all_original_targets_distinct',len({p['target_text_sha256'] for p in pairs})==195)
    check('no_missing_source_event_ids',all(p['source']['event_ids'] for p in pairs))
    check('no_source_event_lost_or_duplicated', [e for p in pairs for e in p['source']['event_ids']]==[e for m in master for e in m['event_ids']] and len({e for p in pairs for e in p['source']['event_ids']})==sum(len(p['source']['event_ids']) for p in pairs))
    check('no_external_expert_claim',all(p['quality']['independent_human_review'] is False for p in pairs))
    check('no_api_claim',all(p['quality']['external_llm_api_called'] is False for p in pairs))
    check('no_unseen_pretraining_claim',all(p['quality']['pretraining_contamination_excluded'] is False for p in pairs))
    check('dedication_not_mislabelled_dialogue',ann[0]['form_de']=='Widmungsgedicht')
    check('prose_scene_forms_preserved','Prosa' in ann[75]['form_de'] and 'Prosa' in ann[76]['form_de'])
    # Phrase similarity is a lexical screen, not a plagiarism or authorship detector.
    target_words=[words(p['target_text']) for p in pairs]
    idx=collections.defaultdict(list)
    for k,w in enumerate(target_words):
        for j in range(len(w)-3):idx[tuple(w[j:j+4])].append((k,j))
    overlaps=[]
    for i,p in enumerate(pairs):
        w=words(p['instruction_de']);matches=set(); longest=0; own=0
        for j in range(len(w)-3):
            for k,t in idx.get(tuple(w[j:j+4]),[]):
                length=4
                while j+length<len(w) and t+length<len(target_words[k]) and w[j+length]==target_words[k][t+length]:length+=1
                longest=max(longest,length)
                if i==k:own=max(own,length)
                matches.add((k,' '.join(w[j:j+length]),length))
        overlaps.append({'segment_id':p['segment_id'],'max_match_words_at_least_4_any_target':longest,
                         'max_match_words_at_least_4_own_target':own,
                         'matches':[{'target_segment_id':pairs[k]['segment_id'],'phrase':s,'length_words':n} for k,s,n in sorted(matches)],
                         'interpretation_zh':('无连续四词及以上匹配。' if not matches else '短语重合已查看；为普通语法、人物名或必要动作表述，不据此认定风格泄漏。')})
    check('no_six_word_original_phrase_in_any_prompt',all(o['max_match_words_at_least_4_any_target']<6 for o in overlaps))
    check('no_Chinese_sidecar_in_training_exports',all(not re.search(r'[\u3400-\u9fff]',r['prompt']) for s in ['train','validation','test'] for r in jl(base/f'splits/sft_v1/{s}.jsonl')))
    text=(base/'docs/sft_v1/剧本说明_中文.txt').read_text(encoding='utf-8')
    check('Chinese_document_covers_all_IDs',all(p['segment_id'] in text and p['description_zh'] in text for p in pairs))
    out=base/'review/sft_v1/lexical_overlap.jsonl'
    out.write_text(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in overlaps),encoding='utf-8')
    lengths=sorted(len(words(p['instruction_de'])) for p in pairs)
    report={'release_id':'faust-sft-0.1.0','all_automated_checks_passed':all(c['passed'] for c in checks),
            'check_count':len(checks),'checks':checks,'failed_checks':[c for c in checks if not c['passed']],
            'rows':len(pairs),'splits':split_stats,'parent_files_unchanged':len(parent['files'])-len(changed),
            'instruction_words':{'min':lengths[0],'median':lengths[len(lengths)//2],'max':lengths[-1]},
            'lexical_screen':{'method':'Unicode words; case-insensitive; punctuation ignored; scan each prompt against all 195 original targets',
                              'rows_with_4_plus_word_matches':sum(bool(o['matches']) for o in overlaps),
                              'max_match_words':max(o['max_match_words_at_least_4_any_target'] for o in overlaps),
                              'rows_with_6_plus_word_matches':sum(o['max_match_words_at_least_4_any_target']>=6 for o in overlaps)},
            'semantic_review':{'reviewed_segments':195,'method':'assistant source reading, annotation and second-pass self-review','independent_external_model_or_expert':False,'external_api_calls':0,
                               'standard_zh':'中性现代德语；按原文推进；区分事实、转述、梦境、意图和人物判断；不过度补全；中文为内容对照而非逐字翻译。'},
            'limitations_zh':['机器检查核验文件、对应关系与字面重合，不证明每项文学理解唯一正确。',
                              '说明有意保留剧情语义，但不逐句保留所有意象、典故和措辞；本数据不是逐句注释版。',
                              '原作已公开多年，不能排除底模预训练见过原文。',
                              '没有运行微调，也没有用特定底模tokenizer计数；训练前须核算prompt与completion总长度。',
                              '各候选模型的模板、EOS和仅completion计算loss须在训练配置中确定。']}
    save(base/'verification/sft_v1/quality_report.json',report)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--corpus',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args()
    try:
        r=audit(args.corpus.resolve())
        print(json.dumps({k:r[k] for k in ['all_automated_checks_passed','check_count','failed_checks','rows','splits','parent_files_unchanged','instruction_words','lexical_screen']},ensure_ascii=False,indent=2))
        raise SystemExit(0 if r['all_automated_checks_passed'] else 1)
    except (OSError,ValueError,KeyError) as exc:ap.exit(2,str(exc)+'\n')
