#!/usr/bin/env python3
"""Regression checks: the auditor must also reject corrupted delivery metadata.
No source text leaves the machine. Temporary copies are deleted after each test.
"""
from pathlib import Path
import json, shutil, tempfile, unittest
from audit import audit, ROOT

class AuditRegressionTests(unittest.TestCase):
    def test_original_delivery_passes(self):
        self.assertTrue(audit(ROOT)['all_checks_passed'])

    def mutate_and_audit(self, relpath, mutate):
        with tempfile.TemporaryDirectory(prefix='faust-audit-test-') as td:
            base=Path(td)
            for name in ('raw','data','review','splits','verification'):
                shutil.copytree(ROOT/name,base/name)
            f=base/relpath
            if str(f).endswith('.jsonl'):
                rows=[json.loads(l) for l in f.read_text('utf-8').splitlines() if l.strip()]
                mutate(rows)
                f.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
            else:
                obj=json.loads(f.read_text('utf-8'));mutate(obj)
                f.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
            return audit(base)

    def test_detects_changed_target_text(self):
        report=self.mutate_and_audit('data/master_segments.jsonl',lambda rows:rows[0].update(target_text=rows[0]['target_text']+' EXTRA'))
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('every_target_word_and_punctuation_matches_source',{c['check'] for c in report['failed_checks']})

    def test_detects_reordered_targets(self):
        def swap(rows): rows[0],rows[1]=rows[1],rows[0]
        report=self.mutate_and_audit('data/master_segments.jsonl',swap)
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('target_segments_partition_events_once_in_source_order',{c['check'] for c in report['failed_checks']})

    def test_detects_missing_boundary_record(self):
        report=self.mutate_and_audit('review/boundary_review.jsonl',lambda rows:rows.pop())
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('every_delivered_boundary_has_review_record',{c['check'] for c in report['failed_checks']})

    def test_detects_false_download_provenance(self):
        report=self.mutate_and_audit('verification/source_manifest.json',lambda obj:obj.update(source_kind='downloaded_commit_pinned_TEI',git_commit='0'*40))
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('no_false_commit_pinned_download_claim',{c['check'] for c in report['failed_checks']})

    def test_detects_fabricated_model_token_counts(self):
        report=self.mutate_and_audit('data/master_segments.jsonl',lambda rows:rows[0].update(token_count=123))
        self.assertFalse(report['all_checks_passed'])
        self.assertIn('not_claiming_unknown_model_token_counts',{c['check'] for c in report['failed_checks']})

if __name__=='__main__': unittest.main(verbosity=2)
