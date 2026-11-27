import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import full_audit as audit
from e4_data import group_split,select_scope,canonical_hash

class AuditTests(unittest.TestCase):
    def test_mutually_truncated_cache_and_split_are_rejected(self):
        rows=[dict(source_order=i,observation_id=str(i),moment_side='query' if i%2==0 else 'document',canonical_text_sha256=canonical_hash(str(i))) for i in range(100)]
        full=group_split(rows,set());scope=select_scope(rows,full,'full')
        record=dict(full=full,selected_scope=scope)
        audit.verify_full_partition(rows,set(),record,rows)
        record['full']['fit_indices']=record['full']['fit_indices'][:-1]
        record['selected_scope']=select_scope(rows,record['full'],'full')
        selected=[rows[i] for i in record['selected_scope']['selected_indices']]
        with self.assertRaisesRegex(ValueError,'replayed full partition'):
            audit.verify_full_partition(rows,set(),record,selected)

    def test_opened_paths_must_be_individually_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files={k:root/k for k in ['input_path','observations_path','split_path']}
            for file in files.values():file.write_text('original')
            p={k:str(v) for k,v in files.items()};p['consumed_files']=audit.snapshot_paths(files.values())
            audit.require_consumed_paths(p)
            forged=root/'unbound';forged.write_text('original');p['input_path']=str(forged)
            with self.assertRaisesRegex(ValueError,'not checksum-bound'):audit.require_consumed_paths(p)

    def test_pre_execution_snapshot_rejects_source_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'worker.py';file.write_text('original')
            snapshot=audit.snapshot_paths([file]);audit.verify_snapshot(snapshot)
            file.write_text('edited after import')
            with self.assertRaisesRegex(ValueError,'changed during full run'):audit.verify_snapshot(snapshot)

    def test_evaluation_audit_reconstructs_ten_real_file_records(self):
        from e4_data import TASKS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);consumed={}
            for task in TASKS:
                folder=root/task;folder.mkdir()
                names=['corpus.jsonl','queries.jsonl'] if task in TASKS[:2] else ['support.jsonl','test.jsonl'] if task in TASKS[2:4] else ['texts.jsonl']
                for name in names:(folder/name).write_text(json.dumps(dict(text=task+'\u2028'+name),ensure_ascii=False)+'\n')
                checks=audit.snapshot_paths(folder/name for name in names)
                (folder/'manifest.json').write_text(json.dumps(dict(files={Path(p).name:v for p,v in checks.items()})))
                consumed.update(audit.snapshot_paths([folder/'manifest.json',*[folder/name for name in names]]))
            hashes,scope=audit.replay_evaluation(consumed)
            self.assertEqual(len(scope),10);self.assertEqual(len(hashes),10)
            self.assertTrue(all(set(r)=={'task','file','rows'} for r in scope))

if __name__=='__main__':unittest.main()
