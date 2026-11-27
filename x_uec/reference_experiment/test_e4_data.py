import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import e4_data as data

class DataTests(unittest.TestCase):
    def pairs(self):
        return [dict(source='ms_marco',triplet_id='calibration:0',source_row=8,query='same\r\ntext',positive='document 0',negative='unused'),
                dict(source='snli',triplet_id='calibration:1',source_row=9,query='same\ntext',positive='document 1',negative='unused')]

    def test_occurrences_and_roles_preserved_without_negatives(self):
        sources={s:dict(repo=s,revision='revision',split='train') for s in ['ms_marco','snli']}
        observations=data.expand_observations(self.pairs(),sources,'selection-hash')
        self.assertEqual(len(observations),4)
        self.assertEqual(len({r['observation_id'] for r in observations}),4)
        self.assertEqual([r['role'] for r in observations],['query','positive_passage','premise','hypothesis'])
        self.assertEqual(observations[0]['text'],'same\r\ntext')
        self.assertEqual(observations[0]['canonical_text_sha256'],observations[2]['canonical_text_sha256'])
        self.assertEqual([r['moment_side'] for r in observations],['query','document','query','document'])

    def observations(self):
        return [dict(observation_id=f'id:{i}',text=f'text {i//2}',canonical_text_sha256=data.canonical_hash(f'text {i//2}'),
            moment_side='query' if i%2==0 else 'document',role='query' if i%2==0 else 'positive_passage',dataset='test') for i in range(100)]

    def test_groups_exclusions_order_and_smoke_roles(self):
        rows=self.observations();excluded={rows[0]['canonical_text_sha256']}
        split=data.group_split(rows,excluded)
        self.assertEqual(split['excluded_indices'],[0,1])
        fit=split['fit_indices'];diag=split['diagnostic_indices']
        self.assertEqual(fit,sorted(fit));self.assertEqual(diag,sorted(diag))
        self.assertEqual({rows[i]['canonical_text_sha256'] for i in fit}&{rows[i]['canonical_text_sha256'] for i in diag},set())
        smoke=data.select_smoke(rows,split,fit_max=16,diagnostic_max=12)
        self.assertEqual(len(smoke['fit_indices']),16)
        self.assertLessEqual(len(smoke['diagnostic_indices']),12)
        checks=smoke['moment_query_indices']+smoke['moment_document_indices']
        self.assertEqual(len({rows[i]['canonical_text_sha256'] for i in checks}),6)
        self.assertTrue(all(rows[i]['moment_side']=='query' for i in smoke['moment_query_indices']))
        self.assertTrue(all(rows[i]['moment_side']=='document' for i in smoke['moment_document_indices']))
        self.assertEqual(smoke,data.select_smoke(rows,split,fit_max=16,diagnostic_max=12))

    def test_cache_rejects_independent_id_and_numerical_errors(self):
        ids=np.array(['query:0','doc:1']);mean=np.array([[1.,0.],[0.,1.]])
        var=np.full((2,2),.01)
        data.validate_cache_arrays(mean,var,ids,ids,ids,2)
        for a,b,c in [(mean,var,ids[::-1]),(mean,-var,ids),(mean*np.nan,var,ids),(mean[:,:1],var,ids),(mean*2,var,ids)]:
            with self.assertRaises(ValueError):data.validate_cache_arrays(a,b,ids,c,ids,2)

    def test_eval_audit_rejects_missing_manifest_declared_corpus(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);query=root/'queries.jsonl';query.write_text('query\n')
            (root/'manifest.json').write_text(json.dumps(dict(files={'queries.jsonl':data.sha(query),'corpus.jsonl':data.canonical_hash('corpus')})))
            with self.assertRaises((ValueError,FileNotFoundError)):data.checked_evaluation_files(root,'SCIDOCS')

    def test_source_checks_reject_changed_original_fitting_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'triplets.jsonl';p.write_text('original selection')
            consumed={str(p):data.sha(p)}
            data.verify_source_checksums(consumed)
            p.write_text('different selection')
            with self.assertRaises(ValueError):data.verify_source_checksums(consumed)

    def test_live_propagation_code_must_match_original_v3_seal(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'runs/original/v3';parent.mkdir(parents=True)
            live=root/'math.py';live.write_text('original arithmetic')
            record=parent/'input_hashes.json';record.write_text(json.dumps(dict(inputs={str(live):data.sha(live)})))
            (parent/'manifest.json').write_text(json.dumps(dict(files={'input_hashes.json':data.sha(record)})))
            data.verify_parent_sources(parent,root,['math.py'])
            live.write_text('different arithmetic')
            with self.assertRaises(ValueError):data.verify_parent_sources(parent,root,['math.py'])

if __name__=='__main__':unittest.main()
