import json,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from standalone import records

class StandaloneDataTests(unittest.TestCase):
    def test_main_selection_samples_before_filter_and_continues_same_rng(self):
        import random
        from standalone.records import main_sample_indices,main_pairs
        ms=[dict(query='q'+str(i),passages=dict(passage_text=['p'+str(i)],is_selected=[int(i%2==0)])) for i in range(100)]
        sn=[dict(premise='a'+str(i),hypothesis='b'+str(i),label=i%3) for i in range(100)]
        rng=random.Random(2025);mi=rng.sample(range(100),5);si=rng.sample(range(100),2)
        self.assertEqual(main_sample_indices(100,100),dict(ms_marco=mi,snli=si))
        selected=main_pairs(dict(ms_marco=dict(zip(mi,[ms[i] for i in mi])),snli=dict(zip(si,[sn[i] for i in si]))),dict(ms_marco=mi,snli=si))
        self.assertEqual([r['source_row'] for r in selected if r['source']=='ms_marco'],[i for i in mi if i%2==0])
        self.assertEqual([r['source_row'] for r in selected if r['source']=='snli'],[i for i in si if i%3==0])
        self.assertTrue(all('negative' not in r and 'negative_hash' not in r for r in selected))

    def test_selection_hash_collapses_whitespace_but_observation_hash_preserves_it(self):
        import hashlib
        q='  A  query\r\n';p='A\t passage'
        selection=[dict(source='ms_marco',source_row=0,triplet_id='one',query_hash=hashlib.sha256(b'A query').hexdigest(),
            positive_hash=hashlib.sha256(b'A passage').hexdigest(),negative_hash=None)]
        pair=records.reconstruct_pairs(selection,{'ms_marco':[dict(query=q,passages=dict(passage_text=[p],is_selected=[1]))]})[0]
        self.assertEqual(pair['query'],q);self.assertEqual(pair['positive'],p)
        self.assertNotEqual(records.observations([pair])[0]['canonical_text_sha256'],selection[0]['query_hash'])

    def fixture(self):
        ms=[dict(query='query'+str(i),query_id=i,passages=dict(passage_text=['not selected','positive'+str(i)],is_selected=[0,1])) for i in range(20)]
        sn=[dict(premise='premise'+str(i),hypothesis='hypothesis'+str(i),label=0) for i in range(20)]
        ids=[]
        for source,rows in [('ms_marco',ms),('snli',sn)]:
            for i,row in enumerate(rows):
                q=row.get('query',row.get('premise'));p=row.get('hypothesis','positive'+str(i))
                ids.append(dict(source=source,source_row=i,triplet_id=source+str(i),query_hash=records.canonical_hash(q),positive_hash=records.canonical_hash(p),negative_hash=None,half=i%2))
        return ms,sn,ids

    def test_reference_rows_preserve_text_roles_and_detect_wrong_revision(self):
        ms,sn,ids=self.fixture();pairs=records.reconstruct_pairs(ids,{'ms_marco':ms,'snli':sn})
        self.assertEqual(pairs[0]['positive'],'positive0');self.assertEqual(pairs[20]['query'],'premise0')
        ms[0]['query']='different revision'
        with self.assertRaisesRegex(ValueError,'reference hash'):records.reconstruct_pairs(ids,{'ms_marco':ms,'snli':sn})

    def test_group_split_and_real_sides_are_independent_of_old_workspace(self):
        ms,sn,ids=self.fixture();pairs=records.reconstruct_pairs(ids,{'ms_marco':ms,'snli':sn});rows=records.observations(pairs)
        blocked={rows[0]['canonical_text_sha256']};split=records.split_rows(rows,blocked)
        self.assertIn(0,split['excluded_indices'])
        a={rows[i]['canonical_text_sha256'] for i in split['fit_indices']};b={rows[i]['canonical_text_sha256'] for i in split['diagnostic_indices']}
        self.assertFalse(a&b);self.assertFalse((a|b)&blocked)
        for key,side,n in [('moment_query_indices','query',2),('moment_document_indices','document',4)]:
            self.assertEqual(len(split[key]),n);self.assertTrue(all(rows[i]['moment_side']==side for i in split[key]))
        self.assertEqual(split,records.split_rows(rows,blocked))

    def test_seal_is_portable_and_rejects_modified_data(self):
        from standalone import storage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'values.json';f.write_text('{"value":1}')
            storage.seal(root,['values.json'],dict(kind='test'));self.assertTrue(storage.verify(root))
            self.assertFalse(any('/data' in x for x in json.loads((root/'manifest.json').read_text())['files']))
            f.write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError,'checksum'):storage.verify(root)

    def test_extra_evaluation_exclusion_uses_observation_hash_not_selection_hash(self):
        from standalone.assets import prepare_data
        from standalone.storage import write_jsonl,read_jsonl
        ms,sn,ids=self.fixture();ms[0]['query']='  A  query\r\n'
        ids[0]['query_hash']=records.selection_hash(ms[0]['query'])
        pairs=records.reconstruct_pairs(ids,{'ms_marco':ms,'snli':sn})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evaluation=root/'eval.jsonl';write_jsonl(evaluation,[dict(text=ms[0]['query'])])
            with patch('standalone.assets.download_datasets',return_value={}),patch('standalone.assets.resolve_reference',return_value=pairs):
                prepare_data(root,'full',root/'cache',evaluation_texts=[evaluation])
            self.assertNotIn(ids[0]['triplet_id'],[r['triplet_id'] for r in read_jsonl(root/'data/curvature_pairs.jsonl')])
            self.assertNotIn(ms[0]['query'],[r['text'] for r in read_jsonl(root/'data/observations.jsonl')])

    def test_completed_data_rejects_changed_published_reference(self):
        from standalone import assets,storage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            storage.seal(data,[],dict(profile='full',reference_manifest_sha256='older-reference'))
            with self.assertRaisesRegex(ValueError,'reference changed'):assets.prepare_data(root,'full',root/'cache')

if __name__=='__main__':unittest.main()
