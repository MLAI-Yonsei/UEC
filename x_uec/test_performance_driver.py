import importlib.util,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from standalone.storage import sha

def driver():
    spec=importlib.util.spec_from_file_location('performance_driver',Path(__file__).parent/'tools/run_alignment.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class PerformanceDriverTests(unittest.TestCase):
    def test_map_fit_input_digest_must_match_scored_inputs_even_after_move(self):
        m=driver()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);root.mkdir(exist_ok=True)
            for name in ['alignment_fit.npz','uec_input_provenance.json','manifest.json']:(root/name).write_text('source-'+name)
            receipt=dict(input_files={'/original/inputs/'+name:sha(root/name) for name in ['alignment_fit.npz','uec_input_provenance.json','manifest.json']})
            m.verify_alignment_input_binding(root,receipt)
            (root/'uec_input_provenance.json').write_text('different valid preparation')
            with self.assertRaisesRegex(ValueError,'fitted input'):m.verify_alignment_input_binding(root,receipt)

    def test_internal_matching_uses_only_diagnostic_endpoints_and_all_known_positives(self):
        m=driver()
        rows=[]
        for i,(q,p) in enumerate([('q','a'),('q','b'),('other','c'),('train','heldout')]):
            for role,text in [('query',q),('positive_passage',p)]:
                rows.append(dict(dataset='ms_marco',original_triplet_id=str(i),observation_id=str(len(rows)),moment_side='query' if role=='query' else 'document',role=role,text=text,canonical_text_sha256=m.canonical_hash(text)))
        arrays={name:np.arange(len(rows)*d,dtype=float).reshape(len(rows),d)+1 for name,d in [('qwen',4),('harrier',3),('gte',2)]}
        data=SimpleNamespace(observations=rows,means=arrays)
        task,native,audit=m.diagnostic_task(data,np.array([0,1,2,3,4,5,7]),'ms_marco')
        self.assertEqual(task['kind'],'Retrieval');self.assertEqual(len(task['roles']['queries']),2)
        self.assertEqual(len(task['roles']['corpus']),4)
        qid=task['roles']['queries'][0]['row_id']
        self.assertEqual(len(task['qrels'][qid]),2)
        self.assertTrue(set(task['qrels'][qid])<=set(r['row_id'] for r in task['roles']['corpus']))
        self.assertEqual(native['queries']['gte'].shape,(2,2))
        self.assertTrue(audit['alignment_only_heldout']);self.assertFalse(audit['posterior_unseen'])

    def test_internal_matching_rejects_missing_both_sides(self):
        m=driver();rows=[dict(dataset='snli',original_triplet_id='0',observation_id='0',moment_side='query',role='premise',text='q',canonical_text_sha256=m.canonical_hash('q'))]
        data=SimpleNamespace(observations=rows,means={'gte':np.ones((1,2))})
        with self.assertRaisesRegex(ValueError,'positive'):m.diagnostic_task(data,np.array([0]),'snli')

if __name__=='__main__':unittest.main()
