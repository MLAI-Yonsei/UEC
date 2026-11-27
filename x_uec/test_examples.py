import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import example

class ExampleTests(unittest.TestCase):
    def test_implicit_demo_is_portable_checks_dense_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'demo';script=Path(__file__).with_name('example_implicit.py')
            cmd=[sys.executable,str(script),'--output',str(out)]
            done=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['dense_equivalence_passed'])
            self.assertTrue(summary['document_stream_equivalence_passed'])
            self.assertLessEqual(summary['largest_stream_document_bundle_rows'],3)
            self.assertFalse(summary['uec_score_computed'])
            with np.load(out/'raw_moments.npz',allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'dot_mean','variance_from_document','variance_from_query','variance_cross','dot_variance'})
                self.assertTrue(all(z[k].shape==(4,8) and np.isfinite(z[k]).all() for k in z.files))
            again=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)

    def test_alignment_demo_never_exports_per_sentence_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fit';script=Path(__file__).with_name('example.py')
            done=subprocess.run([sys.executable,str(script),'--demo','--r0-sweeps','3','--max-sweeps','3','--output',str(out)],cwd=tmp,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            for path in out.glob('*.npz'):
                with np.load(path,allow_pickle=False) as z:self.assertTrue(all(z[k].ndim<=2 for k in z.files))
            summary=json.loads((out/'summary.json').read_text())
            self.assertFalse(summary['UEC_fusion_executed'])
            self.assertFalse(summary['alignment_ready_for_handoff'])
            self.assertEqual(summary['solver']['caia']['status'],'budget_exhausted')

    def test_real_loader_rejects_bad_variance_ids_and_stage(self):
        order=('qwen','harrier','gte');dims=[1024,640,768]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);payload={}
            ids=np.array(['pair:0:query','pair:0:positive_passage'])
            for m,d in zip(order,dims):
                mean=np.zeros((2,d));mean[:,0]=1
                payload.update({'mean__'+m:mean,'variance__'+m:np.full((2,d),.001),
                    'mean_ids__'+m:ids.copy(),'variance_ids__'+m:ids.copy()})
            prov=dict(dataset='actual fitting selection',split='alignment_fit',selection_manifest_sha256='a'*64,
                models={m:dict(model_id=repo,revision='test-revision',dimension=d) for m,d,repo in zip(order,dims,['Qwen/Qwen3-Embedding-0.6B','microsoft/harrier-oss-v1-270m','thenlper/gte-base'])},
                uncertainty_source='recipient-approved output posterior',output_stage='l2_normalized',
                covariance_representation='native_output_diagonal_surrogate',role_policy='original policies preserved',
                evaluation_overlap_audit=dict(status='completed',fit_evaluation_canonical_overlap=0,scope='fixed fixture evaluation texts'),alignment_diagnostic_separated=True)
            def write():
                np.savez(p/'inputs.npz',**payload);(p/'provenance.json').write_text(json.dumps(prov))
            write();example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            payload['variance_ids__gte']=ids[::-1];write()
            with self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            payload['variance_ids__gte']=ids.copy();prov['output_stage']='pre_l2';write()
            with self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            prov['output_stage']='l2_normalized'
            original_var=payload['variance__gte'].copy()
            for value in [original_var.astype(complex)+1j,original_var.astype(str),original_var.astype(int)]:
                payload['variance__gte']=value;write()
                with self.subTest(dtype=value.dtype),self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            payload['variance__gte']=original_var
            for field,value in [('dataset',''),('uncertainty_source','REPLACE_WITH_ACTUAL_POSTERIOR'),('role_policy','TODO')]:
                original=prov[field];prov[field]=value;write()
                with self.subTest(field=field),self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
                prov[field]=original
            revision=prov['models']['gte']['revision'];prov['models']['gte']['revision']='REPLACE_WITH_ACTUAL_REVISION';write()
            with self.subTest(field='revision'),self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            prov['models']['gte']['revision']=revision
            prov['evaluation_overlap_audit']['fit_evaluation_canonical_overlap']=False;write()
            with self.subTest(field='boolean_overlap'),self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')
            prov['evaluation_overlap_audit']['fit_evaluation_canonical_overlap']=0
            prov['evaluation_overlap_audit'].pop('scope');write()
            with self.subTest(field='missing_scope'),self.assertRaises(ValueError):example.load_real_inputs(p/'inputs.npz',p/'provenance.json')

if __name__=='__main__':unittest.main()
