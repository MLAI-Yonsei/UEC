"""Runner boundaries: independent IDs, status honesty and isolated measurements."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import copy
import weakref
import unittest
import numpy as np
from test_covariance_ops import fixture, ORDER
import run_covariance_validation as runner

def write_role(path, inputs):
    payload={'alpha':inputs['alpha'],'alpha_ids':inputs['alpha_ids'],'role':np.array(inputs['role'])}
    for i in ORDER:
        for key,source in [('mean','native_means'),('variance','native_vars'),('mean_ids','mean_ids_by_model'),('variance_ids','variance_ids_by_model')]:
            payload[key+'__'+i]=inputs[source][i]
    np.savez(path,**payload)

class RunnerTests(unittest.TestCase):
    def real_fixture(self,path):
        """Real dimensions, controlled sealed source, no actual model downloads."""
        source=path/'source';(source/'maps').mkdir(parents=True)
        dims=dict(zip(ORDER,[1024,640,768]));rng=np.random.default_rng(3)
        for role,n in [('query',1),('document',2)]:
            means={m:rng.normal(size=(n,d)) for m,d in dims.items()}
            means={m:x/np.linalg.norm(x,axis=1,keepdims=True) for m,x in means.items()}
            side=runner.make_side(means,{m:np.full((n,d),.001) for m,d in dims.items()},[f'{role}:{i}' for i in range(n)],role)
            write_role(path/(role+'.npz'),side)
        maps={m:np.eye(1024)[:d] for m,d in dims.items()}
        np.savez(source/'maps/A.npz',**maps)
        maps['harrier']=-maps['harrier'];np.savez(source/'maps/C.npz',**maps)
        models={m:dict(model_id=repo,revision='rev-'+m,native_dim=dims[m]) for m,repo in zip(ORDER,['Qwen/Qwen3-Embedding-0.6B','microsoft/harrier-oss-v1-270m','thenlper/gte-base'])}
        runner.write_json(source/'input_hashes.json',dict(anchor_refs={m:dict(encoder_config=v) for m,v in models.items()}))
        sealed={str(p.relative_to(source)):runner.sha(p) for p in [source/'input_hashes.json',source/'maps/A.npz',source/'maps/C.npz']}
        runner.write_json(source/'manifest.json',dict(files=sealed))
        files=[source/'manifest.json',source/'input_hashes.json',source/'maps/A.npz',source/'maps/C.npz',path/'query.npz',path/'document.npz']
        manifest=dict(query_npz=str(path/'query.npz'),document_npz=str(path/'document.npz'),maps=dict(GPA=str(source/'maps/A.npz'),Teacher_CAIA=str(source/'maps/C.npz')),
            models=models,source_run=str(source),consumed_files={str(p):runner.sha(p) for p in files},output_stage='l2_normalized',
            covariance_representation='native_output_diagonal_surrogate',uncertainty_source='controlled fixture',original_encoding_role='query')
        return manifest

    def test_e0_rejects_wrong_model_arm_role_and_float32_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);manifest=self.real_fixture(p);file=p/'manifest.json';runner.write_json(file,manifest)
            runner.load_real_manifest(file)
            changes=[lambda x:x['models']['gte'].__setitem__('revision','wrong-revision'),
                lambda x:x['models']['gte'].__setitem__('model_id','wrong-model'),
                lambda x:x.__setitem__('maps',{}),
                lambda x:x['maps'].pop('Teacher_CAIA'),
                lambda x:x['maps'].update(dict(GPA=x['maps']['Teacher_CAIA'],Teacher_CAIA=x['maps']['GPA'])),
                lambda x:x.update(query_npz=x['document_npz'],document_npz=x['query_npz'])]
            for mutate in changes:
                invalid=copy.deepcopy(manifest);mutate(invalid);runner.write_json(file,invalid)
                with self.subTest(change=changes.index(mutate)),self.assertRaises(ValueError):runner.load_real_manifest(file)
            # A fully checksummed float32 reference is still not an admissible
            # float64 base map for this experiment.
            target=Path(manifest['maps']['GPA'])
            with np.load(target) as data:maps={m:data[m].astype(np.float32) for m in ORDER}
            np.savez(target,**maps)
            manifest['consumed_files'][str(target)]=runner.sha(target)
            sealfile=p/'source/manifest.json';seal=json.loads(sealfile.read_text());seal['files']['maps/A.npz']=runner.sha(target)
            runner.write_json(sealfile,seal);manifest['consumed_files'][str(sealfile)]=runner.sha(sealfile)
            runner.write_json(file,manifest)
            with self.assertRaises(ValueError):runner.load_real_manifest(file)
    def test_loader_compares_every_id_before_bundle(self):
        maps,q,d=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'q.npz';write_role(path,q)
            loaded=runner.load_role_npz(path)
            np.testing.assert_array_equal(loaded['variance_ids_by_model']['gte'],q['alpha_ids'])
            q['variance_ids_by_model']['gte']=q['alpha_ids'][::-1]
            write_role(path,q)
            with self.assertRaises(ValueError):
                runner.ops.prepare_raw_bundle(maps=maps,**runner.load_role_npz(path))

    def test_synthetic_run_is_fresh_and_real_steps_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'
            status=runner.run_validation(out,dict(seeds=[42],dtypes=['float64'],benchmark=False))
            self.assertEqual(status['E1_synthetic'],'passed')
            self.assertEqual(status['E1_real'],'blocked_missing_cache')
            self.assertFalse(status['downstream_evaluated'])
            self.assertFalse(status['uec_fusion_executed'])
            with self.assertRaises(FileExistsError):runner.run_validation(out,{})
            self.assertTrue((out/'numeric_equivalence.csv').exists())
            self.assertEqual(json.loads((out/'uec_integration.json').read_text())['uec_integration_ready'],False)

    def test_benchmark_worker_is_fresh_and_separates_costs(self):
        maps,q,d=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_role(p/'q.npz',q);write_role(p/'d.npz',d);np.savez(p/'maps.npz',**maps)
            proc=subprocess.run([sys.executable,runner.__file__,'--worker',str(p),'--method','implicit',
                '--nq','4','--nd','8','--dtype','float64'],check=True,capture_output=True,text=True)
            row=json.loads(proc.stdout)
            self.assertGreater(row['peak_rss_bytes'],0)
            self.assertEqual(row['warmups'],2);self.assertEqual(row['repeats'],5)
            self.assertEqual(row['nq'],4);self.assertEqual(row['nd'],8)
            for phase in ['overlap','document_prepare','query_prepare','score']:
                self.assertGreaterEqual(row[phase+'_median_seconds'],0)
            self.assertLess(row['per_document_cache_bytes'],8*8*8)

    def test_e0_failure_records_no_e2_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);manifest=p/'missing.json';manifest.write_text('{}')
            with self.assertRaises((ValueError,KeyError)):
                runner.run_validation(p/'out',dict(seeds=[42],dtypes=['float64'],benchmark=False),manifest)
            status=json.loads((p/'out/status.json').read_text())
            self.assertEqual(status['status'],'failed')
            self.assertNotEqual(status.get('E2'),'completed')

    def test_worker_peak_is_not_inherited_from_large_parent(self):
        maps,q,d=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_role(p/'q.npz',q);write_role(p/'d.npz',d);np.savez(p/'maps.npz',**maps)
            # A real parent with a 128 MiB live allocation launches the toy
            # worker. Linux getrusage high-water can inherit that parent's floor.
            script='''import subprocess,sys
held=bytearray(128*1024**2)
p=subprocess.run(sys.argv[1:],check=True,capture_output=True,text=True)
print(p.stdout)
'''
            proc=subprocess.run([sys.executable,'-c',script,sys.executable,runner.__file__,
                '--worker',str(p),'--method','implicit','--nq','4','--nd','8','--dtype','float64'],
                check=True,capture_output=True,text=True)
            self.assertLess(json.loads(proc.stdout)['peak_rss_bytes'],96*1024**2)

    def test_diagnostic_pair_selection_does_not_allocate_cartesian_product(self):
        # Ten billion possible pairs must require only the requested 1000 IDs.
        qids=np.array([f'q:{i}' for i in range(100000)])
        dids=np.array([f'd:{i}' for i in range(100000)])
        pairs=runner.select_diagnostic_pairs(qids,dids,1000,42)
        self.assertEqual(len(pairs),1000)
        self.assertEqual(len(set(pairs)),1000)
        self.assertEqual(pairs,runner.select_diagnostic_pairs(qids,dids,1000,42))

    def test_measurement_releases_previous_trial_before_allocating_next(self):
        refs=[]
        def allocate():
            self.assertTrue(not refs or refs[-1]() is None)
            value=np.zeros(10);refs.append(weakref.ref(value));return value
        result,median,maximum=runner.measured(allocate)
        self.assertEqual(len(refs),7);self.assertEqual(result.shape,(10,))

    def test_worker_rejects_requested_counts_exceeding_available_cache(self):
        maps,q,d=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_role(p/'q.npz',q);write_role(p/'d.npz',d);np.savez(p/'maps.npz',**maps)
            with self.assertRaises(ValueError):runner.benchmark_worker(p,'implicit',9,8,'float64')

    def test_unsupported_recorded_config_values_are_rejected(self):
        for key,value in [('warmups',3),('repeats',2),('dense_memory_budget_mib',64),('query_scale_count',4)]:
            with tempfile.TemporaryDirectory() as tmp:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    runner.run_validation(Path(tmp)/'out',{key:value,'benchmark':False,'seeds':[42],'dtypes':['float64']})

    def test_zero_variance_real_cache_has_na_relative_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);manifest=self.real_fixture(p)
            for field in ['query_npz','document_npz']:
                filename=Path(manifest[field]);side=runner.load_role_npz(filename)
                for value in side['native_vars'].values():value.fill(0)
                write_role(filename,side);manifest['consumed_files'][str(filename)]=runner.sha(filename)
            runner.write_json(p/'manifest.json',manifest)
            status=runner.run_validation(p/'out',dict(benchmark=False,seeds=[42],dtypes=['float64'],diagnostic_pairs=2),p/'manifest.json')
            self.assertEqual(status['E2'],'completed');self.assertIn('N/A',(p/'out/report.md').read_text())

if __name__ == '__main__':unittest.main()
