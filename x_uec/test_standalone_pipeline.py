import json,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from standalone.storage import seal,write_json,write_jsonl

class PortablePipelineTests(unittest.TestCase):
    def test_same_work_directory_cannot_have_two_preparation_drivers(self):
        from standalone.pipeline import preparation_lock
        with tempfile.TemporaryDirectory() as tmp:
            with preparation_lock(Path(tmp)):
                with self.assertRaisesRegex(RuntimeError,'already running'):
                    with preparation_lock(Path(tmp)):pass

    def test_gpu_transition_waits_for_utilization_but_rejects_real_clients(self):
        import prepare_alignment as module
        rows=['1, GPU-one, 0, 49140, 90\n','','1, GPU-one, 0, 49140, 0\n','']
        with patch.object(module.subprocess,'check_output',side_effect=rows),patch('time.sleep'):
            self.assertEqual(module.check_gpu(1,settle_utilization=True),'GPU-one')
        with patch.object(module.subprocess,'check_output',side_effect=['1, GPU-one, 0, 49140, 90\n','GPU-one, 99\n']),patch('time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'busy'):module.check_gpu(1,settle_utilization=True)
            sleep.assert_not_called()
    def test_export_uses_normalized_output_and_source_encoder(self):
        import torch
        from standalone.export import export_model
        class Fake:
            name='gte';initial_hash='encoder';policy={};asset_hashes={}
            def capture(self,texts):return [None]*len(texts),torch.nn.functional.one_hot(torch.zeros(len(texts),dtype=torch.long),768).double()
            def forward(self,f):
                unit=torch.zeros(768,dtype=torch.float64);unit[0]=1
                return unit*7,unit
            def diagonal(self,f,v):return torch.ones(768,dtype=torch.float64)
            def verify_unchanged(self):pass
            def parameter_hash(self):return 'encoder'
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);data=work/'data';data.mkdir()
            write_jsonl(data/'observations.jsonl',[dict(observation_id='one',text='test')])
            seal(data,['observations.jsonl'],dict(profile='smoke'))
            gauss=work/'models/gte/gaussian';gauss.mkdir(parents=True);seal(gauss,[],{})
            export_model(Fake(),None,work)
            self.assertEqual(float(np.load(work/'models/gte/mean.npy')[0,0]),1.)

    def test_storage_environment_keeps_caches_under_work_directory(self):
        from standalone.pipeline import environment
        with tempfile.TemporaryDirectory() as tmp:
            env=environment(Path(tmp))
            for key in ['HF_HOME','XDG_CACHE_HOME','TMPDIR','TORCH_HOME']:
                self.assertTrue(Path(env[key]).is_relative_to(tmp))

    def test_array_export_checks_mean_and_variance_independent_ids(self):
        from standalone.export import validate_arrays
        ids=np.array(['a','b']);mean=np.array([[1.,0],[0,1.]])
        validate_arrays(mean,np.ones_like(mean),ids,ids,ids,2)
        with self.assertRaisesRegex(ValueError,'IDs'):
            validate_arrays(mean,np.ones_like(mean),ids,ids[::-1],ids,2)
        with self.assertRaisesRegex(ValueError,'variance'):
            validate_arrays(mean,-np.ones_like(mean),ids,ids,ids,2)

    def test_completed_stage_rejects_configuration_change(self):
        from standalone.pipeline import bind_config
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);bind_config(folder,dict(profile='full'))
            bind_config(folder,dict(profile='full'))
            with self.assertRaisesRegex(ValueError,'configuration'):
                bind_config(folder,dict(profile='smoke'))

    def test_smoke_status_can_never_approve_maps(self):
        from standalone.alignment import profile_status
        self.assertEqual(profile_status('alignment_ready_for_handoff','smoke'),'smoke_pass')
        self.assertEqual(profile_status('review_required','smoke'),'smoke_pass')
        self.assertEqual(profile_status('numerical_failure','smoke'),'numerical_failure')
        self.assertEqual(profile_status('review_required','full'),'review_required')

if __name__=='__main__':unittest.main()
