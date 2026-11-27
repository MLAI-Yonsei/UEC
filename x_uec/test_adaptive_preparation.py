import importlib.util,tempfile,unittest,json
from unittest.mock import patch
from pathlib import Path
import numpy as np

def driver():
    spec=importlib.util.spec_from_file_location('adaptive_preparation',Path(__file__).parent/'tools/prepare_alignment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class AdaptivePreparationTests(unittest.TestCase):
    def test_receipt_publication_recovers_after_receipt_write_before_seal(self):
        from standalone.storage import seal,verify
        m=driver()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);np.save(p/'variance_teacher.npy',np.ones(2));seal(p,['variance_teacher.npy'],dict(probes=64))
            real=m.write_json
            def interrupt(path,value):
                if Path(path).name=='manifest.json':raise OSError('publish interrupted')
                return real(path,value)
            with patch.object(m,'write_json',side_effect=interrupt),self.assertRaisesRegex(OSError,'interrupted'):
                m.attach_receipt(p,dict(actual_probes=64))
            self.assertTrue((p/'refinement_receipt.json').exists())
            m.attach_receipt(p,dict(actual_probes=64))
            self.assertIn('refinement_receipt.json',verify(p)['files'])

    def test_cli_is_validated_before_any_old_work_can_launch(self):
        m=driver()
        with patch.object(m,'failed_precision') as identify,patch.object(m.subprocess,'run') as run:
            with self.assertRaises(SystemExit) as exit:m.main(['--help'])
            self.assertEqual(exit.exception.code,0)
            with self.assertRaises(SystemExit):m.main(['--work-dir','/data2/example'])
            identify.assert_not_called();run.assert_not_called()

    def test_changed_gpu_rejected_before_reading_old_failure(self):
        m=driver()
        with tempfile.TemporaryDirectory() as tmp:
            old=m.parse_cli(['--work-dir',tmp,'--gpu','1']);config=m.effective_config(old)
            (Path(tmp)/'preparation_config.json').write_text(json.dumps(config))
            with patch.object(m,'failed_precision') as identify,patch.object(m.subprocess,'run') as run:
                with self.assertRaisesRegex(ValueError,'configuration'):m.main(['--work-dir',tmp,'--gpu','2'])
                identify.assert_not_called();run.assert_not_called()

    def test_nonfinite_values_and_bad_diagonal_replay_never_trigger_refinement(self):
        from standalone.storage import write_json,seal,sha
        m=driver()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);noise=p/'noise';noise.mkdir();seal(noise,[],dict(sizes=[2],probes=32,seed=241128,device_type='cpu'))
            np.save(p/'batch_order.npy',np.array([0,1]));binding=dict(probes=32,seed=241128,logical_batch=512,order_sha256=sha(p/'batch_order.npy'),noise_manifest_sha256=sha(noise/'manifest.json'))
            coords=np.random.default_rng(241129).choice(4,4,replace=False);samples=np.ones((32,4));exact=np.ones(4)*2;diagonal=np.ones((2,2))
            def state(s,e,k,d):
                np.savez(p/'state.npz',samples=s,exact=e,kl=k,diagonal=d,completed=1)
                write_json(p/'progress.json',dict(completed_batches=1,total_batches=1,binding=binding,checkpoint_file='state.npz',checkpoint_sha256=sha(p/'state.npz')))
            state(samples,exact,0.,diagonal);self.assertIsNotNone(m.precision_failure(p))
            for key in ['samples','exact','kl','diagonal']:
                s=samples.copy();e=exact.copy();d=diagonal.copy();k=0.
                if key=='samples':s[0,0]=np.nan
                elif key=='exact':e[0]=np.nan
                elif key=='kl':k=float('nan')
                else:d[0,0]=5.
                state(s,e,k,d)
                with self.assertRaises(ValueError,msg=key):m.precision_failure(p)

    def test_retry_requires_current_explicit_statistical_outcome(self):
        m=driver()
        self.assertFalse(m.authorizes_retry(1,dict(status='mc_precision_failed',attempt_id='old',probes=64),'new',64))
        self.assertFalse(m.authorizes_retry(1,dict(status='error',attempt_id='new',probes=64),'new',64))
        self.assertTrue(m.authorizes_retry(75,dict(status='mc_precision_failed',attempt_id='new',probes=64),'new',64))

    def test_refinement_schedule_is_fixed_and_bounded(self):
        m=driver();self.assertEqual(m.next_probes(32),64);self.assertEqual(m.next_probes(64),128);self.assertIsNone(m.next_probes(128))
        self.assertEqual(m.required_attempts(64),[32]);self.assertEqual(m.required_attempts(128),[32,64])
        with self.assertRaises(ValueError):m.required_attempts(256)

    def test_prefix_check_rejects_replacement_noise_or_coordinate_samples(self):
        m=driver();a=np.arange(24).reshape(4,6);b=np.concatenate([a,a+1])
        m.check_prefix(a,b)
        b[0,0]=-1
        with self.assertRaisesRegex(ValueError,'prefix'):m.check_prefix(a,b)

    def test_extra_receipt_preserves_all_original_numeric_hashes(self):
        from standalone.storage import seal,verify
        m=driver()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);np.save(p/'variance_teacher.npy',np.array([.1,.2]));before=seal(p,['variance_teacher.npy'],dict(probes=64))
            m.attach_receipt(p,dict(actual_probes=64,reason='coordinate MC precision',unchanged_seed=True))
            after=verify(p);self.assertEqual(before['metadata'],after['metadata']);self.assertEqual(before['files']['variance_teacher.npy'],after['files']['variance_teacher.npy'])
            self.assertIn('refinement_driver.py',after['files']);self.assertIn('refinement_receipt.json',after['files'])

if __name__=='__main__':unittest.main()
