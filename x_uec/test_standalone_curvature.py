import itertools,unittest,tempfile,json
from pathlib import Path
import numpy as np
from unittest.mock import patch
import torch
from standalone.curvature import batch_ggn,coordinate_audit,fit_gaussian

class LinearPath:
    def __init__(self):self.weight=torch.tensor([[.3,.2],[.1,-.4]],dtype=torch.float64);self.device=self.weight.device
    def forward(self,f):return self.weight@f['h'],None
    def raw_vjp(self,features,a,chunk=8):return a.T@torch.stack([f['h'] for f in features])
    def coordinate_jacobian(self,features,c):
        rows=c//2;cols=c%2;selector=torch.nn.functional.one_hot(rows,num_classes=2).T.double()
        return torch.stack([selector*f['h'][cols][None] for f in features])

class FittingPath(LinearPath):
    name='qwen';initial_hash='fixed';policy={'versions':{'torch':'test'}};asset_hashes={'tokenizer.json':'original'}
    def capture(self,texts):return [dict(h=torch.tensor([float(t)+1,.2],dtype=torch.float64)) for t in texts],None

class CurvatureTests(unittest.TestCase):
    def test_failure_publishing_progress_preserves_previous_checkpoint(self):
        import standalone.curvature as module
        pairs=[dict(query=str(i),positive=str(5-i)) for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);real=module.write_json;calls=0
            def fail_second(path,value):
                nonlocal calls
                if Path(path).name=='progress.json':
                    calls+=1
                    if calls==2:raise OSError('simulated failure publishing progress')
                return real(path,value)
            with patch.object(module,'write_json',side_effect=fail_second),self.assertRaisesRegex(OSError,'publishing progress'):
                fit_gaussian(FittingPath(),pairs,folder,probes=128,batch_size=2,data_sha256='data')
            # Must load the first valid generation, recompute the second, then finish.
            variance,_=fit_gaussian(FittingPath(),pairs,folder,probes=128,batch_size=2,data_sha256='data')
            self.assertTrue(torch.isfinite(variance).all())

    def test_gaussian_reuse_rejects_tokenizer_or_actual_runtime_change(self):
        pairs=[dict(query=str(i),positive=str(3-i)) for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);fit_gaussian(FittingPath(),pairs,folder,probes=128,batch_size=2,data_sha256='data')
            changed=FittingPath();changed.asset_hashes={'tokenizer.json':'changed'}
            with self.assertRaisesRegex(ValueError,'source/runtime'):
                fit_gaussian(changed,pairs,folder,probes=128,batch_size=2,data_sha256='data')
            changed=FittingPath();changed.policy={'versions':{'torch':'changed'}}
            with self.assertRaisesRegex(ValueError,'source/runtime'):
                fit_gaussian(changed,pairs,folder,probes=128,batch_size=2,data_sha256='data')
    def test_interrupted_gaussian_resumes_without_recomputing_completed_batch(self):
        class Fitter(LinearPath):
            name='qwen';initial_hash='fixed';policy={};asset_hashes={}
            def __init__(self,fail_at=None):super().__init__();self.calls=0;self.fail_at=fail_at
            def capture(self,texts):
                self.calls+=1
                if self.calls==self.fail_at:raise RuntimeError('simulated interruption')
                return [dict(h=torch.tensor([float(t)+1,.2],dtype=torch.float64)) for t in texts],None
        pairs=[dict(query=str(i),positive=str(5-i)) for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'resumed';b=Path(tmp)/'fresh'
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                fit_gaussian(Fitter(fail_at=2),pairs,a,probes=128,batch_size=2,physical_batch=8,data_sha256='data')
            self.assertEqual(json.loads((a/'progress.json').read_text())['completed_batches'],1)
            m=Fitter();v,_=fit_gaussian(m,pairs,a,probes=128,batch_size=2,physical_batch=8,data_sha256='data')
            self.assertEqual(m.calls,2)
            expected,_=fit_gaussian(Fitter(),pairs,b,probes=128,batch_size=2,physical_batch=8,data_sha256='data')
            self.assertTrue(torch.equal(v,expected))
            self.assertTrue(np.array_equal(np.load(a/'diagonal_ggn.npy'),np.load(b/'diagonal_ggn.npy')))

    def test_exhaustive_noise_equals_sum_ggn_with_both_endpoints(self):
        m=LinearPath();fs=[dict(h=torch.tensor(h,dtype=torch.float64)) for h in [[1,.4],[.2,1],[-.3,.8],[.6,-.5]]]
        noise=[torch.tensor(x,dtype=m.weight.dtype).reshape(2,2) for x in itertools.product([-1.,1.],repeat=4)]
        result=batch_ggn(m,fs,torch.arange(4),noise)
        def score(w):
            a=torch.stack([w@f['h'] for f in fs]);return a[:2]@a[2:].T
        scores=score(m.weight);p=scores.softmax(-1);j=torch.autograd.functional.jacobian(score,m.weight).reshape(2,2,4)
        centered=j-(p[:,:,None]*j).sum(1,keepdim=True);expected=(p[:,:,None]*centered.square()).sum((0,1)).reshape(2,2)
        self.assertTrue(torch.allclose(result['diagonal'],expected,atol=1e-12,rtol=1e-10))
        self.assertTrue(torch.allclose(result['exact_coordinates'],expected.flatten(),atol=1e-12))
        self.assertEqual(result['probes'],16)

    def test_coordinate_audit_handles_zero_standard_error_and_rejects_mismatch(self):
        exact=torch.zeros(3);samples=torch.zeros(32,3)
        self.assertTrue(coordinate_audit(samples,exact)['passed'])
        samples[:,0]=1
        self.assertFalse(coordinate_audit(samples,exact)['passed'])

    def test_coordinate_audit_uses_original_absolute_floor_for_near_zero_curvature(self):
        samples=torch.zeros(32,3,dtype=torch.float64)
        self.assertTrue(coordinate_audit(samples,torch.full((3,),1e-9,dtype=torch.float64))['passed'])
        self.assertFalse(coordinate_audit(samples,torch.full((3,),1e-7,dtype=torch.float64))['passed'])

if __name__=='__main__':unittest.main()
