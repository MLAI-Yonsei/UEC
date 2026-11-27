"""Standalone numerical/interface checks; run python -m unittest -v test_caia."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from caia import (NAMES,check_anchor_inputs,fit_caia,transform_gaussian,
                  compute_mean_stats,compute_cov_stats,objective)


def fixture():
    rng=np.random.default_rng(41);n=30
    means={name:rng.normal(size=(n,d)) for name,d in zip(NAMES,[5,3,4])}
    means={name:x/np.linalg.norm(x,axis=1,keepdims=True) for name,x in means.items()}
    variances={name:rng.uniform(.001,.02,size=x.shape) for name,x in means.items()}
    ids={name:np.array([f'pair:{i}:query' for i in range(n)]) for name in NAMES}
    return means,variances,ids


class TestCAIA(unittest.TestCase):
    def test_direct_root_objective_equals_cached_loss(self):
        from types import SimpleNamespace
        x,v,ids=fixture();s=compute_mean_stats(SimpleNamespace(x=x,ids=ids[NAMES[0]].tolist()))
        k=compute_cov_stats(SimpleNamespace(v=v,ids=ids[NAMES[0]].tolist(),source='test'))
        maps={n:np.linalg.svd(a.T@x['qwen'],full_matrices=False)[0]@np.linalg.svd(a.T@x['qwen'],full_matrices=False)[2] for n,a in x.items()}
        z=[x[n]@maps[n] for n in NAMES];u=sum(z)/3
        mean=sum(np.sum((a-u)**2) for a in z)/(3*len(u));cov=0.
        for i in range(len(u)):
            roots=[maps[n].T@np.diag(np.sqrt(v[n][i]))@maps[n] for n in NAMES]
            cov+=sum(np.sum((roots[a]-roots[b])**2) for a,b in [(0,1),(0,2),(1,2)])/(3*len(u))
        out=objective(maps,s,k,.7)
        self.assertAlmostEqual(out['mean'],mean,places=13);self.assertAlmostEqual(out['cov'],cov,places=13)

    def test_fit_preserves_geometry_and_decreases_objective(self):
        x,v,ids=fixture();original={n:a.copy() for n,a in x.items()}
        out=fit_caia(x,v,ids,ids,max_sweeps=15,r0_sweeps=15)
        for arm in ['gpa','caia','shuffled']:
            fitted=out[arm]
            self.assertIn(fitted['status'],['stationary_tolerance','stalled_not_stationary','budget_exhausted'])
            for n,r in fitted['maps'].items():
                np.testing.assert_allclose(r@r.T,np.eye(len(r)),atol=1e-12)
                np.testing.assert_allclose((x[n]@r)@(x[n]@r).T,x[n]@x[n].T,atol=1e-12)
            values=[r['loss']['total'] for r in fitted['trace']]
            self.assertTrue(np.all(np.diff(values)<=1e-11))
        for n in NAMES:np.testing.assert_array_equal(x[n],original[n])
        for arm in ['gpa','caia','shuffled']:
            self.assertAlmostEqual(out[arm]['trace'][0]['loss']['mean'],out['r0']['trace'][-1]['loss']['mean'],places=13)
        evidence=out['gamma_evidence']
        self.assertAlmostEqual(out['gamma'],.1*evidence['direct_mean']/evidence['teacher_cov'])

    def test_row_and_role_mismatch_rejected(self):
        x,v,ids=fixture();bad={**ids,'gte':ids['gte'][::-1]}
        with self.assertRaisesRegex(ValueError,'ID'):check_anchor_inputs(x,v,ids,bad)
        bad={n:a.copy() for n,a in ids.items()};bad['gte'][0]='pair:0:passage'
        with self.assertRaisesRegex(ValueError,'ID'):check_anchor_inputs(x,v,bad,bad)
        bad={n:np.full(len(a),'duplicate') for n,a in ids.items()}
        with self.assertRaises(ValueError):check_anchor_inputs(x,v,bad,bad)

    def test_invalid_norm_variance_and_dimensions_rejected(self):
        x,v,ids=fixture()
        for bad in [2*x['gte'],np.full_like(x['gte'],np.nan),np.zeros_like(x['gte'])]:
            with self.assertRaises(ValueError):check_anchor_inputs({**x,'gte':bad},v,ids,ids)
        with self.assertRaises(ValueError):check_anchor_inputs(x,{**v,'gte':-v['gte']},ids,ids)
        with self.assertRaises(ValueError):check_anchor_inputs(x,{**v,'gte':v['gte'][:,:2]},ids,ids)
        with self.assertRaises(ValueError):fit_caia(x,v,ids,ids,gamma=-1)

    def test_zero_covariance_matches_gpa_continuation(self):
        x,v,ids=fixture();v={n:np.zeros_like(a) for n,a in v.items()}
        out=fit_caia(x,v,ids,ids,gamma=1.,max_sweeps=12,r0_sweeps=3)
        self.assertIn('no_cov_guidance',out['caia']['quality_flags'])
        for n in NAMES:np.testing.assert_array_equal(out['gpa']['maps'][n],out['caia']['maps'][n])

    def test_transport_full_and_diagonal_retains_offdiagonal_and_rank(self):
        r=np.array([[1.,1,0],[-1,1,2.]])
        r=np.linalg.qr(r.T)[0].T;x=np.array([[1.,0],[0.,1]])
        v=np.array([[.2,.7],[.4,.1]])
        mean,cov=transform_gaussian(x,v,r)
        np.testing.assert_allclose(mean,x@r)
        for i in range(2):
            np.testing.assert_allclose(cov[i],r.T@np.diag(v[i])@r,atol=1e-15)
            self.assertEqual(np.linalg.matrix_rank(cov[i]),2)
            self.assertGreaterEqual(np.linalg.eigvalsh(cov[i]).min(),-1e-14)
        self.assertGreater(np.max(abs(cov[0]-np.diag(np.diag(cov[0])))),.01)
        full=np.stack([np.diag(a) for a in v]);m,c=transform_gaussian(x,full,r)
        np.testing.assert_allclose(c,cov,atol=1e-15)

    def test_transport_rejects_invalid_map_and_covariance(self):
        x=np.array([[1.,0]]);r=np.eye(2)
        for cov in [np.array([[-1.,1.]]),np.array([[[1.,2.],[2.,1.]]])]:
            with self.assertRaises(ValueError):transform_gaussian(x,cov,r)
        with self.assertRaises(ValueError):transform_gaussian(x,np.ones_like(x),2*r)

    def test_transport_accepts_float32_normalization_roundoff_without_rescaling(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(3,10)).astype(np.float32)
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        self.assertGreater(np.max(abs(np.linalg.norm(x.astype(np.float64),axis=1)-1)),1e-8)
        variance=np.full(x.shape,.01,dtype=np.float32)
        mean,cov=transform_gaussian(x,variance,np.eye(10))
        np.testing.assert_array_equal(mean,x.astype(np.float64))
        np.testing.assert_array_equal(np.diagonal(cov,axis1=1,axis2=2),variance.astype(np.float64))
        with self.assertRaises(ValueError):transform_gaussian(1.01*x,variance,np.eye(10))

    def test_demo_is_portable_and_refuses_overwrite(self):
        script=Path(__file__).with_name('example.py')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'demo'
            cmd=[sys.executable,str(script),'--demo','--output',str(out)]
            done=subprocess.run(cmd,cwd=directory,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertTrue((out/'caia_maps.npz').exists())
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['synthetic_demo']);self.assertFalse(summary['UEC_fusion_executed'])
            again=subprocess.run(cmd,cwd=directory,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)


if __name__=='__main__':unittest.main()
