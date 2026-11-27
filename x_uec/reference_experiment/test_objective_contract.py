"""Independent characterization of the unchanged v3 objective and MM block."""
from types import SimpleNamespace
import unittest
import numpy as np
import caia

class ObjectiveContractTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(721);dims=dict(qwen=8,harrier=5,gte=6);n=9
        x={m:rng.normal(size=(n,d)) for m,d in dims.items()};x={m:a/np.linalg.norm(a,axis=1,keepdims=True) for m,a in x.items()}
        v={m:rng.uniform(.001,.07,a.shape) for m,a in x.items()}
        r={'qwen':np.eye(8)}
        for m in ['harrier','gte']:r[m]=np.linalg.qr(rng.normal(size=(8,dims[m])))[0].T
        ids=[str(i) for i in range(n)]
        return x,v,r,caia.compute_mean_stats(SimpleNamespace(x=x,ids=ids)),caia.compute_cov_stats(SimpleNamespace(v=v,ids=ids,source='toy'))

    def test_principal_root_for_rectangular_isometry(self):
        x,v,r,s,k=self.fixture()
        for m in r:
            for row in v[m]:
                p=r[m].T@np.diag(np.sqrt(row))@r[m]
                covariance=r[m].T@np.diag(row)@r[m]
                np.testing.assert_allclose(p@p,covariance,atol=1e-14,rtol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(p).min(),-1e-14)

    def test_user_objective_equals_code_times_three_n(self):
        x,v,r,s,k=self.fixture();n=len(x['qwen']);gamma=.37
        z={m:x[m]@r[m] for m in r};u=sum(z.values())/3
        mean=sum(np.sum((a-u)**2) for a in z.values());cov=0.
        names=list(r)
        for i in range(n):
            roots={m:r[m].T@np.diag(np.sqrt(v[m][i]))@r[m] for m in r}
            cov+=sum(np.sum((roots[a]-roots[b])**2) for j,a in enumerate(names) for b in names[j+1:])
        actual=caia.objective(r,s,k,gamma)
        np.testing.assert_allclose([actual['mean'],actual['cov'],actual['total']],[mean/(3*n),cov/(3*n),(mean+gamma*cov)/(3*n)],atol=1e-14,rtol=1e-12)

    def test_svd_matrix_is_correct_mm_linearization(self):
        x,v,r,s,k=self.fixture();gamma=.37;m='harrier';rng=np.random.default_rng(723)
        expected=sum(s.S[m,n]@r[n]+6*gamma*(k.K[m,n]*(r[m]@r[n].T))@r[n] for n in r if n!=m)
        actual=caia.block_matrix(m,r,s,k,gamma);np.testing.assert_allclose(actual,expected,atol=1e-14)
        direction=rng.normal(size=r[m].shape);eps=1e-6
        def h(value):return sum(float(np.sum(value*(s.S[m,n]@r[n])))+3*gamma*float(np.sum(k.K[m,n]*(value@r[n].T)**2)) for n in r if n!=m)
        finite_difference=(h(r[m]+eps*direction)-h(r[m]-eps*direction))/(2*eps)
        self.assertAlmostEqual(finite_difference,float(np.sum(actual*direction)),places=8)
        candidate=caia.polar_rows(actual)[0]
        self.assertLessEqual(caia.objective(dict(r,harrier=candidate),s,k,gamma)['total'],caia.objective(r,s,k,gamma)['total']+1e-12)

if __name__=='__main__':unittest.main()
