"""Independent numerical and rejection tests for raw Gaussian moments."""
import copy
import unittest
import numpy as np
import covariance_ops as ops

ORDER = ('qwen', 'harrier', 'gte')

def fixture(seed=42, dtype=np.float64):
    rng = np.random.default_rng(seed)
    maps = {'qwen': np.eye(8, dtype=dtype)}
    for m, d in zip(ORDER[1:], [5, 6]):
        maps[m] = np.linalg.qr(rng.normal(size=(8, d)))[0].T.astype(dtype)
    def side(n, role):
        means, variances = {}, {}
        ids = np.array([f'toy:{role}:{i}' for i in range(n)])
        for m in ORDER:
            x = rng.normal(size=(n, maps[m].shape[0]))
            means[m] = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(dtype)
            variances[m] = rng.uniform(0.001, 0.2, x.shape).astype(dtype)
        return dict(native_means=means, native_vars=variances,
                    alpha=np.full((n, 3), 1/3, dtype=dtype),
                    mean_ids_by_model={m: ids.copy() for m in ORDER},
                    variance_ids_by_model={m: ids.copy() for m in ORDER},
                    alpha_ids=ids.copy(), role=role)
    return maps, side(4, 'query'), side(8, 'document')

class ContractTests(unittest.TestCase):
    def test_shapes_ids_and_coefficients(self):
        maps, q, _ = fixture()
        ops.prepare_raw_bundle(maps=maps, **q)
        changes = [
            lambda x: x['mean_ids_by_model']['gte'].__setitem__(slice(None), x['alpha_ids'][::-1]),
            lambda x: x['variance_ids_by_model']['harrier'].__setitem__(slice(None), x['alpha_ids'][::-1]),
            lambda x: x['alpha_ids'].__setitem__(slice(None), x['alpha_ids'][::-1]),
            lambda x: x['native_vars']['qwen'].__setitem__((0, 0), -1),
            lambda x: x['native_vars']['gte'].__setitem__((0, 0), np.inf),
            lambda x: x['native_means']['harrier'].__setitem__((0, 0), np.nan),
            lambda x: x['alpha'].__setitem__((0, 0), -0.1),
            lambda x: x['alpha'].__setitem__((0, 0), np.inf),
            lambda x: x['alpha'].__setitem__((0, 0), .6),
            lambda x: x['native_vars'].__setitem__('gte', np.zeros((4, 5))),
            lambda x: x.__setitem__('role', ''),
            lambda x: x['native_means'].__setitem__('gte', x['native_means']['gte'].astype(int)),
        ]
        for change in changes:
            x = copy.deepcopy(q); change(x)
            with self.subTest(change=changes.index(change)), self.assertRaises(ValueError):
                ops.prepare_raw_bundle(maps=maps, **x)
        for replacement in [np.ones((5, 8)), np.full((5, 8), np.nan), np.ones((5, 7))]:
            wrong = dict(maps, harrier=replacement)
            with self.assertRaises(ValueError): ops.prepare_raw_bundle(maps=wrong, **q)
        with self.assertRaises(ValueError):
            ops.prepare_raw_bundle(maps=maps, model_order=ORDER[::-1], **q)

    def test_no_implicit_renormalization_or_input_mutation(self):
        maps, q, d = fixture()
        q['native_means']['qwen'] *= 2.5  # raw API accepts non-unit means
        saved = copy.deepcopy((maps, q, d))
        bundle = ops.prepare_raw_bundle(maps=maps, **q)
        expected = sum(q['alpha'][:, i:i+1] * (q['native_means'][m] @ maps[m]) for i,m in enumerate(ORDER))
        np.testing.assert_allclose(bundle['fused_mean'], expected, atol=1e-15)
        dense = ops.pair_moments_dense_reference(q, d, maps)
        np.testing.assert_allclose(dense['dot_mean'], expected @ sum(d['alpha'][:,i:i+1] * (d['native_means'][m] @ maps[m]) for i,m in enumerate(ORDER)).T)
        def same(a,b):
            if isinstance(a, dict):
                for k in a: same(a[k],b[k])
            elif isinstance(a, tuple):
                for x,y in zip(a,b): same(x,y)
            else: np.testing.assert_array_equal(a,b)
        same((maps,q,d),saved)

    def test_mixed_numerical_dtypes_are_rejected(self):
        maps,q,d=fixture()
        for component,model in [('native_means','harrier'),('native_vars','gte'),('alpha',None)]:
            side=copy.deepcopy(q)
            if model:side[component][model]=side[component][model].astype(np.float32)
            else:side[component]=side[component].astype(np.float32)
            with self.assertRaises(ValueError):ops.prepare_raw_bundle(maps=maps,**side)
        mixed=dict(maps);mixed['harrier']=mixed['harrier'].astype(np.float32)
        with self.assertRaises(ValueError):ops.build_overlap_blocks(mixed,ORDER)
        m32,q32,d32=fixture(dtype=np.float32)
        qb=ops.prepare_raw_bundle(maps=m32,**q32);db=ops.prepare_raw_bundle(maps=maps,**d)
        with self.assertRaises(ValueError):ops.pair_moments_implicit(qb,db,ops.build_overlap_blocks(maps,ORDER))

class MomentTests(unittest.TestCase):
    def compare(self, a, b, dtype=np.float64):
        atol, rtol = (1e-6,1e-4) if dtype == np.float32 else (1e-10,1e-8)
        for k in ops.TERMS:
            np.testing.assert_allclose(a[k],b[k],atol=atol,rtol=rtol,err_msg=k)

    def calculate(self, maps, q, d):
        qb = ops.prepare_raw_bundle(maps=maps,**q)
        db = ops.prepare_raw_bundle(maps=maps,**d)
        return ops.pair_moments_implicit(qb, db, ops.build_overlap_blocks(maps,ORDER))

    def test_each_moment_matches_dense(self):
        for dtype in [np.float64,np.float32]:
            for seed in [42,43,44]:
                maps,q,d = fixture(seed,dtype)
                rng=np.random.default_rng(seed+100)
                conditions=[(np.full((4,3),1/3),np.full((8,3),1/3)),
                    (np.tile([.6,.2,.2],(4,1)),np.tile([.6,.2,.2],(8,1))),
                    (rng.dirichlet([1,1,1],4),rng.dirichlet([1,1,1],8))]
                conditions += [(np.tile(np.eye(3)[i],(4,1)),np.tile(np.eye(3)[j],(8,1))) for i in range(3) for j in range(3)]
                for qa,da in conditions:
                    q['alpha']=qa.astype(dtype); d['alpha']=da.astype(dtype)
                    with self.subTest(seed=seed,dtype=dtype,weights=qa[0]):
                        self.compare(self.calculate(maps,q,d),ops.pair_moments_dense_reference(q,d,maps),dtype)

    def test_same_model_invariance(self):
        maps,q,d = fixture()
        for i,m in enumerate(ORDER):
            q['alpha']=np.tile(np.eye(3)[i],(4,1)); d['alpha']=np.tile(np.eye(3)[i],(8,1))
            fq,fd=q['native_means'][m],d['native_means'][m]
            vq,vd=q['native_vars'][m],d['native_vars'][m]
            expected=dict(zip(ops.TERMS,[fq@fd.T,fq**2@vd.T,vq@(fd**2).T,vq@vd.T,
                fq**2@vd.T+vq@(fd**2).T+vq@vd.T]))
            self.compare(self.calculate(maps,q,d),expected)

    def test_cross_model_onehot(self):
        maps,q,d = fixture()
        for i,j in [(0,1),(1,0),(1,2),(2,1),(2,0),(0,2)]:
            q['alpha']=np.tile(np.eye(3)[i],(4,1));d['alpha']=np.tile(np.eye(3)[j],(8,1))
            self.compare(self.calculate(maps,q,d),ops.pair_moments_dense_reference(q,d,maps))

    def test_zero_variance(self):
        maps,q,d=fixture()
        for side in [q,d]:
            for a in side['native_vars'].values(): a.fill(0)
        result=self.calculate(maps,q,d)
        for k in ops.TERMS[1:]: np.testing.assert_array_equal(result[k],0)

    def test_overlap_blocks(self):
        maps,q,d=fixture(); overlap=ops.build_overlap_blocks(maps,ORDER)
        for m in ORDER:
            np.testing.assert_allclose(overlap['blocks'][m,m],np.eye(maps[m].shape[0]),atol=1e-14)
            for n in ORDER:
                np.testing.assert_array_equal(overlap['blocks'][m,n],overlap['blocks'][n,m].T)
        qb=ops.prepare_raw_bundle(maps=maps,**q); db=ops.prepare_raw_bundle(maps=maps,**d)
        different=fixture(43)[0]
        with self.assertRaises(ValueError):
            ops.pair_moments_implicit(qb,db,ops.build_overlap_blocks(different,ORDER))
        with self.assertRaises(ValueError):
            ops.pair_moments_implicit(db,qb,overlap)

    def test_pair_budget_and_two_sided_blocks(self):
        maps,q,d=fixture(); qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
        ov=ops.build_overlap_blocks(maps,ORDER)
        with self.assertRaises(MemoryError): ops.pair_moments_implicit(qb,db,ov,memory_budget_bytes=1)
        with self.assertRaises(MemoryError): ops.pair_moments_dense_reference(q,d,maps,memory_budget_bytes=1)
        result={k:np.zeros((4,8)) for k in ops.TERMS}
        for qs,ds,block in ops.iter_pair_moments(qb,db,ov,query_block=2,document_block=3):
            for k in result: result[k][qs,ds]=block[k]
        self.compare(result,ops.pair_moments_dense_reference(q,d,maps))

class DiagonalTests(MomentTests):
    def test_aligned_diagonal_matches_dense_diagonal(self):
        maps,q,d=fixture()
        qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
        qref=ops.prepare_dense_reference(q,maps);dref=ops.prepare_dense_reference(d,maps)
        for ref in [qref,dref]:
            c=ref['covariance']
            for i in range(len(c)): c[i]=np.diag(np.diag(c[i]))
        diag=ops.pair_moments_diagonal_ablation(qb,db,maps)
        self.compare(diag,ops.pair_moments_prepared_dense(qref,dref))
        full=self.calculate(maps,q,d)
        self.assertGreater(np.max(np.abs(full['dot_variance']-diag['dot_variance'])),1e-5)
        np.testing.assert_array_equal(diag['dot_mean'],full['dot_mean'])

    def test_qwen_identity_diagonal_ablation(self):
        maps,q,d=fixture()
        q['alpha']=np.tile([1.,0.,0.],(4,1));d['alpha']=np.tile([1.,0.,0.],(8,1))
        qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
        self.compare(self.calculate(maps,q,d),ops.pair_moments_diagonal_ablation(qb,db,maps))

if __name__ == '__main__': unittest.main()
