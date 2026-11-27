"""Raw dot moments for independent native-diagonal Gaussian surrogates.

No normalization, weight selection, posterior fitting, inverse, or UEC score.
All APIs reject invalid input rather than repairing it. Dense is a small oracle.
"""
import hashlib
import numpy as np

MODEL_ORDER = ('qwen', 'harrier', 'gte')
TERMS = ('dot_mean', 'variance_from_document', 'variance_from_query',
         'variance_cross', 'dot_variance')
DEFAULT_BUDGET = 512 * 1024**2

def _float_array(value, name):
    a = np.asarray(value)
    if a.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(a).all():
        raise ValueError(f'{name}: finite float32/float64 required')
    return a

def map_metadata(maps, model_order=MODEL_ORDER):
    if tuple(model_order) != MODEL_ORDER or set(maps) != set(MODEL_ORDER):
        raise ValueError('model order must be qwen, harrier, gte')
    d, common, errors = {}, None, {}
    dtype = None
    digest = hashlib.sha256()
    for m in MODEL_ORDER:
        r = _float_array(maps[m], 'map '+m)
        dtype = r.dtype if dtype is None else dtype
        if r.dtype != dtype: raise ValueError('all maps must have the same dtype')
        if r.ndim != 2 or not 0 < r.shape[0] <= r.shape[1]:
            raise ValueError('invalid map shape')
        common = r.shape[1] if common is None else common
        if r.shape[1] != common: raise ValueError('common dimension mismatch')
        tol = 1e-5 if r.dtype == np.float32 else 1e-8
        err = float(np.max(np.abs(r @ r.T - np.eye(r.shape[0], dtype=r.dtype))))
        if err > tol: raise ValueError('map is not row-isometric: '+m)
        if m == 'qwen' and (r.shape[0] != common or np.max(np.abs(r - np.eye(common))) > tol):
            raise ValueError('qwen map must be identity')
        d[m] = r.shape[0]; errors[m] = err
        digest.update(m.encode()); digest.update(str((r.shape, r.dtype.str)).encode())
        digest.update(np.ascontiguousarray(r).tobytes())
    return dict(model_order=MODEL_ORDER, dimensions=d, common_dimension=common,
                map_checksum=digest.hexdigest(), row_isometry_error=errors,
                moment_stage='raw_linear_fusion',dtype=dtype.name)

def _ids(value, n, name):
    a = np.asarray(value)
    if a.dtype.kind != 'U' or a.shape != (n,) or np.any(a == ''):
        raise ValueError(name+': Unicode row IDs required')
    return a

def validate_inputs(native_means, native_vars, maps, alpha, mean_ids_by_model,
                    variance_ids_by_model, alpha_ids, role, model_order=MODEL_ORDER):
    meta = map_metadata(maps, model_order)
    if role not in ('query', 'document'): raise ValueError('query/document computation role required')
    for values in (native_means, native_vars, mean_ids_by_model, variance_ids_by_model):
        if set(values) != set(MODEL_ORDER): raise ValueError('model keys differ')
    alpha = _float_array(alpha, 'alpha')
    if alpha.dtype.name != meta['dtype']: raise ValueError('alpha/map dtype mismatch; explicitly cast all inputs together')
    if alpha.ndim != 2 or alpha.shape[1] != 3 or not alpha.shape[0]: raise ValueError('alpha shape')
    n = alpha.shape[0]; ids = _ids(alpha_ids, n, 'alpha_ids')
    tol = 1e-6 if alpha.dtype == np.float32 else 1e-10
    if np.any(alpha < 0) or np.any(np.abs(alpha.sum(axis=1)-1) > tol):
        raise ValueError('alpha must be nonnegative with row sum 1')
    for m in MODEL_ORDER:
        mu = _float_array(native_means[m], 'mean '+m)
        var = _float_array(native_vars[m], 'variance '+m)
        if mu.dtype.name != meta['dtype'] or var.dtype.name != meta['dtype']:
            raise ValueError('mean/variance/map dtype mismatch; explicitly cast all inputs together')
        shape = (n, meta['dimensions'][m])
        if mu.shape != shape or var.shape != shape or np.any(var < 0):
            raise ValueError('native shape or negative variance: '+m)
        for values, label in ((mean_ids_by_model, 'mean'), (variance_ids_by_model, 'variance')):
            if not np.array_equal(_ids(values[m], n, label+' IDs '+m), ids):
                raise ValueError('row ID/order mismatch: '+label+' '+m)
    return meta

def prepare_raw_bundle(native_means, native_vars, maps, alpha, mean_ids_by_model,
                       variance_ids_by_model, alpha_ids, role, model_order=MODEL_ORDER):
    meta = validate_inputs(native_means, native_vars, maps, alpha, mean_ids_by_model,
                           variance_ids_by_model, alpha_ids, role, model_order)
    dtype = np.result_type(alpha, *native_means.values(), *native_vars.values(), *maps.values())
    f = np.zeros((len(alpha), meta['common_dimension']), dtype=dtype)
    weights = []
    for i, m in enumerate(MODEL_ORDER):
        f += alpha[:, i:i+1] * (native_means[m] @ maps[m])
        weights.append(alpha[:, i:i+1]**2 * native_vars[m])
    return dict(meta, fused_mean=f, weighted_native_variance=np.concatenate(weights, axis=1),
                projected_mean_squared=np.concatenate([(f @ maps[m].T)**2 for m in MODEL_ORDER], axis=1),
                row_ids=np.asarray(alpha_ids).copy(), role=role)

def _check_output_budget(nq, nd, dtype, budget):
    if nq * nd * np.dtype(dtype).itemsize * 5 > budget:
        raise MemoryError('five pair outputs exceed budget; use iter_pair_moments')

def dense_memory_estimate(nq, nd, common_dimension, dtype):
    return ((nq+nd+4)*common_dimension**2 + 5*nq*nd)*np.dtype(dtype).itemsize

def prepare_dense_reference(inputs, maps, memory_budget_bytes=DEFAULT_BUDGET):
    meta = validate_inputs(maps=maps, **inputs)
    means, variances, alpha = inputs['native_means'], inputs['native_vars'], inputs['alpha']
    dtype = np.result_type(alpha, *means.values(), *variances.values(), *maps.values())
    n, dim = len(alpha), meta['common_dimension']
    if dense_memory_estimate(n, 0, dim, dtype) > memory_budget_bytes:
        raise MemoryError('dense oracle exceeds memory budget')
    # Independent oracle: literal per-row covariance transport, no B/a/w helpers.
    f = np.zeros((n,dim), dtype=dtype); cov = np.zeros((n,dim,dim), dtype=dtype)
    for row in range(n):
        for j,m in enumerate(MODEL_ORDER):
            r = maps[m]
            f[row] += alpha[row,j] * (means[m][row] @ r)
            cov[row] += alpha[row,j]**2 * (r.T @ (variances[m][row,:,None] * r))
    return dict(meta, fused_mean=f, covariance=cov, row_ids=inputs['alpha_ids'].copy(), role=inputs['role'])

def _compatible(q, d, extra=None):
    if q['role'] != 'query' or d['role'] != 'document': raise ValueError('wrong query/document roles')
    for other in (d, extra) if extra is not None else (d,):
        for key in ('model_order','dimensions','common_dimension','map_checksum','moment_stage','dtype'):
            if q[key] != other[key]: raise ValueError('incompatible '+key)

def _moments(mean, td, tq, cross):
    result = dict(zip(TERMS, (mean, td, tq, cross, td+tq+cross)))
    for k, a in result.items():
        if not np.isfinite(a).all(): raise ValueError('nonfinite '+k)
        tol = 1e-6 if a.dtype == np.float32 else 1e-10
        if k != 'dot_mean' and np.any(a < -tol): raise ValueError('negative variance term '+k)
    return result

def pair_moments_prepared_dense(q, d, memory_budget_bytes=DEFAULT_BUDGET):
    _compatible(q,d)
    fq, fd, cq, cd = q['fused_mean'], d['fused_mean'], q['covariance'], d['covariance']
    _check_output_budget(len(fq),len(fd),np.result_type(fq,fd), memory_budget_bytes)
    return _moments(fq @ fd.T, np.einsum('qi,dij,qj->qd',fq,cd,fq),
                    np.einsum('di,qij,dj->qd',fd,cq,fd),
                    cq.reshape(len(cq),-1) @ cd.reshape(len(cd),-1).T)

def pair_moments_dense_reference(query_inputs, document_inputs, maps, memory_budget_bytes=DEFAULT_BUDGET):
    dim = maps['qwen'].shape[1]
    dtype = np.result_type(*maps.values(), *query_inputs['native_means'].values(),
                           *document_inputs['native_means'].values(),
                           *query_inputs['native_vars'].values(), *document_inputs['native_vars'].values(),
                           query_inputs['alpha'], document_inputs['alpha'])
    if dense_memory_estimate(len(query_inputs['alpha']),len(document_inputs['alpha']),dim,dtype) > memory_budget_bytes:
        raise MemoryError('dense oracle exceeds memory budget')
    q = prepare_dense_reference(query_inputs,maps,memory_budget_bytes)
    d = prepare_dense_reference(document_inputs,maps,memory_budget_bytes)
    return pair_moments_prepared_dense(q,d,memory_budget_bytes)

def build_overlap_blocks(maps, model_order=MODEL_ORDER):
    meta = map_metadata(maps,model_order)
    blocks = {}
    for i,m in enumerate(MODEL_ORDER):
        for n in MODEL_ORDER[i:]:
            blocks[m,n] = (maps[m] @ maps[n].T)**2
            blocks[n,m] = blocks[m,n].T
    return dict(meta, blocks=blocks)

def pair_moments_implicit(query_bundle, document_bundle, overlap_blocks,
                          memory_budget_bytes=DEFAULT_BUDGET):
    q,d,b = query_bundle,document_bundle,overlap_blocks
    _compatible(q,d,b)
    fq,fd=q['fused_mean'],d['fused_mean']
    wq,wd=q['weighted_native_variance'],d['weighted_native_variance']
    aq,ad=q['projected_mean_squared'],d['projected_mean_squared']
    dtype=np.result_type(fq,fd,wq,wd)
    _check_output_budget(len(fq),len(fd),dtype,memory_budget_bytes)
    # B stays in shared blocks; WqB is query-sized, never per-document D x D.
    spans={}; start=0
    for m in MODEL_ORDER:
        spans[m]=slice(start,start+b['dimensions'][m]);start+=b['dimensions'][m]
    wb=np.zeros((len(fq),start),dtype=dtype)
    for m in MODEL_ORDER:
        for n in MODEL_ORDER:
            wb[:,spans[n]] += wq[:,spans[m]] @ b['blocks'][m,n]
    return _moments(fq@fd.T,aq@wd.T,wq@ad.T,wb@wd.T)

def slice_bundle(bundle, rows):
    result=dict(bundle)
    for k in ('fused_mean','weighted_native_variance','projected_mean_squared','row_ids'):
        result[k]=bundle[k][rows]
    return result

def iter_pair_moments(query_bundle, document_bundle, overlap_blocks,
                      query_block=8, document_block=128, memory_budget_bytes=DEFAULT_BUDGET):
    if query_block <= 0 or document_block <= 0: raise ValueError('positive block sizes required')
    _compatible(query_bundle,document_bundle,overlap_blocks)
    for qi in range(0,len(query_bundle['row_ids']),query_block):
        qs=slice(qi,qi+query_block)
        for di in range(0,len(document_bundle['row_ids']),document_block):
            ds=slice(di,di+document_block)
            yield qs,ds,pair_moments_implicit(slice_bundle(query_bundle,qs),
                slice_bundle(document_bundle,ds),overlap_blocks,memory_budget_bytes)

def pair_moments_diagonal_ablation(query_bundle, document_bundle, maps,
                                  memory_budget_bytes=DEFAULT_BUDGET):
    q,d=query_bundle,document_bundle
    _compatible(q,d,map_metadata(maps))
    fq,fd=q['fused_mean'],d['fused_mean']
    _check_output_budget(len(fq),len(fd),np.result_type(fq,fd),memory_budget_bytes)
    def diagonal(bundle):
        delta=np.zeros_like(bundle['fused_mean']);start=0
        for m in MODEL_ORDER:
            stop=start+maps[m].shape[0]
            delta += bundle['weighted_native_variance'][:,start:stop] @ (maps[m]**2)
            start=stop
        return delta
    dq,dd=diagonal(q),diagonal(d)
    return _moments(fq@fd.T,(fq**2)@dd.T,dq@(fd**2).T,dq@dd.T)
