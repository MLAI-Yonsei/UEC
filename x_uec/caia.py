"""Portable three-model CAIA-SVD core, extracted from the accepted v3 numerical implementation.

Only NumPy and the Python standard library are required. Original statistics and
solver functions below preserve their arithmetic; the repository-relative import
is removed. The public fit/transport adapters follow those functions.
"""
from dataclasses import dataclass
from itertools import combinations
import hashlib,json
import numpy as np

@dataclass
class MeanStats:
    S:dict
    fit_ids_hash:str

@dataclass
class CovStats:
    K:dict
    trace_mean:dict
    fit_ids_hash:str
    source:str

def ids_hash(ids):return hashlib.sha256(json.dumps(ids,ensure_ascii=False).encode()).hexdigest()
def pairs(keys):return list(combinations(keys,2))

def compute_mean_stats(means):
    x=means.x;n=len(means.ids)
    if not n or any(len(a)!=n or not np.isfinite(a).all() for a in x.values()):raise ValueError('Mean row shape/finite contract')
    return MeanStats({(a,b):x[a].T@x[b]/n for a in x for b in x},ids_hash(means.ids))

def compute_cov_stats(cov,permutations=None):
    n=len(cov.ids);root={};trace={}
    for name,v in cov.v.items():
        if len(v)!=n or not np.isfinite(v).all() or np.min(v)<0:raise ValueError('Invalid output variance')
        trace[name]=float(v.sum()/n)
        if permutations and name in permutations:
            p=np.asarray(permutations[name])
            if not np.array_equal(np.sort(p),np.arange(n)):raise ValueError('Invalid covariance permutation')
            v=v[p]
        root[name]=np.sqrt(v)
    k={}
    for a,b in pairs(root):k[a,b]=root[a].T@root[b]/n;k[b,a]=k[a,b].T
    return CovStats(k,trace,ids_hash(cov.ids),cov.source)

def nonnegative(raw,scale,kind):
    if not np.isfinite(raw):raise ValueError('Nonfinite '+kind)
    if raw < -1e-12*max(scale,1e-30):raise ValueError('Negative '+kind)
    return max(0.,float(raw))

def objective(maps,S,K=None,gamma=0.):
    if gamma<0 or not np.isfinite(gamma):raise ValueError('Invalid gamma')
    keys=list(maps);mu=0.;c_mu=0.;c_cov=0.;cov=0.;contributions={}
    if K is not None and K.fit_ids_hash!=S.fit_ids_hash:raise ValueError('Mean/covariance row correspondence mismatch')
    for a,b in pairs(keys):
        constant=float(np.trace(S.S[a,a])+np.trace(S.S[b,b]))/9
        mean=constant-2*float(np.sum(maps[a]*(S.S[a,b]@maps[b])))/9
        c_mu+=constant;mu+=mean
        cp=0.
        if K is not None:
            cc=(K.trace_mean[a]+K.trace_mean[b])/3;c_cov+=cc
            cp=cc-2*float(np.sum(K.K[a,b]*(maps[a]@maps[b].T)**2))/3
            cov+=cp
        contributions[a+'_'+b]=dict(mean=mean,cov=cp)
    raw_mu=mu;raw_cov=cov
    mu=nonnegative(mu,c_mu,'mean loss');cov=nonnegative(cov,c_cov,'covariance loss')
    return dict(mean=mu,cov=cov,total=mu+gamma*cov,raw_mean=raw_mu,raw_cov=raw_cov,
                c_mu=c_mu,c_cov=c_cov,pairs=contributions)

"""Sequential row-Stiefel SVD-MM over fixed sufficient statistics, no autograd."""
import time
import numpy as np

def polar_rows(matrix):
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite block matrix')
    u,s,vh=np.linalg.svd(matrix,full_matrices=False)
    return u@vh,s

def initialize(S):
    d=S.S['qwen','qwen'].shape[0];maps={'qwen':np.eye(d)}
    for name in ['harrier','gte']:
        m=S.S[name,'qwen'];maps[name]=polar_rows(m)[0] if np.any(m) else np.eye(len(m),d)
    return maps

def block_matrix(name,maps,S,K=None,gamma=0.):
    old=maps[name];matrix=np.zeros_like(old)
    for other,r in maps.items():
        if other==name:continue
        if other=='qwen' and np.array_equal(r,np.eye(len(r))):
            matrix+=S.S[name,other]
            if K is not None and gamma:matrix+=6*gamma*K.K[name,other]*old
            continue
        matrix+=S.S[name,other]@r
        if K is not None and gamma:
            matrix+=6*gamma*(K.K[name,other]*(old@r.T))@r
    return matrix

def stationarity(maps,S,K=None,gamma=0.,with_gap=False):
    residual=denominator=0.;matrices={}
    for name in ['harrier','gte']:
        m=block_matrix(name,maps,S,K,gamma);r=maps[name];mr=m@r.T
        p=m-(mr+mr.T)/2@r
        residual+=float(np.sum(p*p));denominator+=float(np.sum(m*m));matrices[name]=m
    result=dict(absolute_residual=np.sqrt(residual),relative_residual=np.sqrt(residual)/max(np.sqrt(denominator),1e-30),relative_gap=None,gap_svd_calls=0)
    if with_gap:
        nuclear=sum(float(np.linalg.svd(m,compute_uv=False).sum()) for m in matrices.values())
        raw=nuclear-sum(float(np.sum(maps[name]*m)) for name,m in matrices.items())
        if raw < -1e-12*max(1,nuclear):raise ValueError('Invalid negative polar gap')
        result.update(relative_gap=max(0.,raw)/max(nuclear,1e-30),raw_gap=raw,gap_rounding=raw<0,gap_svd_calls=2)
    return result

def isometry(maps):
    value=max(float(np.max(np.abs(r@r.T-np.eye(len(r))))) for r in maps.values())
    if value>1e-8:raise ValueError('Row isometry violated')
    return value

def fit_svd(initial,S,K,gamma,config,checkpoint=None):
    maps={k:v.copy() for k,v in initial.items()};flags=[];ks=K;gs=gamma
    if K is not None and not any(np.any(v) for v in K.K.values()):
        ks=None;gs=0.;flags.append('no_cov_guidance')
    maximum=config.get('max_sweeps',300);tol=config.get('stat_tol',1e-7);gap_tol=config.get('gap_tol',1e-10)
    stalled_tol=config.get('stalled_tol',1e-10);stalled_steps=config.get('stalled_sweeps',5)
    trace=[];stalled=0;svd_calls=0;blocks=0;start=time.perf_counter()
    def record(sweep,details):
        nonlocal svd_calls
        st=stationarity(maps,S,ks,gs,False)
        if st['relative_residual']<=tol:
            st=stationarity(maps,S,ks,gs,True);svd_calls+=st['gap_svd_calls']
        losses=objective(maps,S,ks,gs);reported=objective(maps,S,K,gamma)
        row=dict(sweep=sweep,loss=losses,reported_loss=reported,stationarity=st,
                 isometry_error=isometry(maps),svd_calls=svd_calls,block_count=blocks,blocks=details,seconds=time.perf_counter()-start)
        trace.append(row)
        if checkpoint:checkpoint(sweep,{k:v.copy() for k,v in maps.items()},row)
        return row
    row=record(0,[]);status=None
    if row['stationarity']['relative_residual']<=tol and row['stationarity']['relative_gap']<=gap_tol:status='stationary_tolerance'
    for sweep in range(1,maximum+1):
        if status:break
        before_sweep=row['loss']['total'];details=[]
        for name in ['harrier','gte']:
            old=maps[name];m=block_matrix(name,maps,S,ks,gs)
            if not np.any(m):
                details.append(dict(model=name,zero_M=True,rank=0));continue
            candidate,s=polar_rows(m);svd_calls+=1
            rank=int(np.sum(s>np.finfo(float).eps*max(m.shape)*s[0]))
            oldloss=objective(maps,S,ks,gs)['total'];trial={**maps,name:candidate}
            isometry(trial);newloss=objective(trial,S,ks,gs)['total']
            if newloss>oldloss+1e-11*max(1,abs(oldloss)):
                return dict(maps=maps,status='numerical_failure',sweeps=sweep-1,trace=trace,quality_flags=flags,failure=dict(model=name,old=oldloss,new=newloss))
            delta=candidate-old
            identity=2/9*(float(np.sum(m*delta))+3*gs*sum(float(np.sum(ks.K[name,n]*(delta@r.T)**2)) for n,r in maps.items() if n!=name)) if ks is not None else 2/9*float(np.sum(m*delta))
            details.append(dict(model=name,rank=rank,minimum_singular=float(s[-1]),maximum_singular=float(s[0]),
                                old_loss=oldloss,new_loss=newloss,decrease_identity=identity,zero_M=False))
            maps[name]=candidate;blocks+=1
        row=record(sweep,details);st=row['stationarity']
        relative=abs(row['loss']['total']-before_sweep)/max(abs(before_sweep),1e-12)
        stalled=stalled+1 if relative<=stalled_tol else 0
        row['relative_loss_change']=relative
        if st['relative_residual']<=tol and st['relative_gap']<=gap_tol:status='stationary_tolerance'
        elif stalled>=stalled_steps:status='stalled_not_stationary'
        elif sweep==maximum:status='budget_exhausted'
    if status is None:status='budget_exhausted'
    final=stationarity(maps,S,ks,gs,True);svd_calls+=2
    return dict(maps=maps,status=status,sweeps=trace[-1]['sweep'],trace=trace,quality_flags=flags,
                final_stationarity=final,svd_calls=svd_calls,block_count=blocks,seconds=time.perf_counter()-start)

def choose_gamma(means,maps,S,K):
    count=len(means.ids);direct=0.
    for begin in range(0,count,128):
        z=[means.x[k][begin:begin+128]@r for k,r in maps.items()];u=sum(z)/3
        direct+=sum(float(np.sum((a-u)**2)) for a in z)
    direct/=3*count;loss=objective(maps,S,K,0)
    if abs(direct-loss['mean'])>1e-10*max(loss['c_mu'],1e-30):raise ValueError('Direct/cached mean loss differ')
    degenerate=direct<=1e-12*max(loss['c_mu'],1e-30) or loss['cov']<=1e-12*max(loss['c_cov'],1e-30)
    gamma=0. if degenerate else .1*direct/loss['cov']
    return gamma,dict(gamma=gamma,direct_mean=direct,teacher_cov=loss['cov'],cached=loss,degenerate_gamma=degenerate)


# Portable adapter: the functions above retain the original numerical arithmetic.
NAMES = ('qwen', 'harrier', 'gte')


def check_anchor_inputs(means, variances, mean_ids, variance_ids):
    """Validate corresponding *same-input* native means and output diagonals.

    IDs must include the actual encoding role, e.g. 'pair:31:query'. Equal text
    with different prompts/roles is a different observation. No rows are reordered.
    Returns independent float64 working copies and the common ordered IDs.
    """
    for value in (means, variances, mean_ids, variance_ids):
        if set(value) != set(NAMES):
            raise ValueError('Exactly qwen, harrier, gte keys are required')
    x, v, common = {}, {}, None
    for name in NAMES:
        ids = np.asarray(mean_ids[name])
        vids = np.asarray(variance_ids[name])
        if ids.ndim != 1 or ids.dtype.kind not in 'US' or not len(ids):
            raise ValueError('IDs must be nonempty one-dimensional string arrays')
        if len(set(ids.tolist())) != len(ids) or any(not str(i) for i in ids):
            raise ValueError('IDs must be unique and nonempty')
        if not np.array_equal(ids, vids) or (common is not None and not np.array_equal(ids, common)):
            raise ValueError('Mean/variance/model ID or role order mismatch')
        common = ids.copy()
        a = np.array(means[name], dtype=np.float64, copy=True)
        b = np.array(variances[name], dtype=np.float64, copy=True)
        if a.ndim != 2 or a.shape[0] != len(ids) or a.shape[1] < 1 or b.shape != a.shape:
            raise ValueError('Means/output variance must both have native shape (N,d)')
        if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b < 0):
            raise ValueError('Nonfinite means or nonfinite/negative output variance')
        norm = np.linalg.norm(a, axis=1)
        tolerance = 2e-3 if name == 'qwen' else 1e-5
        if np.any(norm <= 0) or np.any(abs(norm - 1) > tolerance):
            raise ValueError('Native means must already be L2-normalized')
        x[name], v[name] = a / norm[:, None], b
    if any(a.shape[1] > x['qwen'].shape[1] for a in x.values()):
        raise ValueError('Qwen reference dimension must be at least each native dimension')
    return x, v, common.tolist()


def fit_caia(means, variances, mean_ids, variance_ids, *, gamma=None,
             include_shuffled=True, max_sweeps=300, r0_sweeps=100):
    """Fit saved-method GPA and CAIA controls from a common mean-only R0.

    Covariance diagonals must describe the SAME normalized output as the means,
    not parameter-space variance. Caller owns LA/Jacobian fitting and provenance.
    No model weights are touched. Only the three-model normalization is supported.
    """
    from types import SimpleNamespace
    if gamma is not None and (not np.isfinite(gamma) or gamma < 0):
        raise ValueError('gamma must be finite and nonnegative')
    for limit in (max_sweeps, r0_sweeps):
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
            raise ValueError('Sweep limits must be positive integers')
    x, v, ids = check_anchor_inputs(means, variances, mean_ids, variance_ids)
    mean_data = SimpleNamespace(x=x, ids=ids)
    cov_data = SimpleNamespace(v=v, ids=ids, source='caller_output_diagonal')
    s, k = compute_mean_stats(mean_data), compute_cov_stats(cov_data)
    r0 = fit_svd(initialize(s), s, None, 0., {'max_sweeps': r0_sweeps})
    if r0['status'] == 'numerical_failure':
        raise ValueError('R0 numerical_failure; no CAIA fit attempted')
    if gamma is None:
        gamma, evidence = choose_gamma(mean_data, r0['maps'], s, k)
        evidence['rule'] = 'fit-only 0.1 * Lmean(R0) / Lcov(R0)'
    else:
        gamma = float(gamma)
        evidence = dict(gamma=gamma, rule='caller_prespecified; not selected on downstream results')
    result = dict(r0=r0, gamma=gamma, gamma_evidence=evidence)
    result['gpa'] = fit_svd(r0['maps'], s, None, 0., {'max_sweeps': max_sweeps})
    result['caia'] = fit_svd(r0['maps'], s, k, gamma, {'max_sweeps': max_sweeps})
    if include_shuffled:
        perms = {n: np.random.default_rng(seed).permutation(len(ids))
                 for n, seed in [('harrier', 241304), ('gte', 241305)]}
        shuffled = compute_cov_stats(cov_data, perms)
        result['shuffled'] = fit_svd(r0['maps'], s, shuffled, gamma, {'max_sweeps': max_sweeps})
        result['shuffle_permutations'] = perms
    return result


def transform_gaussian(means, covariance, matrix):
    """Transport a small batch to the common space; preserve off-diagonals.

    means: (B,d), already normalized. covariance: (B,d) native diagonals or
    (B,d,d) full native output covariance. matrix: row-isometry (d,D).
    Returns (means @ matrix, full_covariance[B,D,D]). Stored-unit row norm error
    up to 2e-3 (the original Qwen cache tolerance) is accepted without rescaling
    the supplied mean or covariance. No L2 transform or jitter is added. Process
    small batches; rank<=d can make the returned covariance singular.
    """
    x = np.asarray(means, dtype=np.float64)
    c = np.asarray(covariance, dtype=np.float64)
    r = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or r.ndim != 2 or x.shape[1] != r.shape[0] or not len(x):
        raise ValueError('Invalid mean/map shape')
    if not all(np.isfinite(a).all() for a in (x, c, r)):
        raise ValueError('Nonfinite Gaussian input')
    if np.max(abs(r @ r.T - np.eye(len(r)))) > 1e-8:
        raise ValueError('Map must preserve native row geometry')
    if np.any(abs(np.linalg.norm(x, axis=1) - 1) > 2e-3):
        raise ValueError('Means must already represent normalized outputs')
    if c.shape == x.shape:
        if np.any(c < 0):
            raise ValueError('Negative native output variance')
        aligned = np.einsum('na,ad,ae->nde', c, r, r, optimize=True)
    elif c.shape == (len(x), x.shape[1], x.shape[1]):
        scale = np.maximum(np.max(abs(c), axis=(1, 2)), 1e-30)
        if np.any(np.max(abs(c-c.transpose(0, 2, 1)), axis=(1, 2)) > 1e-8*scale):
            raise ValueError('Native covariance must be symmetric')
        if np.any(np.linalg.eigvalsh(c).min(axis=1) < -1e-8*scale):
            raise ValueError('Native covariance must be PSD')
        aligned = np.einsum('ad,nab,be->nde', r, c, r, optimize=True)
    else:
        raise ValueError('Covariance must be output diagonal (B,d) or full (B,d,d)')
    return x @ r, aligned
