"""Reproducible E0--E3 experiment. Never evaluates a final UEC score."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
import covariance_ops as ops

ORDER=ops.MODEL_ORDER

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write_csv(path,rows):
    if not rows:Path(path).write_text('');return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def load_role_npz(path,rows=slice(None)):
    with np.load(path,allow_pickle=False) as data:
        # Release each full NPZ member immediately after selecting trial rows.
        # Deserialization has a one-member peak, never a full retained corpus.
        def take(key):return data[key][rows].copy()
        return dict(native_means={m:take('mean__'+m) for m in ORDER},
            native_vars={m:take('variance__'+m) for m in ORDER},
            mean_ids_by_model={m:take('mean_ids__'+m) for m in ORDER},
            variance_ids_by_model={m:take('variance_ids__'+m) for m in ORDER},
            alpha=take('alpha'),alpha_ids=take('alpha_ids'),role=str(data['role'].item()))

def save_role_npz(path,side):
    data={'alpha':side['alpha'],'alpha_ids':side['alpha_ids'],'role':np.array(side['role'])}
    for m in ORDER:
        for key,source in [('mean','native_means'),('variance','native_vars'),('mean_ids','mean_ids_by_model'),('variance_ids','variance_ids_by_model')]:
            data[key+'__'+m]=side[source][m]
    np.savez(path,**data)

def side_subset(side,indices,dtype=None):
    def take(a):
        x=a[indices]
        return x.astype(dtype) if dtype is not None and x.dtype.kind=='f' else x.copy()
    return {k:({m:take(a) for m,a in v.items()} if isinstance(v,dict) else v if k=='role' else take(v)) for k,v in side.items()}

def make_side(means,variances,ids,role,dtype=np.float64):
    return dict(native_means={m:np.asarray(means[m],dtype=dtype) for m in ORDER},
        native_vars={m:np.asarray(variances[m],dtype=dtype) for m in ORDER},
        mean_ids_by_model={m:np.asarray(ids,dtype=str).copy() for m in ORDER},
        variance_ids_by_model={m:np.asarray(ids,dtype=str).copy() for m in ORDER},
        alpha=np.full((len(ids),3),1/3,dtype=dtype),alpha_ids=np.asarray(ids,dtype=str),role=role)

def synthetic(seed,dtype):
    rng=np.random.default_rng(seed);maps={'qwen':np.eye(8,dtype=dtype)}
    for m,dim in zip(ORDER[1:],[5,6]):maps[m]=np.linalg.qr(rng.normal(size=(8,dim)))[0].T.astype(dtype)
    def side(n,role):
        means={};variances={}
        for m in ORDER:
            x=rng.normal(size=(n,maps[m].shape[0]));means[m]=x/np.linalg.norm(x,axis=1,keepdims=True)
            variances[m]=rng.uniform(.001,.2,x.shape)
        return make_side(means,variances,[f'toy:{role}:{i}' for i in range(n)],role,dtype)
    return maps,side(4,'query'),side(8,'document')

def weight_conditions(nq,nd,seed,all_onehots=True):
    rng=np.random.default_rng(seed)
    result=[('uniform',np.full((nq,3),1/3),np.full((nd,3),1/3)),
        ('fixed_0.6_0.2_0.2',np.tile([.6,.2,.2],(nq,1)),np.tile([.6,.2,.2],(nd,1))),
        ('input_dirichlet',rng.dirichlet([1,1,1],nq),rng.dirichlet([1,1,1],nd))]
    for i in range(3 if all_onehots else 1):
        for j in range(3 if all_onehots else 1):
            result.append((f'onehot_{ORDER[i]}_{ORDER[j]}',np.tile(np.eye(3)[i],(nq,1)),np.tile(np.eye(3)[j],(nd,1))))
    return result

def compare_terms(fast,reference,arm,dtype,case):
    rows=[];atol,rtol=(1e-6,1e-4) if np.dtype(dtype)==np.float32 else (1e-10,1e-8)
    for term in reference:
        ref,got=reference[term],fast[term];delta=np.abs(got-ref);valid=np.abs(ref)>1e-12
        passed=bool(np.isfinite(got).all() and np.isfinite(ref).all() and np.all(delta<=atol+rtol*np.abs(ref)))
        if term.startswith('variance') or term=='dot_variance':passed &= bool(np.all(got>=-atol) and np.all(ref>=-atol))
        rows.append(dict(arm=arm,dtype=np.dtype(dtype).name,case=case,term=term,max_abs_error=float(delta.max()),
            max_relative_error=float((delta[valid]/np.abs(ref[valid])).max()) if valid.any() else None,
            near_zero_count=int((~valid).sum()),pass_equivalence=passed))
    return rows

def numeric_cases(maps,q,d,arm,dtype,seed,zeros=True):
    maps={m:r.astype(dtype) for m,r in maps.items()}
    q=side_subset(q,slice(None),dtype);d=side_subset(d,slice(None),dtype)
    overlap=ops.build_overlap_blocks(maps,ORDER);rows=[]
    for label,qa,da in weight_conditions(len(q['alpha']),len(d['alpha']),seed):
        q['alpha']=qa.astype(dtype);d['alpha']=da.astype(dtype)
        qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
        fast=ops.pair_moments_implicit(qb,db,overlap)
        refq=ops.prepare_dense_reference(q,maps);refd=ops.prepare_dense_reference(d,maps)
        ref=ops.pair_moments_prepared_dense(refq,refd)
        fast['fused_mean']=qb['fused_mean'];ref['fused_mean']=refq['fused_mean']
        rows.extend(compare_terms(fast,ref,arm,dtype,f'seed{seed}:{label}'))
    if zeros:
        for side in (q,d):
            for v in side['native_vars'].values():v.fill(0)
        fast=ops.pair_moments_implicit(ops.prepare_raw_bundle(maps=maps,**q),ops.prepare_raw_bundle(maps=maps,**d),overlap)
        rows.extend(compare_terms(fast,ops.pair_moments_dense_reference(q,d,maps),arm,dtype,f'seed{seed}:zero_variance'))
        if any(np.any(fast[k]!=0) for k in ops.TERMS[1:]):raise AssertionError('zero variance is not zero')
    return rows

def load_real_manifest(path):
    p=Path(path);manifest=json.loads(p.read_text())
    required=('query_npz','document_npz','maps','models','source_run','output_stage','covariance_representation','uncertainty_source','original_encoding_role','consumed_files')
    if any(k not in manifest for k in required):raise ValueError('incomplete real manifest')
    if manifest['output_stage']!='l2_normalized' or manifest['covariance_representation']!='native_output_diagonal_surrogate':
        raise ValueError('unsupported output stage or covariance representation')
    if tuple(manifest['models'])!=ORDER:raise ValueError('model order mismatch')
    expected_dims=dict(zip(ORDER,[1024,640,768]))
    for m in ORDER:
        if manifest['models'][m]['native_dim']!=expected_dims[m]:raise ValueError('wrong model dimension')
        if not manifest['models'][m].get('revision'):raise ValueError('missing model revision')
    # All external payloads are pinned, not accepted on shape alone.
    for filename,expected in manifest['consumed_files'].items():
        if sha(filename)!=expected:raise ValueError('checksum mismatch '+filename)
    source=Path(manifest['source_run']).resolve();source_manifest=source/'manifest.json';config_path=source/'input_hashes.json'
    if any(str(f) not in manifest['consumed_files'] for f in (source_manifest,config_path)):
        raise ValueError('missing bound source/model configuration evidence')
    source_seal=json.loads(source_manifest.read_text())
    if source_seal['files']['input_hashes.json']!=sha(config_path):raise ValueError('source model config differs from seal')
    configs=json.loads(config_path.read_text())['anchor_refs']
    expected_repos=dict(zip(ORDER,['Qwen/Qwen3-Embedding-0.6B','microsoft/harrier-oss-v1-270m','thenlper/gte-base']))
    for m in ORDER:
        if manifest['models'][m]!=configs[m]['encoder_config'] or manifest['models'][m].get('model_id')!=expected_repos[m]:
            raise ValueError('model policy/revision differs from bound producer configuration')
    if set(manifest['maps'])!={'GPA','Teacher_CAIA'}:raise ValueError('exact GPA/Teacher_CAIA arm set required')
    for arm,relative in [('GPA','maps/A.npz'),('Teacher_CAIA','maps/C.npz')]:
        if Path(manifest['maps'][arm]).resolve()!=source/relative or sha(source/relative)!=source_seal['files'][relative]:
            raise ValueError('arm identity differs from sealed source map')
    files=[manifest['query_npz'],manifest['document_npz'],*manifest['maps'].values()]
    if any(f not in manifest['consumed_files'] for f in files):raise ValueError('unbound input file')
    q=load_role_npz(manifest['query_npz']);d=load_role_npz(manifest['document_npz']);maps={}
    if q['role']!='query' or d['role']!='document':raise ValueError('reversed computation slots')
    for arm,filename in manifest['maps'].items():
        with np.load(filename,allow_pickle=False) as data:maps[arm]={m:data[m] for m in ORDER}
        if any(r.dtype!=np.float64 for r in maps[arm].values()):raise ValueError('base maps must be stored as float64')
        meta=ops.map_metadata(maps[arm])
        if meta['dimensions']!=expected_dims or meta['common_dimension']!=1024:raise ValueError('real map dimension')
        for side in (q,d):ops.validate_inputs(maps=maps[arm],**side)
    for side in (q,d):
        for m in ORDER:
            err=float(np.max(np.abs(np.linalg.norm(side['native_means'][m],axis=1)-1)))
            if err>(2e-3 if m=='qwen' else 1e-5):raise ValueError('declared unit mean norm failed')
    return manifest,q,d,maps

def select_diagnostic_pairs(query_ids,document_ids,pair_count,seed):
    rng=np.random.default_rng(seed);seen=set();pairs=[]
    nq,nd=len(query_ids),len(document_ids)
    counts_q,counts_d=Counter(query_ids),Counter(document_ids)
    possible=nq*nd-sum(n*counts_d.get(k,0) for k,n in counts_q.items())
    target=min(pair_count,possible)
    while len(pairs)<target:
        qi,di=divmod(int(rng.integers(nq*nd)),nd)
        if query_ids[qi]==document_ids[di] or (qi,di) in seen:continue
        seen.add((qi,di));pairs.append((qi,di))
    return pairs

def diagonal_diagnostic(out,q,d,maps,pair_count=1000,seed=42):
    if 'diagnostic_text_hashes_q' in maps:raise ValueError('maps and metadata must be separate')
    pairs=select_diagnostic_pairs(q['alpha_ids'],d['alpha_ids'],pair_count,seed)
    if not pairs:raise ValueError('no distinct diagnostic pairs available')
    write_json(out/'selected_diagnostic_pairs.json',dict(seed=seed,count=len(pairs),pairs=[
        dict(query_id=str(q['alpha_ids'][i]),document_id=str(d['alpha_ids'][j]),query_index=i,document_index=j) for i,j in pairs]))
    records=[];summaries=[]
    for arm,arm_maps in maps.items():
        overlap=ops.build_overlap_blocks(arm_maps,ORDER)
        for condition,qa,da in weight_conditions(len(q['alpha']),len(d['alpha']),seed,False):
            qs=side_subset(q,slice(None));ds=side_subset(d,slice(None));qs['alpha']=qa;ds['alpha']=da
            qb=ops.prepare_raw_bundle(maps=arm_maps,**qs);db=ops.prepare_raw_bundle(maps=arm_maps,**ds)
            fulls={k:[] for k in ops.TERMS};diags={k:[] for k in ops.TERMS}
            # Paired blocks collect just requested entries, no 1000 x 1000 output.
            for start in range(0,len(pairs),16):
                group=pairs[start:start+16];qi=np.array([i for i,j in group]);di=np.array([j for i,j in group])
                a=ops.slice_bundle(qb,qi);b=ops.slice_bundle(db,di)
                full=ops.pair_moments_implicit(a,b,overlap);diag=ops.pair_moments_diagonal_ablation(a,b,arm_maps)
                for k in ops.TERMS:fulls[k].extend(np.diag(full[k]));diags[k].extend(np.diag(diag[k]))
            for k in ops.TERMS:
                full=np.asarray(fulls[k]);diag=np.asarray(diags[k]);diff=diag-full;absolute=np.abs(diff);valid=full>1e-12
                if k=='dot_mean' and np.any(diff!=0):raise AssertionError('ablation changed mean')
                if condition=='onehot_qwen_qwen' and np.max(absolute)>1e-10:raise AssertionError('identity control failed')
                rel=absolute[valid]/np.abs(full[valid])
                summaries.append(dict(arm=arm,condition=condition,term=k,pairs=len(pairs),
                    signed_mean=float(diff.mean()),signed_min=float(diff.min()),signed_max=float(diff.max()),
                    absolute_median=float(np.median(absolute)),absolute_p95=float(np.quantile(absolute,.95)),absolute_max=float(absolute.max()),
                    relative_median=float(np.median(rel)) if len(rel) else None,
                    relative_p95=float(np.quantile(rel,.95)) if len(rel) else None,
                    relative_max=float(rel.max()) if len(rel) else None,near_zero_count=int((~valid).sum())))
                for t,((qi,di),fv,dv) in enumerate(zip(pairs,full,diag)):
                    records.append(dict(arm=arm,condition=condition,pair=t,query_id=str(q['alpha_ids'][qi]),document_id=str(d['alpha_ids'][di]),
                        term=k,full=float(fv),diagonal_ablation=float(dv),signed_difference=float(dv-fv),absolute_difference=float(abs(dv-fv))))
    write_csv(out/'diagonal_ablation.csv',records);write_csv(out/'diagonal_summary.csv',summaries)
    write_json(out/'diagonal_top_examples.json',sorted([r for r in records if r['term']=='dot_variance'],key=lambda x:x['absolute_difference'],reverse=True)[:10])
    return summaries

def measured(fn,warmups=2,repeats=5):
    for _ in range(warmups):
        value=fn();del value
    times=[]
    for index in range(repeats):
        start=time.perf_counter();value=fn();times.append(time.perf_counter()-start)
        if index<repeats-1:del value
    return value,float(np.median(times)),float(max(times))

def benchmark_worker(folder,method,nq,nd,dtype,arm='maps'):
    folder=Path(folder);dtype=np.dtype(dtype)
    q=side_subset(load_role_npz(folder/'q.npz',slice(0,nq)),slice(None),dtype)
    d=side_subset(load_role_npz(folder/'d.npz',slice(0,nd)),slice(None),dtype)
    if nq<=0 or nd<=0 or len(q['alpha'])!=nq or len(d['alpha'])!=nd:raise ValueError('requested trial exceeds available rows')
    with np.load(folder/(arm+'.npz'),allow_pickle=False) as data:maps={m:data[m].astype(dtype) for m in ORDER}
    row=dict(method=method,arm=arm,device='cpu',dtype=dtype.name,nq=nq,nd=nd,warmups=2,repeats=5,pid=os.getpid())
    def phase(name,fn):
        result,median,maximum=measured(fn);row[name+'_median_seconds']=median;row[name+'_max_seconds']=maximum;return result
    if method=='implicit':
        overlap=phase('overlap',lambda:ops.build_overlap_blocks(maps,ORDER))
        db=phase('document_prepare',lambda:ops.prepare_raw_bundle(maps=maps,**d))
        qb=phase('query_prepare',lambda:ops.prepare_raw_bundle(maps=maps,**q))
        value=phase('score',lambda:ops.pair_moments_implicit(qb,db,overlap))
        row['per_document_cache_bytes']=sum(db[k].nbytes for k in ('fused_mean','weighted_native_variance','projected_mean_squared'))//nd
        # Store six unique symmetric blocks; transposed views allocate no data.
        row['overlap_bytes']=sum(overlap['blocks'][m,n].nbytes for i,m in enumerate(ORDER) for n in ORDER[i:])
    elif method=='dense_reference':
        estimate=ops.dense_memory_estimate(nq,nd,1024,dtype)
        if estimate>ops.DEFAULT_BUDGET:raise MemoryError('dense benchmark exceeds 512 MiB')
        row['overlap_median_seconds']=0.;row['overlap_max_seconds']=0.;row['overlap_bytes']=0
        db=phase('document_prepare',lambda:ops.prepare_dense_reference(d,maps))
        qb=phase('query_prepare',lambda:ops.prepare_dense_reference(q,maps))
        value=phase('score',lambda:ops.pair_moments_prepared_dense(qb,db))
        row['per_document_cache_bytes']=(db['fused_mean'].nbytes+db['covariance'].nbytes)//nd
    else:raise ValueError('unknown method')
    row['map_bytes']=sum(r.nbytes for r in maps.values());row['pair_output_bytes']=sum(v.nbytes for v in value.values())
    status=Path('/proc/self/status').read_text().splitlines()
    row['peak_rss_bytes']=int(next(s.split()[1] for s in status if s.startswith('VmHWM:')))*1024
    row['peak_measurement']='Linux /proc/self/status VmHWM, address-space peak reset on exec'
    row['score_checksum']=hashlib.sha256(value['dot_variance'].tobytes()).hexdigest()
    return row

def benchmarks(out,q,d,maps,config):
    folder=out/'benchmark_inputs';folder.mkdir();save_role_npz(folder/'q.npz',side_subset(q,slice(0,8)))
    save_role_npz(folder/'d.npz',side_subset(d,slice(0,1024)))
    for arm,values in maps.items():np.savez(folder/(arm+'.npz'),**values)
    rows=[]
    if len(q['alpha'])<8:raise ValueError('benchmark scale requires 8 available queries')
    for arm in maps:
        for dtype in config.get('dtypes',['float64','float32']):
            for n in config.get('dense_batches',[1,4,8]):
                for method in ['dense_reference','implicit']:
                    if n>len(q['alpha']) or n>len(d['alpha']):raise ValueError('dense batch exceeds available cache')
                    proc=subprocess.run([sys.executable,__file__,'--worker',str(folder),'--method',method,
                        '--nq',str(n),'--nd',str(n),'--dtype',dtype,'--arm',arm],check=True,capture_output=True,text=True)
                    rows.append(json.loads(proc.stdout));write_csv(out/'runtime_memory.csv',rows)
            for n in config.get('document_sizes',[128,512,1024]):
                if n>len(d['alpha']):continue
                proc=subprocess.run([sys.executable,__file__,'--worker',str(folder),'--method','implicit',
                    '--nq','8','--nd',str(n),'--dtype',dtype,'--arm',arm],check=True,capture_output=True,text=True)
                rows.append(json.loads(proc.stdout));write_csv(out/'runtime_memory.csv',rows)
    return rows

def validate_config(config):
    locked=dict(synthetic_common_dimension=8,synthetic_dimensions=[8,5,6],synthetic_variance_range=[.001,.2],
                query_scale_count=8,warmups=2,repeats=5,dense_memory_budget_mib=512)
    for key,value in locked.items():
        if key in config and config[key]!=value:raise ValueError('unsupported fixed protocol override: '+key)
    if not config.get('seeds',[42,43,44]) or not config.get('dtypes',['float64','float32']):raise ValueError('empty experiment grid')
    if any(dtype not in ('float64','float32') for dtype in config.get('dtypes',[])):raise ValueError('unsupported dtype')
    for key in ['dense_batches','document_sizes']:
        if key in config and (not config[key] or any(not isinstance(n,int) or n<=0 for n in config[key])):raise ValueError('invalid sizes')
    if config.get('diagnostic_pairs',1000)<=0:raise ValueError('positive pair count required')

def run_validation(output,config,real_manifest=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',E0='pending',E1_synthetic='pending',E1_real='blocked_missing_cache',E2='pending',E3='pending',
        E4='blocked_missing_uec_fitting_outputs',E5='blocked_missing_uec_score_contract',encoder_weights_updated=False,
        alignment_refit_performed=False,uec_fusion_executed=False,downstream_evaluated=False,theorem_extension_proved=False)
    write_json(out/'status.json',status);write_json(out/'config.json',config)
    integration=dict(uec_integration_ready=False,executed=False,reason='actual three-model fitting output caches and fixed score/weight/normalization contract not supplied')
    write_json(out/'uec_integration.json',integration)
    write_json(out/'alignment_refit.json',dict(performed=False,status=status['E4'],reason='available outputs are AG News teacher-surrogate anchors, not actual role-aware UEC fitting observations'))
    try:
        validate_config(config)
        source_files={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py')}
        manifest=dict(code_files=source_files,model_order=list(ORDER),covariance_representation='native_output_diagonal_surrogate',
            moment_stage='raw_linear_fusion',uec_integration_ready=False,independence_assumptions=['models independent','query and document random vectors independent'],
            coefficient_function='external deterministic alpha; synthetic diagnostic weights only',score_function=None,
            numpy=np.__version__,python=platform.python_version(),device='cpu',threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
        real=None
        if real_manifest:
            real=load_real_manifest(real_manifest);manifest['real_inputs']=real[0];manifest['real_manifest_sha256']=sha(real_manifest)
        write_json(out/'manifest.json',manifest);status['E0']='passed'
        rows=[]
        for seed in config.get('seeds',[42,43,44]):
            for dtype in config.get('dtypes',['float64','float32']):
                maps,q,d=synthetic(seed,np.dtype(dtype));rows.extend(numeric_cases(maps,q,d,'synthetic',dtype,seed))
        write_csv(out/'numeric_equivalence.csv',rows)
        if not all(r['pass_equivalence'] for r in rows):raise AssertionError('E1 synthetic failed; E2/E5 stopped')
        status['E1_synthetic']='passed'
        fixture_dir=out/'small_fixture';fixture_dir.mkdir()
        fm,fq,fd=synthetic(42,np.float64);save_role_npz(fixture_dir/'q.npz',fq);save_role_npz(fixture_dir/'d.npz',fd);np.savez(fixture_dir/'maps.npz',**fm)
        if real:
            metadata,q,d,maps=real
            for arm,arm_maps in maps.items():
                for dtype in config.get('dtypes',['float64','float32']):
                    rows.extend(numeric_cases(arm_maps,side_subset(q,slice(0,2)),side_subset(d,slice(0,4)),arm,dtype,42))
            write_csv(out/'numeric_equivalence.csv',rows)
            if not all(r['pass_equivalence'] for r in rows):raise AssertionError('E1 real failed; E2/E5 stopped')
            status['E1_real']='passed';write_json(out/'status.json',status)
            diagonal_diagnostic(out,q,d,maps,config.get('diagnostic_pairs',1000),42);status['E2']='completed'
            write_json(out/'status.json',status)
            if config.get('benchmark',True):
                benchmarks(out,q,d,maps,config);status['E3']='completed'
            else:status['E3']='not_requested'
        else:
            status['E2']='blocked_missing_cache';status['E3']='blocked_missing_cache'
        status['status']='completed_available_stages'
        write_json(out/'status.json',status)
        write_report(out,status,rows)
        return status
    except Exception as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        if status['E0']=='pending':status['E0']='failed'
        write_json(out/'status.json',status)
        (out/'report.md').write_text('Experiment failed; no later-stage success is claimed.\n'+json.dumps(status,indent=2)+'\n')
        raise

def write_report(out,status,rows):
    lines=['# CAIA UEC covariance validation','',
        'Native-diagonal output surrogate의 raw 선형 결합 내적 모멘트 검증. 최종 UEC 점수 또는 과제 성능 실험이 아니다.','',
        f"E1: {len(rows)} term comparisons; max absolute error {max(r['max_abs_error'] for r in rows):.6g}.",'',
        '| Stage | Status |','|---|---|']
    for k in ['E0','E1_synthetic','E1_real','E2','E3','E4','E5']:lines.append(f'| {k} | {status[k]} |')
    lines.extend(['','AG News query-style 인코딩을 query/document 계산 슬롯에 배치했다. 인코딩 역할은 바꾸지 않았다.',
        '진단은 기존 alignment diagnostic 자료이며 posterior까지 완전히 미사용인 독립 평가로 부르지 않는다.',
        '원래 teacher surrogate의 품질 경고는 manifest에 보존했다. UEC fitting 관측의 3모델 출력 캐시와 실제 score 계약이 없으므로 E4/E5는 실행하지 않았다.','',
        '## Numeric comparisons','', '| Arm | dtype | max abs error |','|---|---|---|'])
    for arm,dtype in sorted(set((r['arm'],r['dtype']) for r in rows)):
        lines.append(f"| {arm} | {dtype} | {max(r['max_abs_error'] for r in rows if r['arm']==arm and r['dtype']==dtype):.6g} |")
    diag=out/'diagonal_summary.csv'
    if diag.exists():
        lines+=['','## Diagonal ablation','','Relative difference = abs(diag-full)/full; numeric variance effect only.','',
            '| Arm | weights | V median relative | V p95 relative | V max relative |','|---|---|---|---|---|']
        def relative_cell(value):return f'{float(value):.6%}' if value else 'N/A'
        for r in csv.DictReader(io.StringIO(diag.read_text())):
            if r['term']=='dot_variance':
                lines.append(f"| {r['arm']} | {r['condition']} | {relative_cell(r['relative_median'])} | {relative_cell(r['relative_p95'])} | {relative_cell(r['relative_max'])} |")
                if int(r['near_zero_count']):lines.append(f"\n{r['arm']} / {r['condition']}: {r['near_zero_count']} near-zero pairs excluded from relative statistics.\n")
    rt=out/'runtime_memory.csv'
    if rt.exists():
        lines+=['','## CPU costs','','Warm-up 2, repeats 5, fresh process per arm/dtype/method/size. Peak includes interpreter, inputs, maps and warm-up; phase times include validation.',
            'Map/B preparation is a shared one-time cost; document/query preparation and score are separate. Dense 512 MiB guard is checked before allocation.','',
            '| Arm | dtype | method | q×d | document prep ms | query prep ms | score ms | peak MiB | cache bytes/document |','|---|---|---|---|---|---|---|---|---|']
        for r in csv.DictReader(io.StringIO(rt.read_text())):
            lines.append(f"| {r['arm']} | {r['dtype']} | {r['method']} | {r['nq']}×{r['nd']} | {float(r['document_prepare_median_seconds'])*1000:.3f} | {float(r['query_prepare_median_seconds'])*1000:.3f} | {float(r['score_median_seconds'])*1000:.3f} | {int(r['peak_rss_bytes'])/1024**2:.1f} | {r['per_document_cache_bytes']} |")
        lines+=['','A speed ratio is only meaningful between equal arm/dtype/batch/device. This does not compare throughput with original diagonal UEC.']
    lines+=['','## Scope flags','',json.dumps({k:v for k,v in status.items() if k in ('encoder_weights_updated','alignment_refit_performed','uec_fusion_executed','downstream_evaluated','theorem_extension_proved')},indent=2),'']
    (out/'report.md').write_text('\n'.join(lines))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--config');parser.add_argument('--manifest')
    parser.add_argument('--worker');parser.add_argument('--method');parser.add_argument('--nq',type=int);parser.add_argument('--nd',type=int)
    parser.add_argument('--dtype',default='float64');parser.add_argument('--arm',default='maps');args=parser.parse_args()
    if args.worker:print(json.dumps(benchmark_worker(args.worker,args.method,args.nq,args.nd,args.dtype,args.arm)));return
    if not args.output:parser.error('--output required')
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    status=run_validation(args.output,config,args.manifest);print(json.dumps(status,indent=2))

if __name__=='__main__':main()
