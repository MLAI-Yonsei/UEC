"""E4 smoke adapter executing the exact v3 delivery numerical files."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import time
import numpy as np
import caia
import covariance_ops as ops
from e4_data import DIMS,canonical_hash,validate_cache_arrays
from run_covariance_validation import sha,write_json,write_csv,compare_terms,make_side

@dataclass
class E4Inputs:
    means:dict
    variances:dict
    mean_ids:dict
    variance_ids:dict
    observations:list
    provenance:dict
    checksums:dict

@dataclass
class AlignmentSplit:
    fit_indices:np.ndarray
    diagnostic_indices:np.ndarray
    excluded_indices:np.ndarray
    audit:dict

def scope_flags():
    return dict(delivery_version='v3',mode='smoke',alignment_ready_for_handoff=False,
        encoder_weights_updated=False,inference_executed_in_this_run=False,posterior_refit_performed=False,
        uec_final_score_validated=False,uec_fusion_executed=False,downstream_evaluated=False,theorem_extension_proved=False)

def verify_backend_files(config_files,provenance_files):
    def normalize(values):
        result={}
        for key,value in values.items():
            name=Path(key).name
            if name in result and result[name]!=value:raise ValueError('ambiguous backend file '+name)
            result[name]=value
        return result
    config_files,provenance_files=normalize(config_files),normalize(provenance_files)
    for name in ['caia.py','covariance_ops.py','run_covariance_validation.py']:
        actual=sha(Path(__file__).parent/name)
        if config_files.get(name)!=actual or provenance_files.get(name)!=actual:raise ValueError('not pinned v3 backend '+name)
    if Path(caia.__file__).resolve()!=Path(__file__).with_name('caia.py').resolve() or Path(ops.__file__).resolve()!=Path(__file__).with_name('covariance_ops.py').resolve():raise ValueError('imported backend is outside actual delivery folder')

def write_json_atomic(path,value):
    path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def load_verified_inputs(config):
    provenance_path=Path(config['provenance_path']);p=json.loads(provenance_path.read_text())
    if config.get('mode')=='full':
        from full_audit import require_consumed_paths
        require_consumed_paths(p)
    expected_scope='alignment_full' if config.get('mode')=='full' else 'smoke_only'
    if p['delivery_version']!='v3' or p['cache_scope']!=expected_scope:raise ValueError('expected declared v3 '+expected_scope+' cache')
    if p['uncertainty_source']!='frozen_teacher_v3_reused' or not p['mean_variance_same_final_stage']:raise ValueError('unapproved uncertainty source/stage')
    if p['output_stage']!='l2_normalized' or p['covariance_representation']!='native_output_diagonal_surrogate':raise ValueError('unsupported output contract')
    for file,expected in p['consumed_files'].items():
        if sha(file)!=expected:raise ValueError('consumed input changed '+file)
    verify_backend_files(config['backend_files'],p['backend_files'])
    rows=[json.loads(s) for s in Path(p['observations_path']).read_text().splitlines()]
    ids=np.array([r['observation_id'] for r in rows],dtype=str)
    if len(set(ids))!=len(ids):raise ValueError('observation IDs not unique')
    if [r['source_order'] for r in rows]!=sorted(r['source_order'] for r in rows):raise ValueError('original observation order changed')
    for row in rows:
        roles={'query':'query','positive_passage':'document','premise':'query','hypothesis':'document'}
        if row['role'] not in roles or row['moment_side']!=roles[row['role']] or canonical_hash(row['text'])!=row['canonical_text_sha256']:
            raise ValueError('invalid actual role/moment side/hash')
    with np.load(p['input_path'],allow_pickle=False) as z:
        means={m:z['mean__'+m] for m in ops.MODEL_ORDER};variances={m:z['variance__'+m] for m in ops.MODEL_ORDER}
        mids={m:z['mean_ids__'+m] for m in ops.MODEL_ORDER};vids={m:z['variance_ids__'+m] for m in ops.MODEL_ORDER}
    for m in ops.MODEL_ORDER:
        policy=p['models'][m]['policy']
        if policy!=config['models'][m] or policy['native_dim']!=DIMS[m]:raise ValueError('model policy changed')
        validate_cache_arrays(means[m],variances[m],mids[m],vids[m],ids,DIMS[m])
        if np.max(np.abs(np.linalg.norm(means[m],axis=1)-1))>(2e-3 if m=='qwen' else 1e-5):raise ValueError('native norm contract failed')
        summary=p['models'][m]
        if summary['original_parameter_hash_before']!=summary['original_parameter_hash_after']:raise ValueError('encoder changed')
        # Read the two producer ID sidecars independently of the NPZ ID arrays.
        for kind,actual in [('mean_ids',mids[m]),('variance_ids',vids[m])]:
            candidates=[Path(f) for f in summary['output_checksums'] if Path(f).name==kind+'.npy']
            if len(candidates)!=1 or not np.array_equal(np.load(candidates[0],allow_pickle=False),actual):raise ValueError('independent sidecar mismatch')
    return E4Inputs(means,variances,mids,vids,rows,p,dict(p['consumed_files'],**{str(provenance_path):sha(provenance_path)}))

def make_alignment_split(data,config):
    record=json.loads(Path(data.provenance['split_path']).read_text());smoke=record['smoke']
    lookup={r['source_order']:i for i,r in enumerate(data.observations)}
    def remap(indices):return np.array([lookup[i] for i in indices],dtype=int)
    fit=remap(smoke['fit_indices']);diag=remap(smoke['diagnostic_indices'])
    if not len(fit) or not len(diag) or len(fit)>2048 or len(diag)>128:raise ValueError('smoke split sizes')
    if set(fit)&set(diag) or set(fit)|set(diag)!=set(range(len(data.observations))):raise ValueError('smoke cache/split coverage mismatch')
    if {data.observations[i]['canonical_text_sha256'] for i in fit}&{data.observations[i]['canonical_text_sha256'] for i in diag}:raise ValueError('text group leakage')
    q=remap(smoke['moment_query_indices']);d=remap(smoke['moment_document_indices'])
    if len(q)!=2 or len(d)!=4 or not set(q)|set(d)<=set(diag):raise ValueError('missing moment roles')
    six=list(q)+list(d)
    if len({data.observations[i]['canonical_text_sha256'] for i in six})!=6:raise ValueError('moment texts overlap')
    for indices,side in [(q,'query'),(d,'document')]:
        if any(data.observations[i]['moment_side']!=side for i in indices):raise ValueError('moment role relabeling')
    return AlignmentSplit(fit,diag,np.array([],dtype=int),dict(record,moment_query_indices=q.tolist(),moment_document_indices=d.tolist()))

def preflight(config):
    try:
        data=load_verified_inputs(config);split=make_alignment_split(data,config)
        return dict(status='ready_for_smoke',rows=len(data.observations),fit_rows=len(split.fit_indices),diagnostic_rows=len(split.diagnostic_indices),**scope_flags())
    except (FileNotFoundError,KeyError) as error:
        return dict(status='blocked_missing_inputs',reason=str(error),**scope_flags())

def fit_e4(data,split,config,progress=None):
    # Transparent timing wrappers in this isolated worker; original arithmetic
    # and caia.py bytes stay unchanged. Restore all functions even on failure.
    originals={k:getattr(caia,k) for k in ['compute_mean_stats','compute_cov_stats','fit_svd']}
    times={};fit_count=0
    def timed(name,fn):
        def call(*args,**kwargs):
            nonlocal fit_count
            phase=name
            if name=='fit_svd':phase=['r0','gpa','caia'][fit_count];fit_count+=1
            if progress:progress(phase)
            if name=='fit_svd' and config.get('mode')=='full' and progress:
                # Existing v3 observer hook only; no update/objective change.
                def checkpoint(sweep,maps,row):progress(f'{phase}/sweep_{sweep}')
                kwargs['checkpoint']=checkpoint
            start=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:times[phase]=time.perf_counter()-start
        return call
    before={m:float(np.max(np.abs(np.linalg.norm(data.means[m][split.fit_indices],axis=1)-1))) for m in ops.MODEL_ORDER}
    for name,fn in originals.items():setattr(caia,name,timed(name,fn))
    try:
        ix=split.fit_indices
        result=caia.fit_caia({m:x[ix] for m,x in data.means.items()},
            {m:x[ix] for m,x in data.variances.items()},{m:x[ix] for m,x in data.mean_ids.items()},
            {m:x[ix] for m,x in data.variance_ids.items()},gamma=None,include_shuffled=False,
            r0_sweeps=100 if config.get('mode')=='full' else 3,max_sweeps=300 if config.get('mode')=='full' else 3)
    finally:
        for name,fn in originals.items():setattr(caia,name,fn)
    result['phase_seconds']=times;result['fit_max_norm_correction']=before
    for arm in ['r0','gpa','caia']:
        if result[arm]['status']=='numerical_failure':raise ValueError(arm+' numerical_failure')
    return result

def validate_e4_transport(data,split,result,config):
    qix=np.array(split.audit['moment_query_indices']);dix=np.array(split.audit['moment_document_indices'])
    def side(ix,role):
        return make_side({m:a[ix] for m,a in data.means.items()},{m:a[ix] for m,a in data.variances.items()},data.mean_ids['qwen'][ix],role)
    q,d=side(qix,'query'),side(dix,'document');rows=[];geometry={}
    conditions=[('uniform',np.full((2,3),1/3),np.full((4,3),1/3))]
    conditions += [(f'onehot_{m}_{n}',np.tile(np.eye(3)[i],(2,1)),np.tile(np.eye(3)[j],(4,1))) for i,m in enumerate(ops.MODEL_ORDER) for j,n in enumerate(ops.MODEL_ORDER)]
    for arm in ['gpa','caia']:
        maps=result[arm]['maps'];geometry[arm]=ops.map_metadata(maps);overlap=ops.build_overlap_blocks(maps)
        for label,qa,da in conditions:
            q['alpha']=qa;d['alpha']=da
            qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
            fast=ops.pair_moments_implicit(qb,db,overlap)
            refq=ops.prepare_dense_reference(q,maps);refd=ops.prepare_dense_reference(d,maps)
            reference=ops.pair_moments_prepared_dense(refq,refd)
            for key,left,right in [('fused_mean_query',qb['fused_mean'],refq['fused_mean']),('fused_mean_document',db['fused_mean'],refd['fused_mean'])]:
                fast[key]=left;reference[key]=right
            rows+=compare_terms(fast,reference,arm,'float64',label)
            if label.startswith('onehot_'):
                for i,m in enumerate(ops.MODEL_ORDER):
                    if label==f'onehot_{m}_{m}':
                        fq,fd=q['native_means'][m],d['native_means'][m];vq,vd=q['native_vars'][m],d['native_vars'][m]
                        native=dict(zip(ops.TERMS,[fq@fd.T,(fq**2)@vd.T,vq@(fd**2).T,vq@vd.T,(fq**2)@vd.T+vq@(fd**2).T+vq@vd.T]))
                        if any(not np.allclose(fast[k],native[k],atol=1e-10,rtol=1e-8) for k in ops.TERMS):raise ValueError('native same-model moment invariance failed')
    if not all(r['pass_equivalence'] for r in rows):raise ValueError('dense/implicit numerical_failure')
    return rows,geometry

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

def write_e4_outputs(output,data,split,result,moments,geometry):
    output=Path(output)
    if any(result[a]['status']=='numerical_failure' for a in ['r0','gpa','caia']):raise ValueError('failed map export prohibited')
    if len(moments)!=140 or not all(r['pass_equivalence'] for r in moments):raise ValueError('incomplete or failed smoke E1 diagnostics')
    for arm in ['gpa','caia']:
        actual=ops.map_metadata(result[arm]['maps'])
        if arm not in geometry or geometry[arm].get('map_checksum')!=actual['map_checksum']:raise ValueError('missing/stale geometry diagnostics')
    diagnostic=output/'diagnostic_only';diagnostic.mkdir(exist_ok=False)
    for arm in ['gpa','caia']:np.savez(diagnostic/(arm+'_maps.npz'),**result[arm]['maps'])
    summary={key:clean(value) for key,value in result.items() if key not in ['r0','gpa','caia']}
    for arm in ['r0','gpa','caia']:summary[arm]=clean({k:v for k,v in result[arm].items() if k!='maps'})
    write_json(output/'solver_summary.json',summary);write_json(output/'geometry.json',geometry)
    write_csv(output/'moment_equivalence.csv',moments)
    write_json(output/'split.json',clean(dict(fit_indices=split.fit_indices,diagnostic_indices=split.diagnostic_indices,audit=split.audit)))
    active=result['gamma']>0 and 'no_cov_guidance' not in result['caia'].get('quality_flags',[])
    status=dict(status='smoke_pass',alignment_refit_performed=True,covariance_guidance_active=active,**scope_flags())
    write_json(output/'status.json',status)
    write_json(output/'handoff_manifest.json',dict(status,approved_maps=[],diagnostic_maps={str(f):sha(f) for f in diagnostic.glob('*.npz')},
        input_files=data.checksums,config_file=data.provenance.get('execution_config'),
        code_files={str(Path(__file__).parent/f):sha(Path(__file__).parent/f) for f in ['caia.py','covariance_ops.py','run_covariance_validation.py','e4_refit.py','e4_data.py','run_e4_refit.py']},
        uncertainty_source='frozen_teacher_v3_reused',prior_run_v2_used_as_new_evidence=False))
    lines=['# CAIA E4 v3 smoke','', 'status: smoke_pass; alignment_ready_for_handoff=false.','',
        f'Fit {len(split.fit_indices)} / diagnostic {len(split.diagnostic_indices)} observations; new fit-only gamma={result["gamma"]:.12g}.',
        f'Float64 dense/implicit: {len(moments)} term comparisons; max abs error '+str(max((r['max_abs_error'] for r in moments),default=0)),
        '', '| Arm | status | sweeps | seconds |','|---|---|---|---|']
    for a in ['r0','gpa','caia']:lines.append(f'| {a} | {result[a]["status"]} | {result[a]["sweeps"]} | {result[a]["seconds"]:.6f} |')
    lines+=['','3 sweeps are a wiring check, not convergence or downstream validation. GPA is the continued gpa result, not R0.',
        'Original teacher fitting endpoint roles and query-style encoding policies are retained; this is not a new original UEC-LA posterior or standard retrieval document-path validation.',
        'Cache preparation includes new frozen encoder inference; the E4 numerical process does not. Full E4, E2, repeated cost benchmarks and downstream scores are not run.',
        'Only diagnostic_only maps are saved. None are approved for final handoff.','']
    (output/'report.md').write_text('\n'.join(lines))

def run_with_deadline(command,output,budget=300):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    with (output/'worker.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            status=dict(status='smoke_timeout',alignment_refit_performed=False,**scope_flags())
        else:
            path=output/'status.json'
            try:status=json.loads(path.read_text()) if path.exists() else {}
            except json.JSONDecodeError:status={}
            if not status or (code!=0 and status.get('status')=='smoke_pass'):
                status=dict(status='smoke_failed',exit_code=code,alignment_refit_performed=False,**scope_flags())
    status['wall_seconds']=time.perf_counter()-start;status['wall_time_budget_seconds']=budget
    progress=output/'progress.json'
    if progress.exists():
        try:status['last_progress']=json.loads(progress.read_text())
        except json.JSONDecodeError:status['last_progress_error']='interrupted progress write; worker exit/timeout status preserved'
    write_json(output/'status.json',status);return status
