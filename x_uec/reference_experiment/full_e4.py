"""Full E4 fitting and diagnostics; unchanged CAIA v3 numerical backend."""
from collections import Counter
import json
import shutil
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import caia
import covariance_ops as ops
from e4_refit import (AlignmentSplit,load_verified_inputs,fit_e4,clean,write_json_atomic)
from run_covariance_validation import sha,write_json,write_csv,compare_terms,weight_conditions
from full_audit import audit_full_inputs,snapshot_paths,verify_snapshot

def make_full_split(data):
    record=json.loads(Path(data.provenance['split_path']).read_text());scope=record['selected_scope']
    lookup={r['source_order']:i for i,r in enumerate(data.observations)}
    def remap(indices):return np.asarray([lookup[i] for i in indices],dtype=int)
    fit,diag=remap(record['full']['fit_indices']),remap(record['full']['diagnostic_indices'])
    if not len(fit) or not len(diag) or set(fit)&set(diag) or set(fit)|set(diag)!=set(range(len(data.observations))):raise ValueError('full split/caches do not cover all eligible observations')
    if list(scope['selected_indices'])!=sorted(record['full']['fit_indices']+record['full']['diagnostic_indices']):raise ValueError('full cache truncated to subset')
    if {data.observations[i]['canonical_text_sha256'] for i in fit}&{data.observations[i]['canonical_text_sha256'] for i in diag}:raise ValueError('fit/diagnostic canonical overlap')
    q,d=remap(scope['moment_query_indices']),remap(scope['moment_document_indices'])
    if len(q)!=2 or len(d)!=4 or len({data.observations[i]['canonical_text_sha256'] for i in list(q)+list(d)})!=6 or not set(q)|set(d)<=set(diag):raise ValueError('invalid fixed moment observations')
    for ix,side in [(q,'query'),(d,'document')]:
        if any(data.observations[i]['moment_side']!=side for i in ix):raise ValueError('moment side mismatch')
    return AlignmentSplit(fit,diag,np.array([],dtype=int),dict(record,moment_query_indices=q.tolist(),moment_document_indices=d.tolist()))

def evaluate_alignment_losses(data,split,result):
    rows=[];gamma=result['gamma']
    for label,indices in [('fit',split.fit_indices),('diagnostic',split.diagnostic_indices)]:
        x,v,ids=caia.check_anchor_inputs({m:a[indices] for m,a in data.means.items()},{m:a[indices] for m,a in data.variances.items()},
            {m:a[indices] for m,a in data.mean_ids.items()},{m:a[indices] for m,a in data.variance_ids.items()})
        s=caia.compute_mean_stats(SimpleNamespace(x=x,ids=ids));k=caia.compute_cov_stats(SimpleNamespace(v=v,ids=ids,source=data.provenance.get('uncertainty_source','frozen_teacher_v3_reused')))
        for arm in ['r0','gpa','caia']:
            loss=caia.objective(result[arm]['maps'],s,k,gamma)
            for pair,values in [('all',loss),*loss['pairs'].items()]:
                rows.append(dict(split=label,arm=arm,model_pair=pair,observations=len(indices),gamma=gamma,
                    mean_loss=values['mean'],cov_loss=values['cov'],total_loss=values['mean']+gamma*values['cov']))
    return rows

def select_pairs(left,right,observations,maximum,seed):
    left,right=np.asarray(left),np.asarray(right);lh=[observations[i]['canonical_text_sha256'] for i in left];rh=[observations[i]['canonical_text_sha256'] for i in right]
    lc,rc=Counter(lh),Counter(rh);available=len(left)*len(right)-sum(n*rc.get(h,0) for h,n in lc.items())
    target=min(maximum,available);rng=np.random.default_rng(seed);seen=set();result=[]
    while len(result)<target:
        i,j=divmod(int(rng.integers(len(left)*len(right))),len(right))
        if lh[i]==rh[j] or (i,j) in seen:continue
        seen.add((i,j));result.append((int(left[i]),int(right[j])))
    return result

def geometry_checks(data,split,result,output):
    pairs=select_pairs(split.diagnostic_indices,split.diagnostic_indices,data.observations,1000,42)
    if not pairs:raise ValueError('no distinct diagnostic geometry pairs')
    write_json(output/'geometry_pairs.json',dict(seed=42,pairs=[dict(left_id=data.observations[i]['observation_id'],right_id=data.observations[j]['observation_id'],left_index=i,right_index=j) for i,j in pairs]))
    a=np.array([i for i,j in pairs]);b=np.array([j for i,j in pairs]);rows=[];passed=True
    for arm in ['gpa','caia']:
        meta=ops.map_metadata(result[arm]['maps'])
        for m,r in result[arm]['maps'].items():
            left,right=data.means[m][a],data.means[m][b]
            before=(left*right).sum(1)/(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1))
            al,ar=left@r,right@r;after=(al*ar).sum(1)/(np.linalg.norm(al,axis=1)*np.linalg.norm(ar,axis=1))
            delta=np.abs(after-before);ok=bool(np.isfinite(after).all() and np.all(delta<=1e-10+1e-8*np.abs(before)))
            passed &= ok
            rows.append(dict(arm=arm,model=m,pairs=len(pairs),max_cosine_abs_error=float(delta.max()),row_isometry_error=meta['row_isometry_error'][m],passed=ok))
    write_csv(output/'geometry_preservation.csv',rows)
    return passed,rows

def role_inputs(data,indices,role,dtype=np.float64):
    indices=np.asarray(indices)
    return dict(native_means={m:a[indices].astype(dtype) for m,a in data.means.items()},
        native_vars={m:a[indices].astype(dtype) for m,a in data.variances.items()},
        mean_ids_by_model={m:a[indices] for m,a in data.mean_ids.items()},variance_ids_by_model={m:a[indices] for m,a in data.variance_ids.items()},
        alpha=np.full((len(indices),3),1/3,dtype=dtype),alpha_ids=np.array([data.observations[i]['observation_id'] for i in indices]),role=role)

def full_moment_checks(data,split,result,output):
    qi,di=split.audit['moment_query_indices'],split.audit['moment_document_indices'];rows=[];map_audit={}
    conditions=weight_conditions(2,4,42,True)
    np.savez(output/'moment_weights.npz',**{f'{i}_{side}':values for i,(_,q,d) in enumerate(conditions) for side,values in [('query',q),('document',d)]})
    write_json(output/'moment_observations.json',dict(query_ids=[data.observations[i]['observation_id'] for i in qi],document_ids=[data.observations[i]['observation_id'] for i in di],conditions=[x[0] for x in conditions]))
    for arm in ['gpa','caia']:
        ops.map_metadata(result[arm]['maps']);map_audit[arm]={}
        for dtype in [np.float64,np.float32]:
            maps={m:r.astype(dtype) for m,r in result[arm]['maps'].items()};map_audit[arm][np.dtype(dtype).name]=ops.map_metadata(maps)
            overlap=ops.build_overlap_blocks(maps);q=role_inputs(data,qi,'query',dtype);d=role_inputs(data,di,'document',dtype)
            for label,qa,da in conditions:
                q['alpha']=qa.astype(dtype);d['alpha']=da.astype(dtype)
                qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
                fast=ops.pair_moments_implicit(qb,db,overlap)
                rq=ops.prepare_dense_reference(q,maps);rd=ops.prepare_dense_reference(d,maps);reference=ops.pair_moments_prepared_dense(rq,rd)
                for key,a,b in [('fused_mean_query',qb['fused_mean'],rq['fused_mean']),('fused_mean_document',db['fused_mean'],rd['fused_mean'])]:fast[key]=a;reference[key]=b
                rows+=compare_terms(fast,reference,arm,dtype,label)
                for m in ops.MODEL_ORDER:
                    if label==f'onehot_{m}_{m}':
                        fq,fd=q['native_means'][m],d['native_means'][m];vq,vd=q['native_vars'][m],d['native_vars'][m]
                        native=dict(zip(ops.TERMS,[fq@fd.T,(fq**2)@vd.T,vq@(fd**2).T,vq@vd.T,(fq**2)@vd.T+vq@(fd**2).T+vq@vd.T]))
                        rows+=compare_terms(fast,native,arm,dtype,'native_invariance_'+m)
    write_csv(output/'moment_equivalence.csv',rows);write_json(output/'map_geometry_by_dtype.json',map_audit)
    return bool(rows) and all(r['pass_equivalence'] for r in rows),rows

def diagonal_checks(data,split,result,output):
    qix=np.array([i for i in split.diagnostic_indices if data.observations[i]['moment_side']=='query']);dix=np.array([i for i in split.diagnostic_indices if data.observations[i]['moment_side']=='document'])
    pairs=select_pairs(qix,dix,data.observations,1000,42)
    if not pairs:raise ValueError('no true-side distinct diagnostic pairs')
    write_json(output/'diagnostic_pairs.json',dict(seed=42,count=len(pairs),same_original_triplet=sum(data.observations[i]['original_triplet_id']==data.observations[j]['original_triplet_id'] for i,j in pairs),
        pairs=[dict(query_id=data.observations[i]['observation_id'],document_id=data.observations[j]['observation_id'],query_index=i,document_index=j) for i,j in pairs]))
    qlookup={i:j for j,i in enumerate(qix)};dlookup={i:j for j,i in enumerate(dix)}
    conditions=weight_conditions(len(qix),len(dix),42,False)
    np.savez(output/'diagnostic_weights.npz',**{f'{i}_{side}':values for i,(_,q,d) in enumerate(conditions) for side,values in [('query',q),('document',d)]})
    long=[];summary=[];controls=True
    for arm in ['gpa','caia']:
        maps=result[arm]['maps'];overlap=ops.build_overlap_blocks(maps)
        for label,qa,da in conditions:
            q=role_inputs(data,qix,'query');d=role_inputs(data,dix,'document');q['alpha']=qa;d['alpha']=da
            qb=ops.prepare_raw_bundle(maps=maps,**q);db=ops.prepare_raw_bundle(maps=maps,**d)
            full={k:[] for k in ops.TERMS};diag={k:[] for k in ops.TERMS}
            for start in range(0,len(pairs),64):
                group=pairs[start:start+64];qi=np.array([qlookup[i] for i,j in group]);di=np.array([dlookup[j] for i,j in group])
                smallq=ops.slice_bundle(qb,qi);smalld=ops.slice_bundle(db,di)
                a=ops.pair_moments_implicit(smallq,smalld,overlap);b=ops.pair_moments_diagonal_ablation(smallq,smalld,maps)
                for k in ops.TERMS:full[k].extend(np.diag(a[k]));diag[k].extend(np.diag(b[k]))
            for k in ops.TERMS:
                fv,dv=np.array(full[k]),np.array(diag[k]);delta=dv-fv;absolute=np.abs(delta);valid=fv>1e-12;relative=absolute[valid]/fv[valid]
                if k=='dot_mean' or label=='onehot_qwen_qwen':controls &= bool(np.all(absolute<=1e-10+1e-8*np.abs(fv)))
                summary.append(dict(arm=arm,condition=label,term=k,pairs=len(pairs),signed_mean=float(delta.mean()),signed_min=float(delta.min()),signed_max=float(delta.max()),
                    absolute_median=float(np.median(absolute)),absolute_p95=float(np.quantile(absolute,.95)),absolute_max=float(absolute.max()),
                    relative_median=float(np.median(relative)) if len(relative) else None,relative_p95=float(np.quantile(relative,.95)) if len(relative) else None,
                    relative_max=float(relative.max()) if len(relative) else None,near_zero_count=int((~valid).sum())))
                for index,((i,j),f,v) in enumerate(zip(pairs,fv,dv)):
                    long.append(dict(arm=arm,condition=label,pair=index,query_id=data.observations[i]['observation_id'],document_id=data.observations[j]['observation_id'],term=k,
                        full=float(f),diagonal_ablation=float(v),signed_difference=float(v-f),absolute_difference=float(abs(v-f))))
    write_csv(output/'diagonal_diagnostic.csv',long);write_csv(output/'diagonal_summary.csv',summary)
    write_json(output/'diagonal_top_examples.json',sorted([r for r in long if r['term']=='dot_variance'],key=lambda r:r['absolute_difference'],reverse=True)[:10])
    return controls,summary

def readiness(result,diagnostics):
    if any(result[a]['status']=='numerical_failure' for a in ['r0','gpa','caia']) or not all(diagnostics.get(k,False) for k in ['geometry_passed','moment_passed','diagonal_completed']):return 'numerical_failure'
    for arm in ['gpa','caia']:
        state=result[arm].get('final_stationarity',{})
        residual,gap=state.get('relative_residual'),state.get('relative_gap')
        if result[arm]['status']!='stationary_tolerance' or residual is None or gap is None or not np.isfinite([residual,gap]).all() or not 0<=residual<=1e-7 or not 0<=gap<=1e-10:return 'review_required'
    return 'alignment_ready_for_handoff'

def complete_diagnostics(losses,geometry,moments,diagonal,geometry_pairs,diagonal_pairs):
    def exact(rows,fields,expected):
        keys=[tuple(r[f] for f in fields) for r in rows]
        return len(keys)==len(expected) and set(keys)==expected
    arms=['gpa','caia'];dtypes=['float64','float32']
    cases=[r[0] for r in weight_conditions(2,4,42,True)]
    terms=[*ops.TERMS,'fused_mean_query','fused_mean_document']
    expected={(a,d,c,t) for a in arms for d in dtypes for c in cases for t in terms}
    expected|={(a,d,'native_invariance_'+m,t) for a in arms for d in dtypes for m in ops.MODEL_ORDER for t in ops.TERMS}
    if not exact(moments,['arm','dtype','case','term'],expected) or not all(r['pass_equivalence'] and np.isfinite(r['max_abs_error']) for r in moments):return False
    if not exact(geometry,['arm','model'],{(a,m) for a in arms for m in ops.MODEL_ORDER}) or geometry_pairs<1:return False
    if not all(r['passed'] and r['pairs']==geometry_pairs and np.isfinite([r['max_cosine_abs_error'],r['row_isometry_error']]).all() for r in geometry):return False
    if not exact(diagonal,['arm','condition','term'],{(a,c[0],t) for a in arms for c in weight_conditions(2,4,42,False) for t in ops.TERMS}) or diagonal_pairs<1:return False
    if not all(r['pairs']==diagonal_pairs and np.isfinite([r['signed_mean'],r['absolute_max']]).all() for r in diagonal):return False
    if not exact(losses,['split','arm','model_pair'],{(s,a,p) for s in ['fit','diagnostic'] for a in ['r0','gpa','caia'] for p in ['all','qwen_harrier','qwen_gte','harrier_gte']}):return False
    return all(np.isfinite([r['mean_loss'],r['cov_loss'],r['total_loss']]).all() for r in losses)

def run_full(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();stage='input_validation'
    def progress(phase):
        nonlocal stage
        stage=phase;write_json_atomic(output/'progress.json',dict(phase=phase,seconds=time.perf_counter()-start))
    flags=dict(delivery_version='v3',mode=config.get('mode','full') if config.get('input_profile')=='standalone_v1' else 'full',encoder_weights_updated=False,inference_executed_in_this_run=False,posterior_refit_performed=False,
        alignment_refit_performed=False,alignment_only_heldout=True,posterior_unseen=False,uec_final_score_validated=False,uec_fusion_executed=False,downstream_evaluated=False,theorem_extension_proved=False)
    write_json_atomic(output/'status.json',dict(status='running',**flags))
    try:
        code=snapshot_paths(Path(__file__).parent.glob('*.py'))
        write_json(output/'executed_code_before.json',code)
        standalone=config.get('input_profile')=='standalone_v1'
        if standalone:
            import sys
            sys.path.append(str(Path(__file__).resolve().parents[1]))
            from standalone.alignment import load_inputs,audit_and_split,profile_status
            data=load_inputs(config);audit,split=audit_and_split(data)
            flags['mode']=config['mode']
            code.update(snapshot_paths((Path(__file__).resolve().parents[1]/'standalone').glob('*.py')))
            write_json(output/'executed_code_before.json',code)
        else:
            config=dict(config,mode='full');data=load_verified_inputs(config);split=make_full_split(data)
            audit=audit_full_inputs(data)
        progress(stage);write_json(output/'verified_full_audit.json',audit)
        if config.get('runtime_verification_path'):
            file=Path(config['runtime_verification_path'])
            if sha(file)!=config['runtime_verification_sha256']:raise ValueError('runtime verification changed')
            runtime=json.loads(file.read_text());data.checksums[str(file)]=sha(file)
            for path,expected in runtime['additional_consumed_files'].items():
                if sha(path)!=expected:raise ValueError('retrospective runtime source changed')
                data.checksums[path]=expected
            write_json(output/'runtime_policy_verification.json',runtime)
        if '_config_path' in config:data.checksums[config['_config_path']]=sha(config['_config_path'])
        write_json(output/'preflight.json',dict(status='passed',cache_scope=data.provenance['cache_scope'],fit_observations=len(split.fit_indices),diagnostic_observations=len(split.diagnostic_indices)))
        write_json(output/'split.json',clean(split.audit));input_seconds=time.perf_counter()-start
        result=fit_e4(data,split,config,progress);flags['alignment_refit_performed']=True
        summary={k:clean(v) for k,v in result.items() if k not in ['r0','gpa','caia']}
        for arm in ['r0','gpa','caia']:summary[arm]=clean({k:v for k,v in result[arm].items() if k!='maps'})
        write_json(output/'solver_summary.json',summary)
        progress('matched_fit_diagnostic_losses');losses=evaluate_alignment_losses(data,split,result);write_csv(output/'alignment_losses.csv',losses)
        progress('geometry');geometry_ok,geometry=geometry_checks(data,split,result,output)
        progress('dense_implicit_moments');moment_ok,moments=full_moment_checks(data,split,result,output)
        progress('diagonal_diagnostic');diagonal_ok,diagonal=diagonal_checks(data,split,result,output) if geometry_ok and moment_ok else (False,[])
        gp=len(select_pairs(split.diagnostic_indices,split.diagnostic_indices,data.observations,1000,42))
        qi=[i for i in split.diagnostic_indices if data.observations[i]['moment_side']=='query'];di=[i for i in split.diagnostic_indices if data.observations[i]['moment_side']=='document']
        dp=len(select_pairs(qi,di,data.observations,1000,42))
        coverage=complete_diagnostics(losses,geometry,moments,diagonal,gp,dp)
        diagnostics=dict(geometry_passed=geometry_ok,moment_passed=moment_ok,diagonal_completed=diagonal_ok,diagnostic_coverage_passed=coverage)
        if not coverage:diagnostics['moment_passed']=False
        state=readiness(result,diagnostics)
        if standalone:state=profile_status(state,config['mode'])
        progress('rechecking_inputs_and_export')
        for file,expected in data.checksums.items():
            if sha(file)!=expected:raise ValueError('input changed during full run '+file)
        verify_snapshot(code)
        approved=state=='alignment_ready_for_handoff';maps={}
        folder=None;staging=output/'.maps_staging'
        if state!='numerical_failure':
            folder=output/('maps' if approved else 'diagnostic_only');staging.mkdir()
            for arm in ['gpa','caia']:
                file=staging/(arm+'_maps.npz');np.savez(file,**result[arm]['maps']);maps[str(folder/file.name)]=sha(file)
        status=dict(status=state,alignment_ready_for_handoff=approved,covariance_guidance_active=result['gamma']>0 and 'no_cov_guidance' not in result['caia']['quality_flags'],
            wall_seconds=time.perf_counter()-start,input_validation_seconds=input_seconds,**flags,**diagnostics)
        write_json(output/'handoff_manifest.json',dict(status,uncertainty_source=data.provenance['uncertainty_source'],gamma=result['gamma'],input_files=data.checksums,code_files=code,
            maps=maps,approved_maps=list(maps) if approved else [],actual_models={m:record['policy'] for m,record in data.provenance['models'].items()}))
        lines=[f'# CAIA v3 {flags["mode"]} E4','',f'status: {state}',f'fit={len(split.fit_indices)}, diagnostic={len(split.diagnostic_indices)}, gamma={result["gamma"]:.12g}',
            f'geometry={geometry_ok}, raw moment terms={len(moments)} all_pass={moment_ok}, diagonal controls={diagonal_ok}',
            '', '| arm | status | sweeps | seconds |','|---|---|---|---|']
        for arm in ['r0','gpa','caia']:lines.append(f'| {arm} | {result[arm]["status"]} | {result[arm]["sweeps"]} | {result[arm]["seconds"]:.6f} |')
        lines+=['',f'Uncertainty source: {data.provenance["uncertainty_source"]}. Original UEC-LA, downstream scores and theorem extension are not validated.',
            'GPA covariance loss in alignment_losses.csv is recomputed with matched K and the same fit gamma; GPA trace cov=0 is not the covariance diagnostic.',
            'All plots/tables describe alignment/numerics, not retrieval or STS performance. Nonstationary maps remain diagnostic_only.','']
        (output/'report.md').write_text('\n'.join(lines))
        if folder is not None:
            required=['preflight.json','split.json','verified_full_audit.json','solver_summary.json','alignment_losses.csv','geometry_preservation.csv','moment_equivalence.csv',
                'map_geometry_by_dtype.json','moment_weights.npz','moment_observations.json','geometry_pairs.json','diagnostic_pairs.json','diagnostic_weights.npz',
                'diagonal_diagnostic.csv','diagonal_summary.csv','handoff_manifest.json','report.md','executed_code_before.json']
            if not all((output/name).is_file() for name in required):raise ValueError('incomplete required diagnostic artifacts')
            staging.rename(folder)
        progress('completed')
        write_json_atomic(output/'status.json',status)
        return status
    except Exception as error:
        # Only this fresh run's own generated map directories can exist here.
        for name in ['.maps_staging','maps']:
            folder=output/name
            if folder.exists():shutil.rmtree(folder)
        status=dict(status='numerical_failure' if isinstance(error,ValueError) else 'full_failed',error_type=type(error).__name__,error=str(error),last_phase=stage,wall_seconds=time.perf_counter()-start,alignment_ready_for_handoff=False,**flags)
        write_json_atomic(output/'status.json',status);raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());config['_config_path']=str(Path(a.config).resolve())
    status=run_full(config,a.output);print(json.dumps(status,indent=2))
    if status['status']=='numerical_failure':raise SystemExit(2)
