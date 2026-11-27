"""Portable v3 fitting example. Supply an already audited alignment-fit cache."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from caia import NAMES,check_anchor_inputs,fit_caia
from covariance_ops import map_metadata

MODEL_INFO=dict(qwen=('Qwen/Qwen3-Embedding-0.6B',1024),
    harrier=('microsoft/harrier-oss-v1-270m',640),gte=('thenlper/gte-base',768))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value

def require_declaration(value,name):
    def strings(item):
        if isinstance(item,str):return [item]
        if isinstance(item,dict):return [s for v in item.values() for s in strings(v)]
        if isinstance(item,(list,tuple)):return [s for v in item for s in strings(v)]
        return []
    fields=strings(value)
    if not fields or not any(s.strip() for s in fields) or any(re.search(r'replace_with|placeholder|\bTODO\b|\bTBD\b',s,re.IGNORECASE) for s in fields):
        raise ValueError('complete the actual non-placeholder '+name+' declaration')

def load_real_inputs(input_path,provenance_path):
    p=json.loads(Path(provenance_path).read_text())
    for key in ['dataset','split','selection_manifest_sha256','models','uncertainty_source','output_stage',
                'covariance_representation','role_policy','evaluation_overlap_audit','alignment_diagnostic_separated']:
        if key not in p:raise ValueError('missing provenance field '+key)
    if p['split']!='alignment_fit' or p['alignment_diagnostic_separated'] is not True:
        raise ValueError('supply only a previously fixed alignment-fit subset, with diagnostic separated')
    for name in ['dataset','uncertainty_source','role_policy']:require_declaration(p[name],name)
    if p['output_stage']!='l2_normalized' or p['covariance_representation']!='native_output_diagonal_surrogate':
        raise ValueError('expected approved native diagonal variance at final L2 output')
    audit=p['evaluation_overlap_audit']
    if not isinstance(audit,dict) or audit.get('status')!='completed' or type(audit.get('fit_evaluation_canonical_overlap')) is not int or audit['fit_evaluation_canonical_overlap']!=0:
        raise ValueError('completed fit/evaluation text-overlap audit required')
    require_declaration(audit.get('scope'),'evaluation_overlap_audit.scope')
    selection_hash=p['selection_manifest_sha256']
    if not isinstance(selection_hash,str) or len(selection_hash)!=64 or any(c not in '0123456789abcdef' for c in selection_hash):
        raise ValueError('selection_manifest_sha256 must identify the actual existing selection')
    if set(p['models'])!=set(NAMES) or not p['uncertainty_source'] or not p['role_policy']:raise ValueError('missing model/source/role declaration')
    with np.load(input_path,allow_pickle=False) as z:
        means={m:z['mean__'+m] for m in NAMES};variances={m:z['variance__'+m] for m in NAMES}
        mids={m:z['mean_ids__'+m] for m in NAMES};vids={m:z['variance_ids__'+m] for m in NAMES}
    for m,(repo,dim) in MODEL_INFO.items():
        model=p['models'][m]
        if model.get('model_id')!=repo or model.get('dimension')!=dim or not isinstance(model.get('revision'),str) or not model['revision']:
            raise ValueError('model ID/revision/native dimension declaration: '+m)
        require_declaration(model['revision'],m+' revision')
        if means[m].dtype not in (np.dtype('float32'),np.dtype('float64')) or variances[m].dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('real float32/float64 mean and output variance required; no complex/string/integer coercion')
        if means[m].ndim!=2 or means[m].shape[1]!=dim:raise ValueError('native dimension mismatch '+m)
        if mids[m].dtype.kind!='U' or vids[m].dtype.kind!='U':raise ValueError('Unicode producer IDs required')
    # Validate all original arrays without returning the normalized working copies.
    check_anchor_inputs(means,variances,mids,vids)
    return means,variances,mids,vids,p

def demo_inputs():
    rng=np.random.default_rng(41)
    means={m:rng.normal(size=(40,d)) for m,d in zip(NAMES,[8,5,6])}
    means={m:x/np.linalg.norm(x,axis=1,keepdims=True) for m,x in means.items()}
    var={m:rng.uniform(.001,.02,x.shape) for m,x in means.items()}
    ids={m:np.array([f'toy:{i}:query' for i in range(40)]) for m in NAMES}
    return means,var,ids,{m:a.copy() for m,a in ids.items()},dict(dataset='synthetic demo; not an experiment score')

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo',action='store_true');mode.add_argument('--input',type=Path)
    parser.add_argument('--provenance',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--r0-sweeps',type=int,default=100);parser.add_argument('--max-sweeps',type=int,default=300)
    parser.add_argument('--include-shuffled',action='store_true');args=parser.parse_args()
    if args.output.exists():parser.error('output exists; use a fresh directory')
    if not args.demo and args.provenance is None:parser.error('--provenance is required for real inputs')
    try:
        means,variances,mids,vids,provenance=demo_inputs() if args.demo else load_real_inputs(args.input,args.provenance)
        result=fit_caia(means,variances,mids,vids,gamma=None,include_shuffled=args.include_shuffled,
                        r0_sweeps=args.r0_sweeps,max_sweeps=args.max_sweeps)
        arms=['r0','gpa','caia']+(['shuffled'] if args.include_shuffled else [])
        for arm in arms:
            if result[arm]['status']=='numerical_failure':raise ValueError(arm+' numerical_failure; no maps exported')
            map_metadata(result[arm]['maps'])
    except (ValueError,KeyError,FileNotFoundError) as error:parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    for arm in arms:
        if arm!='r0':np.savez(args.output/(arm+'_maps.npz'),**result[arm]['maps'])
    np.savez(args.output/'fit_ids.npz',ids=mids['qwen'])
    if args.include_shuffled:np.savez(args.output/'shuffle_permutations.npz',**result['shuffle_permutations'])
    summary=dict(delivery_version='v3',synthetic_demo=args.demo,provenance=provenance,
        input_sha256={} if args.demo else {str(p):digest(p) for p in [args.input,args.provenance]},
        code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('caia.py'),Path(__file__).with_name('covariance_ops.py')]},
        gamma=result['gamma'],gamma_evidence=result['gamma_evidence'],
        solver={a:{k:v for k,v in result[a].items() if k!='maps'} for a in arms},
        solver_stationary=all(result[a]['status']=='stationary_tolerance' for a in ['gpa','caia']),
        covariance_guidance_active=result['gamma']>0 and 'no_cov_guidance' not in result['caia']['quality_flags'],
        alignment_ready_for_handoff=False,UEC_fusion_executed=False,uec_score_computed=False,
        note='Candidate maps only. Review solver status, independent geometry/moment diagnostics and your actual UEC contract before use. No dense per-observation covariance is exported.')
    (args.output/'summary.json').write_text(json.dumps(clean(summary),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),solver_status={a:result[a]['status'] for a in arms},alignment_ready_for_handoff=False)))

if __name__=='__main__':main()
