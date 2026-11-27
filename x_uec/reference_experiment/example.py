"""Standalone synthetic demo or fit on arrays exported by the recipient's UEC."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from caia import NAMES, fit_caia, transform_gaussian


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo',action='store_true')
    mode.add_argument('--input',type=Path,help='Native means/output variance and per-model IDs NPZ')
    parser.add_argument('--provenance',type=Path,help='Required for real data; see README')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gamma',type=float,default=None,help='Prespecified value; default fit-only rule')
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; use a fresh directory')
    hashes={}
    if args.demo:
        rng=np.random.default_rng(41);means={n:rng.normal(size=(40,d)) for n,d in zip(NAMES,[8,5,6])}
        means={n:a/np.linalg.norm(a,axis=1,keepdims=True) for n,a in means.items()}
        variances={n:rng.uniform(.001,.02,size=a.shape) for n,a in means.items()}
        ids={n:np.array([f'demo:{i}:texts' for i in range(40)]) for n in NAMES};vids=ids
        provenance=dict(dataset='synthetic; no measured UEC result',covariance_source='synthetic_nonnegative')
    else:
        if args.provenance is None:parser.error('--provenance is required for real UEC arrays')
        provenance=json.loads(args.provenance.read_text())
        for key in ['dataset','split','selection_manifest_sha256','models','uncertainty_source','output_stage','role_policy','evaluation_overlap_audit']:
            if key not in provenance:parser.error('Missing provenance key: '+key)
        if provenance['output_stage']!='l2_normalized':parser.error('Expected output_stage=l2_normalized; propagate L2 covariance first')
        if set(provenance['models'])!=set(NAMES):parser.error('Provenance requires all three models')
        with np.load(args.input,allow_pickle=False) as pack:
            means={n:pack['mean__'+n] for n in NAMES}
            variances={n:pack['variance__'+n] for n in NAMES}
            ids={n:pack['mean_ids__'+n] for n in NAMES}
            vids={n:pack['variance_ids__'+n] for n in NAMES}
        hashes={str(p.resolve()):digest(p) for p in [args.input,args.provenance]}
    result=fit_caia(means,variances,ids,vids,gamma=args.gamma)
    args.output.mkdir(parents=True,exist_ok=False)
    for arm in ['gpa','caia','shuffled']:
        np.savez(args.output/(arm+'_maps.npz'),**result[arm]['maps'])
    np.savez(args.output/'shuffle_permutations.npz',**result['shuffle_permutations'])
    np.savez(args.output/'fit_ids.npz',ids=ids['qwen'])
    # Demonstrate exact transport with only two rows; never save a full fit N×D×D tensor.
    from caia import check_anchor_inputs
    x,v,_=check_anchor_inputs(means,variances,ids,vids)
    m,c=transform_gaussian(x['harrier'][:2],v['harrier'][:2],result['caia']['maps']['harrier'])
    np.savez(args.output/'transport_example.npz',mean=m,covariance=c)
    summary=dict(synthetic_demo=args.demo,UEC_fusion_executed=False,dataset_alignment_evaluated=False,
        provenance=provenance,input_sha256=hashes,
        source_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('caia.py')]},
        gamma=result['gamma'],gamma_evidence=result['gamma_evidence'],
        solver={a:{k:v for k,v in result[a].items() if k!='maps'} for a in ['r0','gpa','caia','shuffled']},
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
        note='budget/stalled status is not convergence; inspect every status before scientific claims')
    (args.output/'summary.json').write_text(json.dumps(clean(summary),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),synthetic_demo=args.demo,
        solver_status={a:result[a]['status'] for a in ['r0','gpa','caia','shuffled']})))
    if any(result[a]['status']=='numerical_failure' for a in ['gpa','caia','shuffled']):raise SystemExit(2)


if __name__=='__main__':main()
