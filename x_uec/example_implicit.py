"""Small standalone raw-moment demonstration; no encoder or real UEC score."""
import argparse
import json
from pathlib import Path
import numpy as np
from covariance_ops import (MODEL_ORDER,TERMS,build_overlap_blocks,prepare_raw_bundle,
    pair_moments_implicit,pair_moments_dense_reference,iter_pair_moments)

def fixture():
    rng=np.random.default_rng(42);maps={'qwen':np.eye(8)}
    for m,d in zip(MODEL_ORDER[1:],[5,6]):maps[m]=np.linalg.qr(rng.normal(size=(8,d)))[0].T
    def side(n,role):
        means={};var={}
        for m in MODEL_ORDER:
            x=rng.normal(size=(n,len(maps[m])));means[m]=x/np.linalg.norm(x,axis=1,keepdims=True)
            var[m]=rng.uniform(.001,.02,x.shape)
        ids=np.array([f'toy:{role}:{i}' for i in range(n)])
        return dict(native_means=means,native_vars=var,alpha=np.full((n,3),1/3),
            mean_ids_by_model={m:ids.copy() for m in MODEL_ORDER},
            variance_ids_by_model={m:ids.copy() for m in MODEL_ORDER},alpha_ids=ids.copy(),role=role)
    return maps,side(4,'query'),side(8,'document')

def run_demo(output):
    output=Path(output)
    if output.exists():raise FileExistsError('output exists; use a fresh directory')
    maps,q,d=fixture();overlap=build_overlap_blocks(maps)
    qb=prepare_raw_bundle(maps=maps,**q);db=prepare_raw_bundle(maps=maps,**d)
    moments=pair_moments_implicit(qb,db,overlap)
    # Tiny D=8 independent oracle only. Do not scale dense reference to a corpus.
    reference=pair_moments_dense_reference(q,d,maps)
    errors={k:float(np.max(np.abs(moments[k]-reference[k]))) for k in TERMS}
    if not all(np.allclose(moments[k],reference[k],atol=1e-10,rtol=1e-8) for k in TERMS):raise ValueError('dense/implicit mismatch')
    blocked={k:np.empty_like(a) for k,a in moments.items()}
    for qs,ds,block in iter_pair_moments(qb,db,overlap,query_block=2,document_block=3):
        for key in TERMS:blocked[key][qs,ds]=block[key]
    if not all(np.allclose(blocked[k],moments[k],atol=1e-10,rtol=1e-8) for k in TERMS):raise ValueError('block mismatch')
    # The entire d bundle above exists only for the tiny demonstration oracle.
    # Production can read one native document chunk, prepare, consume, discard.
    streamed={k:np.empty_like(a) for k,a in moments.items()};largest=0
    for start in range(0,len(d['alpha_ids']),3):
        rows=slice(start,start+3)
        inputs={k:({m:a[rows] for m,a in v.items()} if isinstance(v,dict) else v if k=='role' else v[rows]) for k,v in d.items()}
        document_chunk=prepare_raw_bundle(maps=maps,**inputs);largest=max(largest,len(document_chunk['row_ids']))
        for qs,ds,block in iter_pair_moments(qb,document_chunk,overlap,query_block=2,document_block=3):
            # Only this tiny fixture collects a full result for verification.
            global_documents=slice(start+ds.start,min(start+ds.stop,len(d['alpha_ids'])))
            for key in TERMS:streamed[key][qs,global_documents]=block[key]
        del document_chunk
    if not all(np.allclose(streamed[k],reference[k],atol=1e-10,rtol=1e-8) for k in TERMS):raise ValueError('document streaming mismatch')
    output.mkdir(parents=True,exist_ok=False);np.savez(output/'raw_moments.npz',**streamed)
    summary=dict(synthetic_demo=True,dimension=8,query_count=4,document_count=8,
        moment_stage='raw_linear_fusion',dense_equivalence_passed=True,block_equivalence_passed=True,
        document_stream_equivalence_passed=True,largest_stream_document_bundle_rows=largest,
        max_abs_errors=errors,uec_score_computed=False,
        note='The synthetic dense oracle is tiny; production uses shared maps/overlap blocks and vector caches.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:summary=run_demo(args.output)
    except (ValueError,FileExistsError) as error:parser.error(str(error))
    print(json.dumps(summary,indent=2))
