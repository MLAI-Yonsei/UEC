"""Bind existing sealed AG News caches without inference or renormalization.

Legacy variance NPY files have separately sealed per-model rows.json sidecars.
Check every mean row list and variance sidecar before assigning computational
roles. Full native covariance is not read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_covariance_validation import sha,write_json,save_role_npz,ORDER

def verify_binding_rows(anchors,mean_rows,metadata_rows):
    expected=[(r['row_id'],r['exact_hash']) for r in anchors]
    for rows in (mean_rows,metadata_rows):
        if [(r['row_id'],r['exact_hash']) for r in rows]!=expected:
            raise ValueError('original per-model row ID/hash/order differs')
    return np.asarray([r['row_id']+':encoding_query' for r in anchors],dtype=str)

def verify_variance_rows(anchors,sidecar,model,shape,weight_hash):
    expected=[(r['row_id'],r['exact_hash']) for r in anchors]
    if (sidecar.get('model')!=model or sidecar.get('source')!='teacher'
            or sidecar.get('shape')!=list(shape) or sidecar.get('original_weight_hash')!=weight_hash
            or len(sidecar.get('ids',[]))!=len(anchors) or len(sidecar.get('hashes',[]))!=len(anchors)
            or list(zip(sidecar['ids'],sidecar['hashes']))!=expected):
        raise ValueError('variance sidecar ID/hash/model/source/shape/checkpoint mismatch')
    return np.asarray([s+':encoding_query' for s in sidecar['ids']],dtype=str)

def export(source,output):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    consumed={}
    seal_path=source/'manifest.json';seal=json.loads(seal_path.read_text());consumed[str(seal_path)]=sha(seal_path)
    def sealed(relative):
        path=source/relative;actual=sha(path)
        if actual!=seal['files'][relative]:raise ValueError('sealed source changed '+relative)
        consumed[str(path)]=actual;return path
    anchor_path=sealed('anchor_rows.jsonl');anchors=[json.loads(s) for s in anchor_path.read_text().splitlines()]
    bindings=json.loads(sealed('input_hashes.json').read_text())
    producer_code={}
    for rel in ['source/preflight/0_workflow.py','source_final/caia_svd_pilot_v3/uncertainty.py',
                'source_final/caia_svd_pilot_v2/uncertainty.py']:
        path=sealed(rel);producer_code[str(path)]=consumed[str(path)]
    if len(anchors)!=10000 or any(r['split']!=('fit' if i<8000 else 'diagnostic') for i,r in enumerate(anchors)):
        raise ValueError('source fit/diagnostic split changed')
    # Use distinct text observations, predetermined before looking at E2 values.
    fit_hashes={r['canonical_hash'] for r in anchors if r['split']=='fit'}
    eligible=[];seen=set()
    for i,r in enumerate(anchors):
        if r['split']!='diagnostic' or r['canonical_hash'] in fit_hashes or r['canonical_hash'] in seen:continue
        seen.add(r['canonical_hash']);eligible.append(i)
    # The same original role is retained in both mathematical slots, disjoint IDs.
    selected=np.random.default_rng(42).permutation(eligible)
    if len(selected)<1032:raise ValueError('insufficient held-out distinct observations for 8+1024')
    # Use all eligible rows in disjoint query/document slots for E2; E3 uses
    # just the first 8 query rows and at most 1024 documents.
    qi,di=selected[:-1024],selected[-1024:]
    means={};variances={};mean_ids={};variance_ids={};models={};quality={};id_evidence={}
    for m in ORDER:
        ref=bindings['anchor_refs'][m];configs=ref['encoder_config'];models[m]=configs
        rows_path=Path(ref['rows_path']);meta_path=Path(ref['meta_path'])
        for p in (rows_path,meta_path):
            actual=sha(p)
            if actual!=ref['files'][str(p)]:raise ValueError('original row metadata changed')
            consumed[str(p)]=actual
        original_rows=json.loads(rows_path.read_text());meta=json.loads(meta_path.read_text())
        mean_ids[m]=verify_binding_rows(anchors,original_rows,meta['contract']['rows'])
        mean_path=sealed('means/'+m+'.npy');variance_path=sealed('covariance/'+m+'/teacher/diagonal.npy')
        bulk_path=sealed('covariance/'+m+'/bulk_order.npy');bulk=np.load(bulk_path,allow_pickle=False)
        if not np.array_equal(np.sort(bulk),np.arange(len(anchors))):raise ValueError('incomplete variance producer row permutation')
        means[m]=np.load(mean_path,allow_pickle=False,mmap_mode='r');variances[m]=np.load(variance_path,allow_pickle=False,mmap_mode='r')
        if means[m].shape!=(10000,configs['native_dim']) or variances[m].shape!=means[m].shape:raise ValueError('native dimensions changed')
        summary=json.loads(sealed('covariance/'+m+'/summary.json').read_text());quality[m]=summary.get('quality')
        if summary['original_parameter_hash_before']!=summary['original_parameter_hash_after']:
            raise ValueError('original checkpoint changed during covariance production')
        variance_rows=sealed('covariance/'+m+'/teacher/rows.json');sidecar=json.loads(variance_rows.read_text())
        variance_ids[m]=verify_variance_rows(anchors,sidecar,m,variances[m].shape,summary['original_parameter_hash_before'])
        gaussian=sealed('gaussian/'+m+'/summary.json')
        id_evidence[m]=dict(mean_rows_path=str(rows_path),mean_metadata_path=str(meta_path),variance_rows_path=str(variance_rows),
            variance_rows_sha256=consumed[str(variance_rows)],variance_binding='independently checked per-model sealed rows.json IDs/hashes/model/source/shape/checkpoint; NPY positional rows bound to its sidecar',
            bulk_order_sha256=consumed[str(bulk_path)],posterior_config_path=str(gaussian),posterior_config_sha256=consumed[str(gaussian)])
    maps={}
    for arm,filename in [('GPA','A.npz'),('Teacher_CAIA','C.npz')]:maps[arm]=str(sealed('maps/'+filename))
    payloads=[]
    for indices,role,filename in [(qi,'query','q.npz'),(di,'document','d.npz')]:
        side=dict(native_means={m:np.asarray(means[m][indices]) for m in ORDER},native_vars={m:np.asarray(variances[m][indices]) for m in ORDER},
            mean_ids_by_model={m:mean_ids[m][indices] for m in ORDER},variance_ids_by_model={m:variance_ids[m][indices] for m in ORDER},
            alpha=np.full((len(indices),3),1/3),alpha_ids=mean_ids['qwen'][indices].copy(),role=role)
        # The loader subsequently compares all six ID arrays and alpha IDs.
        save_role_npz(output/filename,side);consumed[str(output/filename)]=sha(output/filename);payloads.append(side)
    selection=dict(seed=42,original_role='query-style model-specific STS policy',query_indices=qi.tolist(),document_indices=di.tolist(),
        query_ids=payloads[0]['alpha_ids'].tolist(),document_ids=payloads[1]['alpha_ids'].tolist(),eligible_diagnostic_distinct=len(eligible),
        fit_canonical_hash_overlap=0,query_document_canonical_hash_overlap=0)
    write_json(output/'selection.json',selection);consumed[str(output/'selection.json')]=sha(output/'selection.json')
    manifest=dict(query_npz=str(output/'q.npz'),document_npz=str(output/'d.npz'),maps=maps,models=models,
        output_stage='l2_normalized',covariance_representation='native_output_diagonal_surrogate',
        uncertainty_source='frozen teacher-anchored GGN/Jacobian local output sensitivity surrogate; not verified as original UEC LA',
        original_encoding_role='query-style for both mathematical slots; original prompt/pooling retained',
        source_run=str(source),alignment_diagnostic_scope='held-out for alignment; not independent of uncertainty-model design or previous pilot inspection',
        row_id_evidence=id_evidence,producer_code_files=producer_code,source_quality_flags=quality,selection=selection,
        coefficient_function='fixed numeric diagnostics, not actual UEC weights',score_function=None,uec_integration_ready=False,
        native_full_covariance_consumed=False,normalization_applied_by_exporter=False,consumed_files=consumed)
    write_json(output/'manifest.json',manifest);return output/'manifest.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(export(args.source,args.output))
