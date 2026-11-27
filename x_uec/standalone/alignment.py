"""Portable, sealed real-data input loader for the unchanged v3 alignment core."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from .storage import sha,verify,read_jsonl,code_hashes
from .records import observations,split_rows,canonical_hash,selection_hash

def profile_status(state,profile):
    if profile=='smoke' and state!='numerical_failure':return 'smoke_pass'
    return state

def inspect_inputs(folder):
    folder=Path(folder).resolve();verify(folder)
    p=json.loads((folder/'uec_input_provenance.json').read_text());work=folder.parent
    if p['input_profile']!='standalone_v1' or p['profile'] not in ['full','smoke']:raise ValueError('unsupported standalone input profile')
    if p['paths']!=dict(work='..',input='alignment_fit.npz',data='../data'):raise ValueError('unsupported portable stage layout')
    config=json.loads((work/'preparation_config.json').read_text())
    if sha(work/'preparation_config.json')!=p['preparation_config_sha256'] or config['code_files']!=p['code_files']:raise ValueError('preparation config changed')
    if p['code_files']!=code_hashes():raise ValueError('preparation source differs from this checkout; use the same code revision')
    if config['profile']!=p['profile'] or config['probes']!=32 or config['logical_batch']!=512:raise ValueError('unapproved preparation settings')
    if p['uncertainty_source']!='frozen_teacher_v3_fitted_in_preparation' or not p['mean_variance_same_final_stage']:raise ValueError('unapproved uncertainty contract')
    if sha(folder/'alignment_fit.npz')!=p['input_sha256'] or sha(work/'data/manifest.json')!=p['data_manifest_sha256']:raise ValueError('prepared input changed')
    verify(work/'data')
    for m in ['qwen','harrier','gte']:
        stage=work/'models'/m;verify(stage);verify(stage/'gaussian')
        if sha(stage/'manifest.json')!=p['stages'][m]:raise ValueError('model stage changed')
        summary=json.loads((stage/'summary.json').read_text())
        if summary!=p['models'][m] or summary['data_sha256']!=p['data_manifest_sha256']:raise ValueError('model summary/source mismatch')
        if summary['encoder_hash_before']!=summary['encoder_hash_after']:raise ValueError('encoder weights updated')
        if sha(stage/'gaussian/manifest.json')!=summary['gaussian_manifest_sha256']:raise ValueError('Gaussian changed')
        gaussian=json.loads((stage/'gaussian/summary.json').read_text())
        if not gaussian['coordinate_audit']['passed'] or not gaussian['parameter_gaussian_fitted_here']:raise ValueError('Gaussian verification failed')
    return p

def load_inputs(config):
    from e4_refit import E4Inputs
    from .export import DIMS,validate_arrays
    folder=Path(config['provenance_path']).resolve().parent;p=inspect_inputs(folder);work=folder.parent
    if config['mode']!=p['profile']:raise ValueError('alignment/input profile mismatch')
    rows=read_jsonl(work/'data/observations.jsonl');ids=np.array([r['observation_id'] for r in rows])
    if len(set(ids))!=len(ids):raise ValueError('duplicate observation IDs')
    means={};variances={};mids={};vids={}
    with np.load(folder/'alignment_fit.npz',allow_pickle=False) as z:
        expected={key+'__'+m for m in DIMS for key in ['mean','variance','mean_ids','variance_ids']}
        if set(z.files)!=expected:raise ValueError('input must contain all twelve arrays')
        for m,dim in DIMS.items():
            mean,var,mi,vi=[z[k+'__'+m] for k in ['mean','variance','mean_ids','variance_ids']]
            validate_arrays(mean,var,mi,vi,ids,dim)
            for key,value in [('mean',mean),('variance',var),('mean_ids',mi),('variance_ids',vi)]:
                if not np.array_equal(value,np.load(work/'models'/m/(key+'.npy'),allow_pickle=False,mmap_mode='r')):raise ValueError('independent producer array mismatch')
            means[m]=mean;variances[m]=var;mids[m]=mi;vids[m]=vi
    checks={str(folder/'manifest.json'):sha(folder/'manifest.json'),str(folder/'uec_input_provenance.json'):sha(folder/'uec_input_provenance.json'),
        str(folder/'alignment_fit.npz'):sha(folder/'alignment_fit.npz'),str(work/'preparation_config.json'):sha(work/'preparation_config.json')}
    for stage in [work/'data',*[work/'models'/m for m in DIMS],*[work/'models'/m/'gaussian' for m in DIMS]]:
        record=verify(stage);checks[str(stage/'manifest.json')]=sha(stage/'manifest.json')
        checks.update({str(stage/k):v for k,v in record['files'].items()})
    # Runtime paths are expanded here; exported JSON contains only portable stage links.
    p=dict(p,split_path=str(work/'data/split.json'),observations_path=str(work/'data/observations.jsonl'),work_path=str(work))
    return E4Inputs(means,variances,mids,vids,rows,p,checks)

def audit_and_split(data):
    from e4_refit import AlignmentSplit
    from .assets import REFERENCE
    work=Path(data.provenance['work_path']);verify(REFERENCE)
    pairs=read_jsonl(work/'data/pairs.jsonl');reference=read_jsonl(REFERENCE/'selection_ids.jsonl')
    if data.provenance['profile']=='smoke':reference=[r for source in ['ms_marco','snli'] for r in [r for r in reference if r['source']==source][:32]]
    if len(pairs)!=len(reference):raise ValueError('selected pair count mismatch')
    for row,original in zip(pairs,reference):
        if any(row.get(k)!=v for k,v in original.items()):raise ValueError('published reference selection changed')
        for key,field in [('query_hash','query'),('positive_hash','positive'),('negative_hash','negative')]:
            if key not in row:continue
            if selection_hash(row[field])!=row[key]:raise ValueError('selected text hash changed')
    all_rows=observations(pairs)
    if all_rows!=read_jsonl(work/'data/observations_all.jsonl'):raise ValueError('source observation replay failed')
    evaluation=json.loads((work/'data/evaluation.json').read_text());blocked=set(evaluation['hashes'])
    inventory=json.loads((REFERENCE/'evaluation_hashes.json').read_text())
    historical=set(inventory['hashes'])
    if evaluation['historical_inventory_unique_hashes']!=len(historical):raise ValueError('historical inventory scope changed')
    historical_overlap=sum(r['canonical_text_sha256'] in historical for r in all_rows)
    if historical_overlap!=evaluation['historical_overlap_observations']:raise ValueError('historical overlap audit changed')
    replay=split_rows(all_rows,blocked)
    if replay!=json.loads((work/'data/split.json').read_text()):raise ValueError('independent split replay failed')
    if data.observations!=[all_rows[i] for i in replay['selected_indices']]:raise ValueError('selected rows changed')
    curvature=[r for r in pairs if canonical_hash(r['query']) not in blocked and canonical_hash(r['positive']) not in blocked]
    if curvature!=read_jsonl(work/'data/curvature_pairs.jsonl'):raise ValueError('Gaussian fitting exclusion replay failed')
    lookup={r['source_order']:i for i,r in enumerate(data.observations)}
    def remap(indices):return np.asarray([lookup[i] for i in indices],dtype=int)
    audit=dict(replay,moment_query_indices=remap(replay['moment_query_indices']).tolist(),moment_document_indices=remap(replay['moment_document_indices']).tolist(),
        source_counts=dict(Counter(r['source'] for r in pairs)),curvature_pairs=len(curvature),evaluation_scope=evaluation['scope'],evaluation_unique_hashes=len(blocked),
        historical_external_overlap_observations=historical_overlap,historical_overlap_excluded_by_default=False,
        posterior_unseen=False,author_exact_selection_asserted=False,matches_uec_main_default_selection=True)
    split=AlignmentSplit(remap(replay['fit_indices']),remap(replay['diagnostic_indices']),np.array([],dtype=int),audit)
    return audit,split

def run(config,output):
    # Run in a fresh interpreter with reference_experiment first on sys.path;
    # the exact same caia.py/covariance_ops.py are used by its existing diagnostics.
    from full_e4 import run_full
    return run_full(config,output)
