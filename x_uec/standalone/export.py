"""Export real encoder means/native diagonals and their independent row IDs."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .storage import sha,write_json,read_jsonl,seal,verify

DIMS={'qwen':1024,'harrier':640,'gte':768}

def validate_arrays(mean,variance,mean_ids,variance_ids,expected_ids,dimension):
    shape=(len(expected_ids),dimension)
    if mean.shape!=shape or variance.shape!=shape:raise ValueError('mean/variance shape mismatch')
    if not np.array_equal(mean_ids,expected_ids) or not np.array_equal(variance_ids,expected_ids):raise ValueError('independent mean/variance IDs mismatch')
    if not np.isfinite(mean).all():raise ValueError('nonfinite mean')
    if not np.isfinite(variance).all() or (variance<0).any():raise ValueError('invalid native variance')
    if np.max(np.abs(np.linalg.norm(mean,axis=1)-1))>1e-5:raise ValueError('native mean norm contract')

@torch.no_grad()
def export_model(model,variance,work,physical_batch=8):
    work=Path(work);folder=work/'models'/model.name;folder.mkdir(parents=True,exist_ok=True)
    data_manifest=verify(work/'data');data_sha=sha(work/'data/manifest.json')
    if (folder/'manifest.json').exists():
        record=verify(folder)
        if record['metadata']['data_sha256']!=data_sha or record['metadata']['encoder_hash']!=model.initial_hash:raise ValueError('completed model source changed')
        return record
    rows=read_jsonl(work/'data/observations.jsonl');ids=np.array([r['observation_id'] for r in rows]);n=len(rows);dim=DIMS[model.name]
    mean=np.lib.format.open_memmap(folder/'mean.npy',mode='w+',dtype='float64',shape=(n,dim))
    native=np.lib.format.open_memmap(folder/'variance.npy',mode='w+',dtype='float64',shape=(n,dim))
    worst=0.;start=time.perf_counter()
    for begin in range(0,n,physical_batch):
        batch=rows[begin:begin+physical_batch];features,actual=model.capture([r['text'] for r in batch])
        for j,feature in enumerate(features):
            _,mu=model.forward(feature);v=model.diagonal(feature,variance)
            mean[begin+j]=mu.cpu().numpy();native[begin+j]=v.cpu().numpy()
            error=float((mu.cpu()-actual[j]).abs().max());worst=max(worst,error)
            if error>(1e-3 if model.name=='qwen' else 1e-5):raise ValueError('final-path/encoder center mismatch '+model.name)
        mean.flush();native.flush()
        write_json(folder/'export_progress.json',dict(rows=min(begin+physical_batch,n),total=n,seconds=time.perf_counter()-start))
        if begin%128==0:print('EXPORT',model.name,begin,n,flush=True)
    # Write each sidecar independently, then validate both against selected source rows.
    np.save(folder/'mean_ids.npy',ids);np.save(folder/'variance_ids.npy',np.array([r['observation_id'] for r in rows]))
    validate_arrays(mean,native,np.load(folder/'mean_ids.npy'),np.load(folder/'variance_ids.npy'),ids,dim)
    model.verify_unchanged()
    summary=dict(model=model.name,policy=model.policy,encoder_hash_before=model.initial_hash,encoder_hash_after=model.parameter_hash(),
        asset_hashes=model.asset_hashes,mean_variance_same_final_stage=True,native_output_dim=dim,observations=n,
        max_encoder_center_abs_error=worst,seconds=time.perf_counter()-start,data_sha256=data_sha,
        gaussian_manifest_sha256=sha(folder/'gaussian/manifest.json'),gaussian_fitted_in_preparation=True,
        uncertainty_source='frozen_teacher_v3_fitted_in_preparation',profile=data_manifest['metadata']['profile'])
    write_json(folder/'summary.json',summary)
    return seal(folder,['mean.npy','variance.npy','mean_ids.npy','variance_ids.npy','summary.json','gaussian/manifest.json'],
        dict(data_sha256=data_sha,encoder_hash=model.initial_hash))

def assemble(work):
    work=Path(work);folder=work/'inputs';folder.mkdir(exist_ok=True);verify(work/'data')
    if (folder/'manifest.json').exists():return verify(folder)
    rows=read_jsonl(work/'data/observations.jsonl');ids=np.array([r['observation_id'] for r in rows]);arrays={};models={}
    for m,dim in DIMS.items():
        stage=work/'models'/m;verify(stage);verify(stage/'gaussian')
        values={k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['mean','variance','mean_ids','variance_ids']}
        validate_arrays(values['mean'],values['variance'],values['mean_ids'],values['variance_ids'],ids,dim)
        arrays.update({k+'__'+m:v for k,v in values.items()});models[m]=json.loads((stage/'summary.json').read_text())
    np.savez(folder/'alignment_fit.npz',**arrays)
    config=json.loads((work/'preparation_config.json').read_text())
    provenance=dict(input_profile='standalone_v1',profile=config['profile'],delivery_version='v3',models=models,
        uncertainty_source='frozen_teacher_v3_fitted_in_preparation',mean_variance_same_final_stage=True,
        preparation_config_sha256=sha(work/'preparation_config.json'),code_files=config['code_files'],
        input_sha256=sha(folder/'alignment_fit.npz'),data_manifest_sha256=sha(work/'data/manifest.json'),
        stages={m:sha(work/'models'/m/'manifest.json') for m in DIMS},
        paths=dict(work='..',input='alignment_fit.npz',data='../data'),
        cache_scope='alignment_full' if config['profile']=='full' else 'smoke_only',
        author_exact_selection_asserted=False,matches_uec_main_default_selection=True,posterior_unseen=False)
    write_json(folder/'uec_input_provenance.json',provenance)
    return seal(folder,['alignment_fit.npz','uec_input_provenance.json'],dict(profile=config['profile']))
