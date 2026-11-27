"""New fitting-text outputs; reuse immutable original v3 model/propagation code."""
import argparse
import os
import json
from pathlib import Path
import sys
import time
ROOT=Path(os.environ['XUEC_EXPERIMENT_WORKSPACE']).resolve()
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.insert(1,str(ROOT))
import e4_data
from e4_data import prepare_data,assemble_cache,DIMS,verify_parent_sources
from run_covariance_validation import sha,write_json
sys.path.append(str(Path(__file__).resolve().parent.parent))
import preparation_checks
from preparation_checks import verify_model_policy

def worker(name,folder):
    import numpy as np
    import torch
    from caia_svd_pilot_v3.uncertainty import capture,native_forward,diagonal
    folder=Path(folder).resolve();out=folder/name;out.mkdir(exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    parent=ROOT/'runs/caia_svd_frozen_triple_20261124/v3';seal=json.loads((parent/'manifest.json').read_text())
    consumed={str(folder/'observations.jsonl'):sha(folder/'observations.jsonl'),str(parent/'manifest.json'):sha(parent/'manifest.json'),str(Path(__file__)):sha(__file__)}
    consumed.update(verify_parent_sources(parent,ROOT))
    consumed[str(Path(preparation_checks.__file__).resolve())]=sha(preparation_checks.__file__)
    consumed[str(Path(e4_data.__file__).resolve())]=sha(e4_data.__file__)
    def bound(relative):
        path=parent/relative;actual=sha(path)
        if actual!=seal['files'][relative]:raise ValueError('parent Gaussian/source seal changed')
        consumed[str(path)]=actual;return path
    summary=json.loads(bound(f'gaussian/{name}/summary.json').read_text())
    original_cov=json.loads(bound(f'covariance/{name}/summary.json').read_text())
    weight=np.load(bound(f'gaussian/{name}/mean_weight.npy'),allow_pickle=False)
    variance=np.load(bound(f'gaussian/{name}/variance_teacher.npy'),allow_pickle=False)
    for rel in ['caia_svd_pilot_v3/uncertainty.py','caia_svd_pilot_v2/sensitivity_math.py','caia_pair_pilot/qwen_math.py','caia_pair_pilot/qwen_adapter.py','caia_svd_pilot_v3/harrier_adapter.py','gte_la/model.py']:
        consumed[str(ROOT/rel)]=sha(ROOT/rel)
    if name=='qwen':
        author=ROOT/'runs/qwen_reproduction_20261110/author_protocol/reference/qwen_author_model.py'
        if sha(author)!=summary['policy']['source_sha256']:raise ValueError('Qwen author source differs from original policy')
        consumed[str(author)]=sha(author)
        from caia_pair_pilot.qwen_adapter import QwenPath
        model=QwenPath()
    elif name=='harrier':
        from caia_svd_pilot_v3.harrier_adapter import HarrierPath
        model=HarrierPath()
    else:
        from gte_la.model import GTEPath
        model=GTEPath()
    runtime_policy=verify_model_policy(name,model,summary['policy'])
    from importlib.metadata import version
    runtime_versions={package:version(package) for package in summary['policy']['versions']}
    if runtime_versions!=summary['policy']['versions']:raise ValueError('runtime versions differ from original producer policy')
    if not np.array_equal(model.weight.cpu().numpy(),weight):raise ValueError('frozen posterior center mismatch')
    if model.initial_hash!=original_cov['original_parameter_hash_before']:raise ValueError('original encoder checksum mismatch')
    parameter_variance=torch.as_tensor(variance,device=model.device)
    with (folder/'observations.jsonl').open() as stream:rows=[json.loads(s) for s in stream]
    n=len(rows);dim=DIMS[name]
    means=np.lib.format.open_memmap(out/'mean.npy',mode='w+',shape=(n,dim),dtype=np.float64)
    variances=np.lib.format.open_memmap(out/'variance.npy',mode='w+',shape=(n,dim),dtype=np.float64)
    ids=np.array([r['observation_id'] for r in rows],dtype=str);mean_ids=np.full(ids.shape,'',dtype=ids.dtype);variance_ids=mean_ids.copy()
    load_seconds=time.perf_counter()-started;inference=propagation=0.;max_error=0.
    try:
        for begin in range(0,n,8):
            indices=list(range(begin,min(n,begin+8)));t=time.perf_counter()
            features,actual=capture(name,model,[rows[i]['text'] for i in indices]);torch.cuda.synchronize();inference+=time.perf_counter()-t
            for j,(index,features_one) in enumerate(zip(indices,features)):
                t=time.perf_counter()
                with torch.no_grad():
                    mu=native_forward(name,model,features_one)[1];var=diagonal(name,model,features_one,parameter_variance)
                if not torch.isfinite(mu).all() or not torch.isfinite(var).all() or float(var.min())<0:raise ValueError('invalid output moment; no clipping')
                err=float((mu-actual[j].double()).abs().max());max_error=max(max_error,err)
                if err>1e-3:raise ValueError('native frozen-path center differs from original encoder')
                means[index]=mu.cpu().numpy();mean_ids[index]=rows[index]['observation_id']
                variances[index]=var.cpu().numpy();variance_ids[index]=rows[index]['observation_id']
                torch.cuda.synchronize();propagation+=time.perf_counter()-t
            means.flush();variances.flush()
            write_json(out/'progress.json',dict(status='running',completed_rows=begin+len(indices),total_rows=n,seconds=time.perf_counter()-started))
            if begin%128==0:print('OUTPUT_CACHE',name,begin+len(indices),n,'seconds',time.perf_counter()-started,flush=True)
        verify_model_policy(name,model,summary['policy'])
        if {package:version(package) for package in runtime_versions}!=runtime_versions:raise ValueError('runtime versions changed during generation')
        final=model.parameter_hash()
        if final!=model.initial_hash:raise ValueError('encoder changed')
        np.save(out/'mean_ids.npy',mean_ids);np.save(out/'variance_ids.npy',variance_ids)
        outputs={str(out/(k+'.npy')):sha(out/(k+'.npy')) for k in ['mean','variance','mean_ids','variance_ids']}
        write_json(out/'summary.json',dict(status='completed',model=name,rows=n,dimension=dim,policy=summary['policy'],
            uncertainty_source='frozen_teacher_v3_reused',posterior_refit_performed=False,encoder_weights_updated=False,
            original_parameter_hash_before=model.initial_hash,original_parameter_hash_after=final,
            center_vs_original_max_abs=max_error,inherited_quality_flags=original_cov.get('quality'),
            model_load_seconds=load_seconds,inference_seconds=inference,propagation_seconds=propagation,
            runtime_versions=runtime_versions,runtime_policy=runtime_policy,policy_origin='inherited_original_v3; source and encoder hashes checked; runtime versions verified',
            total_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            output_checksums=outputs,consumed_files=consumed))
    except Exception as error:
        write_json(out/'summary.json',dict(status='failed',model=name,error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['data','model','assemble']);p.add_argument('--output',required=True);p.add_argument('--model',choices=['qwen','harrier','gte']);a=p.parse_args()
    if a.stage=='data':print(prepare_data(ROOT,a.output,mode='full'))
    elif a.stage=='assemble':print(assemble_cache(a.output))
    else:worker(a.model,a.output)
