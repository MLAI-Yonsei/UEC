"""Prepare with the original v3 fitter; refine failed MC precision at 64/128.

The base preparation code and acceptance rule stay unchanged. Every refinement
replays the SAME seeded probe prefix, preserves the failed attempt, and records
this extra driver with the Gaussian before native outputs are sealed.
"""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone.storage import sha,verify,write_json,safe_file,code_hashes

def next_probes(count):return {32:64,64:128}.get(count)

def check_prefix(previous,extended):
    a=np.asarray(previous);b=np.asarray(extended)[:len(a)]
    same=a.shape==b.shape and (np.allclose(a,b,rtol=1e-8,atol=1e-12) if a.dtype.kind=='f' else np.array_equal(a,b))
    if not same:raise ValueError('seeded probe/coordinate prefix changed')

def attach_receipt(folder,receipt):
    folder=Path(folder);record=verify(folder)
    if (folder/'refinement_receipt.json').exists():return
    write_json(folder/'base_fitter_manifest.json',record)
    shutil.copyfile(__file__,folder/'refinement_driver.py')
    receipt=dict(receipt,driver_sha256=sha(__file__),base_fitter_manifest_sha256=sha(folder/'base_fitter_manifest.json'))
    write_json(folder/'refinement_receipt.json',receipt)
    # Extend the base fitter's seal before the model export/parent seal exists.
    # Original numerical hashes and the fitter's metadata are unchanged.
    for name in ['base_fitter_manifest.json','refinement_driver.py','refinement_receipt.json']:record['files'][name]=sha(folder/name)
    write_json(folder/'manifest.json',record);verify(folder)

def precision_failure(folder):
    from standalone.curvature import coordinate_audit
    folder=Path(folder);progress=folder/'progress.json'
    if (folder/'manifest.json').exists() or not progress.exists():return None
    state=json.loads(progress.read_text())
    if state['completed_batches']!=state['total_batches']:return None
    file=safe_file(folder,state['checkpoint_file'])
    if sha(file)!=state['checkpoint_sha256']:raise ValueError('failed checkpoint checksum changed')
    with np.load(file,allow_pickle=False) as z:
        audit=coordinate_audit(z['samples'],z['exact']);n=len(np.load(folder/'batch_order.npy',allow_pickle=False))
        if not np.isfinite(z['diagonal']).all() or (z['diagonal']<0).any() or abs(float(z['kl']))/n>1e-10:return None
    return (state,audit) if not audit['passed'] else None

def failed_precision(work):
    for model in ['qwen','harrier','gte']:
        failed=precision_failure(Path(work)/'models'/model/'gaussian')
        if failed:return model,*failed
    return None

def ready_refinement(work):
    for model in ['qwen','harrier','gte']:
        stage=Path(work)/'models'/model
        if (stage/'manifest.json').exists():continue
        old=stage/'gaussian'
        if (old/'refinement_receipt.json').exists():
            return model,json.loads((old/'summary.json').read_text())['probes']
        if not old.exists() and (stage/'gaussian_rejected_32').exists():
            for count in [64,128]:
                if (stage/f'gaussian_refined_{count}/refinement_receipt.json').exists():return model,count
    return None

def refine(work,model,probes):
    from standalone.pipeline import environment,preparation_lock
    from prepare_alignment import check_gpu
    work=Path(work).resolve();config=json.loads((work/'preparation_config.json').read_text())
    if config['code_files']!=code_hashes():raise ValueError('base preparation source changed')
    os.environ.update(environment(work,config['cache_dir'],config['offline']))
    os.environ['CUDA_VISIBLE_DEVICES']=check_gpu(config['gpu'],settle_utilization=True)
    from standalone.models import ModelPath
    from standalone.curvature import fit_gaussian
    from standalone.export import export_model
    from standalone.storage import read_jsonl
    with preparation_lock(work):
        stage=work/'models'/model;old=stage/'gaussian';new=stage/f'gaussian_refined_{probes}';archive=stage/'gaussian_rejected_32'
        # Recover a crash in the short directory publication window.
        if not old.exists() and archive.exists() and (new/'manifest.json').exists():new.rename(old)
        if (old/'manifest.json').exists():
            m=ModelPath(model,config['cache_dir'],config['device'],config['offline'],config['qwen_attention'])
            binding=verify(old)['metadata']
            if binding['policy']!=m.policy or binding['asset_hashes']!=m.asset_hashes or binding['encoder_hash']!=m.initial_hash:raise ValueError('refined Gaussian source/runtime changed')
            import torch
            variance=torch.as_tensor(np.load(old/'variance_teacher.npy'),device=m.device)
            export_model(m,variance,work,config['physical_batch']);return
        previous=json.loads((old/'progress.json').read_text());binding=previous['binding'];verify(old/'noise')
        m=ModelPath(model,config['cache_dir'],config['device'],config['offline'],config['qwen_attention'])
        if binding['encoder_hash']!=m.initial_hash or binding['policy']!=m.policy or binding['asset_hashes']!=m.asset_hashes:raise ValueError('original attempt source/runtime changed')
        pairs=read_jsonl(work/'data/curvature_pairs.jsonl');data_sha=sha(work/'data/manifest.json')
        if binding['data_sha256']!=data_sha or binding['probes']!=32:raise ValueError('unapproved refinement source')
        variance,_=fit_gaussian(m,pairs,new,probes=probes,batch_size=config['logical_batch'],physical_batch=config['physical_batch'],data_sha256=data_sha)
        for p in sorted((old/'noise').glob('batch_*.npy')):check_prefix(np.load(p,mmap_mode='r'),np.load(new/'noise'/p.name,mmap_mode='r'))
        with np.load(safe_file(old,previous['checkpoint_file']),allow_pickle=False) as before,np.load(new/'coordinate_audit.npz',allow_pickle=False) as after:
            check_prefix(before['samples'],after['per_probe'])
            if not np.allclose(before['exact'],after['exact'],rtol=1e-8,atol=1e-12):raise ValueError('exact coordinate target changed')
            from standalone.curvature import coordinate_audit
            original_audit=coordinate_audit(before['samples'],before['exact'])
        attach_receipt(new,dict(reason='initial 32-probe coordinate MC precision check failed',initial_probes=32,actual_probes=probes,
            initial_audit=original_audit,acceptance_rule_unchanged=True,seed_unchanged=True,noise_and_coordinate_prefix_verified=True,
            initial_checkpoint_sha256=previous['checkpoint_sha256'],initial_noise_manifest_sha256=sha(old/'noise/manifest.json'),
            preparation_config_sha256=sha(work/'preparation_config.json')))
        if archive.exists():raise FileExistsError('unexpected previous attempt archive')
        old.rename(archive);new.rename(old)
        export_model(m,variance,work,config['physical_batch'])

def main():
    if len(sys.argv)>1 and sys.argv[1]=='_refine':
        refine(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]));return
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--work-dir',type=Path)
    a,_=p.parse_known_args();counts={}
    while True:
        ready=ready_refinement(a.work_dir) if a.work_dir else None
        if ready:
            model,count=ready
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'_refine',str(a.work_dir.resolve()),model,str(count)],check=True)
        failed=failed_precision(a.work_dir) if a.work_dir and (a.work_dir/'preparation_config.json').exists() else None
        if failed:
            model,state,audit=failed;count=next_probes(counts.get(model,32))
            if count is None:raise RuntimeError('coordinate precision failed through 128 probes; no approved Gaussian')
            counts[model]=count
            print(f'REFINE {model}: unchanged seed/rule; {count} probes; preserve original failed attempt',flush=True)
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_refine',str(a.work_dir.resolve()),model,str(count)])
            if result.returncode:
                if precision_failure(a.work_dir/'models'/model/f'gaussian_refined_{count}') is None:raise SystemExit(result.returncode)
                continue
        result=subprocess.run([sys.executable,str(ROOT/'prepare_alignment.py'),*sys.argv[1:]])
        if result.returncode==0:return
        if a.work_dir is None or failed_precision(a.work_dir) is None:raise SystemExit(result.returncode)

if __name__=='__main__':main()
