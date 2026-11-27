"""Prepare with the original v3 fitter; refine failed MC precision at 64/128.

The base preparation code and acceptance rule stay unchanged. Every refinement
replays the SAME seeded probe prefix, preserves the failed attempt, and records
this extra driver with the Gaussian before native outputs are sealed.
"""
import argparse,json,os,shutil,subprocess,sys,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone.storage import sha,verify,write_json,safe_file,code_hashes

def next_probes(count):return {32:64,64:128}.get(count)

def required_attempts(probes):
    if probes not in [64,128]:raise ValueError('supported refinement counts are 64 and 128')
    return [32] if probes==64 else [32,64]

def authorizes_retry(exit_code,outcome,attempt_id,probes):
    return exit_code==75 and outcome.get('status')=='mc_precision_failed' and outcome.get('attempt_id')==attempt_id and outcome.get('probes')==probes

def check_prefix(previous,extended):
    a=np.asarray(previous);b=np.asarray(extended)[:len(a)]
    same=a.shape==b.shape and (np.allclose(a,b,rtol=1e-8,atol=1e-12) if a.dtype.kind=='f' else np.array_equal(a,b))
    if not same:raise ValueError('seeded probe/coordinate prefix changed')

def attach_receipt(folder,receipt):
    folder=Path(folder);record=verify(folder)
    extra=['base_fitter_manifest.json','refinement_driver.py','refinement_receipt.json']
    if all(name in record['files'] for name in extra):return
    write_json(folder/'base_fitter_manifest.json',record)
    shutil.copyfile(__file__,folder/'refinement_driver.py')
    receipt=dict(receipt,driver_sha256=sha(__file__),base_fitter_manifest_sha256=sha(folder/'base_fitter_manifest.json'))
    write_json(folder/'refinement_receipt.json',receipt)
    # Extend the base fitter's seal before the model export/parent seal exists.
    # Original numerical hashes and the fitter's metadata are unchanged.
    for name in extra:record['files'][name]=sha(folder/name)
    write_json(folder/'manifest.json',record);verify(folder)

def precision_failure(folder):
    from standalone.curvature import coordinate_audit
    folder=Path(folder);progress=folder/'progress.json'
    if (folder/'manifest.json').exists() or not progress.exists():return None
    state=json.loads(progress.read_text())
    if state['completed_batches']!=state['total_batches']:return None
    binding=state['binding'];order=np.load(folder/'batch_order.npy',allow_pickle=False);noise=verify(folder/'noise')
    if len(order)<1 or sorted(order.tolist())!=list(range(len(order))) or sha(folder/'batch_order.npy')!=binding['order_sha256']:raise ValueError('invalid bound batch order')
    if sha(folder/'noise/manifest.json')!=binding['noise_manifest_sha256'] or noise['metadata']['probes']!=binding['probes'] or noise['metadata']['seed']!=binding['seed']:raise ValueError('invalid bound noise')
    if sum(noise['metadata']['sizes'])!=len(order) or len(noise['metadata']['sizes'])!=state['total_batches']:raise ValueError('invalid complete batch count')
    file=safe_file(folder,state['checkpoint_file'])
    if sha(file)!=state['checkpoint_sha256']:raise ValueError('failed checkpoint checksum changed')
    with np.load(file,allow_pickle=False) as z:
        samples,exact,diagonal=z['samples'],z['exact'],z['diagonal'];kl=float(z['kl']);n=len(order)
        if not all(np.isfinite(a).all() for a in [samples,exact,diagonal]) or not np.isfinite(kl):raise ValueError('nonfinite coordinate audit or teacher KL')
        if diagonal.ndim!=2 or samples.shape!=(binding['probes'],min(16,diagonal.size)) or exact.shape!=(samples.shape[1],) or int(z['completed'])!=state['total_batches']:raise ValueError('invalid coordinate audit dimensions')
        if any((a<0).any() for a in [samples,exact,diagonal]) or abs(kl)/n>1e-10:raise ValueError('deterministic Gaussian audit failed')
        coords=np.random.default_rng(binding['seed']+1).choice(diagonal.size,len(exact),replace=False)
        if not np.allclose(diagonal.flatten()[coords],samples.mean(0),rtol=1e-10,atol=1e-10):raise ValueError('coordinate diagonal replay mismatch')
        audit=coordinate_audit(samples,exact)
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

def refine(work,model,probes,attempt_id):
    required_attempts(probes)
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
        verify(work/'data')
        stage=work/'models'/model;old=stage/'gaussian';new=stage/f'gaussian_refined_{probes}';archive=stage/'gaussian_rejected_32'
        # Recover a crash in the short directory publication window.
        if not old.exists() and archive.exists() and (new/'manifest.json').exists():new.rename(old)
        if (old/'manifest.json').exists():
            if not all(name in verify(old)['files'] for name in ['refinement_receipt.json','refinement_driver.py','base_fitter_manifest.json']):raise ValueError('refinement receipt not completely sealed')
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
        history=[]
        for count in required_attempts(probes):
            folder=old if count==32 else stage/f'gaussian_refined_{count}'
            failure=precision_failure(folder)
            if failure is None:raise ValueError('required previous MC precision failure is missing')
            state,audit=failure
            if state['binding']['probes']!=count or state['binding']['data_sha256']!=data_sha or state['binding']['encoder_hash']!=m.initial_hash or state['binding']['policy']!=m.policy or state['binding']['asset_hashes']!=m.asset_hashes:raise ValueError('previous refinement source/runtime changed')
            history.append((count,folder,state,audit))
        try:
            variance,_=fit_gaussian(m,pairs,new,probes=probes,batch_size=config['logical_batch'],physical_batch=config['physical_batch'],data_sha256=data_sha)
        except ValueError as error:
            # Only this exact failure from this attempt may authorize more work.
            failure=precision_failure(new) if str(error).startswith('Gaussian coordinate/teacher-center audit failed:') else None
            if failure is None:raise
            state,audit=failure
            write_json(stage/f'refinement_outcome_{attempt_id}.json',dict(status='mc_precision_failed',attempt_id=attempt_id,probes=probes,
                checkpoint_sha256=state['checkpoint_sha256'],audit=audit,driver_sha256=sha(__file__)))
            raise SystemExit(75)
        for count,folder,state,audit in history:
            for p in sorted((folder/'noise').glob('batch_*.npy')):check_prefix(np.load(p,mmap_mode='r'),np.load(new/'noise'/p.name,mmap_mode='r'))
            with np.load(safe_file(folder,state['checkpoint_file']),allow_pickle=False) as before,np.load(new/'coordinate_audit.npz',allow_pickle=False) as after:
                check_prefix(before['samples'],after['per_probe'])
                if not np.allclose(before['exact'],after['exact'],rtol=1e-8,atol=1e-12):raise ValueError('exact coordinate target changed')
        attach_receipt(new,dict(reason='initial 32-probe coordinate MC precision check failed',initial_probes=32,actual_probes=probes,
            initial_audit=history[0][3],previous_attempts=[dict(probes=n,audit=a,checkpoint_sha256=s['checkpoint_sha256'],noise_manifest_sha256=sha(f/'noise/manifest.json')) for n,f,s,a in history],
            acceptance_rule_unchanged=True,seed_unchanged=True,noise_and_coordinate_prefix_verified=True,
            initial_checkpoint_sha256=previous['checkpoint_sha256'],initial_noise_manifest_sha256=sha(old/'noise/manifest.json'),
            preparation_config_sha256=sha(work/'preparation_config.json')))
        if archive.exists():raise FileExistsError('unexpected previous attempt archive')
        old.rename(archive);new.rename(old)
        export_model(m,variance,work,config['physical_batch'])

def parse_cli(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--profile',choices=['full','smoke'],default='full');p.add_argument('--cache-dir',type=Path)
    p.add_argument('--offline',action='store_true');p.add_argument('--physical-batch',type=int,default=8)
    p.add_argument('--evaluation-texts',type=Path,action='append',default=[])
    p.add_argument('--qwen-attention',choices=['sdpa','flash_attention_2'],default='sdpa')
    a=p.parse_args(argv);a.work_dir=a.work_dir.resolve()
    if a.physical_batch<1:p.error('--physical-batch must be positive')
    return a

def effective_config(a):
    from standalone.models import package_versions
    cache=(a.cache_dir or a.work_dir/'cache/huggingface/hub').resolve()
    if any(str(p).startswith('/home/') for p in [a.work_dir,cache]):raise ValueError('use a data disk outside /home')
    return dict(profile=a.profile,probes=32,logical_batch=512,physical_batch=a.physical_batch,device='cuda:0',gpu=a.gpu,versions=package_versions(),
        cache_dir=str(cache),offline=a.offline,qwen_attention=a.qwen_attention,code_files=code_hashes(),
        additional_evaluation=[dict(path=str(p.resolve()),sha256=sha(p)) for p in a.evaluation_texts])

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=='_refine':
        if len(argv)!=5:raise ValueError('private refinement invocation requires work/model/count/attempt ID')
        refine(Path(argv[1]),argv[2],int(argv[3]),argv[4]);return
    a=parse_cli(argv);counts={};file=a.work_dir/'preparation_config.json'
    expected=effective_config(a)
    if file.exists() and json.loads(file.read_text())!=expected:raise ValueError('preparation configuration/code changed; use the original invocation or a new work directory')
    while True:
        ready=ready_refinement(a.work_dir) if a.work_dir else None
        if ready:
            model,count=ready
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'_refine',str(a.work_dir),model,str(count),uuid.uuid4().hex],check=True)
        failed=failed_precision(a.work_dir) if a.work_dir and (a.work_dir/'preparation_config.json').exists() else None
        if failed:
            model,state,audit=failed;count=next_probes(counts.get(model,32))
            if count is None:raise RuntimeError('coordinate precision failed through 128 probes; no approved Gaussian')
            counts[model]=count
            print(f'REFINE {model}: unchanged seed/rule; {count} probes; preserve original failed attempt',flush=True)
            token=uuid.uuid4().hex
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_refine',str(a.work_dir),model,str(count),token])
            if result.returncode:
                outcome_file=a.work_dir/'models'/model/f'refinement_outcome_{token}.json'
                outcome=json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
                if not authorizes_retry(result.returncode,outcome,token,count):raise SystemExit(result.returncode)
                continue
        result=subprocess.run([sys.executable,str(ROOT/'prepare_alignment.py'),*argv])
        if result.returncode==0:return
        if a.work_dir is None or failed_precision(a.work_dir) is None:raise SystemExit(result.returncode)

if __name__=='__main__':main()
