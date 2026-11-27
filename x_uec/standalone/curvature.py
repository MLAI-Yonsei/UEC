"""Frozen-teacher SUM categorical GGN; both endpoints share one weight noise."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .storage import write_json,sha,seal,verify

def projected_noise(probabilities,noise):
    a=probabilities.detach().sqrt()*noise
    return a-probabilities.detach()*a.sum(-1,keepdim=True)

def score_coordinate_jacobian(q,p,dq,dp):
    return torch.einsum('idk,jd->ijk',dq,p)+torch.einsum('id,jdk->ijk',q,dp)

def coordinate_audit(samples,exact):
    samples=torch.as_tensor(samples,dtype=torch.float64);exact=torch.as_tensor(exact,dtype=torch.float64)
    observed=samples.mean(0);se=samples.std(0,unbiased=True)/len(samples)**.5;nonzero=se>0
    z=torch.zeros_like(exact);z[nonzero]=(observed[nonzero]-exact[nonzero])/se[nonzero]
    # Exact accepted v3 check_coordinate_ggn rule, including its absolute floor.
    # Raw categorical scores can give essentially zero curvature (e.g. 1e-33);
    # a z-only test would mistake roundoff there for a failed derivative.
    tolerance=3.29*se+1e-8*torch.maximum(torch.ones_like(exact),exact.abs())
    passed=bool(torch.isfinite(samples).all() and torch.isfinite(exact).all() and ((observed-exact).abs()<=tolerance).all())
    return dict(passed=passed,max_abs_z=float(z.abs().max()),zero_se_count=int((~nonzero).sum()),
        rule='abs(estimate-exact) <= 3.29*SE + 1e-8*max(1,abs(exact))',
        max_abs_error=float((observed-exact).abs().max()),max_tolerance=float(tolerance.max()))

@torch.no_grad()
def batch_ggn(model,features,coordinates,noises):
    n=len(features)//2
    if len(features)!=2*n or not n:raise ValueError('query/positive feature count')
    means=torch.stack([model.forward(f)[0] for f in features]);q,p=means[:n],means[n:]
    scores=q@p.T;prob=scores.softmax(-1).detach()
    jq=model.coordinate_jacobian(features[:n],coordinates);jp=model.coordinate_jacobian(features[n:],coordinates)
    js=score_coordinate_jacobian(q,p,jq,jp);centered=js-(prob[:,:,None]*js).sum(1,keepdim=True)
    exact=(prob[:,:,None]*centered.square()).sum((0,1))
    total=torch.zeros_like(model.weight);samples=[]
    for noise in noises:
        noise=noise.detach().clone().to(device=model.device,dtype=model.weight.dtype) if torch.is_tensor(noise) else torch.as_tensor(np.array(noise,copy=True),device=model.device,dtype=model.weight.dtype)
        r=projected_noise(prob,noise)
        # Same parameter affects query AND passage: add their VJPs before square.
        gradient=model.raw_vjp(features[:n],r@p)+model.raw_vjp(features[n:],r.T@q)
        squared=gradient.square();total+=squared;samples.append(squared.flatten()[coordinates].cpu())
    if len(samples)<2:raise ValueError('at least two curvature probes required')
    return dict(diagonal=total/len(samples),coordinate_samples=torch.stack(samples),exact_coordinates=exact,
        teacher_kl_sum=float(torch.nn.functional.kl_div(scores.log_softmax(-1),prob,reduction='sum')),probes=len(samples))

def generate_noise(folder,sizes,probes,seed,device):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        record=verify(folder)
        if record['metadata']!=dict(sizes=sizes,probes=probes,seed=seed,device_type=torch.device(device).type):raise ValueError('noise configuration changed')
        return [folder/f'batch_{i}.npy' for i in range(len(sizes))]
    arrays=[np.lib.format.open_memmap(folder/f'batch_{i}.npy',mode='w+',dtype=np.int8,shape=(probes,n,n)) for i,n in enumerate(sizes)]
    generator=torch.Generator(device=device).manual_seed(seed)
    for probe in range(probes):
        for arr,n in zip(arrays,sizes):arr[probe]=(2*torch.randint(0,2,(n,n),device=device,generator=generator)-1).to(torch.int8).cpu().numpy()
    for arr in arrays:arr.flush()
    del arrays
    seal(folder,[f'batch_{i}.npy' for i in range(len(sizes))],dict(sizes=sizes,probes=probes,seed=seed,device_type=torch.device(device).type))
    return [folder/f'batch_{i}.npy' for i in range(len(sizes))]

def fit_gaussian(model,pairs,folder,*,probes=32,batch_size=512,physical_batch=8,data_sha256):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if (folder/'manifest.json').exists():verify(folder)
    seed={'qwen':241241,'harrier':241301,'gte':241128}[model.name];n=len(pairs)
    if not n:raise ValueError('no calibration pairs')
    order=torch.randperm(n,generator=torch.Generator().manual_seed(2025)).numpy();sizes=[min(batch_size,n-i) for i in range(0,n,batch_size)]
    order_file=folder/'batch_order.npy'
    if order_file.exists():
        if not np.array_equal(np.load(order_file,allow_pickle=False),order):raise ValueError('Gaussian batch order changed')
    else:np.save(order_file,order)
    noises=generate_noise(folder/'noise',sizes,probes,seed,model.device)
    binding=dict(data_sha256=data_sha256,encoder_hash=model.initial_hash,policy=model.policy,asset_hashes=model.asset_hashes,
        probes=probes,logical_batch=batch_size,physical_batch=physical_batch,seed=seed,
        order_sha256=sha(folder/'batch_order.npy'),noise_manifest_sha256=sha(folder/'noise/manifest.json'))
    if (folder/'manifest.json').exists():
        record=verify(folder)
        if record['metadata']!=binding:raise ValueError('Gaussian source/runtime changed')
        if not np.array_equal(np.load(folder/'mean_weight.npy'),model.weight.cpu().numpy()):raise ValueError('Gaussian weight center changed')
        return torch.as_tensor(np.load(folder/'variance_teacher.npy'),device=model.device),True
    count=min(16,model.weight.numel());coordinates=np.random.default_rng(seed+1).choice(model.weight.numel(),count,replace=False)
    coords=torch.as_tensor(coordinates,device=model.device);diagonal=torch.zeros_like(model.weight);samples=np.zeros((probes,count));exact=np.zeros(count);completed=0;kl=0.
    checkpoint=None;progress=folder/'progress.json'
    if progress.exists():
        state=json.loads(progress.read_text())
        if state['binding']!=binding:raise ValueError('Gaussian checkpoint source/runtime changed')
        from .storage import safe_file
        checkpoint=safe_file(folder,state['checkpoint_file'])
        if sha(checkpoint)!=state['checkpoint_sha256']:raise ValueError('Gaussian checkpoint changed')
        with np.load(checkpoint,allow_pickle=False) as z:
            diagonal.copy_(torch.as_tensor(z['diagonal'],device=model.device));samples=z['samples'];exact=z['exact'];completed=int(z['completed']);kl=float(z['kl'])
    for bi in range(completed,len(sizes)):
        chosen=order[bi*batch_size:bi*batch_size+sizes[bi]];texts=[pairs[i]['query'] for i in chosen]+[pairs[i]['positive'] for i in chosen]
        features=[]
        for begin in range(0,len(texts),physical_batch):features.extend(model.capture(texts[begin:begin+physical_batch])[0])
        result=batch_ggn(model,features,coords,np.load(noises[bi],mmap_mode='r',allow_pickle=False))
        diagonal+=result['diagonal'];samples+=result['coordinate_samples'].numpy();exact+=result['exact_coordinates'].cpu().numpy();kl+=result['teacher_kl_sum']
        # Keep the last committed generation intact until its successor's pointer
        # is published. A kill between these writes can always resume the old one.
        successor=folder/f'checkpoint_{bi+1:06d}.npz';temporary=folder/f'checkpoint.{os.getpid()}.tmp.npz'
        np.savez(temporary,diagonal=diagonal.cpu().numpy(),samples=samples,exact=exact,completed=bi+1,kl=kl);os.replace(temporary,successor)
        write_json(progress,dict(completed_batches=bi+1,total_batches=len(sizes),binding=binding,checkpoint_file=successor.name,checkpoint_sha256=sha(successor)))
        if checkpoint is not None and checkpoint!=successor:checkpoint.unlink(missing_ok=True)
        checkpoint=successor
        print('GAUSSIAN',model.name,bi+1,len(sizes),'seconds',time.perf_counter()-start,flush=True)
        del features,result
    audit=coordinate_audit(samples,exact)
    if not audit['passed'] or abs(kl/n)>1e-10:raise ValueError('Gaussian coordinate/teacher-center audit failed: '+str(audit))
    if not torch.isfinite(diagonal).all() or diagonal.min()<0:raise ValueError('invalid GGN diagonal')
    if not np.allclose(diagonal.flatten()[coords].cpu().numpy(),samples.mean(0),atol=1e-10,rtol=1e-10):raise ValueError('probe coordinate replay mismatch')
    precision=1/float(model.weight.square().mean());prior=torch.full_like(diagonal,1/precision);teacher=1/(precision+diagonal)
    for key,value in [('mean_weight',model.weight),('diagonal_ggn',diagonal),('variance_prior',prior),('variance_teacher',teacher)]:np.save(folder/(key+'.npy'),value.cpu().numpy())
    np.savez(folder/'coordinate_audit.npz',coordinates=coordinates,exact=exact,per_probe=samples)
    write_json(folder/'summary.json',dict(method='accepted v3 frozen-teacher SUM categorical GGN; no encoder update',prior_precision=precision,teacher_kl_per_observation=kl/n,
        coordinate_audit=audit,probes=probes,logical_batch=batch_size,physical_batch=physical_batch,seed=seed,
        parameter_gaussian_fitted_here=True,seconds=time.perf_counter()-start,policy=model.policy))
    seal(folder,['mean_weight.npy','diagonal_ggn.npy','variance_prior.npy','variance_teacher.npy','coordinate_audit.npz','batch_order.npy','summary.json','noise/manifest.json'],binding)
    return teacher,False
