"""Public-data preparation driver; no original experiment directory is needed."""
import json,os,subprocess,sys
import fcntl
from contextlib import contextmanager
from pathlib import Path
from .storage import write_json,code_hashes,sha,verify

def environment(work,cache=None,offline=False):
    work=Path(work).resolve();cache=Path(cache or work/'cache/huggingface/hub').resolve()
    for path in [work,cache]:
        if str(path).startswith('/home/'):raise ValueError('put work directory and caches on a data disk, outside /home')
    values=dict(HF_HOME=str(cache.parent),HF_HUB_CACHE=str(cache),HUGGINGFACE_HUB_CACHE=str(cache),
        HF_DATASETS_CACHE=str(work/'cache/datasets'),XDG_CACHE_HOME=str(work/'cache'),TORCH_HOME=str(work/'cache/torch'),TMPDIR=str(work/'tmp'))
    for path in values.values():Path(path).mkdir(parents=True,exist_ok=True)
    values.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1' if offline else '0',TRANSFORMERS_OFFLINE='1' if offline else '0')
    env=dict(os.environ,**values);env.pop('TRANSFORMERS_CACHE',None)
    return env

def bind_config(work,config):
    file=Path(work)/'preparation_config.json'
    if file.exists():
        if json.loads(file.read_text())!=config:raise ValueError('preparation configuration/code changed; use a new work directory')
    else:write_json(file,config)

@contextmanager
def preparation_lock(work):
    with (Path(work)/'.preparation.lock').open('a+') as file:
        try:fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('preparation is already running in this work directory')
        try:yield
        finally:fcntl.flock(file,fcntl.LOCK_UN)

def prepare(work,**kwargs):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    with preparation_lock(work):return _prepare(work,**kwargs)

def _prepare(work,*,profile='full',gpu=0,cache=None,offline=False,physical_batch=8,evaluation_texts=(),qwen_attention='sdpa'):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    env=environment(work,cache,offline);os.environ.update({k:v for k,v in env.items() if k not in os.environ or k in ['HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']})
    if profile not in ['full','smoke'] or physical_batch<1:raise ValueError('invalid profile or physical batch')
    from .storage import sha
    from .models import package_versions,SPECS
    config=dict(profile=profile,probes=32,logical_batch=512,physical_batch=physical_batch,device='cuda:0',gpu=gpu,versions=package_versions(),
        cache_dir=env['HF_HUB_CACHE'],offline=offline,qwen_attention=qwen_attention,code_files=code_hashes(),
        additional_evaluation=[dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in evaluation_texts])
    bind_config(work,config)
    from .assets import prepare_data
    prepare_data(work,profile,env['HF_HUB_CACHE'],offline,evaluation_texts)
    from prepare_alignment import check_gpu
    root=Path(__file__).resolve().parents[1]
    previous_worker=False
    for model in ['qwen','harrier','gte']:
        stage=work/'models'/model
        if (stage/'manifest.json').exists():
            verify(stage);verify(stage/'gaussian');summary=json.loads((stage/'summary.json').read_text());s=SPECS[model]
            snapshot=Path(env['HF_HUB_CACHE'])/('models--'+s['model_id'].replace('/','--'))/'snapshots'/s['revision']
            for name,expected in summary['asset_hashes'].items():
                if sha(snapshot/name)!=expected:raise ValueError('completed model source/runtime changed')
            continue
        env['CUDA_VISIBLE_DEVICES']=check_gpu(gpu,settle_utilization=previous_worker)
        subprocess.run([sys.executable,'-m','standalone.worker','--work',str(work),'--model',model],cwd=root,env=env,check=True)
        previous_worker=True
    from .export import assemble
    assemble(work)
    print('Prepared inputs:',work/'inputs',flush=True)
    return work/'inputs/uec_input_provenance.json'
