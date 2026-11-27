"""Download public inputs and fit v3 uncertainty; no private workspace needed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PRODUCER=Path(__file__).resolve().parent/'reference_experiment/prepare_cache.py'

def check_workspace(root):
    required=['caia_svd_pilot_v3/uncertainty.py','caia_pair_pilot/qwen_adapter.py','gte_la/model.py',
        'runs/gte_la_validation_20261124/v1/data/triplets.jsonl',
        'runs/caia_svd_frozen_triple_20261124/v3/manifest.json',
        '.venv/bin/python','.venv-reproduction/bin/python']
    missing=[name for name in required if not (root/name).is_file()]
    if missing:raise FileNotFoundError('original experiment workspace required; missing: '+', '.join(missing))

def check_gpu(gpu,*,settle_utilization=False):
    import time
    for attempt in range(6 if settle_utilization else 1):
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        row=next((line for line in lines if int(line.split(',')[0])==gpu),None)
        if row is None:raise ValueError('GPU index does not exist')
        _,uuid,memory,total,util=[v.strip() for v in row.split(',')]
        if int(total)<46000:raise RuntimeError('GPU memory below the original supported 46000 MiB profile')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
        if int(memory)>1024 or any(a.split(',')[0].strip()==uuid for a in apps):break
        if int(util)==0:return uuid
        if settle_utilization and attempt<5:time.sleep(1)
    raise RuntimeError('GPU is busy; select an idle GPU and preserve existing jobs')

def storage_environment(workspace):
    source=workspace.parent/'home-offload/storage-env.sh'
    if not source.is_file():raise FileNotFoundError('original storage environment required: '+str(source))
    code='import os,json; print(json.dumps({k:os.environ.get(k) for k in ["HF_HOME","XDG_CACHE_HOME","TMPDIR"]}))'
    text=subprocess.check_output(['/bin/bash','-c','source "$1"; exec "$2" -c "$3"','xuec-storage',str(source),sys.executable,code],text=True)
    env=json.loads(text)
    for key,value in env.items():
        if not value or not str(Path(value).resolve()).startswith(('/data1/','/data2/')):raise ValueError('off-home storage setting required: '+key)
    env['HUGGINGFACE_HUB_CACHE']=str(Path(env['HF_HOME'])/'hub');env['TRANSFORMERS_CACHE']=env['HUGGINGFACE_HUB_CACHE']
    env['HF_DATASETS_CACHE']=str(Path(env['HF_HOME'])/'datasets');env['TORCH_HOME']=str(Path(env['XDG_CACHE_HOME'])/'torch')
    return env

def prepare(workspace,output,*,gpu):
    workspace,output=Path(workspace).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('use a new output directory: '+str(output))
    check_workspace(workspace)
    env=dict(os.environ,XUEC_EXPERIMENT_WORKSPACE=str(workspace),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env.update(storage_environment(workspace))
    regular=workspace/'.venv/bin/python';qwen_gte=workspace/'.venv-reproduction/bin/python'
    def invoke(interpreter,stage,*extra):
        subprocess.run([str(interpreter),str(PRODUCER),stage,'--output',str(output),*extra],cwd=str(workspace),env=env,check=True)
    check_gpu(gpu)
    invoke(regular,'data')
    for model in ['qwen','harrier','gte']:
        env['CUDA_VISIBLE_DEVICES']=check_gpu(gpu)
        invoke(regular if model=='harrier' else qwen_gte,'model','--model',model)
    invoke(regular,'assemble')
    return output/'uec_input_provenance.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--work-dir',type=Path,help='Data disk directory for public downloads, new Gaussian and generated inputs')
    group.add_argument('--workspace',type=Path,help='Legacy only: reuse the original archived experiment workspace')
    p.add_argument('--output',type=Path,help='Legacy --workspace output only')
    p.add_argument('--gpu',type=int,required=True,help='Idle physical GPU index; models are processed sequentially')
    p.add_argument('--profile',choices=['full','smoke'],default='full',help='full: 3983 MS MARCO + 3775 SNLI pairs; smoke: 32+32')
    p.add_argument('--cache-dir',type=Path,help='Optional existing standard Hugging Face hub cache on a data disk')
    p.add_argument('--offline',action='store_true',help='Use only already downloaded public assets in cache')
    p.add_argument('--physical-batch',type=int,default=8,help='Encoder chunk size; logical curvature batch stays 512')
    p.add_argument('--evaluation-texts',type=Path,action='append',default=[],help='Additional evaluation JSONL with text field; exclude matching texts')
    p.add_argument('--qwen-attention',choices=['sdpa','flash_attention_2'],default='sdpa')
    a=p.parse_args()
    try:
        if a.work_dir:
            if a.output:p.error('--output is legacy only; standalone inputs are written under --work-dir/inputs')
            from standalone.pipeline import prepare as standalone_prepare
            print(standalone_prepare(a.work_dir,profile=a.profile,gpu=a.gpu,cache=a.cache_dir,offline=a.offline,
                physical_batch=a.physical_batch,evaluation_texts=a.evaluation_texts,qwen_attention=a.qwen_attention))
        else:
            if a.output is None:p.error('--workspace requires --output')
            print(prepare(a.workspace,a.output,gpu=a.gpu))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,subprocess.CalledProcessError) as e:p.exit(1,str(e)+'\n')

if __name__=='__main__':main()
