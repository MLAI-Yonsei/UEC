"""One encoder per process, releasing GPU memory between model stages."""
import argparse,json
from pathlib import Path
from .storage import verify,sha,read_jsonl,code_hashes

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',required=True,choices=['qwen','harrier','gte'])
    a=p.parse_args();work=a.work.resolve();config=json.loads((work/'preparation_config.json').read_text())
    if config['code_files']!=code_hashes():raise ValueError('preparation code changed')
    verify(work/'data');pairs=read_jsonl(work/'data/curvature_pairs.jsonl')
    from .models import ModelPath
    from .curvature import fit_gaussian
    from .export import export_model
    model=ModelPath(a.model,config['cache_dir'],device=config['device'],offline=config['offline'],qwen_attention=config['qwen_attention'])
    variance,_=fit_gaussian(model,pairs,work/'models'/a.model/'gaussian',probes=config['probes'],batch_size=config['logical_batch'],
        physical_batch=config['physical_batch'],data_sha256=sha(work/'data/manifest.json'))
    export_model(model,variance,work,config['physical_batch'])

if __name__=='__main__':main()
