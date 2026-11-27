"""Explicit v3 E4 smoke CLI; full execution is never implicit."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from e4_refit import *

def worker(config,output):
    output=Path(output);started=time.perf_counter()
    def progress(phase):write_json_atomic(output/'progress.json',dict(phase=phase,seconds=time.perf_counter()-started))
    progress('input_validation')
    data=load_verified_inputs(config);split=make_alignment_split(data,config)
    if '_config_path' in config:
        data.checksums[config['_config_path']]=sha(config['_config_path'])
        data.provenance['execution_config']=dict(path=config['_config_path'],sha256=sha(config['_config_path']))
    write_json(output/'preflight.json',dict(status='ready_for_smoke',**scope_flags()))
    input_seconds=time.perf_counter()-started
    result=fit_e4(data,split,config,progress)
    progress('geometry_and_raw_moment_equivalence');check_start=time.perf_counter()
    moments,geometry=validate_e4_transport(data,split,result,config)
    for file,expected in data.checksums.items():
        if sha(file)!=expected:raise ValueError('input changed during fit')
    result['input_validation_seconds']=input_seconds;result['validation_seconds']=time.perf_counter()-check_start
    write_e4_outputs(output,data,split,result,moments,geometry);progress('completed')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run','worker']);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['smoke'],default='smoke');a=p.parse_args();config=json.loads(Path(a.config).read_text());out=Path(a.output).resolve()
    config['_config_path']=str(Path(a.config).resolve())
    if a.stage=='worker':
        try:worker(config,out)
        except Exception as error:
            state=dict(status='blocked_missing_inputs' if isinstance(error,(FileNotFoundError,KeyError)) else 'smoke_failed',
                error_type=type(error).__name__,error=str(error),alignment_refit_performed=False,**scope_flags())
            write_json_atomic(out/'status.json',state);traceback.print_exc();sys.exit(2)
        return
    if a.stage=='preflight':
        out.mkdir(parents=True,exist_ok=False)
        try:state=preflight(config)
        except Exception as error:state=dict(status='smoke_failed',error_type=type(error).__name__,error=str(error),**scope_flags())
        write_json(out/'preflight.json',state);print(json.dumps(state,indent=2));sys.exit(0 if state['status']=='ready_for_smoke' else 2)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
    state=run_with_deadline([sys.executable,str(Path(__file__).resolve()),'worker','--config',str(Path(a.config).resolve()),'--output',str(out)],out,300)
    print(json.dumps(state,indent=2))
    if state['status']!='smoke_pass':sys.exit(2)

if __name__=='__main__':main()
