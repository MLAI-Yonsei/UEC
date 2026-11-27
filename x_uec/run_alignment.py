"""Generate full-run configuration from prepared provenance, then run v3."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from preparation_checks import verified_provenance,bind_runtime_evidence

RUNNER=Path(__file__).resolve().parent/'reference_experiment/full_e4.py'

def config_from_inputs(folder):
    file=Path(folder).resolve()/'uec_input_provenance.json'
    if json.loads(file.read_text()).get('input_profile')=='standalone_v1':
        from standalone.alignment import inspect_inputs
        p=inspect_inputs(folder)
        return dict(mode=p['profile'],delivery_version='v3',input_profile='standalone_v1',provenance_path=str(file))
    provenance=Path(folder).resolve()/'uec_input_provenance.json';p=verified_provenance(folder)
    return dict(mode='full',delivery_version='v3',provenance_path=str(provenance),backend_files=p['backend_files'],models={m:r['policy'] for m,r in p['models'].items()})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true',help='Verify prepared inputs and generate config without refitting alignment')
    args=parser.parse_args()
    try:
        config=config_from_inputs(args.inputs)
        if config.get('input_profile')!='standalone_v1':bind_runtime_evidence(args.inputs,verified_provenance(args.inputs),config)
        file=Path(str(args.output)+'.config.json').resolve();file.parent.mkdir(parents=True,exist_ok=True)
        text=json.dumps(config,indent=2)+'\n'
        if file.exists() and file.read_text()!=text:raise ValueError('existing generated config differs; use a new preparation')
        if not file.exists():file.write_text(text)
        if args.check_only:
            if config.get('input_profile')=='standalone_v1':
                sys.path.insert(0,str(RUNNER.parent))
                from standalone.alignment import load_inputs,audit_and_split
                _,split=audit_and_split(load_inputs(config))
                print(json.dumps(dict(status='verified',fit=len(split.fit_indices),diagnostic=len(split.diagnostic_indices),mode=config['mode'])))
                return
            subprocess.run([sys.executable,str(RUNNER.with_name('check_prepared.py')),'--config',str(file)],check=True);return
        if args.output.exists():raise FileExistsError('use a new result directory')
        subprocess.run([sys.executable,str(RUNNER),'--config',str(file),'--output',str(args.output)],check=True)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError,subprocess.CalledProcessError) as e:parser.exit(1,str(e)+'\n')

if __name__=='__main__':main()
