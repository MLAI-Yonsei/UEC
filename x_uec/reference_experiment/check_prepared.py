"""Validate prepared cache in an isolated process without fitting alignment."""
import argparse,json
from pathlib import Path
from e4_refit import load_verified_inputs
from full_audit import audit_full_inputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());data=load_verified_inputs(config);audit=audit_full_inputs(data)
    if config.get('runtime_verification_path'):
        from run_covariance_validation import sha
        file=Path(config['runtime_verification_path'])
        if sha(file)!=config['runtime_verification_sha256']:raise ValueError('runtime evidence changed')
        evidence=json.loads(file.read_text())
        for path,expected in evidence['additional_consumed_files'].items():
            if sha(path)!=expected:raise ValueError('runtime source changed')
    print(json.dumps(dict(status='prepared_inputs_verified',config=a.config,original_observations=audit['original_observations'],alignment_refit_performed=False)))

if __name__=='__main__':main()
