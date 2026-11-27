"""Bind generated provenance to independent producer records and actual runtime."""
import hashlib
import json
from importlib.metadata import distributions
from pathlib import Path
import re

MODELS=('qwen','harrier','gte')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verified_provenance(folder):
    folder=Path(folder).resolve();p=json.loads((folder/'uec_input_provenance.json').read_text())
    if p['delivery_version']!='v3' or p['cache_scope']!='alignment_full':raise ValueError('expected v3 alignment_full preparation')
    if set(p['models'])!=set(MODELS):raise ValueError('missing producer models')
    consumed={str(Path(f).resolve()):h for f,h in p['consumed_files'].items()}
    def bound(file):
        if consumed.get(str(file.resolve()))!=sha(file):raise ValueError('unbound producer summary/input '+str(file))
    for key,name in [('input_path','uec_alignment_inputs.npz'),('observations_path','observations.jsonl'),('split_path','split.json')]:
        file=folder/name
        if Path(p[key]).resolve()!=file.resolve():raise ValueError('prepared directory payload points elsewhere: '+key)
        bound(file)
    for m in MODELS:
        file=folder/m/'summary.json';bound(file);actual=json.loads(file.read_text())
        if actual!=p['models'][m] or actual.get('status')!='completed' or actual.get('model')!=m:raise ValueError('embedded producer summary mismatch '+m)
    return p

def query_versions(interpreter,packages):
    # Inspect the recorded environment's metadata; never execute a path from JSON.
    prefix=Path(interpreter).parent.parent;folders=list((prefix/'lib').glob('python*/site-packages'))
    cfg=prefix/'pyvenv.cfg'
    if cfg.is_file():
        settings={k.strip():v.strip() for line in cfg.read_text().splitlines() if '=' in line for k,v in [line.split('=',1)]}
        if settings.get('include-system-site-packages','').lower()=='true':
            folders+=list((Path(settings['home']).parent/'lib').glob('python*/site-packages'))
    if not folders:raise FileNotFoundError('recorded runtime package metadata unavailable')
    canonical=lambda name:re.sub('[-_.]+','-',name).lower()
    versions={}
    for folder in folders:
        for d in distributions(path=[str(folder)]):
            if d.metadata.get('Name'):versions.setdefault(canonical(d.metadata['Name']),d.version)
    return {name:versions[canonical(name)] for name in packages}

def bind_runtime_evidence(folder,p,config):
    if all('runtime_versions' in r and 'runtime_policy' in r for r in p['models'].values()):
        for m,r in p['models'].items():
            if r['runtime_versions']!=r['policy']['versions'] or not r['runtime_policy']:raise ValueError('producer runtime versions/policy mismatch '+m)
            if any(r['policy'].get(k)!=v for k,v in r['runtime_policy'].items()):raise ValueError('producer runtime policy mismatch '+m)
        return
    file=Path(folder).resolve()/'runtime_policy_verification.json'
    evidence=json.loads(file.read_text())  # Missing historical evidence is an error.
    if evidence['timing']!='retrospective_after_cache_generation':raise ValueError('historical runtime evidence timing mismatch')
    for m,r in p['models'].items():
        actual=evidence['model_runtime_versions'][m];expected=r['policy']['versions']
        if actual['versions']!=expected or query_versions(actual['interpreter'],expected)!=expected:raise ValueError('historical runtime versions mismatch '+m)
    author=False
    for name,expected in evidence['additional_consumed_files'].items():
        if sha(name)!=expected:raise ValueError('historical runtime source mismatch')
        if Path(name).name=='qwen_author_model.py':
            author=expected==p['models']['qwen']['policy']['source_sha256']
    if not author:raise ValueError('historical runtime source missing Qwen author binding')
    config.update(runtime_verification_path=str(file),runtime_verification_sha256=sha(file))

def verify_model_policy(name,model,expected):
    if name in ('qwen','harrier'):
        actual=model.policy
        if actual!=expected:raise ValueError('loaded runtime policy differs from parent '+name)
        return actual
    enc=model.encoder;first=enc[0];config=first.auto_model.config
    snapshot=Path(config._name_or_path).name
    actual=dict(revision=snapshot,tokenizer_revision=snapshot,native_dim=enc.get_sentence_embedding_dimension(),
        max_seq_length=enc.max_seq_length,actual_dtype=str(next(enc.parameters()).dtype),
        actual_attention_backend=config._attn_implementation,actual_padding_side=enc.tokenizer.padding_side,
        actual_truncation_side=enc.tokenizer.truncation_side,tokenizer_class=type(enc.tokenizer).__name__,
        pooling=enc[1].get_config_dict(),modules=[type(m).__name__ for m in enc],default_prompt_name=enc.default_prompt_name,
        do_lower_case=first.do_lower_case,prompt='')
    if any(expected.get(k)!=v for k,v in actual.items()):raise ValueError('loaded runtime policy differs from parent '+name)
    return actual
