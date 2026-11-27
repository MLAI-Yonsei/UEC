import hashlib,json,os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temp,path)

def read_jsonl(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path,rows):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))

def safe_file(folder,name):
    folder=Path(folder).resolve();file=(folder/name).resolve()
    if not file.is_relative_to(folder):raise ValueError('manifest path escapes stage')
    return file

def seal(folder,names,metadata):
    folder=Path(folder);file=folder/'manifest.json'
    if file.exists():raise FileExistsError('stage already sealed: '+str(folder))
    value=dict(metadata=metadata,files={str(n):sha(safe_file(folder,n)) for n in names})
    write_json(file,value);return value

def verify(folder):
    folder=Path(folder);p=json.loads((folder/'manifest.json').read_text())
    for name,expected in p['files'].items():
        if sha(safe_file(folder,name))!=expected:raise ValueError('stage checksum mismatch: '+name)
    return p

def code_hashes():
    root=Path(__file__).resolve().parents[1]
    paths=[*root.glob('*.py'),*root.glob('standalone/*.py'),*root.glob('reference_experiment/*.py')]
    return {str(p.relative_to(root)):sha(p) for p in paths if not p.name.startswith('test_')}
