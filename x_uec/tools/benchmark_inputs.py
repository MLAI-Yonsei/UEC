"""Portable, identity-checked inputs for the exploratory six-task pilot.

The legacy-import command reads an explicitly supplied certified cache index;
it never imports the original experiment package. Public-source reconstruction
and native-only encoding are exposed separately from the frozen alignment code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from standalone.storage import read_jsonl, safe_file, seal, sha, verify, write_json


def text_sha256(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def fingerprint(value):
    return text_sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False))


def canonical_text_sha256(text):
    """Match current alignment exclusion: NFC and newline normalization only."""
    return text_sha256(unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n')))


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(files):
    for path,expected in files.items():
        if sha(path)!=expected:raise ValueError('source checksum mismatch: '+str(path))


def load_certified_cache(ref,rows):
    """Check original model row identities/policy before reading native arrays."""
    verify_hashes(ref['files'])
    meta=read_json(ref['meta_path']);contract=meta['contract'];expected=ref['encoder_config']
    if not meta['complete'] or fingerprint(contract)!=meta['contract_fingerprint']:raise ValueError('incomplete or changed cache contract')
    if contract['encoder_config']!=expected or contract['encoder_fingerprint']!=fingerprint(expected):raise ValueError('cache encoder revision/role/prompt mismatch')
    identity=lambda rs:[{k:r[k] for k in ['row_id','exact_hash']} for r in rs]
    if identity(contract['rows'])!=identity(rows):raise ValueError('original cache row IDs/order mismatch')
    if len({r['row_id'] for r in rows})!=len(rows):raise ValueError('duplicate raw row IDs')
    if read_json(ref['rows_path'])!=contract['rows']:raise ValueError('model-specific row sidecar mismatch')
    for row,cached in zip(rows,contract['rows']):
        if text_sha256(row['text'])!=row['exact_hash'] or cached['cache_key']!=text_sha256(row['text']+fingerprint(expected)):raise ValueError('original cache exact text hash mismatch')
        if 'canonical_hash' in row and text_sha256(' '.join(unicodedata.normalize('NFC',row['text']).split()))!=row['canonical_hash']:raise ValueError('legacy canonical text hash mismatch')
    a=ref['array'];path=Path(a['path'])
    if sha(path)!=a['sha256']:raise ValueError('native array checksum mismatch')
    loaded=np.load(path,allow_pickle=False)
    if a['key'] is not None:
        try:x=np.array(loaded[a['key']],copy=True)
        finally:loaded.close()
    else:x=np.array(loaded,copy=True)
    if sha(path)!=a['sha256']:raise ValueError('native array changed while reading')
    if list(x.shape)!=a['shape'] or str(x.dtype)!=a['dtype'] or list(x.shape)!=contract['shape'] or x.dtype!=np.float32 or x.ndim!=2 or len(x)!=len(rows) or not np.isfinite(x).all():raise ValueError('native shape/dtype/finite contract mismatch')
    norms=np.linalg.norm(x.astype(np.float64),axis=1,keepdims=True)
    if np.any(norms==0):raise ValueError('zero native row')
    return x.astype(np.float64)/norms


def fit_policies(work):
    if work is None:return {},{}
    policies={};sources={}
    for model in ['qwen','harrier','gte']:
        path=Path(work)/'models'/model/'gaussian/manifest.json'
        stage=verify(path.parent);policies[model]=stage['metadata']['policy'];sources[str(path.resolve())]=sha(path)
    return policies,sources


def export_legacy(cache_index,registry,data_dir,output,fit_work=None):
    """Create sealed portable inputs; existing completed exports are verified."""
    cache_index=Path(cache_index);registry=Path(registry);data_dir=Path(data_dir);output=Path(output)
    binding=dict(cache_index_sha256=sha(cache_index),registry_sha256=sha(registry))
    policies,fit_sources=fit_policies(fit_work)
    binding['fit_policy_sources']=fit_sources
    if (output/'manifest.json').exists():
        saved=verify(output)
        if saved['metadata']['binding']!=binding:raise ValueError('benchmark export source binding changed')
        for task in saved['metadata']['tasks']:verify(output/task)
        return saved
    index=read_json(cache_index);entries={r['runtime_task_id']:r for r in read_json(registry)};names=[];sources={str(cache_index.resolve()):sha(cache_index),str(registry.resolve()):sha(registry),**fit_sources};all_files=[]
    for task,roles in index.items():
        source=data_dir/task;meta=read_json(source/'manifest.json')
        verify_hashes({str(safe_file(source,n)):h for n,h in meta['files'].items()})
        sources[str((source/'manifest.json').resolve())]=sha(source/'manifest.json')
        target=output/task
        if (target/'manifest.json').exists():
            old=verify(target)
            if old['metadata']['source_manifest_sha256']!=sha(source/'manifest.json') or old['metadata']['cache_index_sha256']!=binding['cache_index_sha256']:raise ValueError('partial benchmark source binding changed')
        else:
            target.mkdir(parents=True,exist_ok=True);record=dict(task=task,kind=meta['settings']['kind'],roles={},source=meta['settings'],registry=entries[task]);arrays={};cache_policies={};cache_sources={}
            for role,models in roles.items():
                rows=read_jsonl(source/f'{role}.jsonl');record['roles'][role]=[dict(r,canonical_text_sha256=canonical_text_sha256(r['text'])) for r in rows]
                for model,ref in models.items():
                    arrays[f'{role}__{model}']=load_certified_cache(ref,rows);cache_policies.setdefault(model,{})[role]=ref['encoder_config'];cache_sources.update(ref['files'])
            kind=record['kind']
            if kind=='Retrieval':record['qrels']=read_json(source/'qrels.json')
            elif kind=='Classification':record['labels']=read_json(source/'labels.json')
            elif kind=='STS':record['pairs']=read_jsonl(source/'pairs.jsonl')
            else:raise ValueError('unsupported task kind')
            write_json(target/'task.json',record);np.savez(target/'native.npz',**arrays)
            write_json(target/'provenance.json',dict(source_manifest_sha256=sha(source/'manifest.json'),source_files=meta['files'],cache_files=cache_sources,cache_policies=cache_policies,fit_policies=policies,encoding_comparability=dict(bit_identical_to_fit_runtime=False,normalization='certified original native caches converted to float64, then row L2',fit_feature_path='current fit uses reconstructed FP64 final path',known_difference='Qwen evaluation cache uses flash_attention_2 and FP16 output normalization; current fit uses sdpa and FP64 reconstructed output. Task-specific prompts are preserved.'),scope='six-task exploratory pilot; not full MTEB'))
            seal(target,['task.json','native.npz','provenance.json'],dict(task=task,source_manifest_sha256=sha(source/'manifest.json'),cache_index_sha256=binding['cache_index_sha256']))
        names.append(task)
        all_files.extend(f'{task}/{n}' for n in ['task.json','native.npz','provenance.json','manifest.json'])
        print('EXPORTED',task,flush=True)
    write_json(output/'provenance.json',dict(sources=sources,fit_policies=policies,source_mode='explicit certified legacy cache import',scope='six-task exploratory pilot; not full MTEB',uses_old_alignment_maps=False,requires_encoder_inference=False))
    all_files.append('provenance.json')
    return seal(output,all_files,dict(tasks=names,binding=binding,scope='six-task exploratory pilot; not full MTEB',schema_version=1))


REFERENCE=Path(__file__).with_name('benchmark_reference')


def text_row(row_id,text):
    return dict(row_id=str(row_id),text=text,exact_hash=text_sha256(text),canonical_hash=text_sha256(' '.join(unicodedata.normalize('NFC',text).split())),canonical_text_sha256=canonical_text_sha256(text))


def check_reference(roles,reference):
    if set(roles)!=set(reference):raise ValueError('public role identity mismatch')
    for role,rows in roles.items():
        if [[r['row_id'],r['exact_hash']] for r in rows]!=reference[role]:raise ValueError('public source row identity mismatch: '+role)
        if len({r['row_id'] for r in rows})!=len(rows):raise ValueError('duplicate public row identity')
        if any(text_sha256(r['text'])!=r['exact_hash'] for r in rows):raise ValueError('public text exact hash mismatch')


def reconstruct_roles(task,spec,raw):
    """Replay source records without labels/text selection changes."""
    cfg=spec['settings'];kind=cfg['kind']
    if kind=='Classification':
        train=raw['train'];test=raw['test'];labels=spec['labels']
        indices=sorted({i for trial in labels['support_train_indices'] for i in trial})
        if [int(train[i]['label']) for i in indices]!=labels['support'] or [int(r['label']) for r in test]!=labels['test']:raise ValueError('public classification labels differ from fixed protocol')
        return dict(support=[text_row(f'{task}:{cfg["train"]}:{i}',train[i][cfg['text']]) for i in indices],test=[text_row(f'{task}:{cfg["split"]}:{i}',r[cfg['text']]) for i,r in enumerate(test)])
    if kind=='Retrieval':
        corpus=[text_row(r['_id'],((r.get('title','') or '')+' '+r['text']).strip()) for r in raw['corpus.jsonl']]
        queries=[text_row(r['_id'],r['text']) for r in raw['queries.jsonl']];qrels={}
        doc_ids={r['row_id'] for r in corpus};query_ids={r['row_id'] for r in queries}
        for item in raw['qrels/test.jsonl']:
            qid,cid,score=str(item['query-id']),str(item['corpus-id']),int(item['score'])
            if qid not in query_ids or cid not in doc_ids or score<0:raise ValueError('invalid public relevance judgment')
            if cid in qrels.setdefault(qid,{}) and qrels[qid][cid]!=score:raise ValueError('conflicting public relevance judgment')
            qrels[qid][cid]=score
        if qrels!=spec['qrels']:raise ValueError('public qrels differ from fixed protocol')
        return dict(corpus=corpus,queries=[r for r in queries if r['row_id'] in qrels])
    if kind!='STS':raise ValueError('unknown public task kind')
    filename='test/en.jsonl.gz' if cfg['legacy']=='sts22' else 'test.jsonl.gz';texts=[];lookup={};pairs=[]
    prefix='sts22' if cfg['legacy']=='sts22' else 'stsb'
    for index,item in enumerate(raw[filename]):
        indices=[]
        for field in ['sentence1','sentence2']:
            text=item[field];key=text_sha256(text)
            if key not in lookup:lookup[key]=len(texts);texts.append(text_row(prefix+':'+key,text))
            indices.append(lookup[key])
        pair_id=item['id'] if prefix=='sts22' else f'stsb:test:{index}:{item["sid"]}'
        pairs.append(dict(id=pair_id,left_index=indices[0],right_index=indices[1],gold=float(item['score'])))
    expected=[{k:r[k] for k in ['id','left_index','right_index','gold']} for r in spec['pairs']]
    if pairs!=expected:raise ValueError('public STS pair identity/score mismatch')
    return dict(texts=texts)


def prepare_public_sources(output,cache,offline=False):
    """Download pinned data, replay IDs, and seal source text outside Git."""
    import gzip
    import os
    from standalone.pipeline import environment
    output=Path(output).resolve();cache=Path(cache).resolve();os.environ.update(environment(output,cache,offline))
    os.environ['HF_MODULES_CACHE']=str(output/'cache/modules');os.environ['HF_DATASETS_OFFLINE']='1' if offline else '0'
    verify(REFERENCE);reference_sha=sha(REFERENCE/'manifest.json')
    if (output/'manifest.json').exists():
        stage=verify(output)
        if stage['metadata']['reference_sha256']!=reference_sha:raise ValueError('public benchmark reference changed')
        return stage
    from huggingface_hub import hf_hub_download
    protocol=read_json(REFERENCE/'protocol.json')
    with gzip.open(REFERENCE/'row_ids.json.gz','rt') as stream:identities=json.load(stream)
    names=[];files=[]
    for task,spec in protocol['tasks'].items():
        cfg=spec['settings'];folder=output/task
        if (folder/'manifest.json').exists():
            stage=verify(folder)
            if stage['metadata']['reference_sha256']!=reference_sha:raise ValueError('public source reference changed')
        else:
            raw={};sources={}
            for filename,expected in spec['raw_files'].items():
                path=Path(hf_hub_download(cfg['repo'],filename,repo_type='dataset',revision=cfg['revision'],cache_dir=str(cache),local_files_only=offline))
                if sha(path)!=expected['sha256']:raise ValueError('pinned public dataset checksum mismatch: '+task+'/'+filename)
                opener=gzip.open if filename.endswith('.gz') else open
                with opener(path,'rt') as stream:raw[filename]=[json.loads(line) for line in stream if line.strip()]
                sources[filename]=dict(sha256=expected['sha256'],repo=cfg['repo'],revision=cfg['revision'])
            if cfg['kind']=='Classification':
                try:from datasets import load_dataset
                except ImportError as error:raise ImportError('public classification reconstruction requires datasets==2.21.0') from error
                ds=load_dataset(cfg['repo'],revision=cfg['revision'],trust_remote_code=True,cache_dir=str(output/'cache/datasets'))
                raw=dict(train=ds[cfg['train']],test=ds[cfg['split']]);sources=dict(repo=cfg['repo'],revision=cfg['revision'],dataset_fingerprints={k:v._fingerprint for k,v in ds.items()})
            roles=reconstruct_roles(task,spec,raw);check_reference(roles,identities[task])
            record=dict(task=task,kind=cfg['kind'],roles=roles,source=cfg,registry=spec['registry'])
            for key in ['qrels','labels','pairs']:
                if key in spec:record[key]=spec[key]
            write_json(folder/'task.json',record);write_json(folder/'provenance.json',dict(public_sources=sources,reference_sha256=reference_sha))
            seal(folder,['task.json','provenance.json'],dict(task=task,reference_sha256=reference_sha))
        files.extend(f'{task}/{n}' for n in ['task.json','provenance.json','manifest.json']);names.append(task)
        print('PUBLIC SOURCE',task,flush=True)
    return seal(output,files,dict(tasks=names,reference_sha256=reference_sha,scope=protocol['scope']))


def check_encoding_policy(model,expected):
    s=model.spec;p=model.policy
    actual=dict(model_id=s['model_id'],revision=s['revision'],tokenizer_revision=p['tokenizer_revision'],max_seq_length=s['length'],native_dim=s['dimension'],padding_side=p['padding_side'],truncation_side=p['truncation_side'])
    wanted={key:expected[key] for key in ['model_id','revision','tokenizer_revision','max_seq_length','native_dim']}
    wanted.update(padding_side=expected.get('padding_side',expected.get('actual_padding_side')),truncation_side=expected.get('truncation_side',expected.get('actual_truncation_side')))
    pooling=expected['pooling'];desired='last_valid_token' if model.name in ['qwen','harrier'] else 'masked_mean'
    if isinstance(pooling,dict):
        selected='pooling_mode_lasttoken' if model.name=='harrier' else 'pooling_mode_mean_tokens'
        pooling_valid=bool(pooling.get(selected)) and all(not value for key,value in pooling.items() if key.startswith('pooling_mode_') and key!=selected)
    else:pooling_valid=pooling==desired
    if actual!=wanted or p['pooling']!=desired or not pooling_valid:raise ValueError('public encoding policy revision/tokenizer/role/padding/pooling/maxseq mismatch')


def encode_mean_rows(model,rows,prompt,physical_batch=8):
    """Mean-only current FP64 final path, checked against actual encoder center."""
    import torch
    if physical_batch<1:raise ValueError('physical batch must be positive')
    model.spec=dict(model.spec,prompt=prompt);values=[];worst=0.
    with torch.no_grad():
        for begin in range(0,len(rows),physical_batch):
            features,actual=model.capture([r['text'] for r in rows[begin:begin+physical_batch]])
            if len(features)!=len(actual):raise ValueError('mean capture row count mismatch')
            for index,feature in enumerate(features):
                _,mu=model.forward(feature);value=mu.detach().double().cpu().numpy()
                error=float((mu.detach().cpu()-actual[index]).abs().max());worst=max(worst,error)
                if not np.isfinite(value).all() or not np.isfinite(error) or error>(1e-3 if model.name=='qwen' else 1e-5):raise ValueError('final-path/encoder center mismatch '+model.name)
                if value.ndim!=1 or abs(np.linalg.norm(value)-1)>1e-5:raise ValueError('mean native unit norm mismatch')
                values.append(value)
    if len(values)!=len(rows):raise ValueError('mean native row count mismatch')
    return np.asarray(values,dtype=np.float64),dict(max_encoder_center_abs_error=worst,center_checks=len(rows),path='current reconstructed FP64 final path',variance_computed=False)


def public_request(cache,device,offline,qwen_attention,physical_batch):
    from standalone.models import package_versions
    from standalone.storage import code_hashes
    if physical_batch<1 or qwen_attention not in ['sdpa','flash_attention_2']:raise ValueError('invalid public encoding request')
    return dict(mode='public',reference_sha256=sha(REFERENCE/'manifest.json'),implementation_sha256=sha(__file__),producer_sha256=fingerprint(code_hashes()),versions=package_versions(),device=str(device),qwen_attention=qwen_attention,physical_batch=physical_batch,cache=str(Path(cache).resolve()),offline=bool(offline),dtype_policy=dict(qwen='torch.float16' if str(device).startswith('cuda') else 'torch.float32',harrier='torch.float32',gte='torch.float32'),output_path='reconstructed FP64 mean only')


def check_public_request(metadata,request):
    if metadata.get('mode')!='public':raise ValueError('benchmark mode mismatch: public requested')
    if metadata.get('request')!=request:raise ValueError('public benchmark request/configuration/producer changed')


def validate_benchmark_suite(folder):
    """Canonical scoring boundary: seal integrity plus exact published semantics."""
    import gzip
    from types import SimpleNamespace
    folder=Path(folder);reference=verify(REFERENCE);top=verify(folder);metadata=top['metadata'];protocol=read_json(REFERENCE/'protocol.json');specs=protocol['tasks'];names=list(specs)
    canonical={'SCIDOCS','LegalBenchCorporateLobbying','PoemSentimentClassification','TweetTopicSingleClassification','STSBenchmark','STS22.v2'}
    if set(names)!=canonical or len(names)!=6 or metadata.get('tasks')!=names or metadata.get('schema_version')!=1:raise ValueError('benchmark suite task identities/schema mismatch')
    with gzip.open(REFERENCE/'row_ids.json.gz','rt') as stream:identities=json.load(stream)
    provenance=read_json(folder/'provenance.json');public=metadata.get('mode')=='public'
    if metadata.get('mode') not in [None,'legacy','public']:raise ValueError('unsupported benchmark mode')
    if public:
        request=metadata.get('request',{});current=public_request(request['cache'],request['device'],request['offline'],request['qwen_attention'],request['physical_batch']);check_public_request(metadata,current)
        if provenance.get('request')!=request or set(provenance.get('runtime',{}))!={'qwen','harrier','gte'}:raise ValueError('public benchmark runtime/request mismatch')
    else:
        if provenance.get('source_mode')!='explicit certified legacy cache import' or metadata.get('binding',{}).get('cache_index_sha256')!=reference['metadata']['cache_index_sha256']:raise ValueError('legacy benchmark certified cache index mismatch')
    for name,spec in specs.items():
        stage=verify(folder/name);record=read_json(folder/name/'task.json');prov=read_json(folder/name/'provenance.json')
        if record.get('task')!=name or record.get('kind')!=spec['settings']['kind'] or record.get('source')!=spec['settings']:raise ValueError('benchmark task identity/source mismatch')
        registry={k:v for k,v in record.get('registry',{}).items() if k!='encoding_role_config'}
        if registry!=spec['registry']:raise ValueError('benchmark registry/protocol mismatch')
        check_reference(record['roles'],identities[name])
        for role,rows in record['roles'].items():
            if any(r.get('canonical_text_sha256')!=canonical_text_sha256(r['text']) for r in rows):raise ValueError('benchmark canonical hash mismatch')
        for key in ['qrels','labels','pairs']:
            if record.get(key)!=spec.get(key):raise ValueError('benchmark labels/qrels/pairs protocol mismatch')
        if public:
            check_public_request(stage['metadata'],request)
            if prov.get('request')!=request or prov.get('runtime')!=provenance['runtime'] or prov.get('task_role_policies')!=spec['policies']:raise ValueError('public task policy/runtime mismatch')
            for role,policies in spec['policies'].items():
                for model,expected in policies.items():
                    runtime=prov['runtime'][model];p=runtime['policy'];backend=request['qwen_attention'] if model=='qwen' else 'sdpa'
                    if p['actual_attention_backend']!=backend or p['actual_dtype']!=request['dtype_policy'][model] or p['versions']!=request['versions']:raise ValueError('public actual backend/dtype/version mismatch')
                    simulated=SimpleNamespace(name=model,spec=dict(model_id=p['model_id'],revision=p['revision'],length=p['max_seq_length'],dimension=p['native_dim']),policy=p);check_encoding_policy(simulated,expected)
                    audit=prov['center_audits'][role+'__'+model]
                    if audit['center_checks']!=len(record['roles'][role]) or not np.isfinite(audit['max_encoder_center_abs_error']) or audit['max_encoder_center_abs_error']>(1e-3 if model=='qwen' else 1e-5):raise ValueError('public native center audit mismatch')
        else:
            source_sha=reference['metadata']['source_manifests'][name]
            if stage['metadata'].get('source_manifest_sha256')!=source_sha or stage['metadata'].get('cache_index_sha256')!=reference['metadata']['cache_index_sha256'] or prov.get('source_manifest_sha256')!=source_sha:raise ValueError('legacy benchmark source manifest mismatch')
            expected_policies={model:{role:values[model] for role,values in spec['policies'].items()} for model in ['qwen','harrier','gte']}
            if prov.get('cache_policies')!=expected_policies:raise ValueError('legacy benchmark model-role policy mismatch')
        with np.load(folder/name/'native.npz',allow_pickle=False) as arrays:
            expected_keys={role+'__'+model for role in spec['policies'] for model in ['qwen','harrier','gte']}
            if set(arrays.files)!=expected_keys:raise ValueError('benchmark model/role array schema mismatch')
            for role,policies in spec['policies'].items():
                for model,policy in policies.items():
                    array=arrays[role+'__'+model]
                    if array.dtype!=np.float64 or array.shape!=(len(record['roles'][role]),policy['native_dim']) or not np.isfinite(array).all() or np.max(np.abs(np.linalg.norm(array,axis=1)-1))>1e-5:raise ValueError('benchmark native shape/dtype/finite/unit mismatch')
    return dict(metadata,validated=True,mode='public' if public else 'legacy',reference_sha256=sha(REFERENCE/'manifest.json'))


def prepare_public_benchmarks(output,cache,device='cuda:0',offline=False,qwen_attention='sdpa',physical_batch=8):
    """Build the six public benchmark inputs automatically without old caches."""
    import gc
    import os
    from standalone.pipeline import environment,preparation_lock
    output=Path(output).resolve();cache=Path(cache).resolve();os.environ.update(environment(output,cache,offline))
    request=public_request(cache,device,offline,qwen_attention,physical_batch)
    if (output/'manifest.json').exists():
        saved=verify(output);check_public_request(saved['metadata'],request);validate_benchmark_suite(output)
        return saved
    with preparation_lock(output):
        source=output/'public_source';prepare_public_sources(source,cache,offline)
        from standalone.models import ModelPath
        protocol=read_json(REFERENCE/'protocol.json');reference_sha=sha(REFERENCE/'manifest.json');tasks=list(protocol['tasks']);runtime={}
        for model_name in ['qwen','harrier','gte']:
            model=ModelPath(model_name,cache,device,offline,qwen_attention)
            runtime[model_name]=dict(policy=model.policy,asset_hashes=model.asset_hashes,encoder_hash=model.initial_hash)
            for task,spec in protocol['tasks'].items():
                record=read_json(source/task/'task.json')
                for role,rows in record['roles'].items():
                    expected=spec['policies'][role][model_name]
                    check_encoding_policy(model,expected)
                    stage=output/'.encoding'/model_name/task/role
                    binding=dict(mode='public',request=request,source_sha256=sha(source/task/'task.json'),reference_sha256=reference_sha,model=runtime[model_name],prompt=expected['prompt'],row_ids_sha256=fingerprint([r['row_id'] for r in rows]))
                    if (stage/'manifest.json').exists():
                        saved=verify(stage)
                        if saved['metadata']!=binding:raise ValueError('completed public mean encoding configuration changed')
                    else:
                        values,audit=encode_mean_rows(model,rows,expected['prompt'],physical_batch)
                        stage.mkdir(parents=True,exist_ok=True);np.save(stage/'mean.npy',values);np.save(stage/'mean_ids.npy',np.array([r['row_id'] for r in rows]));write_json(stage/'audit.json',audit)
                        seal(stage,['mean.npy','mean_ids.npy','audit.json'],binding)
                    print('PUBLIC NATIVE',model_name,task,role,flush=True)
            model.verify_unchanged();del model;gc.collect()
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        files=[]
        for task,spec in protocol['tasks'].items():
            target=output/task;record=read_json(source/task/'task.json');arrays={};audits={}
            if (target/'manifest.json').exists():
                previous=verify(target);check_public_request(previous['metadata'],request)
                if read_json(target/'provenance.json').get('runtime')!=runtime:raise ValueError('public task runtime changed before promotion')
            else:
                for role,rows in record['roles'].items():
                    for model_name in ['qwen','harrier','gte']:
                        stage=output/'.encoding'/model_name/task/role;encoded=verify(stage)
                        expected_binding=dict(mode='public',request=request,source_sha256=sha(source/task/'task.json'),reference_sha256=reference_sha,model=runtime[model_name],prompt=spec['policies'][role][model_name]['prompt'],row_ids_sha256=fingerprint([r['row_id'] for r in rows]))
                        if encoded['metadata']!=expected_binding:raise ValueError('public role source/policy/runtime changed before promotion')
                        ids=np.load(stage/'mean_ids.npy',allow_pickle=False)
                        if not np.array_equal(ids,np.array([r['row_id'] for r in rows])):raise ValueError('independent public native row IDs mismatch')
                        arrays[f'{role}__{model_name}']=np.load(stage/'mean.npy',allow_pickle=False);audits[f'{role}__{model_name}']=read_json(stage/'audit.json')
                write_json(target/'task.json',record);np.savez(target/'native.npz',**arrays)
                write_json(target/'provenance.json',dict(mode='public',request=request,source_mode='pinned public-source reconstruction and current mean-only encoding',public_source_manifest_sha256=sha(source/task/'manifest.json'),reference_sha256=reference_sha,runtime=runtime,task_role_policies=spec['policies'],center_audits=audits,encoding_comparability=dict(bit_identical_to_legacy_caches=False,known_difference='Current reconstructed FP64 final path; original Qwen cache used flash_attention_2 and FP16 output normalization. Current runtime attention backend recorded in runtime.'),scope='six-task exploratory pilot; not full MTEB',uses_old_alignment_maps=False))
                seal(target,['task.json','native.npz','provenance.json'],dict(task=task,reference_sha256=reference_sha,mode='public',request=request))
            files.extend(f'{task}/{n}' for n in ['task.json','native.npz','provenance.json','manifest.json'])
        write_json(output/'provenance.json',dict(mode='public',request=request,source_mode='pinned public-source reconstruction and current mean-only encoding',runtime=runtime,reference_sha256=reference_sha,scope='six-task exploratory pilot; not full MTEB',requires_encoder_inference=True,variance_computed=False,uses_old_alignment_maps=False))
        saved=seal(output,files+['provenance.json'],dict(tasks=tasks,schema_version=1,mode='public',request=request,scope='six-task exploratory pilot; not full MTEB',reference_sha256=reference_sha))
        validate_benchmark_suite(output);return saved


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    old=sub.add_parser('legacy-import',help='Export explicit certified original native caches with row/policy checks')
    for option in ['cache-index','registry','data-dir','output']:old.add_argument('--'+option,type=Path,required=True)
    old.add_argument('--fit-work',type=Path)
    public=sub.add_parser('public',help='Download pinned public sources and encode current native means')
    public.add_argument('--output',type=Path,required=True);public.add_argument('--cache-dir',type=Path,required=True)
    public.add_argument('--device',default='cuda:0');public.add_argument('--offline',action='store_true');public.add_argument('--qwen-attention',choices=['sdpa','flash_attention_2'],default='sdpa');public.add_argument('--physical-batch',type=int,default=8)
    sources=sub.add_parser('public-sources',help='Reconstruct/check only pinned benchmark text, without model inference')
    sources.add_argument('--output',type=Path,required=True);sources.add_argument('--cache-dir',type=Path,required=True);sources.add_argument('--offline',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='legacy-import':export_legacy(args.cache_index,args.registry,args.data_dir,args.output,args.fit_work)
    elif args.command=='public-sources':prepare_public_sources(args.output,args.cache_dir,args.offline)
    elif args.command=='public':prepare_public_benchmarks(args.output,args.cache_dir,args.device,args.offline,args.qwen_attention,args.physical_batch)


if __name__=='__main__':main()
