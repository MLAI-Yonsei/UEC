"""Pinned fitting observations and leakage-free smoke selection for v3."""
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
from run_covariance_validation import sha,write_json,ORDER

DIMS=dict(qwen=1024,harrier=640,gte=768)
TASKS=('LegalBenchCorporateLobbying','SCIDOCS','PoemSentimentClassification','TweetTopicSingleClassification','STSBenchmark','STS22.v2')
PROPAGATION_SOURCES=('caia_svd_pilot_v3/uncertainty.py','caia_svd_pilot_v3/sensitivity_math.py',
    'caia_svd_pilot_v2/sensitivity_math.py','caia_pair_pilot/qwen_math.py','caia_pair_pilot/qwen_adapter.py',
    'caia_svd_pilot_v3/harrier_adapter.py','caia_svd_pilot_v3/config.py','gte_la/model.py','gte_la/core.py',
    'uec_experiment/reproduce_qwen.py','uec_experiment/gpa_experiment.py')

def verify_source_checksums(consumed):
    for file,expected in consumed.items():
        if sha(file)!=expected:raise ValueError('source checksum mismatch '+file)

def verify_parent_sources(parent,root,relative_paths=PROPAGATION_SOURCES):
    parent,root=Path(parent),Path(root)
    seal=json.loads((parent/'manifest.json').read_text());record=parent/'input_hashes.json'
    if sha(record)!=seal['files']['input_hashes.json']:raise ValueError('parent source evidence changed')
    inputs=json.loads(record.read_text())['inputs'];consumed={str(record):sha(record),str(parent/'manifest.json'):sha(parent/'manifest.json')}
    for relative in relative_paths:
        file=root/relative;actual=sha(file)
        if actual!=inputs[str(file)]:raise ValueError('live propagation source differs from original v3 '+relative)
        consumed[str(file)]=actual
    return consumed

def checked_evaluation_files(folder,task):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if task in ('LegalBenchCorporateLobbying','SCIDOCS'):required=['corpus.jsonl','queries.jsonl']
    elif task in ('PoemSentimentClassification','TweetTopicSingleClassification'):required=['support.jsonl','test.jsonl']
    else:required=['texts.jsonl']
    files=[]
    for name in required:
        file=folder/name
        if name not in manifest['files'] or not file.exists():raise FileNotFoundError('required evaluation file '+str(file))
        if sha(file)!=manifest['files'][name]:raise ValueError('evaluation source checksum mismatch')
        files.append(file)
    return files

def canonical_hash(text):
    text=unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def expand_observations(pairs,sources,selection_hash):
    result=[]
    for pi,pair in enumerate(pairs):
        source=pair['source'];roles=('query','positive_passage') if source=='ms_marco' else ('premise','hypothesis')
        for field,role,side in zip(('query','positive'),roles,('query','document')):
            text=pair[field]
            result.append(dict(observation_id=f"{source}:{pair['triplet_id']}:{role}:0",dataset=source,
                dataset_revision=sources[source]['revision'],source_split=sources[source]['split'],
                source_record_id=pair['source_row'],source_order=len(result),pair_index=pi,
                original_triplet_id=pair['triplet_id'],role=role,moment_side=side,text=text,
                canonical_text_sha256=canonical_hash(text),selection_manifest_sha256=selection_hash))
    if len({r['observation_id'] for r in result})!=len(result):raise ValueError('duplicate observation ID')
    return result

def group_split(observations,evaluation_hashes):
    excluded=[i for i,r in enumerate(observations) if r['canonical_text_sha256'] in evaluation_hashes]
    eligible=[i for i,r in enumerate(observations) if r['canonical_text_sha256'] not in evaluation_hashes]
    groups=sorted({observations[i]['canonical_text_sha256'] for i in eligible})
    shuffled=np.random.default_rng(42).permutation(len(groups));nfit=int(np.floor(.8*len(groups)))
    if not 0<nfit<len(groups):raise ValueError('empty fit/diagnostic group split')
    fit_groups={groups[i] for i in shuffled[:nfit]}
    fit=[i for i in eligible if observations[i]['canonical_text_sha256'] in fit_groups]
    diagnostic=[i for i in eligible if observations[i]['canonical_text_sha256'] not in fit_groups]
    return dict(fit_indices=fit,diagnostic_indices=diagnostic,excluded_indices=excluded,
        seed=42,group_rule='sorted unique NFC/newline canonical hashes; shuffled groups, floor(0.8*G)',
        groups=len(groups),fit_groups=nfit,diagnostic_groups=len(groups)-nfit,
        alignment_only_heldout=True,posterior_unseen=False,duplicate_policy='preserve_occurrences')

def select_smoke(observations,split,fit_max=2048,diagnostic_max=128):
    rng=np.random.default_rng(42);fit=sorted(rng.choice(split['fit_indices'],min(fit_max,len(split['fit_indices'])),replace=False).tolist())
    chosen=[];hashes=set();query=[];document=[]
    for side,count,target in [('query',2,query),('document',4,document)]:
        for i in split['diagnostic_indices']:
            row=observations[i];h=row['canonical_text_sha256']
            if row['moment_side']==side and h not in hashes:
                target.append(i);hashes.add(h);chosen.append(i)
                if len(target)==count:break
        if len(target)<count:raise ValueError('insufficient distinct diagnostic '+side+' observations')
    rest=[i for i in split['diagnostic_indices'] if i not in chosen]
    remaining=min(diagnostic_max-len(chosen),len(rest))
    if remaining<0:raise ValueError('diagnostic limit below six')
    chosen+=np.random.default_rng(42).choice(rest,remaining,replace=False).tolist()
    return dict(fit_indices=fit,diagnostic_indices=sorted(chosen),moment_query_indices=query,moment_document_indices=document,
        selected_indices=sorted(fit+chosen),warnings=['fit_below_1024'] if len(fit)<1024 else [],
        fit_rows_max=fit_max,diagnostic_rows_max=diagnostic_max)

def select_scope(observations,split,mode):
    if mode=='smoke':return select_smoke(observations,split)
    if mode!='full':raise ValueError('mode must be smoke or full')
    moments=select_smoke(observations,split)
    return dict(fit_indices=list(split['fit_indices']),diagnostic_indices=list(split['diagnostic_indices']),
        moment_query_indices=moments['moment_query_indices'],moment_document_indices=moments['moment_document_indices'],
        selected_indices=sorted(split['fit_indices']+split['diagnostic_indices']),
        warnings=['fit_below_1024'] if len(split['fit_indices'])<1024 else [],mode='full')

def partition_summary(observations,indices):
    rows=[observations[i] for i in indices];groups=Counter(r['canonical_text_sha256'] for r in rows)
    return dict(observations=len(rows),unique_text_groups=len(groups),duplicate_occurrences=len(rows)-len(groups),
        maximum_text_multiplicity=max(groups.values(),default=0),datasets=Counter(r['dataset'] for r in rows),
        roles=Counter(r['role'] for r in rows),dataset_roles=Counter(r['dataset']+':'+r['role'] for r in rows))

def validate_cache_arrays(mean,variance,mean_ids,variance_ids,expected_ids,dimension):
    if mean.dtype!=np.float64 or variance.dtype!=np.float64:raise ValueError('cache dtype must be float64')
    if mean.shape!=(len(expected_ids),dimension) or variance.shape!=mean.shape:raise ValueError('native output shape')
    if any(np.asarray(a).dtype.kind!='U' for a in (mean_ids,variance_ids,expected_ids)):raise ValueError('Unicode IDs required')
    if not np.array_equal(mean_ids,expected_ids) or not np.array_equal(variance_ids,expected_ids):raise ValueError('independent ID order mismatch')
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<0):raise ValueError('invalid mean/variance values')
    if np.max(np.abs(np.linalg.norm(mean,axis=1)-1))>2e-3:raise ValueError('not final unit output stage')

def prepare_data(root,output,mode='smoke'):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source=root/'runs/gte_la_validation_20261124/v1/data';seal=json.loads((source/'manifest.json').read_text());consumed={}
    def consume(path,expected=None):
        actual=sha(path)
        if expected and actual!=expected:raise ValueError('source checksum mismatch '+str(path))
        consumed[str(path)]=actual;return path
    consume(source/'manifest.json')
    for filename in ['triplets.jsonl','selection.json','dataset_sources.json']:consume(source/filename,seal['files'][filename])
    pairs=[json.loads(s) for s in (source/'triplets.jsonl').read_text().splitlines()]
    counts=Counter(r['source'] for r in pairs)
    if dict(counts)!=dict(ms_marco=3983,snli=3775):raise ValueError('local selection changed')
    sources=json.loads((source/'dataset_sources.json').read_text())
    observations=expand_observations(pairs,sources,sha(source/'selection.json'))
    eval_hashes=set();scope=[]
    for task in TASKS:
        folder=root/'runs/cross_predictive_partial_pilot6_20261119/data'/task
        manifest=json.loads(consume(folder/'manifest.json').read_text())
        for file in checked_evaluation_files(folder,task):
            consume(file,manifest['files'][file.name]);n=0
            with file.open() as f:
                for line in f:
                    r=json.loads(line);eval_hashes.add(canonical_hash(r['text']));n+=1
            scope.append(dict(task=task,file=str(file),rows=n))
    split=group_split(observations,eval_hashes);smoke=select_smoke(observations,split);scope=select_scope(observations,split,mode)
    (output/'observations_all.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in observations))
    selected=[observations[i] for i in scope['selected_indices']]
    (output/'observations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    write_json(output/'split.json',dict(full=split,smoke=smoke,selected_scope=scope,selected_observation_ids=[r['observation_id'] for r in selected],
        partition_counts={part:partition_summary(observations,split[part+'_indices']) for part in ['fit','diagnostic','excluded']},
        counts=dict(pairs=counts,observations=len(observations),selected=len(selected),
            roles=Counter(r['role'] for r in selected),datasets=Counter(r['dataset'] for r in selected)),
        evaluation_scope=scope,evaluation_unique_hashes=len(eval_hashes),canonicalization='Unicode NFC + CRLF/CR to LF only',
        evaluation_overlaps_removed=len(split['excluded_indices']),fit_diagnostic_hash_overlap=0))
    write_json(output/'source_manifest.json',dict(consumed_files=consumed,source_counts=counts,cache_scope='alignment_full' if mode=='full' else 'smoke_only',
        uncertainty_source='frozen_teacher_v3_reused',positive_endpoints_only=True,negatives_preserved_in_original_triplets=True,
        encoder_policy='original teacher fitting policies; query-style prompts retained for both actual endpoint roles',
        delivery_version='v3',parent_gaussian_run=str(root/'runs/caia_svd_frozen_triple_20261124/v3')))
    return dict(output=str(output),observations=len(observations),cache_rows=len(selected),excluded=len(split['excluded_indices']))

def assemble_cache(folder):
    folder=Path(folder).resolve();observations=[json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()]
    ids=np.array([r['observation_id'] for r in observations],dtype=str);payload={};models={};consumed={}
    source=json.loads((folder/'source_manifest.json').read_text());verify_source_checksums(source['consumed_files'])
    consumed.update(source['consumed_files'])
    parent=Path(source['parent_gaussian_run']);root=parent.parents[2]
    # Clearly retrospective: these live dependency hashes are checked after GPU
    # generation against the immutable original v3 source evidence. Original
    # producer summaries are preserved; no claim of pre-execution sealing.
    post_sources=verify_parent_sources(parent,root);consumed.update(post_sources)
    data_root=root/'runs/gte_la_validation_20261124/v1/data';parent_seal=json.loads((parent/'manifest.json').read_text())
    if sha(data_root/'manifest.json')!=parent_seal['inputs'][str(data_root/'manifest.json')]:raise ValueError('fitting selection differs from original v3 posterior data')
    pairs=[json.loads(s) for s in (data_root/'triplets.jsonl').read_text().splitlines()]
    sources=json.loads((data_root/'dataset_sources.json').read_text())
    all_rows=expand_observations(pairs,sources,sha(data_root/'selection.json'))
    stored_all=[json.loads(s) for s in (folder/'observations_all.jsonl').read_text().splitlines()]
    split=json.loads((folder/'split.json').read_text())
    scope=split.get('selected_scope',split['smoke'])
    if all_rows!=stored_all or observations!=[all_rows[i] for i in scope['selected_indices']]:raise ValueError('observations not bound to original fitting endpoints')
    for file in ['observations_all.jsonl','observations.jsonl','split.json','source_manifest.json']:consumed[str(folder/file)]=sha(folder/file)
    for m in ORDER:
        dest=folder/m;summary=json.loads((dest/'summary.json').read_text())
        if summary['status']!='completed':raise ValueError('incomplete '+m)
        arrays={k:np.load(dest/(k+'.npy'),allow_pickle=False) for k in ['mean','variance','mean_ids','variance_ids']}
        validate_cache_arrays(arrays['mean'],arrays['variance'],arrays['mean_ids'],arrays['variance_ids'],ids,DIMS[m])
        for k,v in arrays.items():
            file=dest/(k+'.npy')
            if sha(file)!=summary['output_checksums'][str(file)]:raise ValueError('output cache changed')
            payload[k+'__'+m]=v;consumed[str(file)]=sha(file)
        consumed[str(dest/'summary.json')]=sha(dest/'summary.json');models[m]=summary
        for file,expected in summary['consumed_files'].items():
            if sha(file)!=expected:raise ValueError('cache producer input changed')
            consumed[file]=expected
    npz=folder/'uec_alignment_inputs.npz'
    if npz.exists():raise FileExistsError(npz)
    np.savez(npz,**payload);consumed[str(npz)]=sha(npz)
    backend=Path(__file__).parent
    provenance=dict(delivery_version='v3',cache_scope=source['cache_scope'],models=models,consumed_files=consumed,
        uncertainty_source='frozen_teacher_v3_reused',output_stage='l2_normalized',covariance_representation='native_output_diagonal_surrogate',
        mean_variance_same_final_stage=True,raw_mean_definition='unchanged frozen final-block FP64 local linearization center, L2-normalized; original encoder comparison recorded',
        backend_files={f:sha(backend/f) for f in ['caia.py','covariance_ops.py','run_covariance_validation.py']},
        propagation_source_verification=dict(timing='post_execution_checked_against_original_v3_input_seal',files=post_sources),
        input_path=str(npz),observations_path=str(folder/'observations.jsonl'),split_path=str(folder/'split.json'),
        posterior_refit_performed=False,encoder_weights_updated=False,inference_executed_in_preparation=True)
    write_json(folder/'uec_input_provenance.json',provenance)
    return str(folder/'uec_input_provenance.json')
