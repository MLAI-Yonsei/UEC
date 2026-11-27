"""Pinned public assets and published reference IDs; no private workspace."""
import json
from pathlib import Path
from .storage import verify,sha,read_jsonl,write_json,write_jsonl,seal
from .records import reconstruct_pairs,observations,split_rows,canonical_hash,main_sample_indices,main_pairs

REFERENCE=Path(__file__).with_name('reference')

def download_datasets(cache,offline=False):
    from huggingface_hub import hf_hub_download
    verify(REFERENCE);sources=json.loads((REFERENCE/'dataset_sources.json').read_text());paths={}
    for name,p in sources.items():
        file=Path(hf_hub_download(p['repo'],p['filename'],repo_type='dataset',revision=p['revision'],cache_dir=str(cache),local_files_only=offline))
        if file.stat().st_size!=p['size'] or sha(file)!=p['sha256']:raise ValueError('public dataset checksum mismatch: '+name)
        paths[name]=file
    return paths

def resolve_reference(paths):
    import pyarrow as pa
    import pyarrow.parquet as pq
    selection=read_jsonl(REFERENCE/'selection_ids.jsonl');sources={};tables={}
    for name,path in paths.items():
        columns=['query','query_id','passages'] if name=='ms_marco' else ['premise','hypothesis','label']
        tables[name]=pq.read_table(path,columns=columns)
    sampled=main_sample_indices(len(tables['ms_marco']),len(tables['snli']))
    for name,path in paths.items():
        indices=sampled[name];selected=tables[name].take(pa.array(indices)).to_pylist()
        sources[name]=dict(zip(indices,selected))
    pairs=main_pairs(sources,sampled)
    if len(pairs)!=len(selection) or any({k:r.get(k) for k in original}!=original for r,original in zip(pairs,selection)):
        raise ValueError('UEC main sampling replay differs from published reference IDs')
    return pairs

def prepare_data(work,profile,cache,offline=False,evaluation_texts=()):
    work=Path(work);folder=work/'data'
    if (folder/'manifest.json').exists():
        record=verify(folder)
        if record['metadata']['profile']!=profile:raise ValueError('data profile changed')
        verify(REFERENCE)
        if record['metadata']['reference_manifest_sha256']!=sha(REFERENCE/'manifest.json'):raise ValueError('published reference changed')
        return record
    folder.mkdir(parents=True,exist_ok=True);verify(REFERENCE)
    paths=download_datasets(cache,offline);pairs=resolve_reference(paths)
    if profile=='smoke':pairs=[r for source in ['ms_marco','snli'] for r in [r for r in pairs if r['source']==source][:32]]
    elif profile!='full':raise ValueError('profile must be full or smoke')
    inventory=json.loads((REFERENCE/'evaluation_hashes.json').read_text());historical=set(inventory['hashes']);hashes=set();extra=[]
    for path in evaluation_texts:
        rows=read_jsonl(path);hashes.update(canonical_hash(r['text']) for r in rows)
        extra.append(dict(file=Path(path).name,sha256=sha(path),rows=len(rows)))
    all_rows=observations(pairs);split=split_rows(all_rows,hashes);rows=[all_rows[i] for i in split['selected_indices']]
    curvature_pairs=[r for r in pairs if canonical_hash(r['query']) not in hashes and canonical_hash(r['positive']) not in hashes]
    write_jsonl(folder/'pairs.jsonl',pairs);write_jsonl(folder/'curvature_pairs.jsonl',curvature_pairs)
    write_jsonl(folder/'observations_all.jsonl',all_rows);write_jsonl(folder/'observations.jsonl',rows)
    write_json(folder/'split.json',split);write_json(folder/'evaluation.json',dict(hashes=sorted(hashes),
        scope='within UEC main selected training data; external exclusions only when explicitly supplied',additional_scope=extra,
        historical_inventory_scope=inventory['scope'],historical_inventory_unique_hashes=len(historical),
        historical_overlap_observations=sum(r['canonical_text_sha256'] in historical for r in all_rows),
        historical_overlap_excluded_by_default=False))
    write_json(folder/'dataset_sources.json',json.loads((REFERENCE/'dataset_sources.json').read_text()))
    return seal(folder,['pairs.jsonl','curvature_pairs.jsonl','observations_all.jsonl','observations.jsonl','split.json','evaluation.json','dataset_sources.json'],
        dict(profile=profile,pairs=len(pairs),observations=len(rows),source_selection='UEC main default training selection reproduced; seed2025, MS5% then SNLI2%, sample before filter',
             curvature_pairs=len(curvature_pairs),
             reference_manifest_sha256=sha(REFERENCE/'manifest.json')))
