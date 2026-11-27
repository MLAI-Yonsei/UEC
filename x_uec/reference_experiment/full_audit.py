"""Independent full-coverage replay; preserves immutable cache-producer files."""
from collections import Counter
import json
from pathlib import Path
from e4_data import (TASKS,checked_evaluation_files,canonical_hash,expand_observations,
    group_split,select_scope,partition_summary)
from run_covariance_validation import sha

def snapshot_paths(paths):
    return {str(Path(p).resolve()):sha(p) for p in paths}

def verify_snapshot(snapshot):
    for path,expected in snapshot.items():
        if sha(path)!=expected:raise ValueError('source/input changed during full run '+path)

def require_consumed_paths(provenance):
    consumed={str(Path(p).resolve()):v for p,v in provenance['consumed_files'].items()}
    for key in ['input_path','observations_path','split_path']:
        path=Path(provenance[key]).resolve()
        if str(path) not in consumed or sha(path)!=consumed[str(path)]:raise ValueError(key+' not checksum-bound to consumed_files')

def replay_evaluation(consumed):
    records=[];hashes=set()
    for task in TASKS:
        candidates=[Path(p) for p in consumed if Path(p).name=='manifest.json' and Path(p).parent.name==task]
        if len(candidates)!=1:raise ValueError('missing/ambiguous evaluation manifest '+task)
        for file in checked_evaluation_files(candidates[0].parent,task):
            if str(file) not in consumed or sha(file)!=consumed[str(file)]:raise ValueError('evaluation source not checksum-bound')
            with file.open() as stream:rows=[json.loads(line) for line in stream]
            hashes.update(canonical_hash(row['text']) for row in rows)
            records.append(dict(task=task,file=str(file),rows=len(rows)))
    return hashes,records

def verify_full_partition(all_rows,evaluation_hashes,record,selected_rows):
    replay=group_split(all_rows,evaluation_hashes)
    if record['full']!=replay:raise ValueError('replayed full partition differs from supplied split')
    selected=select_scope(all_rows,replay,'full')
    if record['selected_scope']!=selected or selected_rows!=[all_rows[i] for i in selected['selected_indices']]:raise ValueError('replayed full selection differs from cache observations')
    return replay

def audit_full_inputs(data):
    p=data.provenance;require_consumed_paths(p);consumed=p['consumed_files']
    candidates=[Path(f) for f in consumed if Path(f).name=='triplets.jsonl']
    if len(candidates)!=1:raise ValueError('missing/ambiguous original fitting triplets')
    folder=candidates[0].parent;seal=json.loads((folder/'manifest.json').read_text())
    for name in ['manifest.json','triplets.jsonl','selection.json','dataset_sources.json']:
        file=folder/name
        if str(file) not in consumed or sha(file)!=consumed[str(file)]:raise ValueError('original fitting source not checksum-bound')
        if name!='manifest.json' and sha(file)!=seal['files'][name]:raise ValueError('original fitting manifest mismatch')
    with (folder/'triplets.jsonl').open() as stream:pairs=[json.loads(line) for line in stream]
    counts=Counter(row['source'] for row in pairs)
    if counts!=dict(ms_marco=3983,snli=3775):raise ValueError('full fitting selection counts changed')
    sources=json.loads((folder/'dataset_sources.json').read_text())
    all_rows=expand_observations(pairs,sources,sha(folder/'selection.json'))
    hashes,scope=replay_evaluation(consumed)
    record=json.loads(Path(p['split_path']).read_text());replay=verify_full_partition(all_rows,hashes,record,data.observations)
    return dict(status='passed',evaluation_scope=scope,evaluation_unique_hashes=len(hashes),
        partition_counts={part:partition_summary(all_rows,replay[part+'_indices']) for part in ['fit','diagnostic','excluded']},
        original_pairs=counts,original_observations=len(all_rows),
        prepared_evaluation_scope_was_mislabeled=record.get('evaluation_scope')!=scope,
        note='Independent replay is authoritative; worker-pinned preparation split remains byte-identical.')
