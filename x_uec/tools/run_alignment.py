"""Run v3 numerical diagnostics, then original/mapped/uniform performance."""
import argparse,csv,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(TOOLS))
from standalone.records import canonical_hash
from standalone.storage import sha,read_jsonl,write_json,seal,verify

def diagnostic_task(data,diagnostic_indices,source):
    """Held-out sentence groups, preserving original positive-pair relations."""
    rows=data.observations;indices=set(map(int,diagnostic_indices));unique={'query':{},'document':{}};positive={};pairs={}
    for i,row in enumerate(rows):
        if row['dataset']!=source:continue
        pairs.setdefault(row['original_triplet_id'],{})[row['moment_side']]=row
        if i in indices:unique[row['moment_side']].setdefault(row['canonical_text_sha256'],i)
    for sides in pairs.values():
        if set(sides)!={'query','document'}:continue
        positive.setdefault(sides['query']['canonical_text_sha256'],set()).add(sides['document']['canonical_text_sha256'])
    docs=unique['document'];queries={h:i for h,i in unique['query'].items() if positive.get(h,set())&docs.keys()}
    if not queries or not docs:raise ValueError('no diagnostic queries with a held-out known positive')
    ix={'queries':list(queries.values()),'corpus':list(docs.values())};roles={}
    for role,items in ix.items():roles[role]=[dict(rows[i],row_id=rows[i]['observation_id']) for i in items]
    docids={h:rows[i]['observation_id'] for h,i in docs.items()}
    qrels={rows[i]['observation_id']:{docids[h]:1 for h in positive[qh]&docs.keys()} for qh,i in queries.items()}
    task=dict(task=source+'_diagnostic_pair_matching',kind='Retrieval',roles=roles,qrels=qrels)
    native={role:{m:a[items] for m,a in data.means.items()} for role,items in ix.items()}
    audit=dict(source=source,queries=len(queries),corpus=len(docs),known_positive_relations=sum(map(len,qrels.values())),
        dropped_queries_without_diagnostic_positive=len(unique['query'])-len(queries),canonical_text_deduplicated=True,
        alignment_only_heldout=True,posterior_unseen=False,benchmark_claim=False,
        note='Positive-pair matching among diagnostic candidates; SNLI entailment-only, not NLI classification or STS.')
    return task,native,audit

def verify_alignment_input_binding(inputs,receipt):
    inputs=Path(inputs)
    for name in ['alignment_fit.npz','uec_input_provenance.json']:
        candidates=[value for path,value in receipt['input_files'].items() if Path(path).name==name]
        if len(candidates)!=1 or sha(inputs/name)!=candidates[0]:raise ValueError('scored inputs differ from the fitted input '+name)
    # The original stage manifest is additionally matched when recorded. The
    # provenance and NPZ hashes above allow a byte-identical preparation to move.
    candidates=[value for path,value in receipt['input_files'].items() if Path(path).name=='manifest.json' and Path(path).parent.name=='inputs']
    if len(candidates)!=1 or sha(inputs/'manifest.json')!=candidates[0]:raise ValueError('scored input manifest differs from the fitted input')

def load_approved_maps(alignment):
    alignment=Path(alignment);status=json.loads((alignment/'status.json').read_text())
    if status['status'] not in ['alignment_ready_for_handoff','review_required','smoke_pass']:raise ValueError('alignment has no usable completed maps')
    folder=alignment/('maps' if status['alignment_ready_for_handoff'] else 'diagnostic_only')
    receipt=json.loads((alignment/'handoff_manifest.json').read_text());expected={Path(k).name:v for k,v in receipt['maps'].items()}
    if any(status[k]!=receipt[k] for k in ['status','alignment_ready_for_handoff','delivery_version','mode']):raise ValueError('alignment status differs from handoff')
    maps={};checks={}
    for arm in ['gpa','caia']:
        file=folder/(arm+'_maps.npz')
        if sha(file)!=expected[file.name]:raise ValueError('alignment map checksum changed')
        with np.load(file,allow_pickle=False) as z:maps[arm]={m:z[m] for m in ['qwen','harrier','gte']}
        checks[str(file)]=sha(file)
    checks.update({str(alignment/name):sha(alignment/name) for name in ['status.json','handoff_manifest.json']})
    return maps,status,checks

def write_report(folder,metrics,metadata):
    rows={}
    for r in metrics:rows.setdefault((r['scope'],r['task']),{})[r['condition']]=r['score100']
    conditions=['native_qwen','gpa_qwen','caia_qwen','native_harrier','gpa_harrier','caia_harrier','native_gte','gpa_gte','caia_gte','uniform_gpa','uniform_caia']
    lines=['# 변환 전후 원본 모델과 균등 결합 성능','',
        '동일 평가 문장·정답·분류 support 추출을 모든 조건에 사용했습니다. 균등 결합은 각 모델의 정렬된 평균을1/3씩 더한 뒤 L2 정규화한 cosine입니다. 불확실성 가중 UEC 최종 점수와 구분합니다.','',
        'internal은 UEC 선택 자료의 정렬 진단 분할에서 수행한 positive-pair matching입니다. 불확실성 피팅에는 쓰인 자료이며, SNLI 분류/STS 성능이 아닙니다. external은 별도6태스크 pilot이며 전체 MMTEB가 아닙니다. 피팅에 사용한 문장과 겹치는 평가 항목은 제외했습니다.','',
        '| 범위 / 과제 | '+' | '.join(conditions)+' |','|---|'+'---:|'*len(conditions)]
    for (scope,task),values in rows.items():lines.append('| '+scope+' / '+task+' | '+' | '.join(f'{values[c]:.6f}' if c in values else '미실행' for c in conditions)+' |')
    lines+=['','점수는 각 과제의 주 지표×100입니다. 검색 NDCG@10, 분류 accuracy, STS Spearman입니다. 원본 모델의 변환 전후 cosine 보존과 분류 head 운반 오차는 각 과제의 diagnostics.json에서 별도로 확인합니다.','',
        '분류는 같은 support와 같은 solver로 다시 피팅하므로 작은 solver 오차가 있을 수 있습니다. 원본 head를 함께 운반했을 때의 logit/accuracy도 진단해 임베딩 변환과 probe 재피팅 효과를 구분합니다.','']
    (folder/'report.md').write_text('\n'.join(lines));write_json(folder/'metrics.json',metrics)
    with (folder/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in metrics for k in row}));writer.writeheader();writer.writerows(metrics)
    write_json(folder/'evaluation_provenance.json',metadata)

def evaluate(inputs,alignment,benchmarks=None,performance_output=None):
    from performance_core import evaluate_task,load_task
    from standalone.alignment import load_inputs,audit_and_split
    sys.path.append(str(ROOT/'reference_experiment'))
    inputs=Path(inputs).resolve();alignment=Path(alignment).resolve();folder=Path(performance_output).resolve() if performance_output else alignment/'performance';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        maps,status,map_checks=load_approved_maps(alignment);source={str(inputs/'manifest.json'):sha(inputs/'manifest.json'),**map_checks}
        verify_alignment_input_binding(inputs,json.loads((alignment/'handoff_manifest.json').read_text()))
        target=Path(benchmarks).resolve() if benchmarks else inputs.parent/'benchmarks'
        if (target/'manifest.json').exists():
            from benchmark_inputs import validate_benchmark_suite
            validate_benchmark_suite(target);source[str(target/'manifest.json')]=sha(target/'manifest.json')
        code={str(p.relative_to(ROOT)):sha(p) for p in TOOLS.glob('*.py')}
        if (folder/'manifest.json').exists():
            record=verify(folder)
            if record['metadata']['inputs']!=source or record['metadata']['code']!=code:raise ValueError('performance input/source changed; use a new performance output')
            return json.loads((folder/'status.json').read_text())
        started=time.perf_counter();write_json(folder/'status.json',dict(status='running',alignment_ready_for_handoff=status['alignment_ready_for_handoff']))
        p=json.loads((inputs/'uec_input_provenance.json').read_text())
        data=load_inputs(dict(mode=p['profile'],provenance_path=str(inputs/'uec_input_provenance.json')));audit,split=audit_and_split(data)
        work=inputs.parent;blocked={r['canonical_text_sha256'] for r in read_jsonl(work/'data/observations_all.jsonl')};metrics=[];internal={}
        for source_name in ['ms_marco','snli']:
            task,native,scope=diagnostic_task(data,split.diagnostic_indices,source_name)
            write_json(folder/'progress.json',dict(phase='internal',task=task['task']))
            result=evaluate_task(task,native,maps,set(),folder/'internal'/task['task'])
            metrics.extend(dict(r,scope='internal',map_approved=status['alignment_ready_for_handoff']) for r in result['metrics']);internal[source_name]=scope
        if not (target/'manifest.json').exists():
            from benchmark_inputs import prepare_public_benchmarks
            from prepare_alignment import check_gpu
            cfg=json.loads((work/'preparation_config.json').read_text())
            os.environ['CUDA_VISIBLE_DEVICES']=check_gpu(cfg['gpu'],settle_utilization=True)
            write_json(folder/'progress.json',dict(phase='prepare_public_benchmarks'))
            prepare_public_benchmarks(target,cfg['cache_dir'],device=cfg['device'],offline=cfg['offline'],qwen_attention=cfg['qwen_attention'],physical_batch=cfg['physical_batch'])
        from benchmark_inputs import validate_benchmark_suite
        validate_benchmark_suite(target)
        benchmark=verify(target);external_tasks=benchmark['metadata']['tasks'];source[str(target/'manifest.json')]=sha(target/'manifest.json')
        external={}
        for name in external_tasks:
            write_json(folder/'progress.json',dict(phase='external',task=name))
            task,native=load_task(target/name)
            result=evaluate_task(task,native,maps,blocked,folder/'external'/name)
            metrics.extend(dict(r,scope='external',map_approved=status['alignment_ready_for_handoff']) for r in result['metrics']);external[name]=result['diagnostics']
        for file,expected in source.items():
            if sha(file)!=expected:raise ValueError('performance source changed '+file)
        verify(inputs);verify(target)
        for name in external_tasks:verify(target/name)
        metadata=dict(inputs=source,code=code,internal=internal,external_tasks=external_tasks,benchmark_metadata=benchmark['metadata'],
            full_mmteb=False,posterior_unseen=False,external_fit_text_exclusions=True,mean_uniform_weights=[1/3]*3,uec_uncertainty_score=False)
        write_report(folder,metrics,metadata)
        final=dict(status='completed',conditions=11,internal_tasks=2,external_tasks=len(external_tasks),metrics=len(metrics),wall_seconds=time.perf_counter()-started)
        write_json(folder/'status.json',final)
        names=[str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name not in ['manifest.json','.lock']]
        seal(folder,names,dict(inputs=source,code=code));return final

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmarks',type=Path);p.add_argument('--performance-output',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();base=[sys.executable,str(ROOT/'run_alignment.py'),'--inputs',str(a.inputs),'--output',str(a.output)]
    if a.check_only or (a.output/'status.json').exists():base.append('--check-only')
    subprocess.run(base,check=True)
    if not a.check_only:print(json.dumps(evaluate(a.inputs,a.output,a.benchmarks,a.performance_output),indent=2))

if __name__=='__main__':main()
