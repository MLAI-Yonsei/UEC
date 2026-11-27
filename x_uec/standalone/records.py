import hashlib,unicodedata
import random
import numpy as np

def canonical_hash(text):
    if not isinstance(text,str):raise ValueError('text must be a string')
    value=unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n'))
    return hashlib.sha256(value.encode()).hexdigest()

def selection_hash(text):
    # Original pair selection collapsed whitespace. Alignment/evaluation grouping
    # uses canonical_hash instead; retain both contracts without changing text.
    return hashlib.sha256(' '.join(unicodedata.normalize('NFC',text).split()).encode()).hexdigest()

def main_sample_indices(ms_count,snli_count):
    """UEC main: ONE Python RNG, MS sample first, SNLI sample second."""
    rng=random.Random(2025)
    return dict(ms_marco=rng.sample(range(ms_count),int(ms_count*.05)),snli=rng.sample(range(snli_count),int(snli_count*.02)))

def main_pairs(sources,indices):
    pairs=[]
    for source in ['ms_marco','snli']:
        for index in indices[source]:
            raw=sources[source][index]
            if source=='ms_marco':
                try:
                    selected=[i for i,y in enumerate(raw['passages']['is_selected']) if y==1][0]
                    positive=raw['passages']['passage_text'][selected]
                except (IndexError,KeyError,TypeError):continue
                query=raw['query']
            else:
                if raw['label']!=0:continue
                query=raw['premise'];positive=raw['hypothesis']
            pairs.append(dict(source=source,source_row=index,triplet_id=f'uec_main_train:{len(pairs)}',query=query,positive=positive,
                query_hash=selection_hash(query),positive_hash=selection_hash(positive),source_query_id=raw.get('query_id')))
    return pairs

def reconstruct_pairs(selection,sources):
    result=[]
    for record in selection:
        source=record['source'];raw=sources[source][record['source_row']]
        if source=='ms_marco':
            query=raw['query'];positive=[t for t,y in zip(raw['passages']['passage_text'],raw['passages']['is_selected']) if y==1 and selection_hash(t)==record['positive_hash']]
        elif source=='snli':
            query=raw['premise'];positive=[raw['hypothesis']] if raw['label']==0 and selection_hash(raw['hypothesis'])==record['positive_hash'] else []
        else:raise ValueError('unsupported pair source')
        if selection_hash(query)!=record['query_hash'] or not positive:raise ValueError('reference hash mismatch; wrong dataset revision/row')
        result.append(dict(record,query=query,positive=positive[0]))
    pool={r['positive_hash']:r['positive'] for r in result}
    for row in result:
        if row.get('negative_hash') is not None:
            if row['negative_hash'] not in pool:raise ValueError('original negative not in reference pool')
            row['negative']=pool[row['negative_hash']]
    return result

def observations(pairs):
    rows=[]
    for index,pair in enumerate(pairs):
        roles=('query','positive_passage') if pair['source']=='ms_marco' else ('premise','hypothesis')
        for field,role,side in zip(('query','positive'),roles,('query','document')):
            text=pair[field]
            if not text.strip():raise ValueError('empty selected text')
            rows.append(dict(observation_id=f"{pair['source']}:{pair['triplet_id']}:{role}:0",source_order=len(rows),pair_index=index,
                dataset=pair['source'],source_record_id=pair['source_row'],original_triplet_id=pair['triplet_id'],
                role=role,moment_side=side,text=text,canonical_text_sha256=canonical_hash(text)))
    if len({r['observation_id'] for r in rows})!=len(rows):raise ValueError('duplicate observation IDs')
    return rows

def split_rows(rows,evaluation_hashes):
    excluded=[i for i,r in enumerate(rows) if r['canonical_text_sha256'] in evaluation_hashes]
    excluded_set=set(excluded);eligible=[i for i in range(len(rows)) if i not in excluded_set]
    groups=sorted({rows[i]['canonical_text_sha256'] for i in eligible});order=np.random.default_rng(42).permutation(len(groups))
    nfit=int(.8*len(groups));fit_groups={groups[i] for i in order[:nfit]}
    fit=[i for i in eligible if rows[i]['canonical_text_sha256'] in fit_groups];diag=[i for i in eligible if rows[i]['canonical_text_sha256'] not in fit_groups]
    if not fit or not diag:raise ValueError('empty alignment split')
    moments={};used=set()
    for name,side,count in [('moment_query_indices','query',2),('moment_document_indices','document',4)]:
        chosen=[]
        for i in diag:
            key=rows[i]['canonical_text_sha256']
            if rows[i]['moment_side']==side and key not in used:chosen.append(i);used.add(key)
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError('insufficient distinct diagnostic query/document observations')
        moments[name]=chosen
    return dict(fit_indices=fit,diagnostic_indices=diag,excluded_indices=excluded,selected_indices=sorted(fit+diag),seed=42,
        groups=len(groups),fit_groups=nfit,diagnostic_groups=len(groups)-nfit,**moments)
