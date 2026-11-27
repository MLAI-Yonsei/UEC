"""Portable fixed-protocol performance scoring after alignment.

Native and mapped single models use the same held-out records as the two
equal-weight mean-embedding baselines.  No uncertainty weights, fitted
hyperparameters, model inference, or private experiment modules are used here.
The input protocol preserves original classification support trials and STS
pair indices.  All arithmetic remains float64; metric differences are retained.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unicodedata
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


MODELS = ('qwen', 'harrier', 'gte')
ARMS = ('gpa', 'caia')
CONDITIONS = tuple(c for m in MODELS for c in (f'native_{m}', f'gpa_{m}', f'caia_{m}')) + (
    'uniform_gpa', 'uniform_caia')
ISOMETRY_TOLERANCE = 1e-8


def canonical_text_sha256(text):
    """Hash NFC text after CRLF/CR to LF conversion, without trimming spaces."""
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    canonical = unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def normalize_rows(values):
    """Return finite float64 row-unit vectors; reject zero or invalid vectors."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError('Expected finite two-dimensional embeddings')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Zero or nonfinite embedding norm')
    return x / norms


def _validate_ids(ids, label):
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Missing or duplicate {label} IDs')


def _rank_unit(queries, corpus, query_ids, corpus_ids, top_k=100, block_size=128):
    """Frozen TREC tie convention: reverse doc ID, then stable score sort."""
    if top_k < 1 or block_size < 1:
        raise ValueError('top_k and block_size must be positive')
    _validate_ids(query_ids, 'query')
    _validate_ids(corpus_ids, 'corpus')
    if len(queries) != len(query_ids) or len(corpus) != len(corpus_ids) or not len(corpus):
        raise ValueError('Retrieval arrays do not match IDs or corpus is empty')
    if queries.shape[1] != corpus.shape[1]:
        raise ValueError('Query/corpus dimensions differ')
    order = np.array(sorted(range(len(corpus_ids)), key=lambda i: corpus_ids[i], reverse=True))
    k = min(top_k, len(corpus_ids))
    ranked = {}
    for offset in range(0, len(queries), block_size):
        scores = queries[offset:offset + block_size] @ corpus[order].T
        selected = np.argsort(-scores, axis=1, kind='stable')[:, :k]
        for j, positions in enumerate(selected):
            ranked[query_ids[offset + j]] = [(corpus_ids[order[p]], float(scores[j, p])) for p in positions]
    return ranked


def exact_cosine_search(queries, corpus, query_ids, corpus_ids, top_k=100, block_size=128):
    """Exact cosine top-k as {query_id: [(doc_id, actual_float_score), ...]}."""
    return _rank_unit(normalize_rows(queries), normalize_rows(corpus), list(query_ids),
                      list(corpus_ids), top_k, block_size)


def retrieval_metrics(qrels, rankings):
    """TREC NDCG@10 (linear graded gain) and recall@100, macro over queries.

    Ranking order is already explicit. A relevance value above zero counts as a
    positive for recall; NDCG uses its numeric grade, as pytrec_eval ndcg_cut does.
    """
    if not qrels or set(qrels) != set(rankings):
        raise ValueError('Retrieval predictions must cover every judged query')
    per_query = {}
    for qid, judgments in qrels.items():
        grades = np.asarray(list(judgments.values()), dtype=np.float64)
        if not len(grades) or not np.isfinite(grades).all() or np.any(grades < 0):
            raise ValueError(f'Invalid relevance grades for {qid}')
        relevant = {doc for doc, grade in judgments.items() if grade > 0}
        if not relevant:
            raise ValueError(f'No positive judgments for {qid}')
        ids = [item[0] for item in rankings[qid]]
        _validate_ids(ids, 'ranked document')
        if any(not np.isfinite(item[1]) for item in rankings[qid]):
            raise ValueError('Nonfinite retrieval prediction')
        gains = np.asarray([judgments.get(doc, 0) for doc in ids[:10]], dtype=np.float64)
        ideal = np.sort(grades)[::-1][:10]
        dcg = float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)))
        idcg = float(np.sum(ideal / np.log2(np.arange(len(ideal)) + 2)))
        per_query[qid] = dict(ndcg_at_10=dcg / idcg,
                              recall_at_100=len(set(ids[:100]) & relevant) / len(relevant))
    return dict(mean={key: float(np.mean([r[key] for r in per_query.values()]))
                      for key in ('ndcg_at_10', 'recall_at_100')}, per_query=per_query)


def load_task(task_folder, native_filename='native.npz'):
    """Load TASK/task.json and ROLE__MODEL arrays without pickle."""
    folder = Path(task_folder)
    data = json.loads((folder / 'task.json').read_text(encoding='utf-8'))
    with np.load(folder / native_filename, allow_pickle=False) as archive:
        arrays = {role: {m: np.asarray(archive[f'{role}__{m}'], dtype=np.float64)
                         for m in MODELS} for role in data['roles']}
    return data, arrays


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _prepare(data, native_by_role, blocked_hashes):
    kind = data.get('kind')
    required = {'STS': ('texts',), 'Retrieval': ('queries', 'corpus'),
                'Classification': ('support', 'test')}
    if kind not in required:
        raise ValueError(f'Unsupported task kind: {kind!r}')
    if not isinstance(data.get('task'), str) or not data['task']:
        raise ValueError('Missing task name')
    roles = data.get('roles', {})
    if set(roles) != set(required[kind]):
        raise ValueError(f'{kind} needs roles {required[kind]}')
    blocked = set(blocked_hashes)
    if any(not isinstance(h, str) or len(h) != 64 for h in blocked):
        raise ValueError('Blocked text hashes must be SHA256 strings')
    keep, audit_roles, input_norms = {}, {}, {}
    dims = {}
    for role, records in roles.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f'Empty or invalid {role} records')
        ids, hashes = [], []
        for r in records:
            if not isinstance(r, dict) or 'row_id' not in r:
                raise ValueError(f'Missing row_id in {role}')
            ids.append(r['row_id'])
            h = canonical_text_sha256(r.get('text'))
            if r.get('canonical_text_sha256') != h:
                raise ValueError(f'Canonical text hash mismatch in {role}: {r["row_id"]}')
            hashes.append(h)
        _validate_ids(ids, role)
        indices = [i for i, h in enumerate(hashes) if h not in blocked]
        keep[role] = indices
        removed = [i for i, h in enumerate(hashes) if h in blocked]
        audit_roles[role] = dict(count_before=len(records), count_after=len(indices),
            blocked_count=len(removed), blocked_original_indices=removed,
            blocked_row_ids=[ids[i] for i in removed],
            blocked_text_hashes=[hashes[i] for i in removed],
            kept_original_indices=indices, row_ids_after=[ids[i] for i in indices])
        if set(native_by_role.get(role, {})) != set(MODELS):
            raise ValueError(f'Missing native model arrays for {role}')
        input_norms[role] = {}
        for model in MODELS:
            x = np.asarray(native_by_role[role][model], dtype=np.float64)
            if x.ndim != 2 or len(x) != len(records) or x.shape[1] == 0 or not np.isfinite(x).all():
                raise ValueError(f'Invalid native embedding array {role}/{model}')
            norms = np.linalg.norm(x, axis=1)
            if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                raise ValueError(f'Zero or nonfinite native norm in {role}/{model}')
            if model in dims and dims[model] != x.shape[1]:
                raise ValueError(f'Inconsistent native dimension for {model}')
            dims[model] = x.shape[1]
            input_norms[role][model] = float(np.max(np.abs(norms - 1)))
    audit = dict(blocked_hash_count=len(blocked), roles=audit_roles,
                 policy='Exclude any evaluation text equal to any uncertainty-fit pair endpoint; NFC and newline canonicalization only')
    protocol = {}
    if kind == 'STS':
        pairs = data.get('pairs', [])
        if not isinstance(pairs, list) or not pairs:
            raise ValueError('Missing STS pairs')
        pair_ids = [p.get('id') for p in pairs]
        _validate_ids(pair_ids, 'pair')
        kept, dropped = [], []
        retained = set(keep['texts'])
        for p in pairs:
            li, ri = p.get('left_index'), p.get('right_index')
            if (type(li) is not int or type(ri) is not int or
                    min(li, ri) < 0 or max(li, ri) >= len(roles['texts'])):
                raise ValueError(f'Invalid original STS indices for {p["id"]}')
            if not isinstance(p.get('gold'), (int, float)) or not np.isfinite(p['gold']):
                raise ValueError(f'Invalid STS gold for {p["id"]}')
            if li not in retained or ri not in retained:
                dropped.append(dict(pair_id=p['id'], left_index=li, right_index=ri,
                                    reason='blocked_fit_endpoint',
                                    blocked_left=li not in retained, blocked_right=ri not in retained))
            else:
                kept.append(p)
        if len(kept) < 2 or np.ptp([p['gold'] for p in kept]) == 0:
            raise ValueError('STS needs at least two surviving pairs with nonconstant gold')
        index_map = {old: new for new, old in enumerate(keep['texts'])}
        protocol = dict(pairs=kept, left=np.array([index_map[p['left_index']] for p in kept]),
                        right=np.array([index_map[p['right_index']] for p in kept]),
                        gold=np.array([p['gold'] for p in kept], dtype=np.float64))
        audit.update(pair_count_before=len(pairs), pair_count_after=len(kept),
                     excluded_pair_ids=[p['pair_id'] for p in dropped], excluded_pairs=dropped,
                     pair_ids_after=[p['id'] for p in kept])
    elif kind == 'Retrieval':
        qrels = data.get('qrels')
        if not isinstance(qrels, dict):
            raise ValueError('Missing retrieval qrels')
        all_q = {r['row_id'] for r in roles['queries']}
        all_c = {r['row_id'] for r in roles['corpus']}
        if set(qrels) - all_q:
            raise ValueError('Qrels contain unknown query IDs')
        for qid, judgments in qrels.items():
            if not isinstance(judgments, dict) or set(judgments) - all_c:
                raise ValueError(f'Qrels contain unknown document IDs for {qid}')
            grades = np.asarray(list(judgments.values()), dtype=np.float64)
            if not np.isfinite(grades).all() or np.any(grades < 0):
                raise ValueError(f'Invalid relevance grades for {qid}')
        corpus_ids = [roles['corpus'][i]['row_id'] for i in keep['corpus']]
        remaining_corpus = set(corpus_ids)
        kept_q, judged, unjudged, lost, original_zero = [], {}, [], [], []
        for index in keep['queries']:
            qid = roles['queries'][index]['row_id']
            if qid not in qrels:
                unjudged.append(qid)
                continue
            filtered = {doc: grade for doc, grade in qrels[qid].items() if doc in remaining_corpus}
            if not any(grade > 0 for grade in qrels[qid].values()):
                original_zero.append(qid)
                continue
            if not any(grade > 0 for grade in filtered.values()):
                lost.append(qid)
                continue
            kept_q.append(index)
            judged[qid] = filtered
        keep['queries'] = kept_q
        query_ids = [roles['queries'][i]['row_id'] for i in kept_q]
        if not corpus_ids or not query_ids:
            raise ValueError('No judged queries with positives remain after exclusions')
        audit_roles['queries'].update(count_after=len(kept_q), kept_original_indices=kept_q,
                                      row_ids_after=query_ids)
        protocol = dict(query_ids=query_ids, corpus_ids=corpus_ids, qrels=judged)
        audit.update(query_ids_after=query_ids, corpus_ids_after=corpus_ids,
                     unjudged_query_ids=unjudged, queries_losing_all_positives=lost,
                     queries_without_original_positives=original_zero,
                     qrel_count_before=sum(len(v) for v in qrels.values()),
                     qrel_count_after=sum(len(v) for v in judged.values()))
    else:
        labels = data.get('labels', {})
        for role in ('support', 'test'):
            if len(labels.get(role, [])) != len(roles[role]):
                raise ValueError(f'Classification labels do not match {role} records')
            if not keep[role]:
                raise ValueError(f'No classification {role} records survive exclusions')
        class_order = labels.get('class_order', [])
        if (not class_order or len(set(class_order)) != len(class_order) or
                not set(labels['support'] + labels['test']).issubset(set(class_order))):
            raise ValueError('Invalid classification class_order')
        trials = labels.get('support_positions', [])
        if len(trials) != 10:
            raise ValueError('Classification requires the original 10 support-position trials')
        remap = {old: new for new, old in enumerate(keep['support'])}
        after, removed_by_trial = [], []
        for trial in trials:
            if not isinstance(trial, list) or any(type(i) is not int or i < 0 or i >= len(roles['support']) for i in trial):
                raise ValueError('Invalid original support_positions')
            positions = [remap[i] for i in trial if i in remap]
            after.append(positions)
            removed_by_trial.append([i for i in trial if i not in remap])
            if len({labels['support'][i] for i in trial if i in remap}) < 2:
                raise ValueError('Leak filtering left a support trial with fewer than two classes; no resampling allowed')
        protocol = dict(y=np.asarray(labels['support'])[keep['support']],
                        gold=np.asarray(labels['test'])[keep['test']], trials=after,
                        class_order=class_order)
        audit.update(class_order=class_order, support_positions_before=trials,
                     support_positions_after=after, support_positions_removed_by_trial=removed_by_trial,
                     support_trial_count_before=len(trials), support_trial_count_after=len(after))
    arrays = {role: {m: normalize_rows(np.asarray(native_by_role[role][m], dtype=np.float64)[keep[role]])
                      for m in MODELS} for role in roles}
    return kind, arrays, dims, protocol, audit, input_norms


def _validate_maps(maps_by_arm, dims):
    if set(maps_by_arm) != set(ARMS):
        raise ValueError('Expected gpa and caia maps')
    target_dims, diagnostics = set(), {}
    for arm in ARMS:
        if set(maps_by_arm[arm]) != set(MODELS):
            raise ValueError(f'Missing {arm} model maps')
        for model in MODELS:
            r = np.asarray(maps_by_arm[arm][model], dtype=np.float64)
            if r.ndim != 2 or r.shape[0] != dims[model] or not np.isfinite(r).all():
                raise ValueError(f'Invalid map {arm}/{model}')
            error = float(np.max(np.abs(r @ r.T - np.eye(r.shape[0]))))
            if error > ISOMETRY_TOLERANCE:
                raise ValueError(f'Map {arm}/{model} is not row-isometric: {error}')
            target_dims.add(r.shape[1])
            diagnostics[f'{arm}_{model}'] = dict(shape=list(r.shape),
                max_abs_row_isometry_error=error, tolerance=ISOMETRY_TOLERANCE)
    if len(target_dims) != 1:
        raise ValueError('All maps must share one common target dimension')
    return diagnostics


def _cosine_geometry(kind, native, mapped, protocol):
    if kind == 'STS':
        a = np.einsum('ij,ij->i', native['texts'][protocol['left']], native['texts'][protocol['right']])
        b = np.einsum('ij,ij->i', mapped['texts'][protocol['left']], mapped['texts'][protocol['right']])
        return dict(max_abs_cosine_error=float(np.max(np.abs(a - b))),
                    compared_cosine_count=len(a), comparison='all surviving STS pairs')
    left, right = ('queries', 'corpus') if kind == 'Retrieval' else ('support', 'test')
    error, count = 0., 0
    for start in range(0, len(native[left]), 128):
        a = native[left][start:start + 128] @ native[right].T
        b = mapped[left][start:start + 128] @ mapped[right].T
        error = max(error, float(np.max(np.abs(a - b))))
        count += a.size
    return dict(max_abs_cosine_error=error, compared_cosine_count=count,
                comparison=f'all surviving {left}-by-{right} cosines')


def _classify(features, protocol):
    predictions, scores, estimators, trial_details = [], [], [], []
    for positions in protocol['trials']:
        clf = LogisticRegression(random_state=42, n_jobs=1, max_iter=100)
        # Convergence outcomes are evidence; retain them instead of changing the
        # frozen estimator parameters to force agreement after a rotation.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clf.fit(features['support'][positions], protocol['y'][positions])
        pred = clf.predict(features['test'])
        predictions.append(pred)
        scores.append([accuracy_score(protocol['gold'], pred),
                       f1_score(protocol['gold'], pred, average='macro', zero_division=0)])
        estimators.append(clf)
        trial_details.append(dict(support_positions=positions, classes=clf.classes_.tolist(),
            n_iter=clf.n_iter_.tolist(), converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
            warnings=[str(w.message) for w in caught],
            parameters=clf.get_params(), coefficients=clf.coef_.tolist(), intercept=clf.intercept_.tolist(),
            accuracy=float(scores[-1][0]), macro_f1=float(scores[-1][1])))
    return dict(predictions=np.asarray(predictions), means=np.mean(scores, axis=0),
                estimators=estimators, trials=trial_details)


def _head_predictions(logits, classes):
    if len(classes) == 2:
        return classes[(np.asarray(logits).reshape(-1) > 0).astype(int)]
    return classes[np.argmax(logits, axis=1)]


def _transport_proof(model, arm, native, mapped, r, native_result, mapped_result, protocol):
    trials, accuracies, f1s = [], [], []
    for i, clf in enumerate(native_result['estimators']):
        native_logits = np.asarray(clf.decision_function(native['test']))
        coefficients = clf.coef_ @ r
        logits = mapped['test'] @ coefficients.T + clf.intercept_
        if len(clf.classes_) == 2:
            logits = logits.reshape(-1)
        max_error = float(np.max(np.abs(logits - native_logits)))
        if max_error > ISOMETRY_TOLERANCE:
            raise ValueError(f'Transported classification logits exceed tolerance for {arm}/{model}: {max_error}')
        pred = _head_predictions(logits, clf.classes_)
        acc = float(accuracy_score(protocol['gold'], pred))
        f1 = float(f1_score(protocol['gold'], pred, average='macro', zero_division=0))
        accuracies.append(acc)
        f1s.append(f1)
        trials.append(dict(trial_index=i, max_abs_logit_error=max_error,
            predictions_equal_native=bool(np.array_equal(pred, native_result['predictions'][i])),
            changed_prediction_count_vs_native=int(np.count_nonzero(pred != native_result['predictions'][i])),
            native_accuracy=native_result['trials'][i]['accuracy'], transported_accuracy=acc,
            native_macro_f1=native_result['trials'][i]['macro_f1'], transported_macro_f1=f1,
            transported_predictions=pred.tolist(),
            refitted_changed_prediction_count_vs_native=int(np.count_nonzero(
                mapped_result['predictions'][i] != native_result['predictions'][i]))))
    return dict(model=model, arm=arm, condition=f'{arm}_{model}', trials=trials,
        max_abs_logit_error=max(t['max_abs_logit_error'] for t in trials),
        all_trial_predictions_equal_native=all(t['predictions_equal_native'] for t in trials),
        native_accuracy=float(native_result['means'][0]), transported_accuracy=float(np.mean(accuracies)),
        refitted_accuracy=float(mapped_result['means'][0]),
        native_macro_f1=float(native_result['means'][1]), transported_macro_f1=float(np.mean(f1s)),
        refitted_macro_f1=float(mapped_result['means'][1]),
        refitted_accuracy_delta_vs_native=float(mapped_result['means'][0] - native_result['means'][0]),
        explanation='Main metrics refit the frozen native protocol; coefficient transport is a separate numerical proof')


def evaluate_task(data, native_by_role, maps_by_arm, blocked_hashes, output_folder=None):
    """Score all 11 conditions and return {metrics, diagnostics}.

    ``native_by_role[role][model]`` holds float64 native rows in the exact order
    of task.json. ``maps_by_arm[arm][model]`` holds a d_native by d_common
    row-isometry. ``blocked_hashes`` MUST include every endpoint of every
    uncertainty-fit pair; this function applies the supplied set without
    estimating it from evaluation labels. Existing predictions are never used
    to select support trials, weights, maps, or estimator settings.
    """
    kind, arrays, dims, protocol, exclusions, input_norms = _prepare(data, native_by_role, blocked_hashes)
    map_diagnostics = _validate_maps(maps_by_arm, dims)
    folder = Path(output_folder) if output_folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'predictions').mkdir(exist_ok=True)
        _write_json(folder / 'exclusion_audit.json', exclusions)
    metrics, predictions, geometry, transport, native_classifications = [], {}, {}, [], {}
    conditions = {}
    for model in MODELS:
        conditions[f'native_{model}'] = {role: by_model[model] for role, by_model in arrays.items()}
        for arm in ARMS:
            r = np.asarray(maps_by_arm[arm][model], dtype=np.float64)
            mapped = {role: normalize_rows(by_model[model] @ r) for role, by_model in arrays.items()}
            condition = f'{arm}_{model}'
            conditions[condition] = mapped
            geometry[condition] = _cosine_geometry(kind, conditions[f'native_{model}'], mapped, protocol)
            if geometry[condition]['max_abs_cosine_error'] > ISOMETRY_TOLERANCE:
                raise ValueError(f'Cosine preservation exceeds tolerance for {condition}')
    for arm in ARMS:
        # Exactly unit(sum_m(X_m @ R_m) / 3); no per-model uncertainty weights.
        conditions[f'uniform_{arm}'] = {
            role: normalize_rows(sum(arrays[role][m] @ np.asarray(maps_by_arm[arm][m], dtype=np.float64)
                                     for m in MODELS) / 3.) for role in arrays}
    for condition in CONDITIONS:
        features = conditions[condition]
        row = dict(task=data['task'], kind=kind, condition=condition)
        if kind == 'STS':
            scores = np.einsum('ij,ij->i', features['texts'][protocol['left']], features['texts'][protocol['right']])
            statistic = float(spearmanr(scores, protocol['gold']).statistic)
            if not np.isfinite(statistic):
                raise ValueError(f'Undefined STS Spearman for {condition}: constant predictions')
            row.update(spearman=statistic, pair_count=len(scores), primary_metric='spearman', score100=100 * statistic)
            prediction = dict(pair_ids=[p['id'] for p in protocol['pairs']],
                left_indices=[p['left_index'] for p in protocol['pairs']],
                right_indices=[p['right_index'] for p in protocol['pairs']],
                scores=scores.tolist(), gold=protocol['gold'].tolist())
        elif kind == 'Retrieval':
            ranked = _rank_unit(features['queries'], features['corpus'], protocol['query_ids'], protocol['corpus_ids'])
            result = retrieval_metrics(protocol['qrels'], ranked)
            row.update(result['mean'], query_count=len(ranked), corpus_count=len(features['corpus']),
                       primary_metric='ndcg_at_10', score100=100 * result['mean']['ndcg_at_10'])
            prediction = dict(query_ids=protocol['query_ids'], rankings=ranked, per_query=result['per_query'])
            if not condition.startswith(('native_', 'uniform_')):
                model = condition.split('_', 1)[1]
                original = predictions[f'native_{model}']['rankings']
                geometry[condition]['queries_with_changed_top100_order'] = sum(
                    [d for d, _ in ranked[q]] != [d for d, _ in original[q]] for q in ranked)
                geometry[condition]['metric_delta_vs_native'] = {
                    k: row[k] - next(r[k] for r in metrics if r['condition'] == f'native_{model}')
                    for k in ('ndcg_at_10', 'recall_at_100')}
        else:
            result = _classify(features, protocol)
            row.update(accuracy=float(result['means'][0]), macro_f1=float(result['means'][1]),
                       trial_count=len(protocol['trials']), test_count=len(protocol['gold']),
                       primary_metric='accuracy', score100=100 * float(result['means'][0]))
            prediction = dict(test_row_ids=exclusions['roles']['test']['row_ids_after'],
                gold=protocol['gold'].tolist(), class_order=protocol['class_order'],
                predictions=result['predictions'].tolist(), trials=result['trials'])
            if condition.startswith('native_'):
                native_classifications[condition.split('_', 1)[1]] = result
            elif not condition.startswith('uniform_'):
                arm, model = condition.split('_', 1)
                transport.append(_transport_proof(model, arm, conditions[f'native_{model}'], features,
                    np.asarray(maps_by_arm[arm][model], dtype=np.float64),
                    native_classifications[model], result, protocol))
        metrics.append(row)
        predictions[condition] = prediction
        if folder is not None:
            _write_json(folder / 'predictions' / f'{condition}.json', prediction)
        print(f'SCORED {data["task"]} {condition} {row["score100"]}', flush=True)
    source = Path(__file__).resolve()
    diagnostics = dict(schema_version=1, task=data['task'], kind=kind, conditions=list(CONDITIONS),
        exclusions=exclusions, geometry=geometry, maps=map_diagnostics,
        input_max_abs_unit_norm_error=input_norms,
        source_provenance=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        native_dimensions=dims, evaluation_dtype='float64',
        uniform_definition='unit(sum_m(unit(native_m) @ map_m) / 3); equal weights 1/3',
        classification_protocol=dict(random_state=42, n_jobs=1, max_iter=100,
            trial_count=len(protocol['trials']) if kind == 'Classification' else None,
            support_resampled=False),
        classification_transport=transport,
        metric_policy='Actual unrounded floats; native and mapped refits share protocol; numerical tie or optimizer changes are retained')
    if folder is not None:
        _write_json(folder / 'metrics.json', metrics)
        keys = list(dict.fromkeys(key for row in metrics for key in row))
        with (folder / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)
        _write_json(folder / 'diagnostics.json', diagnostics)
        if kind == 'Classification':
            _write_json(folder / 'classification_transport.json', transport)
    return dict(metrics=metrics, diagnostics=diagnostics)
