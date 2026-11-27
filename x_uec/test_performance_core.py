"""Portable scoring contract: no model downloads or private experiment imports."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def record(row_id, text=None):
    import hashlib
    text = str(row_id) if text is None else text
    return dict(row_id=str(row_id), text=text,
                canonical_text_sha256=hashlib.sha256(text.encode()).hexdigest())


def maps():
    # Both are row-isometric, including a genuine expansion into dimension 3.
    a = np.array([[1., 0., 0.], [0., 1., 0.]])
    b = np.array([[0., -1., 0.], [1., 0., 0.]])
    return {arm: {model: r.copy() for model in ('qwen', 'harrier', 'gte')}
            for arm, r in [('gpa', a), ('caia', b)]}


def native(roles):
    return {role: {model: np.array(values, dtype=np.float64)
                   for model in ('qwen', 'harrier', 'gte')}
            for role, values in roles.items()}


class PerformanceCoreTests(unittest.TestCase):
    def setUp(self):
        # A missing implementation is an explicit failing assertion on first run.
        self.assertIsNotNone(importlib.util.find_spec('tools.performance_core'),
                             'portable performance scoring core is missing')
        self.core = importlib.import_module('tools.performance_core')

    def sts_fixture(self):
        h = 2 ** -.5
        data = dict(task='tiny-sts', kind='STS', roles={'texts': [record(i) for i in range(3)]},
                    pairs=[dict(id='ab', left_index=0, right_index=1, gold=.2),
                           dict(id='ac', left_index=0, right_index=2, gold=.8),
                           dict(id='bc', left_index=1, right_index=2, gold=.9)])
        values = native({'texts': [[1, 0], [1, 0], [h, h]]})
        values['texts']['harrier'][0] = [0, 1]
        values['texts']['gte'][:2] = [[0, 1], [0, 1]]
        return data, values

    def test_canonical_hash_preserves_spaces_but_unifies_nfc_and_newlines(self):
        f = self.core.canonical_text_sha256
        self.assertEqual(f('e\u0301\r\nx\ry'), f('\u00e9\nx\ny'))
        self.assertNotEqual(f(' x '), f('x'))

    def test_uniform_is_cosine_of_equal_mean_embeddings_and_isometry_is_measured(self):
        data, values = self.sts_fixture()
        result = self.core.evaluate_task(data, values, maps(), set())
        self.assertEqual(len(result['metrics']), 11)
        by_condition = {r['condition']: r for r in result['metrics']}
        self.assertAlmostEqual(by_condition['native_qwen']['spearman'], -np.sqrt(.75))
        self.assertLessEqual(result['diagnostics']['geometry']['gpa_qwen']['max_abs_cosine_error'], 1e-8)
        with tempfile.TemporaryDirectory() as folder:
            self.core.evaluate_task(data, values, maps(), set(), folder)
            p = json.loads((Path(folder) / 'predictions/uniform_gpa.json').read_text())
            np.testing.assert_allclose(p['scores'], [.8, 3 / np.sqrt(10), 3 / np.sqrt(10)], atol=1e-15)
            self.assertEqual(p['pair_ids'], ['ab', 'ac', 'bc'])
            self.assertTrue((Path(folder) / 'metrics.csv').is_file())

    def test_sts_leak_filter_preserves_pair_ids_original_indices_and_audit(self):
        data, values = self.sts_fixture()
        data['roles']['texts'].append(record('unused-blocked'))
        for model in values['texts']:
            values['texts'][model] = np.vstack([values['texts'][model], [1., 0.]])
        blocked = {data['roles']['texts'][0]['canonical_text_sha256'],
                   data['roles']['texts'][3]['canonical_text_sha256']}
        # Add a second surviving pair to permit a defined Spearman coefficient.
        data['pairs'].append(dict(id='cc', left_index=2, right_index=2, gold=1.))
        with tempfile.TemporaryDirectory() as folder:
            result = self.core.evaluate_task(data, values, maps(), blocked, folder)
            p = json.loads((Path(folder) / 'predictions/native_qwen.json').read_text())
            self.assertEqual(p['pair_ids'], ['bc', 'cc'])
            self.assertEqual(p['left_indices'], [1, 2])
            self.assertEqual(p['right_indices'], [2, 2])
            audit = result['diagnostics']['exclusions']
            self.assertEqual(audit['excluded_pair_ids'], ['ab', 'ac'])
            self.assertEqual(audit['roles']['texts']['blocked_row_ids'], ['0', 'unused-blocked'])
            self.assertEqual(audit['pair_count_after'], 2)

    def test_exact_cosine_ties_use_reverse_lexicographic_document_ids(self):
        rankings = self.core.exact_cosine_search(
            np.array([[1., 0.]]), np.array([[1., 0.], [1., 0.], [0., 1.]]),
            ['q'], ['a', 'z', 'm'])
        self.assertEqual([r[0] for r in rankings['q']], ['z', 'a', 'm'])
        result = self.core.retrieval_metrics({'q': {'a': 1, 'z': 1}}, rankings)
        self.assertEqual(result['mean']['ndcg_at_10'], 1.)
        self.assertEqual(result['mean']['recall_at_100'], 1.)

    def test_graded_ndcg_matches_trec_and_multiple_positives_count_for_recall(self):
        rankings = {'q': [('z', .8), ('b', .7), ('a', .6), ('x', .5)]}
        qrels = {'q': {'a': 3, 'b': 1, 'z': 0, 'unretrieved': 2}}
        result = self.core.retrieval_metrics(qrels, rankings)
        expected = (1 / np.log2(3) + 3 / np.log2(4)) / (3 + 2 / np.log2(3) + 1 / np.log2(4))
        self.assertAlmostEqual(result['mean']['ndcg_at_10'], expected)
        self.assertAlmostEqual(result['mean']['recall_at_100'], 2 / 3)
        if importlib.util.find_spec('pytrec_eval') is not None:
            import pytrec_eval
            actual = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.100'}).evaluate(
                {'q': dict(rankings['q'])})['q']
            self.assertAlmostEqual(result['mean']['ndcg_at_10'], actual['ndcg_cut_10'])
            self.assertAlmostEqual(result['mean']['recall_at_100'], actual['recall_100'])

    def test_top100_cutoff_keeps_the_frozen_tie_order_and_counts_missing_positive(self):
        ids = [f'd{i:03d}' for i in range(101)]
        rankings = self.core.exact_cosine_search(
            np.array([[1., 0.]]), np.tile([1., 0.], (101, 1)), ['q'], ids)
        self.assertEqual(len(rankings['q']), 100)
        self.assertEqual(rankings['q'][0][0], 'd100')
        self.assertEqual(rankings['q'][-1][0], 'd001')
        result = self.core.retrieval_metrics({'q': {'d100': 1, 'd000': 1}}, rankings)
        self.assertEqual(result['mean']['recall_at_100'], .5)
        self.assertAlmostEqual(result['mean']['ndcg_at_10'], 1 / (1 + 1 / np.log2(3)))

    def test_retrieval_filters_blocked_documents_queries_and_lost_positive_queries(self):
        data = dict(task='tiny-retrieval', kind='Retrieval',
                    roles={'queries': [record(s) for s in ['q', 'blocked-query', 'lost', 'unjudged']],
                           'corpus': [record(s) for s in ['a', 'b', 'blocked-doc']]},
                    qrels={'q': {'a': 1, 'b': 1}, 'blocked-query': {'a': 1}, 'lost': {'blocked-doc': 1}})
        values = native({'queries': [[1, 0]] * 4, 'corpus': [[1, 0], [1, 0], [0, 1]]})
        blocked = {data['roles']['queries'][1]['canonical_text_sha256'],
                   data['roles']['corpus'][2]['canonical_text_sha256']}
        result = self.core.evaluate_task(data, values, maps(), blocked)
        audit = result['diagnostics']['exclusions']
        self.assertEqual(audit['query_ids_after'], ['q'])
        self.assertEqual(audit['queries_losing_all_positives'], ['lost'])
        self.assertEqual(audit['unjudged_query_ids'], ['unjudged'])
        for metric in result['metrics']:
            self.assertEqual(metric['ndcg_at_10'], 1.)
            self.assertEqual(metric['recall_at_100'], 1.)

    def test_classification_retains_ten_original_support_trials_and_transports_native_heads(self):
        data = dict(task='tiny-classification', kind='Classification',
                    roles={'support': [record('s' + str(i)) for i in range(5)],
                           'test': [record('t' + str(i)) for i in range(3)]},
                    labels=dict(support=[0, 1, 0, 1, 0], test=[0, 1, 0], class_order=[1, 0],
                                support_positions=[[0, 1, 2, 3, 4], [4, 3, 1, 0]] * 5))
        values = native({'support': [[1, 0], [-1, 0], [.8, .6], [-.8, -.6], [0, 1]],
                         'test': [[1, 0], [-1, 0], [0, 1]]})
        blocked = {data['roles']['support'][4]['canonical_text_sha256'],
                   data['roles']['test'][2]['canonical_text_sha256']}
        with tempfile.TemporaryDirectory() as folder:
            result = self.core.evaluate_task(data, values, maps(), blocked, folder)
            for metric in result['metrics']:
                self.assertEqual(metric['accuracy'], 1.)
                self.assertEqual(metric['macro_f1'], 1.)
                self.assertEqual(metric['trial_count'], 10)
            audit = result['diagnostics']['exclusions']
            self.assertEqual(audit['support_positions_after'], [[0, 1, 2, 3], [3, 1, 0]] * 5)
            self.assertEqual(audit['class_order'], [1, 0])
            proof = json.loads((Path(folder) / 'classification_transport.json').read_text())
            self.assertEqual(len(proof), 6)
            for item in proof:
                self.assertLessEqual(item['max_abs_logit_error'], 1e-8)
                self.assertTrue(item['all_trial_predictions_equal_native'])
                self.assertEqual(item['transported_accuracy'], 1.)
                self.assertEqual(len(item['trials']), 10)

    def test_invalid_inputs_fail_instead_of_silently_scoring(self):
        data, values = self.sts_fixture()
        for bad in [np.array([[0., 0.]]), np.array([[np.nan, 0.]])]:
            with self.subTest(values=bad):
                with self.assertRaises(ValueError):
                    self.core.normalize_rows(bad)
        invalid_maps = maps()
        invalid_maps['gpa']['gte'][0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'isometr'):
            self.core.evaluate_task(data, values, invalid_maps, set())
        data['roles']['texts'][0].pop('row_id')
        with self.assertRaisesRegex(ValueError, 'row_id'):
            self.core.evaluate_task(data, values, maps(), set())

    def test_multiclass_native_head_transport_survives_dense_rectangular_isometries(self):
        h = np.sqrt(3) / 2
        x = [[1., 0.], [-.5, h], [-.5, -h]]
        labels = ['red', 'green', 'blue']
        data = dict(task='three-class', kind='Classification',
                    roles={'support': [record('s' + str(i)) for i in range(3)],
                           'test': [record('t' + str(i)) for i in range(3)]},
                    labels=dict(support=labels, test=labels, class_order=labels,
                                support_positions=[[2, 0, 1]] * 10))
        rng = np.random.default_rng(174)
        rotations = {arm: {model: np.linalg.qr(rng.normal(size=(5, 2)))[0].T
                           for model in ('qwen', 'harrier', 'gte')} for arm in ('gpa', 'caia')}
        with tempfile.TemporaryDirectory() as folder:
            result = self.core.evaluate_task(data, native({'support': x, 'test': x}), rotations, set(), folder)
            for metric in result['metrics']:
                self.assertEqual(metric['accuracy'], 1.)
            for item in result['diagnostics']['classification_transport']:
                self.assertLessEqual(item['max_abs_logit_error'], 1e-12)
                self.assertTrue(item['all_trial_predictions_equal_native'])
            p = json.loads((Path(folder) / 'predictions/gpa_qwen.json').read_text())
            self.assertEqual(p['trials'][0]['parameters']['random_state'], 42)
            self.assertEqual(p['trials'][0]['parameters']['n_jobs'], 1)
            self.assertEqual(p['trials'][0]['parameters']['max_iter'], 100)

    def test_load_task_reads_native_npz_without_pickle_or_private_imports(self):
        data, values = self.sts_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'task.json').write_text(json.dumps(data))
            np.savez(path / 'native.npz', **{role + '__' + model: x
                     for role, per_model in values.items() for model, x in per_model.items()})
            loaded, arrays = self.core.load_task(path)
            self.assertEqual(loaded, data)
            np.testing.assert_array_equal(arrays['texts']['gte'], values['texts']['gte'])


if __name__ == '__main__':
    unittest.main()
