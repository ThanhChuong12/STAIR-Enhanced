"""Taxonomy integrity and requested-pair CAE algebraic contracts."""
import hashlib
import json
import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models.stair6_v1_categories import (
    CategoryData, CategoryEvidence, assert_category_provenance,
    binary_user_category_incidence, build_category_edge_scores, build_memberships,
    canonicalize_paths, category_association_scores, load_category_data,
    pool_category_scores, requested_category_pairs,
)


def _taxonomy():
    return [[['Root', 'A']], [['Root', 'B']], [['Root', 'C']], [['Root']]]


def test_exact_independence_never_generates_spurious_positive_evidence():
    # n_A=12, n_B=18, n_AB=9, N*=24: 9*24 == 12*18.
    # Four-log subtraction evaluates to +4.44e-16 on the tested runtime.
    incidence = np.zeros((24, 3), dtype=np.int64)
    incidence[:9, :2] = 1
    incidence[9:12, 0] = 1
    incidence[12:21, 1] = 1
    incidence[21:, 2] = 1
    evidence = category_association_scores(sp.csr_matrix(incidence), np.array([1], dtype=np.int64))
    assert evidence.counts.tolist() == [9]
    assert evidence.scores.tolist() == [0.0]
    assert evidence.metadata["positive_pairs"] == 0


def _inputs():
    memberships, _ = build_memberships(_taxonomy())
    edges = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 4],
                          [0, 0, 1, 0, 1, 2, 2, 3]], dtype=torch.int64)
    graph = sp.csr_matrix(([1., 1., 1., 1.], ([0, 1, 0, 2], [1, 0, 2, 0])), shape=(4, 4))
    return memberships, edges, graph


def _write_files(tmp_path, *, mapping=None, records=None, manifest_updates=None):
    mapping_path = tmp_path / 'mapping.json'
    metadata_path = tmp_path / 'categories.jsonl'
    manifest_path = tmp_path / 'manifest.json'
    mapping_path.write_text(json.dumps(mapping if mapping is not None else ['ASIN0', 'ASIN1', 'ASIN2']), encoding='utf-8')
    if records is None:
        records = [{'item_id': 0, 'asin': 'ASIN0', 'categories': [['Baby', 'Feeding']]},
                   {'item_id': 1, 'asin': 'ASIN1', 'categories': [['Baby']]}]
    metadata_path.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'schema_version': 1, 'dataset': 'ToyDataset', 'n_items': 3,
        'mapping_verified': True, 'mapping_source': 'Original provider preprocessing row order',
        'metadata_source': {'source': 'Amazon2014 provider snapshot', 'snapshot': '2014'},
        'files': {'mapping_sha256': digest(mapping_path), 'metadata_sha256': digest(metadata_path)},
        'baseline': {'features': {'text.pkl': 'a' * 64}, 'splits': {'train.txt': 'b' * 64}},
    }
    manifest.update(manifest_updates or {})
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    return mapping_path, metadata_path, manifest_path


def test_terminal_paths_normalize_without_losing_parent_identity():
    paths = canonicalize_paths([[' Electronics '], [' Electronics ', 'Cameras'],
        ['Electronics', 'Cameras', ' Accessories '], ['Electronics', 'Computers', 'Accessories'],
        ['Electronics', 'Cameras', 'Accessories'], ['Shop', ' Products '],
        ['Baby', ' Cafe\u0301  Accessories ']])
    assert paths == (('Baby', 'Café Accessories'),
                     ('Electronics', 'Cameras', 'Accessories'),
                     ('Electronics', 'Computers', 'Accessories'))


@pytest.mark.parametrize('paths', ['Electronics', [['']], [[]], [['Root', 3]], [None]])
def test_malformed_explicit_category_paths_rejected(paths):
    with pytest.raises(ValueError):
        canonicalize_paths(paths)


def test_membership_rows_normalized_and_root_only_neutral():
    matrix, paths = build_memberships([[['Root', 'A'], ['Root', 'B']], [['Root']], []])
    assert matrix.dtype == np.float32
    assert paths == (('Root', 'A'), ('Root', 'B'))
    np.testing.assert_array_equal(np.asarray(matrix.sum(axis=1)).ravel(), [1, 0, 0])
    np.testing.assert_array_equal(matrix.data, [.5, .5])
    empty, vocabulary = build_memberships([[['Root']], []])
    assert empty.shape == (2, 0) and not empty.nnz and not vocabulary


def test_train_incidence_binary_deduplicated_and_population_excludes_unobserved():
    memberships, edges, graph = _inputs()
    incidence = binary_user_category_incidence(edges, 5, memberships)
    assert incidence.dtype == np.int64
    assert np.all(incidence.data == 1)
    np.testing.assert_array_equal(np.asarray(incidence.sum(axis=1)).ravel(), [2, 2, 1, 1, 0])
    requested = requested_category_pairs(graph, memberships)
    evidence = category_association_scores(incidence, requested)
    assert evidence.metadata['population_users'] == 4
    assert evidence.pair_keys.tolist() == [1, 2]
    assert evidence.counts.tolist() == [2, 0]
    expected = (2 / 7) * math.tanh(math.log(2))
    np.testing.assert_allclose(evidence.scores, [expected, 0], rtol=1e-6)
    scores = pool_category_scores(graph, memberships, evidence)
    weighted = graph.copy()
    weighted.data = scores
    assert (weighted - weighted.T).nnz == 0
    assert weighted[0, 1] == pytest.approx(expected)
    assert weighted[0, 2] == 0


def test_minimum_count_and_shrinkage_not_ochiai():
    memberships, edges, graph = _inputs()
    incidence = binary_user_category_incidence(edges, 5, memberships)
    pairs = requested_category_pairs(graph, memberships)
    assert not category_association_scores(incidence, pairs, min_count=3).scores.any()
    assert category_association_scores(incidence, pairs, shrinkage=0).scores[0] == pytest.approx(.6)


def test_diagonal_membership_mass_not_renormalized():
    memberships, _ = build_memberships([[['Root', 'A'], ['Root', 'B']],
                                       [['Root', 'A'], ['Root', 'B']], []])
    graph = sp.csr_matrix(([1., 1., 1., 1., 1.], ([0, 0, 1, 0, 2], [0, 1, 0, 2, 0])), shape=(3, 3))
    evidence = CategoryEvidence(np.array([1], dtype=np.int64), np.array([2]), np.array([.8], dtype=np.float32), {})
    scores = pool_category_scores(graph, memberships, evidence)
    weighted = graph.copy()
    weighted.data = scores
    assert weighted[0, 1] == pytest.approx(.4)
    assert weighted[1, 0] == pytest.approx(.4)
    assert weighted[0, 0] == weighted[0, 2] == weighted[2, 0] == 0


def test_no_requested_dense_clique_and_explicit_workspace_limits():
    memberships, edges, graph = _inputs()
    assert requested_category_pairs(graph, memberships).tolist() == [1, 2]
    with pytest.raises(MemoryError, match='pairs'):
        requested_category_pairs(graph, memberships, max_pairs=1)
    with pytest.raises(MemoryError, match='combinations'):
        requested_category_pairs(graph, memberships, max_combinations=1)


def test_empty_incidence_categories_and_graph_are_valid_neutral():
    memberships, _ = build_memberships([[], [], []])
    edges = torch.empty((2, 0), dtype=torch.int64)
    incidence = binary_user_category_incidence(edges, 2, memberships)
    pairs = requested_category_pairs(sp.eye(3, format='csr'), memberships)
    evidence = category_association_scores(incidence, pairs)
    assert evidence.metadata['population_users'] == 0
    assert incidence.shape == (2, 0)
    assert not pool_category_scores(sp.eye(3, format='csr'), memberships, evidence).any()


@pytest.mark.parametrize('case', ['weighted_incidence', 'diagonal_query', 'bad_count', 'bad_shrinkage'])
def test_invalid_association_inputs_fail_closed(case):
    incidence = sp.csr_matrix([[1, 1], [0, 1]], dtype=np.int64)
    query = np.array([1], dtype=np.int64)
    kwargs = {}
    if case == 'weighted_incidence':
        incidence.data[0] = 2
    elif case == 'diagonal_query':
        query = np.array([0], dtype=np.int64)
    elif case == 'bad_count':
        kwargs['min_count'] = True
    else:
        kwargs['shrinkage'] = float('nan')
    with pytest.raises(ValueError):
        category_association_scores(incidence, query, **kwargs)


def test_loader_partial_metadata_neutral_and_expected_provenance(tmp_path):
    files = _write_files(tmp_path)
    data = load_category_data(*files, 3, {'dataset': 'ToyDataset'})
    assert data.item_asins == ('ASIN0', 'ASIN1', 'ASIN2')
    assert data.paths == (('Baby', 'Feeding'),)
    assert data.audit['informative_coverage'] == pytest.approx(1 / 3)
    assert data.audit['missing_metadata_rows'] == 1
    assert_category_provenance(data, {'features': {'text.pkl': 'A' * 64}, 'splits': {'train.txt': 'b' * 64}})
    for wrong in ({'dataset': 'AnotherDataset'}, {'features': {'text.pkl': 'c' * 64}},
                  {'splits': {'test.txt': 'b' * 64}}):
        with pytest.raises(ValueError, match='provenance'):
            assert_category_provenance(data, wrong)


def test_mapping_records_may_be_out_of_order_but_ids_are_contiguous(tmp_path):
    files = _write_files(tmp_path, mapping=[{'item_id': 2, 'asin': 'ASIN2'},
        {'item_id': 0, 'asin': 'ASIN0'}, {'item_id': 1, 'asin': 'ASIN1'}])
    assert load_category_data(*files, 3).item_asins == ('ASIN0', 'ASIN1', 'ASIN2')


@pytest.mark.parametrize('mapping', [['ASIN0', 'ASIN0', 'ASIN2'], ['ASIN0', ' ASIN1', 'ASIN2'],
    [{'item_id': 0, 'asin': 'ASIN0'}, {'item_id': 0, 'asin': 'ASIN1'}, {'item_id': 2, 'asin': 'ASIN2'}],
    [{'item_id': True, 'asin': 'ASIN0'}, {'item_id': 1, 'asin': 'ASIN1'}, {'item_id': 2, 'asin': 'ASIN2'}]])
def test_loader_rejects_non_bijective_or_ambiguous_mapping(tmp_path, mapping):
    files = _write_files(tmp_path, mapping=mapping)
    with pytest.raises(ValueError):
        load_category_data(*files, 3)


@pytest.mark.parametrize('records', [
    [{'item_id': 0, 'asin': 'WRONG', 'categories': []}],
    [{'item_id': 3, 'asin': 'ASIN3', 'categories': []}],
    [{'item_id': 0, 'asin': 'ASIN0'}],
    [{'item_id': 0, 'asin': 'ASIN0', 'categories': []}] * 2,
])
def test_loader_rejects_malformed_explicit_metadata(tmp_path, records):
    with pytest.raises(ValueError):
        load_category_data(*_write_files(tmp_path, records=records), 3)


@pytest.mark.parametrize('updates', [
    {'mapping_verified': False}, {'n_items': 4}, {'schema_version': True},
    {'metadata_source': {'source': 'provider'}}, {'baseline': {'features': {}, 'splits': {}}},
])
def test_loader_requires_verified_complete_manifest_contract(tmp_path, updates):
    with pytest.raises(ValueError):
        load_category_data(*_write_files(tmp_path, manifest_updates=updates), 3)


def test_content_hash_tamper_rejected_and_fingerprint_revision_bound(tmp_path):
    files = _write_files(tmp_path)
    original = load_category_data(*files, 3).metadata_fingerprint
    files[1].write_text(files[1].read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA256'):
        load_category_data(*files, 3)
    files = _write_files(tmp_path, manifest_updates={'mapping_source': 'Another verified provider revision'})
    assert load_category_data(*files, 3).metadata_fingerprint != original


def test_wrapper_builds_only_train_evidence_and_returns_cpu_array():
    memberships, edges, graph = _inputs()
    data = CategoryData(memberships, (('Root', 'A'), ('Root', 'B'), ('Root', 'C')),
        ('A0', 'A1', 'A2', 'A3'), {'informative_coverage': .75}, 'toy-fingerprint', {})
    scores, audit = build_category_edge_scores(graph, edges, 5, 4, data)
    assert isinstance(scores, np.ndarray) and scores.dtype == np.float32
    assert audit['population_users'] == 4 and audit['active_semantic_edges'] == 2
    assert audit['metadata_fingerprint'] == 'toy-fingerprint'
