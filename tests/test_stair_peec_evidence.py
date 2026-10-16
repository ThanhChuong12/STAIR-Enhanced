"""PEEC estimator and legacy-bridge contracts. Definitions only; run explicitly."""
import numpy as np
import pytest
import scipy.sparse as sp

from models.stair_icaart_evidence import (
    build_activity_strata, candidate_pooled_statistics, candidate_excess_statistics,
    pooled_activity_score, replicated_excess_score, degree_stratified_permutation,
)
from stair_icaart_config import validate_icaart_config, evidence_identity


@pytest.fixture
def presence():
    # Six active users in one activity bin plus one inactive user.
    return sp.csr_matrix(np.array([
        [1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,1],
        [1,0,1,0], [0,1,0,1], [0,0,0,0],
    ], dtype=np.int8))


def test_full_null_is_not_sum_of_half_nulls(presence):
    pairs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
    bins, sizes = build_activity_strata(presence)
    assert bins.tolist() == [1,1,1,1,1,1,-1]
    assert sizes.tolist() == [0,6]
    counts, mu = candidate_pooled_statistics(presence.tocsc(), pairs, bins, sizes, 2)
    assert counts.tolist() == [2,1,0,0,1,2]
    np.testing.assert_array_equal(mu, np.full(6, 1.5))
    half = np.array([0,0,1,1,0,1,-1])
    groups = np.array([[0,3], [0,3]])
    c_half, mu_half = candidate_excess_statistics(presence.tocsc(), pairs, half, bins, groups)
    np.testing.assert_array_equal(c_half.sum(1), counts)
    assert mu_half[0].sum() == 2.0
    assert mu[0] == 1.5
    score = pooled_activity_score(counts, mu)
    assert score[0] == pytest.approx(.5 / 7.5)
    assert np.count_nonzero(score) == 2


def test_duplicate_storage_does_not_change_presence_or_mutate_input(presence):
    # Construct deliberately uncoalesced duplicate CSR storage.
    duplicates = sp.csr_matrix((np.array([1,2,1,1]), np.array([0,0,1,1]),
                                np.array([0,3,4])), shape=(2,4))
    original = duplicates.data.copy()
    bins, sizes = build_activity_strata(duplicates)
    assert bins.tolist() == [1,0]
    np.testing.assert_array_equal(duplicates.data, original)
    pairs = np.array([[0,1]])
    c, mu = candidate_pooled_statistics(duplicates.tocsc(), pairs, bins, sizes)
    assert c.tolist() == [1]
    assert mu.tolist() == [1.0]


def test_empty_histories_and_singleton_strata():
    empty = sp.csr_matrix((3,4), dtype=np.int8)
    bins, sizes = build_activity_strata(empty)
    c, mu = candidate_pooled_statistics(empty.tocsc(), np.array([[0,1]]), bins, sizes)
    assert bins.tolist() == [-1,-1,-1] and sizes.size == 0
    assert c.tolist() == mu.tolist() == [0]
    assert pooled_activity_score(c, mu).tolist() == [0]
    one = sp.csr_matrix([[1,1,0]])
    bins, sizes = build_activity_strata(one)
    c, mu = candidate_pooled_statistics(one.tocsc(), np.array([[0,1]]), bins, sizes)
    assert c.tolist() == mu.tolist() == [1]
    assert pooled_activity_score(c, mu).tolist() == [0]


def test_full_statistics_invariant_to_user_permutation_and_pair_order(presence):
    pairs = np.array([[2,3], [0,1], [0,2]])
    bins, sizes = build_activity_strata(presence)
    reference = candidate_pooled_statistics(presence.tocsc(), pairs, bins, sizes, 1)
    reordered = presence[[6,4,0,3,1,5,2]]
    bins, sizes = build_activity_strata(reordered)
    actual = candidate_pooled_statistics(reordered.tocsc(), pairs, bins, sizes, 99)
    for a, b in zip(reference, actual):
        np.testing.assert_array_equal(a, b)


def test_legacy_min_and_pooled_half_equations_remain_distinct():
    c = np.array([[2,0], [1,1]])
    mu = np.array([[.25,.25], [.25,.25]])
    replicated = replicated_excess_score(c, mu, 2, "replicated_excess")
    pooled = replicated_excess_score(c, mu, 2, "pooled_excess")
    assert replicated[0] == 0
    np.testing.assert_allclose(pooled, np.array([1.5/6.5, 1.5/6.5]))
    with pytest.raises(ValueError, match="pooled_activity_score"):
        replicated_excess_score(c, mu, mode="pooled_activity_excess")


def test_count_control_and_pooled_regularizer():
    c, mu = np.array([0,1,3]), np.array([0.,.1,1.])
    np.testing.assert_allclose(pooled_activity_score(c, mu), [0,.9/5.1,2/8])
    np.testing.assert_allclose(pooled_activity_score(c, mu, mode="pooled_activity_count"), [0,.2,3/7])
    with pytest.raises(ArithmeticError):
        pooled_activity_score(np.array([1]), np.array([0.]))


def test_pooled_placebo_preserves_scores_inside_train_degree_strata():
    degrees = np.repeat(np.arange(1,6), 4)
    pairs = np.array([[0,4], [1,5], [2,6], [3,7], [8,12], [9,13], [10,14], [11,15]])
    counts, mu = np.arange(1,9), np.full(8, .1)
    scores = pooled_activity_score(counts, mu)
    shuffled = pooled_activity_score(counts, mu, mode="pooled_activity_shuffle",
                                    degrees=degrees, E_sem=pairs, placebo_seed=7)
    for indices in (slice(0,4), slice(4,8)):
        np.testing.assert_array_equal(np.sort(scores[indices]), np.sort(shuffled[indices]))
    np.testing.assert_array_equal(shuffled, degree_stratified_permutation(scores, degrees, pairs, 7))


def test_full_config_ignores_only_partition_seed():
    first = validate_icaart_config(dict(evidence_mode="pooled_activity_excess", evidence_halves=0, evidence_seed=1))
    second = validate_icaart_config(dict(evidence_mode="pooled_activity_excess", evidence_halves=0, evidence_seed=99))
    assert first == second
    assert first["evidence_seed"] is None
    assert evidence_identity(first)["null_scope"] == "full_activity"
    with pytest.raises(ValueError, match="evidence_halves=0"):
        validate_icaart_config(dict(evidence_mode="pooled_activity_excess", evidence_halves=2))
    with pytest.raises(ValueError, match="evidence_halves=2"):
        validate_icaart_config(dict(evidence_mode="pooled_excess", evidence_halves=0))


@pytest.mark.parametrize("pairs", [np.array([[1,0]]), np.array([[0,4]]), np.array([[0,1],[0,1]])])
def test_invalid_candidates_rejected(presence, pairs):
    bins, sizes = build_activity_strata(presence)
    with pytest.raises(ValueError):
        candidate_pooled_statistics(presence.tocsc(), pairs, bins, sizes)
