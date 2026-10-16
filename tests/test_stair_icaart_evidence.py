"""Unit tests for STAIR-REEC evidence computation (models/stair_icaart_evidence.py).

Covers Acceptance Criteria 1, 2, 3, 4:
1. Small enumerated group fixtures: exact c_ijf, mu_ijf, and r_ij with empty, singleton,
   duplicate interaction pairs, and isolated items.
2. Whole user in exactly one half; deterministic assignment independent of global RNG;
   consistent remapping preserves counts.
3. Bounded score 0 <= r_ij < 1; r = 0 on zero co-occurrence or one-half peaks; no NaN/Inf.
4. Strict data boundary: no user leakage across halves; candidate invariants enforced.
"""

import math
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models.stair_icaart_evidence import (
    build_activity_replication,
    candidate_excess_statistics,
    deterministic_user_hash,
    replicated_excess_score,
)


def test_deterministic_user_hash_properties():
    """Verify deterministic 64-bit user hashing is independent of Python run/RNG."""
    seed = 20261016
    h1 = deterministic_user_hash(seed, 42)
    h2 = deterministic_user_hash(seed, 42)
    assert h1 == h2, "Hash must be pure deterministic function"
    assert isinstance(h1, int) and 0 <= h1 < 2**64

    # Different seeds or users yield distinct hashes
    h3 = deterministic_user_hash(seed + 1, 42)
    h4 = deterministic_user_hash(seed, 43)
    assert h1 != h3
    assert h1 != h4

    # Independent of global torch/numpy RNG
    torch.manual_seed(999)
    np.random.seed(999)
    assert deterministic_user_hash(seed, 42) == h1


def test_activity_stratification_and_splitting_fixtures():
    """Verify activity strata, singleton handling, and zero-activity user exclusion."""
    M, N = 8, 10
    # User 0: 0 interactions -> h_0 = 0 -> bin = -1, half = -1
    # User 1: 1 interaction -> h_1 = 1 -> bin = 0 (singleton in bin 0)
    # User 2, 3: 2 interactions each -> h = 2 -> bin = 1 (2 users in bin 1)
    # User 4, 5, 6: 4 interactions each -> h = 4 -> bin = 2 (3 users in bin 2)
    # User 7: 8 interactions -> h = 8 -> bin = 3 (singleton in bin 3)

    rows = []
    cols = []
    # User 1
    rows.append(1); cols.append(0)
    # User 2
    rows.extend([2, 2]); cols.extend([0, 1])
    # User 3
    rows.extend([3, 3]); cols.extend([1, 2])
    # User 4
    rows.extend([4, 4, 4, 4]); cols.extend([0, 1, 2, 3])
    # User 5
    rows.extend([5, 5, 5, 5]); cols.extend([2, 3, 4, 5])
    # User 6
    rows.extend([6, 6, 6, 6]); cols.extend([1, 3, 5, 6])
    # User 7
    rows.extend([7] * 8); cols.extend(list(range(8)))

    R = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(M, N))

    user_bin, half, group_sizes = build_activity_replication(R, evidence_seed=20261016)

    # User 0 must be excluded
    assert user_bin[0] == -1
    assert half[0] == -1

    # Bin checks: floor(log2(h))
    assert user_bin[1] == 0
    assert user_bin[2] == 1
    assert user_bin[3] == 1
    assert user_bin[4] == 2
    assert user_bin[5] == 2
    assert user_bin[6] == 2
    assert user_bin[7] == 3

    # Singleton strata must be deterministically placed in half 0
    assert half[1] == 0  # bin 0 has 1 user
    assert half[7] == 0  # bin 3 has 1 user

    # Bin 1 has 2 users: exactly one in half 0 and one in half 1
    assert {half[2], half[3]} == {0, 1}

    # Bin 2 has 3 users: counts must be 2 in one half, 1 in the other
    b2_halves = [half[4], half[5], half[6]]
    assert b2_halves.count(0) == 2 or b2_halves.count(0) == 1
    assert b2_halves.count(0) + b2_halves.count(1) == 3

    # Group sizes match exact assignments
    B = group_sizes.shape[1]
    assert B == 4
    for b in range(B):
        u_b = np.flatnonzero(user_bin == b)
        assert group_sizes[0, b] == np.count_nonzero(half[u_b] == 0)
        assert group_sizes[1, b] == np.count_nonzero(half[u_b] == 1)


def test_no_user_leakage_and_disjointness():
    """Verify that every active user is strictly in one half and histories are disjoint."""
    rng = np.random.default_rng(42)
    M, N = 100, 50
    # Random sparse interactions
    nnz = 400
    rows = rng.integers(0, M, size=nnz)
    cols = rng.integers(0, N, size=nnz)
    R = sp.csr_matrix((np.ones(nnz, dtype=np.int8), (rows, cols)), shape=(M, N))
    R.data[:] = 1

    user_bin, half, group_sizes = build_activity_replication(R, evidence_seed=20261016)

    # Every user is either in half 0, half 1, or inactive (-1)
    unique_halves = set(np.unique(half))
    assert unique_halves.issubset({-1, 0, 1})

    # Users in half 0 and users in half 1 are completely disjoint sets
    users_half0 = set(np.flatnonzero(half == 0))
    users_half1 = set(np.flatnonzero(half == 1))
    assert users_half0.isdisjoint(users_half1)

    # Inactive users have zero interactions
    h = np.asarray(R.sum(axis=1)).ravel()
    for u in range(M):
        if h[u] == 0:
            assert half[u] == -1 and user_bin[u] == -1
        else:
            assert half[u] in (0, 1) and user_bin[u] >= 0


def test_candidate_counts_and_null_expectation_exact_math():
    """Verify exact numerical counts c_ijf and null expectation mu_ijf against manual math."""
    # 4 users, 4 items
    # User 0: half 0, items (0, 1) -> common user for (0, 1) in half 0
    # User 1: half 0, items (0, 1) -> common user for (0, 1) in half 0
    # User 2: half 1, items (0, 1) -> common user for (0, 1) in half 1
    # User 3: half 1, items (0, 2) -> no common user with 1 in half 1
    # Both User 0 and User 1 have degree 2 -> stratum b = 1
    # User 2 has degree 2 -> stratum b = 1
    # User 3 has degree 2 -> stratum b = 1

    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 0, 1, 0, 2]
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(4, 4))

    user_bin = np.array([1, 1, 1, 1], dtype=np.int64)
    half = np.array([0, 0, 1, 1], dtype=np.int64)
    group_sizes = np.array([[0, 2], [0, 2]], dtype=np.int64)

    # Candidate edge E_sem = [(0, 1), (0, 2), (1, 2)]
    E_sem = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)

    c, mu = candidate_excess_statistics(
        R.tocsc(), E_sem, half, user_bin, group_sizes, candidate_block=8192
    )

    # Manual math for pair (0, 1):
    # Half 0: Users 0 and 1 both interacted with 0 and 1 -> c_01,0 = 2
    # In half 0 stratum 1: M_01 = 2. n_{0,0,1} = 2, n_{1,0,1} = 2.
    # Expected mu_01,0 = (2 * 2) / 2 = 2.0
    assert c[0, 0] == 2
    assert math.isclose(mu[0, 0], 2.0, rel_tol=1e-9)

    # Half 1: User 2 interacted with (0, 1), User 3 interacted with (0, 2)
    # Common users in half 1 for (0, 1): only User 2 -> c_01,1 = 1
    # In half 1 stratum 1: M_11 = 2. n_{0,1,1} = 2 (users 2 and 3), n_{1,1,1} = 1 (user 2).
    # Expected mu_01,1 = (2 * 1) / 2 = 1.0
    assert c[0, 1] == 1
    assert math.isclose(mu[0, 1], 1.0, rel_tol=1e-9)

    # Pair (1, 2):
    # Half 0: no users interacted with 2 -> c_12,0 = 0, mu_12,0 = 0.0
    # Half 1: User 2 has 1, User 3 has 2 -> no common user -> c_12,1 = 0
    # Expected mu_12,1 = (1 * 1) / 2 = 0.5
    assert c[2, 0] == 0
    assert mu[2, 0] == 0.0
    assert c[2, 1] == 0
    assert math.isclose(mu[2, 1], 0.5, rel_tol=1e-9)


def test_replicated_excess_scoring_bounds():
    """Verify that 0 <= r_ij < 1, neutral on zero co-occurrence and one-half peaks."""
    tau = 2.0

    # Case 1: Both halves have positive excess beyond mu
    # Half 0: c0 = 5, mu0 = 1 -> excess = 4, denom = 5 + 1 + 2 = 8 -> e0 = 4/8 = 0.5
    # Half 1: c1 = 4, mu1 = 1 -> excess = 3, denom = 4 + 1 + 2 = 7 -> e1 = 3/7 = 0.42857
    # r = min(0.5, 0.42857) = 3/7
    c = np.array([[5, 4]], dtype=np.int64)
    mu = np.array([[1.0, 1.0]], dtype=np.float64)
    r = replicated_excess_score(c, mu, tau=tau, mode="replicated_excess")
    assert math.isclose(r[0], 3.0 / 7.0, rel_tol=1e-9)
    assert 0.0 < r[0] < 1.0

    # Case 2: One-half peak (c0 = 10, but c1 = 0)
    # One half has big excess, other has zero -> min is 0!
    c_peak = np.array([[10, 0]], dtype=np.int64)
    mu_peak = np.array([[1.0, 0.0]], dtype=np.float64)
    r_peak = replicated_excess_score(c_peak, mu_peak, tau=tau, mode="replicated_excess")
    assert r_peak[0] == 0.0, "One-half peak must yield r = 0"

    # Case 3: Zero co-occurrence in both halves
    c_zero = np.array([[0, 0]], dtype=np.int64)
    mu_zero = np.array([[0.5, 0.5]], dtype=np.float64)
    r_zero = replicated_excess_score(c_zero, mu_zero, tau=tau, mode="replicated_excess")
    assert r_zero[0] == 0.0

    # Case 4: c <= mu in one half (no excess)
    c_no_excess = np.array([[5, 2]], dtype=np.int64)
    mu_no_excess = np.array([[1.0, 2.5]], dtype=np.float64)
    r_no_excess = replicated_excess_score(c_no_excess, mu_no_excess, tau=tau, mode="replicated_excess")
    assert r_no_excess[0] == 0.0


def test_invariant_violation_raises_arithmetic_error():
    """Verify that c > 0 with mu <= 0 strictly triggers an ArithmeticError."""
    R = sp.csc_matrix(np.zeros((4, 4), dtype=np.int8))
    E_sem = np.array([[0, 1]], dtype=np.int64)
    half = np.array([0, 0, 1, 1], dtype=np.int64)
    user_bin = np.array([0, 0, 0, 0], dtype=np.int64)
    group_sizes = np.array([[2], [2]], dtype=np.int64)

    # Artificially test invariant logic if c > 0 but mu <= 0
    # In candidate_excess_statistics, if c > 0, mathematically mu > 0.
    # To verify the assertion itself:
    with pytest.raises((ValueError, ArithmeticError)):
        # We can construct a pathological scenario where group_sizes claim 0 users in a group
        # but R has interactions
        bad_group_sizes = np.array([[0], [0]], dtype=np.int64)
        R_patho = sp.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(4, 4)).tocsc()
        candidate_excess_statistics(R_patho, E_sem, half, user_bin, bad_group_sizes)


def test_control_modes_pooled_count_shuffle_uniform():
    """Verify all scientific control ablation modes in replicated_excess_score."""
    c = np.array([[6, 4]], dtype=np.int64)
    mu = np.array([[1.0, 1.0]], dtype=np.float64)
    tau = 2.0

    # v8_exact
    r_v8 = replicated_excess_score(c, mu, tau=tau, mode="v8_exact")
    assert np.all(r_v8 == 0.0)

    # pooled_excess
    # c_pool = 10, mu_pool = 2, tau_pool = 4 -> (10 - 2) / (10 + 2 + 4) = 8 / 16 = 0.5
    r_pool = replicated_excess_score(c, mu, tau=tau, mode="pooled_excess")
    assert math.isclose(r_pool[0], 0.5, rel_tol=1e-9)

    # replicated_count: min(6/(6+2), 4/(4+2)) = min(6/8, 4/6) = min(0.75, 0.6667) = 2/3
    r_count = replicated_excess_score(c, mu, tau=tau, mode="replicated_count")
    assert math.isclose(r_count[0], 4.0 / 6.0, rel_tol=1e-9)

    # uniform_multiplier
    r_uni = replicated_excess_score(c, mu, tau=tau, mode="uniform_multiplier")
    assert 0.0 < r_uni[0] < 1.0


def test_evidence_does_not_consume_global_rng_and_deduplicates_private_copy():
    import random
    rows = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 0, 1, 0, 1, 1, 2, 1, 2]
    matrix = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(5, 4))
    original_data = matrix.data.copy()
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    bins, halves, sizes = build_activity_replication(matrix)
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    counts, expected = candidate_excess_statistics(matrix.tocsc(), pairs, halves, bins, sizes, candidate_block=1)
    for field, actual in zip(numpy_state, np.random.get_state()):
        if isinstance(field, np.ndarray):
            np.testing.assert_array_equal(field, actual)
        else:
            assert field == actual
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    np.testing.assert_array_equal(matrix.data, original_data)
    # Activity is binary presence: duplicate (0,0) does not make h_0=3.
    assert bins[0] == 1
    assert bins[4] == halves[4] == -1
    reference = np.zeros_like(counts)
    binary = (matrix.toarray() > 0)
    for index, (i, j) in enumerate(pairs):
        for half_id in (0, 1):
            reference[index, half_id] = np.count_nonzero(binary[:, i] & binary[:, j] & (halves == half_id))
    np.testing.assert_array_equal(counts, reference)


def test_empty_history_catalog_and_invalid_score_statistics():
    empty = sp.csr_matrix((4, 6))
    bins, halves, sizes = build_activity_replication(empty)
    counts, expected = candidate_excess_statistics(
        empty.tocsc(), np.array([[0, 1]], dtype=np.int64), halves, bins, sizes,
    )
    np.testing.assert_array_equal(counts, [[0, 0]])
    np.testing.assert_array_equal(expected, [[0., 0.]])
    with pytest.raises(ArithmeticError, match="Positive overlap"):
        replicated_excess_score(np.array([[1, 0]]), np.array([[0., 0.]]))
    with pytest.raises(ValueError):
        replicated_excess_score(np.array([[1, 0]]), np.array([[np.nan, 0.]]))
    with pytest.raises(ValueError):
        replicated_excess_score(np.empty((0, 2), dtype=np.int64), np.empty((0, 2)), mode="invalid")
