"""Unit tests for STAIR-REEC graph construction (models/stair_icaart_graph.py).

Covers Acceptance Criteria 5, 6, 7, 8:
5. Undirected scoring mirrored exactly; topology, zero diagonal, and isolated identity blocks.
6. Exact V8 bypass at delta=0 or r=0; neutral calibration for component-constant evidence.
7. Small reference matrices: true degree normalization, eigenspectrum in [-1, 1],
   finite-difference matching of Eq 10 derivative, stale-degree error catching.
8. Finite BSC polynomial L=3 recurrence, Eq 11 perturbation bound, and detached autograd.
"""

import math
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models.stair_icaart_graph import (
    calibrate_semantic_operator,
    compute_mechanism_diagnostics,
    normalize_degree_isolated,
    recover_v8_raw_semantic_graph,
)
from optimizers.stair5_v8_smoother import STAIR5V8Smoother


def test_undirected_mirroring_and_topology_preservation():
    """Verify that W^delta is strictly symmetric, zero diagonal, with identity on isolates."""
    N = 6
    # 6 nodes: nodes 0, 1, 2 connected in triangle, node 3, 4 connected in edge, node 5 isolated
    rows = [0, 1, 0, 2, 1, 2, 3, 4]
    cols = [1, 0, 2, 0, 2, 1, 4, 3]
    vals = [1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 3.0, 3.0]
    W8 = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    upper = sp.triu(W8, k=1).tocoo()
    E_sem = np.column_stack((upper.row, upper.col)).astype(np.int64)
    r = np.array([0.2, 0.4, 0.1, 0.8], dtype=np.float64)

    delta = 1.0
    S_ICA, audit = calibrate_semantic_operator(
        W8_csr=W8, E_sem=E_sem, r=r, delta=delta, SCF_csr=None, eta=0.0
    )

    S_dense = S_ICA.toarray()

    # 1. Exact symmetry
    assert np.allclose(S_dense, S_dense.T, atol=1e-7)

    # 2. Diagonal: 0.0 on connected nodes 0..4, 1.0 on isolated node 5
    for i in range(5):
        assert S_dense[i, i] == 0.0, f"Connected node {i} must have zero diagonal"
    assert S_dense[5, 5] == 1.0, "Isolated node 5 must have identity diagonal entry"

    # 3. Off-diagonal non-zero pattern strictly matches W8
    s_pattern = (np.abs(S_dense) > 0).astype(int)
    np.fill_diagonal(s_pattern, 0)
    w8_pattern = (np.abs(W8.toarray()) > 0).astype(int)
    assert np.array_equal(s_pattern, w8_pattern), "Topology support must not change"


def test_exact_v8_bypass_and_neutral_constant_calibration():
    """Verify bitwise recovery at delta=0 or r=0, and neutral calibration for constant r."""
    N = 4
    # Complete K4 with zero diagonal
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                rows.append(i); cols.append(j); vals.append(1.0)
    W8 = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    upper = sp.triu(W8, k=1).tocoo()
    E_sem = np.column_stack((upper.row, upper.col)).astype(np.int64)
    E = len(E_sem)

    S_v8_cached = normalize_degree_isolated(W8)

    # Case A: delta = 0.0 -> exact bypass
    r_rand = np.random.default_rng(42).uniform(0.1, 0.9, size=E)
    S_res_delta0, audit0 = calibrate_semantic_operator(
        W8, E_sem, r_rand, delta=0.0, SCF_csr=None, eta=0.0, V8_operator_cached=S_v8_cached
    )
    assert audit0["exact_v8_bypass"] is True
    assert audit0["inactive_reason"] == "delta_zero"
    assert S_res_delta0 is S_v8_cached

    # Case B: all r = 0.0 -> exact bypass
    r_zero = np.zeros(E, dtype=np.float64)
    S_res_r0, audit_r0 = calibrate_semantic_operator(
        W8, E_sem, r_zero, delta=1.5, SCF_csr=None, eta=0.0, V8_operator_cached=S_v8_cached
    )
    assert audit_r0["exact_v8_bypass"] is True
    assert audit_r0["inactive_reason"] == "no_replicated_excess"
    assert S_res_r0 is S_v8_cached

    # Case C: component-constant r -> mathematically identical normalized operator
    c_val = 0.42
    r_const = np.full(E, c_val, dtype=np.float64)
    S_res_const, audit_const = calibrate_semantic_operator(
        W8, E_sem, r_const, delta=1.2, SCF_csr=None, eta=0.0
    )
    assert audit_const.get("neutral_calibration") is True
    # Difference from S_v8 must be within floating-point epsilon
    diff = np.max(np.abs(S_res_const.toarray() - S_v8_cached.toarray()))
    assert diff < 1e-6, f"Constant r reweighting must cancel under degree normalization: diff={diff}"


def test_spectral_norm_contraction_and_eigenspectrum():
    """Verify Theorem 1: spectral norm ||S_ICA||_2 <= 1.0 and real eigenvalues in [-1, 1]."""
    rng = np.random.default_rng(20261016)
    N = 12
    # Random symmetric positive adjacency with isolated nodes
    A = rng.uniform(0.1, 2.0, size=(N, N))
    A = (A + A.T) * 0.5
    np.fill_diagonal(A, 0.0)
    # Zero out rows 10, 11 to create isolated nodes
    A[10:, :] = 0.0
    A[:, 10:] = 0.0

    W8 = sp.csr_matrix(A)
    upper = sp.triu(W8, k=1).tocoo()
    E_sem = np.column_stack((upper.row, upper.col)).astype(np.int64)
    r = rng.uniform(0.0, 0.9, size=len(E_sem))

    # Also create a synthetic CF graph
    A_cf = rng.uniform(0.0, 1.0, size=(N, N))
    A_cf = (A_cf + A_cf.T) * 0.5
    np.fill_diagonal(A_cf, 0.0)
    A_cf[10:, :] = 0.0
    A_cf[:, 10:] = 0.0
    SCF = normalize_degree_isolated(sp.csr_matrix(A_cf))

    for delta in (0.5, 1.0, 2.0):
        for eta in (0.0, 0.1, 0.3):
            S_ICA, _ = calibrate_semantic_operator(
                W8, E_sem, r, delta=delta, SCF_csr=SCF, eta=eta
            )
            dense = S_ICA.toarray().astype(np.float64)
            eigenvalues = np.linalg.eigvalsh(dense)
            min_eig = float(np.min(eigenvalues))
            max_eig = float(np.max(eigenvalues))
            spectral_norm = float(np.max(np.abs(eigenvalues)))

            assert min_eig >= -1.0 - 1e-5, f"Eigenvalue < -1: {min_eig}"
            assert max_eig <= 1.0 + 1e-5, f"Eigenvalue > 1: {max_eig}"
            assert spectral_norm <= 1.0 + 1e-5, f"Spectral norm > 1: {spectral_norm}"


def test_finite_difference_equation_10_derivative():
    """Verify Eq 10 derivative matches finite difference: d(S_delta)/ddelta |_0."""
    N = 5
    rng = np.random.default_rng(42)
    A = rng.uniform(0.2, 1.5, size=(N, N))
    A = (A + A.T) * 0.5
    np.fill_diagonal(A, 0.0)

    W8 = sp.csr_matrix(A)
    upper = sp.triu(W8, k=1).tocoo()
    E_sem = np.column_stack((upper.row, upper.col)).astype(np.int64)
    r = rng.uniform(0.1, 0.8, size=len(E_sem))

    # Analytical derivative at delta = 0 (Eq 10):
    # d(S_delta_ij)/ddelta |_0 = S_sem8_ij * (r_ij - 0.5 * rbar_i - 0.5 * rbar_j)
    S_0 = normalize_degree_isolated(W8, dtype=np.float64).toarray()
    d8 = np.asarray(W8.sum(axis=1)).ravel().astype(np.float64)

    r_mat = np.zeros((N, N), dtype=np.float64)
    for idx, (i, j) in enumerate(E_sem):
        r_mat[i, j] = r[idx]
        r_mat[j, i] = r[idx]

    weighted_r = A * r_mat
    rbar = np.sum(weighted_r, axis=1) / d8

    analytical_deriv = np.zeros((N, N), dtype=np.float64)
    for idx, (i, j) in enumerate(E_sem):
        val = S_0[i, j] * (r[idx] - 0.5 * (rbar[i] + rbar[j]))
        analytical_deriv[i, j] = val
        analytical_deriv[j, i] = val

    # Numerical finite difference in FP64: (S_eps - S_0) / eps
    h = 1e-5
    # Construct W_h = W8 * exp(h * r)
    W_h_dense = A * np.exp(h * r_mat)
    np.fill_diagonal(W_h_dense, 0.0)
    S_h = normalize_degree_isolated(sp.csr_matrix(W_h_dense), dtype=np.float64).toarray()

    W_minus = A * np.exp(-h * r_mat)
    S_minus = normalize_degree_isolated(sp.csr_matrix(W_minus), dtype=np.float64).toarray()
    numerical_deriv = (S_h - S_minus) / (2 * h)

    # Check match on off-diagonal candidate edges
    for i, j in E_sem:
        ana = analytical_deriv[i, j]
        num = numerical_deriv[i, j]
        assert abs(ana - num) <= 1e-7, (
            f"Derivative mismatch at ({i}, {j}): analytical={ana}, numerical={num}"
        )


def test_stale_degree_counterexample_fails_contraction():
    """Verify that using stale d8 instead of true actual d_delta breaks the norm bound."""
    # 3-node counterexample from Section 9.3 of report:
    # W = [[0, 100, 1], [100, 0, 1], [1, 1, 0]]
    W = np.array([
        [0.0, 100.0, 1.0],
        [100.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)

    target = np.full(3, 2.0, dtype=np.float64)
    restored = W / W.sum(axis=1)[:, None] * target[:, None]
    sym = (restored + restored.T) / 2
    stale = sym / np.sqrt(target[:, None] * target[None, :])
    stale_norm = np.linalg.norm(stale, ord=2)
    assert stale_norm == pytest.approx(1.1074819683, abs=1e-10)
    assert not np.allclose(sym.sum(axis=1), target)
    true = normalize_degree_isolated(sp.csr_matrix(sym), dtype=np.float64).toarray()
    assert np.linalg.norm(true, ord=2) <= 1 + 1e-12


def test_bsc_polynomial_and_perturbation_bound_equation_11():
    """Verify finite BSC recurrence L=3 matches formula and satisfies Eq 11 perturbation bound."""
    N = 8
    d = 64
    gamma = 0.2
    k = d // 2

    rng = np.random.default_rng(101)
    A1 = rng.uniform(0.1, 1.0, size=(N, N))
    A1 = (A1 + A1.T) * 0.5
    np.fill_diagonal(A1, 0.0)
    S1 = normalize_degree_isolated(sp.csr_matrix(A1))

    A2 = rng.uniform(0.1, 1.0, size=(N, N))
    A2 = (A2 + A2.T) * 0.5
    np.fill_diagonal(A2, 0.0)
    S2 = normalize_degree_isolated(sp.csr_matrix(A2))

    b_k = 0.1 + 0.9 * (k / d) ** gamma
    omega = np.array([1.0, b_k, b_k**2, b_k**3], dtype=np.float64)
    omega = omega / np.sum(omega)

    # C_k = sum_{l=1}^3 l * omega_l
    C_k = float(np.sum(np.arange(1, 4) * omega[1:]))
    assert C_k <= 3.0, "C_k must be <= L = 3"

    # Matrix polynomials P(S1) and P(S2)
    I = np.eye(N, dtype=np.float64)
    s1_dense = S1.toarray().astype(np.float64)
    s2_dense = S2.toarray().astype(np.float64)

    def poly_mat(S):
        S2 = S @ S
        S3 = S2 @ S
        return omega[0] * I + omega[1] * S + omega[2] * S2 + omega[3] * S3

    P1 = poly_mat(s1_dense)
    P2 = poly_mat(s2_dense)

    norm_delta_S = float(np.linalg.norm(s1_dense - s2_dense, ord=2))
    norm_delta_P = float(np.linalg.norm(P1 - P2, ord=2))

    # Eq 11: ||P_k(S_ICA) - P_k(S_8)||_2 <= C_k * ||S_ICA - S_8||_2
    assert norm_delta_P <= C_k * norm_delta_S + 1e-9, (
        f"Eq 11 bound violated: ||delta P|| = {norm_delta_P}, C_k * ||delta S|| = {C_k * norm_delta_S}"
    )

    # Test detached autograd in STAIR5V8Smoother
    s1_tensor = torch.from_numpy(s1_dense).float().to_sparse_csr()
    beta_vec = 0.1 + 0.9 * (torch.arange(d, dtype=torch.float32) / d).pow(gamma)
    smoother = STAIR5V8Smoother(
        operator=lambda X: torch.sparse.mm(s1_tensor, X),
        beta=beta_vec,
        L=3,
    )
    direction = torch.randn(N, d, requires_grad=False)
    filtered = smoother(direction)
    assert filtered.grad_fn is None, "Filtered direction must be detached from autograd graph"
    assert filtered.shape == (N, d)
    dense_reference = torch.zeros_like(direction)
    dense_graph = s1_tensor.to_dense()
    for power in range(4):
        coefficient = beta_vec.pow(power) / sum(beta_vec.pow(q) for q in range(4))
        dense_reference += (torch.linalg.matrix_power(dense_graph, power) @ direction) * coefficient
    torch.testing.assert_close(filtered, dense_reference, rtol=2e-5, atol=2e-6)


def test_permuted_pair_order_binds_weights_to_ids():
    raw = sp.csr_matrix([[0., 2., 4.], [2., 0., 1.], [4., 1., 0.]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    scores = np.array([.1, .6, .2])
    reference, _ = calibrate_semantic_operator(raw, pairs, scores, 1., eta=0.)
    perm = np.array([2, 0, 1])
    actual, _ = calibrate_semantic_operator(raw, pairs[perm], scores[perm], 1., eta=0.)
    np.testing.assert_array_equal(actual.toarray(), reference.toarray())
    with pytest.raises(ValueError, match="each raw semantic edge"):
        calibrate_semantic_operator(raw, pairs[[0, 0, 2]], scores, 1., eta=0.)


def test_component_constant_scores_exactly_reuse_cached_v8():
    raw = sp.csr_matrix([[0., 2., 0., 0.], [2., 0., 0., 0.],
                         [0., 0., 0., 3.], [0., 0., 3., 0.]])
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    cached = normalize_degree_isolated(raw)
    actual, audit = calibrate_semantic_operator(
        raw, pairs, np.array([.2, .7]), 2., V8_operator_cached=cached,
    )
    assert actual is cached
    assert audit["inactive_reason"] == "component_constant_evidence"
    assert audit["relative_operator_dose"] == 0.
    assert audit["s_ica_fingerprint"]


def test_empty_cf_and_invalid_graph_inputs():
    raw = sp.csr_matrix([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    scores = np.array([.1, .3, .2])
    no_cf, _ = calibrate_semantic_operator(raw, pairs, scores, 1., eta=0.)
    empty, audit = calibrate_semantic_operator(raw, pairs, scores, 1., SCF_csr=sp.csr_matrix((3, 3)))
    np.testing.assert_array_equal(empty.toarray(), no_cf.toarray())
    assert audit["effective_eta"] == 0.
    with pytest.raises(ValueError):
        calibrate_semantic_operator(raw, pairs, np.array([.1, np.nan, .2]), 1.)
    with pytest.raises(ValueError):
        normalize_degree_isolated(sp.csr_matrix([[0., 1.], [1. + 1e-12, 0.]]))
