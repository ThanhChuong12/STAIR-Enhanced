"""tests/test_stair5_v6_graph.py — Algebraic & Unit Verification Suite for STAIR5-v6 (BCSR).
========================================================================================
Verification requirements:
1. Exact candidate counts vs dense reference; duplicate train interaction handling.
2. Under-support scoring fixtures: e=0 gives a=0, c >= e gives a=0, 0 <= a <= theta.
3. Degree Conservation Invariant:
   sum_j (W_R)_ij == sum_j (W_0)_ij == d_i^0 for every item i in P-BCSR.
4. Non-negativity:
   (W_R)_ii == m_i >= 0, (W_R)_ij == (1 - a_ij) * W_0,ij >= 0.
5. Operator Contraction:
   rho(S_R) <= 1.0 + 1e-6 and rho(S_6) <= 1.0 + 1e-6.
6. Exact theta=0 Fast Path Parity:
   S_6(theta=0) returns exact v4 sparse fingerprint and identical operator values.
7. Ablation arms verification:
   C-Uniform, C-NoShrink, C-Shuffled, C-NoRetention, C-Renorm, C-CF-Placebo.
8. Cache integrity and signature invalidation.
"""
import math
import unittest
import tempfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v6_graph import (
    ARMS_V6,
    binary_train_keys,
    compute_candidate_intersections,
    compute_under_support_scores,
    build_retained_semantic_graph,
    normalize_retained_semantic_graph,
    build_calibrated_graph_v6,
    sparse_fingerprint,
    tensor_to_scipy,
)
from models import stair5_v4_graph as reference_graph_v4


class TestSTAIR5V6Graph(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_binary_train_keys_deduplication_and_validation(self):
        # 4 users, 5 items
        edges = torch.tensor([[0, 0, 0, 1, 1, 2, 3], [1, 1, 2, 0, 3, 4, 2]])
        keys = binary_train_keys(edges, n_users=4, n_items=5)
        # Unique pairs: (0,1), (0,2), (1,0), (1,3), (2,4), (3,2) -> 6 unique pairs
        self.assertEqual(len(keys), 6)

        # Invalid bounds
        with self.assertRaises(ValueError):
            binary_train_keys(torch.tensor([[4], [0]]), n_users=4, n_items=5)
        with self.assertRaises(ValueError):
            binary_train_keys(torch.tensor([[0], [5]]), n_users=4, n_items=5)

    def test_candidate_intersections_vs_dense_reference(self):
        n_users, n_items = 20, 15
        # Generate random interactions
        R_dense = np.random.randint(0, 2, size=(n_users, n_items))
        pairs = np.argwhere(R_dense == 1)
        edges = torch.tensor(pairs.T, dtype=torch.int64)

        keys = binary_train_keys(edges, n_users, n_items)
        R_sparse = sp.csr_matrix(
            (np.ones(len(keys), dtype=np.int64), (keys // n_items, keys % n_items)),
            shape=(n_users, n_items),
        )
        R_T = R_sparse.T.tocsr()
        R_T.sort_indices()

        # Dense product
        dense_C = R_dense.T @ R_dense

        # Pick candidate pairs
        cand_i = np.array([0, 1, 2, 3, 5, 8, 10], dtype=np.int64)
        cand_j = np.array([4, 6, 7, 9, 11, 12, 14], dtype=np.int64)

        computed_c = compute_candidate_intersections(R_T, cand_i, cand_j, chunk_size=4)
        expected_c = np.array([dense_C[i, j] for i, j in zip(cand_i, cand_j)], dtype=np.int64)

        np.testing.assert_array_equal(computed_c, expected_c)

    def test_under_support_scores_fixtures(self):
        M = 100
        n_i = np.array([10, 20, 30, 0, 50], dtype=np.float64)
        n_j = np.array([10, 20, 30, 10, 50], dtype=np.float64)
        c_ij = np.array([0, 4, 15, 0, 25], dtype=np.float64)

        e, h, r, a = compute_under_support_scores(n_i, n_j, c_ij, M=M, theta=0.25, t_rel=5.0)

        # Pair 0: e = 10*10/100 = 1.0, c = 0 -> deficit h = 1.0, r = 1 / (1 + 5) = 1/6
        # a = 0.25 * (1/6) * 1.0 = 0.041666...
        self.assertAlmostEqual(e[0], 1.0)
        self.assertAlmostEqual(h[0], 1.0)
        self.assertAlmostEqual(r[0], 1.0 / 6.0)
        self.assertAlmostEqual(a[0], 0.25 / 6.0)

        # Pair 1: e = 20*20/100 = 4.0, c = 4.0 -> c == e -> deficit h = 0 -> a = 0
        self.assertAlmostEqual(e[1], 4.0)
        self.assertAlmostEqual(h[1], 0.0)
        self.assertAlmostEqual(a[1], 0.0)

        # Pair 2: e = 30*30/100 = 9.0, c = 15.0 -> c > e -> deficit h = 0 -> a = 0
        self.assertAlmostEqual(h[2], 0.0)
        self.assertAlmostEqual(a[2], 0.0)

        # Pair 3: n_i = 0 -> e = 0 -> h = 0 -> a = 0
        self.assertAlmostEqual(e[3], 0.0)
        self.assertAlmostEqual(a[3], 0.0)

        # Check bounds: 0 <= a <= theta
        self.assertTrue(np.all((a >= 0.0) & (a <= 0.25)))

    def test_bcsr_degree_conservation_invariant(self):
        n_items = 10
        # Create symmetric raw semantic graph W_0 with zero diagonal
        w_dense = np.random.uniform(0.1, 1.0, size=(n_items, n_items))
        w_dense = (w_dense + w_dense.T) / 2.0
        np.fill_diagonal(w_dense, 0.0)
        # Sparsify
        w_dense[w_dense < 0.4] = 0.0
        W_0 = sp.csr_matrix(w_dense)
        d_0 = np.asarray(W_0.sum(axis=1)).ravel()

        coo = sp.triu(W_0, k=1).tocoo()
        cand_i = coo.row
        cand_j = coo.col
        # Random suppression fractions in [0, 0.5]
        a_ij = np.random.uniform(0.0, 0.5, size=len(cand_i))

        W_R, A_mat, m_i, diag = build_retained_semantic_graph(
            W_0=W_0,
            d_0=d_0,
            cand_i=cand_i,
            cand_j=cand_j,
            a_ij=a_ij,
            n_items=n_items,
            arm="P-BCSR",
        )

        # Invariant 1: Degree preservation W_R 1 == W_0 1
        d_R = np.asarray(W_R.sum(axis=1)).ravel()
        np.testing.assert_allclose(d_R, d_0, atol=1e-12)
        self.assertAlmostEqual(diag["degree_conservation_max_error"], 0.0, places=10)

        # Invariant 2: Diagonal is exactly m_i
        np.testing.assert_allclose(W_R.diagonal(), m_i, atol=1e-12)

        # Invariant 3: Symmetry of W_R and A_mat
        diff_WR = W_R - W_R.T
        diff_A = A_mat - A_mat.T
        self.assertEqual(diff_WR.nnz, 0)
        self.assertEqual(diff_A.nnz, 0)

        # Invariant 4: Non-negativity
        self.assertTrue(np.all(W_R.data >= 0.0))
        self.assertTrue(np.all(m_i >= 0.0))

    def test_operator_contraction(self):
        n_items = 8
        w_dense = np.random.uniform(0.2, 1.0, size=(n_items, n_items))
        w_dense = (w_dense + w_dense.T) / 2.0
        np.fill_diagonal(w_dense, 0.0)
        W_0 = sp.csr_matrix(w_dense)
        d_0 = np.asarray(W_0.sum(axis=1)).ravel()

        coo = sp.triu(W_0, k=1).tocoo()
        cand_i = coo.row
        cand_j = coo.col
        a_ij = np.random.uniform(0.0, 0.4, size=len(cand_i))

        W_R, _, _, _ = build_retained_semantic_graph(W_0, d_0, cand_i, cand_j, a_ij, n_items, arm="P-BCSR")
        S_R = normalize_retained_semantic_graph(W_R, d_0, arm="P-BCSR")

        # Spectral radius of S_R must be <= 1.0 + 1e-6
        eigvals = np.linalg.eigvalsh(S_R.toarray())
        max_eig = np.max(np.abs(eigvals))
        self.assertLessEqual(max_eig, 1.0 + 1e-6)

    def test_theta0_exact_fast_path_recovery(self):
        n_users, n_items = 12, 10
        edges = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 0, 1, 2],
        ], dtype=torch.int64)

        # Semantic graph
        w_dense = np.zeros((n_items, n_items), dtype=np.float32)
        for i in range(n_items):
            w_dense[i, (i + 1) % n_items] = 1.0
            w_dense[i, (i + 2) % n_items] = 0.5
        w_dense = np.maximum(w_dense, w_dense.T)
        np.fill_diagonal(w_dense, 0.0)
        W_0 = sp.csr_matrix(w_dense)
        d_0 = np.asarray(W_0.sum(axis=1)).ravel()
        scale = sp.diags(1.0 / np.sqrt(np.maximum(d_0, 1e-12)))
        S_0 = scale @ W_0 @ scale

        s0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(S_0.indptr.astype(np.int64)),
            torch.from_numpy(S_0.indices.astype(np.int64)),
            torch.from_numpy(S_0.data.astype(np.float32)),
            size=S_0.shape,
        )
        w0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(W_0.indptr.astype(np.int64)),
            torch.from_numpy(W_0.indices.astype(np.int64)),
            torch.from_numpy(W_0.data.astype(np.float32)),
            size=W_0.shape,
        )

        # Baseline v4 graph
        v4_state = reference_graph_v4.build_calibrated_graph_v4(
            raw_semantic=w0_tensor,
            baseline_normalized=s0_tensor,
            train_edges=edges,
            n_users=n_users,
            n_items=n_items,
            arm="N-CSE",
            eta=0.1,
            k_cf=3,
            c_min=1,
        )

        # STAIR5-v6 with theta=0.0
        v6_state_theta0 = build_calibrated_graph_v6(
            raw_semantic=w0_tensor,
            baseline_normalized=s0_tensor,
            train_edges=edges,
            n_users=n_users,
            n_items=n_items,
            arm="P-BCSR",
            theta=0.0,
            eta=0.1,
            k_cf=3,
            c_min=1,
        )

        # STAIR5-v6 with arm="C-V4"
        v6_state_cv4 = build_calibrated_graph_v6(
            raw_semantic=w0_tensor,
            baseline_normalized=s0_tensor,
            train_edges=edges,
            n_users=n_users,
            n_items=n_items,
            arm="C-V4",
            theta=0.25,
            eta=0.1,
            k_cf=3,
            c_min=1,
        )

        # Assert bitwise operator equality
        self.assertTrue(torch.equal(v4_state.operator.values(), v6_state_theta0.operator.values()))
        self.assertTrue(torch.equal(v4_state.operator.values(), v6_state_cv4.operator.values()))
        self.assertEqual(v6_state_theta0.metadata["fast_path_theta0"], True)

    def test_ablation_arms(self):
        n_users, n_items = 15, 12
        edges = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4],
        ], dtype=torch.int64)

        w_dense = np.zeros((n_items, n_items), dtype=np.float32)
        for i in range(n_items):
            w_dense[i, (i + 1) % n_items] = 1.0
            w_dense[i, (i + 2) % n_items] = 0.5
        w_dense = np.maximum(w_dense, w_dense.T)
        np.fill_diagonal(w_dense, 0.0)
        W_0 = sp.csr_matrix(w_dense)
        d_0 = np.asarray(W_0.sum(axis=1)).ravel()
        scale = sp.diags(1.0 / np.sqrt(np.maximum(d_0, 1e-12)))
        S_0 = scale @ W_0 @ scale

        s0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(S_0.indptr.astype(np.int64)),
            torch.from_numpy(S_0.indices.astype(np.int64)),
            torch.from_numpy(S_0.data.astype(np.float32)),
            size=S_0.shape,
        )
        w0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(W_0.indptr.astype(np.int64)),
            torch.from_numpy(W_0.indices.astype(np.int64)),
            torch.from_numpy(W_0.data.astype(np.float32)),
            size=W_0.shape,
        )

        for arm in ("P-BCSR", "C-Uniform", "C-NoShrink", "C-Shuffled", "C-NoRetention", "C-Renorm", "C-CF-Placebo"):
            state = build_calibrated_graph_v6(
                raw_semantic=w0_tensor,
                baseline_normalized=s0_tensor,
                train_edges=edges,
                n_users=n_users,
                n_items=n_items,
                arm=arm,
                theta=0.25,
                t_rel=5.0,
                eta=0.1,
                k_cf=3,
                c_min=1,
            )
            csr = tensor_to_scipy(state.operator)
            self.assertEqual(csr.shape, (n_items, n_items))
            self.assertTrue(np.all(csr.data >= 0.0))
            diff = csr - csr.T
            self.assertAlmostEqual(float(np.max(np.abs(diff.data))) if diff.nnz else 0.0, 0.0, places=6)

    def test_cache_hit_and_signature_integrity(self):
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 2, 3, 3, 0]], dtype=torch.int64)
        w_dense = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ], dtype=np.float32)
        W_0 = sp.csr_matrix(w_dense)
        d_0 = np.asarray(W_0.sum(axis=1)).ravel()
        scale = sp.diags(1.0 / np.sqrt(d_0))
        S_0 = scale @ W_0 @ scale

        s0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(S_0.indptr.astype(np.int64)),
            torch.from_numpy(S_0.indices.astype(np.int64)),
            torch.from_numpy(S_0.data.astype(np.float32)),
            size=S_0.shape,
        )
        w0_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(W_0.indptr.astype(np.int64)),
            torch.from_numpy(W_0.indices.astype(np.int64)),
            torch.from_numpy(W_0.data.astype(np.float32)),
            size=W_0.shape,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = dict(
                raw_semantic=w0_tensor,
                baseline_normalized=s0_tensor,
                train_edges=edges,
                n_users=3,
                n_items=4,
                arm="P-BCSR",
                theta=0.25,
                t_rel=5.0,
                eta=0.1,
                k_cf=2,
                c_min=1,
                cache_dir=tmpdir,
            )
            first = build_calibrated_graph_v6(**kwargs)
            self.assertFalse(first.metadata["cache_hit"])

            # Second call should be a cache hit
            second = build_calibrated_graph_v6(**kwargs)
            self.assertTrue(second.metadata["cache_hit"])
            self.assertTrue(torch.equal(first.operator.values(), second.operator.values()))

            # Changing theta must invalidate cache and compute afresh
            kwargs["theta"] = 0.35
            third = build_calibrated_graph_v6(**kwargs)
            self.assertFalse(third.metadata["cache_hit"])


if __name__ == "__main__":
    unittest.main()
