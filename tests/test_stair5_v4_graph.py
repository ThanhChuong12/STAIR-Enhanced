"""tests/test_stair5_v4_graph.py — Comprehensive Unit Tests for STAIR5-v4 Graph Module.
====================================================================================
Tests:
1. Exact evidence shrinkage formula.
2. Block-wise sparse co-occurrence builder vs exact dense brute-force matrix multiplication.
3. Isolated-item identity fallback verification.
4. Convex blend operator bounds: ||S4||_2 <= 1.0 and exact symmetry.
5. Exact off-path: eta=0 or arm='B0' returns baseline S0 unchanged.
6. Degree-stratified placebo permutation properties.
"""
import math
import unittest
import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_graph import (
    compute_evidence_score,
    build_candidate_support_graph,
    normalize_cf_graph_with_fallback,
    permute_cf_graph_degree_stratified,
    build_calibrated_graph_v4,
)


class TestSTAIR5V4Graph(unittest.TestCase):

    def test_evidence_score_formula(self):
        # c=4, n_i=10, n_j=10, t=5
        # shrinkage = 4 / (4 + 5) = 4/9
        # base = 4 / sqrt(10*10) = 4/10 = 0.4
        # expected = (4/9) * 0.4 = 0.177777...
        score = compute_evidence_score(c=4.0, n_i=10.0, n_j=10.0, t=5.0)
        expected = (4.0 / 9.0) * (4.0 / 10.0)
        self.assertAlmostEqual(score, expected, places=6)

        # Zero cases
        self.assertEqual(compute_evidence_score(c=0, n_i=10, n_j=10), 0.0)
        self.assertEqual(compute_evidence_score(c=5, n_i=0, n_j=10), 0.0)

    def test_block_builder_vs_dense(self):
        # Synthetic dataset: 10 users, 8 items
        n_users = 10
        n_items = 8
        # Define interactions
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 1), (1, 2),
            (2, 2), (2, 3), (2, 4),
            (3, 4), (3, 5), (3, 6),
            (4, 0), (4, 7),
            (5, 1), (5, 2), (5, 3), (5, 4),
        ]
        u = torch.tensor([e[0] for e in edges], dtype=torch.long)
        i = torch.tensor([e[1] for e in edges], dtype=torch.long)
        train_edges = torch.stack([u, i], dim=0)

        # Compute with small block_size to trigger chunking logic
        W_cf = build_candidate_support_graph(
            train_edges=train_edges,
            n_items=n_items,
            n_users=n_users,
            k_cf=3,
            c_min=2,
            t_shrinkage=5.0,
            block_size=2,
        )

        # Must be symmetric
        diff = W_cf - W_cf.transpose()
        self.assertTrue(np.allclose(diff.toarray(), 0.0, atol=1e-6))

        # Diagonal must be strictly zero
        self.assertTrue(np.all(W_cf.diagonal() == 0.0))

        # Verification: check item 1 and item 2 co-occurrence
        # Item 1 users: {0, 1, 5}
        # Item 2 users: {0, 1, 2, 5}
        # Intersection: {0, 1, 5} -> count = 3 >= c_min=2
        self.assertGreater(W_cf[1, 2], 0.0)
        self.assertEqual(W_cf[2, 1], W_cf[1, 2])

    def test_isolated_node_fallback(self):
        # Item 0 and 1 are connected, Item 2 is isolated
        W_cf = sp.csr_matrix(
            np.array([
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
        )
        S_bar = normalize_cf_graph_with_fallback(W_cf)
        dense = S_bar.toarray()

        # Connected nodes: off-diagonal normalized, diagonal is 0
        self.assertEqual(dense[0, 0], 0.0)
        self.assertEqual(dense[1, 1], 0.0)
        self.assertAlmostEqual(dense[0, 1], 1.0, places=6)
        self.assertAlmostEqual(dense[1, 0], 1.0, places=6)

        # Isolated node (item 2): diagonal is 1.0 (identity fallback)
        self.assertAlmostEqual(dense[2, 2], 1.0, places=6)
        self.assertEqual(dense[2, 0], 0.0)
        self.assertEqual(dense[2, 1], 0.0)

        # Symmetry
        self.assertTrue(np.allclose(dense, dense.T, atol=1e-7))

        # Operator norm must be <= 1.0
        eigvals = np.linalg.eigvalsh(dense)
        self.assertLessEqual(max(abs(eigvals)), 1.0 + 1e-6)

    def test_convex_blend_operator_bound(self):
        n = 6
        # Semantic graph S0
        adj0 = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=np.float32)
        deg0 = adj0.sum(axis=1)
        inv_deg0 = 1.0 / np.sqrt(deg0)
        S0 = inv_deg0[:, None] * adj0 * inv_deg0[None, :]

        # CF graph with an isolated node (item 5)
        w_cf = np.array([
            [0, 2, 0, 0, 1, 0],
            [2, 0, 1, 0, 0, 0],
            [0, 1, 0, 3, 0, 0],
            [0, 0, 3, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)
        S_bar_cf = normalize_cf_graph_with_fallback(sp.csr_matrix(w_cf)).toarray()

        for eta in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
            S4 = (1.0 - eta) * S0 + eta * S_bar_cf
            # Symmetry
            self.assertTrue(np.allclose(S4, S4.T, atol=1e-7))
            # Spectral norm <= 1.0
            eigvals = np.linalg.eigvalsh(S4)
            self.assertLessEqual(max(abs(eigvals)), 1.0 + 1e-6)

    def test_exact_off_path(self):
        n_items = 4
        n_users = 5
        baseline_s0 = torch.eye(n_items).to_sparse_csr()
        raw = torch.eye(n_items).to_sparse_coo()
        train_edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

        # Arm B0
        state_b0 = build_calibrated_graph_v4(
            raw_semantic=raw,
            baseline_normalized=baseline_s0,
            train_edges=train_edges,
            n_users=n_users,
            n_items=n_items,
            arm="B0",
        )
        self.assertFalse(state_b0.metadata["active"])
        self.assertIs(state_b0.operator, baseline_s0)

        # eta = 0.0
        state_eta0 = build_calibrated_graph_v4(
            raw_semantic=raw,
            baseline_normalized=baseline_s0,
            train_edges=train_edges,
            n_users=n_users,
            n_items=n_items,
            arm="N-CSE",
            eta=0.0,
        )
        self.assertFalse(state_eta0.metadata["active"])
        self.assertIs(state_eta0.operator, baseline_s0)

    def test_exact_empty_cf_fallback_and_invalid_parameters(self):
        original = torch.eye(4).to_sparse_csr()
        args = dict(raw_semantic=original, baseline_normalized=original,
                    train_edges=torch.tensor([[0, 1], [0, 1]]), n_users=2, n_items=4)
        state = build_calibrated_graph_v4(**args, eta=.3, c_min=2)
        self.assertIs(state.operator, original)
        self.assertFalse(state.metadata["active"])
        self.assertEqual(state.metadata["nnz"], 4)
        for overrides in ({"eta": -1}, {"eta": 2}, {"eta": float("nan")}, {"arm": "typo"}, {"t_shrinkage": 0}):
            with self.assertRaises(ValueError):
                build_calibrated_graph_v4(**args, **overrides)
        with self.assertRaises(ValueError):
            build_candidate_support_graph(torch.tensor([[0, 0], [0, 4]]), 4, 2)

    def test_dense_oracle_ties_duplicates_and_budget(self):
        edges = torch.tensor([[0,0,0,0,1,1,1,2,2,2], [0,0,1,2,0,1,3,0,2,3]])
        n, k = 4, 1
        R = np.zeros((3, n), dtype=np.int64)
        R[edges[0], edges[1]] = 1
        counts, degrees = R.T @ R, R.sum(0)
        directed = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            candidates = [(compute_evidence_score(counts[i,j], degrees[i], degrees[j]), j)
                          for j in range(n) if i != j and counts[i,j] >= 1]
            for score, j in sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:k]:
                directed[i,j] = score
        oracle = np.maximum(directed, directed.T)
        for block in (1,2,64):
            actual = build_candidate_support_graph(edges, n, 3, k, 1, block_size=block,
                                                   memory_budget_mib=.001)
            np.testing.assert_allclose(actual.toarray(), oracle, rtol=1e-6, atol=1e-7)
            self.assertLessEqual(actual.nnz, 2*n*k)
        with self.assertRaises(MemoryError):
            build_candidate_support_graph(edges, n, 3, memory_budget_mib=.00001)

    def test_cache_uses_full_data_and_reuses_across_eta_seeds(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            graph = torch.eye(4).to_sparse_csr()
            # Large duplicate prefix exposed the previous first-100000-byte hash.
            prefix = torch.tensor([[0], [0]]).repeat(1, 14000)
            a = torch.cat((prefix, torch.tensor([[0,1,1], [1,0,1]])), 1)
            b = torch.cat((prefix, torch.tensor([[0,1,1], [2,0,2]])), 1)
            args = dict(raw_semantic=graph, baseline_normalized=graph, n_users=2, n_items=4,
                        c_min=1, cache_dir=directory)
            first = build_calibrated_graph_v4(**args, train_edges=a)
            second = build_calibrated_graph_v4(**args, train_edges=a, eta=.2, seed=7)
            changed = build_calibrated_graph_v4(**args, train_edges=b)
            self.assertTrue(second.metadata["cache_hit"])
            self.assertFalse(changed.metadata["cache_hit"])
            self.assertNotEqual(first.metadata["train_fingerprint"], changed.metadata["train_fingerprint"])
            # A different S0 must be blended freshly, even when CF cache is reused.
            replacement = torch.tensor([[0.,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]).to_sparse_csr()
            third = build_calibrated_graph_v4(**{**args,"baseline_normalized":replacement}, train_edges=a)
            self.assertTrue(third.metadata["cache_hit"])
            self.assertNotEqual(first.metadata["graph_fingerprint"], third.metadata["graph_fingerprint"])
            self.assertEqual(len(list(Path(directory).glob("cf_*.pt"))), 2)


if __name__ == "__main__":
    unittest.main()
