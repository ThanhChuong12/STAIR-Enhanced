"""tests/test_stair5_v52_graph.py — Algebraic & Unit Verification Suite for STAIR5-v5.2.
=====================================================================================
Verification requirements:
1. Toy Graph Wedge Accumulation:
   - Chain (A-B-C): m_AC = 1
   - Diamond (A-B-C, A-D-C): m_AC = 2
   - Direct 1-hop edges strictly excluded from candidate set.
2. Mass Budget Invariant:
   - Assert sum_j (W_add)_ij <= nu * sum_j (W_1)_ij for every row i on random sparse graphs.
3. Operator Contraction:
   - Spectral norm / radius rho(S_star) <= 1.0 + 1e-6 and rho(S_5.2) <= 1.0 + 1e-6 on toy graphs.
4. S_0 Exclusion Test:
   - Candidate pairs overlapping existing S_0 non-zeros are strictly filtered out when exclude_s0=True.
5. Isolated Node Preservation:
   - Nodes with zero degree in W_star retain exact identity diagonal in S_star.
"""
import math
import unittest
import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v52_graph import (
    build_mutual_seed_graph,
    enumerate_path_candidates,
    apply_edge_and_mass_budgets,
    normalize_cf_graph_with_fallback,
    build_calibrated_graph_v52,
)


class TestSTAIR5V52Graph(unittest.TestCase):

    def test_toy_wedge_accumulation_chain_and_diamond(self):
        """Chain (0-1-2) should have m_{0,2} = 1.

        Diamond (0-1-2, 0-3-2) should have m_{0,2} = 2.
        Existing 1-hop edges (0,1), (1,2) must be excluded from candidates.
        """
        # Chain graph: 3 nodes
        # 0 -- 1 -- 2
        W_chain = sp.csr_matrix(
            np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ], dtype=np.float64)
        )
        S0_empty = sp.csr_matrix((3, 3), dtype=np.float64)
        T_chain = build_mutual_seed_graph(W_chain, k_seed=10)
        cands_chain, stats_chain = enumerate_path_candidates(
            T=T_chain, W_1=W_chain, S_0=S0_empty, m_min=1, t_path=1.0, exclude_s0=True
        )
        dense_chain = cands_chain.toarray()

        # m_{0,2} = 1. d_1^T = 2.
        # p_{0,2} = (T_01 * T_12) / d_1^T = (1.0 * 1.0) / 2.0 = 0.5
        # a_{0,2} = (1 / (1 + 1)) * 0.5 = 0.25
        self.assertAlmostEqual(dense_chain[0, 2], 0.25, places=6)
        self.assertAlmostEqual(dense_chain[2, 0], 0.25, places=6)

        # Existing 1-hop edges (0,1), (1,2) must be strictly 0
        self.assertEqual(dense_chain[0, 1], 0.0)
        self.assertEqual(dense_chain[1, 0], 0.0)
        self.assertEqual(dense_chain[1, 2], 0.0)
        self.assertEqual(dense_chain[2, 1], 0.0)
        # Self-loops must be 0
        self.assertEqual(dense_chain[0, 0], 0.0)
        self.assertEqual(dense_chain[1, 1], 0.0)
        self.assertEqual(dense_chain[2, 2], 0.0)

        # Diamond graph: 4 nodes
        # 0 connects to 1 and 3
        # 2 connects to 1 and 3
        W_diamond = sp.csr_matrix(
            np.array([
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ], dtype=np.float64)
        )
        S0_empty4 = sp.csr_matrix((4, 4), dtype=np.float64)
        T_diamond = build_mutual_seed_graph(W_diamond, k_seed=10)
        cands_diamond, stats_diamond = enumerate_path_candidates(
            T=T_diamond, W_1=W_diamond, S_0=S0_empty4, m_min=1, t_path=1.0, exclude_s0=True
        )
        dense_diamond = cands_diamond.toarray()

        # m_{0,2} = 2 (intermediates 1 and 3).
        # d_1^T = 2, d_3^T = 2.
        # p_{0,2} = (1*1)/2 + (1*1)/2 = 1.0
        # a_{0,2} = (2 / (2 + 1)) * 1.0 = 2/3 ≈ 0.666667
        self.assertAlmostEqual(dense_diamond[0, 2], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(dense_diamond[2, 0], 2.0 / 3.0, places=6)

        # In diamond, (1, 3) also has 2 paths through 0 and 2:
        self.assertAlmostEqual(dense_diamond[1, 3], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(dense_diamond[3, 1], 2.0 / 3.0, places=6)

        # Existing 1-hop edges must be strictly 0
        self.assertEqual(dense_diamond[0, 1], 0.0)
        self.assertEqual(dense_diamond[0, 3], 0.0)
        self.assertEqual(dense_diamond[2, 1], 0.0)
        self.assertEqual(dense_diamond[2, 3], 0.0)

    def test_mass_budget_invariant_random_sparse_graphs(self):
        """Assert sum_j (W_add)_ij <= nu * sum_j (W_1)_ij for every row i."""
        rng = np.random.RandomState(42)
        n_items = 50
        # Generate random symmetric W_1
        mat = rng.uniform(0.0, 1.0, (n_items, n_items))
        mask = rng.uniform(0.0, 1.0, (n_items, n_items)) < 0.15
        mat = mat * mask
        np.fill_diagonal(mat, 0.0)
        W_1 = sp.csr_matrix(np.maximum(mat, mat.T))

        # Generate random candidates C
        c_mat = rng.uniform(0.0, 2.0, (n_items, n_items))
        c_mask = rng.uniform(0.0, 1.0, (n_items, n_items)) < 0.25
        c_mat = c_mat * c_mask * (W_1.toarray() == 0)
        np.fill_diagonal(c_mat, 0.0)
        C = sp.csr_matrix(np.maximum(c_mat, c_mat.T))

        for nu in [0.2, 0.5, 0.8]:
            W_add, telemetry = apply_edge_and_mass_budgets(
                candidates=C,
                W_1=W_1,
                k_add=3,
                beta_edges=0.25,
                nu=nu,
                scale_calibration=True,
            )
            d1 = np.asarray(W_1.sum(1)).ravel()
            added_sums = np.asarray(W_add.sum(1)).ravel()

            # Every row i must satisfy: added_sums[i] <= nu * d1[i] + 1e-12
            diff = added_sums - nu * d1
            self.assertTrue(np.all(diff <= 1e-10), f"Mass budget violated for nu={nu}: max excess = {diff.max()}")

            # W_add must be strictly symmetric
            sym_diff = W_add - W_add.T
            self.assertTrue(np.allclose(sym_diff.toarray(), 0.0, atol=1e-10))

    def test_operator_contraction_spectral_norm(self):
        """Verify spectral norm ||S_star||_2 <= 1.0 + 1e-6 and ||S_5.2||_2 <= 1.0 + 1e-6."""
        rng = np.random.RandomState(123)
        n_items = 20

        # Construct connected component and isolated node
        mat = rng.uniform(0.0, 1.0, (n_items, n_items))
        mask = rng.uniform(0.0, 1.0, (n_items, n_items)) < 0.3
        mask = mask | mask.T
        mat = mat * mask
        mat = (mat + mat.T) / 2.0
        # Make node 19 strictly isolated
        mat[19, :] = 0.0
        mat[:, 19] = 0.0
        np.fill_diagonal(mat, 0.0)
        W_star = sp.csr_matrix(mat)

        S_star = normalize_cf_graph_with_fallback(W_star)
        dense_s_star = S_star.toarray()

        # Compute largest singular value / absolute eigenvalue
        eigenvalues = np.linalg.eigvalsh(dense_s_star)
        spectral_radius_star = np.max(np.abs(eigenvalues))
        self.assertLessEqual(spectral_radius_star, 1.0 + 1e-6)

        # Isolated node 19 must have exact 1 on diagonal
        self.assertAlmostEqual(dense_s_star[19, 19], 1.0, places=6)
        self.assertEqual(np.count_nonzero(dense_s_star[19, :]), 1)

        # Build S0 and check S_5.2 contraction
        mat_s0 = rng.uniform(0.0, 1.0, (n_items, n_items))
        mask_s0 = rng.uniform(0.0, 1.0, (n_items, n_items)) < 0.4
        mask_s0 = mask_s0 | mask_s0.T
        mat_s0 = mat_s0 * mask_s0
        mat_s0 = (mat_s0 + mat_s0.T) / 2.0
        np.fill_diagonal(mat_s0, 0.0)
        S0 = normalize_cf_graph_with_fallback(sp.csr_matrix(mat_s0))
        dense_s0 = S0.toarray()

        dense_s52 = 0.9 * dense_s0 + 0.1 * dense_s_star
        eigs_52 = np.linalg.eigvalsh(dense_s52)
        spectral_radius_52 = np.max(np.abs(eigs_52))
        self.assertLessEqual(spectral_radius_52, 1.0 + 1e-6)

    def test_s0_exclusion_test(self):
        """Ensure candidate pairs overlapping existing S_0 non-zeros are filtered out."""
        # Diamond graph: 0-1-2 and 0-3-2
        # Wedge paths connect (0, 2)
        W_1 = sp.csr_matrix(
            np.array([
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ], dtype=np.float64)
        )
        # S0 already contains edge (0, 2)
        S0 = sp.csr_matrix(
            np.array([
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ], dtype=np.float64)
        )
        T = build_mutual_seed_graph(W_1, k_seed=10)

        # With exclude_s0=True, edge (0, 2) must be excluded!
        cands_filtered, stats_filtered = enumerate_path_candidates(
            T=T, W_1=W_1, S_0=S0, m_min=1, t_path=1.0, exclude_s0=True
        )
        dense_filtered = cands_filtered.toarray()
        self.assertEqual(dense_filtered[0, 2], 0.0)
        self.assertEqual(dense_filtered[2, 0], 0.0)
        # But edge (1, 3) was not in S0, so it remains!
        self.assertGreater(dense_filtered[1, 3], 0.0)
        self.assertGreater(dense_filtered[3, 1], 0.0)
        self.assertGreaterEqual(stats_filtered["excluded_s0_count"], 2)

        # With exclude_s0=False (C-Overlap), edge (0, 2) is kept!
        cands_overlap, stats_overlap = enumerate_path_candidates(
            T=T, W_1=W_1, S_0=S0, m_min=1, t_path=1.0, exclude_s0=False
        )
        dense_overlap = cands_overlap.toarray()
        self.assertGreater(dense_overlap[0, 2], 0.0)
        self.assertGreater(dense_overlap[2, 0], 0.0)

    def test_isolated_node_preservation_identity(self):
        """Verify nodes with zero degree in W_star retain exact identity diagonal in S_star."""
        W_star = sp.csr_matrix(
            np.array([
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # Node 2 is isolated
            ], dtype=np.float64)
        )
        S_star = normalize_cf_graph_with_fallback(W_star)
        dense = S_star.toarray()

        self.assertEqual(dense[2, 0], 0.0)
        self.assertEqual(dense[2, 1], 0.0)
        self.assertEqual(dense[0, 2], 0.0)
        self.assertEqual(dense[1, 2], 0.0)
        self.assertEqual(dense[2, 2], 1.0)  # Exact identity fallback

    def test_global_edge_budget_clipping(self):
        """Verify that global edge budget retains at most floor(beta_edges * nnz(W_1) / 2) pairs."""
        n_items = 10
        # W_1 has 8 directed edges (4 pairs)
        W_1 = sp.csr_matrix(
            np.array([
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float64)
        )
        w1_nnz = W_1.nnz  # 8
        # Beta_edges = 0.25 -> B_pairs = floor(0.25 * 8 / 2) = 1 pair!
        expected_max_pairs = int(math.floor(0.25 * w1_nnz / 2.0))
        self.assertEqual(expected_max_pairs, 1)

        # Candidate matrix with 6 pairs
        cand = sp.csr_matrix(
            np.array([
                [0, 0, 0, 5, 4, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float64)
        )
        W_add, telemetry = apply_edge_and_mass_budgets(
            candidates=cand,
            W_1=W_1,
            k_add=3,
            beta_edges=0.25,
            nu=0.5,
            scale_calibration=True,
        )
        # Should keep exactly 1 pair (2 directed entries), which must be the highest score (0, 3)
        self.assertEqual(telemetry["budgeted_pairs"], 1)
        self.assertEqual(W_add.nnz, 2)
        dense = W_add.toarray()
        self.assertGreater(dense[0, 3], 0.0)
        self.assertGreater(dense[3, 0], 0.0)
        self.assertEqual(dense[0, 4], 0.0)

    def test_scale_calibration_kappa(self):
        """Verify scale calibration computes kappa = median(W_1 > 0) / median(Q > 0)."""
        W_1 = sp.csr_matrix(
            np.array([
                [0.0, 10.0],
                [10.0, 0.0],
            ], dtype=np.float64)
        )
        cand = sp.csr_matrix(
            np.array([
                [0.0, 2.0],
                [2.0, 0.0],
            ], dtype=np.float64)
        )
        W_add, telemetry = apply_edge_and_mass_budgets(
            candidates=cand,
            W_1=W_1,
            k_add=3,
            beta_edges=1.0,
            nu=1.0,
            scale_calibration=True,
        )
        # median(W_1) = 10.0, median(cand) = 2.0 -> kappa = 5.0
        self.assertAlmostEqual(telemetry["scale_kappa"], 5.0, places=6)

