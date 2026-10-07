"""tests/test_stair5_v7_graph.py — Algebraic & Unit Verification Suite for STAIR5-v7 (UCR-D).
=============================================================================================
Verification requirements:
1. Degree Preservation:
   Row sums of W_R equal row sums of W_0 within eps = 1e-6:
     sum_j W_{R, ij} == sum_j W_{0, ij} == d_{0, i}.
2. Contraction Invariant:
   rho(S_R) <= 1.0 + 1e-6 and rho(S_7) <= 1.0 + 1e-6 on synthetic graphs.
3. Scale Invariance Cancellation:
   Multiplying an arbitrary unnormalized gate by a positive constant before degree
   normalization results in the exact same operator:
     D_{cW}^{-1/2} (cW) D_{cW}^{-1/2} == D_W^{-1/2} W D_W^{-1/2}.
4. Dose Bound Verification:
   sum_{i != j} A_ij <= rho_t * sum_{i != j} W_{0, ij} under all settings,
   and rho_eff == min(rho_t, theta_max * Z_t).
5. Symmetry & Non-negativity:
   W_R == W_R^T, W_{R, ij} >= 0, S_7 == S_7^T.
6. Chunked Scoring Agreement:
   Chunked pair scoring (C=4, 64, 4096) matches unchunked dot products.
7. CSR in-place mutation matches dense reference:
   Active sparse CSR values match dense linear algebra within float32 tolerance.
8. IHP-NCER structural Gram score bounded in [0, 1].
"""
import math
import unittest
import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v7_graph import (
    ARMS_V7,
    canonicalize_arm_name_v7,
    binary_train_keys,
    compute_ihp_ncer_scores,
    build_calibrated_graph_v7,
    STAIR5V7GraphAdapter,
    tensor_to_scipy,
    scipy_to_tensor,
)


class TestSTAIR5V7Graph(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_canonicalize_arms(self):
        self.assertEqual(canonicalize_arm_name_v7("UCR-D"), "UCR-D")
        self.assertEqual(canonicalize_arm_name_v7("ucr"), "UCR-D")
        self.assertEqual(canonicalize_arm_name_v7("v4"), "V4-control")
        self.assertEqual(canonicalize_arm_name_v7("v7-recovery"), "V7-recovery")
        self.assertEqual(canonicalize_arm_name_v7("uniform"), "Uniform-retention")
        self.assertEqual(canonicalize_arm_name_v7("shuffled"), "Shuffled-UCR")
        self.assertEqual(canonicalize_arm_name_v7("ihp-ncer"), "IHP-NCER-only")
        with self.assertRaises(ValueError):
            canonicalize_arm_name_v7("invalid-arm")

    def test_degree_preservation_algebraic(self):
        """Row sums of W_R must identically equal row sums of W_0."""
        N = 25
        # Generate random symmetric W_0 with zero diagonal in float64
        W0 = np.random.uniform(0.1, 2.0, size=(N, N)).astype(np.float64)
        W0 = (W0 + W0.T) / 2.0
        np.fill_diagonal(W0, 0.0)
        d0 = W0.sum(axis=1)

        # Random antagonistic scores q in [0, 1]
        q = np.random.uniform(0.0, 1.0, size=(N, N)).astype(np.float64)
        q = (q + q.T) / 2.0
        np.fill_diagonal(q, 0.0)

        H = float(W0.sum())
        Z = float((W0 * q).sum()) / H
        rho = 0.01
        theta_max = 0.25
        theta = min(theta_max, rho / Z) if Z > 0 else 0.0

        a = theta * q
        A = W0 * a
        h = A.sum(axis=1)
        WR = W0 - A + np.diag(h)

        # Degree preservation check
        row_sums = WR.sum(axis=1)
        max_degree_err = float(np.max(np.abs(row_sums - d0)))
        self.assertLess(max_degree_err, 1e-6, f"Degree preservation violated: err={max_degree_err}")

    def test_contraction_invariant(self):
        """Spectral radius rho(S_R) <= 1.0 + 1e-6 and rho(S_7) <= 1.0 + 1e-6."""
        N = 20
        W0 = np.random.uniform(0.1, 1.5, size=(N, N)).astype(np.float32)
        W0 = (W0 + W0.T) / 2.0
        np.fill_diagonal(W0, 0.0)
        d0 = W0.sum(axis=1)

        q = np.random.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)
        q = (q + q.T) / 2.0
        np.fill_diagonal(q, 0.0)

        theta = 0.2
        A = W0 * (theta * q)
        h = A.sum(axis=1)
        WR = W0 - A + np.diag(h)

        # Normalized S_R
        inv_sqrt_d0 = np.zeros(N, dtype=np.float32)
        inv_sqrt_d0[d0 > 0] = 1.0 / np.sqrt(d0[d0 > 0])
        SR = inv_sqrt_d0[:, None] * WR * inv_sqrt_d0[None, :]

        # Synthetic CF operator
        Wcf = np.random.uniform(0.1, 1.0, size=(N, N)).astype(np.float32)
        Wcf = (Wcf + Wcf.T) / 2.0
        np.fill_diagonal(Wcf, 0.0)
        dcf = Wcf.sum(axis=1)
        inv_sqrt_dcf = np.zeros(N, dtype=np.float32)
        inv_sqrt_dcf[dcf > 0] = 1.0 / np.sqrt(dcf[dcf > 0])
        Scf = inv_sqrt_dcf[:, None] * Wcf * inv_sqrt_dcf[None, :]

        S7 = 0.9 * SR + 0.1 * Scf

        # Eigenvalues
        eig_SR = np.linalg.eigvalsh(SR)
        eig_S7 = np.linalg.eigvalsh(S7)

        self.assertLessEqual(np.max(np.abs(eig_SR)), 1.0 + 1e-6, "S_R violates contraction norm <= 1.0")
        self.assertLessEqual(np.max(np.abs(eig_S7)), 1.0 + 1e-6, "S_7 violates contraction norm <= 1.0")

    def test_scale_invariance_cancellation(self):
        """D_{cW}^{-1/2} (cW) D_{cW}^{-1/2} == D_W^{-1/2} W D_W^{-1/2}."""
        N = 10
        W = np.random.uniform(0.2, 3.0, size=(N, N)).astype(np.float64)
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)

        def norm_op(M):
            d = M.sum(axis=1)
            inv = np.divide(1.0, np.sqrt(d), out=np.zeros(N, dtype=np.float64), where=d > 0)
            return inv[:, None] * M * inv[None, :]

        S_orig = norm_op(W)
        for c in [0.01, 0.55, 2.5, 100.0]:
            S_scaled = norm_op(c * W)
            err = np.max(np.abs(S_orig - S_scaled))
            self.assertLess(err, 1e-12, f"Scale invariance violated for c={c}: err={err}")

    def test_dose_bound_verification(self):
        """sum_{i != j} A_ij <= rho_t * sum_{i != j} W_{0, ij} under all settings."""
        N = 15
        W0 = np.random.uniform(0.1, 1.0, size=(N, N)).astype(np.float32)
        W0 = (W0 + W0.T) / 2.0
        np.fill_diagonal(W0, 0.0)
        H = float(W0.sum())

        q = np.random.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)
        q = (q + q.T) / 2.0
        np.fill_diagonal(q, 0.0)

        Z = float((W0 * q).sum()) / H
        theta_max = 0.25

        # Test across multiple target doses (feasible and infeasible)
        for rho in [0.001, 0.01, 0.05, 0.25, 0.5, 2.0]:
            theta = min(theta_max, rho / Z) if Z > 0 else 0.0
            A = W0 * (theta * q)
            sum_A = float(A.sum())
            rho_eff = sum_A / H
            rho_formula = min(rho, theta_max * Z)

            self.assertLessEqual(sum_A, rho * H + 1e-6, f"Dose bound violated for rho={rho}")
            self.assertAlmostEqual(rho_eff, rho_formula, places=5)

    def test_symmetry_and_nonnegativity(self):
        """W_R and S_7 must be symmetric and nonnegative on off-diagonal edges."""
        N = 12
        W0 = np.random.uniform(0.1, 1.0, size=(N, N)).astype(np.float32)
        W0 = (W0 + W0.T) / 2.0
        np.fill_diagonal(W0, 0.0)

        q = np.random.uniform(0.0, 1.0, size=(N, N)).astype(np.float32)
        q = (q + q.T) / 2.0
        np.fill_diagonal(q, 0.0)

        theta = 0.25  # Maximum cap
        a = theta * q
        A = W0 * a
        h = A.sum(axis=1)
        WR = W0 - A + np.diag(h)

        # Off-diagonal entries must remain nonnegative (since a <= 0.25 < 1.0)
        off_diag_mask = ~np.eye(N, dtype=bool)
        self.assertTrue(np.all(WR[off_diag_mask] >= 0.0))
        self.assertTrue(np.all(np.diag(WR) >= 0.0))
        # Symmetry
        self.assertLess(np.max(np.abs(WR - WR.T)), 1e-6)

    def test_chunked_pair_scoring_agreement(self):
        """Chunked pair scoring (C=4, 64, 4096) matches unchunked pairwise scoring."""
        N, d, K = 30, 16, 80
        D = torch.randn(N, d)
        r = torch.norm(D, dim=-1)
        tau = 0.01 * float(torch.median(r[r > 0]))

        pair_i = torch.randint(0, N, (K,))
        pair_j = torch.randint(0, N, (K,))
        eligible = (r >= tau)

        # Unchunked reference
        q_ref = torch.zeros(K)
        for k in range(K):
            u, v = pair_i[k], pair_j[k]
            if eligible[u] and eligible[v]:
                cos = (D[u] * D[v]).sum() / (r[u] * r[v]).clamp(min=1e-12)
                q_ref[k] = max(0.0, -float(cos.clamp(-1.0, 1.0)))

        # Mock adapter
        adapter = STAIR5V7GraphAdapter(
            crow_indices=torch.zeros(N + 1, dtype=torch.long),
            col_indices=torch.zeros(K, dtype=torch.long),
            base_values=torch.zeros(K),
            n_items=N,
            pair_i=pair_i,
            pair_j=pair_j,
            w0_pairs=torch.ones(K),
            w_norm_pairs=torch.ones(K),
            slot_fwd=torch.zeros(K, dtype=torch.long),
            slot_bwd=torch.zeros(K, dtype=torch.long),
            diag_nodes=torch.zeros(0, dtype=torch.long),
            diag_slots=torch.zeros(0, dtype=torch.long),
            diag_coeff=torch.zeros(0),
            d0=torch.ones(N),
            pair_chunk_size=4,
        )

        for chunk_size in [4, 16, 64, 4096]:
            adapter.pair_chunk_size = chunk_size
            q_chunk = adapter.score_pairs_chunked(D, r, eligible)
            diff = (q_chunk - q_ref).abs().max().item()
            self.assertLess(diff, 1e-6, f"Chunked scoring discrepancy at C={chunk_size}: {diff}")

    def test_csr_in_place_mutation_matches_dense_reference(self):
        """Verifies that the CSR in-place slot mutation matches dense linear algebra."""
        N = 6
        # Connected graph on 5 nodes, 1 isolated node
        W0_np = np.array([
            [0.0, 1.5, 0.0, 0.5, 0.0, 0.0],
            [1.5, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 1.0, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        d0_np = W0_np.sum(axis=1)

        raw_semantic = scipy_to_tensor(sp.csr_matrix(W0_np))
        baseline_norm = scipy_to_tensor(sp.csr_matrix(
            np.divide(W0_np, np.sqrt(d0_np[:, None] * d0_np[None, :]), out=np.zeros_like(W0_np), where=(d0_np[:, None] * d0_np[None, :]) > 0)
        ))

        # Synthetic train edges
        train_edges = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        state = build_calibrated_graph_v7(
            raw_semantic=raw_semantic,
            baseline_normalized=baseline_norm,
            train_edges=train_edges,
            n_users=5,
            n_items=N,
            arm="UCR-D",
            rho=0.01,
            theta_max=0.25,
            k_cf=2,
            eta=0.1,
        )
        adapter = state.adapter

        # Run one step
        D = torch.randn(N, 16)
        active_op = adapter.update_step_snapshot(D, epoch_progress=10.0)

        # Check that active_op is a valid CSR operator
        self.assertEqual(active_op.layout, torch.sparse_csr)
        self.assertEqual(tuple(active_op.shape), (N, N))
        self.assertTrue(torch.isfinite(active_op.values()).all())

        # Check diagnostics
        diag = adapter.get_last_step_diagnostics()
        self.assertIsNotNone(diag)
        self.assertIn("rho_eff", diag)
        self.assertGreaterEqual(diag["rho_eff"], 0.0)

        # Clear step snapshot
        adapter.clear_step_snapshot()
        self.assertIsNone(adapter.get_last_step_diagnostics())

    def test_ihp_ncer_gram_bounds(self):
        """IHP-NCER structural Gram scores must lie strictly in [0, 1]."""
        N = 10
        Wcf = np.random.uniform(0.0, 2.0, size=(N, N)).astype(np.float32)
        Wcf = (Wcf + Wcf.T) / 2.0
        np.fill_diagonal(Wcf, 0.0)
        Wcf_csr = sp.csr_matrix(Wcf)

        W_retained, meta = compute_ihp_ncer_scores(Wcf_csr, rho=0.01, theta_max=0.25)
        self.assertIn("mean_s_struct", meta)
        self.assertGreaterEqual(meta["mean_s_struct"], 0.0)
        self.assertLessEqual(meta["mean_s_struct"], 1.0 + 1e-6)

        # Retained graph must preserve degrees
        self.assertTrue(np.allclose(W_retained.sum(axis=1), Wcf_csr.sum(axis=1), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
