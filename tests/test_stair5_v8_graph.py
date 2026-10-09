"""tests/test_stair5_v8_graph.py — Comprehensive Unit Tests for STAIR5-v8 Graph & Smoother.
========================================================================================
Tests:
1. Cosine similarity chunking and clamping in [-1, 1].
2. Power scaling f_p(a) = 0.05 + 0.95 * [max(0, a)]^p across p in {1, 1.5, 2, 3}.
3. Multimodal occurrence fusion and max-symmetrization with zero diagonal.
4. Relation-Proxy Gate (RPG): monotonicity, bounds in [1 - kappa, 1], neutral on missing metadata.
5. Symmetric current-degree normalization with isolated-node identity blocks.
6. Theorem 1 Operator Bound: ||S_8||_2 <= 1.0 across semantic_mix in [0, 1].
7. Theorem 2 Dirichlet-Floor BSC Smoother bounds:
   ||D_tilde||_F <= ||D||_F and omega^2 * E_8(D) <= E_8(D_tilde) <= E_8(D).
8. Exact V4-control bypass parity.
"""
import math
import unittest
import types
import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v8_graph import (
    compute_selected_edge_cosines,
    power_scale_similarities,
    compute_relation_proxy_gate,
    normalize_graph_with_isolated_fallback,
    build_wmsg_semantic_graph,
    build_calibrated_graph_v8,
    tensor_to_scipy,
    resolve_v8_config,
)
from optimizers.stair5_v8_smoother import (
    STAIR5V8Smoother,
    compute_dirichlet_energy,
)


class TestSTAIR5V8GraphAndSmoother(unittest.TestCase):

    def test_selected_edge_cosines_and_clamping(self):
        # 4 items with known features
        x = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],   # identical to 0 -> cosine 1.0
            [-1.0, 0.0],  # opposite to 0 -> cosine -1.0
            [0.0, 2.0],   # orthogonal to 0 -> cosine 0.0
        ], dtype=torch.float32)
        # Edges (0->1), (0->2), (0->3)
        edge_index = torch.tensor([
            [0, 0, 0],
            [1, 2, 3],
        ], dtype=torch.long)

        cosines = compute_selected_edge_cosines(x, edge_index, chunk_size=2)
        self.assertEqual(len(cosines), 3)
        self.assertAlmostEqual(float(cosines[0]), 1.0, places=5)
        self.assertAlmostEqual(float(cosines[1]), -1.0, places=5)
        self.assertAlmostEqual(float(cosines[2]), 0.0, places=5)

    def test_power_scaling_formula(self):
        # Tests: f_p(a) = 0.05 + 0.95 * [max(0, a)]^p
        sims = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # For negative and zero: weight must be exactly 0.05
        weights_p2 = power_scale_similarities(sims, p=2.0, floor=0.05)
        self.assertAlmostEqual(weights_p2[0], 0.05, places=6)
        self.assertAlmostEqual(weights_p2[1], 0.05, places=6)

        # For 0.5 with p=2: 0.05 + 0.95 * (0.5)^2 = 0.05 + 0.95 * 0.25 = 0.2875
        self.assertAlmostEqual(weights_p2[2], 0.2875, places=5)

        # For 1.0: 0.05 + 0.95 * 1.0 = 1.0
        self.assertAlmostEqual(weights_p2[3], 1.0, places=6)

        # Check grid p in {1.0, 1.5, 2.0, 3.0}
        for p in (1.0, 1.5, 2.0, 3.0):
            w = power_scale_similarities(sims, p=p, floor=0.05)
            self.assertTrue(np.all(w >= 0.05))
            self.assertTrue(np.all(w <= 1.0))

    def test_relation_proxy_gate_properties(self):
        n_items = 4
        # Metadata: prices and categories
        metadata_dict = {
            "prices": np.array([10.0, 10.0, 100.0, 10.0], dtype=np.float32),
            "categories": [["A"], ["A"], ["B"], ["A"]],
            "price_mask": np.array([True, True, True, True]),
            "category_mask": np.array([True, True, True, True]),
        }
        # Edges: (0, 1) identical price/cat; (0, 2) diff price/cat
        u = np.array([0, 0])
        v = np.array([1, 2])

        gate = compute_relation_proxy_gate(
            u_indices=u,
            v_indices=v,
            n_items=n_items,
            metadata_dict=metadata_dict,
            kappa=0.1,
            tau_s=0.7,
            T_s=0.1,
        )

        # Bounds: in [1 - kappa, 1.0] = [0.9, 1.0]
        self.assertTrue(np.all(gate >= 0.9))
        self.assertTrue(np.all(gate <= 1.0))

        # (0, 1) has h_01 = 1.0 * 1.0 = 1.0 > tau_s -> retention reduced (close to 0.9)
        # (0, 2) has h_02 = 0.1 * 0.0 = 0.0 < tau_s -> retention near 1.0
        self.assertLess(gate[0], gate[1])

        # Neutral on missing metadata
        neutral_gate = compute_relation_proxy_gate(
            u_indices=u,
            v_indices=v,
            n_items=n_items,
            metadata_dict=None,
            kappa=0.1,
        )
        self.assertTrue(np.allclose(neutral_gate, 1.0))

    def test_isolated_node_normalization(self):
        # 3 nodes: 0 and 1 connected, 2 isolated
        W = sp.csr_matrix(
            np.array([
                [0.0, 0.8, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=np.float32)
        )
        S_norm = normalize_graph_with_isolated_fallback(W)
        dense = S_norm.toarray()

        # Connected block
        self.assertAlmostEqual(dense[0, 1], 1.0, places=6)
        self.assertAlmostEqual(dense[1, 0], 1.0, places=6)
        self.assertAlmostEqual(dense[0, 0], 0.0, places=6)
        self.assertAlmostEqual(dense[1, 1], 0.0, places=6)

        # Isolated node 2 gets identity fallback block (1.0 on diagonal)
        self.assertAlmostEqual(dense[2, 2], 1.0, places=6)
        self.assertAlmostEqual(dense[2, 0], 0.0, places=6)
        self.assertAlmostEqual(dense[2, 1], 0.0, places=6)

    def test_theorem_1_operator_norm_bound(self):
        """Theorem 1: ||S_8||_2 <= 1.0 for all epsilon in [0, 1]."""
        n_items = 20
        n_users = 15
        rng = np.random.default_rng(123)

        # Create synthetic modality features
        f_text = torch.from_numpy(rng.standard_normal((n_items, 16)).astype(np.float32))
        f_vis = torch.from_numpy(rng.standard_normal((n_items, 16)).astype(np.float32))

        # Synthetic kNN lists (text: 2 neighbors, vis: 1 neighbor)
        t_neighbors = torch.tensor([[i, (i + 1) % n_items] for i in range(n_items)]).t().contiguous()
        v_neighbors = torch.tensor([[i, (i + 2) % n_items] for i in range(n_items)]).t().contiguous()
        knn_edges = [t_neighbors, v_neighbors]

        # Synthetic train edges
        u_arr = rng.integers(0, n_users, size=50)
        i_arr = rng.integers(0, n_items, size=50)
        train_edges = torch.tensor(np.stack([u_arr, i_arr]), dtype=torch.long)

        # Baseline S0 (symmetrized via max, zero diagonal)
        all_knn = torch.cat(knn_edges, dim=1)
        w0 = torch.ones(all_knn.size(1), dtype=torch.float32)
        raw_coo = torch.sparse_coo_tensor(all_knn, w0, (n_items, n_items)).coalesce()
        raw_csr = sp.csr_matrix((raw_coo.values().numpy(), raw_coo.indices().numpy()), shape=(n_items, n_items))
        raw_sym = raw_csr.maximum(raw_csr.T)
        raw_sym.setdiag(0.0)
        raw_sym.eliminate_zeros()

        raw_sym_coo = raw_sym.tocoo()
        raw_sem = torch.sparse_coo_tensor(
            torch.stack([torch.from_numpy(raw_sym_coo.row).long(), torch.from_numpy(raw_sym_coo.col).long()]),
            torch.from_numpy(raw_sym_coo.data).float(),
            (n_items, n_items),
        ).coalesce()

        S0_scipy = normalize_graph_with_isolated_fallback(raw_sym)
        base_norm = torch.sparse_csr_tensor(
            torch.from_numpy(S0_scipy.indptr).long(),
            torch.from_numpy(S0_scipy.indices).long(),
            torch.from_numpy(S0_scipy.data).float(),
            size=(n_items, n_items),
        )

        for eps in (0.0, 0.5, 1.0):
            graph_state = build_calibrated_graph_v8(
                raw_semantic=raw_sem,
                baseline_normalized=base_norm,
                train_edges=train_edges,
                n_users=n_users,
                n_items=n_items,
                mfeats=[f_text, f_vis],
                knn_edges_list=knn_edges,
                v8_arm="WMSG-core",
                semantic_mix=eps,
                edge_power=2.0,
            )
            # Evaluate 2-norm of S_8
            S8_np = tensor_to_scipy(graph_state.operator).toarray()
            # Symmetric check
            self.assertTrue(np.allclose(S8_np, S8_np.T, atol=1e-5))
            # Spectral norm
            s8_norm = float(np.linalg.norm(S8_np, ord=2))
            self.assertLessEqual(s8_norm, 1.0 + 1e-5)

    def test_theorem_2_dirichlet_smoother_bounds(self):
        """Theorem 2: ||D_tilde||_F <= ||D||_F and omega^2 * E_8(D) <= E_8(D_tilde) <= E_8(D)."""
        n, d = 25, 8
        rng = np.random.default_rng(42)

        # Build valid symmetric normalized S_8 with ||S_8||_2 <= 1
        W = rng.uniform(0.1, 1.0, size=(n, n)).astype(np.float32)
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        S8 = deg_inv_sqrt[:, None] * W * deg_inv_sqrt[None, :]
        S8_csr = sp.csr_matrix(S8)

        # Convert to torch sparse CSR
        S8_tensor = torch.sparse_csr_tensor(
            torch.from_numpy(S8_csr.indptr).long(),
            torch.from_numpy(S8_csr.indices).long(),
            torch.from_numpy(S8_csr.data).float(),
            size=(n, n),
        )

        # Beta vector
        beta = torch.linspace(0.1, 0.95, d, dtype=torch.float32)

        # Test across different residual omega values
        for omega in (0.0, 0.05, 0.1):
            smoother = STAIR5V8Smoother(
                spmm_fn=lambda x: torch.sparse.mm(S8_tensor, x),
                beta=beta,
                L=3,
                omega=omega,
            )

            # Random preconditioned directions D_t
            D_t = torch.from_numpy(rng.standard_normal((n, d)).astype(np.float32))

            D_tilde = smoother(D_t)

            # 1. Frobenius norm contraction: ||D_tilde||_F <= ||D_t||_F
            norm_in = torch.norm(D_t, p="fro").item()
            norm_out = torch.norm(D_tilde, p="fro").item()
            self.assertLessEqual(norm_out, norm_in + 1e-5)

            # 2. Dirichlet energy bounds
            E_in = compute_dirichlet_energy(S8_tensor, D_t)
            E_out = compute_dirichlet_energy(S8_tensor, D_tilde)

            # Upper bound: E_out <= E_in
            self.assertLessEqual(E_out, E_in + 1e-4)

            # Lower bound (Theorem 2): omega^2 * E_in <= E_out
            lower_bound = (omega ** 2) * E_in
            self.assertGreaterEqual(E_out, lower_bound - 1e-4)

    def test_invalid_weights_and_configuration_are_rejected(self):
        for value in (float("nan"), -1.0, float("inf")):
            with self.assertRaises(ValueError):
                normalize_graph_with_isolated_fallback(sp.csr_matrix([[0, value], [value, 0]]))
        with self.assertRaises(ValueError):
            normalize_graph_with_isolated_fallback(sp.csr_matrix([[0, 1], [0, 0]]))
        for power in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                power_scale_similarities(np.array([0.5]), p=power)
        with self.assertRaises(ValueError):
            STAIR5V8Smoother(operator=lambda x: x, beta=torch.tensor([0.5]), omega=0.25)

    def test_named_arms_resolve_components(self):
        for arm, alignment, gate, residual in (
            ("WMSG-core", "off", "off", 0.0),
            ("SAP-only", "stratified_procrustes", "off", 0.0),
            ("WMSG-RPG", "off", "metadata_proxy", 0.0),
            ("WMSG-residual", "off", "off", 0.05),
        ):
            cfg = resolve_v8_config(types.SimpleNamespace(v8_arm=arm))
            self.assertEqual((cfg.alignment, cfg.relation_gate, cfg.bsc_residual), (alignment, gate, residual))
            self.assertEqual(vars(resolve_v8_config(cfg)), vars(cfg))

    def test_rpg_missing_masks_and_generic_roots_are_neutral(self):
        metadata = {"prices": [10, 10], "categories": [["Root"], ["Root"]],
                    "generic_category_roots": ["Root"]}
        kwargs = dict(u_indices=np.array([0]), v_indices=np.array([1]), n_items=2, kappa=0.1)
        self.assertEqual(compute_relation_proxy_gate(metadata_dict=metadata, **kwargs)[0], 1)
        metadata["categories"] = [["A"], ["A"]]
        metadata["price_mask"] = [True, False]
        self.assertEqual(compute_relation_proxy_gate(metadata_dict=metadata, **kwargs)[0], 1)

    def test_modality_sum_precedes_max_symmetrization(self):
        features = [torch.ones(2, 2), torch.ones(2, 2)]
        edges = [torch.tensor([[0], [1]]), torch.tensor([[1], [0]])]
        graph, _ = build_wmsg_semantic_graph(features, edges, 2)
        # max(text_01 + visual_01, text_10 + visual_10) == 1, not 2.
        self.assertEqual(float(graph[0, 1]), 1)

    def test_smoother_does_not_mutate_input_or_retain_autograd(self):
        direction = torch.randn(4, 2, requires_grad=True)
        original = direction.detach().clone()
        smoother = STAIR5V8Smoother(operator=lambda x: x, beta=torch.tensor([0.2, 0.7]), omega=0.05)
        result = smoother(direction)
        self.assertTrue(torch.equal(direction.detach(), original))
        self.assertIsNone(result.grad_fn)
        energy = compute_dirichlet_energy(lambda x: x, direction)
        self.assertIsInstance(energy, torch.Tensor)
        self.assertIsNone(energy.grad_fn)

    def test_empty_cf_and_real_placebo(self):
        n = 30
        generator = torch.Generator().manual_seed(31)
        features = [torch.randn(n, 8, generator=generator), torch.randn(n, 8, generator=generator)]
        edges = [torch.tensor([[i, (i + offset) % n] for i in range(n) for offset in range(1, 5)]).t(),
                 torch.tensor([[i, (i + 7) % n] for i in range(n)]).t()]
        raw, _ = build_wmsg_semantic_graph(features, edges, n, semantic_mode="reference")
        raw_coo = raw.tocoo()
        raw_tensor = torch.sparse_coo_tensor(torch.tensor(np.stack([raw_coo.row, raw_coo.col])),
                                             torch.from_numpy(raw_coo.data), size=(n, n)).coalesce()
        normalized = normalize_graph_with_isolated_fallback(raw)
        baseline = torch.sparse_csr_tensor(torch.from_numpy(normalized.indptr).long(),
                                           torch.from_numpy(normalized.indices).long(),
                                           torch.from_numpy(normalized.data), size=(n, n))
        # Each user sees exactly one item: no collaborative pair can survive c_min.
        train = torch.stack([torch.arange(n), torch.arange(n)])
        kwargs = dict(raw_semantic=raw_tensor, baseline_normalized=baseline, train_edges=train,
                      n_users=n, n_items=n, mfeats=features, knn_edges_list=edges)
        core = build_calibrated_graph_v8(**kwargs)
        self.assertEqual(core.metadata["effective_eta"], 0)
        expected, _ = build_wmsg_semantic_graph(features, edges, n)
        self.assertTrue(np.allclose(tensor_to_scipy(core.operator).toarray(),
                                    normalize_graph_with_isolated_fallback(expected).toarray()))
        placebo = build_calibrated_graph_v8(**kwargs, v8_arm="WMSG-placebo", placebo_seed=11)
        repeated = build_calibrated_graph_v8(**kwargs, v8_arm="WMSG-placebo", placebo_seed=11)
        self.assertGreater(placebo.metadata["placebo_changed_edges"], 0)
        self.assertTrue(torch.equal(placebo.operator.values(), repeated.operator.values()))
        self.assertTrue(torch.equal(placebo.operator.crow_indices(), core.operator.crow_indices()))
        self.assertTrue(torch.equal(placebo.operator.col_indices(), core.operator.col_indices()))


if __name__ == "__main__":
    unittest.main()
