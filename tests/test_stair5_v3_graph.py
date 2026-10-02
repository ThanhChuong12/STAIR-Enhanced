"""STAIR5-v3 graph projection, candidate statistics and smoother unit tests."""
import tempfile
import unittest

import numpy as np
import torch

from models.stair5_v3_graph import (
    build_calibrated_graph_v3, candidate_statistics, candidate_statistics_v3,
    project_to_original_degrees, normalize_graph,
)
from optimizers.stair5_v3_smoother import STAIR5V3Smoother
from optimizers.utils import Smoother
from optimizers.AdamW import AdamWSEvo


def toy_graph():
    # Symmetric 4-node graph with duplicate raw edges; node 3 isolated
    indices = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    return torch.sparse_coo_tensor(indices, torch.tensor([1., 1., 2., 1., 1.]), (4, 4)).coalesce()


def four_cycle_graph():
    # 4-cycle: 0-1, 1-2, 2-3, 3-0
    row = [0, 1, 1, 2, 2, 3, 3, 0]
    col = [1, 0, 2, 1, 3, 2, 0, 3]
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()


def tree_graph():
    # Simple star / tree: 0-1, 0-2, 0-3
    row = [0, 1, 0, 2, 0, 3]
    col = [1, 0, 2, 0, 3, 0]
    indices = torch.tensor([row, col], dtype=torch.long)
    values = torch.ones(len(row), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()


class TestSTAIR5V3Graph(unittest.TestCase):
    def setUp(self):
        self.raw = toy_graph()
        self.baseline = normalize_graph(self.raw)
        # Item-user sets: 0={0,1}, 1={0,1,2}, 2={2}; duplicate (0,0)
        self.edges = torch.tensor([[0, 1, 0, 1, 2, 2, 0], [0, 0, 1, 1, 1, 2, 0]])

    def build(self, **kwargs):
        return build_calibrated_graph_v3(self.raw, self.baseline, self.edges, 4, **kwargs)

    def test_candidate_statistics_v3(self):
        scores, degrees, counts = candidate_statistics_v3(self.raw, self.edges, 4, 5.0)
        torch.testing.assert_close(degrees.float(), torch.tensor([2., 3., 1., 0.]))
        torch.testing.assert_close(counts.float(), torch.tensor([2., 2., 1., 1.]))
        expected = torch.tensor([2 / 7 * 2 / (6 ** 0.5)] * 2 + [1 / 6 / (3 ** 0.5)] * 2)
        torch.testing.assert_close(scores.float(), expected)

    def test_zero_intervention_returns_exact_baseline(self):
        for arm in ("B0", "DP-zero"):
            result = self.build(arm=arm)
            self.assertIs(result.operator, self.baseline)
        for kwargs in ({"strength": 0.0}, {"mix": 0.0}):
            result = self.build(arm="DP-ref", **kwargs)
            self.assertIs(result.operator, self.baseline)

    def test_symmetry_support_isolated_node_and_spectral_bound(self):
        result = self.build(arm="DP-ref", strength=0.5, mix=0.5)
        dense = result.operator.to_dense()
        torch.testing.assert_close(dense, dense.T, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(dense != 0, self.raw.to_dense() != 0))
        self.assertEqual(dense[3].abs().sum().item(), 0)
        max_eig = torch.linalg.eigvalsh(dense).abs().max().item()
        self.assertLessEqual(max_eig, 1.0 + 1e-5)
        self.assertIsNone(result.operator.grad_fn)

    def test_four_cycle_kl_projection_redistribution_and_degree_preservation(self):
        cycle = four_cycle_graph()
        row, col = cycle.indices()
        scores = torch.zeros(len(row), dtype=torch.float32)
        # boost edge (0,1) and (1,0) by factor 1.0 (strength=1.0)
        for idx in range(len(row)):
            if (row[idx].item(), col[idx].item()) in ((0, 1), (1, 0)):
                scores[idx] = 1.0

        projected, diag = project_to_original_degrees(cycle, scores, strength=1.0, tol=1e-8)
        self.assertTrue(diag["converged"])
        dense_w = projected.to_dense()
        d0 = cycle.to_dense().sum(dim=1)
        d_star = dense_w.sum(dim=1)
        torch.testing.assert_close(d_star, d0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dense_w, dense_w.T, atol=1e-6, rtol=1e-6)
        self.assertGreater(dense_w[0, 1].item(), 1.0)
        self.assertLess(dense_w[1, 2].item(), 1.0)
        self.assertAlmostEqual((dense_w[0, 1] + dense_w[0, 3]).item(), 2.0, places=5)

    def test_tree_projection_returns_identity_reweighting(self):
        tree = tree_graph()
        row, col = tree.indices()
        scores = torch.zeros(len(row), dtype=torch.float32)
        for idx in range(len(row)):
            if (row[idx].item(), col[idx].item()) in ((0, 1), (1, 0)):
                scores[idx] = 1.0
        projected, diag = project_to_original_degrees(tree, scores, strength=1.0, tol=1e-8)
        torch.testing.assert_close(projected.to_dense(), tree.to_dense(), atol=1e-6, rtol=1e-6)

    def test_constant_evidence_returns_identity(self):
        cycle = four_cycle_graph()
        row, col = cycle.indices()
        scores = torch.ones(len(row), dtype=torch.float32)
        projected, diag = project_to_original_degrees(cycle, scores, strength=0.5, tol=1e-8)
        torch.testing.assert_close(projected.to_dense(), cycle.to_dense(), atol=1e-6, rtol=1e-6)

    def test_placebo_preserves_symmetry_and_score_marginal(self):
        regular = self.build(arm="DP-ref", strength=0.5, mix=0.5)
        placebo = self.build(arm="DP-placebo", strength=0.5, mix=0.5, placebo_seed=13)
        dense_p = placebo.operator.to_dense()
        torch.testing.assert_close(dense_p, dense_p.T, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(regular.support_scores.sort().values, placebo.support_scores.sort().values)

    def test_cache_roundtrip_and_invalidation(self):
        cycle = four_cycle_graph()
        base_norm = normalize_graph(cycle)
        # Asymmetric interaction: only edge (0,1) has common co-occurrence
        edges = torch.tensor([[0, 0], [0, 1]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            first = build_calibrated_graph_v3(cycle, base_norm, edges, 1, cache_dir=directory, arm="DP-ref", strength=0.25)
            repeated = build_calibrated_graph_v3(cycle, base_norm, edges, 1, cache_dir=directory, arm="DP-ref", strength=0.25)
            torch.testing.assert_close(first.operator.to_dense(), repeated.operator.to_dense())
            changed = build_calibrated_graph_v3(cycle, base_norm, edges, 1, cache_dir=directory, arm="DP-ref", strength=2.0)
            self.assertFalse(torch.allclose(first.operator.to_dense(), changed.operator.to_dense()))

    def test_callback_polynomial_matches_dense_and_baseline(self):
        beta = torch.tensor([0.0, 0.1, 0.5, 0.9])
        features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        smoother = STAIR5V3Smoother(lambda x: self.baseline @ x, beta=beta, L=3)
        expected = torch.zeros_like(features)
        dense = self.baseline.to_dense()
        for power in range(4):
            expected += torch.linalg.matrix_power(dense, power) @ features * beta.pow(power)
        expected *= (1 - beta) / (1 - beta.pow(4))
        torch.testing.assert_close(smoother(features), expected)
        torch.testing.assert_close(smoother(features), Smoother(self.baseline, beta, 3, "neumann")(features))

    def test_adam_update_matches_baseline_smoother(self):
        beta = torch.tensor([0.0, 0.1, 0.5, 0.9])
        parameters = [torch.nn.Parameter(torch.arange(16.0).reshape(4, 4) / 10) for _ in range(2)]
        smoothers = [
            Smoother(self.baseline, beta, 3, "neumann"),
            STAIR5V3Smoother(lambda x: self.baseline @ x, beta=beta, L=3)
        ]
        optimizers = [
            AdamWSEvo([{"params": [p], "smoother": s}], lr=0.01, weight_decay=0.1)
            for p, s in zip(parameters, smoothers)
        ]
        for step in range(3):
            for parameter, optimizer in zip(parameters, optimizers):
                optimizer.zero_grad()
                ((parameter - step / 10).square().mean()).backward()
                optimizer.step()
            torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)
            for name in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    optimizers[0].state[parameters[0]][name],
                    optimizers[1].state[parameters[1]][name],
                    rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
