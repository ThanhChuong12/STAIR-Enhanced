"""Independent graph algebra, evidence and actual optimizer integration tests."""
import tempfile
import unittest

import torch

from models.stair5_v2_graph import build_calibrated_graph, candidate_statistics, normalize_graph
from optimizers.stair5_v2_smoother import STAIR5V2Smoother
from optimizers.utils import Smoother
from optimizers.AdamW import AdamWSEvo


def toy_graph():
    # Duplicate raw edges must merge; item 3 remains isolated.
    indices = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 2, 1]])
    return torch.sparse_coo_tensor(indices, torch.tensor([1., 1., 2., 1., 1.]), (4, 4)).coalesce()


class TestCandidateGraph(unittest.TestCase):
    def setUp(self):
        self.raw = toy_graph()
        self.baseline = normalize_graph(self.raw)
        # Item-user sets: 0={0,1}, 1={0,1,2}, 2={2}; duplicate (0,0).
        self.edges = torch.tensor([[0, 1, 0, 1, 2, 2, 0], [0, 0, 1, 1, 1, 2, 0]])

    def build(self, **kwargs):
        return build_calibrated_graph(self.raw, self.baseline, self.edges, 4, **kwargs)

    def test_candidate_statistics_deduplicate_binary_interactions(self):
        scores, degrees, counts = candidate_statistics(self.raw, self.edges, 4, 5.)
        torch.testing.assert_close(degrees.float(), torch.tensor([2., 3., 1., 0.]))
        torch.testing.assert_close(counts.float(), torch.tensor([2., 2., 1., 1.]))
        expected = torch.tensor([2 / 7 * 2 / (6 ** .5)] * 2 + [1 / 6 / (3 ** .5)] * 2)
        torch.testing.assert_close(scores.float(), expected)

    def test_symmetry_support_isolated_node_and_spectral_bound(self):
        result = self.build(strength=.5, mix=.5)
        dense = result.operator.to_dense()
        torch.testing.assert_close(dense, dense.T)
        self.assertTrue(torch.equal(dense != 0, self.raw.to_dense() != 0))
        self.assertEqual(dense[3].abs().sum().item(), 0)
        self.assertLessEqual(torch.linalg.eigvalsh(dense).abs().max().item(), 1 + 1e-6)
        self.assertFalse(torch.allclose(dense, self.baseline.to_dense()))
        self.assertIsNone(result.operator.grad_fn)

    def test_zero_intervention_returns_exact_original_operator(self):
        for kwargs in ({'strength': 0.}, {'mix': 0.}):
            result = self.build(**kwargs)
            self.assertIs(result.operator, self.baseline)
            self.assertTrue(torch.equal(result.operator.to_dense(), self.baseline.to_dense()))

    def test_constant_boost_cancels_under_degree_normalization(self):
        boosted = torch.sparse_coo_tensor(self.raw.indices(), self.raw.values() * 1.25, self.raw.shape)
        torch.testing.assert_close(normalize_graph(boosted).to_dense(), self.baseline.to_dense())

    def test_only_supplied_train_edges_affect_evidence(self):
        extra = torch.tensor([[3, 3], [0, 2]])
        original = self.build()
        changed = build_calibrated_graph(self.raw, self.baseline, torch.cat([self.edges, extra], 1), 4)
        self.assertFalse(torch.equal(original.item_degrees, changed.item_degrees))
        # Held-out edges exist outside the builder; rerunning the original inputs remains exact.
        repeated = self.build()
        torch.testing.assert_close(original.support_scores, repeated.support_scores)

    def test_cache_split_and_hyperparameter_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.build(cache_dir=directory)
            repeated = self.build(cache_dir=directory)
            changed = self.build(cache_dir=directory, strength=.5)
            changed_edges = build_calibrated_graph(self.raw, self.baseline, self.edges[:, :-2], 4, cache_dir=directory)
            torch.testing.assert_close(first.operator.to_dense(), repeated.operator.to_dense())
            self.assertFalse(torch.allclose(first.operator.to_dense(), changed.operator.to_dense()))
            self.assertFalse(torch.equal(first.item_degrees, changed_edges.item_degrees))

    def test_placebo_preserves_symmetry_and_evidence_marginal(self):
        regular, placebo = self.build(), self.build(placebo=True, seed=13)
        torch.testing.assert_close(placebo.operator.to_dense(), placebo.operator.to_dense().T)
        # Scores on both directions carry the same unordered-edge distribution.
        torch.testing.assert_close(regular.support_scores.sort().values, placebo.support_scores.sort().values)

    def test_callback_polynomial_matches_dense_and_baseline(self):
        beta = torch.tensor([0., .1, .5, .9])
        features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        smoother = STAIR5V2Smoother(lambda x: self.baseline @ x, beta=beta, L=3)
        expected = torch.zeros_like(features)
        dense = self.baseline.to_dense()
        for power in range(4):
            expected += torch.linalg.matrix_power(dense, power) @ features * beta.pow(power)
        expected *= (1 - beta) / (1 - beta.pow(4))
        torch.testing.assert_close(smoother(features), expected)
        torch.testing.assert_close(smoother(features), Smoother(self.baseline, beta, 3, 'neumann')(features))

    def test_actual_adam_update_and_moments_match_baseline_smoother(self):
        beta = torch.tensor([0., .1, .5, .9])
        parameters = [torch.nn.Parameter(torch.arange(16.).reshape(4, 4) / 10) for _ in range(2)]
        smoothers = [Smoother(self.baseline, beta, 3, 'neumann'),
                     STAIR5V2Smoother(lambda x: self.baseline @ x, beta=beta, L=3)]
        optimizers = [AdamWSEvo([{'params': [p], 'smoother': s}], lr=.01, weight_decay=.1)
                      for p, s in zip(parameters, smoothers)]
        for step in range(3):
            for parameter, optimizer in zip(parameters, optimizers):
                optimizer.zero_grad()
                ((parameter - step / 10).square().mean()).backward()
                optimizer.step()
            torch.testing.assert_close(parameters[0], parameters[1], rtol=0, atol=0)
            for name in ('exp_avg', 'exp_avg_sq', 'step'):
                torch.testing.assert_close(optimizers[0].state[parameters[0]][name],
                                           optimizers[1].state[parameters[1]][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
