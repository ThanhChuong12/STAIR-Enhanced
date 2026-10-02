"""Audited contrastive geometry and multi-positive objective tests."""
import unittest

import torch
import torch.nn.functional as F

from models.stair5_v2_geometry import AuditedGeometry
from models.stair5_v2_objectives import AuditedContrastiveLoss, remove_self_return


class TestAuditedObjectives(unittest.TestCase):
    def test_checkpointed_blocks_match_full_loss_and_gradients(self):
        for arm in ('H0', 'E0'):
            full = self.make_loss(arm)
            blocked = self.make_loss(arm)
            full.chunk_size, blocked.chunk_size = 100, 1
            first = {k: v.detach().clone().requires_grad_() for k,v in self.inputs.items()}
            second = {k: v.detach().clone().requires_grad_() for k,v in self.inputs.items()}
            a, _ = full(**first, **self.kwargs)
            b, _ = blocked(**second, **self.kwargs)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            a.backward()
            b.backward()
            for key in first:
                torch.testing.assert_close(first[key].grad, second[key].grad, rtol=2e-4, atol=2e-6)
    def setUp(self):
        torch.manual_seed(12)
        self.inputs = {name: torch.randn(count, 4, requires_grad=True)
                       for name, count in [('u_0', 3), ('u_1', 3), ('i_0', 4), ('i_1', 4)]}
        self.positive = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
        self.kwargs = dict(P=self.positive, deg_u=torch.ones(3), deg_i=torch.ones(4),
                           beta=torch.ones(4), q_u0=1., q_i0=1., q_u1=1., q_i1=1., M_norm=100.)

    def make_loss(self, arm='E0'):
        return AuditedContrastiveLoss(AuditedGeometry(arm=arm), radius_cap=2., tau=.3, disable_reweight=True)

    def test_uniform_multi_positive_target_matches_manual_log_softmax(self):
        objective = self.make_loss()
        loss, _ = objective(**self.inputs, **self.kwargs)
        geo = objective.geo
        # Independent reference: positive averaging, rows without positives omitted.
        pairs = [(self.inputs['u_0'], self.inputs['i_1'], self.positive),
                 (self.inputs['i_0'], self.inputs['u_1'], self.positive.T)]
        values = []
        for query, key, positive in pairs:
            query, key = geo.cap_radius(query), geo.cap_radius(key)
            distance = (query[:, None] - key[None, :]).square().sum(-1)
            logits = -distance / (2 * 2. ** 2 * .3)
            valid = positive.sum(1) > 0
            values.append((-(F.log_softmax(logits, 1) * positive).sum(1)[valid]
                           / positive.sum(1)[valid]).mean())
        torch.testing.assert_close(loss, torch.stack(values).mean())

    def test_no_positive_and_empty_context_batches_remain_differentiable(self):
        objective = self.make_loss()
        for shape in [(3, 4), (0, 4), (3, 0)]:
            users, items = shape
            inputs = {name: torch.randn(count, 4, requires_grad=True)
                      for name, count in [('u_0', users), ('u_1', users), ('i_0', items), ('i_1', items)]}
            kwargs = dict(self.kwargs, P=torch.zeros(shape), deg_u=torch.ones(users), deg_i=torch.ones(items))
            loss, _ = objective(**inputs, **kwargs)
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(loss.requires_grad)
            loss.backward()

    def test_gradients_flow_through_both_branches_over_multiple_steps(self):
        for arm in ['H0', 'E0', 'HC']:
            inputs = {name: value.detach().clone().requires_grad_() for name, value in self.inputs.items()}
            optimizer = torch.optim.SGD(list(inputs.values()), lr=.05)
            objective = self.make_loss(arm)
            for _ in range(3):
                optimizer.zero_grad()
                loss, diagnostics = objective(**inputs, **self.kwargs)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for value in inputs.values():
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                    self.assertGreater(value.grad.abs().sum().item(), 0.)
                optimizer.step()
                self.assertIn('alignment', diagnostics)
                self.assertIn('uniformity', diagnostics)

    def test_near_origin_and_cap_have_finite_geodesic_gradients(self):
        geometry = AuditedGeometry()
        for magnitude in [0., 1e-8, 1., 1e4]:
            query = (torch.randn(5, 4) * magnitude).requires_grad_()
            key = torch.randn(3, 4, requires_grad=True)
            q = geometry.lorentz_exp0(geometry.cap_radius(query))
            k = geometry.lorentz_exp0(geometry.cap_radius(key))
            distance = geometry.pairwise_lorentz_distance_squared(q, k)
            self.assertTrue((distance >= 0).all())
            distance.sum().backward()
            self.assertTrue(torch.isfinite(query.grad).all())
            self.assertTrue(torch.isfinite(key.grad).all())

    def test_hyperboloid_and_self_distance(self):
        geometry = AuditedGeometry()
        x = geometry.lorentz_exp0(geometry.cap_radius(torch.randn(8, 4)))
        torch.testing.assert_close(-x[:, 0].square() + x[:, 1:].square().sum(1), -torch.ones(8), atol=1e-4, rtol=1e-4)
        distance = geometry.pairwise_lorentz_distance_squared(x, x)
        torch.testing.assert_close(distance.diag(), torch.zeros(8), atol=2e-5, rtol=0)

    def test_optional_self_return_uses_original_edge_coefficient_and_masks_empty(self):
        source = torch.tensor([[1., 2., 3., 4.], [2., 1., 0., 3.]])
        beta = torch.tensor([.9, .5, .01, 0.])
        # Weighted original normalized UI coefficients, not (degree-1) normalization.
        coefficient = torch.tensor([.25, 1.])
        other_context = torch.tensor([[.1, .2, 0., .3], [0., 0., 0., 0.]])
        h1 = source * coefficient[:, None] * beta + other_context
        corrected, valid = remove_self_return(h1, source, coefficient, beta)
        torch.testing.assert_close(corrected, other_context)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(corrected[1].abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
