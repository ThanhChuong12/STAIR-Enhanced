# -*- coding: utf-8 -*-
"""
tests/test_stair5_v1_geometry.py — Unit Tests for LorentzGeometryModule
=======================================================================
Verifies:
1. Minkowski hyperboloid constraint: <x, x>_L = -1/kappa.
2. Self-distance equality: D_kappa(x, x) = 0.
3. Gradient finiteness & stability at zero distance (no NaN / inf gradients).
4. Radius capping bounded by R = 2.0.
5. Constant-radius control (HC arm) enforces ||v||_2 = 1.0.
6. Smooth continuity between Taylor expansion and acosh branches at t = eps_taylor.
7. Euclidean control arm (E0) exact squared Euclidean distance.
8. Hybrid kernel interpolation consistency.
"""

import math
import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stair5_v1_geometry import LorentzGeometryModule


class TestLorentzGeometryModule(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.geo = LorentzGeometryModule(kappa=1.0, radius_cap=2.0, w_hybrid=0.0).to(self.device)

    def test_01_radius_capping_bounds(self):
        """Test that cap_radius strictly enforces ||v||_2 < R = 2.0 for extreme vectors."""
        # Test random normal, small, and huge magnitudes
        norms_test = [0.0, 1e-4, 1.0, 2.0, 10.0, 100.0, 1000.0]
        for mag in norms_test:
            a = torch.randn(10, self.dim, device=self.device)
            a = F.normalize(a, p=2, dim=-1) * mag
            v = self.geo.cap_radius(a)
            v_norm = torch.norm(v, p=2, dim=-1)
            self.assertTrue(
                torch.all(v_norm <= 2.0 + 1e-5),
                f"Radius cap violated for input magnitude {mag}: max norm is {v_norm.max().item()}"
            )

    def test_02_constant_radius_hc_arm(self):
        """Test that HC arm enforces unit norm ||v||_2 == 1.0."""
        geo_hc = LorentzGeometryModule(kappa=1.0, arm="HC").to(self.device)
        a = torch.randn(20, self.dim, device=self.device) * 5.0
        v = geo_hc.cap_radius(a)
        v_norm = torch.norm(v, p=2, dim=-1)
        self.assertTrue(
            torch.allclose(v_norm, torch.ones_like(v_norm), atol=1e-5),
            "Constant-radius arm must project vectors to unit sphere!"
        )

    def test_03_minkowski_hyperboloid_constraint(self):
        """Test that lorentz_exp0 produces points satisfying <x, x>_L = -1/kappa."""
        for kappa in [0.5, 1.0, 2.0]:
            geo = LorentzGeometryModule(kappa=kappa, radius_cap=2.0).to(self.device)
            a = torch.randn(30, self.dim, device=self.device)
            v = geo.cap_radius(a)
            x = geo.lorentz_exp0(v)

            # Minkowski inner product: -x0^2 + ||xs||^2
            x0 = x[:, 0:1]
            xs = x[:, 1:]
            minkowski_norm = - (x0 ** 2) + torch.sum(xs ** 2, dim=-1, keepdim=True)
            expected = -1.0 / kappa

            self.assertTrue(
                torch.allclose(minkowski_norm, torch.full_like(minkowski_norm, expected), atol=1e-4),
                f"Minkowski constraint violated for kappa={kappa}!"
            )
            # Forward sheet: x0 > 0
            self.assertTrue(torch.all(x0 > 0), "Lorentz points must lie on forward sheet (x0 > 0)!")

    def test_04_self_distance_zero(self):
        """Test that geodesic distance between identical points is exactly 0."""
        a = torch.randn(15, self.dim, device=self.device)
        v = self.geo.cap_radius(a)
        x = self.geo.lorentz_exp0(v)

        d2 = self.geo.pairwise_lorentz_distance_squared(x, x)
        self.assertEqual(d2.shape, (15, 15))

        diag = torch.diagonal(d2)
        self.assertTrue(
            torch.allclose(diag, torch.zeros_like(diag), atol=1e-5),
            f"Self-distance must be 0, got max diagonal value: {diag.max().item()}"
        )

    def test_05_gradient_finite_at_zero_distance(self):
        """CRITICAL: Test that backprop through pairwise_lorentz_distance_squared at distance 0 yields NO NaNs."""
        v = torch.randn(10, self.dim, device=self.device, requires_grad=True)
        v_capped = self.geo.cap_radius(v)
        x = self.geo.lorentz_exp0(v_capped)

        # Distance to self (distance = 0)
        d2 = self.geo.pairwise_lorentz_distance_squared(x, x)
        loss = d2.sum()
        loss.backward()

        self.assertFalse(torch.isnan(v.grad).any(), "NaN detected in gradient at distance 0!")
        self.assertFalse(torch.isinf(v.grad).any(), "Inf detected in gradient at distance 0!")

    def test_06_continuity_at_taylor_boundary(self):
        """Test smooth continuity between Taylor series and acosh branch at t = eps_taylor."""
        eps = 1e-4
        t_bound = torch.tensor([[eps]], dtype=torch.float32, device=self.device)

        # Taylor value
        val_taylor = (
            2.0 * t_bound
            - (t_bound ** 2) / 3.0
            + 4.0 * (t_bound ** 3) / 45.0
            - (t_bound ** 4) / 35.0
        )
        # Acosh value
        val_acosh = torch.acosh(1.0 + t_bound).pow(2)

        rel_diff = torch.abs(val_taylor - val_acosh) / (val_acosh + 1e-8)
        self.assertLess(
            rel_diff.item(),
            1e-3,
            f"Discontinuity between Taylor and acosh at boundary t={eps}: rel_diff={rel_diff.item()}"
        )

    def test_07_euclidean_control_arm(self):
        """Test that E0 arm calculates exact squared Euclidean distance."""
        geo_e0 = LorentzGeometryModule(arm="E0").to(self.device)
        u = torch.randn(8, self.dim, device=self.device)
        v = torch.randn(12, self.dim, device=self.device)

        d_geo = geo_e0.compute_distance_matrix(u, v)

        # Ground truth Euclidean distance squared
        d_true = torch.cdist(u, v, p=2).pow(2)
        self.assertTrue(
            torch.allclose(d_geo, d_true, atol=1e-5),
            "E0 arm must produce exact squared Euclidean distance!"
        )

    def test_08_hybrid_kernel(self):
        """Test that hybrid kernel with w=0.5 computes (1-w)*d2 + w*log1p(d2)."""
        geo_hybrid = LorentzGeometryModule(kappa=1.0, w_hybrid=0.5).to(self.device)
        geo_base = LorentzGeometryModule(kappa=1.0, w_hybrid=0.0).to(self.device)

        u = torch.randn(5, self.dim, device=self.device)
        v = torch.randn(6, self.dim, device=self.device)

        d_base = geo_base.compute_distance_matrix(u, v)
        d_hybrid = geo_hybrid.compute_distance_matrix(u, v)

        expected = 0.5 * d_base + 0.5 * torch.log1p(d_base)
        self.assertTrue(
            torch.allclose(d_hybrid, expected, atol=1e-5),
            "Hybrid kernel must correctly compute convex combination!"
        )


if __name__ == "__main__":
    unittest.main()
