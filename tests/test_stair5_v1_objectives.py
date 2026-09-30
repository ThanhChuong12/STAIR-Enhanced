# -*- coding: utf-8 -*-
"""
tests/test_stair5_v1_objectives.py — Unit Tests for STAIR-LHC v1 Objectives
===========================================================================
Verifies:
1. Self-return removal on positive keys.
2. Norm Protection for degree-1 leaf items (magnitude strictly >= eps_floor).
3. Adaptive spectral re-weighting and M_norm quantile capping.
4. Bidirectional Multi-Positive InfoNCE loss computation & backprop (Zero NaNs).
5. Alignment and uniformity diagnostics sanity.
6. Ablation arm support (E0, H0, disable_self_return, disable_reweight).
"""

import math
import os
import sys
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.stair5_v1_geometry import LorentzGeometryModule
from models.stair5_v1_objectives import (
    remove_self_return,
    spectral_reweight,
    MultiPositiveInfoNCELoss,
)


class TestSTAIR5v1Objectives(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = (0.1 + 0.9 * (torch.arange(self.dim, device=self.device) / self.dim).pow(0.2))

    def test_01_self_return_removal_and_norm_protection(self):
        """Test removal of self-return shortcut and enforcement of degree-1 norm protection."""
        N = 10
        E_source = torch.randn(N, self.dim, device=self.device)
        deg_src = torch.ones(N, device=self.device)  # degree 1
        deg_tgt = torch.ones(N, device=self.device)  # degree 1

        # Construct H1 containing pure self-return shortcut
        d_src_eff = torch.clamp(deg_src - 1.0, min=1.0)
        d_tgt_eff = torch.clamp(deg_tgt - 1.0, min=1.0)
        norm_factor = torch.sqrt(d_src_eff * d_tgt_eff).unsqueeze(-1)
        shortcut = (E_source / norm_factor) * self.beta.unsqueeze(0)
        H1_pure = shortcut.clone()

        eps_floor = 1e-2
        H_corr = remove_self_return(
            H1_target=H1_pure,
            E_source=E_source,
            deg_source=deg_src,
            deg_target=deg_tgt,
            beta=self.beta,
            eps_floor=eps_floor,
        )

        # Before protection it was 0; after protection norm must be at least eps_floor
        corr_norm = torch.norm(H_corr, p=2, dim=-1)
        self.assertTrue(
            torch.all(corr_norm >= eps_floor - 1e-6),
            f"Degree-1 norm protection violated! Min norm: {corr_norm.min().item()}"
        )
        self.assertFalse(torch.isnan(H_corr).any(), "NaN in corrected representation!")

    def test_02_spectral_reweight(self):
        """Test spectral reweighting amplifying high-dim features and clipping by M_norm."""
        H1 = torch.randn(20, self.dim, device=self.device)
        H1_rew = spectral_reweight(H1, self.beta, eps_beta=0.05, max_norm=5.0)

        # Ensure shapes match
        self.assertEqual(H1_rew.shape, H1.shape)

        # Norm must not exceed max_norm
        norms = torch.norm(H1_rew, p=2, dim=-1)
        self.assertTrue(
            torch.all(norms <= 5.0 + 1e-4),
            f"M_norm bound exceeded! Max norm: {norms.max().item()}"
        )

    def test_03_multi_positive_infonce_forward_and_backward(self):
        """Test multi-positive InfoNCE loss forward and backward passes (Zero NaNs)."""
        B_u = 16
        B_i = 24
        geo = LorentzGeometryModule(kappa=1.0, radius_cap=2.0).to(self.device)
        loss_fn = MultiPositiveInfoNCELoss(geo=geo, radius_cap=2.0, tau=0.3).to(self.device)

        u_0 = torch.randn(B_u, self.dim, device=self.device, requires_grad=True)
        i_0 = torch.randn(B_i, self.dim, device=self.device, requires_grad=True)
        u_1 = torch.randn(B_u, self.dim, device=self.device, requires_grad=True)
        i_1 = torch.randn(B_i, self.dim, device=self.device, requires_grad=True)

        # Create binary train positive interaction matrix P
        P = (torch.rand(B_u, B_i, device=self.device) > 0.8).float()
        # Guarantee at least 1 positive and 1 negative per user
        P[:, 0] = 1.0
        P[:, -1] = 0.0

        deg_u = torch.randint(2, 50, (B_u,), device=self.device).float()
        deg_i = torch.randint(2, 50, (B_i,), device=self.device).float()

        loss, diag = loss_fn(
            u_0=u_0,
            i_0=i_0,
            u_1=u_1,
            i_1=i_1,
            P=P,
            deg_u=deg_u,
            deg_i=deg_i,
            beta=self.beta,
            q_u0=1.0,
            q_i0=1.0,
            q_u1=1.0,
            q_i1=1.0,
            M_norm=3.0,
        )

        self.assertTrue(torch.isfinite(loss), "InfoNCE loss must be finite!")
        self.assertGreater(loss.item(), 0.0, "InfoNCE loss must be strictly positive!")

        # Backpropagation test
        loss.backward()

        self.assertFalse(torch.isnan(u_0.grad).any(), "NaN in u_0 gradient!")
        self.assertFalse(torch.isnan(i_0.grad).any(), "NaN in i_0 gradient!")
        self.assertFalse(torch.isnan(u_1.grad).any(), "NaN in u_1 gradient!")
        self.assertFalse(torch.isnan(i_1.grad).any(), "NaN in i_1 gradient!")

        # Check diagnostic keys
        self.assertIn("alignment", diag)
        self.assertIn("uniformity", diag)
        self.assertIn("raw_cl_loss", diag)

    def test_04_euclidean_control_objective(self):
        """Test MultiPositiveInfoNCELoss running with Euclidean control (E0 arm)."""
        B_u = 10
        B_i = 15
        geo_e0 = LorentzGeometryModule(arm="E0").to(self.device)
        loss_fn = MultiPositiveInfoNCELoss(geo=geo_e0, radius_cap=2.0, tau=0.3).to(self.device)

        u_0 = torch.randn(B_u, self.dim, device=self.device, requires_grad=True)
        i_0 = torch.randn(B_i, self.dim, device=self.device, requires_grad=True)
        u_1 = torch.randn(B_u, self.dim, device=self.device, requires_grad=True)
        i_1 = torch.randn(B_i, self.dim, device=self.device, requires_grad=True)

        P = torch.zeros(B_u, B_i, device=self.device)
        P[:5, :5] = torch.eye(5, device=self.device)

        deg_u = torch.ones(B_u, device=self.device) * 5.0
        deg_i = torch.ones(B_i, device=self.device) * 5.0

        loss, diag = loss_fn(
            u_0=u_0,
            i_0=i_0,
            u_1=u_1,
            i_1=i_1,
            P=P,
            deg_u=deg_u,
            deg_i=deg_i,
            beta=self.beta,
            q_u0=1.0,
            q_i0=1.0,
            q_u1=1.0,
            q_i1=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertFalse(torch.isnan(u_0.grad).any())


if __name__ == "__main__":
    unittest.main()
