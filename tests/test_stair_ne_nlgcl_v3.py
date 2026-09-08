# -*- coding: utf-8 -*-
"""
tests/test_stair_ne_nlgcl_v3.py
================================
Unit test suite for STAIR-NE-NLGCL+ (v3) architecture:
1. Regularized Diagonal Spectral Projector (0-rotation, L2 anchor).
2. True Sign-Preserving Spectral Noise (|noise| quadrant invariance).
3. Dynamic Slicing in Thresholded MFNA (OOM prevention on large catalog).
4. Hybrid Dynamic HANS Scheduler (Cosine Ceiling Cap + Loss-Gated feedback).
5. Contrastive MLP Projection Head & End-to-End Gradient Flow.
"""

import math
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.stair_ne_nlgcl_plus import (
    STAIR_NE_NLGCL_Plus,
    RegularizedDiagonalSpectralProjector,
)


class TestSTAIR_NE_NLGCL_v3(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.n_users = 200
        self.n_items = 500
        self.batch_size = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_01_regularized_diagonal_projector(self):
        """Test Pillar 1: 0-rotation and L2 anchoring loss."""
        projector = RegularizedDiagonalSpectralProjector(dim=self.dim, reg_weight=1e-4).to(self.device)
        
        # 1. Weights initialized to exactly 1.0
        self.assertTrue(torch.allclose(projector.w, torch.ones(self.dim, device=self.device)))
        
        # 2. Initial anchoring loss is 0.0
        initial_loss = projector.get_anchoring_loss()
        self.assertAlmostEqual(initial_loss.item(), 0.0, places=6)

        # 3. Element-wise multiplication without cross-dimensional rotation
        x = torch.randn(self.batch_size, self.dim, device=self.device)
        y = projector(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x))  # Because w=1.0 initially

        # 4. Anchoring loss penalizes deviations from 1.0
        with torch.no_grad():
            projector.w[0] = 2.0  # (2.0 - 1.0)^2 = 1.0
        anchor_loss = projector.get_anchoring_loss()
        self.assertAlmostEqual(anchor_loss.item(), 1e-4 * 1.0, places=6)
        print("  [PASS] Test 1: RegularizedDiagonalSpectralProjector passed.")

    def test_02_quadrant_invariance_sign_preserving_noise(self):
        """Test Pillar 2: Quadrant Invariance Theorem for |noise|."""
        module = STAIR_NE_NLGCL_Plus(
            dim=self.dim,
            eps=0.15,
        ).to(self.device)
        module.train()

        # Create diverse positive and negative latent embeddings
        h = torch.randn(500, self.dim, device=self.device)
        beta = torch.linspace(0.9, 0.1, self.dim, device=self.device)

        # Inject noise multiple times across random draws
        for trial in range(10):
            h_perturbed = module.inject_spectral_noise(h, beta)
            
            # Quadrant Invariance: sign(h_perturbed) must match sign(h) for 100% of non-zero elements
            non_zero = h != 0.0
            sign_match = torch.sign(h_perturbed[non_zero]) == torch.sign(h[non_zero])
            total_elements = non_zero.sum().item()
            matching_elements = sign_match.sum().item()

            self.assertEqual(
                matching_elements, total_elements,
                f"Quadrant drift detected! {total_elements - matching_elements} elements changed sign."
            )
        print("  [PASS] Test 2: True Sign-Preserving Noise (100% Quadrant Invariance) passed.")

    def test_03_dynamic_slicing_memory_safety(self):
        """Test Pillar 4: Dynamic Slicing prevents OOM on large catalog."""
        large_n_items = 20000
        large_n_users = 1000
        module = STAIR_NE_NLGCL_Plus(
            dim=self.dim,
            n_users=large_n_users,
            n_items=large_n_items,
            tau_thresh=0.85,
        ).to(self.device)
        module.train()

        batch_size = 128
        users = torch.randint(0, large_n_users, (batch_size,), device=self.device)
        positives = torch.randint(0, large_n_items, (batch_size,), device=self.device)
        beta = torch.rand(self.dim, device=self.device)

        # Full catalog modality matrix (20,000 items)
        full_catalog_modals = torch.randn(large_n_items, self.dim, device=self.device)

        # Dummy layer representations [H^(0), H^(1)] for full graph
        total_nodes = large_n_users + large_n_items
        h0 = torch.randn(total_nodes, self.dim, device=self.device)
        h1 = torch.randn(total_nodes, self.dim, device=self.device)
        layer_embeds = [h0, h1]

        # Forward pass should dynamically slice without computing (20000 x 20000 = 400M elements = 1.6GB) matrix!
        weighted_loss, raw_loss, lam = module(
            layer_embeds=layer_embeds,
            users=users,
            positives=positives,
            beta=beta,
            item_modals=full_catalog_modals,
        )

        self.assertFalse(torch.isnan(weighted_loss))
        self.assertFalse(torch.isinf(weighted_loss))
        self.assertGreater(raw_loss, 0.0)
        print("  [PASS] Test 3: Dynamic Slicing Memory Safety ([B x B] on 20K catalog) passed.")

    def test_04_hybrid_hans_scheduler_dynamics(self):
        """Test Pillar 5: Hybrid Dynamic HANS Scheduler dynamics."""
        module = STAIR_NE_NLGCL_Plus(
            lambda_max=0.010,
            lambda_min=0.002,
            gamma_max=0.35,
            gamma_min=0.05,
            warmup_epochs=50,
            total_epochs=500,
        )

        # 1. Warmup phase (Epochs 1-50): lambda increases linearly, gamma_h stays at gamma_min
        for ep in range(1, 51):
            module.update_scheduler(current_cl_loss=1.0)
        self.assertAlmostEqual(module.current_lambda, 0.010, places=4)
        self.assertAlmostEqual(module.current_gamma_h, 0.05, places=4)

        # 2. Cooling phase with plateauing loss (should increase gamma_h, but bounded by cosine cap)
        plateau_loss = 0.80
        for ep in range(51, 100):
            module.update_scheduler(current_cl_loss=plateau_loss)

        # gamma_h should have adapted upward due to plateau
        self.assertGreater(module.current_gamma_h, 0.05)
        # lambda should have started decaying according to cosine curve
        self.assertLess(module.current_lambda, 0.010)

        # 3. Final convergence phase (Epoch 500): both lambda and gamma_h must approach their mins
        for ep in range(100, 501):
            module.update_scheduler(current_cl_loss=0.10)  # Decreasing loss
        self.assertAlmostEqual(module.current_lambda, 0.002, places=3)
        self.assertAlmostEqual(module.current_gamma_h, 0.05, places=3)
        print("  [PASS] Test 4: Hybrid HANS Scheduler (Cosine Ceiling + Loss-Gated Feedback) passed.")

    def test_05_projection_head_and_gradient_flow(self):
        """Test Pillar 3 & Backward Pass: Gradients flow cleanly to all components."""
        projector = RegularizedDiagonalSpectralProjector(dim=self.dim).to(self.device)
        module = STAIR_NE_NLGCL_Plus(
            dim=self.dim,
            n_users=self.n_users,
            n_items=self.n_items,
        ).to(self.device)
        module.train()

        total_nodes = self.n_users + self.n_items
        raw_embeds = torch.randn(total_nodes, self.dim, device=self.device, requires_grad=True)
        h0 = raw_embeds
        h1 = raw_embeds * 0.9  # simulated 1-hop convolution

        batch_size = 32
        users = torch.randint(0, self.n_users, (batch_size,), device=self.device)
        positives = torch.randint(0, self.n_items, (batch_size,), device=self.device)
        beta = torch.rand(self.dim, device=self.device)
        item_modals = torch.randn(self.n_items, self.dim, device=self.device)

        # Forward
        weighted_cl_loss, raw_loss, _ = module(
            layer_embeds=[h0, h1],
            users=users,
            positives=positives,
            beta=beta,
            item_modals=item_modals,
        )
        anchor_loss = projector.get_anchoring_loss()
        total_loss = weighted_cl_loss + anchor_loss

        # Backward
        total_loss.backward()

        # Check gradients exist on inputs and module parameters
        self.assertIsNotNone(raw_embeds.grad)
        self.assertFalse(torch.isnan(raw_embeds.grad).any())

        # Check Projection Head receives gradients
        proj_weight_grad = module.proj_head[0].weight.grad
        self.assertIsNotNone(proj_weight_grad)
        self.assertFalse(torch.isnan(proj_weight_grad).any())
        self.assertGreater(proj_weight_grad.abs().sum().item(), 0.0)

        # Check Spectral Projector w receives gradients
        self.assertIsNotNone(projector.w.grad)
        print("  [PASS] Test 5: Contrastive MLP Projection Head & End-to-End Gradient Flow passed.")


if __name__ == '__main__':
    print("=" * 75)
    print("RUNNING COMPREHENSIVE TEST SUITE: STAIR-NE-NLGCL+ (v3)")
    print("=" * 75)
    unittest.main(verbosity=2)
