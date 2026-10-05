"""tests/test_stair5_v52_objectives.py — Unit Tests for STAIR5-v5.2 NLGCL Loss Module.
======================================================================================
Tests:
1. Faithful in-batch InfoNCE loss computation & finite scalar check.
2. Exact numerical equivalence between unchunked and chunked InfoNCE.
3. Gradient propagation to both query and key embeddings.
4. Input validation (positive entity counts, finite tau, alpha in [0, 1]).
"""
import unittest
import torch
import torch.nn.functional as F

from models.stair5_v4_objectives import NLGCL_Module


class TestSTAIR5V52Objectives(unittest.TestCase):

    def test_nlgcl_module_forward_and_backward(self):
        n_users = 25
        n_items = 20
        dim = 16
        B = 8

        # Create dummy layer embeds: [H0, H1]
        H0 = torch.randn(n_users + n_items, dim, requires_grad=True)
        H1 = torch.randn(n_users + n_items, dim, requires_grad=True)

        users = torch.randint(0, n_users, (B,))
        pos_items = torch.randint(0, n_items, (B,))

        mod = NLGCL_Module(n_users=n_users, n_items=n_items, G=1, tau=0.2, alpha=0.5)
        loss = mod([H0, H1], users, pos_items)

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        # Test backward pass
        loss.backward()
        self.assertIsNotNone(H0.grad)
        self.assertIsNotNone(H1.grad)
        self.assertTrue(torch.isfinite(H0.grad).all())
        self.assertTrue(torch.isfinite(H1.grad).all())

    def test_nlgcl_chunked_vs_unchunked_equivalence(self):
        n_users = 30
        n_items = 25
        dim = 32
        B = 16

        torch.manual_seed(42)
        H0 = torch.randn(n_users + n_items, dim)
        H1 = torch.randn(n_users + n_items, dim)

        users = torch.randint(0, n_users, (B,))
        pos_items = torch.randint(0, n_items, (B,))

        # Unchunked
        mod_unchunked = NLGCL_Module(n_users=n_users, n_items=n_items, G=1, tau=0.2, alpha=0.5, chunk_size=None)
        loss_unchunked = mod_unchunked([H0, H1], users, pos_items)

        # Chunked with chunk_size = 4
        mod_chunked = NLGCL_Module(n_users=n_users, n_items=n_items, G=1, tau=0.2, alpha=0.5, chunk_size=4)
        loss_chunked = mod_chunked([H0, H1], users, pos_items)

        self.assertTrue(torch.allclose(loss_unchunked, loss_chunked, atol=1e-5))

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            NLGCL_Module(n_users=0, n_items=10)
        with self.assertRaises(ValueError):
            NLGCL_Module(n_users=10, n_items=10, tau=-0.1)
        with self.assertRaises(ValueError):
            NLGCL_Module(n_users=10, n_items=10, alpha=1.5)
