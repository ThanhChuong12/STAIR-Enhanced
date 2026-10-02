"""tests/test_stair5_v4_objectives.py — Unit Tests for STAIR5-v4 NLGCL Loss Modules.
====================================================================================
Tests:
1. NLGCL_Module InfoNCE loss computation & finite scalar check.
2. Exact numerical equivalence between unchunked and chunked InfoNCE.
3. Gradient propagation to both query and key embeddings.
4. Positive-Aware NLGCL Module execution and gradient check.
"""
import unittest
import torch

from models.stair5_v4_objectives import NLGCL_Module, NLGCL_PositiveAware_Module


class TestSTAIR5V4Objectives(unittest.TestCase):

    def test_nlgcl_module_forward(self):
        n_users = 20
        n_items = 15
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

    def test_nlgcl_chunked_equivalence(self):
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

    def test_nlgcl_positive_aware_module(self):
        n_users = 20
        n_items = 15
        dim = 16
        B = 8

        # Some train interaction keys
        train_pairs = torch.tensor([
            [0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
        ], dtype=torch.long)
        train_keys = torch.unique(train_pairs[:, 0] * n_items + train_pairs[:, 1], sorted=True)

        H0 = torch.randn(n_users + n_items, dim, requires_grad=True)
        H1 = torch.randn(n_users + n_items, dim, requires_grad=True)

        users = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        pos_items = torch.tensor([1, 2, 2, 3, 4, 5, 6, 7], dtype=torch.long)

        mod_pos = NLGCL_PositiveAware_Module(
            n_users=n_users, n_items=n_items, train_pair_keys=train_keys, G=1, tau=0.2, alpha=0.5
        )
        loss = mod_pos([H0, H1], users, pos_items)

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(H0.grad)
        self.assertIsNotNone(H1.grad)
        self.assertTrue(torch.isfinite(H0.grad).all())


if __name__ == "__main__":
    unittest.main()
