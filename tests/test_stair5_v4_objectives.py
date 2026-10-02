"""tests/test_stair5_v4_objectives.py — Unit Tests for STAIR5-v4 NLGCL Loss Modules.
====================================================================================
Tests:
1. NLGCL_Module InfoNCE loss computation & finite scalar check.
2. Exact numerical equivalence between unchunked and chunked InfoNCE.
3. Gradient propagation to both query and key embeddings.
4. Positive-Aware NLGCL Module execution and gradient check.
"""
import unittest
from pathlib import Path
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

    def test_checkpointed_loss_matches_reference_gradients_with_repeats(self):
        import ast
        from typing import List
        import torch.nn as nn
        import torch.nn.functional as F
        source = Path(__file__).resolve().parents[1] / "main_stair_nlgcl_v4.py"
        node = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "NLGCL_Module")
        namespace = dict(torch=torch, nn=nn, F=F, List=List)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        generator = torch.Generator().manual_seed(851)
        a = [torch.randn(11, 5, generator=generator, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        b = [h.detach().clone().requires_grad_() for h in a]
        users = torch.tensor([0, 0, 1, 2, 3, 1, 4])
        items = torch.tensor([0, 1, 1, 2, 3, 4, 5])
        reference = namespace["NLGCL_Module"](5, 6, G=2)
        proposed = NLGCL_Module(5, 6, G=2, chunk_size=3)
        old, new = reference(a, users, items), proposed(b, users, items)
        torch.testing.assert_close(old, new, rtol=1e-12, atol=1e-12)
        old.backward(); new.backward()
        for left, right in zip(a, b):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-12)

    def test_chunking_does_not_save_full_logits_for_backward(self):
        from torch.autograd.graph import saved_tensors_hooks
        anchor = torch.randn(129, 8, requires_grad=True)
        keys = torch.randn(129, 8, requires_grad=True)
        saved = []
        with saved_tensors_hooks(lambda x: (saved.append(tuple(x.shape)) or x), lambda x: x):
            loss = NLGCL_Module(1, 1, chunk_size=17).info_nce_in_batch(anchor, keys, keys)
        # Recomputed logit blocks are not retained by the forward graph.
        self.assertNotIn((17, 129), saved)
        self.assertNotIn((129, 129), saved)
        loss.backward()
        self.assertGreater(float(keys.grad.norm()), 0)
        self.assertGreater(float(anchor.grad.norm()), 0)

    def test_positive_aware_deduplication_and_no_fake_positives(self):
        keys = torch.tensor([0, 1, 3, 4])  # users 0,1 each know items 0,1
        module = NLGCL_PositiveAware_Module(2, 3, keys.flip(0), chunk_size=1)
        layers = [torch.randn(5, 4, requires_grad=True) for _ in range(2)]
        users, items = torch.tensor([0, 1]), torch.tensor([0, 1])
        sparse_batch = module(layers, users, items)
        duplicate_batch = module(layers, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]))
        torch.testing.assert_close(sparse_batch, duplicate_batch, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "train split"):
            module(layers, torch.tensor([0]), torch.tensor([2]))
        with self.assertRaises(ValueError):
            module.positive_aware_info_nce(torch.randn(2, 4), torch.randn(3, 4), torch.zeros(2, 3, dtype=torch.bool))
        sparse_batch.backward()
        self.assertTrue(all(torch.isfinite(h.grad).all() for h in layers))

    def test_invalid_loss_contracts(self):
        for params in ({"tau": 0}, {"tau": float("nan")}, {"alpha": 2}, {"G": 0}, {"chunk_size": -1}):
            with self.assertRaises(ValueError):
                NLGCL_Module(2, 3, **params)


if __name__ == "__main__":
    unittest.main()
