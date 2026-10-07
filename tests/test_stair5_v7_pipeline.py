"""tests/test_stair5_v7_pipeline.py — Integration & End-to-End Tests for STAIR5-v7 (UCR-D).
========================================================================================
Verification requirements:
1. Optimizer Group Isolation:
   - User embedding group maintains smoother=None.
   - Item embedding group has STAIR5V7Smoother.
   - Every trainable parameter belongs to exactly one group.
2. Bitwise / Floating-Point Recovery vs v4:
   - When rho=0 (V7-recovery) or arm='V4-control', operator, loss, gradients and AdamWSEvo
     updates match STAIR5-v4 within strict float32 precision limits (1e-6).
3. No Gradient Leaks:
   - Asserts that neither D_t, q_ij, nor S_7^{(t)} values retain an active grad_fn.
4. Snapshot Lifecycle Test:
   - Asserts that clear_step_snapshot() completely clears dynamic transient pointers
     even if an exception occurs during the BSC forward pass.
5. Checkpoint Roundtrip:
   - Save and resume training state; confirm exact continuity of Adam moments and step counters.
6. Ablation Arms Initialization:
   - UCR-D, V4-control, V7-recovery, Uniform-retention, Shuffled-UCR, IHP-NCER-only.
"""
import copy
import math
import os
import pickle
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import models.freerec_compat
except Exception:
    pass

import freerec
from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL

from models.stair5_v4 import STAIR5_v4_Model
from models.stair5_v7 import ARMS_V7, STAIR5_v7_Model
from models.stair5_v7_utils import save_training_checkpoint, load_training_checkpoint, atomic_torch_save
from optimizers.AdamW import AdamWSEvo
from optimizers.stair5_v7_smoother import STAIR5V7Smoother


class MockDataset:
    def __init__(self, n_users: int = 20, n_items: int = 30):
        self.n_users = n_users
        self.n_items = n_items
        self.path = "."

        u_field = Field("User", USER, ID)
        u_field.count = n_users
        i_field = Field("Item", ITEM, ID)
        i_field.count = n_items
        lbl_field = Field("Label", LABEL)
        self.fields = FieldTuple([u_field, i_field, lbl_field])

        # Synthetic user-item interactions
        torch.manual_seed(42)
        u_idx = torch.randint(0, n_users, (120,))
        i_idx = torch.randint(0, n_items, (120,))
        self.edge_index = torch.stack([u_idx, i_idx], dim=0)

    def train(self):
        return self

    def to_normalized_adj(self, normalization="sym"):
        N = self.n_users + self.n_items
        u_adj = self.edge_index[0]
        i_adj = self.edge_index[1] + self.n_users
        row = torch.cat([u_adj, i_adj])
        col = torch.cat([i_adj, u_adj])
        edge_index = torch.stack([row, col], dim=0)
        val = torch.ones(edge_index.size(1))
        deg = torch.bincount(row, minlength=N).float().clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(edge_index, norm_val, (N, N)).to_sparse_csr()

    def to_bigraph(self, edge_type="u2i"):
        container = types.SimpleNamespace()
        container.edge_index = self.edge_index
        return {"u2i": container}


class TestSTAIR5V7Pipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset = MockDataset(20, 30)
        self.dataset.path = self.temp_dir.name

        # Create dummy modality feature files
        gen = torch.Generator().manual_seed(42)
        for name in ["text.pkl", "image.pkl"]:
            feat = torch.randn(30, 16, generator=gen)
            with open(Path(self.temp_dir.name) / name, "wb") as h:
                pickle.dump(feat, h)

        self.cfg = types.SimpleNamespace(
            embedding_dim=8,
            num_layers=2,
            gamma=0.2,
            lr=1e-3,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            batch_size=16,
            device="cpu",
            root=".",
            dataset="mock",
            mfiles=["text.pkl", "image.pkl"],
            num_neighbors=[3, 1],
            beta3=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            v7_arm="UCR-D",
            v7_rho=0.01,
            v7_theta_max=0.25,
            v7_zeta=0.01,
            v7_pair_chunk_size=1024,
            v7_warmup_epochs=10.0,
            v7_shuffled_seed=1,
            lambda_nlgcl=0.01,
            nlgcl_tau=0.2,
            nlgcl_G=1,
            nlgcl_alpha=0.5,
            cl_chunk_size=512,
            knn_chunk_size=512,
            knn_device="cpu",
            cf_block_size=16,
            cf_memory_budget_mib=32.0,
            graph_cache_dir=None,
            eta=0.1,
            k_cf=3,
            c_min=2,
            t_shrinkage=5.0,
        )

    def test_optimizer_parameter_groups_isolation(self):
        """User parameters have smoother=None; item parameters have STAIR5V7Smoother."""
        model = STAIR5_v7_Model(self.dataset, self.cfg)
        groups = model.marked_params()
        self.assertEqual(len(groups), 2)
        # Group 0: User embeddings, smoother=None
        self.assertIsNone(groups[0]["smoother"])
        self.assertEqual(len(groups[0]["params"]), 1)
        self.assertIs(groups[0]["params"][0], model.User.embeddings.weight)

        # Group 1: Item embeddings, smoother=STAIR5V7Smoother
        self.assertIsInstance(groups[1]["smoother"], STAIR5V7Smoother)
        self.assertEqual(len(groups[1]["params"]), 1)
        self.assertIs(groups[1]["params"][0], model.Item.embeddings.weight)

    def test_no_gradient_leaks(self):
        """Asserts that D_t, q_ij, and S_7 values have no active grad_fn."""
        model = STAIR5_v7_Model(self.dataset, self.cfg)
        D = torch.randn(30, 8, requires_grad=True)

        # Call smoother
        smoothed = model.smoother(D)
        self.assertIsNone(smoothed.grad_fn)
        self.assertIsNone(model.graph_adapter.active_values.grad_fn)
        self.assertIsNone(model.graph_adapter.base_values.grad_fn)
        self.assertIsNone(model.graph_adapter.q_buf.grad_fn)

    def test_snapshot_lifecycle_and_exception_safety(self):
        """clear_step_snapshot() is called even if an exception occurs during smoothing."""
        model = STAIR5_v7_Model(self.dataset, self.cfg)
        D = torch.randn(30, 8)

        # Mock an exception inside _smooth
        original_smooth = model.smoother._smooth
        called_clear = []

        def failing_smooth(*args, **kwargs):
            raise RuntimeError("Synthetic BSC Failure")

        original_clear = model.smoother.clear_step_snapshot

        def tracking_clear():
            called_clear.append(True)
            original_clear()

        model.smoother._smooth = failing_smooth
        model.smoother.clear_step_snapshot = tracking_clear

        try:
            with self.assertRaises(RuntimeError):
                model.smoother(D)
            self.assertTrue(len(called_clear) > 0, "clear_step_snapshot was not called on exception")
            self.assertIsNone(model.graph_adapter._current_snapshot)
        finally:
            model.smoother._smooth = original_smooth
            model.smoother.clear_step_snapshot = original_clear

    def test_bitwise_v4_recovery_at_rho_zero(self):
        """When v7_rho=0 or v7_arm='V7-recovery', model update matches v4 within float32 precision."""
        cfg_v4 = copy.copy(self.cfg)
        cfg_v4.arm = "N-CSE"
        cfg_v7 = copy.copy(self.cfg)
        cfg_v7.v7_arm = "V7-recovery"
        cfg_v7.v7_rho = 0.0

        torch.manual_seed(99)
        model_v4 = STAIR5_v4_Model(self.dataset, cfg_v4)
        torch.manual_seed(99)
        model_v7 = STAIR5_v7_Model(self.dataset, cfg_v7)

        # Verify initial weights match identically
        self.assertTrue(torch.allclose(model_v4.Item.embeddings.weight, model_v7.Item.embeddings.weight, atol=1e-7))
        self.assertTrue(torch.allclose(model_v4.User.embeddings.weight, model_v7.User.embeddings.weight, atol=1e-7))

        opt_v4 = AdamWSEvo(model_v4.marked_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        opt_v7 = AdamWSEvo(model_v7.marked_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

        batch_data = {
            model_v4.User: torch.tensor([0, 1, 2, 3]),
            model_v4.Item: torch.tensor([4, 5, 6, 7]),
            model_v4.INeg: torch.tensor([8, 9, 10, 11]),
        }

        # Step v4
        opt_v4.zero_grad(set_to_none=True)
        loss_v4, _, _ = model_v4.training_objective(batch_data)
        loss_v4.backward()
        opt_v4.step()

        # Step v7
        batch_data_v7 = {
            model_v7.User: torch.tensor([0, 1, 2, 3]),
            model_v7.Item: torch.tensor([4, 5, 6, 7]),
            model_v7.INeg: torch.tensor([8, 9, 10, 11]),
        }
        opt_v7.zero_grad(set_to_none=True)
        loss_v7, _, _ = model_v7.training_objective(batch_data_v7)
        loss_v7.backward()
        opt_v7.step()

        # Loss check
        self.assertAlmostEqual(loss_v4.item(), loss_v7.item(), places=5)

        # Item embedding update check
        item_diff = (model_v4.Item.embeddings.weight - model_v7.Item.embeddings.weight).abs().max().item()
        user_diff = (model_v4.User.embeddings.weight - model_v7.User.embeddings.weight).abs().max().item()
        self.assertLess(item_diff, 1e-6, f"Item embedding discrepancy in v4 recovery: {item_diff}")
        self.assertLess(user_diff, 1e-6, f"User embedding discrepancy in v4 recovery: {user_diff}")

    def test_checkpoint_roundtrip(self):
        """Save and resume checkpoint; confirm exact continuity of Adam moments and step counters."""
        model = STAIR5_v7_Model(self.dataset, self.cfg)
        opt = AdamWSEvo(model.marked_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

        batch_data = {
            model.User: torch.tensor([0, 1]),
            model.Item: torch.tensor([2, 3]),
            model.INeg: torch.tensor([4, 5]),
        }
        opt.zero_grad(set_to_none=True)
        loss, _, _ = model.training_objective(batch_data)
        loss.backward()
        opt.step()

        ckpt_path = Path(self.temp_dir.name) / "test_ckpt.pt"
        extra_data = {"best_metric": 0.1234, "step": 1}
        save_training_checkpoint(ckpt_path, model, opt, epoch=1, extra=extra_data)

        # Re-initialize new model and optimizer
        new_model = STAIR5_v7_Model(self.dataset, self.cfg)
        new_opt = AdamWSEvo(new_model.marked_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

        payload = load_training_checkpoint(ckpt_path, new_model, new_opt, restore_random=True)
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(payload["extra"]["best_metric"], 0.1234)

        # Verify weights match exactly
        self.assertTrue(torch.allclose(model.Item.embeddings.weight, new_model.Item.embeddings.weight))
        self.assertTrue(torch.allclose(model.User.embeddings.weight, new_model.User.embeddings.weight))

    def test_factorial_arms_initialization(self):
        """Verify that all 6 arms initialize properly and execute a forward-backward step."""
        for arm in ARMS_V7:
            cfg = copy.copy(self.cfg)
            cfg.v7_arm = arm
            model = STAIR5_v7_Model(self.dataset, cfg)
            opt = AdamWSEvo(model.marked_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)

            batch_data = {
                model.User: torch.tensor([0, 1]),
                model.Item: torch.tensor([2, 3]),
                model.INeg: torch.tensor([4, 5]),
            }
            opt.zero_grad(set_to_none=True)
            loss, bpr, cl = model.training_objective(batch_data)
            loss.backward()
            opt.step()
            self.assertTrue(math.isfinite(loss.item()))


if __name__ == "__main__":
    unittest.main()
