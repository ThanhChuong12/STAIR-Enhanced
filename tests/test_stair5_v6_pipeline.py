"""tests/test_stair5_v6_pipeline.py — Integration & End-to-End Tests for STAIR5-v6 (BCSR).
========================================================================================
Verification requirements:
1. Optimizer Groups:
   - Every trainable parameter belongs to exactly one group.
   - Item embedding has STAIR5V6Smoother.
   - User embedding has smoother=None.
2. Bitwise / Exact Reference Recovery:
   - When theta=0 or arm='C-V4', operator, loss, gradients and AdamWSEvo updates
     match STAIR5-v4 bitwise or within strict float epsilon.
3. Checkpoint Roundtrip:
   - Save and resume checkpoint; verify graph fingerprints, parameter states and metadata match.
4. Factorial Arms Construction:
   - P-BCSR, C-V4, C-Uniform, C-NoShrink, C-Shuffled, C-NoRetention, C-Renorm, C-CF-Placebo.
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
from models.stair5_v6 import ARMS_V6, STAIR5_v6_Model
from models.stair5_v6_utils import save_training_checkpoint, load_training_checkpoint, atomic_torch_save
from optimizers.AdamW import AdamWSEvo
from optimizers.stair5_v6_smoother import STAIR5V6Smoother


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


class TestSTAIR5V6Pipeline(unittest.TestCase):

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
            v6_arm="P-BCSR",
            v6_theta=0.25,
            v6_t_rel=5.0,
            v6_candidate_chunk_size=1024,
            lambda_nlgcl=0.01,
            nlgcl_tau=0.2,
            nlgcl_G=1,
            nlgcl_alpha=0.5,
            eta=0.1,
            k_cf=3,
            c_min=1,
            t_shrinkage=5.0,
            cl_chunk_size=1024,
            knn_device="cpu",
            knn_chunk_size=1024,
            cf_block_size=16,
            cf_memory_budget_mib=32.0,
            graph_cache_dir=None,
        )

    def test_optimizer_parameter_groups_and_smoother(self):
        """Verify parameter membership and item-only smoother binding."""
        model = STAIR5_v6_Model(self.dataset, self.cfg)
        groups = model.marked_params()

        self.assertEqual(len(groups), 2)
        # Group 0: Users (no smoother)
        self.assertIsNone(groups[0]["smoother"])
        # Group 1: Items (STAIR5V6Smoother)
        self.assertIsInstance(groups[1]["smoother"], STAIR5V6Smoother)
        self.assertEqual(groups[1]["smoother"].L, 2)

        # Unique & complete parameter membership
        all_params = [p for g in groups for p in g["params"]]
        self.assertEqual(len(all_params), len(set(id(p) for p in all_params)))
        model_trainables = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(all_params), len(model_trainables))

    def test_bitwise_reference_recovery_vs_v4(self):
        """When v6_theta=0.0 or v6_arm='C-V4', graph and loss/grad match v4 bitwise or within strict eps."""
        cfg_v4 = copy.copy(self.cfg)
        cfg_v4.v4_arm = "N-CSE"

        cfg_v6 = copy.copy(self.cfg)
        cfg_v6.v6_arm = "C-V4"

        # Fix RNG seed before creating models
        torch.manual_seed(999)
        model_v4 = STAIR5_v4_Model(self.dataset, cfg_v4)

        torch.manual_seed(999)
        model_v6 = STAIR5_v6_Model(self.dataset, cfg_v6)

        # Verify graph operators match within float32 precision
        mAdj_v4 = model_v4.mAdj.to_dense()
        mAdj_v6 = model_v6.mAdj.to_dense()
        self.assertTrue(torch.allclose(mAdj_v4, mAdj_v6, atol=1e-6))

        # Forward pass on identical batch
        batch = {
            model_v4.User: torch.tensor([0, 1, 2, 3]),
            model_v4.Item: torch.tensor([4, 5, 6, 7]),
            model_v4.INeg: torch.tensor([8, 9, 10, 11]),
        }
        loss_v4, bpr_v4, cl_v4 = model_v4.training_objective(batch)
        loss_v6, bpr_v6, cl_v6 = model_v6.training_objective(batch)

        self.assertAlmostEqual(loss_v4.item(), loss_v6.item(), places=5)
        self.assertAlmostEqual(bpr_v4.item(), bpr_v6.item(), places=5)
        self.assertAlmostEqual(cl_v4.item(), cl_v6.item(), places=5)

        # Backward step
        loss_v4.backward()
        loss_v6.backward()

        for (n4, p4), (n6, p6) in zip(model_v4.named_parameters(), model_v6.named_parameters()):
            self.assertEqual(n4, n6)
            self.assertTrue(torch.allclose(p4.grad, p6.grad, atol=1e-5))

        # Multi-step update parity
        opt4 = AdamWSEvo(model_v4.marked_params(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        opt6 = AdamWSEvo(model_v6.marked_params(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        for _ in range(3):
            for m, opt in ((model_v4, opt4), (model_v6, opt6)):
                opt.zero_grad(set_to_none=True)
                m.fit(batch).backward()
                opt.step()
            for p4, p6 in zip(model_v4.parameters(), model_v6.parameters()):
                self.assertTrue(torch.equal(p4, p6))
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(opt4.state[p4][key], opt6.state[p6][key]))

    def test_checkpoint_roundtrip_restoration(self):
        """Save and resume checkpoint; verify graph fingerprints and parameters match."""
        model = STAIR5_v6_Model(self.dataset, self.cfg)
        optimizer = AdamWSEvo(model.marked_params(), lr=1e-3)

        batch = {
            model.User: torch.tensor([0, 1]),
            model.Item: torch.tensor([2, 3]),
            model.INeg: torch.tensor([4, 5]),
        }
        loss, _, _ = model.training_objective(batch)
        loss.backward()
        optimizer.step()

        checkpoint_path = Path(self.temp_dir.name) / "ckpt.pt"
        extra = {"test_metric": 0.0542}
        save_training_checkpoint(checkpoint_path, model, optimizer, epoch=10, extra=extra)

        model_restored = STAIR5_v6_Model(self.dataset, self.cfg)
        opt_restored = AdamWSEvo(model_restored.marked_params(), lr=1e-3)

        payload = load_training_checkpoint(checkpoint_path, model_restored, opt_restored)
        self.assertEqual(payload["epoch"], 10)
        self.assertEqual(payload["extra"]["test_metric"], 0.0542)

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_restored.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))

        self.assertEqual(model.graph_fingerprint, model_restored.graph_fingerprint)

    def test_factorial_arms_construction(self):
        """Verify each of the registered v6 arms instantiates and compiles correctly."""
        for arm in ARMS_V6:
            cfg_arm = copy.copy(self.cfg)
            cfg_arm.v6_arm = arm
            model = STAIR5_v6_Model(self.dataset, cfg_arm)
            self.assertEqual(model.v6_arm, arm)
            self.assertIsNotNone(model.mAdj)
            self.assertEqual(model.mAdj.shape, (self.dataset.n_items, self.dataset.n_items))


if __name__ == "__main__":
    unittest.main()
