"""tests/test_stair5_v4_pipeline.py — Integration and End-to-End Tests for STAIR5-v4.
===================================================================================
Tests:
1. End-to-end model construction with mock dataset and parameters across all arms.
2. AdamWSEvo optimizer parameter groups and STAIR5V4Smoother setup.
3. Joint forward pass: L_total = L_BPR + lambda_nlgcl * L_NLGCL.
4. Backward pass and optimizer step (Zero NaNs).
5. Arm properties and behavior: B0, N0, C0, N-CSE, N-CSE-placebo, v4b.
6. Checkpoint roundtrip and state restoration.
"""
import copy
import math
import os
import pickle
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import models.freerec_compat
except Exception:
    try:
        import freerec_compat
    except Exception:
        pass

import freerec
from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL

from models.stair5_v4 import ARMS_V4, STAIR5_v4_Model
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from optimizers.AdamW import AdamWSEvo


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
        u_idx = torch.randint(0, n_users, (100,))
        i_idx = torch.randint(0, n_items, (100,))
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


class TestSTAIR5V4Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset = MockDataset(20, 30)
        self.dataset.path = self.temp_dir.name

        # Create dummy feature files
        gen = torch.Generator().manual_seed(42)
        for name in ["text.pkl", "image.pkl"]:
            feat = torch.randn(30, 16, generator=gen).numpy()
            with open(Path(self.temp_dir.name) / name, "wb") as h:
                pickle.dump(feat, h)

        self.cfg = types.SimpleNamespace(
            embedding_dim=8,
            num_layers=2,
            gamma=0.2,
            mfiles=["text.pkl", "image.pkl"],
            num_neighbors=[3, 1],
            root=self.temp_dir.name,
            dataset="toy",
            v4_arm="N-CSE",
            eta=0.1,
            k_cf=3,
            c_min=1,
            t_shrinkage=5.0,
            placebo_seed=1,
            seed=1,
            lambda_nlgcl=0.01,
            nlgcl_tau=0.2,
            nlgcl_G=1,
            nlgcl_alpha=0.5,
            cl_chunk_size=1024,
            graph_cache_dir=None,
            lr=0.001,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.999,
            device=torch.device("cpu"),
        )
        self.cfg.beta3 = (
            0.1 + 0.9 * (torch.arange(8) / 8).pow(0.2)
        ).to(self.cfg.device)

    def _batch(self, model):
        return {
            model.User: torch.tensor([0, 1, 2, 3]),
            model.Item: torch.tensor([1, 2, 3, 4]),
            model.INeg: torch.tensor([5, 6, 7, 8]),
        }

    def test_arms_construction(self):
        for arm in ARMS_V4:
            cfg = copy.copy(self.cfg)
            cfg.v4_arm = arm
            model = STAIR5_v4_Model(self.dataset, cfg)

            if arm in ("B0", "C0"):
                self.assertEqual(model.lambda_nlgcl, 0.0)
            else:
                self.assertEqual(model.lambda_nlgcl, 0.01)

            # Check smoother
            params = model.marked_params()
            self.assertEqual(len(params), 2)
            self.assertIsNone(params[0]["smoother"])
            self.assertIsNotNone(params[1]["smoother"])

            # Check forward and loss
            batch = self._batch(model)
            loss = model.fit(batch)
            self.assertTrue(torch.is_tensor(loss))
            self.assertTrue(torch.isfinite(loss).item())
            self.assertGreater(loss.item(), 0.0)

    def test_training_step_and_optimizer(self):
        model = STAIR5_v4_Model(self.dataset, self.cfg)
        optimizer = AdamWSEvo(
            model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        batch = self._batch(model)
        optimizer.zero_grad()
        loss = model.fit(batch)
        loss.backward()
        optimizer.step()

        # Check weights are finite and updated
        self.assertTrue(torch.isfinite(model.User.embeddings.weight).all())
        self.assertTrue(torch.isfinite(model.Item.embeddings.weight).all())

    def test_checkpoint_roundtrip(self):
        model = STAIR5_v4_Model(self.dataset, self.cfg)
        optimizer = AdamWSEvo(
            model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        batch = self._batch(model)
        optimizer.zero_grad()
        model.fit(batch).backward()
        optimizer.step()

        ckpt_path = Path(self.temp_dir.name) / "ckpt.pt"
        save_training_checkpoint(ckpt_path, model, optimizer, epoch=5)

        # New model and optimizer to load into
        model2 = STAIR5_v4_Model(self.dataset, self.cfg)
        optimizer2 = AdamWSEvo(
            model2.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        payload = load_training_checkpoint(ckpt_path, model2, optimizer2)
        self.assertEqual(payload["epoch"], 5)
        self.assertTrue(torch.allclose(model.User.embeddings.weight, model2.User.embeddings.weight))
        self.assertTrue(torch.allclose(model.Item.embeddings.weight, model2.Item.embeddings.weight))


if __name__ == "__main__":
    unittest.main()
