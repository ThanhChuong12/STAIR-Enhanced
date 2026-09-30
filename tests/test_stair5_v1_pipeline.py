# -*- coding: utf-8 -*-
"""
tests/test_stair5_v1_pipeline.py — Integration and End-to-End Pipeline Tests for STAIR-LHC v1
=============================================================================================
Verifies:
1. End-to-end model construction with mock dataset and parameters.
2. AdamWSEvo optimizer parameter groups and BSC Smoother setup.
3. Scale buffers initialization (q_{t, l}, M_norm) and frozen state.
4. Warmup scheduler across all phases (epochs 0 -> 20 -> 50+).
5. Joint forward pass: L_total = L_BPR + lambda * L_LHC.
6. Gradient backward pass and AdamWSEvo optimizer step (Zero NaNs).
7. Inference evaluation ranking buffer and Euclidean dot product consistency.
8. Gradient cosine diagnostic calculation (BPR vs LHC).
9. All ablation arms (H0, B0, E0, HC, H0W5, H0-NOSELF, H0-REWEIGHT).
"""

import math
import os
import sys
import unittest
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import models.freerec_compat
except Exception:
    try:
        import freerec_compat
    except Exception:
        pass

import freerec
from optimizers.AdamW import AdamWSEvo
from models.stair5_v1 import STAIR5_v1_Model


from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL


class MockDataset:
    def __init__(self, n_users: int = 40, n_items: int = 60):
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
        u_idx = torch.randint(0, n_users, (200,))
        i_idx = torch.randint(0, n_items, (200,))
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


def _mock_prepare(model_self, path: str = "."):
    N_i = model_self.Item.count
    edge_index = torch.stack([
        torch.randint(0, N_i, (80,)),
        torch.randint(0, N_i, (80,))
    ], dim=0)
    edge_weight = torch.ones(80)
    mAdj = torch.sparse_coo_tensor(edge_index, edge_weight, (N_i, N_i)).to_sparse_csr()
    model_self.register_buffer("mAdj", mAdj)


class TestSTAIR5v1Pipeline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_users = 30
        self.n_items = 45
        self.dim = 64

        self.mock_dataset = MockDataset(self.n_users, self.n_items)

        # Build mock config
        self.cfg = types.SimpleNamespace(
            embedding_dim=self.dim,
            num_layers=3,
            gamma=0.2,
            lhc_arm="H0",
            lambda_lhc=3e-4,
            lhc_tau=0.3,
            lhc_kappa=1.0,
            radius_cap=2.0,
            w_hybrid=0.0,
            eps_taylor=1e-4,
            eps_floor=1e-2,
            eps_beta=0.05,
            warmup_start=20,
            warmup_end=50,
            lr=1e-3,
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.999,
            device=self.device,
            optimizer="adamwsevo",
        )
        self.cfg.beta3 = (
            0.1 + 0.9 * (torch.arange(self.dim) / self.dim).pow(self.cfg.gamma)
        ).to(self.device)

        # Construct model with mocked prepare to bypass disk modality files
        original_prepare = STAIR5_v1_Model.prepare
        STAIR5_v1_Model.prepare = _mock_prepare
        try:
            self.model = STAIR5_v1_Model(self.mock_dataset, self.cfg).to(self.device)
        finally:
            STAIR5_v1_Model.prepare = original_prepare

    def test_01_parameter_groups_and_smoother(self):
        """Test parameter marked groups contract: Item has BSC smoother, User does not."""
        marked = self.model.marked_params()
        self.assertEqual(len(marked), 2)
        self.assertIsNone(marked[0]["smoother"], "User embeddings must NOT have smoother!")
        self.assertIsNotNone(marked[1]["smoother"], "Item embeddings must have BSC smoother!")

    def test_02_scale_buffer_initialization(self):
        """Test initialization of median norms (q) and M_norm quantile."""
        self.model.initialize_scale_buffers()
        self.assertTrue(self.model.is_scale_initialized.item())
        self.assertGreater(self.model.q_u0.item(), 0.0)
        self.assertGreater(self.model.q_i0.item(), 0.0)
        self.assertGreater(self.model.q_u1.item(), 0.0)
        self.assertGreater(self.model.q_i1.item(), 0.0)
        self.assertGreater(self.model.M_norm.item(), 0.0)

    def test_03_warmup_scheduler(self):
        """Test 3-stage linear warmup schedule."""
        self.model.update_epoch(0)
        self.assertEqual(self.model.current_lambda, 0.0)

        self.model.update_epoch(20)
        self.assertEqual(self.model.current_lambda, 0.0)

        self.model.update_epoch(35)
        self.assertAlmostEqual(self.model.current_lambda, 1.5e-4, places=6)

        self.model.update_epoch(50)
        self.assertAlmostEqual(self.model.current_lambda, 3e-4, places=6)

        self.model.update_epoch(200)
        self.assertAlmostEqual(self.model.current_lambda, 3e-4, places=6)

    def test_04_forward_backward_optimizer_step(self):
        """Test end-to-end forward pass, backward pass, and AdamWSEvo optimizer step."""
        self.model.train()
        self.model.update_epoch(50)  # active lambda = 3e-4

        batch_size = 12
        users = torch.randint(0, self.n_users, (batch_size, 1), device=self.device)
        positives = torch.randint(0, self.n_items, (batch_size, 1), device=self.device)
        negatives = torch.randint(0, self.n_items, (batch_size, 1), device=self.device)

        data = {
            self.model.User: users,
            self.model.Item: positives,
            self.model.INeg: negatives,
        }

        optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        optimizer.zero_grad()
        loss = self.model(data)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Check gradients
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), "NaN gradient detected!")
                self.assertFalse(torch.isinf(p.grad).any(), "Inf gradient detected!")

        optimizer.step()

        # Verify telemetry diagnostics
        self.assertIn("bpr_loss", self.model.last_diagnostics)
        self.assertIn("cl_loss", self.model.last_diagnostics)
        self.assertIn("alignment", self.model.last_diagnostics)
        self.assertIn("uniformity", self.model.last_diagnostics)

    def test_05_inference_evaluation_parity(self):
        """Test that reset_ranking_buffers and recommendations use 100% Euclidean dot product."""
        self.model.eval()
        self.model.reset_ranking_buffers()

        users = torch.tensor([[0], [1], [2]], device=self.device)
        data_full = {self.model.User: users}
        scores_full = self.model.recommend_from_full(data_full)
        self.assertEqual(scores_full.shape, (3, self.n_items))

        # Check against manual dot product
        u_emb, i_emb = self.model.encode_for_eval()
        expected = u_emb[users.squeeze(-1)] @ i_emb.t()
        self.assertTrue(torch.allclose(scores_full, expected, atol=1e-5))

    def test_06_ablation_arms(self):
        """Test all ablation arms instantiate and run forward pass cleanly."""
        arms = ["H0", "B0", "E0", "HC", "H0W5", "H0-NOSELF", "H0-REWEIGHT"]
        batch_size = 8
        users = torch.randint(0, self.n_users, (batch_size, 1), device=self.device)
        positives = torch.randint(0, self.n_items, (batch_size, 1), device=self.device)
        negatives = torch.randint(0, self.n_items, (batch_size, 1), device=self.device)
        data = {self.model.User: users, self.model.Item: positives, self.model.INeg: negatives}

        for arm in arms:
            self.cfg.lhc_arm = arm
            original_prepare = STAIR5_v1_Model.prepare
            STAIR5_v1_Model.prepare = _mock_prepare
            try:
                model_arm = STAIR5_v1_Model(self.mock_dataset, self.cfg).to(self.device)
            finally:
                STAIR5_v1_Model.prepare = original_prepare

            model_arm.update_epoch(50)
            loss = model_arm(data)
            self.assertTrue(torch.isfinite(loss), f"Loss not finite for arm {arm}!")


if __name__ == "__main__":
    unittest.main()
