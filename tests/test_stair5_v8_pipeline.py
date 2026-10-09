"""tests/test_stair5_v8_pipeline.py — Integration and Pipeline Tests for STAIR5-v8.
==================================================================================
Tests:
1. Stage 1 SAP Procrustes recovery on synthetic rotation and guard fallbacks.
2. End-to-end model construction with mock dataset across arms (WMSG-core, V4-control).
3. Parameter invariant: user smoother is None, item smoother is STAIR5V8Smoother.
4. Joint forward pass: total loss = BPR + lambda_nlgcl * NLGCL.
5. Backward pass and optimizer step without NaNs.
6. Checkpoint contract, extra state verification, and restoration roundtrip.
"""
import copy
import math
import os
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import models.freerec_compat
except Exception:
    pass

import freerec
from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v4 import STAIR5_v4_Model
from models.stair5_v8_alignment import (
    align_modality_features,
    compute_weighted_procrustes,
)
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from optimizers.AdamW import AdamWSEvo
from optimizers.stair5_v8_smoother import STAIR5V8Smoother


class MockDataset:
    def __init__(self, n_users: int = 20, n_items: int = 35):
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


class TestSTAIR5V8Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset = MockDataset(20, 35)
        self.dataset.path = self.temp_dir.name

        # Create dummy feature files
        gen = torch.Generator().manual_seed(42)
        for name in ["text.pkl", "image.pkl"]:
            feat = torch.randn(35, 16, generator=gen)
            with open(Path(self.temp_dir.name) / name, "wb") as h:
                pickle.dump(feat, h)

        self.cfg = types.SimpleNamespace(
            embedding_dim=8,
            num_layers=2,
            mfiles=["text.pkl", "image.pkl"],
            num_neighbors=[5, 1],
            gamma=0.2,
            beta1=0.9,
            beta2=0.999,
            eta=0.1,
            k_cf=3,
            c_min=2,
            t_shrinkage=5.0,
            lambda_nlgcl=0.01,
            nlgcl_tau=0.2,
            nlgcl_G=1,
            nlgcl_alpha=0.5,
            cl_chunk_size=1024,
            knn_chunk_size=1024,
            cf_block_size=16,
            cf_memory_budget_mib=128.0,
            knn_device="cpu",
            graph_cache_dir=str(self.temp_dir.name),
            artifact_dir=str(self.temp_dir.name),
            device="cpu",
            root=str(self.temp_dir.name),
            dataset="mock_dataset",
            lr=0.001,
            weight_decay=0.1,
            # V8 specific parameters
            v8_arm="WMSG-core",
            semantic_mode="weighted",
            edge_power=2.0,
            edge_floor=0.05,
            semantic_mix=1.0,
            alignment="off",
            anchor_seed=2718,
            anchor_per_stratum=1024,
            relation_gate="off",
            relation_strength=0.0,
            relation_threshold=0.7,
            relation_temperature=0.1,
            bsc_residual=0.0,
            lambda_dirichlet=0.0,
        )
        self.cfg.beta3 = (
            0.1 + 0.9 * (torch.arange(self.cfg.embedding_dim) / self.cfg.embedding_dim).pow(self.cfg.gamma)
        )

    def test_sap_procrustes_algebra_and_guards(self):
        """Tests Stage 1 SAP weighted Procrustes recovery and fallback guards."""
        d = 6
        n = 30
        rng = np.random.default_rng(42)

        # 1. Exact orthogonal transform recovery
        Z_v = rng.standard_normal((n, d))
        Q_true, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Z_t = Z_v @ Q_true
        weights = rng.uniform(0.1, 1.0, size=n)
        weights /= weights.sum()

        Q_est, Sigma, cond = compute_weighted_procrustes(Z_v, Z_t, weights)
        self.assertTrue(np.allclose(Z_v @ Q_est, Z_t, atol=1e-6))
        self.assertGreater(cond, 1e-4)

        # 2. Test align_modality_features with alignment="off"
        cfg_off = types.SimpleNamespace(alignment="off")
        Z_t_t = torch.randn(30, d)
        Z_v_t = torch.randn(30, d)
        degrees = torch.randint(1, 10, (30,))
        fused_off, Q_off, audit_off = align_modality_features(Z_t_t, Z_v_t, degrees, cfg_off)
        expected_off = (5.0 * Z_t_t + Z_v_t) / 6.0
        self.assertTrue(torch.allclose(fused_off, expected_off))
        self.assertEqual(audit_off["alignment_mode"], "off")

        # 3. Test fallback behavior when audit fails or data insufficient
        cfg_sap = types.SimpleNamespace(
            alignment="stratified_procrustes",
            anchor_seed=2718,
            anchor_per_stratum=1024,
            embedding_dim=d,
        )
        # Fewer than ten items cannot give each stratum fit and audit rows.
        fused_fb, Q_fb, audit_fb = align_modality_features(Z_t_t[:9], Z_v_t[:9], degrees[:9], cfg_sap)
        self.assertEqual(audit_fb["status"], "FALLBACK_INSUFFICIENT_ITEMS")
        self.assertTrue(torch.allclose(Q_fb, torch.eye(d, dtype=torch.float64)))

        source = torch.from_numpy(rng.standard_normal((100, d)).astype(np.float32))
        target = source @ torch.from_numpy(Q_true.astype(np.float32))
        fused, transform, audit = align_modality_features(target, source, torch.arange(1, 101), cfg_sap)
        self.assertTrue(audit["passed"])
        self.assertTrue(torch.allclose(transform.float(), torch.from_numpy(Q_true).float(), atol=1e-5))
        self.assertTrue(torch.allclose(fused, target, atol=1e-5))
        self.assertEqual(len(audit["selected_anchor_ids"]), 100)
        self.assertEqual(len(audit["transform_values"]), d)

    def test_model_construction_and_parameter_groups(self):
        """Verifies STAIR5_v8_Model initialization and exact optimizer groups."""
        model = STAIR5_v8_Model(self.dataset, self.cfg)

        # Marked params check
        groups = model.marked_params()
        self.assertEqual(len(groups), 2)

        # User group
        self.assertEqual(groups[0]["params"], list(model.User.parameters()))
        self.assertIsNone(groups[0]["smoother"])

        # Item group
        self.assertEqual(groups[1]["params"], list(model.Item.parameters()))
        self.assertIsInstance(groups[1]["smoother"], STAIR5V8Smoother)

        # Operator buffer
        self.assertTrue(hasattr(model, "mAdj"))
        self.assertTrue(model.mAdj.is_sparse_csr)

    def test_sap_degrees_count_distinct_users(self):
        captured = {}
        def capture_alignment(Z_t, Z_v, train_degrees, cfg):
            captured["degrees"] = train_degrees.clone()
            return align_modality_features(Z_t, Z_v, train_degrees, cfg)
        with patch("models.stair5_v8.align_modality_features", side_effect=capture_alignment):
            STAIR5_v8_Model(self.dataset, self.cfg)
        unique_keys = torch.unique(self.dataset.edge_index[0] * self.dataset.n_items + self.dataset.edge_index[1])
        expected = torch.bincount(unique_keys % self.dataset.n_items, minlength=self.dataset.n_items)
        self.assertTrue(torch.equal(captured["degrees"], expected))

    def test_forward_backward_step_pipeline(self):
        """Verifies forward objective, backward pass, and AdamWSEvo optimizer step."""
        model = STAIR5_v8_Model(self.dataset, self.cfg)
        optimizer = AdamWSEvo(
            model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        batch_size = 16
        data = {
            model.User: torch.randint(0, model.User.count, (batch_size,)),
            model.Item: torch.randint(0, model.Item.count, (batch_size,)),
            model.INeg: torch.randint(0, model.Item.count, (batch_size,)),
        }

        # Forward pass
        optimizer.zero_grad()
        loss, bpr, cl = model.training_objective(data)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(bpr))
        self.assertTrue(torch.isfinite(cl))
        self.assertAlmostEqual(loss.item(), (bpr + model.lambda_nlgcl * cl).item(), places=5)

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

        # Optimizer step
        optimizer.step()

        # Check embeddings updated and finite
        self.assertTrue(torch.isfinite(model.User.embeddings.weight).all())
        self.assertTrue(torch.isfinite(model.Item.embeddings.weight).all())

    def test_v4_control_parity(self):
        """Verifies that v8_arm='V4-control' executes exact V4 bypass."""
        cfg_control = copy.copy(self.cfg)
        cfg_control.v8_arm = "V4-control"

        model_control = STAIR5_v8_Model(self.dataset, cfg_control)
        self.assertEqual(model_control.v8_arm, "V4-control")
        self.assertTrue(
            model_control.graph_metadata.get("v4_bypass", False)
            or model_control.graph_metadata.get("note") == "Exact STAIR5-v4 control reference bypass."
        )
        cfg_v4 = copy.copy(self.cfg)
        cfg_v4.v4_arm = "N-CSE"
        reference = STAIR5_v4_Model(copy.deepcopy(self.dataset), cfg_v4)
        self.assertTrue(torch.equal(model_control.mAdj.to_dense(), reference.mAdj.to_dense()))
        self.assertTrue(torch.equal(model_control.Item.embeddings.weight, reference.Item.embeddings.weight))
        users, positives, negatives = torch.arange(8), torch.arange(8), torch.arange(8) + 10
        data_control = {model_control.User: users, model_control.Item: positives, model_control.INeg: negatives}
        data_reference = {reference.User: users, reference.Item: positives, reference.INeg: negatives}
        loss_control = model_control.training_objective(data_control)[0]
        loss_reference = reference.training_objective(data_reference)[0]
        self.assertTrue(torch.allclose(loss_control, loss_reference, atol=1e-6, rtol=1e-6))
        loss_control.backward()
        loss_reference.backward()
        self.assertTrue(torch.allclose(model_control.Item.embeddings.weight.grad, reference.Item.embeddings.weight.grad, atol=1e-6, rtol=1e-6))
        direction = torch.randn_like(model_control.Item.embeddings.weight)
        self.assertTrue(torch.allclose(model_control.marked_params()[1]["smoother"](direction), reference.marked_params()[1]["smoother"](direction), atol=1e-6))

    def test_checkpoint_saving_and_restoration(self):
        """Verifies checkpoint roundtrip and immutable provenance contract."""
        model = STAIR5_v8_Model(self.dataset, self.cfg)
        optimizer = AdamWSEvo(model.marked_params(), lr=0.001)
        data = {model.User: torch.arange(8), model.Item: torch.arange(8), model.INeg: torch.arange(8) + 10}
        model.training_objective(data)[0].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint_path = Path(self.temp_dir.name) / "test_checkpoint.pt"
        extra = {"epoch": 1, "score": 0.05}

        save_training_checkpoint(checkpoint_path, model, optimizer, epoch=1, extra=extra)
        self.assertTrue(checkpoint_path.is_file())

        # Load checkpoint into fresh model
        model2 = STAIR5_v8_Model(self.dataset, self.cfg)
        optimizer2 = AdamWSEvo(model2.marked_params(), lr=0.001)

        payload = load_training_checkpoint(checkpoint_path, model2, optimizer2)
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(payload["extra"]["score"], 0.05)
        self.assertTrue(
            torch.allclose(
                model.Item.embeddings.weight,
                model2.Item.embeddings.weight,
            )
        )
        self.assertTrue(torch.equal(model.User.embeddings.weight, model2.User.embeddings.weight))
        for group1, group2 in zip(optimizer.param_groups, optimizer2.param_groups):
            for p1, p2 in zip(group1["params"], group2["params"]):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(optimizer.state[p1][key], optimizer2.state[p2][key]))
        data2 = {model2.User: data[model.User], model2.Item: data[model.Item], model2.INeg: data[model.INeg]}
        for current, current_optimizer, batch in ((model, optimizer, data), (model2, optimizer2, data2)):
            current.training_objective(batch)[0].backward()
            current_optimizer.step()
        self.assertTrue(torch.equal(model.Item.embeddings.weight, model2.Item.embeddings.weight))


if __name__ == "__main__":
    unittest.main()
