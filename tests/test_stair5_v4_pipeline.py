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
import ast
from typing import Dict, List, Tuple
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
            feat = torch.randn(30, 16, generator=gen)
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
            model.User: self.dataset.edge_index[0, :4].clone(),
            model.Item: self.dataset.edge_index[1, :4].clone(),
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

    def test_n0_reference_forward_backward_optimizer_and_ranking_parity(self):
        from optimizers.utils import Smoother
        cfg = copy.copy(self.cfg); cfg.v4_arm = "N0"; cfg.cl_chunk_size = 3; cfg.knn_chunk_size = 5; cfg.knn_device = "cpu"
        path = Path(__file__).resolve().parents[1] / "main_stair_nlgcl_v4.py"
        nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.ClassDef) and n.name in ("NLGCL_Module", "STAIR_NLGCL")]
        namespace = dict(cfg=cfg, torch=torch, nn=nn, F=F, math=math, os=os,
                         freerec=freerec, Smoother=Smoother, Dict=Dict, List=List, Tuple=Tuple)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        torch.manual_seed(121); old = namespace["STAIR_NLGCL"](self.dataset)
        torch.manual_seed(121); new = STAIR5_v4_Model(self.dataset, cfg)
        for left, right in zip(old.encode()[:2], new.encode()[:2]):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(old.mAdj.to_dense(), new.mAdj.to_dense(), rtol=0, atol=0)
        opts = [AdamWSEvo(m.marked_params(), lr=.001, weight_decay=.1) for m in (old,new)]
        for _ in range(2):
            losses = []
            for model,opt in zip((old,new), opts):
                data = {field: ids[:,None] for field,ids in self._batch(model).items()}
                opt.zero_grad(set_to_none=True)
                loss = model.fit(data); loss.backward(); losses.append(loss.detach())
            torch.testing.assert_close(*losses, rtol=1e-6, atol=1e-7)
            for left,right in zip(old.parameters(), new.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
            for opt in opts: opt.step()
            for left,right in zip(old.parameters(), new.parameters()):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opts[0].state[left][key], opts[1].state[right][key], rtol=1e-5, atol=1e-8)
        old.eval(); new.eval(); old.reset_ranking_buffers(); new.reset_ranking_buffers()
        data_old, data_new = self._batch(old), self._batch(new)
        data_old[old.User] = data_old[old.User][:,None]; data_new[new.User] = data_new[new.User][:,None]
        pool = torch.tensor([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
        data_old[old.IUnseen] = pool; data_new[new.IUnseen] = pool
        torch.testing.assert_close(old.recommend_from_full(data_old), new.recommend_from_full(data_new), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(old.recommend_from_pool(data_old), new.recommend_from_pool(data_new), rtol=1e-5, atol=1e-7)
        params = [p for g in new.marked_params() for p in g["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual({id(p) for p in params}, {id(p) for p in new.parameters()})

    def test_checkpoint_rejects_mismatch_and_continues_exact_update(self):
        model = STAIR5_v4_Model(self.dataset,self.cfg)
        opt = AdamWSEvo(model.marked_params(), lr=.001, weight_decay=.1)
        model.fit(self._batch(model)).backward(); opt.step()
        path = Path(self.temp_dir.name)/"checked.pt"
        save_training_checkpoint(path,model,opt,1)
        resumed = STAIR5_v4_Model(self.dataset,self.cfg)
        resumed_opt = AdamWSEvo(resumed.marked_params(),lr=.001,weight_decay=.1)
        load_training_checkpoint(path,resumed,resumed_opt)
        for current,optimizer in ((model,opt),(resumed,resumed_opt)):
            optimizer.zero_grad(); current.fit(self._batch(current)).backward(); optimizer.step()
        for left,right in zip(model.parameters(),resumed.parameters()):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        changed = copy.copy(self.cfg); changed.eta=.2
        mismatch = STAIR5_v4_Model(self.dataset,changed)
        before = mismatch.Item.embeddings.weight.detach().clone()
        mismatch_opt = AdamWSEvo(mismatch.marked_params(),lr=.001)
        with self.assertRaisesRegex(ValueError,"provenance"):
            load_training_checkpoint(path,mismatch,mismatch_opt)
        torch.testing.assert_close(before,mismatch.Item.embeddings.weight,rtol=0,atol=0)
        self.assertFalse(mismatch_opt.state)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_checkpointed_loss_and_static_csr_update(self):
        model = STAIR5_v4_Model(self.dataset,self.cfg).to(torch.device("cuda"))
        opt = AdamWSEvo(model.marked_params(),lr=.001)
        batch = {k:v.cuda() for k,v in self._batch(model).items()}
        model.fit(batch).backward(); opt.step()
        self.assertTrue(torch.isfinite(model.Item.embeddings.weight).all())


if __name__ == "__main__":
    unittest.main()
