"""Source-level baseline parity, arm integration and checkpoint roundtrip for STAIR5-v3."""
import ast
import copy
import io
import importlib.util
import math
import os
import pickle
from pathlib import Path
import tempfile
import types
import unittest
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.freerec_compat
import freerec

from models.stair5_v3 import STAIR5_v3_Model
from models.stair5_v3_utils import save_training_checkpoint, load_training_checkpoint
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

_fixture_spec = importlib.util.spec_from_file_location(
    "stair5_v1_fixture", Path(__file__).with_name("test_stair5_v1_pipeline.py")
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
MockDataset = _fixture_module.MockDataset


def baseline_class(cfg):
    """Execute the unmodified baseline class only, excluding parser/training side effects."""
    path = Path(__file__).resolve().parents[1] / "main.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "STAIR")
    namespace = dict(
        torch=torch, nn=nn, F=F, freerec=freerec, os=os, math=math,
        cfg=cfg, Dict=Dict, Tuple=Tuple, Smoother=Smoother
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["STAIR"]


class TestSTAIR5V3Pipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.dataset = MockDataset(4, 6)
        self.dataset.path = self.directory.name

        generator = torch.Generator().manual_seed(17)
        for name in ["text.pkl", "image.pkl"]:
            with open(Path(self.directory.name) / name, "wb") as handle:
                pickle.dump(torch.randn(6, 8, generator=generator).numpy(), handle)

        self.cfg = types.SimpleNamespace(
            embedding_dim=4, num_layers=3, gamma=0.2, mfiles=["text.pkl", "image.pkl"],
            num_neighbors=[2, 1], root=self.directory.name, dataset="toy",
            v3_arm="B0", edge_strength=0.25, edge_mix=0.25, evidence_shrinkage=5.0,
            placebo_seed=1, seed=1, graph_cache_dir=None, knn_chunk_size=1024,
            knn_device="cpu", lambda_lhc=0.0, lhc_tau=0.3, lhc_kappa=1.0,
            radius_cap=2.0, eps_taylor=1e-4, eps_floor=0.01, eps_beta=0.05,
            warmup_start=20, warmup_end=50, grad_diag_every=10, lhc_chunk_size=256,
            lr=0.001, weight_decay=0.1, beta1=0.9, beta2=0.999,
            device=torch.device("cpu")
        )
        self.cfg.beta3 = 0.1 + 0.9 * (torch.arange(4) / 4).pow(0.2)

    def construct(self, arm="B0"):
        cfg = copy.copy(self.cfg)
        cfg.v3_arm = arm
        torch.manual_seed(2)
        return STAIR5_v3_Model(self.dataset, cfg)

    @staticmethod
    def batch(model):
        return {
            model.User: torch.tensor([[0], [0], [1], [2]]),
            model.Item: torch.tensor([[0], [0], [1], [2]]),
            model.INeg: torch.tensor([[3], [4], [4], [5]])
        }

    def test_b0_exact_parity_with_baseline(self):
        torch.manual_seed(2)
        baseline = baseline_class(self.cfg)(self.dataset)
        model = self.construct("B0")

        torch.testing.assert_close(baseline.User.embeddings.weight, model.User.embeddings.weight, rtol=0, atol=0)
        torch.testing.assert_close(baseline.Item.embeddings.weight, model.Item.embeddings.weight, rtol=0, atol=0)

        optimizers = [
            AdamWSEvo(m.marked_params(), lr=0.001, weight_decay=0.1)
            for m in [baseline, model]
        ]
        for _ in range(3):
            torch.testing.assert_close(baseline.encode()[0], model.encode_for_eval()[0], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(baseline.encode()[1], model.encode_for_eval()[1], rtol=1e-6, atol=1e-7)
            losses = [m.fit(self.batch(m)) for m in [baseline, model]]
            torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-7)
            for loss, optimizer in zip(losses, optimizers):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.testing.assert_close(baseline.Item.embeddings.weight, model.Item.embeddings.weight, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(baseline.User.embeddings.weight, model.User.embeddings.weight, rtol=1e-6, atol=1e-7)

        for m in [baseline, model]:
            m.reset_ranking_buffers()
        torch.testing.assert_close(
            baseline.recommend_from_full(self.batch(baseline)),
            model.recommend_from_full(self.batch(model)),
            rtol=1e-6, atol=1e-7
        )

        pool = torch.tensor([[0, 1, 2], [2, 3, 4], [0, 4, 5], [1, 3, 5]])
        old_data, new_data = self.batch(baseline), self.batch(model)
        old_data[baseline.IUnseen], new_data[model.IUnseen] = pool, pool
        torch.testing.assert_close(
            baseline.recommend_from_pool(old_data),
            model.recommend_from_pool(new_data),
            rtol=1e-6, atol=1e-7
        )

    def test_all_v3_arms_parameter_isolation_and_finite_gradients(self):
        for arm in ("B0", "ET-ref", "DP-ref", "DP-placebo", "DP-zero"):
            with self.subTest(arm=arm):
                model = self.construct(arm)
                groups = model.marked_params()
                groups = [{**group, "params": list(group["params"])} for group in groups]
                ids = [id(p) for group in groups for p in group["params"]]
                self.assertEqual(len(ids), len(set(ids)), "Duplicate parameters across optimizer groups.")
                self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
                self.assertIsNone(groups[0]["smoother"])
                self.assertIsNotNone(groups[1]["smoother"])

                loss = model.fit(self.batch(model))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_blocked_knn_matches_dense_baseline(self):
        baseline = baseline_class(self.cfg)(self.dataset)
        model = self.construct("B0")
        model.cfg.knn_chunk_size = 1
        features = torch.randn(6, 7, generator=torch.Generator().manual_seed(812))
        expected = baseline.get_knn_graph(features, 2)
        computed = model.get_knn_graph(features, 2)
        torch.testing.assert_close(computed, expected)

    def test_checkpoint_roundtrip_and_resume(self):
        model = self.construct("DP-ref")
        optimizer = AdamWSEvo(model.marked_params(), lr=0.001, weight_decay=0.1)
        loss = model.fit(self.batch(model))
        loss.backward()
        optimizer.step()

        path = Path(self.directory.name) / "resume_v3.pt"
        save_training_checkpoint(path, model, optimizer, 10, extra={"best": 0.5})

        resumed = self.construct("DP-ref")
        resumed_optimizer = AdamWSEvo(resumed.marked_params(), lr=0.001, weight_decay=0.1)
        payload = load_training_checkpoint(path, resumed, resumed_optimizer)
        self.assertEqual(payload["epoch"], 10)
        self.assertEqual(payload["extra"]["best"], 0.5)

        for current, opt in [(model, optimizer), (resumed, resumed_optimizer)]:
            opt.zero_grad()
            current.fit(self.batch(current)).backward()
            opt.step()

        torch.testing.assert_close(model.Item.embeddings.weight, resumed.Item.embeddings.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.User.embeddings.weight, resumed.User.embeddings.weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
