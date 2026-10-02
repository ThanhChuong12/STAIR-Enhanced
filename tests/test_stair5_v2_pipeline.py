"""Source-level baseline parity, arm integration and checkpoint roundtrip."""
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

import models.freerec_compat  # Install the same compatibility layer as the engine.
import freerec

from models.stair5_v2 import STAIR5_v2_Model
from models.stair5_v1 import STAIR5_v1_Model
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother
from models.stair5_v2_utils import save_training_checkpoint, load_training_checkpoint
_fixture_spec = importlib.util.spec_from_file_location('stair5_v1_fixture', Path(__file__).with_name('test_stair5_v1_pipeline.py'))
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
MockDataset = _fixture_module.MockDataset


def baseline_class(cfg):
    """Execute the unmodified baseline class only, excluding parser/training side effects."""
    path = Path(__file__).resolve().parents[1] / 'main.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'STAIR')
    namespace = dict(torch=torch, nn=nn, F=F, freerec=freerec, os=os, math=math,
                     cfg=cfg, Dict=Dict, Tuple=Tuple, Smoother=Smoother)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['STAIR']


class TestV2Pipeline(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.dataset = MockDataset(4, 6)
        self.dataset.path = self.directory.name
        # Distinct modality data, real SVD + kNN + MI initialization exercised.
        generator = torch.Generator().manual_seed(17)
        for name in ['text.pkl', 'image.pkl']:
            with open(Path(self.directory.name) / name, 'wb') as handle:
                pickle.dump(torch.randn(6, 8, generator=generator).numpy(), handle)
        self.cfg = types.SimpleNamespace(
            embedding_dim=4, num_layers=3, gamma=.2, mfiles=['text.pkl', 'image.pkl'],
            num_neighbors=[2, 1], root=self.directory.name, dataset='toy',
            v2_arm='B0', edge_strength=.25, edge_mix=.25, evidence_shrinkage=5.,
            graph_cache_dir=None, lhc_arm='B0', lambda_lhc=3e-4, lhc_tau=.3,
            lhc_kappa=1., radius_cap=2., w_hybrid=0., eps_taylor=1e-4,
            eps_floor=.01, eps_beta=.05, warmup_start=20, warmup_end=50,
            lr=.001, weight_decay=.1, beta1=.9, beta2=.999, device=torch.device('cpu'))
        self.cfg.beta3 = .1 + .9 * (torch.arange(4) / 4).pow(.2)

    def construct(self, arm='B0'):
        cfg = copy.copy(self.cfg)
        cfg.v2_arm = arm
        torch.manual_seed(2)
        return STAIR5_v2_Model(self.dataset, cfg)

    @staticmethod
    def batch(model):
        # Duplicate IDs deliberately stress deduplication and positive masking.
        return {model.User: torch.tensor([[0], [0], [1], [2]]),
                model.Item: torch.tensor([[0], [0], [1], [2]]),
                model.INeg: torch.tensor([[3], [4], [4], [5]])}

    def test_b0_actual_baseline_mi_fsc_bpr_ranking_and_optimizer_steps(self):
        torch.manual_seed(2)
        baseline = baseline_class(self.cfg)(self.dataset)
        model = self.construct()
        torch.testing.assert_close(baseline.User.embeddings.weight, model.User.embeddings.weight, rtol=0, atol=0)
        torch.testing.assert_close(baseline.Item.embeddings.weight, model.Item.embeddings.weight, rtol=0, atol=0)
        optimizers = [AdamWSEvo(m.marked_params(), lr=.001, weight_decay=.1) for m in [baseline, model]]
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
        torch.testing.assert_close(baseline.recommend_from_full(self.batch(baseline)), model.recommend_from_full(self.batch(model)), rtol=1e-6, atol=1e-7)
        pool = torch.tensor([[0, 1, 2], [2, 3, 4], [0, 4, 5], [1, 3, 5]])
        old_data, new_data = self.batch(baseline), self.batch(model)
        old_data[baseline.IUnseen], new_data[model.IUnseen] = pool, pool
        torch.testing.assert_close(baseline.recommend_from_pool(old_data), model.recommend_from_pool(new_data), rtol=1e-6, atol=1e-7)

    def test_all_arms_finite_gradients_and_isolated_groups(self):
        for arm in ['B0', 'ET', 'H0-A', 'E0-A', 'HC-A', 'ET-H0', 'ET-placebo']:
            with self.subTest(arm=arm):
                model = self.construct(arm)
                model.update_epoch(50)
                groups = model.marked_params()
                groups = [{**group, 'params': list(group['params'])} for group in groups]
                ids = [id(parameter) for group in groups for parameter in group['params']]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
                self.assertIsNone(groups[0]['smoother'])
                self.assertIsNotNone(groups[1]['smoother'])
                loss = model.fit(self.batch(model))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_positive_matrix_handles_duplicate_ids_and_unseen_users(self):
        model = self.construct('H0-A')
        users = torch.tensor([0, 1, 3])
        items = torch.tensor([0, 1, 4])
        training_pairs = {tuple(pair) for pair in self.dataset.edge_index.T.tolist()}
        expected = torch.tensor([[(int(u), int(i)) in training_pairs for i in items] for u in users])
        torch.testing.assert_close(model._build_positive_matrix_vectorized(users, items, torch.device('cpu')), expected)

    def test_blocked_knn_matches_dense_baseline_without_ties(self):
        model = self.construct()
        model.cfg.knn_chunk_size = 1
        features = torch.randn(6, 7, generator=torch.Generator().manual_seed(812))
        expected = STAIR5_v1_Model.get_knn_graph(model, features, 2)
        torch.testing.assert_close(model.get_knn_graph(features, 2), expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda_sparse_step_and_lhc(self):
        model = self.construct('ET-H0').cuda()
        model.update_epoch(50)
        optimizer = AdamWSEvo(model.marked_params(), lr=.001, weight_decay=.1)
        data = {key: value.cuda() for key,value in self.batch(model).items()}
        loss = model.fit(data)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(model.Item.embeddings.weight).all())

    def test_checkpoint_model_optimizer_scales_and_resume_update(self):
        model = self.construct('ET-H0')
        model.update_epoch(50)
        optimizer = AdamWSEvo(model.marked_params(), lr=.001, weight_decay=.1)
        loss = model.fit(self.batch(model))
        loss.backward()
        optimizer.step()
        scales = {name: getattr(model, name).clone() for name in ['q_u0', 'q_u1', 'q_i0', 'q_i1', 'M_norm']}
        path = Path(self.directory.name)/'resume.pt'
        save_training_checkpoint(path, model, optimizer, 50)
        resumed = self.construct('ET-H0')
        resumed_optimizer = AdamWSEvo(resumed.marked_params(), lr=.001, weight_decay=.1)
        payload = load_training_checkpoint(path, resumed, resumed_optimizer)
        resumed.update_epoch(payload['epoch'])
        for name, value in scales.items():
            torch.testing.assert_close(value, getattr(resumed, name), rtol=0, atol=0)
        for current, opt in [(model, optimizer), (resumed, resumed_optimizer)]:
            opt.zero_grad()
            current.fit(self.batch(current)).backward()
            opt.step()
        torch.testing.assert_close(model.Item.embeddings.weight, resumed.Item.embeddings.weight, rtol=0, atol=0)
        torch.testing.assert_close(model.User.embeddings.weight, resumed.User.embeddings.weight, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
