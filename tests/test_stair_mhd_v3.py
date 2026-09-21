"""Numerical tests against real PyTorch/FreeRec, using tiny synthetic data.

Run with python -m pytest tests/test_stair_mhd_v3.py -q. No trained results
or user datasets are required. The baseline class is compiled from its AST
so importing it cannot parse CLI arguments or launch a training run.
"""
import ast
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict, Tuple
try:
    import models.freerec_compat
except Exception:
    pass
import freerec
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from models.stair_mhd_v3 import MHDOptions, STAIR_MHD_v3, symmetric_infonce
from models.stair_mhd_v3_utils import (
    apply_P_H, behavioral_statistics, build_incidence, exact_knn, graph_state,
)
from optimizers.mhd_smoother import MHDSmoother
from optimizers.utils import Smoother
from optimizers.AdamW import AdamWSEvo
from main_stair_mhd_v3 import (
    build_optimizer, training_step, save_training_checkpoint,
    load_training_checkpoint, load_config, CoachForMHD, seed_everything,
)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class TinyDataset:
    """Dataset adapter with real FreeRec Field objects and graph functions."""
    def __init__(self, path):
        from freerec.data.fields import Field
        from freerec.data.tags import USER, ITEM, ID
        self.path = str(path)
        user, item = Field('USER', USER, ID), Field('ITEM', ITEM, ID)
        user.count, item.count = 4, 8
        self.fields = [user, item]
        self.edges = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                                   [0, 1, 2, 1, 3, 4, 2, 4, 5, 5, 6, 7]])

    def train(self):
        return self

    def to_bigraph(self, edge_type):
        return {'u2i': SimpleNamespace(edge_index=self.edges)}

    def to_normalized_adj(self, normalization):
        edges = self.edges.clone()
        edges[1] += 4
        edges = torch.cat((edges, edges.flip(0)), 1)
        edges, weights = freerec.graph.to_normalized(edges, normalization=normalization)
        return torch.sparse_coo_tensor(edges, weights, (12, 12)).to_sparse_csr()


@pytest.fixture
def setup(tmp_path):
    generator = torch.Generator().manual_seed(911)
    for name, width in [('textual_modality.pkl', 5), ('visual_modality.pkl', 7)]:
        with (tmp_path / name).open('wb') as stream:
            pickle.dump(torch.randn(8, width, generator=generator), stream)
    cfg = SimpleNamespace(embedding_dim=4, num_layers=3, gamma=0.2,
                          mfiles='textual_modality.pkl,visual_modality.pkl', num_neighbors='3-1',
                          root=str(tmp_path), dataset='toy', optimizer='adamwsevo',
                          lr=0.001, weight_decay=0.1, beta1=0.9, beta2=0.999,
                          warmup_epochs=0, ramp_epochs=1, knn_block_size=3,
                          cl_weight_target=0.1, budget_weight_target=0.0001)
    return TinyDataset(tmp_path), cfg


def batch(model):
    return {model.User: torch.tensor([[0], [1], [2], [3]]),
            model.Item: torch.tensor([[0], [3], [5], [7]]),
            model.INeg: torch.tensor([[7], [6], [1], [0]])}


def baseline_class(cfg):
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'STAIR')
    cfg.beta3 = 0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    # Reproduce main.py's post-Parser normalization before class construction.
    if isinstance(cfg.mfiles, str):
        cfg.mfiles = cfg.mfiles.split(',')
    if isinstance(cfg.num_neighbors, str):
        cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))
    namespace = dict(torch=torch, nn=nn, F=F, freerec=freerec, cfg=cfg,
                     os=os, math=math, Dict=Dict, Tuple=Tuple, Smoother=Smoother)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / 'main.py'), 'exec'), namespace)
    return namespace['STAIR']


def test_factorized_operator_psd_and_scaling():
    h = build_incidence(torch.tensor([[1, 2], [0, 2], [0, 1], [1, 2]])).double()
    w = torch.tensor([.2, .8, .5, .4], dtype=torch.double)
    x = torch.randn(4, 3, dtype=torch.double)
    state = graph_state(h, w)
    dense = h.to_dense()
    inv = state['degree'].rsqrt()
    expected = inv[:, None] * ((dense * (w / dense.sum(0))) @ dense.T) * inv[None, :]
    actual = apply_P_H(x, (h,), (state,), x.new_ones(1))
    torch.testing.assert_close(actual, expected @ x)
    torch.testing.assert_close(expected, expected.T)
    eigenvalues = torch.linalg.eigvalsh(expected)
    assert eigenvalues.min() >= -1e-12 and eigenvalues.max() <= 1 + 1e-12
    torch.testing.assert_close(actual, apply_P_H(x, (h,), (graph_state(h, 3*w),), x.new_ones(1)))


def test_autograd_includes_weighted_degrees():
    h = build_incidence(torch.tensor([[1], [2], [0], [2]])).double()
    w = torch.tensor([.2, .4, .7, .9], dtype=torch.double, requires_grad=True)
    x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda weights, feats: apply_P_H(
        feats, (h,), (graph_state(h, weights),), feats.new_ones(1)), (w, x))


def test_knn_matches_baseline_and_binary_incidence():
    features = torch.randn(7, 5, generator=torch.Generator().manual_seed(22))
    neighbors = exact_knn(features, 2, block_size=2)
    dense = F.normalize(features, dim=-1) @ F.normalize(features, dim=-1).T
    dense.fill_diagonal_(-10.)
    edges, _ = freerec.graph.get_knn_graph(dense, 2, symmetric=False)
    torch.testing.assert_close(neighbors, edges[1].reshape(7, 2))
    h = build_incidence(torch.tensor([[0, 1, 1], [1, 0, 0]])).to_dense()
    assert torch.equal(h, torch.ones(2, 2))
    with pytest.raises(ValueError):
        exact_knn(torch.zeros(4, 2), 1)


def test_behavior_is_candidate_only_and_deduplicated():
    features = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
    edges = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
    neighbors = torch.tensor([[1], [0], [1]])
    C, rho, x = behavioral_statistics(edges, 3, 3, neighbors, features, support_s=2)
    torch.testing.assert_close(C, torch.tensor([1., 1., 0.]))
    torch.testing.assert_close(rho, torch.tensor([.5, .5, 1/3]))
    duplicated = behavioral_statistics(torch.cat((edges, edges[:, :2]), 1), 3, 3, neighbors, features, 2)
    for original, duplicate in zip((C, rho, x), duplicated):
        torch.testing.assert_close(original, duplicate)
    assert torch.isfinite(x).all()


def test_duplicate_ids_and_singletons():
    first, second = torch.randn(5, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True)
    duplicated = symmetric_infonce(first, second, torch.tensor([1, 3, 1, 3]), nn.Identity(), .2)
    unique = symmetric_infonce(first, second, torch.tensor([1, 3]), nn.Identity(), .2)
    torch.testing.assert_close(duplicated, unique)
    singleton = symmetric_infonce(first, second, torch.tensor([1, 1]), nn.Identity(), .2)
    singleton.backward()
    assert singleton.item() == 0 and torch.isfinite(first.grad).all()


def test_smoother_baseline_snapshot_isolation_and_lifecycle():
    adj = torch.eye(3).to_sparse_csr()
    beta = torch.tensor([.1, .5, .9])
    smoother = MHDSmoother(lambda x, state: adj @ x if state is None else x*state['scale'], beta, 3)
    x = torch.randn(3, 3)
    with pytest.raises(RuntimeError):
        smoother(x)
    smoother.use_baseline()
    torch.testing.assert_close(smoother(x), Smoother(adj, beta, 3, 'neumann')(x), rtol=0, atol=0)
    scale = torch.tensor(.5, requires_grad=True)
    smoother.set_step_snapshot({'scale': scale*2})
    snapshot = smoother.step_snapshot['scale']
    assert snapshot.grad_fn is None and not snapshot.requires_grad
    with torch.no_grad():
        scale.fill_(.1)
    assert snapshot.item() == 1
    smoother.clear_step_snapshot()
    with pytest.raises(RuntimeError):
        smoother(x)


def test_static_prepare_groups_and_live_gate_steps(setup):
    ds, cfg = setup
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(1)
    assert all(b.grad_fn is None and not b.requires_grad for b in model.buffers())
    groups = model.marked_params()
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert [g['lr'] for g in groups] == [cfg.lr, cfg.lr, .1*cfg.lr]
    assert [g['weight_decay'] for g in groups] == [cfg.weight_decay, cfg.weight_decay, 0]
    optimizer = build_optimizer(model, cfg)
    last_before = model.gates[0][-1].weight.detach().clone()
    hidden_grads = []
    for _ in range(4):
        loss, diagnostics = training_step(model, optimizer, batch(model))
        assert torch.isfinite(loss)
        assert model.smoother.step_snapshot is None
        hidden_grads.append(diagnostics['gate_0_hidden_grad'])
        assert diagnostics['gate_0_output_grad'] > 0
    assert hidden_grads[0] == 0 and max(hidden_grads[1:]) > 0
    assert not torch.equal(last_before, model.gates[0][-1].weight)


def test_bpr_does_not_train_gate(setup):
    ds, cfg = setup
    cfg.cl_weight_target = cfg.budget_weight_target = cfg.bsc_mix_target = 0.
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(100)
    model.fit(batch(model)).backward()
    assert all(p.grad is None for p in model.gates.parameters())
    model.smoother.clear_step_snapshot()


def test_baseline_recovery_loss_gradient_update_and_rankings(setup):
    ds, cfg = setup
    cfg.cl_weight_target = cfg.budget_weight_target = cfg.bsc_mix_target = 0.
    torch.manual_seed(123)
    baseline = baseline_class(cfg)(ds)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    model = STAIR_MHD_v3(ds, cfg)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    torch.testing.assert_close(model.Item.embeddings.weight, baseline.Item.embeddings.weight, rtol=0, atol=0)
    torch.testing.assert_close(model.mAdj.to_dense(), baseline.mAdj.to_dense())
    torch.testing.assert_close(model.User.embeddings.weight, baseline.User.embeddings.weight, rtol=0, atol=0)
    model.set_epoch(100)
    l0, l1 = baseline.fit(batch(baseline)), model.fit(batch(model))
    torch.testing.assert_close(l0, l1, rtol=0, atol=0)
    l0.backward()
    l1.backward()
    for left, right in [(baseline.User, model.User), (baseline.Item, model.Item)]:
        torch.testing.assert_close(left.embeddings.weight.grad, right.embeddings.weight.grad, rtol=0, atol=0)
    base_optimizer = AdamWSEvo(baseline.marked_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = build_optimizer(model, cfg)
    base_optimizer.step()
    optimizer.step()
    model.smoother.clear_step_snapshot()
    for left, right in [(baseline.User, model.User), (baseline.Item, model.Item)]:
        torch.testing.assert_close(left.embeddings.weight, right.embeddings.weight, rtol=0, atol=0)
    baseline.reset_ranking_buffers()
    model.reset_ranking_buffers()
    torch.testing.assert_close(baseline.recommend_from_full(batch(baseline)),
                               model.recommend_from_full(batch(model)), rtol=0, atol=0)
    pools = torch.tensor([[0, 2, 5], [1, 3, 7], [2, 4, 6], [3, 5, 0]])
    b0, b1 = batch(baseline), batch(model)
    b0[baseline.IUnseen], b1[model.IUnseen] = pools, pools
    torch.testing.assert_close(baseline.recommend_from_pool(b0), model.recommend_from_pool(b1), rtol=0, atol=0)


def test_checkpoint_rng_optimizer_roundtrip(setup, tmp_path):
    ds, cfg = setup
    seed_everything(41)
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(2)
    optimizer = build_optimizer(model, cfg)
    training_step(model, optimizer, batch(model))
    checkpoint = tmp_path / 'state.pt'
    save_training_checkpoint(checkpoint, model, optimizer, epoch=2)
    payload = torch.load(checkpoint, weights_only=True)
    assert all('smoother' not in group for group in payload['optimizer']['param_groups'])
    expected_rng = (torch.rand(3), np.random.rand(3), random.random())
    restored = STAIR_MHD_v3(ds, cfg)
    restored_optimizer = build_optimizer(restored, cfg)
    assert load_training_checkpoint(checkpoint, restored, restored_optimizer) == 2
    assert restored.current_epoch == 2
    actual_rng = (torch.rand(3), np.random.rand(3), random.random())
    torch.testing.assert_close(actual_rng[0], expected_rng[0], rtol=0, atol=0)
    np.testing.assert_array_equal(actual_rng[1], expected_rng[1])
    assert actual_rng[2] == expected_rng[2]
    assert restored_optimizer.param_groups[1]['smoother'] is restored.smoother
    for m in [model, restored]:
        m.reset_ranking_buffers()
    torch.testing.assert_close(model.recommend_from_full(batch(model)), restored.recommend_from_full(batch(restored)), rtol=0, atol=0)
    training_step(model, optimizer, batch(model))
    training_step(restored, restored_optimizer, batch(restored))
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    restored.data_hashes = {**restored.data_hashes, 'train': 'different'}
    with pytest.raises(ValueError, match='data_hashes'):
        load_training_checkpoint(checkpoint, restored, restored_optimizer)


def test_cleanup_even_on_optimizer_failure(setup):
    ds, cfg = setup
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(1)
    optimizer = build_optimizer(model, cfg)
    def fail():
        raise RuntimeError('deliberate test failure')
    optimizer.step = fail
    with pytest.raises(RuntimeError, match='deliberate'):
        training_step(model, optimizer, batch(model))
    assert model.smoother.step_snapshot is None


def test_config_inherits_baseline_and_evaluation_methods_unchanged():
    cfg = load_config(ROOT / 'configs/dataset_mhd_v3.yaml')
    baseline = load_config(ROOT / 'configs/Amazon2014Baby_550_MMRec.yaml')
    for key, value in baseline.items():
        assert cfg[key] == value
    for method in ('evaluate', 'check_best', 'save_best', 'remove_seen'):
        assert getattr(CoachForMHD, method) is getattr(freerec.launcher.Coach, method)


@pytest.mark.parametrize('mode', ['concrete', 'straight_through'])
def test_optional_stochastic_gate_has_gradient(setup, mode):
    ds, cfg = setup
    cfg.gate_mode = mode
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(1)
    loss, diag = training_step(model, build_optimizer(model, cfg), batch(model))
    assert torch.isfinite(loss) and diag['gate_0_output_grad'] > 0


def test_warmup_boundary_and_snapshot_precedes_step(setup):
    ds, cfg = setup
    cfg.warmup_epochs, cfg.ramp_epochs = 10, 20
    model = STAIR_MHD_v3(ds, cfg)
    model.set_epoch(10)
    assert model.coefficients() == (0., 0., 0.)
    model.set_epoch(11)
    assert model.coefficients()[0] == pytest.approx(cfg.cl_weight_target / 20)
    state = model.fresh_graph_state(stochastic=False)
    model.fit(batch(model))
    snapshot = model.smoother.step_snapshot
    for expected, actual in zip(state['states'], snapshot['states']):
        torch.testing.assert_close(expected['weights'], actual['weights'])
        assert actual['weights'].grad_fn is None and actual['degree'].grad_fn is None
    model.smoother.clear_step_snapshot()


def test_real_freerec_cli_training_and_checkpoint(tmp_path):
    """Exercise real Dataset/DataLoader2/Coach/evaluation, with a 90s bound."""
    folder = tmp_path / 'data' / 'Processed' / 'mhd_toy'
    folder.mkdir(parents=True)
    interactions = {'train': [(i % 4, i) for i in range(24)],
                    'valid': [(u, (u + 1) % 4) for u in range(4)],
                    'test': [(u, (u + 2) % 4) for u in range(4)]}
    for split, pairs in interactions.items():
        (folder / f'{split}.txt').write_text('USER\tITEM\n' + ''.join(
            f'{u}\t{i}\n' for u, i in pairs), encoding='utf-8')
    g = torch.Generator().manual_seed(17)
    for filename, width in [('textual_modality.pkl', 5), ('visual_modality.pkl', 7)]:
        with (folder / filename).open('wb') as stream:
            pickle.dump(torch.randn(24, width, generator=g), stream)
    config = {**load_config(ROOT / 'configs/dataset_mhd_v3.yaml'),
              'root': str(tmp_path / 'data'), 'dataset': 'mhd_toy', 'device': 'cpu',
              'embedding_dim': 4, 'epochs': 2, 'batch_size': 8, 'num_workers': 0,
              'warmup_epochs': 0, 'ramp_epochs': 1, 'eval_freq': 1,
              'CHECKPOINT_FREQ': 1, 'log2console': False, 'seed': 17}
    config_path = tmp_path / 'smoke.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MPLCONFIGDIR': str(tmp_path / 'mpl'),
           'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, str(ROOT / 'main_stair_mhd_v3.py'),
                             '--config', str(config_path), '--id', 'mhd-smoke'], cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    logs = list(tmp_path.rglob('mhd_diagnostics.jsonl'))
    assert len(logs) == 1
    import json
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert [record['epoch'] for record in records] == [1, 2]
    assert records[0]['zeta'] == pytest.approx(.25)
    assert any(record['gate_0_output_grad'] > 0 for record in records)
    # Load the real Coach checkpoint (saved before the final training epoch),
    # including FreeRec monitor/scheduler state and callback-free optimizer.
    resumed = subprocess.run([sys.executable, str(ROOT / 'main_stair_mhd_v3.py'),
                              '--config', str(config_path), '--id', 'mhd-smoke', '--resume'],
                             cwd=tmp_path, env=env, capture_output=True, text=True,
                             encoding='utf-8', timeout=90)
    assert resumed.returncode == 0, resumed.stdout[-5000:] + resumed.stderr[-5000:]
    continued = [json.loads(line) for line in logs[0].read_text().splitlines()][-1]
    assert continued['epoch'] == 2
    for key in records[-1]:
        if key not in ('seconds', 'cuda_peak_allocated', 'cuda_peak_reserved'):
            assert continued[key] == records[-1][key], key
