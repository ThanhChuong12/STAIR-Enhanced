"""Production-path CSGC tests, including the actual baseline class from main.py."""
from __future__ import annotations

import ast
import copy
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.freerec_compat
import freerec

from models.stair4_v4 import STAIR4V4
from models.stair4_v4_graph import CSGCOptions, build_baseline_raw_graph, build_operator, symmetric_normalize
from models.stair4_v4_utils import (atomic_save, build_train_csr, build_candidate_statistics,
    calibrated_signal, degree_strata, midrank_cdf, capture_rng, restore_rng)
from optimizers.stair4_v4_smoother import STAIR4V4Smoother
from optimizers.utils import Smoother
from optimizers.AdamW import AdamWSEvo

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def pytest_configure(config):
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        Path(basetemp).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def toy(tmp_path):
    from main_stair4_v4 import build_dataset
    path = tmp_path / "Processed" / "CSGCToy"
    path.mkdir(parents=True)
    # Every item appears in training, with shared and unique histories.
    for split, offset in (("train", 0), ("valid", 7), ("test", 11)):
        lines = ["USER\tITEM\tTIMESTAMP"]
        for user in range(8):
            items = [(3 * user + j) % 24 for j in range(5)] if split == "train" else [(3 * user + offset) % 24]
            lines += [f"{user}\t{item}\t1" for item in items]
        (path / f"{split}.txt").write_text("\n".join(lines) + "\n")
    rng = np.random.default_rng(13)
    for name in ("textual_modality.pkl", "visual_modality.pkl"):
        with (path / name).open("wb") as stream:
            pickle.dump(rng.normal(size=(24, 8)).astype(np.float32), stream)
    config = SimpleNamespace(root=str(tmp_path), dataset="CSGCToy", embedding_dim=4,
        num_layers=3, gamma=.2, mfiles="textual_modality.pkl,visual_modality.pkl", num_neighbors="5-1",
        lr=.001, weight_decay=.1, alpha=0., ablation_id="V4-B1", knn_block_size=24,
        graph_cache_dir=str(tmp_path / "graph-cache"))
    return build_dataset(config), config


def actual_baseline_class(config):
    """Execute the unchanged STAIR class without running main.py's CLI parser."""
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "STAIR")
    config = copy.copy(config)
    config.mfiles = config.mfiles.split(",")
    config.num_neighbors = list(map(int, config.num_neighbors.split("-")))
    config.beta3 = .1 + .9 * (torch.arange(config.embedding_dim) / config.embedding_dim).pow(config.gamma)
    scope = dict(torch=torch, nn=nn, F=F, freerec=freerec, cfg=config, math=math, os=os,
                 Dict=Dict, Tuple=Tuple, Optional=Optional, Smoother=Smoother)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "main.py"), "exec"), scope)
    return scope["STAIR"]


def test_candidate_statistics_dense_oracle_duplicates():
    edges = np.array([[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 0, 2, 0, 1, 3]])
    train = build_train_csr(edges, 4, 5)
    pairs = np.array([[0, 1], [0, 2], [1, 3], [2, 4]])
    stats = build_candidate_statistics(train, pairs)
    dense = train.toarray()
    weights = 1 / np.maximum(dense.sum(1), 1)
    oracle = dense.T @ (weights[:, None] * dense)
    oracle_v = dense.T @ (weights[:, None] ** 2 * dense)
    np.testing.assert_allclose(stats.c, oracle[pairs[:, 0], pairs[:, 1]])
    np.testing.assert_allclose(stats.v, oracle_v[pairs[:, 0], pairs[:, 1]])
    np.testing.assert_allclose(stats.q, dense.T @ weights)
    assert stats.c[-1] == stats.effective_support[-1] == 0


def test_midrank_ties_and_degree_bins():
    np.testing.assert_allclose(midrank_cdf([0, 0, 1, 2]), [.25, .25, .625, .875])
    strata = degree_strata(np.ones(4), np.array([[0, 1], [1, 2], [2, 3]]), 4)
    assert len(set(strata)) == 1


def test_shrinkage_and_controls():
    train = build_train_csr(np.array([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 0, 1]]), 4, 4)
    stats = build_candidate_statistics(train, [[0, 1], [0, 2], [1, 3]])
    opts = CSGCOptions(min_stratum_edges=1)
    h, r, _ = calibrated_signal(stats, opts)
    n, d = stats.effective_support, stats.item_degree
    expected = n / (n + 5) * np.sqrt(d[stats.pairs[:, 0]] / (d[stats.pairs[:, 0]] + 10)
                                             * d[stats.pairs[:, 1]] / (d[stats.pairs[:, 1]] + 10))
    np.testing.assert_allclose(r, expected)
    assert h[-1] == r[-1] == 0
    _, no_shrink, _ = calibrated_signal(stats, CSGCOptions(ablation_id="V4-NS"))
    np.testing.assert_array_equal(no_shrink, [1, 1, 0])
    shuffled = CSGCOptions(ablation_id="V4-SH")
    np.testing.assert_array_equal(calibrated_signal(stats, shuffled)[0], calibrated_signal(stats, shuffled)[0])


@pytest.mark.parametrize("seed", range(4))
def test_sparse_graph_algebra_and_no_new_edges(seed):
    gen = torch.Generator().manual_seed(seed)
    feats = [torch.randn(12, 5, generator=gen), torch.randn(12, 6, generator=gen)]
    raw, _ = build_baseline_raw_graph(feats, [3, 1], 4)
    rng = np.random.default_rng(seed)
    edges = np.stack([rng.integers(0, 6, 40), rng.integers(0, 12, 40)])
    bundle = build_operator(raw, edges, 6, CSGCOptions(min_stratum_edges=2))
    s = bundle.operator.to_dense()
    torch.testing.assert_close(s, s.T)
    assert torch.equal(raw.to_dense() > 0, s > 0)
    assert torch.linalg.eigvalsh(s).abs().max() <= 1 + 1e-6
    assert bundle.operator.grad_fn is None
    assert .5 <= bundle.diagnostics["multiplier_quantiles"][0] <= bundle.diagnostics["multiplier_quantiles"][-1] <= 1.5


def test_empty_and_isolated_graph():
    raw = torch.zeros(4, 4).to_sparse_csr()
    bundle = build_operator(raw, np.empty((2, 0), dtype=int), 3, CSGCOptions())
    assert bundle.operator._nnz() == 0
    assert bundle.diagnostics["candidates"] == 0
    raw = torch.diag(torch.tensor([1., 0, 2, 0])).to_sparse_csr()
    result = build_operator(raw, np.empty((2, 0), dtype=int), 3, CSGCOptions())
    torch.testing.assert_close(result.operator.to_dense(), symmetric_normalize(raw).to_dense())


def test_cache_identity_and_corruption(tmp_path):
    raw = (torch.ones(4, 4) - torch.eye(4)).to_sparse_csr()
    edges = np.array([[0, 0, 1, 1], [0, 1, 1, 2]])
    opts = CSGCOptions()
    first = build_operator(raw, edges, 2, opts, cache_dir=tmp_path, data_identity={"mapping": "a"})
    first_path = next(tmp_path.glob("*.pt"))
    second = build_operator(raw, edges, 2, opts, cache_dir=tmp_path, data_identity={"mapping": "a"})
    assert second.diagnostics["cache_hit"]
    torch.testing.assert_close(first.operator.to_dense(), second.operator.to_dense(), rtol=0, atol=0)
    for identity in ({"mapping": "b"}, {"mapping": "a", "features": "changed"}):
        assert not build_operator(raw, edges, 2, opts, cache_dir=tmp_path, data_identity=identity).diagnostics["cache_hit"]
    changed = edges.copy()
    changed[1, 0] = 3
    assert not build_operator(raw, changed, 2, opts, cache_dir=tmp_path).diagnostics["cache_hit"]
    path = first_path
    payload = torch.load(path, weights_only=True)
    payload["values"][0] += 1
    atomic_save(path, payload)
    with pytest.raises(ValueError, match="checksum"):
        build_operator(raw, edges, 2, opts, cache_dir=tmp_path, data_identity={"mapping": "a"})


def test_gate0_skips_statistics_and_rng(monkeypatch):
    import models.stair4_v4_graph as module
    def forbidden(*args, **kwargs):
        raise AssertionError("Gate 0 must skip calibration")
    monkeypatch.setattr(module, "build_candidate_statistics", forbidden)
    raw = torch.eye(3).to_sparse_csr()
    baseline = symmetric_normalize(raw)
    rng = torch.get_rng_state().clone()
    bundle = build_operator(raw, np.empty((2, 0), dtype=int), 2,
                            CSGCOptions(alpha=0, ablation_id="V4-B1"), baseline=baseline)
    assert bundle.operator is baseline
    assert torch.equal(rng, torch.get_rng_state())


def test_production_gate0_multistep_and_ranking(toy):
    dataset, config = toy
    torch.manual_seed(3)
    baseline = actual_baseline_class(config)(dataset)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(3)
    model = STAIR4V4(dataset, config)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    torch.testing.assert_close(model.mAdj.to_dense(), baseline.mAdj.to_dense(), atol=0, rtol=0)
    groups = model.parameter_groups()
    assert len(groups) == 2 and sum(len(g["params"]) for g in groups) == 2
    opt_a = AdamWSEvo(baseline.marked_params(), lr=config.lr, weight_decay=config.weight_decay)
    opt_b = AdamWSEvo(groups, lr=config.lr, weight_decay=config.weight_decay)
    for _ in range(4):
        data = {model.User: torch.tensor([[0], [1], [1], [2]]),
                model.Item: torch.tensor([[0], [3], [3], [6]]),
                model.INeg: torch.tensor([[13], [15], [16], [18]])}
        for current, opt in ((baseline, opt_a), (model, opt_b)):
            opt.zero_grad()
            loss = current.fit(data)
            loss.backward()
            if current is model:
                current.smoother.arm_static()
            opt.step()
            if current is model:
                current.smoother.clear_step_snapshot()
        for a, b in zip(baseline.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(opt_a.state[a][key], opt_b.state[b][key], rtol=0, atol=0)
    baseline.reset_ranking_buffers()
    model.reset_ranking_buffers()
    data[model.IUnseen] = torch.tensor([[5, 7, 8]] * 4)
    torch.testing.assert_close(model.recommend_from_full(data), baseline.recommend_from_full(data), rtol=0, atol=0)
    torch.testing.assert_close(model.recommend_from_pool(data), baseline.recommend_from_pool(data), rtol=0, atol=0)


def test_model_checkpoint_and_no_leakage(toy, tmp_path):
    dataset, config = toy
    config.alpha, config.ablation_id = .25, "V4-C"
    model = STAIR4V4(dataset, config)
    path = tmp_path / "model.pt"
    atomic_save(path, model.state_dict())
    restored = STAIR4V4(dataset, config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for a, b in zip(model.encode(), restored.encode()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    payload["_extra_state"]["schema"] = -1
    before = restored.Item.embeddings.weight.clone()
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_dict(payload)
    assert torch.equal(before, restored.Item.embeddings.weight)
    # Held-out files do not enter the graph builder or its identity.
    (Path(dataset.path) / "valid.txt").write_text("USER\tITEM\tTIMESTAMP\n0\t23\t99\n")
    another = STAIR4V4(dataset, config)
    assert another.graph_identity == model.graph_identity


def test_smoother_lifecycle_and_identity_control():
    matrix = torch.tensor([[0., 1.], [1., 0.]])
    beta = torch.tensor([.1, .5, .8])
    x = torch.randn(2, 3)
    base = Smoother(matrix, beta, 3, "neumann")
    smoother = STAIR4V4Smoother(lambda z: matrix @ z, beta, 3, .2)
    with pytest.raises(RuntimeError, match="arm_static"):
        smoother(x)
    smoother.arm_static()
    torch.testing.assert_close(smoother(x), .8 * base(x) + .2 * x)
    smoother.clear_step_snapshot()
    assert not smoother._armed


def test_rng_roundtrip(tmp_path):
    import random
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    atomic_save(tmp_path / "rng.pt", state)
    restore_rng(torch.load(tmp_path / "rng.pt", weights_only=True))
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_real_cli_train_evaluate_and_resume(toy, tmp_path):
    """Exercise real FreeRec sampling, masking, checkpointing and selected metrics."""
    import json
    import subprocess
    import sys
    import yaml
    _, config = toy
    values = vars(config).copy()
    values.update(epochs=2, batch_size=16, seed=17, num_workers=0, device="cpu",
        optimizer="adamwsevo", description="CSGCToy", CHECKPOINT_FREQ=1,
        monitors=["LOSS", "Recall@20", "NDCG@20"], which4best="NDCG@20",
        alpha=.25, ablation_id="V4-C")
    yaml_path = tmp_path / "toy.yaml"
    yaml_path.write_text(yaml.safe_dump(values))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", PYTHONUTF8="1")
    def run(identifier, directory, extra=()):
        command = [sys.executable, str(ROOT / "main_stair4_v4.py"), "--config", str(yaml_path),
                   "--id", identifier, "--run-dir", str(directory), *extra]
        completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=150)
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-7000:]
        return json.loads((directory / "run_manifest.json").read_text())
    identifier = "v4_test_" + tmp_path.name
    original_dir = tmp_path / "original"
    manifest = run(identifier, original_dir)
    assert manifest["status"] == "completed"
    records = [json.loads(line) for line in (original_dir / "evaluation.jsonl").read_text().splitlines()]
    assert {r["mode"] for r in records if r["selected_checkpoint"]} == {"valid", "test"}
    resumed_dir = tmp_path / "resumed"
    # Same run ID points to the epoch-1 checkpoint (before evaluation and train).
    resumed = run(identifier, resumed_dir, ("--resume",))
    assert resumed["selected_epoch"] == manifest["selected_epoch"]
    expected = json.loads((original_dir / "epochs.jsonl").read_text().splitlines()[-1])
    actual = json.loads((resumed_dir / "epochs.jsonl").read_text().splitlines()[-1])
    assert actual["epoch"] == expected["epoch"] == 2
    assert actual["bpr"] == expected["bpr"]


def test_config_inheritance_and_validation():
    from main_stair4_v4 import load_config
    for dataset, batch, decay, gamma in (("baby", 1024, .3, .1), ("sports", 1024, .1, .2), ("electronics", 4096, .1, .4)):
        config = load_config(ROOT / f"configs/dataset_stair4_v4_{dataset}.yaml")
        assert (config["batch_size"], config["weight_decay"], config["gamma"]) == (batch, decay, gamma)
        assert config["which4best"] == "NDCG@20" and config["epochs"] == 500
        CSGCOptions.from_config(SimpleNamespace(**config))
    for changes in ({"alpha": float("nan")}, {"edge_epsilon": 1}, {"support_tau": 0},
                    {"ablation_id": "V4-NA"}, {"identity_mix": .2}, {"degree_bins": 0}):
        with pytest.raises(ValueError):
            CSGCOptions(**changes)


def test_bsc_polynomial_spectrum_and_spmm_count():
    matrix = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    beta = torch.tensor([.2, .6], dtype=torch.float64)
    count = [0]
    def operator(x):
        count[0] += 1
        return matrix @ x
    smoother = STAIR4V4Smoother(operator, beta, 3)
    smoother.arm_static()
    x = torch.eye(2, dtype=torch.float64)
    result = smoother(x)
    expected = torch.zeros_like(x)
    for j in range(2):
        polynomial = sum(beta[j] ** k * torch.linalg.matrix_power(matrix, k) for k in range(4))
        polynomial *= (1 - beta[j]) / (1 - beta[j] ** 4)
        assert torch.linalg.eigvalsh(polynomial).abs().max() <= 1 + 1e-12
        expected[:, j] = polynomial @ x[:, j]
    torch.testing.assert_close(result, expected)
    assert count[0] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_cuda_model_backward_step_and_scoring(toy):
    dataset, config = toy
    config.alpha, config.ablation_id = .25, "V4-C"
    model = STAIR4V4(dataset, config).cuda()
    opt = AdamWSEvo(model.parameter_groups())
    data = {model.User: torch.tensor([[0], [1]], device="cuda"),
            model.Item: torch.tensor([[0], [3]], device="cuda"),
            model.INeg: torch.tensor([[9], [10]], device="cuda")}
    for _ in range(2):
        opt.zero_grad()
        model.fit(data).backward()
        model.smoother.arm_static()
        try:
            opt.step()
        finally:
            model.smoother.clear_step_snapshot()
    model.reset_ranking_buffers()
    assert torch.isfinite(model.recommend_from_full(data)).all()
