"""Integration contracts for the actual v5 model, baseline and CLI."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_stair4_v4 import toy, actual_baseline_class
from models.stair4_v5 import STAIR4V5
from models.stair4_v5_features import TrainHistory, build_train_csr, prepare_modality_features
from models.stair4_v5_objectives import masked_candidate_cross_entropy
from models.stair4_v5_sampling import UniformTrainUnseenSampler
from models.stair4_v5_utils import calibrate_scale, id_only_features

ROOT = Path(__file__).resolve().parents[1]


def make_model(dataset, config):
    baseline = actual_baseline_class(config)(dataset)
    with torch.no_grad():
        hu, hi = baseline.encode()
    edges = dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index.cpu()
    csr = build_train_csr(edges, len(hu), len(hi))
    features = id_only_features(hi, 6, seed=22)
    history = TrainHistory(csr, features)
    model = STAIR4V5(dataset, hu, hi, history, eta=.5, residual_dim=3, identity={"teacher": "toy"})
    return baseline, model, csr


def test_actual_baseline_disabled_full_pool_and_state_roundtrip(toy, tmp_path):
    dataset, config = toy
    baseline, model, _ = make_model(dataset, config)
    baseline.reset_ranking_buffers()
    model.reset_ranking_buffers()
    users = torch.tensor([[0], [1], [0]])
    pool = torch.tensor([[1, 4, 1], [3, 5, 6], [2, 0, 2]])
    data = {model.User: users, model.IUnseen: pool}
    assert torch.equal(baseline.recommend_from_full(data), model.recommend_from_full(data))
    assert torch.equal(baseline.recommend_from_pool(data), model.recommend_from_pool(data))
    model.residual_enabled = True
    model.reset_ranking_buffers()
    expected = model.recommend_from_pool(data).clone()
    file = tmp_path / "head.pt"
    torch.save(model.state_dict(), file)
    restored = copy.deepcopy(model)
    restored.residual_enabled = False
    restored.load_state_dict(torch.load(file, weights_only=True))
    restored.reset_ranking_buffers()
    assert torch.equal(expected, restored.recommend_from_pool(data))
    bad = copy.deepcopy(model.state_dict())
    bad["_extra_state"]["identity"] = {"teacher": "wrong"}
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_dict(bad)


def test_frozen_teacher_multistep_training_and_duplicate_candidates(toy):
    dataset, config = toy
    baseline, model, csr = make_model(dataset, config)
    frozen = (model.h_user.clone(), model.h_item.clone())
    original_baseline = [p.detach().clone() for p in baseline.parameters()]
    model.residual_enabled = True
    rows, pos = csr.nonzero()
    users, positives = torch.tensor(rows[:8]), torch.tensor(pos[:8])
    sampler = UniformTrainUnseenSampler(csr, seed=12)
    optimizer = torch.optim.AdamW(model.head.parameter_groups())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in model.head.parameters())
    initial = {n: p.detach().clone() for n, p in model.head.named_parameters()}
    for _ in range(3):
        negatives, mask = sampler.sample(users, 4)
        candidates = torch.cat([positives[:, None], negatives], 1)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_candidate_cross_entropy(model.score_candidates(users, positives, candidates), mask)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
        optimizer.step()
        model.invalidate_ranking_cache()
        assert not model.ranking_buffer
    assert all(not torch.equal(initial[n], p) for n, p in model.head.named_parameters())
    assert torch.equal(frozen[0], model.h_user) and torch.equal(frozen[1], model.h_item)
    assert model.h_user.grad_fn is None and model.h_user.grad is None
    assert all(torch.equal(old, new) for old, new in zip(original_baseline, baseline.parameters()))


def test_calibration_rng_isolation_and_singleton_dense_users(toy):
    dataset, config = toy
    _, model, csr = make_model(dataset, config)
    rng = torch.get_rng_state().clone()
    sigma, metadata = calibrate_scale(model.h_user, model.h_item, csr, seed=98, limit=20)
    assert sigma > 0 and metadata["samples"] == 20
    assert torch.equal(rng, torch.get_rng_state())
    assert (sigma, metadata) == calibrate_scale(model.h_user, model.h_item, csr, seed=98, limit=20)


def test_real_cli_two_stages_and_standalone_teacher_validation(toy, tmp_path):
    _, cfg = toy
    env = {**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    run = tmp_path / "run"
    args = [sys.executable, str(ROOT / "main_stair4_v5.py"), "--root", cfg.root,
            "--dataset", cfg.dataset, "--device", "cpu", "--epochs", "2", "--head-epochs", "2",
            "--embedding-dim", "4", "--residual-dim", "3", "--pca-dim", "6", "--batch-size", "16",
            "--microbatch-size", "8", "--eval-freq", "1", "--num-negatives", "3",
            "--calibration-samples", "20", "--knn-block-size", "24",
            "--graph-cache-dir", str(tmp_path / "graphs"), "--feature-cache-dir", str(tmp_path / "features"),
            "--gamma", ".2", "--weight-decay", ".1", "--seed", "1"]
    result = subprocess.run(args + ["--stage", "all", "--run-dir", str(run)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]
    for phase in ("stage_a", "stage_b"):
        manifest = json.loads((run / phase / "run_manifest.json").read_text())
        assert manifest["status"] == "completed"
        rows = [json.loads(line) for line in (run / phase / "evaluation.jsonl").read_text().splitlines()]
        tests = [r for r in rows if r["mode"] == "test"]
        assert len(tests) == 1 and tests[0]["selected_checkpoint"]
        assert tests[0]["epoch"] == manifest["selected_epoch"]
        assert any(r["epoch"] == 0 and r["mode"] == "valid" for r in rows)
    payload = torch.load(run / "stage_a" / "teacher.pt", weights_only=True)
    assert payload["training_state"]["epoch"] == payload["selected_epoch"]
    second = tmp_path / "standalone"
    result = subprocess.run(args + ["--stage", "b", "--teacher-dir", str(run / "stage_a"),
                                   "--run-dir", str(second), "--no-final-test", "--arm", "id-ss"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]
    rows = [json.loads(line) for line in (second / "stage_b" / "evaluation.jsonl").read_text().splitlines()]
    assert not any(r["mode"] == "test" for r in rows)
    # Exercise the trainable-backbone objective-only control with selected Adam moments.
    control = tmp_path / "control"
    result = subprocess.run(args + ["--stage", "b", "--teacher-dir", str(run / "stage_a"),
                                   "--run-dir", str(control), "--no-final-test", "--arm", "baseline-ss"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]
    manifest = json.loads((control / "stage_b" / "run_manifest.json").read_text())
    assert manifest["optimizer_initialization"] == "selected_stage_a_moments"
    # Interrupt before epoch two, then resume in a new interpreter. Last model,
    # optimizer and private sampling RNG must match the uninterrupted MM run.
    interrupted = tmp_path / "interrupted"
    resume_args = args + ["--stage", "b", "--teacher-dir", str(run / "stage_a"),
                          "--run-dir", str(interrupted)]
    script = (
        "import sys\nimport main_stair4_v5 as m\n"
        f"sys.argv = {resume_args[1:]!r}\n"
        "original = m.ResidualCoach.train_per_epoch\n"
        "def fail(self, epoch):\n"
        "    if epoch == 2: raise RuntimeError('intentional interruption')\n"
        "    return original(self, epoch)\n"
        "m.ResidualCoach.train_per_epoch = fail\nm.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode != 0 and "intentional interruption" in result.stderr
    result = subprocess.run(resume_args + ["--resume"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]
    expected = torch.load(run / "stage_b/checkpoints/last.pt", weights_only=True)
    actual = torch.load(interrupted / "stage_b/checkpoints/last.pt", weights_only=True)
    assert expected["selection"] == actual["selection"]
    assert expected["steps"] == actual["steps"]
    for key, value in expected["model"].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual["model"][key]), key
        else:
            assert value == actual["model"][key]
    assert torch.equal(expected["negative_sampler"]["rng_state"], actual["negative_sampler"]["rng_state"])
    for key, state in expected["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, actual["optimizer"]["state"][key][name]), (key, name)
    # The Stage A checkpoint must carry the selected optimizer snapshot too.
    baseline_interrupted = tmp_path / "baseline_interrupted"
    a_args = args + ["--stage", "a", "--run-dir", str(baseline_interrupted)]
    script = (
        "import sys\nimport main_stair4_v5 as m\n"
        f"sys.argv = {a_args[1:]!r}\n"
        "original = m.BaselineCoach.train_per_epoch\n"
        "def fail(self, epoch):\n"
        "    if epoch == 2: raise RuntimeError('intentional baseline interruption')\n"
        "    return original(self, epoch)\n"
        "m.BaselineCoach.train_per_epoch = fail\nm.main()\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode != 0 and "intentional baseline interruption" in result.stderr
    result = subprocess.run(a_args + ["--resume"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-5000:]
    recovered = torch.load(baseline_interrupted / "stage_a/teacher.pt", weights_only=True)
    assert recovered["selected_epoch"] == payload["selected_epoch"]
    assert torch.equal(recovered["user_vectors"], payload["user_vectors"])
    assert torch.equal(recovered["item_vectors"], payload["item_vectors"])
