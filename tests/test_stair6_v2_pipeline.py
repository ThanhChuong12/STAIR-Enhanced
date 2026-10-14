"""Native V8 parity, autograd, checkpoint and actual FreeRec entrypoint tests."""

import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import types

import pytest
import torch
import yaml

import models.freerec_compat
from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL
from main_stair6_v2 import build_config
from models.stair5_v8 import STAIR5_v8_Model
from models.stair6_v2 import STAIR6_v2_Model
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from optimizers.AdamW import AdamWSEvo


class Dataset:
    def __init__(self, path):
        self.path = str(path)
        user, item = Field("User", USER, ID), Field("Item", ITEM, ID)
        user.count, item.count = 12, 72
        self.fields = FieldTuple([user, item, Field("Label", LABEL)])
        pairs = [(u, i) for u in range(4) for i in range(24)]
        pairs += [(u, i) for u in range(4, 12) for i in range(24, 72)]
        pairs += [(0, 0), (0, 1)]
        self.edge_index = torch.tensor(pairs, dtype=torch.int64).T.contiguous()

    def train(self):
        return self

    def to_bigraph(self, edge_type="u2i"):
        return {"u2i": types.SimpleNamespace(edge_index=self.edge_index)}

    def to_normalized_adj(self, normalization="sym"):
        users, items = self.edge_index
        items = items + 12
        rows, cols = torch.cat((users, items)), torch.cat((items, users))
        degrees = torch.bincount(rows, minlength=84).float().clamp_min(1)
        weights = degrees[rows].rsqrt() * degrees[cols].rsqrt()
        return torch.sparse_coo_tensor(torch.stack((rows, cols)), weights, (84, 84)).coalesce().to_sparse_csr()


@pytest.fixture
def setup(tmp_path):
    generator = torch.Generator().manual_seed(29)
    for name in ("text.pkl", "image.pkl"):
        feature = torch.randn(72, 80, generator=generator)
        feature[1:24:2] = feature[:24:2] + 1e-5 * torch.randn(12, 80, generator=generator)
        with (tmp_path / name).open("wb") as handle:
            pickle.dump(feature, handle)
    for name in ("train.txt", "valid.txt", "test.txt"):
        (tmp_path / name).write_text("USER\tITEM\tTIMESTAMP\n0\t0\t1\n", encoding="utf-8")
    cfg = types.SimpleNamespace(
        embedding_dim=8, num_layers=3, gamma=0.2, mfiles=["text.pkl", "image.pkl"], num_neighbors=[5, 1],
        lr=0.001, weight_decay=0.1, beta1=0.9, beta2=0.999, batch_size=4, seed=1,
        eval_freq=5, num_workers=0, ranking="full", which4best="NDCG@20",
        lambda_nlgcl=0.01, nlgcl_G=1, nlgcl_tau=0.2, nlgcl_alpha=0.5, cl_chunk_size=2,
        eta=0.1, k_cf=5, c_min=2, t_shrinkage=5.0,
        v8_arm="WMSG-core", semantic_mode="weighted", edge_power=2.0, edge_floor=0.05, semantic_mix=1.0,
        alignment="off", relation_gate="off", relation_strength=0.0, bsc_residual=0.0, lambda_dirichlet=0.0,
        anchor_seed=2718, anchor_per_stratum=1024, relation_threshold=0.7, relation_temperature=0.1,
        placebo_seed=1, knn_device="cpu", knn_chunk_size=32, cf_block_size=16, cf_memory_budget_mib=128.0,
        graph_cache_dir=str(tmp_path / "cache"), artifact_dir=str(tmp_path / "artifacts"),
        root=str(tmp_path), dataset="ToyDataset", can_mode="prune", can_interpolation=.25,
    )
    cfg.beta3 = .1 + .9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    return Dataset(tmp_path), cfg, tmp_path


def model(setup, parent=False, **updates):
    dataset, cfg, _ = setup
    cfg = copy.copy(cfg)
    for name, value in updates.items():
        setattr(cfg, name, value)
    torch.manual_seed(31)
    return (STAIR5_v8_Model if parent else STAIR6_v2_Model)(dataset, cfg)


def batch(current):
    return {current.User: torch.tensor([0, 0, 1, 4]), current.Item: torch.tensor([0, 1, 1, 24]),
            current.INeg: torch.tensor([24, 25, 26, 0])}


def optimizer(current):
    return AdamWSEvo(current.marked_params(), lr=current.cfg.lr, weight_decay=current.cfg.weight_decay)


@pytest.mark.parametrize("bypass", [{"can_mode": "off"}, {"can_interpolation": 0.0}])
def test_exact_parent_rng_gradients_and_two_step_recovery(setup, bypass):
    reference = model(setup, parent=True)
    rng = torch.get_rng_state().clone()
    control = model(setup, **bypass)
    assert torch.equal(rng, torch.get_rng_state())
    assert reference.graph_fingerprint == control.graph_fingerprint
    assert reference.data_fingerprint == control.data_fingerprint
    optimizers = [optimizer(reference), optimizer(control)]
    for _ in range(2):
        for current, opt in zip((reference, control), optimizers):
            opt.zero_grad(set_to_none=True)
            current.fit(batch(current)).backward()
        for left, right in zip(reference.parameters(), control.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        for opt in optimizers:
            opt.step()
        for left, right in zip(reference.parameters(), control.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_active_changes_only_operator_and_does_not_retain_preprocessing(setup):
    reference = model(setup, parent=True)
    rng = torch.get_rng_state().clone()
    current = model(setup)
    assert torch.equal(rng, torch.get_rng_state())
    assert current.can_metadata["active"] and current.can_metadata["final_support_equal_v8"]
    assert current.graph_fingerprint != reference.graph_fingerprint
    assert current.baseline_graph_fingerprint == reference.graph_fingerprint
    assert current.can_metadata["relative_operator_delta_vs_v8"] > 0
    for left, right in zip(reference.parameters(), current.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(reference.training_objective(batch(reference)), current.training_objective(batch(current))):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    groups = current.marked_params()
    assert len(groups) == 2 and groups[0]["smoother"] is None and groups[1]["smoother"] is not None
    assert len({id(p) for group in groups for p in group["params"]}) == len(list(current.parameters()))
    assert all(buffer.grad_fn is None for buffer in current.buffers())
    assert not hasattr(current, "_can_features") and not hasattr(current, "_can_candidates")
    assert not hasattr(current, "can_probabilities")


def test_backward_and_exact_checkpoint_continuation(setup):
    current = model(setup)
    opt = optimizer(current)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        current.fit(batch(current)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters())
        opt.step()
    path = setup[2] / "checkpoint.pt"
    save_training_checkpoint(path, current, opt, 3)
    restored = model(setup)
    restored_opt = optimizer(restored)
    assert load_training_checkpoint(path, restored, restored_opt)["epoch"] == 3
    for instance, instance_opt in ((current, opt), (restored, restored_opt)):
        instance_opt.zero_grad(set_to_none=True)
        instance.fit(batch(instance)).backward()
        instance_opt.step()
    for left, right in zip(current.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_changed_dose_or_mask_checkpoint_rejected_before_weight_mutation(setup):
    current = model(setup)
    for changes in ({"can_interpolation": .5}, {"can_mode": "fixed"}):
        other = model(setup, **changes)
        before = other.Item.embeddings.weight.detach().clone()
        with pytest.raises(ValueError, match="provenance"):
            other.load_state_dict(current.state_dict())
        torch.testing.assert_close(other.Item.embeddings.weight, before, rtol=0, atol=0)


def test_fsc_full_and_pool_ranking_unchanged_at_same_weights(setup):
    parent, current = model(setup, parent=True), model(setup)
    for instance in (parent, current):
        instance.eval()
        instance.reset_ranking_buffers()
    users = torch.tensor([0, 4])
    ref = parent.recommend_from_full({parent.User: users})
    full = current.recommend_from_full({current.User: users})
    torch.testing.assert_close(ref, full, rtol=0, atol=0)
    items = torch.tensor([[0, 24, 60], [1, 25, 61]])
    pool = current.recommend_from_pool({current.User: users, current.IUnseen: items})
    torch.testing.assert_close(pool, full.gather(1, items))


def test_all_three_configs_and_control_cli(tmp_path):
    for dataset in ("Baby", "Sports", "Electronics"):
        args = ["--config", f"configs/Amazon2014{dataset}_STAIR6_v2.yaml", "--device", "cpu",
                "--artifact-dir", str(tmp_path / dataset), "--preprocess-only"]
        cfg = build_config(args)
        assert cfg.can_mode == "prune" and cfg.can_interpolation == .25
        assert cfg.num_neighbors == [5, 1] and cfg.embedding_dim == 64 and cfg.preprocess_only
        assert cfg.cl_chunk_size == (128 if dataset == "Electronics" else 1024)
        assert build_config([*args, "--can-mode", "off"]).can_mode == "off"
    with pytest.raises(ValueError, match="v8_arm"):
        build_config([*args, "--v8-arm", "WMSG-SAP"])


def test_actual_training_entrypoint_smoke_and_resume(setup):
    dataset, cfg, path = setup
    directory = path / "ToyDataset"
    directory.mkdir()
    for name in ("text.pkl", "image.pkl"):
        shutil.copy2(path / name, directory / name)
    lines = ["USER\tITEM\tTIMESTAMP\n"]
    lines.extend(f"{u}\t{i}\t{step+1}\n" for step, (u, i) in enumerate(dataset.edge_index.T.tolist()))
    (directory / "train.txt").write_text("".join(lines), encoding="utf-8")
    for filename, offset in (("valid.txt", 0), ("test.txt", 12)):
        rows = ["USER\tITEM\tTIMESTAMP\n"]
        rows.extend(f"{u}\t{24+u+offset//12 if u < 4 else u+offset}\t1000\n" for u in range(12))
        (directory / filename).write_text("".join(rows), encoding="utf-8")
    config = {key: value for key, value in vars(cfg).items() if key != "beta3"}
    config.update(epochs=2, eval_freq=1, CHECKPOINT_FREQ=1, device="cpu", batch_size=64,
                  optimizer="adamwsevo", eval_valid=True, eval_test=False)
    config_path = path / "native_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    def execute(*extra):
        return subprocess.run([sys.executable, "-X", "utf8", str(root / "main_stair6_v2.py"),
            "--config", str(config_path), *extra], cwd=root, env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=120)
    result = execute()
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    artifact = Path(cfg.artifact_dir)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["can"]["active"] and manifest["selection"] == "validation NDCG@20"
    assert torch.load(artifact / "training_checkpoint.pt", weights_only=True)["epoch"] == 2
    assert (artifact / "best_model.pt").is_file()
    resumed = execute("--epochs", "3", "--resume-from", str(artifact / "training_checkpoint.pt"))
    assert resumed.returncode == 0, resumed.stdout[-6000:] + resumed.stderr[-6000:]
    records = [json.loads(line) for line in (artifact / "training_telemetry.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    assert (artifact / "resume_manifest.json").is_file()
