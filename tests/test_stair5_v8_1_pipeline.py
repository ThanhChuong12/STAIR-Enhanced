"""V8.1 integration contracts and real FreeRec subprocess smoke/resume."""

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
from main_stair5_v8_1 import build_config
from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v8_1 import STAIR5_v8_1_Model
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from optimizers.AdamW import AdamWSEvo


class Dataset:
    def __init__(self, path, n_items=72):
        self.path = str(path)
        user, item = Field("User", USER, ID), Field("Item", ITEM, ID)
        user.count, item.count = 12, n_items
        self.n_items = n_items
        self.fields = FieldTuple([user, item, Field("Label", LABEL)])
        pairs = [(u, i) for u in range(4) for i in range(24)]
        pairs += [(u, i) for u in range(4, 12) for i in range(24, n_items)]
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
        degrees = torch.bincount(rows, minlength=self.n_items + 12).float().clamp_min(1)
        weights = degrees[rows].rsqrt() * degrees[cols].rsqrt()
        return torch.sparse_coo_tensor(torch.stack((rows, cols)), weights, (self.n_items + 12, self.n_items + 12)).coalesce().to_sparse_csr()


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
        embedding_dim=64, num_layers=3, gamma=0.2, mfiles=["text.pkl", "image.pkl"], num_neighbors=[5, 1],
        lr=0.001, weight_decay=0.1, beta1=0.9, beta2=0.999, batch_size=4, seed=1,
        eval_freq=5, num_workers=0, ranking="full", which4best="NDCG@20",
        lambda_nlgcl=0.01, nlgcl_G=1, nlgcl_tau=0.2, nlgcl_alpha=0.5, cl_chunk_size=2,
        eta=0.1, k_cf=5, c_min=2, t_shrinkage=5.0,
        v8_arm="WMSG-core", semantic_mode="weighted", edge_power=2.0, edge_floor=0.05, semantic_mix=1.0,
        alignment="off", relation_gate="off", relation_strength=0.0, bsc_residual=0.0, lambda_dirichlet=0.0,
        anchor_seed=2718, anchor_per_stratum=1024, relation_threshold=0.7, relation_temperature=0.1,
        placebo_seed=1, knn_device="cpu", knn_chunk_size=32, cf_block_size=16, cf_memory_budget_mib=128.0,
        graph_cache_dir=str(tmp_path / "cache"), artifact_dir=str(tmp_path / "artifacts"),
        root=str(tmp_path), dataset="ToyDataset", consensus_mode="soft", consensus_diagnostic_interval=1,
    )
    cfg.beta3 = .1 + .9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    return Dataset(tmp_path), cfg, tmp_path


def model(setup, parent=False, **updates):
    dataset, cfg, _ = setup
    cfg = copy.copy(cfg)
    for name, value in updates.items():
        setattr(cfg, name, value)
    torch.manual_seed(31)
    return (STAIR5_v8_Model if parent else STAIR5_v8_1_Model)(dataset, cfg)


def batch(current):
    return {current.User: torch.tensor([0, 0, 1, 4]), current.Item: torch.tensor([0, 1, 1, 24]),
            current.INeg: torch.tensor([24, 25, 26, 0])}


def optimizer(current):
    return AdamWSEvo(current.marked_params(), lr=current.cfg.lr, weight_decay=current.cfg.weight_decay)


@pytest.mark.parametrize("bypass", [{"consensus_mode": "off"}, {"consensus_rho": 0.0}, {}])
def test_parent_rng_and_two_step_warmup_recovery(setup, bypass):
    reference = model(setup, parent=True)
    rng = torch.get_rng_state().clone()
    control = model(setup, **bypass)
    assert torch.equal(rng, torch.get_rng_state())
    assert reference.graph_fingerprint == control.graph_fingerprint
    assert reference.data_fingerprint == control.data_fingerprint
    optimizers = [optimizer(reference), optimizer(control)]
    for epoch in (1, 2):
        control.current_epoch = epoch
        for current, opt in zip((reference, control), optimizers):
            opt.zero_grad(set_to_none=True)
            current.fit(batch(current)).backward()
        for left, right in ((reference.User, control.User), (reference.Item, control.Item)):
            torch.testing.assert_close(left.embeddings.weight.grad, right.embeddings.weight.grad, rtol=0, atol=0)
        assert all(p.grad is None for p in control.consensus_residual.parameters())
        for opt in optimizers:
            opt.step()
        for left, right in ((reference.User, control.User), (reference.Item, control.Item)):
            torch.testing.assert_close(left.embeddings.weight, right.embeddings.weight, rtol=0, atol=0)
        assert all(p not in optimizers[1].state for p in control.consensus_residual.parameters())


def test_active_live_gradient_and_fresh_step_graph(setup):
    current = model(setup)
    current.configure_consensus_backend()
    current.current_epoch = 61
    reference = model(setup, parent=True)
    assert current.consensus_metadata["cf_cache_hit"]
    assert current.consensus_backend["verified"]
    assert not hasattr(current, "_consensus_features")
    for left, right in zip(current.encode(), reference.encode()):
        if isinstance(left, list):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    before = [p.detach().clone() for p in current.consensus_residual.parameters()]
    opt = optimizer(current)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        total, bpr, _ = current.training_objective(batch(current))
        torch.testing.assert_close(bpr, STAIR5_v8_Model.training_objective(current, batch(current))[1], rtol=0, atol=0)
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in current.consensus_residual.parameters())
        opt.step()
        assert all(buffer.grad_fn is None for buffer in current.buffers())
        assert all(value.grad_fn is None for value in current.consensus_diagnostics().values())
    assert all(not torch.equal(old, new) for old, new in zip(before, current.consensus_residual.parameters()))
    assert current.consensus_diagnostics()["residual_ratio_max"] <= current.consensus_rho + 1e-6


@pytest.mark.parametrize("gate", ["learned", "fixed"])
def test_groups_and_unchanged_full_pool_scoring(setup, gate):
    current, parent = model(setup, consensus_gate=gate), model(setup, parent=True)
    groups = current.marked_params()
    assert len(groups) == (3 if gate == "learned" else 2)
    assert groups[0]["smoother"] is None and groups[1]["smoother"] is not None
    assert len({id(p) for group in groups for p in group["params"]}) == len(list(current.parameters()))
    if gate == "learned":
        assert groups[2]["lr"] == .0001 and groups[2]["weight_decay"] == 0
        assert groups[2]["smoother"] is None
        assert sum(p.numel() for p in groups[2]["params"]) == 41
    for instance in (parent, current):
        instance.eval()
        instance.reset_ranking_buffers()
    users = torch.tensor([0, 4])
    full = current.recommend_from_full({current.User: users})
    torch.testing.assert_close(full, parent.recommend_from_full({parent.User: users}), rtol=0, atol=0)
    items = torch.tensor([[0, 24, 60], [1, 25, 61]])
    torch.testing.assert_close(current.recommend_from_pool({current.User: users, current.IUnseen: items}), full.gather(1, items))


def test_duplicate_items_inverse_expansion_matches_full_batch(setup):
    current = model(setup)
    current.current_epoch = 61
    _, _, layers = current.encode()
    data = batch(current)
    messages = torch.sparse.mm(current.consensus_csr, layers[0][current.User.count:])
    ids = data[current.Item]
    query = current.consensus_residual(layers[1][current.User.count:][ids], messages[ids],
                                      current.consensus_coverage[ids], current.consensus_rho)
    expected = current.nlgcl(layers, data[current.User], ids, item_query=query)
    torch.testing.assert_close(current.training_objective(data)[2], expected)


def test_checkpoint_next_step_exact_and_provenance_rejection(setup):
    current = model(setup)
    current.configure_consensus_backend()
    current.current_epoch = 61
    opt = optimizer(current)
    current.fit(batch(current)).backward()
    opt.step()
    path = setup[2] / "checkpoint.pt"
    save_training_checkpoint(path, current, opt, 61)
    restored = model(setup)
    restored.configure_consensus_backend()
    restored_opt = optimizer(restored)
    assert load_training_checkpoint(path, restored, restored_opt)["epoch"] == 61
    assert restored.current_epoch == 61 and restored.consensus_rho == .2
    for instance, instance_opt in ((current, opt), (restored, restored_opt)):
        instance.current_epoch = 62
        instance_opt.zero_grad(set_to_none=True)
        instance.fit(batch(instance)).backward()
        instance_opt.step()
    for left, right in zip(current.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for changes in ({"consensus_rho": .1}, {"consensus_floor": .1}, {"consensus_mode": "cf-only"}):
        incompatible = model(setup, **changes)
        before = incompatible.Item.embeddings.weight.detach().clone()
        with pytest.raises(ValueError, match="provenance"):
            incompatible.load_state_dict(current.state_dict())
        torch.testing.assert_close(incompatible.Item.embeddings.weight, before, rtol=0, atol=0)


def test_diagnostic_pooling_uses_counts_global_max_and_second_moments(setup):
    current = model(setup)
    current.reset_consensus_diagnostics()
    current._accumulate_consensus_diagnostics({"count": torch.tensor(2.), "gate_mean": torch.tensor(.2),
        "gate_second_moment": torch.tensor(.04), "gate_std": torch.tensor(0.), "residual_ratio_max": torch.tensor(.15)})
    current._accumulate_consensus_diagnostics({"count": torch.tensor(1.), "gate_mean": torch.tensor(.8),
        "gate_second_moment": torch.tensor(.64), "gate_std": torch.tensor(0.), "residual_ratio_max": torch.tensor(.1)})
    result = current.consensus_diagnostics()
    torch.testing.assert_close(result["gate_mean"], torch.tensor(.4))
    torch.testing.assert_close(result["gate_std"], torch.tensor(.08).sqrt())
    torch.testing.assert_close(result["residual_ratio_max"], torch.tensor(.15))
    assert result["count"] == 3


def test_sparse_layouts_backward_and_auto_fallback(setup, monkeypatch):
    csr = model(setup, consensus_sparse_layout="csr")
    coo = model(setup, consensus_sparse_layout="coo")
    for current in (csr, coo):
        current.configure_consensus_backend()
        current.current_epoch = 61
        current.fit(batch(current)).backward()
    torch.testing.assert_close(csr.Item.embeddings.weight.grad, coo.Item.embeddings.weight.grad, rtol=1e-5, atol=1e-7)
    auto = model(setup)
    original = auto._probe_sparse_backward
    def probe(layout, device):
        if layout == "csr":
            raise NotImplementedError("CSR backward not supported")
        return original(layout, device)
    monkeypatch.setattr(auto, "_probe_sparse_backward", probe)
    with pytest.warns(RuntimeWarning, match="using COO"):
        auto.configure_consensus_backend()
    assert auto.consensus_csr.layout == torch.sparse_coo
    assert auto.consensus_backend["verified"] and auto.consensus_backend["fallback_reason"]
    def oom(*args):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(auto, "_probe_sparse_backward", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        auto.configure_consensus_backend()


def test_all_six_cli_configs_and_raw_arm_rejection(tmp_path):
    for dataset in ("Baby", "Sports", "Electronics"):
        for dimension in (64, 256):
            suffix = "" if dimension == 64 else "_256D"
            args = ["--config", f"configs/Amazon2014{dataset}_STAIR5_v8_1{suffix}.yaml", "--device", "cpu",
                    "--artifact-dir", str(tmp_path / f"{dataset}_{dimension}"), "--preprocess-only"]
            cfg = build_config(args)
            assert cfg.consensus_rho == .2 and cfg.embedding_dim == dimension
            assert cfg.beta3.shape == (dimension,) and cfg.num_neighbors == [5, 1]
            assert cfg.cl_chunk_size == (128 if dataset == "Electronics" else 1024)
            assert build_config([*args, "--consensus-mode", "off"]).consensus_mode == "off"
    with pytest.raises(ValueError, match="v8_arm"):
        build_config([*args, "--v8-arm", "WMSG-SAP"])


def test_stale_beta_schedule_rejected(setup):
    with pytest.raises(ValueError, match="beta3"):
        model(setup, beta3=torch.ones(64))


def test_256d_model_active_backward(setup):
    _, cfg, path = setup
    generator = torch.Generator().manual_seed(256)
    for name in ("text.pkl", "image.pkl"):
        with (path / name).open("wb") as handle:
            pickle.dump(torch.randn(272, 280, generator=generator), handle)
    cfg = copy.copy(cfg)
    cfg.embedding_dim = 256
    cfg.beta3 = .1 + .9 * (torch.arange(256) / 256).pow(cfg.gamma)
    current = STAIR5_v8_1_Model(Dataset(path, n_items=272), cfg)
    current.configure_consensus_backend()
    current.current_epoch = 61
    current.fit(batch(current)).backward()
    assert current.Item.embeddings.weight.shape == (272, 256)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters())


def test_zero_dose_bypasses_consensus_modules(setup, monkeypatch):
    current = model(setup, consensus_rho=0)
    assert current.consensus_is_empty and current.consensus_metadata["mode"] == "off"
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero dose must not evaluate the gate")
    monkeypatch.setattr(current.consensus_residual, "forward", forbidden)
    current.current_epoch = 61
    current.fit(batch(current)).backward()
    assert all(p.grad is None for p in current.consensus_residual.parameters())


def test_actual_entrypoint_preprocess_training_and_resume(setup):
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
                  optimizer="adamwsevo", eval_valid=True, eval_test=False,
                  consensus_warmup=0, consensus_ramp=0)
    config_path = path / "native_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    def execute(*extra):
        return subprocess.run([sys.executable, "-X", "utf8", str(root / "main_stair5_v8_1.py"),
            "--config", str(config_path), *extra], cwd=root, env=environment,
            capture_output=True, text=True, encoding="utf-8", timeout=300)
    preprocess_artifact = path / "preprocess"
    preflight = execute("--preprocess-only", "--artifact-dir", str(preprocess_artifact))
    assert preflight.returncode == 0, preflight.stdout[-6000:] + preflight.stderr[-6000:]
    assert not (preprocess_artifact / "training_checkpoint.pt").exists()
    assert not json.loads((preprocess_artifact / "manifest.json").read_text())["consensus_backend"]["verified"]
    result = execute()
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    artifact = Path(cfg.artifact_dir)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["consensus"]["active"] and manifest["selection"] == "validation NDCG@20"
    assert manifest["consensus_backend"]["verified"]
    assert torch.load(artifact / "training_checkpoint.pt", weights_only=True)["epoch"] == 2
    resumed = execute("--epochs", "3", "--resume-from", str(artifact / "training_checkpoint.pt"))
    assert resumed.returncode == 0, resumed.stdout[-6000:] + resumed.stderr[-6000:]
    records = [json.loads(line) for line in (artifact / "consensus_telemetry.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    assert all(record["rho"] == .2 and record["last_batch_gate_grad_norm"] > 0 for record in records)
    assert (artifact / "resume_manifest.json").is_file()


