"""Real CPU integration checks against the inherited V8 implementation."""

import copy
import hashlib
import json
from pathlib import Path
import pickle
import os
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
from main_stair6_v1 import build_config
from models.stair5_v8 import STAIR5_v8_Model
from models.stair6_v1 import STAIR6_v1_Model
from models.stair6_v1_graph import InactiveCategorySignal
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from models.stair5_v8_graph import tensor_to_scipy
from optimizers.AdamW import AdamWSEvo
from stair6_v1_config import validate_category_config
from tools.audit_stair6_v1_metadata import audit_dataset


class Dataset:
    def __init__(self, path):
        self.path = str(path)
        user, item = Field("User", USER, ID), Field("Item", ITEM, ID)
        user.count, item.count = 12, 72
        self.fields = FieldTuple([user, item, Field("Label", LABEL)])
        pairs = [(u, i) for u in range(4) for i in range(24)]
        pairs += [(u, i) for u in range(4, 12) for i in range(24, 72)]
        pairs += [(0, 0), (0, 1)]  # Native adjacency keeps duplicates; statistics deduplicate.
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
    dataset = Dataset(tmp_path)
    for name in ("train.txt", "valid.txt", "test.txt"):
        (tmp_path / name).write_text("USER\tITEM\tTIMESTAMP\n0\t0\t1\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps([f"ASIN{i}" for i in range(72)]), encoding="utf-8")
    records = [{"item_id": i, "asin": f"ASIN{i}",
                "categories": [["Toy", "A" if i < 24 and i % 2 == 0 else "B" if i < 24 else "C"]]}
               for i in range(72)]
    metadata = tmp_path / "categories.jsonl"
    metadata.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "dataset": "ToyDataset", "n_items": 72, "mapping_verified": True,
        "mapping_source": "Synthetic fixture with explicit item-order mapping",
        "metadata_source": {"source": "Synthetic fixture", "snapshot": "fixture-v1"},
        "files": {"mapping_sha256": digest(mapping), "metadata_sha256": digest(metadata)},
        "baseline": {"features": {name: digest(tmp_path / name) for name in ("text.pkl", "image.pkl")},
                     "splits": {name: digest(tmp_path / name) for name in ("train.txt", "valid.txt", "test.txt")}},
    }), encoding="utf-8")
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
        root=str(tmp_path), dataset="ToyDataset", category_mode="cae", category_source="metadata", category_strength=0.5,
        category_mapping_file=str(mapping), category_metadata_file=str(metadata), category_manifest_file=str(manifest),
    )
    cfg.beta3 = 0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    return dataset, cfg, tmp_path


def make_model(setup, reference=False, **updates):
    dataset, cfg, _ = setup
    cfg = copy.copy(cfg)
    for name, value in updates.items():
        setattr(cfg, name, value)
    torch.manual_seed(31)
    return (STAIR5_v8_Model if reference else STAIR6_v1_Model)(dataset, cfg)


def batch(model):
    return {model.User: torch.tensor([0, 0, 1, 4]), model.Item: torch.tensor([0, 1, 1, 24]),
            model.INeg: torch.tensor([24, 25, 26, 0])}


def optimizer(model):
    return AdamWSEvo(model.marked_params(), lr=model.cfg.lr, weight_decay=model.cfg.weight_decay)


@pytest.mark.parametrize("bypass", [{"category_strength": 0.0}, {"category_mode": "off"}])
def test_exact_zero_intervention_forward_rng_gradient_and_two_step_recovery(setup, bypass):
    reference = make_model(setup, True)
    reference_rng = torch.get_rng_state().clone()
    control = make_model(setup, **bypass)
    assert reference.User.embeddings.weight is not control.User.embeddings.weight
    assert reference.Item.embeddings.weight is not control.Item.embeddings.weight
    assert torch.equal(reference_rng, torch.get_rng_state())
    assert reference.graph_fingerprint == control.graph_fingerprint
    assert reference.data_fingerprint == control.data_fingerprint
    for ref_layer, new_layer in zip(reference.encode()[2], control.encode()[2]):
        torch.testing.assert_close(ref_layer, new_layer, rtol=0, atol=0)
    optimizers = [optimizer(reference), optimizer(control)]
    for _ in range(2):
        results = []
        for model, opt in zip((reference, control), optimizers):
            opt.zero_grad(set_to_none=True)
            loss, bpr, cl = model.training_objective(batch(model))
            results.append((loss.detach(), bpr.detach(), cl.detach()))
            loss.backward()
        for ref, new in zip(*results):
            torch.testing.assert_close(ref, new, rtol=0, atol=0)
        for ref, new in zip(reference.parameters(), control.parameters()):
            torch.testing.assert_close(ref.grad, new.grad, rtol=0, atol=0)
        for opt in optimizers:
            opt.step()
        for ref, new in zip(reference.parameters(), control.parameters()):
            torch.testing.assert_close(ref, new, rtol=0, atol=0)


def test_active_model_changes_only_graph_before_training_and_preserves_parameter_groups(setup):
    reference = make_model(setup, True)
    rng = torch.get_rng_state().clone()
    model = make_model(setup)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.graph_fingerprint != reference.graph_fingerprint
    assert model.baseline_graph_fingerprint == reference.graph_fingerprint
    assert model.category_metadata["category_active"]
    assert model.category_metadata["relative_operator_delta_vs_v8"] > 0
    assert model.category_metadata["bsc_probe_relative_delta"] > 0
    for ref, new in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(ref, new, rtol=0, atol=0)
    for ref, new in zip(reference.training_objective(batch(reference)), model.training_objective(batch(model))):
        torch.testing.assert_close(ref, new, rtol=0, atol=0)
    groups = model.marked_params()
    assert len(groups) == 2 and groups[0]["smoother"] is None and groups[1]["smoother"] is not None
    assert len({id(p) for g in groups for p in g["params"]}) == len(list(model.parameters()))
    assert all(t.grad_fn is None for t in model.buffers())
    assert not hasattr(model, "_category_raw_features")
    assert not hasattr(model, "category_memberships")
    left, right = tensor_to_scipy(reference.mAdj), tensor_to_scipy(model.mAdj)
    assert (left.astype(bool) != right.astype(bool)).nnz == 0


def test_active_autograd_and_checkpoint_continuation(setup):
    model = make_model(setup)
    opt = optimizer(model)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model.fit(batch(model)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    path = setup[2] / "checkpoint.pt"
    save_training_checkpoint(path, model, opt, 3)
    restored = make_model(setup)
    restored_opt = optimizer(restored)
    assert load_training_checkpoint(path, restored, restored_opt)["epoch"] == 3
    for current, current_opt in ((model, opt), (restored, restored_opt)):
        current_opt.zero_grad(set_to_none=True)
        current.fit(batch(current)).backward()
        current_opt.step()
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_same_weights_keep_full_pool_and_fsc_evaluation_identical(setup):
    reference, model = make_model(setup, True), make_model(setup)
    for current in (reference, model):
        current.eval()
        current.reset_ranking_buffers()
    users = torch.tensor([0, 4])
    full_ref = reference.recommend_from_full({reference.User: users})
    full = model.recommend_from_full({model.User: users})
    torch.testing.assert_close(full_ref, full, rtol=0, atol=0)
    pool_ids = torch.tensor([[0, 24, 60], [1, 25, 61]])
    pool = model.recommend_from_pool({model.User: users, model.IUnseen: pool_ids})
    torch.testing.assert_close(pool, full.gather(1, pool_ids))


def test_missing_metadata_fails_before_whitening_but_explicit_control_runs(setup, monkeypatch):
    def expensive(*args, **kwargs):
        raise AssertionError("Whitening must not run for missing category inputs")
    monkeypatch.setattr(STAIR5_v8_Model, "whitening", expensive)
    with pytest.raises(FileNotFoundError, match="category_mapping_file"):
        make_model(setup, category_mapping_file="missing.json")


def test_root_only_metadata_fails_before_whitening(setup, monkeypatch):
    path = setup[2]
    rows = [{"item_id": i, "asin": f"ASIN{i}", "categories": [["Toy"]]} for i in range(72)]
    metadata = path / "categories.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(STAIR5_v8_Model, "whitening", lambda *args: pytest.fail("Unexpected SVD"))
    with pytest.raises(InactiveCategorySignal, match="terminal paths"):
        make_model(setup)


def test_binding_changed_features_splits_and_checkpoint_strength(setup):
    model = make_model(setup)
    changed = make_model(setup, category_strength=0.25)
    before = changed.Item.embeddings.weight.detach().clone()
    with pytest.raises(ValueError, match="provenance"):
        changed.load_state_dict(model.state_dict())
    torch.testing.assert_close(before, changed.Item.embeddings.weight, rtol=0, atol=0)
    (setup[2] / "valid.txt").write_text("USER\tITEM\tTIMESTAMP\n1\t0\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="splits provenance"):
        make_model(setup)


def test_all_three_cli_configs_and_explicit_control_override(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("Baby", "Sports", "Electronics"):
        args = ["--config", str(root / "configs" / f"Amazon2014{name}_STAIR6_v1.yaml"),
                "--device", "cpu", "--artifact-dir", str(tmp_path / name)]
        cfg = build_config(args, check_metadata=False)
        assert cfg.category_strength == 0.5 and cfg.embedding_dim == 64
        assert cfg.category_source == "latent" and cfg.latent_clusters == 50
        assert cfg.cl_chunk_size == (128 if name == "Electronics" else 1024)
        cfg = build_config([*args, "--category-mode", "off"])
        assert cfg.category_mode == "off" and cfg.which4best == "NDCG@20"


def test_metadata_audit_returns_no_go_and_valid_binding(setup):
    _, cfg, path = setup
    missing = audit_dataset(path)
    assert not missing["ready"] and missing["missing_inputs"]
    valid = audit_dataset(path, cfg.category_mapping_file, cfg.category_metadata_file,
                          cfg.category_manifest_file, 72, "ToyDataset", ("text.pkl", "image.pkl"))
    assert valid["ready"] and valid["status"] == "READY_FOR_GRAPH_DIAGNOSTICS"


@pytest.mark.parametrize("category_source", ["metadata", "latent"])
def test_native_training_entrypoint_two_epochs(setup, category_source):
    """Exercise actual FreeRec dataset, datapipe, coach, validation and save/load."""
    dataset, cfg, path = setup
    cfg = copy.copy(cfg)
    if category_source == "latent":
        from models.stair6_v1_latent_categories import build_latent_category_data
        cfg.category_source = "latent"
        cfg.latent_clusters, cfg.latent_projection_dim, cfg.latent_iterations = 6, 8, 10
        with (path / "text.pkl").open("rb") as handle:
            text = pickle.load(handle)
        with (path / "image.pkl").open("rb") as handle:
            image = pickle.load(handle)
        latent = build_latent_category_data([text, image], dataset="ToyDataset",
            feature_manifest=[{"file": "text.pkl", "sha256": "a" * 64},
                              {"file": "image.pkl", "sha256": "b" * 64}],
            split_manifest={"train.txt": "c" * 64}, clusters=6, projection_dim=8, iterations=10)
        # Construct controlled TRAIN co-interest across two content groups.
        # This tests live integration without assuming random groups carry signal.
        items_a = [i for i, group in enumerate(latent.assignments) if group in (0, 1)]
        items_b = [i for i, group in enumerate(latent.assignments) if group not in (0, 1)]
        dataset.edge_index = torch.tensor([(u, i) for u in range(12)
                                          for i in (items_a if u < 4 else items_b)]).T.contiguous()
    train_lines = ["USER\tITEM\tTIMESTAMP\n"]
    train_lines.extend(f"{u}\t{i}\t{step + 1}\n" for step, (u, i) in enumerate(dataset.edge_index.T.tolist()))
    (path / "train.txt").write_text("".join(train_lines), encoding="utf-8")
    for filename, offset in (("valid.txt", 0), ("test.txt", 12)):
        lines = ["USER\tITEM\tTIMESTAMP\n"]
        for user in range(12):
            item = 24 + user + (offset // 12) if user < 4 else user + offset
            lines.append(f"{user}\t{item}\t1000\n")
        (path / filename).write_text("".join(lines), encoding="utf-8")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["baseline"]["splits"] = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("train.txt", "valid.txt", "test.txt")
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    directory = path / "ToyDataset"
    directory.mkdir()
    for filename in ("train.txt", "valid.txt", "test.txt", "text.pkl", "image.pkl",
                     "mapping.json", "categories.jsonl", "manifest.json"):
        if category_source == "metadata" or filename not in ("mapping.json", "categories.jsonl", "manifest.json"):
            shutil.copy2(path / filename, directory / filename)
    config = {key: value for key, value in vars(cfg).items() if key != "beta3"}
    config.update(epochs=2, eval_freq=1, CHECKPOINT_FREQ=1, device="cpu", batch_size=64,
                  optimizer="adamwsevo", eval_valid=True, eval_test=False,
                  category_mapping_file=str(directory / "mapping.json"),
                  category_metadata_file=str(directory / "categories.jsonl"),
                  category_manifest_file=str(directory / "manifest.json"))
    config_file = path / "native_smoke.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-X", "utf8", str(root / "main_stair6_v1.py"),
                             "--config", str(config_file)], cwd=root, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    artifact = Path(config["artifact_dir"])
    records = [json.loads(line) for line in (artifact / "training_telemetry.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]
    recorded_manifest = json.loads((artifact / "manifest.json").read_text())
    assert recorded_manifest["category"]["category_active"]
    assert recorded_manifest["category"]["category_source"] == category_source
    assert recorded_manifest["selection"] == "validation NDCG@20"
    checkpoint = torch.load(artifact / "training_checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert (artifact / "best_model.pt").is_file()
    resumed = subprocess.run([sys.executable, "-X", "utf8", str(root / "main_stair6_v1.py"),
                              "--config", str(config_file), "--epochs", "3", "--resume-from",
                              str(artifact / "training_checkpoint.pt")], cwd=root, env=environment,
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert resumed.returncode == 0, resumed.stdout[-8000:] + resumed.stderr[-8000:]
    records = [json.loads(line) for line in (artifact / "training_telemetry.jsonl").read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    assert (artifact / "resume_manifest.json").is_file()
    checkpoint = torch.load(artifact / "training_checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 3


@pytest.mark.parametrize("name,value", [("category_strength", float("nan")),
    ("category_strength", 1.01), ("category_min_count", True), ("category_shrinkage", 0),
    ("category_probe_dim", 10_000), ("relation_gate", "metadata_proxy"), ("bsc_residual", 0.1),
    ("category_source", "auto"), ("latent_clusters", True), ("latent_projection_dim", 0),
    ("latent_iterations", 1000), ("latent_memory_budget_mib", float("nan"))])
def test_invalid_config_rejected(name, value):
    with pytest.raises(ValueError):
        validate_category_config({name: value})


def test_latent_zero_strength_skips_clustering_and_missing_metadata(setup, monkeypatch):
    monkeypatch.setattr("models.stair6_v1.build_latent_category_data",
                        lambda *args, **kwargs: pytest.fail("Unexpected clustering in bypass"))
    reference = make_model(setup, reference=True)
    model = make_model(setup, category_source="latent", category_strength=0,
                       category_mapping_file="", category_metadata_file="", category_manifest_file="")
    assert model.graph_fingerprint == reference.graph_fingerprint


def test_latent_audit_cli_generates_groups_without_external_metadata(setup):
    path = setup[2]
    root = Path(__file__).resolve().parents[1]
    output = path / "latent_audit.json"
    result = subprocess.run([sys.executable, str(root / "tools/audit_stair6_v1_latent.py"),
        "--dataset-dir", str(path), "--mfiles", "text.pkl,image.pkl", "--clusters", "6",
        "--projection-dim", "8", "--iterations", "10", "--cache-dir", str(path / "latent_cache"),
        "--output", str(output)], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record["status"] == "READY_FOR_GRAPH_DIAGNOSTICS"
    assert not record["audit"]["asin_mapping_used"] and record["audit"]["informative_items"] == 72


def test_preprocess_only_cli_option_is_registered(tmp_path):
    cfg = build_config(["--config", "configs/Amazon2014Baby_STAIR6_v1.yaml", "--device", "cpu",
                        "--artifact-dir", str(tmp_path / "preflight"), "--preprocess-only"])
    assert cfg.preprocess_only and cfg.category_source == "latent"
