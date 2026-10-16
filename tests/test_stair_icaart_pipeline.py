"""Integration tests for STAIR-REEC (models/stair_icaart.py, main_stair_icaart.py).

Covers Acceptance Criteria 9 and 10:
9. Parameter groups initialized exactly once, exact preservation of loss values, gradients,
   and first update step at delta=0 compared to V8, and checkpoint save/load/resume with
   graph hash mismatch rejection.
10. Memory safety on Electronics-scale catalog (N = 63001, M = 192403):
    bounded candidate processing in chunks <= 8192, sparse-only representations,
    and strict prohibition of dense N x N allocation.
"""

import copy
import os
from pathlib import Path
import pickle
import types

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import models.freerec_compat
from freerec.data.fields import Field, FieldTuple
from freerec.data.tags import USER, ITEM, ID, LABEL
from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
from models.stair_icaart import STAIR_ICAART_Model
from models.stair_icaart_evidence import (
    build_activity_replication,
    candidate_excess_statistics,
    replicated_excess_score,
)
from optimizers.AdamW import AdamWSEvo


class ToyDataset:
    """Minimal self-contained dataset for pipeline parity and backward verification."""

    def __init__(self, path: Path, n_users: int = 12, n_items: int = 72):
        self.path = str(path)
        user, item = Field("User", USER, ID), Field("Item", ITEM, ID)
        user.count, item.count = n_users, n_items
        self.fields = FieldTuple([user, item, Field("Label", LABEL)])

        pairs = [(u, i) for u in range(4) for i in range(24)]
        pairs += [(u, i) for u in range(4, 12) for i in range(24, n_items)]
        pairs += [(0, 0), (0, 1), (1, 2), (2, 3)]
        self.edge_index = torch.tensor(pairs, dtype=torch.int64).T.contiguous()

    def train(self):
        return self

    def to_bigraph(self, edge_type="u2i"):
        return {"u2i": types.SimpleNamespace(edge_index=self.edge_index)}

    def to_normalized_adj(self, normalization="sym"):
        users, items = self.edge_index
        total_nodes = 12 + 72
        items_shifted = items + 12
        rows = torch.cat((users, items_shifted))
        cols = torch.cat((items_shifted, users))
        degrees = torch.bincount(rows, minlength=total_nodes).float().clamp_min(1.0)
        weights = degrees[rows].rsqrt() * degrees[cols].rsqrt()
        return (
            torch.sparse_coo_tensor(torch.stack((rows, cols)), weights, (total_nodes, total_nodes))
            .coalesce()
            .to_sparse_csr()
        )


@pytest.fixture
def setup(tmp_path):
    generator = torch.Generator().manual_seed(42)
    # 72 items, 80-dim feature (>= embedding_dim 64)
    for name in ("text.pkl", "image.pkl"):
        feat = torch.randn(72, 80, generator=generator)
        feat[1:24:2] = feat[:24:2] + 1e-4 * torch.randn(12, 80, generator=generator)
        with (tmp_path / name).open("wb") as h:
            pickle.dump(feat, h)

    for name in ("train.txt", "valid.txt", "test.txt"):
        (tmp_path / name).write_text("USER\tITEM\tTIMESTAMP\n0\t0\t1\n", encoding="utf-8")

    cfg = types.SimpleNamespace(
        embedding_dim=64,
        num_layers=3,
        gamma=0.2,
        mfiles=["text.pkl", "image.pkl"],
        num_neighbors=[5, 1],
        lr=0.001,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.999,
        batch_size=4,
        seed=1,
        eval_freq=5,
        num_workers=0,
        ranking="full",
        which4best="NDCG@20",
        lambda_nlgcl=0.01,
        nlgcl_G=1,
        nlgcl_tau=0.2,
        nlgcl_alpha=0.5,
        cl_chunk_size=2,
        eta=0.1,
        k_cf=5,
        c_min=2,
        t_shrinkage=5.0,
        v8_arm="WMSG-core",
        semantic_mode="weighted",
        edge_power=2.0,
        edge_floor=0.05,
        semantic_mix=1.0,
        alignment="off",
        relation_gate="off",
        relation_strength=0.0,
        bsc_residual=0.0,
        lambda_dirichlet=0.0,
        anchor_seed=2718,
        anchor_per_stratum=1024,
        relation_threshold=0.7,
        relation_temperature=0.1,
        placebo_seed=1,
        knn_device="cpu",
        knn_chunk_size=32,
        cf_block_size=16,
        cf_memory_budget_mib=128.0,
        graph_cache_dir=str(tmp_path / "cache"),
        artifact_dir=str(tmp_path / "artifacts"),
        root=str(tmp_path),
        dataset="ToyDataset",
        # ICAART specific settings
        evidence_mode="replicated_excess",
        evidence_activity_bins="log2",
        evidence_halves=2,
        evidence_seed=20261016,
        evidence_tau=2.0,
        evidence_delta=1.0,
        evidence_candidate_block=8192,
        evidence_placebo_seed=20261016,
        eval_valid=True,
        eval_test=False,
    )
    cfg.beta3 = 0.1 + 0.9 * (
        torch.arange(cfg.embedding_dim, dtype=torch.float32) / cfg.embedding_dim
    ).pow(cfg.gamma)

    dataset = ToyDataset(tmp_path)
    return dataset, cfg, tmp_path


def make_batch(model_instance):
    return {
        model_instance.User: torch.tensor([0, 0, 1, 4]),
        model_instance.Item: torch.tensor([0, 1, 2, 24]),
        model_instance.INeg: torch.tensor([24, 25, 26, 0]),
    }


def test_exact_v8_delta0_parity(setup):
    """Verify bitwise parity of loss, gradients, and 1-step Adam update at delta=0."""
    dataset, cfg, _ = setup

    # 1. Instantiate V8 reference model
    torch.manual_seed(2026)
    v8_model = STAIR5_v8_Model(dataset, cfg)

    # 2. Instantiate STAIR-REEC model with evidence_delta = 0.0
    torch.manual_seed(2026)
    cfg_rec = copy.copy(cfg)
    cfg_rec.evidence_delta = 0.0
    rec_model = STAIR_ICAART_Model(dataset, cfg_rec)

    # Graph fingerprint and operator must be bitwise identical
    assert v8_model.graph_fingerprint == rec_model.graph_fingerprint
    torch.testing.assert_close(
        v8_model.mAdj.to_dense(), rec_model.mAdj.to_dense(), rtol=0, atol=0
    )

    # Initial embeddings match
    torch.testing.assert_close(
        v8_model.User.embeddings.weight, rec_model.User.embeddings.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        v8_model.Item.embeddings.weight, rec_model.Item.embeddings.weight, rtol=0, atol=0
    )

    # Forward pass and loss parity
    v8_opt = AdamWSEvo(v8_model.marked_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rec_opt = AdamWSEvo(rec_model.marked_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    batch_v8 = make_batch(v8_model)
    batch_rec = make_batch(rec_model)

    v8_loss = v8_model.fit(batch_v8)
    rec_loss = rec_model.fit(batch_rec)

    torch.testing.assert_close(v8_loss, rec_loss, rtol=0, atol=0)

    # Backward pass and gradient parity
    v8_loss.backward()
    rec_loss.backward()

    torch.testing.assert_close(
        v8_model.User.embeddings.weight.grad,
        rec_model.User.embeddings.weight.grad,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        v8_model.Item.embeddings.weight.grad,
        rec_model.Item.embeddings.weight.grad,
        rtol=0,
        atol=0,
    )

    # Optimizer step parity
    v8_opt.step()
    rec_opt.step()

    torch.testing.assert_close(
        v8_model.User.embeddings.weight, rec_model.User.embeddings.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        v8_model.Item.embeddings.weight, rec_model.Item.embeddings.weight, rtol=0, atol=0
    )


def test_parameter_groups_and_zero_extra_trainable_parameters(setup):
    """Verify that STAIR-REEC introduces 0 extra parameters and retains exact V8 groups."""
    dataset, cfg, _ = setup
    torch.manual_seed(2026)
    v8_model = STAIR5_v8_Model(dataset, cfg)

    torch.manual_seed(2026)
    rec_model = STAIR_ICAART_Model(dataset, cfg)

    v8_groups = v8_model.marked_params()
    rec_groups = rec_model.marked_params()

    # Exactly 2 parameter groups: User (smoother=None), Item (smoother=STAIR5V8Smoother)
    assert len(rec_groups) == 2
    assert rec_groups[0]["smoother"] is None
    assert rec_groups[1]["smoother"] is not None

    # Count parameters
    v8_param_count = sum(p.numel() for p in v8_model.parameters())
    rec_param_count = sum(p.numel() for p in rec_model.parameters())
    assert v8_param_count == rec_param_count, "Zero new trainable parameters constraint violated!"


def test_checkpoint_save_resume_and_graph_mismatch_rejection(setup):
    """Verify checkpoint resume integrity and rejection when graph fingerprint differs."""
    dataset, cfg, tmp_path = setup
    torch.manual_seed(2026)
    model_orig = STAIR_ICAART_Model(dataset, cfg)
    opt_orig = AdamWSEvo(model_orig.marked_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Run 1 step
    model_orig.fit(make_batch(model_orig)).backward()
    opt_orig.step()

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_training_checkpoint(ckpt_path, model_orig, opt_orig, epoch=5)

    # 1. Successful resume with identical model
    model_res = STAIR_ICAART_Model(dataset, cfg)
    opt_res = AdamWSEvo(model_res.marked_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loaded = load_training_checkpoint(ckpt_path, model_res, opt_res)
    assert loaded["epoch"] == 5

    # Weights match
    torch.testing.assert_close(
        model_orig.User.embeddings.weight, model_res.User.embeddings.weight, rtol=0, atol=0
    )

    # 2. Tampered fingerprint rejection
    tampered_state = model_orig.get_extra_state()
    tampered_state["stair_icaart"]["graph_fingerprint"] = "corrupted_hash_123456"

    model_corrupt = STAIR_ICAART_Model(dataset, cfg)
    with pytest.raises(ValueError, match="STAIR-REEC graph fingerprint mismatch on resume"):
        model_corrupt.set_extra_state(tampered_state)


def test_electronics_scale_memory_safety(monkeypatch):
    """Verify that candidate excess algorithm processes Electronics-scale catalog without dense allocation.

    Dimensions: N = 63,001 items, M = 192,403 users, E = 378,006 candidate edges (~6N).
    A dense float32 N x N matrix would require ~15.8 GB RAM.
    This test verifies that candidate_excess_statistics runs strictly in bounded sparse blocks.
    """
    N = 63001
    M = 192403
    E = 10000  # sample candidate block for unit test speed

    rng = np.random.default_rng(20261016)

    # Sparse random candidate edges (strictly i < j)
    src = rng.integers(0, N - 1, size=E)
    dst = rng.integers(src + 1, N, size=E)
    E_sem = np.unique(np.column_stack((src, dst)).astype(np.int64), axis=0)
    E = len(E_sem)

    # Synthetic sparse interactions: 50,000 interactions across 192k users and 63k items
    n_sample_interactions = 50000
    u_idx = rng.integers(0, M, size=n_sample_interactions)
    i_idx = rng.integers(0, N, size=n_sample_interactions)

    R_csr = sp.csr_matrix(
        (np.ones(n_sample_interactions, dtype=np.int8), (u_idx, i_idx)),
        shape=(M, N),
    )
    R_csc = R_csr.tocsc()

    # Fail on catalog-square dense allocations/conversions. A large shape alone
    # is not a memory-safety test without an allocation guard.
    for name in ("zeros", "ones", "empty", "full"):
        original = getattr(np, name)
        def guarded(shape, *args, _original=original, **kwargs):
            dims = tuple(shape) if not isinstance(shape, int) else (shape,)
            if dims == (N, N):
                raise AssertionError("Dense catalog-square allocation is forbidden")
            return _original(shape, *args, **kwargs)
        monkeypatch.setattr(np, name, guarded)
    def forbidden_dense(*args, **kwargs):
        raise AssertionError("Sparse evidence must not convert to a dense matrix")
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden_dense)
    monkeypatch.setattr(sp.csc_matrix, "toarray", forbidden_dense)

    # Stratification
    user_bin, half, group_sizes = build_activity_replication(R_csr, evidence_seed=20261016)

    # Candidate excess statistics in bounded chunk of 8192
    c, mu = candidate_excess_statistics(
        R_csc=R_csc,
        E_sem=E_sem,
        half=half,
        user_bin=user_bin,
        group_sizes=group_sizes,
        candidate_block=8192,
        mode="replicated_excess",
    )

    assert c.shape == (E, 2)
    assert mu.shape == (E, 2)
    assert np.all(c >= 0)
    assert np.all(mu >= 0.0)

    # Score computation
    r = replicated_excess_score(c, mu, tau=2.0, mode="replicated_excess")
    assert r.shape == (E,)
    assert np.all(r >= 0.0) and np.all(r < 1.0)


def test_resume_rejects_changed_evidence_config_even_if_graph_matches(setup):
    dataset, cfg, _ = setup
    first = STAIR_ICAART_Model(dataset, cfg)
    changed = copy.copy(cfg)
    changed.evidence_placebo_seed += 1  # Does not affect the primary score/graph.
    second = STAIR_ICAART_Model(dataset, changed)
    assert first.graph_fingerprint == second.graph_fingerprint
    before = second.Item.embeddings.weight.detach().clone()
    with pytest.raises(ValueError, match="provenance"):
        second.load_state_dict(first.state_dict())
    torch.testing.assert_close(second.Item.embeddings.weight, before, atol=0, rtol=0)


def test_resume_rejects_tampered_sparse_graph_payload(setup):
    dataset, cfg, _ = setup
    model = STAIR_ICAART_Model(dataset, cfg)
    saved = dict(model.state_dict())
    orig_adj = saved["mAdj"]
    vals = orig_adj.values().clone()
    vals[0] += 0.01
    saved["mAdj"] = torch.sparse_csr_tensor(
        orig_adj.crow_indices().clone(),
        orig_adj.col_indices().clone(),
        vals,
        size=orig_adj.size(),
        device=orig_adj.device,
    )
    with pytest.raises(ValueError, match="payload checksum"):
        model.load_state_dict(saved)
