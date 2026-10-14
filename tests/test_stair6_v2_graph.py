"""Sparse normalization, native recovery, cache and resource contracts."""

import types

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models.stair5_v8_graph import (
    GraphStateV8, build_calibrated_graph_v8, build_wmsg_semantic_graph,
    normalize_graph_with_isolated_fallback, power_scale_similarities,
)
from models.stair5_v4_graph import scipy_to_tensor, tensor_to_scipy
from models.stair6_v2_graph import (
    build_stair6_v2_graph, interpolate_operators, semantic_from_weights, validate_candidates,
)


@pytest.fixture
def graph_setup(tmp_path):
    n = 32
    generator = torch.Generator().manual_seed(21)
    features = [torch.randn(n, 16, generator=generator) for _ in range(2)]
    edges = []
    for feat, cap in zip(features, (5, 1)):
        norm = torch.nn.functional.normalize(feat, dim=1)
        similarities = norm @ norm.T  # Tiny test fixture only; never production.
        similarities.fill_diagonal_(-10)
        edges.append(torch.stack((torch.arange(n).repeat_interleave(cap),
                                  similarities.topk(cap, dim=1).indices.reshape(-1))))
    train = torch.tensor([(u, i) for u in range(8) for i in range(n) if i % 2 == u % 2]).T
    cfg = types.SimpleNamespace(eta=0.1, k_cf=5, c_min=2, t_shrinkage=5.0,
        cf_block_size=8, cf_memory_budget_mib=16, graph_cache_dir=str(tmp_path), can_probe_dim=4)
    parent = build_calibrated_graph_v8(
        raw_semantic=scipy_to_tensor(sp.eye(n, dtype=np.float32)),
        baseline_normalized=scipy_to_tensor(sp.eye(n, dtype=np.float32)),
        train_edges=train, n_users=8, n_items=n, mfeats=features, knn_edges_list=edges,
        cache_dir=str(tmp_path), block_size=8, memory_budget_mib=16)
    return parent, features, edges, train, cfg, n


def build(setup, **updates):
    parent, features, edges, train, cfg, n = setup
    cfg = types.SimpleNamespace(**vars(cfg))
    for name, value in updates.items():
        setattr(cfg, name, value)
    beta = 0.1 + 0.9 * (torch.arange(8) / 8).pow(0.2)
    return build_stair6_v2_graph(parent, features, edges, train, 8, n, cfg, beta)


def test_sum_modalities_before_maximum_symmetry_and_native_magnitudes():
    text = torch.tensor([[0, 1], [1, 0]])
    image = torch.tensor([[0], [1]])
    weights = [np.array([0.2, 0.6], dtype=np.float32), np.array([0.5], dtype=np.float32)]
    raw = semantic_from_weights([text, image], weights, 3)
    assert raw[0, 1] == pytest.approx(0.7)
    assert raw[1, 0] == raw[0, 1]  # max(.2+.5, .6), not max(.2,.6)+.5.
    auxiliary = semantic_from_weights([text, image], weights, 3, np.array([False, True]))
    assert auxiliary[0, 1] == pytest.approx(0.6)
    np.testing.assert_allclose(power_scale_similarities(np.array([-1, 0, 0.5, 1])), [.05, .05, .2875, 1])


def test_contraction_isolated_fallback_and_convex_dose():
    raw = sp.csr_matrix(np.array([[0, .2, 1, 0], [.2, 0, .1, 0], [1, .1, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    pruned = raw.copy()
    pruned[0, 1] = pruned[1, 0] = 0
    pruned.eliminate_zeros()
    semantic = normalize_graph_with_isolated_fallback(raw)
    auxiliary = normalize_graph_with_isolated_fallback(pruned)
    cf = sp.csr_matrix(raw.shape, dtype=np.float32)
    parent, effective = interpolate_operators(semantic, auxiliary, cf, .1, 0)
    assert effective == 0
    assert parent[3, 3] == 1
    hard, _ = interpolate_operators(semantic, auxiliary, cf, .1, 1)
    mixed, _ = interpolate_operators(semantic, auxiliary, cf, .1, .25)
    np.testing.assert_allclose((mixed - parent).toarray(), .25 * (hard - parent).toarray(), atol=1e-7)
    assert np.max(np.abs(np.linalg.eigvalsh(mixed.toarray()))) <= 1 + 2e-7


@pytest.mark.parametrize("updates", [{"can_mode": "off"}, {"can_interpolation": 0}])
def test_bypass_is_original_tensor_without_solver(graph_setup, updates, monkeypatch):
    monkeypatch.setattr("models.stair6_v2_graph.calibrate_can", lambda *args: pytest.fail("Unexpected solver"))
    state = build(graph_setup, **updates)
    assert state.operator is graph_setup[0].operator
    assert not state.metadata["active"]


def test_active_native_parent_cache_identity_and_spectral_bound(graph_setup):
    first = build(graph_setup)
    second = build(graph_setup)
    assert first.contract == second.contract
    assert second.metadata["cache_hit"]
    assert first.metadata["actual_mean_text_k"] == 4
    assert first.metadata["final_support_equal_v8"]
    assert first.metadata["relative_operator_delta_vs_v8"] > 0
    final = tensor_to_scipy(first.operator)
    assert final.dtype == np.float32 and first.operator.layout == torch.sparse_csr
    assert np.max(np.abs(np.linalg.eigvalsh(final.toarray()))) <= 1 + 2e-6
    assert first.operator.grad_fn is None
    # Use a dense fixture only to verify the own-degree formula independently.
    raw, _ = build_wmsg_semantic_graph(graph_setup[1], graph_setup[2], graph_setup[-1])
    degree = np.asarray(raw.sum(1)).ravel()
    expected = raw.toarray() / np.sqrt(degree[:, None] * degree[None, :])
    np.testing.assert_allclose(normalize_graph_with_isolated_fallback(raw).toarray(), expected, atol=2e-7)


def test_corrupt_cache_warns_and_rebuilds(graph_setup):
    first = build(graph_setup)
    from pathlib import Path
    path = Path(graph_setup[4].graph_cache_dir) / f"can_{first.metadata['cache_key']}.pt"
    path.write_bytes(b"broken cache")
    with pytest.warns(RuntimeWarning, match="invalid CAN cache"):
        rebuilt = build(graph_setup)
    assert first.contract == rebuilt.contract
    assert not rebuilt.metadata["cache_hit"]


def test_fixed_shuffled_and_hard_arms_share_parent(graph_setup):
    adaptive = build(graph_setup)
    fixed = build(graph_setup, can_mode="fixed", can_fixed_text_k=4)
    shuffled = build(graph_setup, can_mode="shuffled")
    hard = build(graph_setup, can_interpolation=1)
    assert fixed.metadata["actual_mean_text_k"] == 4
    assert shuffled.metadata["k_distribution"] == adaptive.metadata["k_distribution"]
    assert adaptive.contract["baseline_graph_fingerprint"] == hard.contract["baseline_graph_fingerprint"]
    assert hard.metadata["final_nnz"] <= adaptive.metadata["final_nnz"]
    assert adaptive.metadata["relative_operator_delta_vs_v8"] == pytest.approx(
        .25 * hard.metadata["relative_operator_delta_vs_v8"], rel=1e-5)


def test_malformed_candidates_and_parent_fail_loudly(graph_setup):
    parent, features, edges, train, cfg, n = graph_setup
    malformed = edges[0].clone()
    malformed[:, 1] = malformed[:, 0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_candidates(malformed, n, 5)
    damaged = GraphStateV8(parent.operator * .5, parent.metadata)
    with pytest.raises(ValueError, match="fingerprint"):
        build_stair6_v2_graph(damaged, features, edges, train, 8, n, cfg, torch.full((8,), .5))


def test_empty_cf_retains_semantic_only_behavior(graph_setup):
    parent, features, edges, _, cfg, n = graph_setup
    empty = torch.empty((2, 0), dtype=torch.int64)
    raw, _ = build_wmsg_semantic_graph(features, edges, n)
    semantic = scipy_to_tensor(normalize_graph_with_isolated_fallback(raw))
    current = build_stair6_v2_graph(GraphStateV8(semantic, {}), features, edges, empty, 8, n,
                                   cfg, torch.full((8,), .5))
    assert current.metadata["effective_eta"] == 0
    assert current.metadata["cf_nnz"] == 0
