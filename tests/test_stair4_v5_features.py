"""Independent fixed-data and sampling contracts for STAIR-RAM v5."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from models.stair4_v5_features import TrainHistory, build_train_csr, prepare_modality_features
from models.stair4_v5_sampling import UniformTrainUnseenSampler


@pytest.fixture
def graph():
    # Duplicate edge (0, 1), singleton user 1, empty user 2, saturated user 3.
    return build_train_csr(
        np.array([[0, 0, 0, 1, 3, 3, 3, 3, 3], [0, 1, 1, 2, 0, 1, 2, 3, 4]]), 4, 5,
    )


@pytest.mark.parametrize("algorithm", ["exact", "randomized"])
def test_pca_missing_rank_sign_and_isolated_rng(algorithm):
    raw = torch.tensor([[1., 2., 3.], [2., -1., 1.], [-2., 2., 1.],
                        [0., 0., 0.], [float("nan"), 2., 3.]])
    rng = torch.get_rng_state().clone()
    result = prepare_modality_features(raw, output_dim=8, algorithm=algorithm, seed=87)
    assert torch.equal(rng, torch.get_rng_state())
    assert result.values.shape == (5, 8)
    assert result.metadata["actual_rank"] == 2
    assert result.metadata["num_missing"] == 2
    assert not result.valid_mask[3:].any()
    assert not result.values[3:].count_nonzero()
    assert not result.components[:, 2:].count_nonzero()
    assert torch.allclose(result.values[:3].norm(dim=1), torch.ones(3), atol=1e-6)
    components = result.components[:, :2]
    assert (components[components.abs().argmax(dim=0), torch.arange(2)] > 0).all()
    assert torch.allclose(result.explained_variance_ratio.sum(), torch.tensor(1.), atol=1e-5)
    repeat = prepare_modality_features(raw, output_dim=8, algorithm=algorithm, seed=87)
    assert torch.equal(result.values, repeat.values)
    assert all(not value.requires_grad for value in result.state_dict().values() if isinstance(value, torch.Tensor))


def test_pca_cache_rejects_stale_and_corrupt_data(tmp_path):
    raw = torch.randn(12, 6, generator=torch.Generator().manual_seed(42))
    cache = tmp_path / "features.pt"
    original = prepare_modality_features(raw, output_dim=4, cache_path=cache)
    restored = prepare_modality_features(raw, output_dim=4, cache_path=cache)
    assert torch.equal(original.values, restored.values)
    changed = raw.clone()
    changed[0, 0] += .1
    with pytest.raises(ValueError, match="identity"):
        prepare_modality_features(changed, output_dim=4, cache_path=cache)
    with pytest.raises(ValueError, match="identity"):
        prepare_modality_features(raw, output_dim=3, cache_path=cache)
    payload = torch.load(cache, weights_only=True)
    payload["values"][0, 0] += .1
    torch.save(payload, cache)
    with pytest.raises(ValueError, match="integrity"):
        prepare_modality_features(raw, output_dim=4, cache_path=cache)


def test_pca_invalid_and_constant_features():
    with pytest.raises(ValueError, match="at least two"):
        prepare_modality_features(torch.zeros(6, 3))
    with pytest.raises(ValueError, match="internal item"):
        prepare_modality_features(torch.ones(6, 3), item_ids=np.arange(6)[::-1])
    constant = prepare_modality_features(torch.ones(6, 3), output_dim=5)
    assert constant.metadata["actual_rank"] == 0
    assert not constant.values.count_nonzero()
    assert torch.isfinite(constant.explained_variance_ratio).all()


def test_history_loo_preserves_duplicate_rows_and_uses_binary_degrees(graph):
    features = torch.tensor([[1., 2.], [3., 5.], [0., 0.], [7., 8.], [11., 13.]])
    history = TrainHistory(graph, {"text": features, "image": features * 2})
    assert history.degrees.tolist() == [2, 1, 0, 5]
    result = history.loo(torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 0, 2]))
    assert torch.equal(result["text"], torch.stack([features[1], features[0], features[1], torch.zeros(2)]))
    assert torch.equal(result["image"], result["text"] * 2)
    full = history.full(torch.tensor([0, 1, 2, 3]))["text"]
    assert torch.allclose(full[0], features[:2].mean(dim=0))
    assert not full[1:3].count_nonzero()
    assert torch.allclose(full[3], features.mean(dim=0))  # Missing row counts in degree.
    assert history.loo(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))["text"].shape == (0, 2)
    with pytest.raises(ValueError, match="not a member"):
        history.loo([0], [4])
    with pytest.raises(ValueError, match="integer"):
        history.loo([0.5], [0])


def test_history_rejects_dynamic_features_and_invalid_edges(graph):
    with pytest.raises(ValueError, match="without autograd"):
        TrainHistory(graph, {"text": torch.ones(5, 4, requires_grad=True)})
    with pytest.raises(ValueError, match="outside"):
        build_train_csr(np.array([[0], [5]]), 4, 5)
    with pytest.raises(ValueError, match="integer"):
        build_train_csr(np.array([[0.], [1.]]), 4, 5)
    with pytest.raises(ValueError, match="nonnegative"):
        TrainHistory(sp.csr_matrix(np.array([[1., -1.]])), {"text": torch.ones(2, 3)})


def test_sampler_unique_padding_global_rng_and_resume(graph):
    sampler = UniformTrainUnseenSampler(graph, seed=19)
    rng = torch.get_rng_state().clone()
    users = torch.tensor([0, 1, 2, 3, 0])
    ids, mask = sampler.sample(users, 7)
    assert torch.equal(rng, torch.get_rng_state())
    assert mask.sum(dim=1).tolist() == [3, 4, 5, 0, 3]
    for row, user in enumerate(users.tolist()):
        valid = ids[row][mask[row]].tolist()
        assert len(valid) == len(set(valid))
        assert not set(valid).intersection(graph.indices[graph.indptr[user]:graph.indptr[user + 1]])
    assert not ids[~mask].count_nonzero()
    state = sampler.state_dict()
    expected = sampler.sample(users, 2)
    restored = UniformTrainUnseenSampler(graph, seed=1)
    restored.load_state_dict(state)
    actual = restored.sample(users, 2)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    changed = UniformTrainUnseenSampler(sp.csr_matrix((4, 5)), seed=1)
    with pytest.raises(ValueError, match="does not match"):
        changed.load_state_dict(state)


def test_sparse_rejection_and_enumeration_are_uniform():
    # Sparse path (100 items, 5 seen) and dense path (7 items, 3 seen).
    for num_items, seen, k in [(100, list(range(5)), 2), (7, [0, 2, 4], 2)]:
        graph = build_train_csr(np.array([[0] * len(seen), seen]), 1, num_items)
        sampler = UniformTrainUnseenSampler(graph, seed=813)
        ids, mask = sampler.sample(torch.zeros(10_000, dtype=torch.long), k)
        assert mask.all()
        assert (ids[:, 0] != ids[:, 1]).all()
        counts = torch.bincount(ids.flatten(), minlength=num_items).double()
        assert not counts[seen].count_nonzero()
        unseen = [i for i in range(num_items) if i not in seen]
        expected = 10_000 * k / len(unseen)
        # Loose deterministic smoke test, not a general proof of uniformity.
        assert torch.max(torch.abs(counts[unseen] - expected)) < 6 * expected ** .5


def test_sampler_empty_and_invalid_requests(graph):
    sampler = UniformTrainUnseenSampler(graph)
    ids, mask = sampler.sample(torch.empty(0, dtype=torch.long), 3)
    assert ids.shape == mask.shape == (0, 3)
    with pytest.raises(ValueError, match="positive integer"):
        sampler.sample([0], 0)
    with pytest.raises(ValueError, match="outside"):
        sampler.sample([4], 1)
