"""Numerical, cache and missing-content contracts for latent grouping."""
import random

import numpy as np
import pytest
import torch

from models.stair6_v1_latent_categories import build_latent_category_data, _project_views
from models.stair6_v1_categories import binary_user_category_incidence, category_association_scores
from models.stair6_v1_categories import requested_category_pairs, pool_category_scores


def inputs():
    generator = np.random.default_rng(29)
    labels = np.repeat(np.arange(4), 12)
    features = []
    for dimension in (16, 24):
        centers = generator.normal(size=(4, dimension)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        features.append((centers[labels] + .02 * generator.normal(size=(48, dimension))).astype(np.float32))
    return features


def build(features=None, **options):
    values = dict(dataset="Toy", feature_manifest=[{"file": "text.pkl", "sha256": "a" * 64},
                  {"file": "image.pkl", "sha256": "b" * 64}], split_manifest={"train.txt": "c" * 64},
                  clusters=4, projection_dim=12, iterations=10, chunk_size=7, seed=2718)
    values.update(options)
    return build_latent_category_data(inputs() if features is None else features, **values)


def test_hard_sparse_assignment_no_fake_asins_and_unit_centers():
    state = build()
    assert state.data.memberships.shape == (48, 4)
    assert state.data.memberships.nnz == 48
    np.testing.assert_array_equal(state.data.memberships.sum(axis=1), np.ones((48, 1)))
    np.testing.assert_allclose(np.linalg.norm(state.centers, axis=1), 1, atol=1e-6)
    assert state.data.item_asins == ()
    assert not state.data.audit["ground_truth_taxonomy"]
    assert state.data.audit["occupied_clusters"] == 4


def test_reproducibility_and_no_training_rng_changes():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first, second = build(), build()
    np.testing.assert_array_equal(first.assignments, second.assignments)
    np.testing.assert_array_equal(first.centers, second.centers)
    assert first.data.metadata_fingerprint == second.data.metadata_fingerprint
    assert random.getstate() == python_state
    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, np.random.get_state()))
    assert torch.equal(torch_state, torch.get_rng_state())


def test_missing_modalities_are_neutral_without_claiming_full_coverage():
    features = inputs()
    features[0][0] = 0
    features[1][0] = 0
    features[0][1] = 0
    state = build(features)
    assert state.assignments[0] == -1 and state.assignments[1] >= 0
    assert state.data.memberships[0].nnz == 0
    assert state.data.audit["informative_items"] == 47


def test_projection_concatenation_weighted_cosine_and_detached_cpu_input():
    features = inputs()
    fused, valid = _project_views(features, 12, 7, 2718)
    assert valid.all()
    left, right = fused[:, :12] / np.sqrt(5 / 6), fused[:, 12:] / np.sqrt(1 / 6)
    expected = (5 / 6) * np.dot(left[0], left[13]) + (1 / 6) * np.dot(right[0], right[13])
    np.testing.assert_allclose(np.dot(fused[0], fused[13]), expected, atol=1e-6)
    tensors = [torch.from_numpy(f).requires_grad_() for f in features]
    build(tensors)
    assert all(t.grad is None for t in tensors)


def test_cache_roundtrip_corruption_recovery_and_input_binding(tmp_path):
    first, cached = build(cache_dir=tmp_path), build(cache_dir=tmp_path)
    assert not first.metadata["cache_hit"] and cached.metadata["cache_hit"]
    assert first.data.metadata_fingerprint == cached.data.metadata_fingerprint
    path = next(tmp_path.glob("latent_*.pt"))
    path.write_bytes(b"corrupt payload")
    with pytest.warns(RuntimeWarning, match="invalid latent cache"):
        restored = build(cache_dir=tmp_path)
    np.testing.assert_array_equal(first.assignments, restored.assignments)
    features = inputs()
    features[0][0, 0] += .01
    changed = build(features, cache_dir=tmp_path)
    assert first.metadata["cache_key"] != changed.metadata["cache_key"]
    assert not changed.metadata["cache_hit"]


@pytest.mark.parametrize("change", [{"clusters": True}, {"clusters": 1}, {"clusters": 100},
    {"projection_dim": 0}, {"iterations": 0}, {"chunk_size": 0}, {"seed": -1},
    {"memory_budget_mib": .00001}])
def test_invalid_size_and_workspace_rejected(change):
    with pytest.raises((ValueError, MemoryError)):
        build(**change)


def test_nonfinite_invalid_and_degenerate_features_fail_explicitly():
    features = inputs()
    features[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        build(features)
    with pytest.raises(ValueError, match="informative items"):
        build([np.zeros_like(f) for f in inputs()])
    with pytest.raises(ValueError, match="collapsed"):
        build([np.ones_like(f) for f in inputs()])


def test_dense_soft_membership_is_not_a_drop_in_category_probability():
    import scipy.sparse as sp
    membership = sp.csr_matrix(np.full((4, 3), 1 / 3, dtype=np.float32))
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    incidence = binary_user_category_incidence(edges, 4, membership)
    evidence = category_association_scores(incidence, np.array([1, 2, 5]), min_count=2)
    assert evidence.metadata["population_users"] == 4
    np.testing.assert_array_equal(evidence.scores, 0)


def test_group_id_permutation_is_an_invariance_not_a_placebo():
    import scipy.sparse as sp
    state = build()
    membership = state.data.memberships
    permutation = np.array([2, 0, 3, 1])
    renamed = membership[:, permutation]
    raw = sp.csr_matrix(np.ones((48, 48), dtype=np.float32) - np.eye(48, dtype=np.float32))
    pairs = [(u, i) for u in range(6) for i, group in enumerate(state.assignments)
             if group in ((0, 1) if u < 3 else (2, 3))]
    edges = torch.tensor(pairs).T.contiguous()
    outputs = []
    for current in (membership, renamed):
        evidence = category_association_scores(binary_user_category_incidence(edges, 6, current),
            requested_category_pairs(raw, current))
        outputs.append(pool_category_scores(raw, current, evidence))
    assert np.any(outputs[0] > 0)
    np.testing.assert_array_equal(*outputs)
