"""Algebra, gradient, and optimizer contracts for the actual v5 head."""
from __future__ import annotations

import copy

import pytest
import torch
from torch.nn import functional as F

from models.stair4_v5_heads import ResidualHead
from models.stair4_v5_objectives import masked_candidate_cross_entropy


def fixture(dtype=torch.float64, gate_mode="personalized"):
    torch.manual_seed(27)
    head = ResidualHead(6, {"text": 7, "image": 8}, 4, eta=0.7,
                        max_log_degree=3.0, gate_mode=gate_mode).to(dtype)
    users = torch.randn(5, 6, dtype=dtype, requires_grad=True)
    items = torch.randn(9, 6, dtype=dtype, requires_grad=True)
    profiles = {name: torch.randn(5, dim, dtype=dtype, requires_grad=True)
                for name, dim in head.feature_dims.items()}
    features = {name: torch.randn(9, dim, dtype=dtype, requires_grad=True)
                for name, dim in head.feature_dims.items()}
    degrees = torch.tensor([1, 2, 3, 4, 5])
    return head, users, items, profiles, features, degrees


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("amplitude_sign", [1.0, -1.0])
def test_separable_scores_bounds_and_duplicate_candidates(dtype, amplitude_sign):
    head, users, items, profiles, features, degrees = fixture(dtype)
    with torch.no_grad():
        head.theta.fill_(amplitude_sign * 1.7)
        head.gate.weight.normal_(std=0.2)
    candidates = torch.tensor([[1, 1, 2], [3, 2, 7], [0, 3, 5], [2, 4, 2], [1, 8, 6]])
    pool = head.score(users, items[candidates], profiles,
                      {name: value[candidates] for name, value in features.items()}, degrees)
    user_vectors, item_vectors = head.evaluation_vectors(users, items, profiles, features, degrees)
    full = user_vectors @ item_vectors.T
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(pool, full.gather(1, candidates), rtol=tolerance, atol=tolerance)
    assert pool[0, 0].item() == pool[0, 1].item()
    baseline = (users[:, None] * items[candidates]).sum(-1)
    assert (pool - baseline).abs().max() <= head.eta + tolerance
    # A residual cannot reverse a base margin strictly greater than 2 * eta.
    differences = (pool - baseline)[:, :1] - (pool - baseline)[:, 1:]
    assert differences.abs().max() <= 2 * head.eta + tolerance


def test_gate_zero_uses_original_width_and_skips_invalid_feature_inputs():
    head, users, items, _, _, degrees = fixture()
    user_vectors, item_vectors = head.evaluation_vectors(users, items, {}, {}, degrees, enabled=False)
    assert user_vectors.data_ptr() == users.data_ptr()
    assert item_vectors.data_ptr() == items.data_ptr()
    assert torch.equal(user_vectors @ item_vectors.T, users @ items.T)
    pair_scores = head.score(users, items[:5], {}, {}, degrees, enabled=False)
    assert torch.equal(pair_scores, (users * items[:5]).sum(-1))


def test_initialization_copies_values_without_aliasing_and_keeps_branch_open():
    head, *_ = fixture()
    assert head.amplitude.item() == pytest.approx(head.eta.item() * 0.05, rel=1e-7)
    for name in head.modalities:
        key = head.item_projectors[name].weight
        profile = head.profile_projectors[name].weight
        assert torch.equal(key, profile)
        assert key.data_ptr() != profile.data_ptr()
        torch.testing.assert_close(key @ key.T, torch.eye(4, dtype=key.dtype), atol=3e-7, rtol=3e-7)
    assert not head.gate.weight.count_nonzero()
    assert not head.gate.bias.count_nonzero()


def test_multiple_steps_train_every_head_parameter_without_backbone_gradient():
    head, users, items, profiles, features, degrees = fixture()
    optimizer = torch.optim.AdamW(head.parameter_groups())
    initial = {name: parameter.detach().clone() for name, parameter in head.named_parameters()}
    candidates = torch.tensor([[0, 1, 2], [1, 3, 5], [3, 2, 7], [2, 5, 8], [4, 6, 1]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        scores = head.score(users, items[candidates], profiles,
                            {name: value[candidates] for name, value in features.items()}, degrees)
        loss = masked_candidate_cross_entropy(scores)
        loss.backward()
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.norm() > 0, name
        assert users.grad is None and items.grad is None
        assert all(value.grad is None for value in [*profiles.values(), *features.values()])
        diagnostics = head.diagnostics(users, degrees)
        assert diagnostics["gradient_norm/projection"] > 0
        optimizer.step()
    for name, parameter in head.named_parameters():
        assert not torch.equal(initial[name], parameter), name


@pytest.mark.parametrize("gate_mode", ["personalized", "global", "fixed"])
def test_groups_are_disjoint_complete_and_gate_is_a_simplex(gate_mode):
    head, users, _, _, _, degrees = fixture(gate_mode=gate_mode)
    groups = head.parameter_groups()
    identities = [id(p) for group in groups for p in group["params"]]
    assert len(identities) == len(set(identities))
    assert set(identities) == {id(p) for p in head.parameters()}
    assert all(group["weight_decay"] == 0 for group in groups if group["name"] != "projection")
    weights = head.gate_weights(users, degrees)
    torch.testing.assert_close(weights.sum(-1), torch.ones(5, dtype=users.dtype))
    torch.testing.assert_close(weights, torch.full((5, 2), 0.5, dtype=users.dtype))


def test_zero_modality_features_produce_zero_keys_and_finite_scores():
    head, users, items, profiles, features, degrees = fixture()
    features = {name: torch.zeros_like(value) for name, value in features.items()}
    keys = head.encode_items(features)
    assert all(not value.count_nonzero() for value in keys.values())
    score = head.score(users, items[:5], profiles, {name: value[:5] for name, value in features.items()}, degrees)
    torch.testing.assert_close(score, (users * items[:5]).sum(-1), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_k_one_temperature_one_matches_bpr_value_and_gradient(dtype):
    score = torch.tensor([[5.0, -2.0], [-4.0, 5.0], [0.1, 0.2]], dtype=dtype, requires_grad=True)
    ce = masked_candidate_cross_entropy(score)
    bpr = F.softplus(score[:, 1] - score[:, 0]).mean()
    torch.testing.assert_close(ce, bpr)
    grad_ce = torch.autograd.grad(ce, score, retain_graph=True)[0]
    grad_bpr = torch.autograd.grad(bpr, score)[0]
    torch.testing.assert_close(grad_ce, grad_bpr)


def test_masked_loss_excludes_empty_rows_and_ignores_invalid_values():
    scores = torch.tensor([[1.0, 2.0, 900.0], [1.0, float("nan"), float("inf")],
                           [4.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False], [False, False], [True, True]])
    loss = masked_candidate_cross_entropy(scores, mask)
    expected = (F.softplus(torch.tensor(1.0, dtype=torch.float64))
                + torch.logsumexp(torch.tensor([0., -2., -1.], dtype=torch.float64), dim=0)) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.equal(scores.grad[~torch.cat((mask.any(-1, keepdim=True), mask), dim=1)],
                       torch.zeros(4, dtype=scores.dtype))


@pytest.mark.parametrize("shape", [(4, 3), (4, 1), (0, 3)])
def test_all_empty_batches_return_differentiable_zero(shape):
    scores = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros((shape[0], shape[1] - 1), dtype=torch.bool)
    loss = masked_candidate_cross_entropy(scores, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_microbatch_accumulation_matches_full_gradient_with_ineligible_rows():
    torch.manual_seed(7)
    score = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.rand(7, 4) > 0.4
    mask[1] = False
    full_grad = torch.autograd.grad(masked_candidate_cross_entropy(score, mask), score)[0]
    micro_loss = sum(masked_candidate_cross_entropy(score[start:start+2], mask[start:start+2], reduction="sum")
                     for start in range(0, 7, 2)) / mask.any(-1).sum()
    micro_grad = torch.autograd.grad(micro_loss, score)[0]
    torch.testing.assert_close(full_grad, micro_grad)


def test_state_dict_roundtrip_preserves_train_and_evaluation_scores():
    head, users, items, profiles, features, degrees = fixture()
    with torch.no_grad():
        head.theta.fill_(-0.3)
        head.gate.weight.normal_()
    clone = copy.deepcopy(head)
    clone.reset_parameters()
    clone.load_state_dict(head.state_dict())
    original = head.evaluation_vectors(users, items, profiles, features, degrees)
    loaded = clone.evaluation_vectors(users, items, profiles, features, degrees)
    assert all(torch.equal(a, b) for a, b in zip(original, loaded))

