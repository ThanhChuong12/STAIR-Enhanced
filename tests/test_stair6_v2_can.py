"""Independent optimization and calibration contracts for the CAN selector."""

import numpy as np
import pytest
from scipy.optimize import minimize

from models.stair6_v2_can import (
    CANBudgetError, calibrate_can, kkt_residual, prefix_mask, shuffled_counts, solve_can,
)
from stair6_v2_config import validate_can_config


def test_exact_examples_full_support_and_uniform_ties():
    for costs in ([[0, 0.2, 0.8]], [[0, 0.2]]):
        result = solve_can(costs, 0.3)
        np.testing.assert_allclose(result.probabilities[0, :2], [2 / 3, 1 / 3], atol=1e-14)
        assert result.counts[0] == 2
    tied = solve_can(np.full((3, 5), 0.7), 0.2)
    np.testing.assert_allclose(tied.probabilities, 0.2, atol=1e-14)
    assert tied.mask.all()


def test_independent_slsqp_reference_and_translation_invariance():
    rng = np.random.default_rng(19)
    for cap in (5, 15):
        costs = rng.uniform(-2, 2, (20, cap))
        for strength in (0.05, 0.3, 2.0):
            result = solve_can(costs, strength)
            shifted = solve_can(costs + rng.uniform(-10, 10, (len(costs), 1)), strength)
            np.testing.assert_allclose(result.probabilities, shifted.probabilities, atol=2e-13)
            assert kkt_residual(costs, result.probabilities, strength) < 1e-12
            for row, actual in zip(costs, result.probabilities):
                reference = minimize(lambda p: row @ p + strength * (p @ p),
                    np.full(cap, 1 / cap), jac=lambda p: row + 2 * strength * p,
                    bounds=[(0, 1)] * cap, constraints={"type": "eq", "fun": lambda p: p.sum() - 1,
                    "jac": lambda p: np.ones(cap)}, method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 300})
                assert reference.success
                np.testing.assert_allclose(actual, reference.x, atol=3e-6)


def test_support_monotonicity_and_exact_breakpoint():
    costs = np.random.default_rng(4).uniform(0, 2, (90, 5))
    counts = np.stack([solve_can(costs, strength).counts for strength in np.geomspace(0.001, 10, 25)])
    assert (np.diff(counts, axis=0) >= 0).all()
    # At lambda=.1, the second coordinate has exactly zero probability.
    assert solve_can([[0, 0.2, 0.8]], 0.1).counts[0] == 1
    assert solve_can([[0, 0.2, 0.8]], 0.100001).counts[0] == 2


def test_feature_only_budget_and_original_position_mapping():
    costs = np.random.default_rng(51).uniform(0, 2, (101, 5))
    result = calibrate_can(costs)
    assert result.counts.sum() == 404
    assert np.array_equal(result.mask, prefix_mask(costs, result.counts))
    order = np.array([4, 1, 3, 0, 2])
    permuted = calibrate_can(costs[:, order])
    np.testing.assert_allclose(result.probabilities[:, order], permuted.probabilities, atol=1e-14)
    assert result.metadata["kkt_residual"] < 1e-12


def test_tied_budget_no_go_neutral_and_noninteger_target():
    with pytest.raises(CANBudgetError, match="NO-GO"):
        calibrate_can(np.ones((12, 5)), target_mean=4)
    full = calibrate_can(np.ones((12, 5)), target_mean=5)
    assert full.mask.all()
    result = calibrate_can(np.random.default_rng(9).uniform(0, 2, (100, 5)), target_mean=4.5)
    assert result.counts.sum() == 450


def test_placebo_is_local_rng_degree_stratified_distribution_preserving():
    counts = np.tile(np.arange(1, 6), 20)
    degrees = np.repeat([0, 1, 2, 8], 25).astype(np.int64)
    np.random.seed(913)
    state = np.random.get_state()
    shuffled = shuffled_counts(counts, degrees)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    for group in (0, 1, 2, 8):
        ids = degrees == group
        np.testing.assert_array_equal(np.sort(counts[ids]), np.sort(shuffled[ids]))
    np.testing.assert_array_equal(shuffled, shuffled_counts(counts, degrees))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_regularizer(value):
    with pytest.raises(ValueError):
        solve_can([[0, 1]], value)


def test_workspace_guard_and_nonfinite_costs():
    with pytest.raises(MemoryError):
        calibrate_can(np.ones((100, 5)), memory_budget_mib=0.001)
    with pytest.raises(ValueError):
        solve_can([[0, float("nan")]], 1)


def test_electronics_scale_workspace_is_items_times_five():
    """Exercise catalog-scale calibration without any catalog-square allocation."""
    costs = np.random.default_rng(47).uniform(0, 2, (63001, 5)).astype(np.float32)
    result = calibrate_can(costs, memory_budget_mib=64)
    assert result.probabilities.shape == (63001, 5)
    assert result.counts.sum() == 4 * 63001
    assert result.metadata["kkt_residual"] < 1e-12


@pytest.mark.parametrize("name,value", [
    ("can_mode", "expand"), ("can_interpolation", float("nan")), ("can_interpolation", 1.01),
    ("can_text_candidate_cap", 15), ("can_fixed_text_k", True), ("can_probe_dim", 65),
    ("can_memory_budget_mib", 0), ("can_solver_dtype", "float32"), ("can_target_mean_text_k", 0),
    ("alignment", "stratified_procrustes"), ("edge_power", 1.5), ("bsc_residual", 0.05),
    ("v8_arm", "WMSG-SAP"), ("num_neighbors", "4-1"),
])
def test_configuration_rejects_confounds(name, value):
    with pytest.raises(ValueError):
        validate_can_config({name: value})
