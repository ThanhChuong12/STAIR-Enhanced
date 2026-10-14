"""Exact restricted CAN simplex solver and feature-only support calibration.

Solve d.T p + lambda ||p||^2 on the simplex. Probabilities select support;
they never replace native WMSG weights. All arrays are bounded CPU workspaces.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


class CANBudgetError(ValueError):
    """The requested mean support cannot be achieved within its declared tolerance."""


@dataclass(frozen=True)
class CANResult:
    probabilities: np.ndarray
    mask: np.ndarray
    counts: np.ndarray
    regularizer: float
    metadata: dict


def _costs(costs, memory_budget_mib=64.0):
    array = np.asarray(costs)
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
        raise ValueError("CAN costs must be a nonempty real [items, candidates] array")
    if not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("CAN workspace budget must be finite and positive")
    # Includes sorting, projection, thresholds, permutation and returned arrays.
    # Input features and sparse construction are explicitly outside this budget.
    estimate = 128 * array.size + 64 * len(array)
    if estimate > memory_budget_mib * 2**20:
        raise MemoryError(f"CAN workspace estimate {estimate} bytes exceeds its declared budget")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("CAN costs must be finite")
    return array


def _sorted(costs):
    order = np.argsort(costs, axis=1, kind="stable")
    distances = np.take_along_axis(costs, order, axis=1)
    distances = distances - distances[:, :1]
    rank = np.arange(1, costs.shape[1] + 1, dtype=np.float64)
    cumulative = np.cumsum(distances, axis=1)
    thresholds = 0.5 * (rank * distances - cumulative)
    # Roundoff can perturb repeated-cost thresholds at the last bit. Construct
    # the mathematically monotone sequence directly from nonnegative gaps.
    thresholds[:, 0] = 0.0
    if costs.shape[1] > 1:
        thresholds[:, 1:] = np.cumsum(0.5 * rank[:-1] * np.diff(distances, axis=1), axis=1)
    return order, distances, cumulative, thresholds


def kkt_residual(costs, probabilities, regularizer):
    """Maximum primal/stationarity residual, without silently repairing output."""
    costs = np.asarray(costs, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gradients = costs + 2 * regularizer * probabilities
    active = probabilities > 0
    multiplier = np.sum(np.where(active, gradients, 0.0), axis=1) / active.sum(axis=1)
    active_error = np.max(np.where(active, np.abs(gradients - multiplier[:, None]), 0.0))
    inactive_error = np.max(np.where(~active, np.maximum(multiplier[:, None] - gradients, 0.0), 0.0))
    return float(max(active_error, inactive_error, np.max(np.abs(probabilities.sum(1) - 1)),
                     max(0.0, -float(probabilities.min()))))


def _project(costs, order, distances, cumulative, regularizer):
    ranks = np.arange(1, costs.shape[1] + 1, dtype=np.float64)
    theta = (2 * regularizer + cumulative) / ranks
    counts = np.sum(distances < theta, axis=1, dtype=np.int64)
    if np.any(counts == 0):
        raise FloatingPointError("CAN active set is empty; regularizer is numerically too small")
    level = theta[np.arange(len(costs)), counts - 1]
    probabilities_sorted = np.maximum((level[:, None] - distances) / (2 * regularizer), 0.0)
    probabilities = np.empty_like(probabilities_sorted)
    np.put_along_axis(probabilities, order, probabilities_sorted, axis=1)
    mask = probabilities > 0
    if not np.array_equal(mask.sum(1), counts):
        raise FloatingPointError("CAN threshold/positive support disagreement")
    residual = kkt_residual(costs - costs.min(1, keepdims=True), probabilities, regularizer)
    if not math.isfinite(residual) or residual > 1e-10 * max(1.0, regularizer, float(np.ptp(costs))):
        raise FloatingPointError(f"CAN KKT validation failed: {residual}")
    return probabilities, mask, counts, residual


def solve_can(costs, regularizer, memory_budget_mib=64.0):
    """Exact positive-lambda solution, including ties and nonuniform full support."""
    if isinstance(regularizer, bool) or not math.isfinite(regularizer) or regularizer <= 0:
        raise ValueError("CAN regularizer must be finite and strictly positive")
    costs = _costs(costs, memory_budget_mib)
    order, distances, cumulative, _ = _sorted(costs)
    p, mask, counts, residual = _project(costs, order, distances, cumulative, float(regularizer))
    return CANResult(p, mask, counts, float(regularizer), {"kkt_residual": residual})


# Canonical alias for simplex solver contract
solve_can_simplex = solve_can


def calibrate_can(costs, target_mean=4.0, relative_tolerance=0.01, memory_budget_mib=64.0):
    """Choose a shared positive lambda from content-only support breakpoints.

    No accuracy-driven tuning or forced top-k fallback. Infeasible tied budgets
    raise CANBudgetError. Equal costs retain every candidate at positive lambda.
    """
    costs = _costs(costs, memory_budget_mib)
    n_items, cap = costs.shape
    if (not math.isfinite(target_mean) or not 1 <= target_mean <= cap
            or not math.isfinite(relative_tolerance) or not 0 <= relative_tolerance <= 0.01):
        raise ValueError("Invalid target support or relative budget tolerance")
    order, distances, cumulative, thresholds = _sorted(costs)
    ordered = np.sort(thresholds.reshape(-1))
    target = target_mean * n_items
    boundary = ordered[min(max(int(math.floor(target)), 0), len(ordered) - 1)]
    candidates = []
    lower = ordered[max(0, min(int(math.floor(target)) - 1, len(ordered) - 1))]
    if lower < boundary:
        candidates.append(float(lower + 0.5 * (boundary - lower)))
    # Use interiors of feasible intervals, rather than exact floating-point
    # breakpoints: the left interval excludes the whole tie block, the right
    # interval activates it. This avoids boundary cancellation in theta.
    if boundary > 0:
        before = np.searchsorted(ordered, boundary, side="left")
        previous = ordered[before - 1] if before else 0.0
        candidates.append(float(previous + 0.5 * (boundary - previous)))
    higher_pos = np.searchsorted(ordered, boundary, side="right")
    higher = ordered[higher_pos] if higher_pos < len(ordered) else boundary + max(1.0, abs(boundary))
    candidates.append(float(boundary + 0.5 * (higher - boundary)))
    candidates = sorted(set(value for value in candidates if math.isfinite(value) and value > 0))
    totals = [int(np.count_nonzero(thresholds < value)) for value in candidates]
    chosen = min(range(len(candidates)), key=lambda index: (abs(totals[index] - target), totals[index]))
    regularizer, achieved = candidates[chosen], totals[chosen] / n_items
    if abs(achieved - target_mean) > relative_tolerance * target_mean + 1e-12:
        raise CANBudgetError(f"NO-GO: target mean {target_mean} is infeasible; closest achievable mean {achieved}")
    p, mask, counts, residual = _project(costs, order, distances, cumulative, regularizer)
    if int(counts.sum()) != totals[chosen]:
        raise FloatingPointError("Calibrated support differs from threshold staircase")
    preceding = ordered[ordered < regularizer]
    succeeding = ordered[ordered >= regularizer]
    metadata = {
        "regularizer": regularizer, "target_mean_text_k": float(target_mean),
        "actual_mean_text_k": float(counts.mean()), "budget_error": float(counts.mean() - target_mean),
        "relative_budget_tolerance": float(relative_tolerance), "kkt_residual": residual,
        "regularizer_interval": [float(preceding[-1]) if preceding.size else 0.0,
                                 float(succeeding[0]) if succeeding.size else None],
        "k_distribution": {str(k): int(np.count_nonzero(counts == k)) for k in range(1, cap + 1)},
        "flat_cost_rows": int(np.count_nonzero(np.ptp(costs, axis=1) == 0)),
        "can_probability_effective_neighbors_mean": float(np.mean(1 / np.sum(p * p, axis=1))),
        "solver": "restricted_simplex_fp64_v1", "costs": "signed_raw_text_cosine_distance",
    }
    return CANResult(p, mask, counts, regularizer, metadata)


def prefix_mask(costs, counts):
    """Select top count_i from a frozen pool, with stable original-rank ties."""
    costs = np.asarray(costs, dtype=np.float64)
    counts = np.asarray(counts)
    if costs.ndim != 2 or counts.shape != (len(costs),) or counts.dtype.kind not in "iu":
        raise ValueError("Prefix counts must be an integer vector aligned with cost rows")
    if not np.isfinite(costs).all() or np.any(counts < 1) or np.any(counts > costs.shape[1]):
        raise ValueError("Invalid prefix costs/counts")
    order = np.argsort(costs, axis=1, kind="stable")
    mask = np.zeros(costs.shape, dtype=bool)
    np.put_along_axis(mask, order, np.arange(costs.shape[1])[None, :] < counts[:, None], axis=1)
    return mask


def shuffled_counts(counts, train_degrees, seed=2718):
    """Permute k within binary-train-degree log2 strata using a local RNG."""
    counts, degrees = np.asarray(counts), np.asarray(train_degrees)
    if (degrees.shape != counts.shape or degrees.dtype.kind not in "iu" or np.any(degrees < 0)
            or isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
        raise ValueError("Invalid train degree strata or placebo seed")
    bins = np.where(degrees == 0, 0, np.floor(np.log2(np.maximum(degrees, 1))).astype(np.int64) + 1)
    result = counts.copy()
    rng = np.random.default_rng(seed)
    for group in np.unique(bins):
        ids = np.flatnonzero(bins == group)
        result[ids] = rng.permutation(counts[ids])
    return result
