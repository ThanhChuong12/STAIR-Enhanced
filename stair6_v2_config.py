"""Pure configuration contract for STAIR6-v2 / AKP-WMSG-CSE.

Candidate caps concern the auxiliary graph only. Native MI always uses 5:1.
No parser, dataset, accelerator or filesystem side effects occur on import.
"""

import math
from numbers import Integral, Real


CAN_OPTIONS = (
    ("can-mode", str, "prune"),
    ("can-text-candidate-cap", int, 5),
    ("can-visual-candidate-cap", int, 1),
    ("can-target-mean-text-k", float, 4.0),
    ("can-budget-relative-tolerance", float, 0.01),
    ("can-interpolation", float, 0.25),
    ("can-fixed-text-k", int, 4),
    ("can-solver-dtype", str, "float64"),
    ("can-memory-budget-mib", float, 64.0),
    ("can-probe-dim", int, 8),
    ("can-probe-seed", int, 2718),
    ("can-placebo-seed", int, 2718),
)
CAN_NAMES = tuple(flag.replace("-", "_") for flag, _, _ in CAN_OPTIONS)


def _read(config, name, default):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def can_enabled(config):
    """Return whether any CAN work is necessary; zero dose is an exact bypass."""
    return _read(config, "can_mode", "prune") != "off" and _read(config, "can_interpolation", 0.25) > 0


def validate_can_config(config):
    """Return validated settings, rejecting changes to the canonical V8 parent."""
    values = {flag.replace("-", "_"): _read(config, flag.replace("-", "_"), default)
              for flag, _, default in CAN_OPTIONS}
    if values["can_mode"] not in ("off", "prune", "fixed", "shuffled"):
        raise ValueError("can_mode must be off, prune, fixed or shuffled; expansion is a separate study")
    for name in ("can_text_candidate_cap", "can_visual_candidate_cap", "can_fixed_text_k",
                 "can_probe_dim", "can_probe_seed", "can_placebo_seed"):
        if isinstance(values[name], bool) or not isinstance(values[name], Integral):
            raise ValueError(f"{name} must be an integer")
        values[name] = int(values[name])
    if (values["can_text_candidate_cap"], values["can_visual_candidate_cap"]) != (5, 1):
        raise ValueError("AKP uses the native text5/visual1 candidate pool, independently of MI")
    if not 1 <= values["can_fixed_text_k"] <= 5:
        raise ValueError("can_fixed_text_k must be between 1 and 5")
    if not 0 <= values["can_probe_dim"] <= min(64, _read(config, "embedding_dim", 64)):
        raise ValueError("can_probe_dim must be between 0 and min(64, embedding_dim)")
    if min(values["can_probe_seed"], values["can_placebo_seed"]) < 0:
        raise ValueError("CAN seeds must be nonnegative")
    for name in ("can_target_mean_text_k", "can_budget_relative_tolerance", "can_interpolation",
                 "can_memory_budget_mib"):
        if isinstance(values[name], bool) or not isinstance(values[name], Real) or not math.isfinite(values[name]):
            raise ValueError(f"{name} must be a finite real number")
        values[name] = float(values[name])
    if not 1 <= values["can_target_mean_text_k"] <= 5:
        raise ValueError("can_target_mean_text_k must lie in [1, 5]")
    if not 0 <= values["can_budget_relative_tolerance"] <= 0.01:
        raise ValueError("CAN relative budget tolerance must lie in [0, 0.01]")
    if not 0 <= values["can_interpolation"] <= 1:
        raise ValueError("can_interpolation must lie in [0, 1]")
    if values["can_memory_budget_mib"] <= 0 or values["can_solver_dtype"] != "float64":
        raise ValueError("Require a positive CAN workspace budget and float64 solver")
    required = {"v8_arm": "WMSG-core", "semantic_mode": "weighted", "edge_power": 2.0,
                "edge_floor": 0.05, "semantic_mix": 1.0, "alignment": "off", "relation_gate": "off",
                "relation_strength": 0.0, "bsc_residual": 0.0, "lambda_dirichlet": 0.0,
                "knn_device": "cpu"}
    for name, expected in required.items():
        if _read(config, name, expected) != expected:
            raise ValueError(f"STAIR6-v2 requires {name}={expected!r}; avoid parent-architecture confounds")
    neighbors = _read(config, "num_neighbors", [5, 1])
    if neighbors not in ("5-1", [5, 1], (5, 1)):
        raise ValueError("Keep native num_neighbors=5-1 for MI, not adaptive counts")
    return values
