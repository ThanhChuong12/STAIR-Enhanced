"""Dependency-free contracts for the capacity-only STAIR5-v8 experiment.

Dimension changes must not silently enable a new graph, objective, or optimizer.
This module deliberately does not import the research runtime or model package.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EMBEDDING_DIM = 256
EXPERIMENT_ID = "STAIR5-v8-WMSG-CSE-256D"
CONTRACT_VERSION = 1
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "Amazon2014Sports_STAIR5_v8_256D.yaml"

# These are architectural constants of the measured WMSG-core, not tuning defaults.
CAPACITY_CONTRACT = {
    "embedding_dim": EMBEDDING_DIM,
    "num_layers": 3,
    "v8_arm": "WMSG-core",
    "semantic_mode": "weighted",
    "edge_power": 2.0,
    "edge_floor": 0.05,
    "semantic_mix": 1.0,
    "eta": 0.1,
    "k_cf": 5,
    "c_min": 2,
    "t_shrinkage": 5.0,
    "lambda_nlgcl": 0.01,
    "nlgcl_tau": 0.2,
    "nlgcl_G": 1,
    "nlgcl_alpha": 0.5,
    "alignment": "off",
    "relation_gate": "off",
    "relation_strength": 0.0,
    "bsc_residual": 0.0,
    "lambda_dirichlet": 0.0,
    "knn_device": "cpu",
    "ranking": "full",
}


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_capacity_config(config: Any) -> None:
    """Reject accidental architectural changes before model construction.

Epoch budget, seed, loss chunk size, learning rate and decay remain explicit
runtime options so that smoke runs and equally budgeted tuning are possible.
"""
    errors = []
    for name, expected in CAPACITY_CONTRACT.items():
        actual = _get(config, name)
        if actual != expected:
            errors.append(f"{name}={actual!r}; required {expected!r}")
    neighbors = _get(config, "num_neighbors")
    if neighbors == "5-1":
        neighbors = [5, 1]
    if not isinstance(neighbors, (tuple, list)) or list(neighbors) != [5, 1]:
        errors.append("num_neighbors must be text/visual [5, 1]")
    if str(_get(config, "optimizer", "")).lower() != "adamwsevo":
        errors.append("optimizer must be adamwsevo")
    if _get(config, "eval_valid") is not True or _get(config, "eval_test") is not False:
        errors.append("select by validation only; eval_valid=true and eval_test=false")
    if str(_get(config, "which4best", "")).upper() != "NDCG@20":
        errors.append("which4best must be NDCG@20")
    for name in ("gamma", "lr"):
        value = _get(config, name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and positive")
    for name in ("lambda_cl", "lambda_qcl", "consensus_dose"):
        if _get(config, name, 0.0) != 0.0:
            errors.append(f"{name} is excluded from the capacity-only experiment")
    if errors:
        raise ValueError("Invalid V8-256D capacity experiment:\n  " + "\n  ".join(errors))


def with_default_config(arguments: Sequence[str]) -> list[str]:
    """Supply the Sports-256D config only when the caller did not select one."""
    arguments = list(arguments)
    if any(arg in ("--config", "-c") or arg.startswith("--config=") for arg in arguments):
        return arguments
    return ["--config", str(DEFAULT_CONFIG), *arguments]


def tensor_payload_ledger(n_users: int, n_items: int, dimension: int = EMBEDDING_DIM) -> dict:
    """Return analytical FP32 payload sizes; these are not measured GPU peaks."""
    for name, value in (("n_users", n_users), ("n_items", n_items), ("dimension", dimension)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    count = (n_users + n_items) * dimension
    return {
        "n_users": n_users,
        "n_items": n_items,
        "dimension": dimension,
        "embedding_parameters": count,
        "embedding_weights_bytes_fp32": 4 * count,
        "embedding_weights_grad_two_adam_moments_bytes_fp32": 16 * count,
        "one_joint_activation_bytes_fp32": 4 * count,
        "one_item_activation_bytes_fp32": 4 * n_items * dimension,
        "interpretation": "Analytical tensor payload only; not a peak, OOM guarantee or runtime forecast.",
    }
