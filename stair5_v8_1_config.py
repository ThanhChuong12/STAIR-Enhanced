"""Dependency-free configuration contracts for the isolated V8.1 experiment."""

import math
from collections.abc import Mapping


CONSENSUS_OPTIONS = (
    ("consensus-mode", str, "soft"),
    ("consensus-gate", str, "learned"),
    ("consensus-rho", float, 0.2),
    ("consensus-warmup", int, 30),
    ("consensus-ramp", int, 30),
    ("consensus-floor", float, 0.05),
    ("consensus-text-weight", float, 5.0 / 6.0),
    ("consensus-hidden", int, 8),
    ("consensus-eps", float, 1e-8),
    ("consensus-seed", int, 2718),
    ("consensus-placebo-seed", int, 1),
    ("consensus-memory-budget-mib", float, 64.0),
    ("consensus-sparse-layout", str, "auto"),
    ("consensus-diagnostic-interval", int, 25),
    ("lr-aux-scale", float, 0.1),
)
CONSENSUS_NAMES = tuple(flag.replace("-", "_") for flag, _, _ in CONSENSUS_OPTIONS)
DEFAULTS = {flag.replace("-", "_"): value for flag, _, value in CONSENSUS_OPTIONS}


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_consensus_config(config):
    """Reject invalid doses and unintended backbone changes before initialization.

    CSE eta may be zero for the pre-registered redundancy ablation; the
    auxiliary graph still uses the same raw CF substrate in that case.
    """
    values = {name: _get(config, name, default) for name, default in DEFAULTS.items()}
    choices = {
        "consensus_mode": ("off", "soft", "cf-only", "placebo"),
        "consensus_gate": ("learned", "fixed"),
        "consensus_sparse_layout": ("auto", "csr", "coo"),
    }
    for name, allowed in choices.items():
        if values[name] not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    for name in ("consensus_warmup", "consensus_ramp", "consensus_seed", "consensus_placebo_seed",
                 "consensus_diagnostic_interval", "consensus_hidden"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if values["consensus_hidden"] != 8:
        raise ValueError("The V8.1 scalar gate uses hidden size 8 (41 parameters)")
    for name in ("consensus_rho", "consensus_floor", "consensus_text_weight", "consensus_eps",
                 "consensus_memory_budget_mib", "lr_aux_scale"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= values["consensus_rho"] < 1:
        raise ValueError("consensus_rho must be in [0, 1)")
    if not 0 < values["consensus_floor"] <= 1:
        raise ValueError("consensus_floor must be in (0, 1] to preserve CF support")
    if not 0 <= values["consensus_text_weight"] <= 1:
        raise ValueError("consensus_text_weight must be in [0, 1]")
    if any(values[name] <= 0 for name in ("consensus_eps", "consensus_memory_budget_mib", "lr_aux_scale")):
        raise ValueError("Epsilon, feature workspace budget and auxiliary LR scale must be positive")
    required = {
        "embedding_dim": (64, 256), "num_layers": (3,), "nlgcl_G": (1,),
        "v8_arm": ("WMSG-core",), "alignment": ("off",), "relation_gate": ("off",),
        "relation_strength": (0.0,), "bsc_residual": (0.0,), "lambda_dirichlet": (0.0,),
        "semantic_mode": ("weighted",), "semantic_mix": (1.0,),
        "edge_power": (2.0,), "edge_floor": (0.05,), "knn_device": ("cpu",),
    }
    for name, allowed in required.items():
        value = _get(config, name, allowed[0])
        if isinstance(value, bool) or value not in allowed:
            raise ValueError(f"V8.1 requires {name} in {allowed}; received {value!r}")
    neighbors = _get(config, "num_neighbors", [5, 1])
    if isinstance(neighbors, str):
        neighbors = list(map(int, neighbors.split("-")))
    modalities = _get(config, "mfiles", ["textual_modality.pkl", "visual_modality.pkl"])
    if isinstance(modalities, str):
        modalities = modalities.split(",")
    if list(neighbors) != [5, 1] or len(modalities) != 2:
        raise ValueError("V8.1 requires two text/visual modalities with num_neighbors=[5, 1]")
    return values


def analytical_consensus_payload(n_items, dimension, nnz, learned_gate=True):
    """Tensor payload only; not an allocator, evaluation or no-OOM guarantee."""
    if min(n_items, dimension) <= 0 or nnz < 0:
        raise ValueError("Require positive item count/dimension and nonnegative nnz")
    return {
        "csr_int64_fp32_bytes": 8 * (n_items + 1) + 12 * nnz,
        "one_fp32_message_bytes": 4 * n_items * dimension,
        "additional_parameters": 41 if learned_gate else 0,
        "additional_parameter_grad_adam_payload_bytes": 16 * 41 if learned_gate else 0,
        "scope": "Payload estimates exclude activation lifetimes, sparse kernels, allocator and evaluation buffers.",
    }
