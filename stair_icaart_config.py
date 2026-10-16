"""Versioned, dependency-free contracts for PEEC and legacy REEC controls.

Platform: STAIR5-v8 / WMSG-core (64-dimensional primary).
Calibrates pre-BSC semantic graph weights using pooled-full evidence or historical
half-conditioned controls; all use binary TRAIN presence.
"""

from collections.abc import Mapping
import math
from typing import Any, Dict, Tuple


ICAART_OPTIONS: Tuple[Tuple[str, type, Any], ...] = (
    ("evidence-mode", str, "replicated_excess"),
    ("evidence-activity-bins", str, "log2"),
    ("evidence-halves", int, 2),
    ("evidence-seed", int, 20261016),
    ("evidence-tau", float, 2.0),
    ("evidence-delta", float, 1.0),
    ("evidence-candidate-block", int, 8192),
    ("evidence-placebo-seed", int, 20261016),
)

ICAART_NAMES: Tuple[str, ...] = tuple(flag.replace("-", "_") for flag, _, _ in ICAART_OPTIONS)
DEFAULTS: Dict[str, Any] = {flag.replace("-", "_"): value for flag, _, value in ICAART_OPTIONS}

ALLOWED_EVIDENCE_MODES: Tuple[str, ...] = (
    "replicated_excess",
    "v8_exact",
    "pooled_excess",
    "unstratified_replication",
    "replicated_count",
    "degree_stratified_shuffle",
    "uniform_multiplier",
    "pooled_activity_excess",
    "pooled_activity_count",
    "pooled_activity_shuffle",
)

FULL_ACTIVITY_MODES = frozenset((
    "pooled_activity_excess", "pooled_activity_count", "pooled_activity_shuffle",
))
EVIDENCE_CONTRACT_VERSION = 2


def evidence_identity(values: Mapping) -> Dict[str, Any]:
    """Describe the active estimator, without mislabeling pooled scores as replication."""
    mode = values["evidence_mode"]
    full = mode in FULL_ACTIVITY_MODES
    return {
        "contract_version": EVIDENCE_CONTRACT_VERSION,
        "architecture": "STAIR-PEEC-versionICAART-R2" if full else "STAIR-REEC-versionICAART",
        "null_scope": "full_activity" if full else (
            "unstratified_halves" if mode == "unstratified_replication" else "half_activity"
        ),
        "active_partition": not full,
        "active_halves": 0 if full else 2,
        "score_kind": mode,
        "pooled_regularizer": 2.0 * values["evidence_tau"] if full or mode == "pooled_excess" else None,
        "normalization": "actual_degree",
    }


def _get(config: Any, name: str, default: Any = None) -> Any:
    """Retrieve attribute or dict key uniformly."""
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def validate_icaart_config(config: Any) -> Dict[str, Any]:
    """Validate PEEC and legacy-control contracts before model execution.

    Enforces strict mathematical bounds on evidence estimation and preserves
    the unchanged STAIR5-v8 WMSG-core backbone contracts.
    """
    values = {name: _get(config, name, default) for name, default in DEFAULTS.items()}

    mode = values["evidence_mode"]
    if mode not in ALLOWED_EVIDENCE_MODES:
        raise ValueError(f"evidence_mode must be one of {ALLOWED_EVIDENCE_MODES}; got {mode!r}")

    if values["evidence_activity_bins"] != "log2":
        raise ValueError("evidence_activity_bins must be 'log2'")

    halves = values["evidence_halves"]
    full = mode in FULL_ACTIVITY_MODES
    expected_halves = 0 if full else 2
    if halves is None:
        values["evidence_halves"] = expected_halves
        halves = expected_halves
    if isinstance(halves, bool) or not isinstance(halves, int) or halves != expected_halves:
        raise ValueError(f"{mode} requires evidence_halves={expected_halves}")

    for int_field in ("evidence_seed", "evidence_placebo_seed", "evidence_candidate_block"):
        if int_field == "evidence_seed" and full:
            # No hashing, partition or graph identity depends on this inactive seed.
            values[int_field] = None
            continue
        val = values[int_field]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise ValueError(f"{int_field} must be a non-negative integer")
    if values["evidence_candidate_block"] < 1:
        raise ValueError("evidence_candidate_block must be at least 1")

    if isinstance(values["evidence_tau"], bool) or isinstance(values["evidence_delta"], bool):
        raise ValueError("evidence_tau and evidence_delta must be numeric, not boolean")
    tau = float(values["evidence_tau"])
    if not math.isfinite(tau) or not math.isfinite(2.0 * tau) or tau <= 0:
        raise ValueError(f"evidence_tau must be finite and > 0; got {tau}")
    values["evidence_tau"] = tau

    delta = float(values["evidence_delta"])
    if not math.isfinite(delta) or not 0.0 <= delta <= 2.0:
        raise ValueError(f"evidence_delta must be finite and in [0.0, 2.0]; got {delta}")
    values["evidence_delta"] = delta

    # Backbone contract validations: STAIR5-v8 WMSG-core constraints
    required_backbone = {
        "embedding_dim": (64, 256),
        "num_layers": (3,),
        "nlgcl_G": (1,),
        "lambda_nlgcl": (0.01,),
        "nlgcl_tau": (0.2,),
        "nlgcl_alpha": (0.5,),
        "v8_arm": ("WMSG-core",),
        "alignment": ("off",),
        "relation_gate": ("off",),
        "relation_strength": (0.0,),
        "bsc_residual": (0.0,),
        "lambda_dirichlet": (0.0,),
        "semantic_mode": ("weighted",),
        "semantic_mix": (1.0,),
        "edge_power": (2.0,),
        "edge_floor": (0.05,),
        "knn_device": ("cpu",),
    }

    for name, allowed in required_backbone.items():
        val = _get(config, name, allowed[0])
        if isinstance(val, bool) or val not in allowed:
            raise ValueError(f"STAIR-REEC requires {name} in {allowed}; got {val!r}")

    neighbors = _get(config, "num_neighbors", [5, 1])
    if isinstance(neighbors, str):
        neighbors = list(map(int, neighbors.split("-")))
    modalities = _get(config, "mfiles", ["textual_modality.pkl", "visual_modality.pkl"])
    if isinstance(modalities, str):
        modalities = modalities.split(",")
    if list(neighbors) != [5, 1] or len(modalities) != 2:
        raise ValueError("STAIR-REEC requires two modalities [text, vision] with num_neighbors=[5, 1]")
    if "visual" in str(modalities[0]).lower() or "textual" in str(modalities[1]).lower():
        raise ValueError("Modality order must be [text, vision], not [vision, text]")

    return values


def analytical_icaart_payload(n_items: int, n_edges: int, *, pooled: bool = False) -> Dict[str, Any]:
    """Analytical memory footprint of graph and temporary evidence structures."""
    if n_items <= 0 or n_edges < 0:
        raise ValueError("Require positive item count and non-negative edge count")
    return {
        "semantic_directed_csr_bytes": 8 * (n_items + 1) + 24 * n_edges,
        "edge_count_convention": "n_edges counts undirected semantic pairs; final CF union is additional",
        "evidence_pairs_int64_bytes": 16 * n_edges,
        "evidence_counts_int64_bytes": (8 if pooled else 16) * n_edges,
        "evidence_mu_fp64_bytes": (8 if pooled else 16) * n_edges,
        "evidence_score_fp64_bytes": 8 * n_edges,
        "evidence_weight_fp64_bytes": 8 * n_edges,
        "additional_trainable_parameters": 0,
        "scope": "All evidence structures reside transiently on CPU and are freed before GPU training.",
    }
