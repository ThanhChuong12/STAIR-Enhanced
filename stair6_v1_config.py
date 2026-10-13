"""Configuration and data-readiness contracts for STAIR6-v1 / CAE-WMSG-CSE."""

from collections.abc import Mapping
import math
from pathlib import Path


CATEGORY_OPTIONS = (
    ("category-mode", str, "cae"),
    ("category-strength", float, 0.5),
    ("category-min-count", int, 2),
    ("category-shrinkage", float, 5.0),
    ("category-mapping-file", str, ""),
    ("category-metadata-file", str, ""),
    ("category-manifest-file", str, ""),
    ("category-memory-budget-mib", float, 128.0),
    ("category-probe-dim", int, 8),
    ("category-probe-seed", int, 2718),
)
CATEGORY_NAMES = tuple(flag.replace("-", "_") for flag, _, _ in CATEGORY_OPTIONS)
DEFAULTS = {flag.replace("-", "_"): default for flag, _, default in CATEGORY_OPTIONS}


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def category_enabled(config):
    """Zero strength and explicit off mode bypass category preprocessing exactly."""
    return _get(config, "category_mode", "cae") == "cae" and _get(config, "category_strength", 0.5) > 0


def validate_category_config(config, check_files=False):
    """Return defaults while rejecting unintended changes to the V8 backbone."""
    values = {name: _get(config, name, default) for name, default in DEFAULTS.items()}
    if values["category_mode"] not in ("off", "cae"):
        raise ValueError("category_mode must be 'off' or 'cae'")
    for name in ("category_strength", "category_shrinkage", "category_memory_budget_mib"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= values["category_strength"] <= 1:
        raise ValueError("category_strength must be in [0, 1]")
    if values["category_shrinkage"] <= 0 or values["category_memory_budget_mib"] <= 0:
        raise ValueError("Category shrinkage and CPU workspace budget must be positive")
    for name, minimum in (("category_min_count", 1), ("category_probe_dim", 0), ("category_probe_seed", 0)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if values["category_probe_dim"] > 64:
        raise ValueError("CPU graph probes are bounded to at most 64 coordinates")
    for name in ("category_mapping_file", "category_metadata_file", "category_manifest_file"):
        if not isinstance(values[name], (str, Path)):
            raise ValueError(f"{name} must be a filesystem path")
        values[name] = str(values[name])
    fixed = {
        "v8_arm": "WMSG-core", "semantic_mode": "weighted", "semantic_mix": 1.0,
        "alignment": "off", "relation_gate": "off", "relation_strength": 0.0,
        "bsc_residual": 0.0, "lambda_dirichlet": 0.0,
        "edge_power": 2.0, "edge_floor": 0.05, "knn_device": "cpu", "nlgcl_G": 1,
    }
    for name, expected in fixed.items():
        actual = _get(config, name, expected)
        if actual != expected:
            raise ValueError(f"STAIR6-v1 requires {name}={expected!r}; received {actual!r}")
    # Dimension is an explicit experiment factor. Default configs are 64D;
    # matched 256D and small synthetic fixtures remain possible without new layers.
    dimension, layers = _get(config, "embedding_dim", 64), _get(config, "num_layers", 3)
    if (type(dimension) is not int or dimension <= 0 or type(layers) is not int or layers != 3):
        raise ValueError("Require positive embedding dimension and the reference three FSC/BSC layers")
    if type(_get(config, "nlgcl_G", 1)) is not int:
        raise ValueError("nlgcl_G must be the integer 1, not a boolean")
    if check_files and category_enabled(values):
        for name in ("category_mapping_file", "category_metadata_file", "category_manifest_file"):
            if not values[name] or not Path(values[name]).is_file():
                raise FileNotFoundError(
                    f"Category-enabled training requires {name}: {values[name]!r}. "
                    "Recover a verified ASIN mapping and metadata, or explicitly use "
                    "--category-mode off / --category-strength 0 for the V8 control."
                )
    return values
