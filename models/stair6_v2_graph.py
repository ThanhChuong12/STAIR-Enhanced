"""CPU-only CAN auxiliary WMSG construction and normalized-operator interpolation.

Only the final CSR is retained by the model. CAN probabilities select text
support; visual support, native magnitudes and CF normalization stay unchanged.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_graph import (
    binary_train_keys, normalize_cf_graph_with_fallback, scipy_to_tensor,
    sparse_fingerprint, tensor_to_scipy,
)
from models.stair5_v4_utils import atomic_torch_save, file_sha256
from models.stair5_v8_graph import (
    compute_selected_edge_cosines, normalize_graph_with_isolated_fallback, power_scale_similarities,
)
from models.stair6_v1_graph import load_native_raw_cf
from models.stair6_v2_can import calibrate_can, prefix_mask, shuffled_counts
from stair6_v2_config import CAN_NAMES, can_enabled, validate_can_config


@dataclass(frozen=True)
class GraphStateSTAIR6V2:
    operator: torch.Tensor
    metadata: dict
    contract: dict


def array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_candidates(edges, n_items, cap):
    """Validate frozen IDs, returning stable source-grouping positions."""
    if (edges.device.type != "cpu" or edges.dtype != torch.int64
            or tuple(edges.shape) != (2, n_items * cap)):
        raise ValueError("Frozen candidates must be CPU int64 [2, items*cap]")
    ids = edges.numpy()
    if ids.min() < 0 or ids.max() >= n_items or np.any(ids[0] == ids[1]):
        raise ValueError("Frozen candidates contain self edges or out-of-catalog IDs")
    if (not np.array_equal(np.bincount(ids[0], minlength=n_items), np.full(n_items, cap))
            or np.unique(ids[0] * n_items + ids[1]).size != ids.shape[1]):
        raise ValueError("Frozen candidates contain duplicate pairs or invalid per-row counts")
    return np.argsort(ids[0], kind="stable")


def semantic_from_weights(edges, weights, n_items, text_keep=None):
    """Sum modality occurrences first, then maximum-symmetrize like native V8."""
    if len(edges) != 2 or len(weights) != 2:
        raise ValueError("Require text and visual occurrences")
    keep = np.ones(edges[0].shape[1], dtype=bool) if text_keep is None else np.asarray(text_keep)
    if keep.shape != (edges[0].shape[1],) or keep.dtype != np.bool_:
        raise ValueError("Text keep mask must align with the original edge order")
    src = np.concatenate((edges[0][0].numpy()[keep], edges[1][0].numpy()))
    dst = np.concatenate((edges[0][1].numpy()[keep], edges[1][1].numpy()))
    data = np.concatenate((weights[0][keep], weights[1])).astype(np.float32, copy=False)
    directed = sp.csr_matrix((data, (src, dst)), shape=(n_items, n_items))
    directed.sum_duplicates()
    raw = directed.maximum(directed.T).tocsr()
    raw.setdiag(0)
    raw.eliminate_zeros()
    raw.sort_indices()
    return raw


def interpolate_operators(semantic, auxiliary, raw_cf, eta, kappa):
    """Blend separately normalized operators; never normalize their mixture again."""
    if not np.isfinite([eta, kappa]).all() or not 0 <= eta <= 1 or not 0 <= kappa <= 1:
        raise ValueError("eta and kappa must be finite fractions")
    effective_eta = float(eta) if raw_cf.nnz else 0.0
    # Preserve the exact endpoint operation order from native V8.
    mixed = semantic if kappa == 0 else auxiliary if kappa == 1 else (
        (1 - kappa) * semantic + kappa * auxiliary).tocsr()
    result = ((1 - effective_eta) * mixed + effective_eta * normalize_cf_graph_with_fallback(raw_cf)
              ).tocsr() if effective_eta else mixed.copy()
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result.astype(np.float32), effective_eta


def _norm(matrix):
    return float(np.sqrt(np.dot(matrix.data.astype(np.float64), matrix.data.astype(np.float64))))


def _validate_operator(matrix, n_items):
    if not sp.issparse(matrix) or matrix.shape != (n_items, n_items):
        raise ValueError("CAN operator must be a square catalog-aligned sparse matrix")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("CAN operator must be finite and nonnegative")
    difference = matrix - matrix.T
    if difference.nnz and np.max(np.abs(difference.data)) > 2e-7:
        raise ValueError("CAN operator must be symmetric")


@torch.no_grad()
def _probe(parent, final, beta, layers, dim, seed):
    if dim == 0:
        return {}
    from optimizers.stair5_v8_smoother import STAIR5V8Smoother
    if beta is None or len(beta) < dim:
        raise ValueError("Diagnostic dimension must fit the native coordinate schedule")
    values = np.random.default_rng(seed).standard_normal((parent.shape[0], dim)).astype(np.float32)
    probe = torch.from_numpy(values)
    old_tensor, new_tensor = scipy_to_tensor(parent), scipy_to_tensor(final)
    outputs = []
    for matrix in (old_tensor, new_tensor):
        smoother = STAIR5V8Smoother(lambda x: torch.sparse.mm(matrix, x), beta=beta[:dim], L=layers)
        outputs.append(smoother(probe))
    return {"probe_seed": int(seed), "probe_dim": int(dim), "probe_coordinate_policy": "first_native_coordinates",
            "bsc_probe_relative_delta": float(torch.linalg.vector_norm(outputs[1] - outputs[0]) /
                                               torch.linalg.vector_norm(outputs[0]).clamp_min(1e-12))}


@torch.no_grad()
def build_stair6_v2_graph(baseline_state, features=None, candidates=None, train_edges=None,
                          n_users=None, n_items=None, cfg=None, beta=None, num_layers=3):
    """Build a static correction, returning the original tensor for off/zero dose.

    Reuses the parent's content-addressed raw CF cache. No second kNN discovery,
    whitening, catalog-wide CF product, accelerator allocation or global RNG use.
    """
    settings = validate_can_config(cfg if cfg is not None else {})
    baseline = baseline_state.operator
    parent = tensor_to_scipy(baseline)
    parent_hash = sparse_fingerprint(parent)
    contract = {"version": 1, "settings": settings, "baseline_graph_fingerprint": parent_hash}
    if not can_enabled(settings):
        return GraphStateSTAIR6V2(baseline, {
            "requested_mode": settings["can_mode"], "effective_mode": "off", "active": False,
            "reason": "explicit_off_or_zero_interpolation", "graph_fingerprint": parent_hash,
            "baseline_graph_fingerprint": parent_hash,
        }, contract)
    started = time.perf_counter()
    if baseline.device.type != "cpu" or len(features) != 2 or len(candidates) != 2:
        raise ValueError("CAN preparation requires the native CPU parent and two modality inputs")
    grouped = validate_candidates(candidates[0], n_items, 5)
    validate_candidates(candidates[1], n_items, 1)
    budget = settings["can_memory_budget_mib"]
    cosines = [compute_selected_edge_cosines(feature, edges, memory_budget_mib=budget)
               for feature, edges in zip(features, candidates)]
    costs = 1 - cosines[0][grouped].reshape(n_items, 5).astype(np.float64)
    mode = settings["can_mode"]
    diagnostic = {}
    if mode == "fixed":
        counts = np.full(n_items, settings["can_fixed_text_k"], dtype=np.int64)
        mask = prefix_mask(costs, counts)
        diagnostic.update(actual_mean_text_k=float(counts.mean()), regularizer=None)
    else:
        result = calibrate_can(costs, settings["can_target_mean_text_k"],
                               settings["can_budget_relative_tolerance"], budget)
        counts, mask = result.counts, result.mask
        diagnostic.update(result.metadata)
        if mode == "shuffled":
            pairs = binary_train_keys(train_edges, n_users, n_items)
            degrees = np.bincount(pairs % n_items, minlength=n_items)
            shuffled = shuffled_counts(counts, degrees, settings["can_placebo_seed"])
            mask = prefix_mask(costs, shuffled)
            diagnostic.update(placebo_changed_rows=int(np.count_nonzero(counts != shuffled)),
                              placebo_strata="binary_train_degree_log2")
            counts = shuffled
        del result
    edge_mask = np.empty(n_items * 5, dtype=bool)
    edge_mask[grouped] = mask.reshape(-1)
    weights = [power_scale_similarities(values, edge_power=2.0, edge_floor=0.05) for values in cosines]
    native_raw = semantic_from_weights(candidates, weights, n_items)
    auxiliary_raw = semantic_from_weights(candidates, weights, n_items, edge_mask)
    native_sem = normalize_graph_with_isolated_fallback(native_raw)
    auxiliary_sem = normalize_graph_with_isolated_fallback(auxiliary_raw)
    cache_dir = getattr(cfg, "graph_cache_dir", "") or None
    raw_cf, cf_audit = load_native_raw_cf(
        train_edges, n_users, n_items, k_cf=cfg.k_cf, c_min=cfg.c_min, t_shrinkage=cfg.t_shrinkage,
        cache_dir=cache_dir, block_size=cfg.cf_block_size, memory_budget_mib=cfg.cf_memory_budget_mib,
    )
    reconstructed, effective_eta = interpolate_operators(native_sem, native_sem, raw_cf, cfg.eta, 0)
    if sparse_fingerprint(reconstructed) != parent_hash:
        raise ValueError("Native parent fingerprint mismatch; do not attribute a normalization change to CAN")
    identity = {
        **contract, "cost_sha256": array_digest(costs), "mask_sha256": array_digest(edge_mask),
        "candidate_sha256": [array_digest(edges.numpy()) for edges in candidates],
        "raw_semantic_fingerprint": sparse_fingerprint(native_raw),
        "auxiliary_semantic_fingerprint": sparse_fingerprint(auxiliary_raw),
        "cf_fingerprint": cf_audit["cf_fingerprint"], "effective_eta": effective_eta,
        "regularizer": diagnostic["regularizer"],
        "torch": str(torch.__version__), "numpy": np.__version__,
        "sources": {p.name: file_sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("stair6_v2_can.py"),
            Path(__file__).with_name("stair6_v1_graph.py"),
            Path(__file__).resolve().parents[1] / "stair6_v2_config.py")},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"can_{key}.pt" if cache_dir else None
    final, hit = None, False
    if path and path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            loaded = tensor_to_scipy(payload["operator"])
            _validate_operator(loaded, n_items)
            if payload["identity"] != identity or payload["fingerprint"] != sparse_fingerprint(loaded):
                raise ValueError("CAN cache identity/checksum mismatch")
            final, hit = loaded, True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, TypeError, IndexError,
                AttributeError, pickle.UnpicklingError) as error:
            warnings.warn(f"Ignoring invalid CAN cache: {error}", RuntimeWarning)
    if final is None:
        final, _ = interpolate_operators(native_sem, auxiliary_sem, raw_cf, cfg.eta, settings["can_interpolation"])
        _validate_operator(final, n_items)
        if path:
            atomic_torch_save({"operator": scipy_to_tensor(final), "identity": identity,
                               "fingerprint": sparse_fingerprint(final)}, path)
    fingerprint = sparse_fingerprint(final)
    identity["graph_fingerprint"] = fingerprint
    metadata = {
        **diagnostic, **cf_audit, "requested_mode": mode, "effective_mode": mode, "active": True,
        "architecture": "AKP-WMSG-CSE", "interpolation": settings["can_interpolation"],
        "effective_eta": effective_eta, "mask_sha256": identity["mask_sha256"],
        "k_distribution": {str(k): int(np.count_nonzero(counts == k)) for k in range(1, 6)},
        "semantic_nnz_v8": int(native_raw.nnz), "semantic_nnz_auxiliary": int(auxiliary_raw.nnz),
        "final_nnz_v8": int(parent.nnz), "final_nnz": int(final.nnz),
        "removed_raw_semantic_mass_fraction": float(1 - auxiliary_raw.data.astype(np.float64).sum() /
                                                    native_raw.data.astype(np.float64).sum()),
        "relative_operator_delta_vs_v8": _norm(final - parent) / max(_norm(parent), 1e-12),
        "final_support_equal_v8": (np.array_equal(final.indptr, parent.indptr)
                                   and np.array_equal(final.indices, parent.indices)),
        "baseline_graph_fingerprint": parent_hash, "graph_fingerprint": fingerprint,
        "cache_hit": hit, "cache_key": key, "preprocessing_seconds": time.perf_counter() - started,
        **_probe(parent, final, beta, num_layers, settings["can_probe_dim"], settings["can_probe_seed"]),
    }
    return GraphStateSTAIR6V2(scipy_to_tensor(final), metadata, identity)
