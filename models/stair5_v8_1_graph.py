"""Static, restricted-support consensus view for STAIR5-v8.1.

This auxiliary operator never modifies V8's ranking or BSC graphs. Construction
is CPU-only and scores only the upper-triangle edges already retained by CSE.
Its isolated rows are zero, deliberately unlike V8's BSC identity fallback.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_graph import (
    CACHE_VERSION, _digest_array, binary_train_keys,
    build_candidate_support_graph, scipy_to_tensor, sparse_fingerprint,
    tensor_to_scipy,
)
from models.stair5_v4_utils import atomic_torch_save, file_sha256
from models.stair5_v8_graph import compute_selected_edge_cosines


CONSENSUS_CACHE_VERSION = 1
CONSENSUS_MODES = ("off", "soft", "cf-only", "placebo")
_CACHE_ERRORS = (OSError, RuntimeError, ValueError, KeyError, EOFError,
                 TypeError, AttributeError, IndexError, pickle.UnpicklingError)


@dataclass
class GraphStateConsensus:
    """Detached CPU CSR, covered item rows, and serializable provenance."""

    operator: torch.Tensor
    coverage: torch.Tensor
    metadata: dict


def _validate_raw_graph(matrix, n_items):
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape != (n_items, n_items):
        raise ValueError("Consensus graph shape differs from the item catalog.")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Consensus weights must be finite and nonnegative.")
    difference = matrix - matrix.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-7:
        raise ValueError("Consensus weights must be symmetric.")
    if np.any(matrix.diagonal() != 0):
        raise ValueError("Raw CSE/consensus graphs must have zero diagonal.")
    return matrix


def normalize_consensus_graph(weights):
    """Symmetric normalization with FP64 degrees and ZERO isolated rows."""
    weights = _validate_raw_graph(weights, weights.shape[0])
    degrees = np.asarray(weights.sum(axis=1, dtype=np.float64)).ravel()
    inverse = np.zeros_like(degrees)
    positive = degrees > 0
    inverse[positive] = 1.0 / np.sqrt(degrees[positive])
    scale = sp.diags(inverse, format="csr")
    operator = (scale @ weights.astype(np.float64) @ scale).astype(np.float32)
    operator.eliminate_zeros()
    operator.sort_indices()
    return operator, positive


def _raw_cf_graph(train_edges, n_users, n_items, cache_dir, k_cf, c_min,
                  t_shrinkage, block_size, memory_budget_mib, pairs):
    """Reuse the EXACT raw-CF cache signature/payload used by V8."""
    pair_hash = hashlib.sha256()
    _digest_array(pair_hash, pairs)
    signature = {
        "cache_version": CACHE_VERSION, "train": pair_hash.hexdigest(),
        "users": n_users, "items": n_items, "k": k_cf, "cmin": c_min,
        "t": t_shrinkage,
        "source": file_sha256(Path(__file__).with_name("stair5_v4_graph.py")),
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"cf_{key}.pt" if cache_dir else None
    if cache and cache.is_file():
        try:
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            candidate = tensor_to_scipy(payload["operator"])
            if payload.get("signature") != signature:
                raise ValueError("Raw CF cache signature mismatch.")
            if payload.get("fingerprint") != sparse_fingerprint(candidate):
                raise ValueError("Raw CF cache checksum mismatch.")
            return _validate_raw_graph(candidate, n_items), signature, pairs, True
        except _CACHE_ERRORS as exc:
            warnings.warn(f"Ignoring invalid raw CF cache: {exc}", RuntimeWarning)
    weights = build_candidate_support_graph(
        train_edges=train_edges, n_users=n_users, n_items=n_items, k_cf=k_cf,
        c_min=c_min, t_shrinkage=t_shrinkage, block_size=block_size,
        memory_budget_mib=memory_budget_mib,
    )
    weights = _validate_raw_graph(weights, n_items)
    if cache:
        atomic_torch_save({"operator": scipy_to_tensor(weights),
                           "signature": signature,
                           "fingerprint": sparse_fingerprint(weights)}, cache)
    return weights, signature, pairs, False


def _feature_signature(features, n_items):
    """Hash all raw rows in their supplied item order; not retained SVD columns."""
    if len(features) != 2:
        raise ValueError("Consensus requires [text_features, visual_features].")
    signatures = []
    zero_counts = []
    for name, feature in zip(("text", "visual"), features):
        if (not isinstance(feature, torch.Tensor) or feature.device.type != "cpu"
                or feature.dtype != torch.float32 or feature.ndim != 2
                or feature.shape[0] != n_items or feature.shape[1] < 1):
            raise ValueError(f"{name} features must be CPU FP32 [n_items, feature_dim].")
        if not torch.isfinite(feature).all():
            raise ValueError(f"{name} features contain NaN or Inf.")
        digest = hashlib.sha256()
        _digest_array(digest, feature.detach().numpy())
        signatures.append({"modality": name, "shape": list(feature.shape),
                           "content_and_row_order": digest.hexdigest()})
        zero_counts.append(int(torch.count_nonzero(
            torch.linalg.vector_norm(feature, dim=1) == 0)))
    return signatures, zero_counts


def _stratified_placebo(values, row, column, pairs, n_items, seed):
    """Shuffle symmetric semantic multipliers within degree-pair quintiles.

    Rank-based strata break degree ties deterministically by item ID. Neither
    support nor the CF q weights are permuted; weighted degrees may change.
    """
    degrees = np.bincount(pairs % n_items, minlength=n_items)
    order = np.lexsort((np.arange(n_items), degrees))
    strata = np.empty(n_items, dtype=np.int64)
    strata[order] = np.minimum(4, np.arange(n_items, dtype=np.int64) * 5 // n_items)
    low = np.minimum(strata[row], strata[column])
    high = np.maximum(strata[row], strata[column])
    edge_strata = low * 5 + high
    result = values.copy()
    generator = np.random.default_rng(seed)
    movable = 0
    for group in np.unique(edge_strata):
        indices = np.flatnonzero(edge_strata == group)
        if len(indices) > 1:
            movable += len(indices)
            result[indices] = values[generator.permutation(indices)]
    return result, {
        "placebo_movable_edges": int(movable),
        "placebo_changed_edges": int(np.count_nonzero(result != values)),
        "placebo_strata": "binary_train_degree_rank_quintile_pair",
    }


@torch.no_grad()
def build_consensus_graph(
    train_edges, n_users, n_items, mfeats, cache_dir=None, k_cf=5, c_min=2,
    t_shrinkage=5.0, mode="soft", floor=0.05, text_weight=5.0 / 6.0,
    placebo_seed=1, block_size=64, memory_budget_mib=128.0,
    feature_memory_budget_mib=64.0,
):
    """Build the report's fixed-support soft consensus, entirely on CPU.

    ``cf-only`` uses the same zero-isolate normalization without semantic
    multipliers. ``placebo`` shuffles those multipliers within degree-pair
    strata before normalization. ``off`` skips CF and feature preprocessing.
    Budgets bound selected-pair/SpGEMM workspaces, not total process memory.
    """
    started = time.perf_counter()
    if mode not in CONSENSUS_MODES:
        raise ValueError(f"Unknown consensus mode: {mode}")
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("Consensus floor must be finite in (0, 1].")
    if not math.isfinite(text_weight) or not 0 <= text_weight <= 1:
        raise ValueError("Text weight must be finite in [0, 1].")
    if (int(k_cf) != k_cf or k_cf < 0 or int(c_min) != c_min or c_min < 1
            or not math.isfinite(t_shrinkage) or t_shrinkage <= 0):
        raise ValueError("Invalid CSE support construction settings.")
    if (int(block_size) != block_size or block_size < 1
            or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0
            or not math.isfinite(feature_memory_budget_mib)
            or feature_memory_budget_mib <= 0):
        raise ValueError("Workspace budgets and block size must be positive.")
    if int(placebo_seed) != placebo_seed or placebo_seed < 0:
        raise ValueError("Placebo seed must be a nonnegative integer.")
    pairs = binary_train_keys(train_edges, n_users, n_items)
    if mode == "off":
        zero = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        return GraphStateConsensus(scipy_to_tensor(zero), torch.zeros(n_items, dtype=torch.bool),
                                   {"mode": mode, "active": False, "cache_hit": False,
                                    "cf_cache_hit": False, "nnz": 0,
                                    "coverage_count": 0, "graph_fingerprint": sparse_fingerprint(zero),
                                    "graph_build_seconds": time.perf_counter() - started})

    feature_hashes, zero_counts = _feature_signature(mfeats, n_items)
    raw_cf, raw_signature, pairs, cf_hit = _raw_cf_graph(
        train_edges, n_users, n_items, cache_dir, k_cf, c_min, t_shrinkage,
        block_size, memory_budget_mib, pairs,
    )
    signature = {
        "version": CONSENSUS_CACHE_VERSION, "source": file_sha256(__file__),
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "cosine_source": file_sha256(Path(__file__).with_name("stair5_v8_graph.py")),
        "raw_cf": raw_signature, "raw_cf_fingerprint": sparse_fingerprint(raw_cf),
        "features": feature_hashes, "mode": mode, "floor": float(floor),
        "text_weight": float(text_weight),
        "placebo_seed": int(placebo_seed) if mode == "placebo" else None,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"consensus_v81_{key}.pt" if cache_dir else None
    metadata = {"mode": mode, "cache_hit": False, "cf_cache_hit": cf_hit,
                "cache_key": key, "raw_cf_nnz": int(raw_cf.nnz),
                "raw_cf_fingerprint": signature["raw_cf_fingerprint"],
                "feature_hashes": feature_hashes,
                "text_zero_rows": zero_counts[0], "visual_zero_rows": zero_counts[1],
                "floor": float(floor), "text_weight": float(text_weight)}
    if cache and cache.is_file():
        try:
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            candidate = _validate_raw_graph(tensor_to_scipy(payload["operator"]), n_items)
            coverage = np.diff(candidate.indptr) > 0
            if payload.get("signature") != signature:
                raise ValueError("Consensus cache signature mismatch.")
            if payload.get("fingerprint") != sparse_fingerprint(candidate):
                raise ValueError("Consensus cache checksum mismatch.")
            if (payload["coverage"].dtype != torch.bool
                    or not np.array_equal(payload["coverage"].numpy(), coverage)):
                raise ValueError("Consensus cache coverage mismatch.")
            if ((candidate.astype(bool) != raw_cf.astype(bool)).nnz):
                raise ValueError("Consensus cache changed the retained CF support.")
            metadata.update(payload.get("audit", {}), cache_hit=True, active=bool(candidate.nnz),
                            nnz=int(candidate.nnz), coverage_count=int(coverage.sum()),
                            graph_fingerprint=sparse_fingerprint(candidate),
                            graph_build_seconds=time.perf_counter() - started)
            return GraphStateConsensus(scipy_to_tensor(candidate), torch.from_numpy(coverage), metadata)
        except _CACHE_ERRORS as exc:
            warnings.warn(f"Ignoring invalid consensus cache: {exc}", RuntimeWarning)

    upper = sp.triu(raw_cf, k=1, format="coo")
    row, column = upper.row.astype(np.int64), upper.col.astype(np.int64)
    audit = {}
    if mode == "cf-only" or not upper.nnz:
        multipliers = np.ones(upper.nnz, dtype=np.float64)
    else:
        edges = torch.from_numpy(np.stack((row, column)))
        text = compute_selected_edge_cosines(
            mfeats[0], edges, memory_budget_mib=feature_memory_budget_mib)
        visual = compute_selected_edge_cosines(
            mfeats[1], edges, memory_budget_mib=feature_memory_budget_mib)
        agreement = text_weight * np.maximum(text, 0).astype(np.float64)
        agreement += (1.0 - text_weight) * np.maximum(visual, 0).astype(np.float64)
        multipliers = floor + (1.0 - floor) * agreement
        audit.update(semantic_agreement_mean=float(agreement.mean()),
                     semantic_agreement_std=float(agreement.std()))
        if mode == "placebo":
            multipliers, placebo_audit = _stratified_placebo(
                multipliers, row, column, pairs, n_items, int(placebo_seed))
            audit.update(placebo_audit)
    values = upper.data.astype(np.float64) * multipliers
    weights = sp.csr_matrix((np.concatenate((values, values)),
                            (np.concatenate((row, column)), np.concatenate((column, row)))),
                           shape=(n_items, n_items))
    operator, coverage = normalize_consensus_graph(weights)
    if (operator.astype(bool) != raw_cf.astype(bool)).nnz:
        raise ValueError("Consensus normalization lost retained CF edges; inspect numeric scale.")
    cf_degrees = np.asarray(raw_cf.sum(axis=1, dtype=np.float64)).ravel()
    new_degrees = np.asarray(weights.sum(axis=1, dtype=np.float64)).ravel()
    audit["relative_weighted_degree_change"] = float(
        np.linalg.norm(new_degrees - cf_degrees) / max(1e-12, np.linalg.norm(cf_degrees)))
    tensor = scipy_to_tensor(operator)
    coverage_tensor = torch.from_numpy(coverage)
    fingerprint = sparse_fingerprint(operator)
    if cache:
        atomic_torch_save({"operator": tensor, "coverage": coverage_tensor,
                           "signature": signature, "fingerprint": fingerprint,
                           "audit": audit}, cache)
    metadata.update(audit, active=bool(operator.nnz), nnz=int(operator.nnz),
                    coverage_count=int(coverage.sum()), graph_fingerprint=fingerprint,
                    graph_build_seconds=time.perf_counter() - started)
    return GraphStateConsensus(tensor, coverage_tensor, metadata)
