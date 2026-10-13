"""CPU-only, frozen-support category enrichment of the native V8 graph.

The final CSR replaces S8; category statistics are never retained on the GPU.
Raw CF caching deliberately shares the V8 cache key and deterministic builder.
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
    CACHE_VERSION, _digest_array, binary_train_keys, build_candidate_support_graph,
    normalize_cf_graph_with_fallback, scipy_to_tensor, sparse_fingerprint, tensor_to_scipy,
)
from models.stair5_v4_utils import atomic_torch_save, file_sha256
from models.stair5_v8_graph import normalize_graph_with_isolated_fallback
from models.stair6_v1_categories import build_category_edge_scores

CACHE_VERSION_STAIR6 = 1


class InactiveCategorySignal(ValueError):
    """Enabled metadata supplies no informative positive category association."""


@dataclass
class GraphStateSTAIR6:
    operator: torch.Tensor
    metadata: dict


def _canonical_graph(matrix, n_items, name, zero_diagonal=False):
    if not sp.issparse(matrix) or matrix.shape != (n_items, n_items):
        raise ValueError(f"{name} must be a catalog-sized sparse matrix.")
    result = matrix.astype(np.float32).tocsr(copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all() or np.any(result.data < 0):
        raise ValueError(f"{name} weights must be finite and nonnegative.")
    difference = result - result.T
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError(f"{name} must be exactly symmetric.")
    if zero_diagonal and np.any(result.diagonal() != 0):
        raise ValueError(f"{name} must have a zero diagonal.")
    return result


def _pair_digest(train_edges, n_users, n_items):
    hasher = hashlib.sha256()
    _digest_array(hasher, binary_train_keys(train_edges, n_users, n_items))
    return hasher.hexdigest()


@torch.no_grad()
def load_native_raw_cf(train_edges, n_users, n_items, k_cf=5, c_min=2,
                       t_shrinkage=5.0, cache_dir=None, block_size=64, memory_budget_mib=128):
    """Return (raw symmetric CSR, audit), validating native content-addressed caches."""
    if not isinstance(k_cf, int) or k_cf < 0 or not isinstance(c_min, int) or c_min < 1:
        raise ValueError("CF top-k/count must be nonnegative/positive integers.")
    if not math.isfinite(t_shrinkage) or t_shrinkage <= 0:
        raise ValueError("CF shrinkage must be finite and positive.")
    signature = {"cache_version": CACHE_VERSION,
                 "train": _pair_digest(train_edges, n_users, n_items),
                 "users": n_users, "items": n_items, "k": k_cf,
                 "cmin": c_min, "t": t_shrinkage,
                 "source": file_sha256(Path(__file__).with_name("stair5_v4_graph.py"))}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"cf_{key}.pt" if cache_dir else None
    raw, hit = None, False
    if path and path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            candidate = _canonical_graph(tensor_to_scipy(payload["operator"]), n_items, "Cached raw CF", True)
            if payload.get("signature") != signature or payload.get("fingerprint") != sparse_fingerprint(candidate):
                raise ValueError("CF cache signature/checksum mismatch.")
            raw, hit = candidate, True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, TypeError, pickle.UnpicklingError) as error:
            warnings.warn(f"Ignoring invalid native CF cache: {error}", RuntimeWarning)
    if raw is None:
        raw = build_candidate_support_graph(train_edges, n_items, n_users, k_cf, c_min,
                                            t_shrinkage, block_size, memory_budget_mib)
        raw = _canonical_graph(raw, n_items, "Raw CF", True)
        if path:
            atomic_torch_save({"operator": scipy_to_tensor(raw), "signature": signature,
                               "fingerprint": sparse_fingerprint(raw)}, path)
    return raw, {"cf_cache_hit": hit, "cf_cache_key": key,
                 "cf_fingerprint": sparse_fingerprint(raw), "cf_nnz": int(raw.nnz)}


def enrich_existing_weights(raw_semantic, scores, strength):
    """Multiply CSR-aligned values without removing, adding or reordering edges."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Category enrichment strength must be finite in [0, 1].")
    raw = _canonical_graph(raw_semantic, raw_semantic.shape[0], "Raw semantic", True)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != raw.data.shape or not np.isfinite(scores).all():
        raise ValueError("Category scores must be finite and aligned with canonical CSR values.")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Category association scores must lie in [0, 1].")
    scored = sp.csr_matrix((scores, raw.indices.copy(), raw.indptr.copy()), shape=raw.shape)
    difference = scored - scored.T
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("Pooled category scores must be exactly symmetric.")
    enriched = raw.copy()
    enriched.data *= 1.0 + np.float32(strength) * scores
    return enriched


def _blend(raw_semantic, raw_cf, eta):
    semantic = normalize_graph_with_isolated_fallback(raw_semantic)
    effective_eta = float(eta) if raw_cf.nnz else 0.0
    final = ((1 - effective_eta) * semantic + effective_eta * normalize_cf_graph_with_fallback(raw_cf)).tocsr() if effective_eta else semantic
    final.sum_duplicates()
    final.eliminate_zeros()
    final.sort_indices()
    return final, effective_eta


def _frobenius(matrix):
    values = matrix.data.astype(np.float64)
    return float(np.sqrt(np.dot(values, values)))


def _same_support(left, right):
    return (left.shape == right.shape and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices))


def _probe_diagnostics(old, new, dim, seed, beta=None, num_layers=3):
    """Bounded deterministic CPU probes; not eigensolver or ranking guarantees."""
    if not isinstance(dim, int) or dim < 0 or not isinstance(num_layers, int) or num_layers < 0:
        raise ValueError("Probe dimension/layers must be nonnegative integers.")
    if dim == 0:
        return {}
    probe = np.random.default_rng(seed).standard_normal((old.shape[0], dim)).astype(np.float32)
    reference, candidate = old @ probe, new @ probe
    result = {"operator_probe_relative_delta": float(np.linalg.norm(candidate - reference) / max(1e-12, np.linalg.norm(reference))),
              "probe_dim": dim, "probe_seed": int(seed)}
    if beta is not None:
        coefficient = np.asarray(beta, dtype=np.float64).reshape(-1)
        if not coefficient.size or not np.isfinite(coefficient).all() or np.any(coefficient < 0) or np.any(coefficient >= 1):
            raise ValueError("BSC beta coefficients must lie in [0, 1).")
        sampled = coefficient[np.linspace(0, coefficient.size - 1, dim).astype(np.int64)]
        old_current, new_current = probe.copy(), probe.copy()
        old_sum, new_sum = probe.copy(), probe.copy()
        for _ in range(num_layers):
            old_current = (old @ old_current) * sampled
            new_current = (new @ new_current) * sampled
            old_sum += old_current
            new_sum += new_current
        correction = (1 - sampled) / (1 - sampled ** (num_layers + 1))
        old_sum *= correction
        new_sum *= correction
        result["bsc_probe_relative_delta"] = float(np.linalg.norm(new_sum - old_sum) / max(1e-12, np.linalg.norm(old_sum)))
        result["bsc_probe_num_layers"] = num_layers
    return result


@torch.no_grad()
def build_stair6_v1_graph(baseline_state, raw_semantic=None, raw_cf=None,
                          train_edges=None, n_users=None, n_items=None,
                          category_data=None, category_mode="cae", strength=0.5,
                          min_count=2, shrinkage=5.0, eta=0.1, cache_dir=None,
                          memory_budget_mib=128, probe_dim=8, probe_seed=2718,
                          beta=None, num_layers=3):
    """Enrich raw WMSG; off/zero strength returns the exact original tensor.

    Enabled runs require verified metadata and positive association. Signatures
    bind sources, raw weights, membership/provenance and unique train pairs.
    """
    if category_mode not in ("off", "cae"):
        raise ValueError("category_mode must be 'off' or 'cae'.")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Category strength must be finite in [0, 1].")
    metadata = dict(baseline_state.metadata)
    metadata.update(version=6, architecture="STAIR6-v1", category_mode=category_mode,
                    category_strength=float(strength), category_active=False, category_cache_hit=False)
    if category_mode == "off" or strength == 0:
        metadata.update(category_effective_mode="off", category_bypass=True,
                        category_bypass_reason="disabled" if category_mode == "off" else "zero_strength")
        return GraphStateSTAIR6(baseline_state.operator, metadata)
    started = time.perf_counter()
    if category_data is None:
        raise ValueError("CAE requires verified item mapping and category metadata.")
    if not isinstance(min_count, int) or min_count < 1 or not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("Category count/shrinkage must be positive.")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("CF mixing must be finite in [0, 1].")
    if not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("CPU workspace budget must be finite and positive.")
    if n_items is None or n_users is None or train_edges is None:
        raise ValueError("Enabled CAE requires catalog sizes and train interactions.")
    if not isinstance(probe_dim, int) or probe_dim < 0:
        raise ValueError("Probe dimension must be a nonnegative integer.")
    if 64 * n_items * probe_dim > memory_budget_mib * 2**20:
        raise MemoryError("CPU probe workspace exceeds the configured budget; reduce probe_dim.")
    raw = _canonical_graph(raw_semantic, n_items, "Raw WMSG", True)
    cf = _canonical_graph(raw_cf, n_items, "Raw CF", True)
    original = _canonical_graph(tensor_to_scipy(baseline_state.operator), n_items, "V8 operator")
    reconstructed, _ = _blend(raw, cf, eta)
    mismatch = reconstructed - original
    if not _same_support(reconstructed, original) or (mismatch.nnz and np.max(np.abs(mismatch.data)) > 1e-7):
        raise ValueError("Raw WMSG/CF inputs do not reconstruct the native WMSG-core operator.")
    fingerprint = getattr(category_data, "metadata_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Category data must carry verified artifact provenance.")
    signature = {"cache_version": CACHE_VERSION_STAIR6,
                 "train": _pair_digest(train_edges, n_users, n_items),
                 "users": n_users, "items": n_items,
                 "raw_semantic": sparse_fingerprint(raw), "raw_cf": sparse_fingerprint(cf),
                 "baseline": sparse_fingerprint(original), "category": fingerprint,
                 "membership": sparse_fingerprint(category_data.memberships),
                 "strength": float(strength), "min_count": min_count,
                 "shrinkage": float(shrinkage), "eta": float(eta),
                 "sources": {name: file_sha256(Path(__file__).with_name(name)) for name in
                             ("stair6_v1_graph.py", "stair6_v1_categories.py", "stair5_v8_graph.py", "stair5_v4_graph.py")}}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"cae_{key}.pt" if cache_dir else None
    scores, audit = None, None
    if path and path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("signature") != signature:
                raise ValueError("CAE cache signature mismatch.")
            scores = payload["scores"].detach().cpu().numpy()
            hasher = hashlib.sha256()
            _digest_array(hasher, scores)
            if payload.get("score_fingerprint") != hasher.hexdigest():
                raise ValueError("CAE cache score checksum mismatch.")
            audit = payload["audit"]
            if not isinstance(audit, dict):
                raise ValueError("CAE cache audit must be a dictionary.")
            enrich_existing_weights(raw, scores, strength)
            metadata["category_cache_hit"] = True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, TypeError, AttributeError, pickle.UnpicklingError) as error:
            warnings.warn(f"Ignoring invalid CAE cache: {error}", RuntimeWarning)
            scores, audit = None, None
    if scores is None:
        scores, audit = build_category_edge_scores(raw, train_edges, n_users, n_items, category_data,
                                                   min_count=min_count, shrinkage=shrinkage,
                                                   memory_budget_mib=memory_budget_mib)
    enriched = enrich_existing_weights(raw, scores, strength)
    if not np.any(scores > 0):
        raise InactiveCategorySignal("CAE has no positive informative category association; use off mode as V8 control.")
    if path and not metadata["category_cache_hit"]:
        hasher = hashlib.sha256()
        _digest_array(hasher, scores)
        atomic_torch_save({"signature": signature, "scores": torch.from_numpy(np.asarray(scores).copy()),
                           "score_fingerprint": hasher.hexdigest(), "audit": audit}, path)
    final, effective_eta = _blend(enriched, cf, eta)
    if not _same_support(final, original):
        raise RuntimeError("CAE changed normalized support; expected frozen-support enrichment.")
    delta = final - original
    metadata.update(audit)
    metadata.update(category_effective_mode="cae", category_active=True, category_bypass=False,
                    category_cache_key=key, category_metadata_fingerprint=fingerprint,
                    category_graph_signature=signature, category_mapping_audit=dict(category_data.audit),
                    effective_eta=effective_eta, category_semantic_nnz=int(raw.nnz), s6_nnz=int(final.nnz),
                    category_support_unchanged=True, category_score_mean=float(np.mean(scores)),
                    category_score_std=float(np.std(scores)),
                    category_score_quantiles=np.quantile(scores, [0, .25, .5, .75, .95, 1]).tolist(),
                    category_active_semantic_mass=float(raw.data[scores > 0].sum(dtype=np.float64) / max(1e-12, raw.data.sum(dtype=np.float64))),
                    relative_operator_delta_vs_v8=_frobenius(delta) / max(1e-12, _frobenius(original)),
                    graph_fingerprint=sparse_fingerprint(final))
    metadata.update(_probe_diagnostics(original, final, probe_dim, probe_seed, beta, num_layers))
    metadata["category_preprocessing_seconds"] = time.perf_counter() - started
    return GraphStateSTAIR6(scipy_to_tensor(final), metadata)
