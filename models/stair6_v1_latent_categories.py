"""Static latent content groups for LCAE; these are not Amazon taxonomy labels.

CPU-only, blocked Gaussian projection and spherical Lloyd clustering. Local
NumPy generators never change training RNG. Only hard sparse memberships enter
the binary user-group incidence: dense positive soft assignments are invalid.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import pickle
from pathlib import Path
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_utils import atomic_torch_save, file_sha256
from models.stair6_v1_categories import CategoryData

ALGORITHM = "raw-normalized-gaussian-concat-spherical-hard-v1"
WEIGHTS = (5.0 / 6.0, 1.0 / 6.0)
EPS = 1e-8


@dataclass(frozen=True)
class LatentCategoryState:
    data: CategoryData
    assignments: np.ndarray
    centers: np.ndarray
    metadata: dict


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _feature_arrays(features):
    if len(features) != 2:
        raise ValueError("Latent groups require text and visual views in that order")
    arrays = []
    for feature in features:
        if isinstance(feature, torch.Tensor):
            if feature.device.type != "cpu":
                raise ValueError("Latent preprocessing requires CPU feature tensors")
            feature = feature.detach().numpy()
        array = np.asarray(feature)
        if array.dtype != np.float32 or array.ndim != 2 or min(array.shape) < 1:
            raise ValueError("Latent features must be FP32 [items, feature_dim] arrays")
        if not array.flags.c_contiguous:
            raise ValueError("Latent features must be contiguous to avoid a hidden catalog-size copy")
        arrays.append(array)
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError("Both modalities must use the same processed item order")
    return arrays


def _workspace(n, dimensions, clusters, projection_dim, chunk_size, budget_mib):
    width = 2 * projection_dim
    # Resident fused features, labels/scores, sparse accumulation and centers.
    base = 4 * n * width + 40 * n + 12 * clusters * width
    base += 4 * max(dimensions) * projection_dim
    # Conservative block allowance; caller's raw feature arrays are excluded.
    per_row = 12 * max(dimensions) + 16 * width + 12 * clusters + 128
    available = int(budget_mib * 2**20) - base
    block = min(chunk_size, available // per_row)
    if block < 1:
        raise MemoryError("Latent CPU workspace exceeds budget; reduce projection_dim/clusters or increase budget")
    return int(block), int(base + block * per_row)


def _project_views(arrays, projection_dim, block, seed):
    n = arrays[0].shape[0]
    fused = np.zeros((n, 2 * projection_dim), dtype=np.float32)
    valid_views = np.zeros((n, 2), dtype=bool)
    for modal, (features, weight) in enumerate(zip(arrays, WEIGHTS)):
        generator = np.random.default_rng(np.random.SeedSequence([seed, modal, 1409]))
        projection = generator.standard_normal((features.shape[1], projection_dim), dtype=np.float32)
        projection *= np.float32(1 / math.sqrt(projection_dim))
        for start in range(0, n, block):
            stop = min(start + block, n)
            values = features[start:stop]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite latent modality features")
            # Scale before L2 normalization to avoid overflow of finite FP32.
            scale = np.max(np.abs(values), axis=1, keepdims=True)
            normalized = values / np.maximum(scale, EPS)
            norm = np.linalg.norm(normalized, axis=1, keepdims=True)
            normalized /= np.maximum(norm, EPS)
            projected = normalized @ projection
            projected_norm = np.linalg.norm(projected, axis=1, keepdims=True)
            valid = (scale[:, 0] > EPS) & (projected_norm[:, 0] > EPS)
            projected /= np.maximum(projected_norm, EPS)
            projected[~valid] = 0
            fused[start:stop, modal * projection_dim:(modal + 1) * projection_dim] = projected * math.sqrt(weight)
            valid_views[start:stop, modal] = valid
        del projection
    norm = np.linalg.norm(fused, axis=1, keepdims=True)
    fused /= np.maximum(norm, EPS)
    return fused, valid_views


def _initialize(fused, valid_ids, clusters, block, seed):
    """Cosine-distance k-means++ with O(N), rather than NxN, state."""
    generator = np.random.default_rng(np.random.SeedSequence([seed, 1907]))
    centers = np.empty((clusters, fused.shape[1]), dtype=np.float32)
    chosen = np.zeros(len(fused), dtype=bool)
    distance = np.zeros(len(fused), dtype=np.float64)
    distance[valid_ids] = 2.0
    item = int(generator.choice(valid_ids))
    for cluster in range(clusters):
        centers[cluster] = fused[item]
        chosen[item] = True
        for start in range(0, len(fused), block):
            stop = min(start + block, len(fused))
            current = np.maximum(0.0, 2.0 - 2.0 * (fused[start:stop] @ centers[cluster]))
            distance[start:stop] = np.minimum(distance[start:stop], current)
        distance[chosen] = 0
        total = float(distance.sum())
        if cluster + 1 < clusters:
            if total > 1e-10:
                item = int(generator.choice(len(fused), p=distance / total))
            else:
                remaining = valid_ids[~chosen[valid_ids]]
                item = int(remaining[0])
    return centers


def _assign(fused, centers, valid, block):
    labels = np.full(len(fused), -1, dtype=np.int64)
    scores = np.zeros(len(fused), dtype=np.float32)
    margins = np.zeros(len(fused), dtype=np.float32)
    for start in range(0, len(fused), block):
        stop = min(start + block, len(fused))
        similarity = fused[start:stop] @ centers.T
        ids = similarity.argmax(axis=1)  # Ties use the smallest cluster ID.
        best = similarity[np.arange(stop - start), ids]
        second = np.partition(similarity, -2, axis=1)[:, -2]
        labels[start:stop] = np.where(valid[start:stop], ids, -1)
        scores[start:stop] = best
        margins[start:stop] = best - second
    return labels, scores, margins


def spherical_kmeans(fused, valid, *, clusters, iterations, block, seed):
    """Full blocked spherical Lloyd passes; no Euclidean MiniBatchKMeans claim."""
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size < clusters:
        raise ValueError("Fewer informative items than requested latent clusters; lower latent_clusters explicitly")
    centers = _initialize(fused, valid_ids, clusters, block, seed)
    previous = None
    repairs, objectives = 0, []
    for iteration in range(iterations):
        labels, scores, margins = _assign(fused, centers, valid, block)
        objectives.append(float(scores[valid].mean()))
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        accumulator = sp.csr_matrix((np.ones(valid_ids.size, dtype=np.float32),
                                     (labels[valid_ids], valid_ids)), shape=(clusters, len(fused)))
        sums = accumulator @ fused
        norms = np.linalg.norm(sums, axis=1)
        good = norms > EPS
        centers[good] = sums[good] / norms[good, None]
        # Deterministic worst-fit reseeding. Duplicate points can still leave
        # fewer occupied clusters; report actual occupancy instead of inventing it.
        repair_ids = valid_ids[np.argsort(scores[valid_ids], kind="stable")]
        for offset, cluster in enumerate(np.flatnonzero(~good)):
            centers[cluster] = fused[repair_ids[offset % repair_ids.size]]
            repairs += 1
    labels, scores, margins = _assign(fused, centers, valid, block)
    counts = np.bincount(labels[valid], minlength=clusters)
    if np.count_nonzero(counts) < 2:
        raise ValueError("Latent clustering collapsed to one occupied group; no informative partition")
    probabilities = counts[counts > 0] / valid_ids.size
    diagnostic = {
        "occupied_clusters": int(np.count_nonzero(counts)), "cluster_counts": counts.tolist(),
        "largest_cluster_fraction": float(counts.max() / valid_ids.size),
        "normalized_cluster_entropy": float(-(probabilities * np.log(probabilities)).sum() / math.log(clusters)),
        "iterations_executed": iteration + 1, "empty_or_zero_centroid_repairs": repairs,
        "mean_assigned_cosine": float(scores[valid].mean()),
        "assignment_margin_quantiles": np.quantile(margins[valid], [0, .25, .5, .75, .95, 1]).tolist(),
        "mean_cosine_history": objectives,
    }
    return labels, centers, diagnostic


def build_latent_category_data(features, *, dataset, feature_manifest, split_manifest,
                              clusters=50, projection_dim=64, iterations=20, chunk_size=2048,
                              seed=2718, memory_budget_mib=128.0, cache_dir=None):
    """Generate hashed one-hot groups in native feature row order, without ASINs."""
    started = time.perf_counter()
    for name, value, minimum in (("clusters", clusters, 2), ("projection_dim", projection_dim, 1),
                                 ("iterations", iterations, 1), ("chunk_size", chunk_size, 1), ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(memory_budget_mib, bool) or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("Latent CPU workspace budget must be positive and finite")
    arrays = _feature_arrays(features)
    n = len(arrays[0])
    if clusters > n:
        raise ValueError("Requested latent clusters exceed catalog size")
    block, estimated_bytes = _workspace(n, [v.shape[1] for v in arrays], clusters,
                                         projection_dim, chunk_size, memory_budget_mib)
    if not dataset or not feature_manifest or not split_manifest:
        raise ValueError("Latent artifacts require dataset, feature and split provenance")
    signature = {
        "algorithm": ALGORITHM, "dataset": dataset, "n_items": n,
        "raw_arrays": [_array_digest(v) for v in arrays], "feature_files": feature_manifest,
        "splits": split_manifest, "clusters": clusters, "projection_dim": projection_dim,
        "iterations": iterations, "seed": seed, "effective_block": block,
        "modality_weights": list(WEIGHTS), "numpy": np.__version__,
        "torch": str(torch.__version__), "torch_cpu_threads": torch.get_num_threads(),
        "source": file_sha256(Path(__file__)),
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"latent_{key}.pt" if cache_dir else None
    labels = centers = diagnostic = None
    cache_hit = False
    if path and path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            labels, centers = payload["assignments"].numpy(), payload["centers"].numpy()
            if (payload["signature"] != signature or labels.shape != (n,) or labels.dtype != np.int64
                    or centers.shape != (clusters, 2 * projection_dim) or centers.dtype != np.float32
                    or np.any(labels < -1) or np.any(labels >= clusters) or not np.isfinite(centers).all()
                    or not np.allclose(np.linalg.norm(centers, axis=1), 1, rtol=1e-5, atol=1e-5)
                    or payload["assignments_digest"] != _array_digest(labels)
                    or payload["centers_digest"] != _array_digest(centers)):
                raise ValueError("Invalid latent cache payload")
            diagnostic = payload["diagnostic"]
            if not isinstance(diagnostic, dict) or np.unique(labels[labels >= 0]).size < 2:
                raise ValueError("Inactive or invalid latent cache diagnostics")
            cache_hit = True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, TypeError, AttributeError,
                pickle.UnpicklingError) as error:
            warnings.warn(f"Ignoring invalid latent cache: {error}", RuntimeWarning)
            labels = centers = diagnostic = None
    if labels is None:
        fused, valid_views = _project_views(arrays, projection_dim, block, seed)
        valid = valid_views.any(axis=1)
        labels, centers, diagnostic = spherical_kmeans(fused, valid, clusters=clusters,
            iterations=iterations, block=block, seed=seed)
        diagnostic.update(informative_items=int(valid.sum()), informative_coverage=float(valid.mean()),
                          valid_items_by_modality=valid_views.sum(axis=0).tolist())
        if path:
            atomic_torch_save({"signature": signature, "assignments": torch.from_numpy(labels.copy()),
                "centers": torch.from_numpy(centers.copy()), "diagnostic": diagnostic,
                "assignments_digest": _array_digest(labels), "centers_digest": _array_digest(centers)}, path)
        del fused, valid_views
    valid_ids = np.flatnonzero(labels >= 0)
    memberships = sp.csr_matrix((np.ones(valid_ids.size, dtype=np.float32),
                                (valid_ids, labels[valid_ids])), shape=(n, clusters))
    provenance = {"schema_version": 1, "source_kind": "latent_content_groups", "dataset": dataset,
                  "n_items": n, "baseline": {"features": {r["file"]: r["sha256"] for r in feature_manifest},
                                             "splits": split_manifest}, "clustering": signature,
                  "assignment_sha256": _array_digest(labels), "center_sha256": _array_digest(centers)}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    audit = {**diagnostic, "n_items": n, "n_categories": clusters,
             "source_kind": "latent_content_groups", "membership": "hard one-hot; invalid content rows zero",
             "ground_truth_taxonomy": False, "asin_mapping_used": False}
    data = CategoryData(memberships, tuple(("latent", f"cluster-{i}") for i in range(clusters)),
                        (), audit, fingerprint, provenance)
    return LatentCategoryState(data, labels, centers, {"cache_hit": cache_hit,
        "cache_key": key, "preprocessing_seconds": time.perf_counter() - started,
        "workspace_estimate_bytes": estimated_bytes, "effective_block_size": block, **diagnostic})
