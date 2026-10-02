"""Train-only, candidate-restricted edge calibration for STAIR's BSC graph.

No dense item-item matrix or full R.T @ R is constructed here. Raw modality
weights must be supplied before degree normalization. The off path returns the
original operator object to avoid changing baseline floating-point arithmetic.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
import torch


@dataclass
class GraphCalibration:
    operator: torch.Tensor
    support_scores: torch.Tensor
    item_degrees: torch.Tensor
    common_counts: torch.Tensor
    metadata: dict


def _coo(graph: torch.Tensor) -> torch.Tensor:
    if graph.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise ValueError("Expected a sparse COO/CSR graph.")
    graph = graph.detach().to(device="cpu").to_sparse_coo().coalesce()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("The graph must be square.")
    if not torch.isfinite(graph.values()).all() or (graph.values() < 0).any():
        raise ValueError("Graph weights must be finite and nonnegative.")
    transpose = graph.transpose(0, 1).coalesce()
    if not torch.equal(graph.indices(), transpose.indices()) or not torch.allclose(
        graph.values(), transpose.values(), rtol=1e-6, atol=1e-8
    ):
        raise ValueError("The graph must be symmetric before calibration.")
    return graph


def normalize_graph(raw: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalization; zero-degree vertices remain isolated."""
    graph = _coo(raw)
    rows, cols = graph.indices()
    degree = torch.zeros(graph.shape[0], dtype=graph.dtype)
    degree.scatter_add_(0, rows, graph.values())
    inv = torch.zeros_like(degree)
    valid = degree > 0
    inv[valid] = degree[valid].rsqrt()
    values = graph.values() * inv[rows] * inv[cols]
    return torch.sparse_coo_tensor(graph.indices(), values, graph.shape).coalesce().to_sparse_csr()


def _binary_edges(edges: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("train_edges must be an int64 tensor of shape [2, E].")
    edges = edges.detach().cpu()
    if num_users <= 0 or num_items <= 0:
        raise ValueError("Node counts must be positive.")
    if edges.numel() and ((edges < 0).any() or (edges[0] >= num_users).any()
                         or (edges[1] >= num_items).any()):
        raise ValueError("Train interaction IDs are outside the declared node range.")
    keys = torch.unique(edges[0]*num_items + edges[1], sorted=True)
    return torch.stack((keys//num_items, keys%num_items))


def candidate_statistics(raw: torch.Tensor, train_edges: torch.Tensor, num_users: int,
                         shrinkage: float = 5.0):
    """Return (q, train item degrees, common-user counts), aligned to COO edges.

    Intersect sorted item-user lists for each undirected candidate. Intersection
    uses binary search in the longer list; cost depends on actual degrees.
    """
    if not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("evidence_shrinkage must be finite and positive.")
    graph = _coo(raw)
    n_items = graph.shape[0]
    edges = _binary_edges(train_edges, num_users, n_items).numpy()
    order = np.lexsort((edges[0], edges[1]))
    users, items = edges[0, order], edges[1, order]
    degrees = np.bincount(items, minlength=n_items)
    offsets = np.concatenate(([0], np.cumsum(degrees)))
    row, col = graph.indices().numpy()
    counts = np.zeros(len(row), dtype=np.int64)
    keys = row * n_items + col
    for pos in np.flatnonzero(row <= col):
        i, j = row[pos], col[pos]
        left = users[offsets[i]:offsets[i + 1]]
        right = users[offsets[j]:offsets[j + 1]]
        if len(left) > len(right):
            left, right = right, left
        if len(left) and len(right):
            indexes = np.searchsorted(right, left)
            valid = indexes < len(right)
            count = np.count_nonzero(right[indexes[valid]] == left[valid])
        else:
            count = 0
        counts[pos] = count
        reverse = np.searchsorted(keys, j * n_items + i)
        counts[reverse] = count
    denom = np.sqrt(degrees[row].astype(np.float64) * degrees[col])
    ochiai = np.divide(counts, denom, out=np.zeros(len(row), dtype=np.float64), where=denom > 0)
    scores = counts / (counts + shrinkage) * ochiai
    return (torch.from_numpy(scores).to(graph.dtype), torch.from_numpy(degrees),
            torch.from_numpy(counts))


def _placebo(scores, raw, degrees, seed):
    """Permute undirected edge scores within log2 endpoint-degree strata."""
    row, col = raw.indices()
    unique = torch.nonzero(row <= col).flatten()
    bins = torch.floor(torch.log2(degrees.float().clamp_min(1))).long()
    low = torch.minimum(bins[row[unique]], bins[col[unique]])
    high = torch.maximum(bins[row[unique]], bins[col[unique]])
    strata = low * (int(bins.max()) + 1) + high
    result = scores.clone()
    generator = torch.Generator().manual_seed(seed)
    shuffled = 0
    for group in torch.unique(strata):
        indexes = unique[strata == group]
        perm = torch.randperm(len(indexes), generator=generator)
        result[indexes] = scores[indexes[perm]]
        shuffled += int((perm != torch.arange(len(indexes))).sum())
    keys = row * raw.shape[0] + col
    reverse = torch.searchsorted(keys, col[unique] * raw.shape[0] + row[unique])
    result[reverse] = result[unique]
    return result, shuffled


def _fingerprint(raw, baseline, edges, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for tensor in (raw.indices(), raw.values(), baseline.indices(), baseline.values(), edges):
        array = tensor.contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_calibrated_graph(raw_graph, baseline_graph, train_edges, num_users,
                           strength=.25, mix=.25, shrinkage=5., placebo=False,
                           seed=1, cache_dir: Optional[str] = None) -> GraphCalibration:
    """Build a frozen sparse blended operator and optionally cache its state."""
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("edge_strength must be >= 0 and edge_mix must be in [0, 1].")
    if not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("evidence_shrinkage must be positive.")
    if strength == 0 or mix == 0:
        empty = torch.empty(0)
        return GraphCalibration(baseline_graph, empty, empty.long(), empty.long(),
                                {"active": False, "strength": strength, "mix": mix})
    raw, baseline = _coo(raw_graph), _coo(baseline_graph)
    if raw.shape != baseline.shape or not torch.equal(raw.indices(), baseline.indices()):
        raise ValueError("Raw and baseline graphs must have matching support.")
    expected = normalize_graph(raw).to_sparse_coo().coalesce()
    if not torch.allclose(expected.values(), baseline.values(), rtol=2e-5, atol=1e-7):
        raise ValueError("baseline_graph is not the normalized supplied raw graph.")
    edges = _binary_edges(train_edges, num_users, raw.shape[0])
    settings = dict(version=2, strength=strength, mix=mix, shrinkage=shrinkage,
                    placebo=placebo, seed=seed, num_users=num_users, num_items=raw.shape[0],
                    torch=str(torch.__version__))
    key = _fingerprint(raw, baseline, edges, settings)
    cache = Path(cache_dir) / f"c_het_{key}.pt" if cache_dir else None
    if cache and cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload["metadata"]["fingerprint"] != key:
            raise ValueError("Graph cache fingerprint mismatch.")
        payload["metadata"]["cache_hit"] = True
        return GraphCalibration(**payload)
    scores, degrees, counts = candidate_statistics(raw, edges, num_users, shrinkage)
    permuted = 0
    if placebo:
        scores, permuted = _placebo(scores, raw, degrees, seed)
    boosted = torch.sparse_coo_tensor(raw.indices(), raw.values() * (1 + strength * scores), raw.shape)
    plus = normalize_graph(boosted).to_sparse_coo().coalesce()
    values = (1 - mix) * baseline.values() + mix * plus.values()
    operator = torch.sparse_coo_tensor(raw.indices(), values, raw.shape).coalesce().to_sparse_csr()
    metadata = dict(settings, active=True, fingerprint=key, cache_hit=False, nnz=raw._nnz(),
                    nonzero_score_fraction=float((scores > 0).float().mean()) if len(scores) else 0.,
                    score_mean=float(scores.mean()) if len(scores) else 0.,
                    score_max=float(scores.max()) if len(scores) else 0., placebo_permuted=permuted)
    probes = torch.randn((raw.shape[0], 4), generator=torch.Generator().manual_seed(0), dtype=raw.dtype)
    baseline_action = baseline @ probes
    metadata["relative_operator_probe_delta"] = float(
        ((operator @ probes)-baseline_action).norm()/baseline_action.norm().clamp_min(1e-12))
    result = GraphCalibration(operator, scores, degrees, counts, metadata)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".c_het_", suffix=".pt", dir=cache.parent)
        os.close(fd)
        try:
            torch.save(vars(result), temporary)
            os.replace(temporary, cache)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return result
