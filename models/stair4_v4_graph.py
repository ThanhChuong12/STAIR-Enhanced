"""Static, support-preserving CSGC graph construction and content-addressed cache."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, to_undirected

from .stair4_v4_utils import (
    array_hash, atomic_save, build_candidate_statistics, build_train_csr,
    calibrated_signal, identity_hash, source_hash,
)


@dataclass(frozen=True)
class CSGCOptions:
    alpha: float = 0.25
    edge_epsilon: float = 0.5
    support_tau: float = 5.0
    degree_tau: float = 10.0
    degree_bins: int = 4
    min_stratum_edges: int = 200
    activity_weighting: str = "inverse_degree"
    ablation_id: str = "V4-C"
    shuffle_seed: int = 1
    identity_mix: float = 0.0
    knn_block_size: int = 256

    def __post_init__(self):
        for name in ("alpha", "edge_epsilon", "support_tau", "degree_tau", "identity_mix"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.alpha <= 1 or not 0 <= self.edge_epsilon < 1:
            raise ValueError("alpha in [0,1] and edge_epsilon in [0,1) required")
        if min(self.support_tau, self.degree_tau) <= 0:
            raise ValueError("positive shrinkage constants required")
        for name in ("degree_bins", "min_stratum_edges", "knn_block_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.activity_weighting not in ("inverse_degree", "uniform"):
            raise ValueError("invalid activity weighting")
        if self.ablation_id not in ("V4-B1", "V4-C", "V4-NS", "V4-ND", "V4-NA", "V4-SH", "V4-SM"):
            raise ValueError("V4-B0 runs original main.py; unknown v4 ablation")
        if self.ablation_id in ("V4-B1", "V4-SM") and self.alpha != 0:
            raise ValueError("V4-B1 and V4-SM require alpha=0")
        if not 0 <= self.identity_mix <= 1 or (self.identity_mix and self.ablation_id != "V4-SM"):
            raise ValueError("identity_mix is only permitted for the V4-SM control")
        if self.ablation_id == "V4-SM" and self.identity_mix == 0:
            raise ValueError("V4-SM requires a positive identity_mix")
        if self.ablation_id == "V4-NA" and self.activity_weighting != "uniform":
            raise ValueError("V4-NA requires activity_weighting=uniform")

    @classmethod
    def from_config(cls, cfg):
        return cls(**{key: getattr(cfg, key) for key in cls.__dataclass_fields__ if hasattr(cfg, key)})


@torch.no_grad()
def exact_cosine_knn(features, k, block_size=256):
    """Baseline top-k cosine rule, with bounded temporary memory.

    Ties use torch.topk's backend order. Record neighbor hashes; compare baseline
    on the same backend. Blocking need not be bitwise identical across BLAS/GPU.
    """
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("finite two-dimensional modality features required")
    n = len(features)
    if not 0 < k < n or block_size < 1:
        raise ValueError("0 < k < item count and block_size > 0 required")
    normalized = F.normalize(features.float(), dim=-1)
    chunks = []
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        sim = normalized[start:stop] @ normalized.T
        sim[torch.arange(stop - start, device=sim.device), torch.arange(start, stop, device=sim.device)] = -10.0
        chunks.append(sim.topk(k, dim=1).indices)
    return torch.cat(chunks)


@torch.no_grad()
def build_baseline_raw_graph(modalities, neighbors, block_size=256, cache_dir=None):
    if not modalities or len(modalities) != len(neighbors):
        raise ValueError("one positive neighbor count per modality required")
    n = len(modalities[0])
    hashes = {f"features_{m}": array_hash(f) for m, f in enumerate(modalities)}
    cache_identity = {"source": source_hash(), "schema": 1, "features": hashes.copy(),
                      "neighbors": list(neighbors), "block_size": block_size,
                      "torch": str(torch.__version__), "device": str(modalities[0].device),
                      "tie_policy": "torch.topk"}
    cache = Path(cache_dir) / f"raw_{identity_hash(cache_identity)}.pt" if cache_dir else None
    if cache and cache.exists():
        saved = torch.load(cache, map_location=modalities[0].device, weights_only=True)
        if saved["identity"] != cache_identity:
            raise ValueError("Raw graph cache identity mismatch")
        raw = torch.sparse_csr_tensor(saved["crow"], saved["col"], saved["values"], (n, n), check_invariants=True)
        if graph_hash(raw) != saved["checksum"]:
            raise ValueError("Raw graph cache checksum mismatch")
        if any(array_hash(nn) != saved["hashes"][f"neighbors_{m}"] for m, nn in enumerate(saved["neighbors"])):
            raise ValueError("Raw neighbor cache checksum mismatch")
        return raw, saved["hashes"]
    indices, raw_neighbors = [], []
    for m, (features, k) in enumerate(zip(modalities, neighbors)):
        if len(features) != n:
            raise ValueError("modality item mappings differ")
        nn = exact_cosine_knn(features, k, block_size)
        raw_neighbors.append(nn.cpu())
        hashes[f"features_{m}"] = array_hash(features)
        hashes[f"neighbors_{m}"] = array_hash(nn)
        row = torch.arange(n, device=nn.device)[:, None].expand_as(nn)
        indices.append(torch.stack((row.flatten(), nn.flatten())))
    indices = torch.cat(indices, dim=1)
    weights = torch.ones(indices.shape[1], dtype=torch.float32, device=indices.device)
    indices, weights = coalesce(indices, weights, num_nodes=n, reduce="sum")
    indices, weights = to_undirected(indices, weights, num_nodes=n, reduce="max")
    raw = torch.sparse_coo_tensor(indices, weights, (n, n)).coalesce().to_sparse_csr()
    if cache:
        atomic_save(cache, {"identity": cache_identity, "crow": raw.crow_indices().cpu(),
            "col": raw.col_indices().cpu(), "values": raw.values().cpu(), "neighbors": raw_neighbors,
            "hashes": hashes, "checksum": graph_hash(raw)})
    return raw, hashes


@torch.no_grad()
def symmetric_normalize(raw):
    coo = raw.to_sparse_coo().coalesce()
    row, col = coo.indices()
    degree = torch.zeros(raw.shape[0], device=raw.device, dtype=raw.dtype)
    degree.scatter_add_(0, row, coo.values())
    inv = degree.pow(-0.5)
    inv.masked_fill_(torch.isinf(inv), 0)
    values = inv[row] * coo.values() * inv[col]
    return torch.sparse_coo_tensor(coo.indices(), values, raw.shape).coalesce().to_sparse_csr()


def graph_hash(graph):
    return identity_hash({"shape": list(graph.shape), "crow": array_hash(graph.crow_indices()),
                          "col": array_hash(graph.col_indices()), "values": array_hash(graph.values())})


@dataclass(frozen=True)
class GraphBundle:
    operator: torch.Tensor
    baseline: torch.Tensor
    diagnostics: dict
    identity: dict


@torch.no_grad()
def build_operator(raw, train_edges, num_users, options, *, baseline=None, data_identity=None, cache_dir=None):
    """Calibrate only existing off-diagonal edges, then mix normalized operators.

    `baseline` may be the exact FreeRec-normalized graph for strict Gate 0.
    Cached values never replace current indices or bypass content validation.
    """
    if raw.layout != torch.sparse_csr or raw.shape[0] != raw.shape[1]:
        raise ValueError("raw graph must be square CSR")
    if raw.requires_grad or not torch.isfinite(raw.values()).all() or (raw.values() <= 0).any():
        raise ValueError("raw graph must be static with positive finite weights")
    coo = raw.to_sparse_coo().coalesce()
    row, col = coo.indices().cpu().numpy()
    reverse = torch.sparse_coo_tensor(coo.indices().flip(0), coo.values(), raw.shape).coalesce()
    if not torch.equal(coo.indices(), reverse.indices()) or not torch.equal(coo.values(), reverse.values()):
        raise ValueError("raw graph must be symmetric")
    s0 = symmetric_normalize(raw) if baseline is None else baseline
    if not torch.equal(s0.crow_indices(), raw.crow_indices()) or not torch.equal(s0.col_indices(), raw.col_indices()):
        raise ValueError("baseline and raw graph support/order differ")
    train = build_train_csr(train_edges, num_users, raw.shape[0])
    identity = {"schema": 1, "source": source_hash(), "options": asdict(options),
                "raw": graph_hash(raw), "baseline": graph_hash(s0),
                "train_crow": array_hash(train.indptr), "train_col": array_hash(train.indices),
                "train_shape": list(train.shape), "data": data_identity or {},
                "torch": str(torch.__version__)}
    # Do not intersect, rank, cache, shuffle or consume RNG in the baseline path.
    if options.alpha == 0:
        return GraphBundle(s0, s0, {"gate0": True, "nnz": raw._nnz()}, identity)
    cache = Path(cache_dir) / f"csgc_{identity_hash(identity)}.pt" if cache_dir else None
    if cache and cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["checksum"] != array_hash(saved["values"]):
            raise ValueError("CSGC cache identity/checksum mismatch")
        values = saved["values"].to(raw.device)
        if values.shape != raw.values().shape or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("invalid cached operator values")
        operator = torch.sparse_csr_tensor(raw.crow_indices(), raw.col_indices(), values, raw.shape)
        return GraphBundle(operator, s0, {**saved["diagnostics"], "cache_hit": True}, identity)
    mask = row < col
    pairs = np.stack((row[mask], col[mask]), axis=1)
    stats = build_candidate_statistics(train, pairs, options.activity_weighting)
    h, reliability, strata = calibrated_signal(stats, options)
    # Every directed off-diagonal edge finds its canonical counterpart.
    keys = pairs[:, 0] * raw.shape[0] + pairs[:, 1]
    all_keys = np.minimum(row, col) * raw.shape[0] + np.maximum(row, col)
    multipliers = np.ones(len(row), dtype=np.float64)
    off = row != col
    multipliers[off] += options.edge_epsilon * h[np.searchsorted(keys, all_keys[off])]
    wr_values = raw.values() * torch.as_tensor(multipliers, device=raw.device, dtype=raw.dtype)
    wr = torch.sparse_csr_tensor(raw.crow_indices(), raw.col_indices(), wr_values, raw.shape)
    sr = symmetric_normalize(wr)
    values = (1 - options.alpha) * s0.values() + options.alpha * sr.values()
    operator = torch.sparse_csr_tensor(raw.crow_indices(), raw.col_indices(), values, raw.shape)
    def quantiles(x):
        return np.quantile(x, [0, .25, .5, .75, 1]).tolist() if len(x) else []
    diagnostics = {"gate0": False, "cache_hit": False, "nnz": raw._nnz(),
        "candidates": len(pairs), "evidence_fraction": float(np.mean(stats.c > 0)) if len(pairs) else 0.,
        "support_quantiles": quantiles(stats.effective_support), "reliability_quantiles": quantiles(reliability),
        "supported_edge_support_quantiles": quantiles(stats.effective_support[stats.c > 0]),
        "supported_edge_reliability_quantiles": quantiles(reliability[stats.c > 0]),
        "positive_signal_edges": int(np.sum(h > 0)), "negative_signal_edges": int(np.sum(h < 0)),
        "signal_quantiles": quantiles(h), "multiplier_quantiles": quantiles(multipliers),
        "strata_count": len(np.unique(strata)),
        "operator_relative_frobenius": float(torch.linalg.vector_norm(values - s0.values()) /
                                               torch.linalg.vector_norm(s0.values()).clamp_min(1e-30))}
    # Fixed local probe does not consume the training RNG stream.
    generator = torch.Generator(device=raw.device).manual_seed(719)
    probe = torch.randn((raw.shape[0], 8), generator=generator, device=raw.device, dtype=raw.dtype)
    base_action = s0 @ probe
    diagnostics["operator_relative_probe_action"] = float(torch.linalg.vector_norm(operator @ probe - base_action) /
                                                        torch.linalg.vector_norm(base_action).clamp_min(1e-30))
    if cache:
        atomic_save(cache, {"identity": identity, "values": values.cpu(), "checksum": array_hash(values),
                            "diagnostics": diagnostics})
    return GraphBundle(operator, s0, diagnostics, identity)
