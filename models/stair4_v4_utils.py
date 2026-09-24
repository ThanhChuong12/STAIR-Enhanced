"""Train-only, candidate-restricted statistics for STAIR4-CSGC.

No item-by-item co-occurrence matrix is constructed. The production intersection
kernel is compiled by Numba; Python loops are confined to small test oracles.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator


def array_hash(value) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "models").glob("stair4_v4*.py")),
             root / "optimizers/stair4_v4_smoother.py", root / "optimizers/AdamW.py",
             root / "models/freerec_compat.py", root / "main_stair4_v4.py"]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def atomic_save(path, payload) -> None:
    """Atomic replacement in the destination filesystem, without partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng() -> dict:
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, pos, gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def sampler_launchers(datapipe):
    """Locate FreeRec launchers in deterministic datapipe traversal order."""
    from torch.utils.data.graph import traverse_dps
    from torch.utils.data.graph_settings import get_all_graph_pipes
    return [pipe for pipe in get_all_graph_pipes(traverse_dps(datapipe))
            if type(pipe).__module__ == "freerec.data.postprocessing.base" and type(pipe).__name__ == "Launcher"]


def capture_sampler(datapipe):
    # FreeRec shuffles this list IN PLACE. RNG alone does not recover next epoch.
    return [{"source": list(pipe.source), "rng": pipe._rng.getstate()} for pipe in sampler_launchers(datapipe)]


def restore_sampler(datapipe, states):
    launchers = sampler_launchers(datapipe)
    if len(launchers) != len(states):
        raise ValueError("FreeRec sampler graph differs from checkpoint")
    for pipe, state in zip(launchers, states):
        if sorted(state["source"]) != sorted(pipe.source):
            raise ValueError("FreeRec sampler population differs from checkpoint")
        pipe.source[:] = state["source"]
        pipe._rng.setstate(state["rng"])


def build_train_csr(edges, num_users: int, num_items: int) -> sp.csr_matrix:
    """Binary, deduplicated train interactions. Caller supplies train edges only."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("train edges must have shape [2, interactions]")
    if edges.size and (edges.min() < 0 or edges[0].max() >= num_users or edges[1].max() >= num_items):
        raise ValueError("train edge ID outside dataset mapping")
    result = sp.csr_matrix((np.ones(edges.shape[1]), edges), shape=(num_users, num_items))
    result.sum_duplicates()
    result.data[:] = 1.0
    result.sort_indices()
    return result


@njit(cache=True)
def _intersections(indptr, indices, pairs, weights):
    c = np.zeros(len(pairs), dtype=np.float64)
    v = np.zeros(len(pairs), dtype=np.float64)
    for edge in range(len(pairs)):
        i, j = pairs[edge]
        left, right = indptr[i], indptr[j]
        while left < indptr[i + 1] and right < indptr[j + 1]:
            u, other = indices[left], indices[right]
            if u == other:
                w = weights[u]
                c[edge] += w
                v[edge] += w * w
                left += 1
                right += 1
            elif u < other:
                left += 1
            else:
                right += 1
    return c, v


@dataclass(frozen=True)
class CandidateStatistics:
    pairs: np.ndarray
    c: np.ndarray
    v: np.ndarray
    q: np.ndarray
    item_degree: np.ndarray
    user_degree: np.ndarray
    association: np.ndarray
    effective_support: np.ndarray


def build_candidate_statistics(train_csr, candidate_pairs, activity_weighting="inverse_degree"):
    train = train_csr.copy().tocsr()
    train.eliminate_zeros()
    train.sum_duplicates()
    train.data[:] = 1.0
    train.sort_indices()
    pairs = np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) and (pairs.min() < 0 or pairs.max() >= train.shape[1] or np.any(pairs[:, 0] >= pairs[:, 1])):
        raise ValueError("candidates must be valid canonical unordered pairs i < j")
    if activity_weighting not in ("inverse_degree", "uniform"):
        raise ValueError("activity_weighting must be inverse_degree or uniform")
    du = np.diff(train.indptr).astype(np.float64)
    item_users = train.T.tocsr()
    item_users.sort_indices()
    di = np.diff(item_users.indptr).astype(np.float64)
    weights = 1.0 / np.maximum(1.0, du) if activity_weighting == "inverse_degree" else np.ones_like(du)
    q = np.asarray(item_users @ weights).ravel()
    c, v = _intersections(item_users.indptr, item_users.indices, pairs, weights)
    denom = np.sqrt(q[pairs[:, 0]] * q[pairs[:, 1]])
    association = np.divide(c, denom, out=np.zeros_like(c), where=denom > 0)
    effective = np.divide(c * c, v, out=np.zeros_like(c), where=v > 0)
    return CandidateStatistics(pairs, c, v, q, di, du, association, effective)


def midrank_cdf(values):
    """Empirical CDF at tie midpoints, including zero-evidence candidates."""
    values = np.asarray(values)
    if not len(values):
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midpoints = (np.cumsum(counts) - 0.5 * counts) / len(values)
    return midpoints[inverse]


def degree_strata(degrees, pairs, bins):
    values = np.log1p(degrees)
    cuts = np.unique(np.quantile(values, np.arange(1, bins) / bins)) if len(values) else np.empty(0)
    labels = np.searchsorted(cuts, values, side="right")
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    return np.minimum(a, b) * (len(cuts) + 1) + np.maximum(a, b)


def calibrated_signal(stats, options):
    pairs = stats.pairs
    n = stats.effective_support
    d = stats.item_degree
    reliability = n / (n + options.support_tau)
    reliability *= np.sqrt(d[pairs[:, 0]] / (d[pairs[:, 0]] + options.degree_tau)
                           * d[pairs[:, 1]] / (d[pairs[:, 1]] + options.degree_tau))
    if options.ablation_id == "V4-NS":
        reliability = (stats.c > 0).astype(float)
    strata = degree_strata(d, pairs, options.degree_bins)
    ranks = midrank_cdf(stats.association)
    if options.ablation_id != "V4-ND":
        for label in np.unique(strata):
            mask = strata == label
            if mask.sum() >= options.min_stratum_edges:
                ranks[mask] = midrank_cdf(stats.association[mask])
    h = reliability * (2 * ranks - 1)
    if options.ablation_id == "V4-SH":
        rng = np.random.default_rng(options.shuffle_seed)
        for label in np.unique(strata):
            mask = strata == label
            h[mask] = rng.permutation(h[mask])
    return h, reliability, strata
