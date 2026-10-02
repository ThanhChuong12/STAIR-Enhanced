"""Train-only, static Candidate Support Expansion for optimizer-side BSC.

Exact int64 co-occurrence counts, deterministic sparse top-k union, separately
normalized operators, explicit isolated identity fallback and content-addressed
CF caching. No dense item-item product or per-step graph computation.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_utils import atomic_torch_save

ARMS = ("B0", "N0", "C0", "N-CSE", "N-CSE-placebo", "v4b")
CACHE_VERSION = 2


@dataclass
class GraphStateV4:
    operator: torch.Tensor
    metadata: dict


def binary_train_keys(train_edges, n_users, n_items):
    """Validate IDs before flattening; deduplicate all train pairs, not a prefix."""
    if n_users <= 0 or n_items <= 0 or n_users * n_items > np.iinfo(np.int64).max:
        raise ValueError("Invalid entity counts or int64 pair-key overflow.")
    if train_edges.ndim != 2 or train_edges.shape[0] != 2 or train_edges.dtype not in (torch.int32, torch.int64):
        raise ValueError("train_edges must be a [2, E] integer tensor.")
    edges = train_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges[0].max() >= n_users or edges[1].max() >= n_items):
        raise ValueError("Train interaction IDs are outside the declared entity counts.")
    return np.unique(edges[0] * n_items + edges[1])


def _digest_array(hasher, array):
    array = np.ascontiguousarray(array)
    hasher.update(str((array.shape, array.dtype.str)).encode())
    hasher.update(memoryview(array).cast("B"))


def sparse_fingerprint(matrix):
    """Hash canonical shape, indices and values without materializing dense data."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    h = hashlib.sha256(str(matrix.shape).encode())
    for array in (matrix.indptr.astype(np.int64), matrix.indices.astype(np.int64), matrix.data):
        _digest_array(h, array)
    return h.hexdigest()


def tensor_to_scipy(matrix):
    matrix = matrix.detach().cpu()
    if matrix.layout == torch.sparse_csr:
        return sp.csr_matrix((matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()), shape=tuple(matrix.shape))
    if matrix.layout == torch.sparse_coo:
        matrix = matrix.coalesce()
        indices = matrix.indices().numpy()
        return sp.csr_matrix((matrix.values().numpy(), (indices[0], indices[1])), shape=tuple(matrix.shape))
    raise ValueError("Expected a sparse COO or CSR graph.")


def scipy_to_tensor(matrix):
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(matrix.indptr.astype(np.int64)),
                                  torch.from_numpy(matrix.indices.astype(np.int64)),
                                  torch.from_numpy(matrix.data), size=matrix.shape)


def compute_evidence_score(c, n_i, n_j, t=5.0):
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Evidence shrinkage must be finite and positive.")
    if n_i <= 0 or n_j <= 0 or c <= 0:
        return 0.0
    return float(c / (c + t) * c / math.sqrt(n_i * n_j))


def _deterministic_topk(ids, scores, k):
    """Partition in linear time; sort only selected entries and boundary ties."""
    if len(scores) > k:
        boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
        above = np.flatnonzero(scores > boundary)
        ties = np.flatnonzero(scores == boundary)
        needed = k - len(above)
        if len(ties) > needed:
            selected_ties = np.argpartition(ids[ties], needed - 1)[:needed]
            ties = ties[selected_ties]
        chosen = np.concatenate((above, ties[:needed]))
        ids, scores = ids[chosen], scores[chosen]
    order = np.lexsort((ids, -scores))
    return ids[order], scores[order]


def build_candidate_support_graph(train_edges, n_items, n_users, k_cf=5, c_min=2,
                                  t_shrinkage=5.0, block_size=64, memory_budget_mib=128):
    """Exact binary R.T@R in bounded CSR blocks; no history cap or approximation.

    The conservative dense-block bound budgets result and workspace before SpGEMM.
    Blocks shrink to one row when needed. The budget covers product blocks, not
    the input R, retained O(n_items*k_cf) graph, or SciPy allocator overhead.
    """
    if k_cf < 0 or int(k_cf) != k_cf or c_min < 1 or int(c_min) != c_min:
        raise ValueError("Require integer k_cf >= 0 and c_min >= 1.")
    if block_size < 1 or int(block_size) != block_size or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("Require a positive integer block size and finite memory budget.")
    compute_evidence_score(0, 0, 0, t_shrinkage)
    flat = binary_train_keys(train_edges, n_users, n_items)
    if k_cf == 0 or not len(flat):
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    # Integer multiplication is exact even when a pair exceeds FP32's 2**24 range.
    R = sp.csr_matrix((np.ones(len(flat), dtype=np.int64), (flat // n_items, flat % n_items)), shape=(n_users, n_items))
    items = R.T.tocsr()
    counts = np.diff(items.indptr).astype(np.float64)
    budget = int(memory_budget_mib * 2**20)
    safe_rows = budget // (3 * (16 * n_items + 8))
    if safe_rows < 1:
        raise MemoryError("CF product budget cannot fit a single catalog row; increase cf_memory_budget_mib.")
    block_size = min(int(block_size), safe_rows)
    k_cf = min(int(k_cf), max(0, n_items - 1))
    if k_cf == 0:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    rows, cols, data = [], [], []
    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        product = (items[start:stop] @ R).tocsr()
        for local, item in enumerate(range(start, stop)):
            left, right = product.indptr[local:local + 2]
            ids, common = product.indices[left:right], product.data[left:right]
            valid = (ids != item) & (common >= c_min)
            ids, common = ids[valid], common[valid].astype(np.float64)
            if not len(ids):
                continue
            scores = common / (common + t_shrinkage) * common / np.sqrt(counts[item] * counts[ids])
            ids, scores = _deterministic_topk(ids, scores, k_cf)
            rows.append(np.full(len(ids), item, dtype=np.int64))
            cols.append(ids.copy())
            data.append(scores.astype(np.float32))
        del product
    if not rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    directed = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items))
    result = directed.maximum(directed.T).tocsr()
    result.sort_indices()
    return result


def normalize_cf_graph_with_fallback(W_cf):
    if W_cf.shape[0] != W_cf.shape[1] or not np.isfinite(W_cf.data).all() or (W_cf.data < 0).any():
        raise ValueError("CF graph must be square, finite and nonnegative.")
    diff = W_cf - W_cf.T
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-7:
        raise ValueError("CF graph must be symmetric before normalization.")
    degrees = np.asarray(W_cf.sum(1)).ravel().astype(np.float64)
    inv = np.zeros(len(degrees), dtype=np.float64)
    inv[degrees > 0] = 1 / np.sqrt(degrees[degrees > 0])
    scale = sp.diags(inv)
    normalized = (scale @ W_cf @ scale).astype(np.float32)
    return (normalized + sp.diags((degrees == 0).astype(np.float32))).tocsr()


def permute_cf_graph_degree_stratified(W_cf, train_edges, n_items, seed=1, return_metadata=False):
    if W_cf.shape != (n_items, n_items):
        raise ValueError("CF graph shape differs from the catalog.")
    # Train edges may contain duplicate rows; strata use binary item degrees.
    n_users = int(train_edges[0].max()) + 1 if train_edges.numel() else 1
    pairs = binary_train_keys(train_edges, n_users, n_items)
    degrees = np.bincount(pairs % n_items, minlength=n_items)
    bins = np.floor(np.log2(np.maximum(degrees, 1))).astype(np.int64) + 1
    bins[degrees == 0] = 0
    rng, permutation = np.random.RandomState(seed), np.arange(n_items)
    for group in np.unique(bins):
        ids = np.flatnonzero(bins == group)
        permutation[ids] = rng.permutation(ids)
    result = W_cf[permutation, :][:, permutation].tocsr()
    metadata = {"placebo_changed_labels": int(np.count_nonzero(permutation != np.arange(n_items))),
                "placebo_changed_fraction": float(np.mean(permutation != np.arange(n_items))),
                "placebo_strata": "binary_train_degree_log2"}
    return (result, metadata) if return_metadata else result


def build_calibrated_graph_v4(raw_semantic, baseline_normalized, train_edges, n_users, n_items,
                              arm="N-CSE", eta=0.1, k_cf=5, c_min=2, t_shrinkage=5.0,
                              placebo_seed=1, seed=1, cache_dir: Optional[str]=None,
                              block_size=64, memory_budget_mib=128):
    if arm not in ARMS or not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("Unknown arm or eta outside [0, 1].")
    if k_cf < 0 or int(k_cf) != k_cf or c_min < 1 or int(c_min) != c_min:
        raise ValueError("Invalid CF top-k or minimum count.")
    compute_evidence_score(0, 0, 0, t_shrinkage)
    if tuple(baseline_normalized.shape) != (n_items, n_items):
        raise ValueError("Semantic graph shape differs from the catalog.")
    metadata = {"version": 4, "arm": arm, "eta": float(eta), "k_cf": int(k_cf),
                "c_min": int(c_min), "t_shrinkage": float(t_shrinkage),
                "num_users": n_users, "num_items": n_items, "cache_hit": False}
    def off(reason):
        metadata.update(active=False, effective_eta=0.0, nnz=int(baseline_normalized._nnz()), note=reason)
        return GraphStateV4(baseline_normalized, metadata)
    if arm in ("B0", "N0") or eta == 0 or k_cf == 0:
        return off("Exact original S0; CF branch disabled.")

    started = time.perf_counter()
    pairs = binary_train_keys(train_edges, n_users, n_items)
    pair_hash = hashlib.sha256()
    _digest_array(pair_hash, pairs)
    signature = {"cache_version": CACHE_VERSION, "train": pair_hash.hexdigest(),
                 "users": n_users, "items": n_items, "k": k_cf, "cmin": c_min, "t": t_shrinkage,
                 "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"cf_{cache_key}.pt" if cache_dir else None
    W_cf = None
    if cache and cache.is_file():
        try:
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            candidate = tensor_to_scipy(payload["operator"])
            if payload["signature"] != signature or payload["fingerprint"] != sparse_fingerprint(candidate):
                raise ValueError("CF cache signature/content mismatch.")
            if candidate.shape != (n_items, n_items):
                raise ValueError("CF cache shape mismatch.")
            W_cf = candidate
            metadata["cache_hit"] = True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError) as error:
            warnings.warn(f"Ignoring invalid CF cache {cache.name}: {error}", RuntimeWarning)
    if W_cf is None:
        W_cf = build_candidate_support_graph(train_edges, n_items, n_users, k_cf, c_min,
                                             t_shrinkage, block_size, memory_budget_mib)
        if cache:
            atomic_torch_save({"operator": scipy_to_tensor(W_cf), "signature": signature,
                               "fingerprint": sparse_fingerprint(W_cf)}, cache)
    metadata.update(train_fingerprint=signature["train"], cf_cache_key=cache_key,
                    cf_preprocessing_seconds=time.perf_counter() - started,
                    cf_block_size_requested=block_size, cf_memory_budget_mib=memory_budget_mib)
    if W_cf.nnz == 0:
        return off("Exact original S0; CF graph has no eligible edges.")
    if arm == "N-CSE-placebo":
        W_cf, placebo = permute_cf_graph_degree_stratified(W_cf, train_edges, n_items, placebo_seed, True)
        metadata.update(placebo)
    S0 = tensor_to_scipy(baseline_normalized).astype(np.float32)
    cf = normalize_cf_graph_with_fallback(W_cf)
    mixed = ((1 - eta) * S0 + eta * cf).tocsr()
    mixed.eliminate_zeros()
    mixed.sort_indices()
    # Sparse intersection replaces millions of Python tuple/set objects.
    overlap = int(W_cf.sign().multiply(S0.sign()).nnz)
    isolated = int(np.count_nonzero(np.asarray(W_cf.sum(1)).ravel() == 0))
    delta = mixed - S0
    ratio = float(np.sqrt(np.dot(delta.data.astype(np.float64), delta.data.astype(np.float64))) /
                  max(1e-12, np.sqrt(np.dot(S0.data.astype(np.float64), S0.data.astype(np.float64)))))
    metadata.update(active=True, effective_eta=float(eta), s0_nnz=int(S0.nnz), cf_nnz=int(W_cf.nnz),
                    s4_nnz=int(mixed.nnz), cf_isolated_nodes=isolated, cf_isolated_fraction=isolated / n_items,
                    jaccard_semantic_cf=overlap / max(1, W_cf.nnz + S0.nnz - overlap),
                    new_cf_edges_count=int(W_cf.nnz - overlap), new_cf_fraction=1 - overlap / W_cf.nnz,
                    relative_operator_delta=ratio, relative_operator_delta_kind="frobenius_ratio",
                    graph_fingerprint=sparse_fingerprint(mixed), is_placebo=arm == "N-CSE-placebo")
    return GraphStateV4(scipy_to_tensor(mixed).to(baseline_normalized.device), metadata)
