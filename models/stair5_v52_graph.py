"""models/stair5_v52_graph.py — STAIR5-v5.2 (NLGCL-BPE) Graph Builder.
===================================================================
Budgeted Path Expansion (BPE) for Item Update Smoothing in RecSys.

Builds:
    W_star = W_1 + W_add
    S_star = SymNormIso(W_star)
    S_5.2  = (1 - eta) * S_0 + eta * S_star

Key Invariants:
1. Exact W_1 builder reuse from v4 (c_min=2, k_cf=5, t_shrinkage=5.0).
2. Mutual bounded seed graph T (K_seed=10, degree <= 10).
3. Path count m_ij and weighted path score p_ij = sum_{k: d_k^T > 0} (T_ik * T_kj) / d_k^T.
4. Confidence shrinkage a_ij = (m_ij / (m_ij + 1)) * p_ij.
5. Strict candidate exclusion of self-loops, existing direct W_1 edges, and baseline S_0 edges.
6. Local mutual top-k_add (k_add=3) selection.
7. Global edge budget B_pairs = floor(beta_edges * nnz(W_1) / 2) with beta_edges=0.25.
8. Median scale calibration kappa = median(W_1 > 0) / median(Q > 0).
9. Endpoint mass limiter (W_add)_ij = Q_hat_ij * min(u_i, u_j), u_i = min(1, nu * d_i / r_i).
10. SymNormIso: D_star^{-1/2} W_star D_star^{-1/2} + diag(1[d_i^star == 0]).
11. Exact reference recovery: if expansion_enabled=False, nu=0, beta_edges=0, or C is empty,
    delegate bitwise to the exact v4 graph path.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v52_utils import atomic_torch_save, validate_csr_operator

ARMS_V52 = (
    "P-BPE",
    "P-BPE-low",
    "C-V4",
    "C-Direct",
    "C-Radius",
    "C-NoScale",
    "C-M2",
    "C-Overlap",
)
CACHE_VERSION_V52 = 52


@dataclass
class GraphStateV52:
    operator: torch.Tensor
    metadata: dict


def binary_train_keys(train_edges: torch.Tensor, n_users: int, n_items: int) -> np.ndarray:
    """Validate IDs before flattening; deduplicate all train pairs, not a prefix."""
    if n_users <= 0 or n_items <= 0 or n_users * n_items > np.iinfo(np.int64).max:
        raise ValueError("Invalid entity counts or int64 pair-key overflow.")
    if train_edges.ndim != 2 or train_edges.shape[0] != 2 or train_edges.dtype not in (torch.int32, torch.int64):
        raise ValueError("train_edges must be a [2, E] integer tensor.")
    edges = train_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges[0].max() >= n_users or edges[1].max() >= n_items):
        raise ValueError("Train interaction IDs are outside the declared entity counts.")
    return np.unique(edges[0] * n_items + edges[1])


def _digest_array(hasher, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    hasher.update(str((array.shape, array.dtype.str)).encode())
    hasher.update(memoryview(array).cast("B"))


def sparse_fingerprint(matrix: sp.csr_matrix) -> str:
    """Hash canonical shape, indices and values without materializing dense data."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    h = hashlib.sha256(str(matrix.shape).encode())
    for array in (matrix.indptr.astype(np.int64), matrix.indices.astype(np.int64), matrix.data):
        _digest_array(h, array)
    return h.hexdigest()


def tensor_to_scipy(matrix: torch.Tensor) -> sp.csr_matrix:
    matrix = matrix.detach().cpu()
    if matrix.layout == torch.sparse_csr:
        return sp.csr_matrix(
            (matrix.values().numpy(), matrix.col_indices().numpy(), matrix.crow_indices().numpy()),
            shape=tuple(matrix.shape)
        )
    if matrix.layout == torch.sparse_coo:
        matrix = matrix.coalesce()
        indices = matrix.indices().numpy()
        return sp.csr_matrix(
            (matrix.values().numpy(), (indices[0], indices[1])),
            shape=tuple(matrix.shape)
        )
    raise ValueError("Expected a sparse COO or CSR graph.")


def scipy_to_tensor(matrix: sp.csr_matrix) -> torch.Tensor:
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data),
        size=matrix.shape
    )


def compute_evidence_score(c: float, n_i: float, n_j: float, t: float = 5.0) -> float:
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Evidence shrinkage must be finite and positive.")
    if n_i <= 0 or n_j <= 0 or c <= 0:
        return 0.0
    return float(c / (c + t) * c / math.sqrt(n_i * n_j))


def _deterministic_topk(ids: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select top-k entries deterministically with tie-breaking by id ascending."""
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


def build_candidate_support_graph(
    train_edges: torch.Tensor,
    n_items: int,
    n_users: int,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
    return_reserve: bool = False,
    reserve_k: int = 15,
) -> Any:
    """Exact binary R.T @ R in bounded CSR blocks (v4 logic).

    If return_reserve=True, also returns reserve candidate items outside top-k_cf for C-Direct.
    """
    if k_cf < 0 or int(k_cf) != k_cf or c_min < 1 or int(c_min) != c_min:
        raise ValueError("Require integer k_cf >= 0 and c_min >= 1.")
    if block_size < 1 or int(block_size) != block_size or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("Require a positive integer block size and finite memory budget.")
    compute_evidence_score(0, 0, 0, t_shrinkage)
    flat = binary_train_keys(train_edges, n_users, n_items)
    if k_cf == 0 or not len(flat):
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        return (empty, empty) if return_reserve else empty

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
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        return (empty, empty) if return_reserve else empty

    rows, cols, data = [], [], []
    res_rows, res_cols, res_data = [], [], []

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
            # Full sort for selection
            order = np.lexsort((ids, -scores))
            top_ids = ids[order[:k_cf]]
            top_scores = scores[order[:k_cf]]
            rows.append(np.full(len(top_ids), item, dtype=np.int64))
            cols.append(top_ids.copy())
            data.append(top_scores.astype(np.float32))

            if return_reserve and len(order) > k_cf:
                res_sel = order[k_cf:k_cf + reserve_k]
                r_ids = ids[res_sel]
                r_scores = scores[res_sel]
                res_rows.append(np.full(len(r_ids), item, dtype=np.int64))
                res_cols.append(r_ids.copy())
                res_data.append(r_scores.astype(np.float32))
        del product

    if not rows:
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        return (empty, empty) if return_reserve else empty

    directed = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items))
    result = directed.maximum(directed.T).tocsr()
    result.sort_indices()

    if return_reserve:
        if res_rows:
            res_mat = sp.csr_matrix((np.concatenate(res_data), (np.concatenate(res_rows), np.concatenate(res_cols))), shape=(n_items, n_items))
        else:
            res_mat = sp.csr_matrix((n_items, n_items), dtype=np.float32)
        return result, res_mat

    return result


def normalize_cf_graph_with_fallback(W_cf: sp.csr_matrix) -> sp.csr_matrix:
    """SymNormIso: Symmetric degree normalization with exact identity diagonal on isolated nodes."""
    if W_cf.shape[0] != W_cf.shape[1] or not np.isfinite(W_cf.data).all() or (W_cf.data < 0).any():
        raise ValueError("CF graph must be square, finite and nonnegative.")
    diff = W_cf - W_cf.T
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-6:
        raise ValueError("CF graph must be symmetric before normalization.")
    degrees = np.asarray(W_cf.sum(1)).ravel().astype(np.float64)
    inv = np.zeros(len(degrees), dtype=np.float64)
    inv[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    scale = sp.diags(inv)
    normalized = (scale @ W_cf @ scale).astype(np.float32)
    return (normalized + sp.diags((degrees == 0).astype(np.float32))).tocsr()


def build_mutual_seed_graph(W_1: sp.csr_matrix, k_seed: int = 10) -> sp.csr_matrix:
    """Builds bounded mutual seed graph T from W_1:

    T_ij = W_{1, ij} * 1[j in Top_{k_seed}(i)] * 1[i in Top_{k_seed}(j)]
    T is symmetric, non-negative, zero diagonal, with degree count d_k^T <= k_seed.
    """
    n_items = W_1.shape[0]
    rows, cols, data = [], [], []
    for i in range(n_items):
        start, end = W_1.indptr[i], W_1.indptr[i + 1]
        if start == end:
            continue
        ids = W_1.indices[start:end]
        scores = W_1.data[start:end]
        sel = np.lexsort((ids, -scores))[:k_seed]
        rows.append(np.full(len(sel), i, dtype=np.int64))
        cols.append(ids[sel])
        data.append(scores[sel])
    if not rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float64)
    directed = sp.csr_matrix(
        (np.concatenate(data).astype(np.float64), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items)
    )
    # Mutual selection: both i->j and j->i must be selected
    T = directed.minimum(directed.T).tocsr()
    T.setdiag(0)
    T.eliminate_zeros()
    T.sort_indices()
    return T


def enumerate_path_candidates(
    T: sp.csr_matrix,
    W_1: sp.csr_matrix,
    S_0: sp.csr_matrix,
    m_min: int = 1,
    t_path: float = 1.0,
    exclude_s0: bool = True,
    block_size: int = 512,
) -> Tuple[sp.csr_matrix, dict]:
    """Computes path scores a_ij for candidates in C without instantiating an N x N dense matrix.

    m_ij = sum_k 1[T_ik > 0] * 1[T_kj > 0]
    p_ij = sum_{k: d_k^T > 0} (T_ik * T_kj) / d_k^T
    a_ij = (m_ij / (m_ij + t_path)) * p_ij

    Candidate Set: C = {(i, j) : i != j, m_ij >= m_min, W_1,ij == 0, (and S_0,ij == 0 if exclude_s0)}.
    """
    n_items = T.shape[0]
    degrees_T = np.diff(T.indptr).astype(np.float64)
    inv_d = np.zeros_like(degrees_T)
    np.divide(1.0, degrees_T, out=inv_d, where=degrees_T > 0)
    D_inv = sp.diags(inv_d)

    # Binary seed for path counting
    T_bin = T.copy().astype(np.float64)
    T_bin.data = np.ones_like(T_bin.data)

    # Scaled seed for path score
    T_scaled = (D_inv @ T).tocsr()  # row k is T[k] / d_k^T

    cand_rows, cand_cols, cand_scores = [], [], []
    total_m1_candidates = 0
    total_m2_candidates = 0
    excluded_w1_count = 0
    excluded_s0_count = 0

    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        # Compute count matrix block: M_block = T_bin[start:stop] @ T_bin
        M_block = (T_bin[start:stop] @ T_bin).tocsr()
        # Compute weighted path score block: P_block = T[start:stop] @ T_scaled
        P_block = (T[start:stop] @ T_scaled).tocsr()

        for local, item in enumerate(range(start, stop)):
            # Row item
            m_l, m_r = M_block.indptr[local:local + 2]
            p_l, p_r = P_block.indptr[local:local + 2]
            if m_l == m_r:
                continue

            j_indices = M_block.indices[m_l:m_r]
            m_vals = M_block.data[m_l:m_r]
            p_vals = P_block.data[p_l:p_r]

            # 1. Exclude self-loops
            valid = (j_indices != item)
            j_indices, m_vals, p_vals = j_indices[valid], m_vals[valid], p_vals[valid]
            if not len(j_indices):
                continue

            # Track counts
            total_m1_candidates += len(j_indices)
            total_m2_candidates += int(np.count_nonzero(m_vals >= 2))

            # 2. Filter by m_min
            if m_min > 1:
                valid_m = (m_vals >= m_min)
                j_indices, m_vals, p_vals = j_indices[valid_m], m_vals[valid_m], p_vals[valid_m]
                if not len(j_indices):
                    continue

            # 3. Exclude existing W_1 edges
            w1_l, w1_r = W_1.indptr[item:item + 2]
            if w1_l < w1_r:
                existing_w1_ids = W_1.indices[w1_l:w1_r]
                # Fast sorted membership test
                pos = np.searchsorted(existing_w1_ids, j_indices)
                in_w1 = (pos < len(existing_w1_ids)) & (existing_w1_ids[np.minimum(pos, len(existing_w1_ids) - 1)] == j_indices)
                excluded_w1_count += int(np.count_nonzero(in_w1))
                not_in_w1 = ~in_w1
                j_indices, m_vals, p_vals = j_indices[not_in_w1], m_vals[not_in_w1], p_vals[not_in_w1]
                if not len(j_indices):
                    continue

            # 4. Exclude existing S_0 edges (if exclude_s0 is True)
            if exclude_s0:
                s0_l, s0_r = S_0.indptr[item:item + 2]
                if s0_l < s0_r:
                    existing_s0_ids = S_0.indices[s0_l:s0_r]
                    pos_s0 = np.searchsorted(existing_s0_ids, j_indices)
                    in_s0 = (pos_s0 < len(existing_s0_ids)) & (existing_s0_ids[np.minimum(pos_s0, len(existing_s0_ids) - 1)] == j_indices)
                    excluded_s0_count += int(np.count_nonzero(in_s0))
                    not_in_s0 = ~in_s0
                    j_indices, m_vals, p_vals = j_indices[not_in_s0], m_vals[not_in_s0], p_vals[not_in_s0]
                    if not len(j_indices):
                        continue

            # Confidence shrinkage: a_ij = (m_ij / (m_ij + t_path)) * p_ij
            a_vals = (m_vals / (m_vals + t_path)) * p_vals
            pos_a = a_vals > 0
            if np.any(pos_a):
                cand_rows.append(np.full(int(np.count_nonzero(pos_a)), item, dtype=np.int64))
                cand_cols.append(j_indices[pos_a].astype(np.int64))
                cand_scores.append(a_vals[pos_a].astype(np.float64))

        del M_block, P_block

    stats = {
        "total_m1_candidates": total_m1_candidates,
        "total_m2_candidates": total_m2_candidates,
        "excluded_w1_count": excluded_w1_count,
        "excluded_s0_count": excluded_s0_count,
    }

    if not cand_rows:
        return sp.csr_matrix((n_items, n_items), dtype=np.float64), stats

    candidate_matrix = sp.csr_matrix(
        (np.concatenate(cand_scores), (np.concatenate(cand_rows), np.concatenate(cand_cols))),
        shape=(n_items, n_items)
    )
    candidate_matrix.sort_indices()
    return candidate_matrix, stats


def apply_edge_and_mass_budgets(
    candidates: sp.csr_matrix,
    W_1: sp.csr_matrix,
    k_add: int = 3,
    beta_edges: float = 0.25,
    nu: float = 0.5,
    scale_calibration: bool = True,
) -> Tuple[sp.csr_matrix, dict]:
    """Applies local mutual top-k, global edge budget, scale calibration, and endpoint mass limiter:

    1. Local Mutual Top-k_add per row -> Q_mutual
    2. Global Edge Budget: keep at most floor(beta_edges * nnz(W_1) / 2) undirected pairs -> Q
    3. Median scale calibration: kappa = median(W_1 > 0) / median(Q > 0), Q_hat = kappa * Q
    4. Endpoint mass limiter: (W_add)_ij = Q_hat_ij * min(u_i, u_j), u_i = min(1, nu * d_i / r_i)
    """
    n_items = W_1.shape[0]
    w1_nnz = int(W_1.nnz)

    if candidates.nnz == 0 or w1_nnz == 0 or beta_edges <= 0 or nu <= 0 or k_add <= 0:
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float64)
        return empty, {
            "scale_kappa": 1.0,
            "mutual_k_add_nnz": 0,
            "budgeted_pairs": 0,
            "added_edges_nnz": 0,
            "added_mass_total": 0.0,
            "base_mass_total": float(np.sum(W_1.data)),
            "added_mass_ratio_median": 0.0,
            "added_mass_ratio_max": 0.0,
            "cap_active_fraction": 0.0,
        }

    # 1. Local Top-k_add selection per node
    sel_rows, sel_cols, sel_vals = [], [], []
    for i in range(n_items):
        start, end = candidates.indptr[i], candidates.indptr[i + 1]
        if start == end:
            continue
        ids = candidates.indices[start:end]
        scores = candidates.data[start:end]
        sel = np.lexsort((ids, -scores))[:k_add]
        sel_rows.append(np.full(len(sel), i, dtype=np.int64))
        sel_cols.append(ids[sel])
        sel_vals.append(scores[sel])

    if not sel_rows:
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float64)
        return empty, {"scale_kappa": 1.0, "added_edges_nnz": 0}

    dt = sp.csr_matrix(
        (np.concatenate(sel_vals), (np.concatenate(sel_rows), np.concatenate(sel_cols))),
        shape=(n_items, n_items)
    )
    # Mutual selection
    Q_mut = dt.minimum(dt.T).tocsr()
    Q_mut.setdiag(0)
    Q_mut.eliminate_zeros()
    Q_mut.sort_indices()

    mutual_nnz = int(Q_mut.nnz)
    if mutual_nnz == 0:
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float64)
        return empty, {"scale_kappa": 1.0, "mutual_k_add_nnz": 0, "added_edges_nnz": 0}

    # 2. Global Edge Budget
    # Consider undirected pairs i < j
    upper_coo = sp.triu(Q_mut, k=1).tocoo()
    B_pairs = int(math.floor(beta_edges * w1_nnz / 2.0))

    if len(upper_coo.data) > B_pairs:
        order = np.lexsort((upper_coo.col, upper_coo.row, -upper_coo.data))[:B_pairs]
        upper = sp.csr_matrix(
            (upper_coo.data[order], (upper_coo.row[order], upper_coo.col[order])),
            shape=(n_items, n_items)
        )
    else:
        upper = upper_coo.tocsr()

    Q = (upper + upper.T).tocsr()
    Q.eliminate_zeros()
    Q.sort_indices()

    if Q.nnz == 0:
        empty = sp.csr_matrix((n_items, n_items), dtype=np.float64)
        return empty, {"scale_kappa": 1.0, "mutual_k_add_nnz": mutual_nnz, "budgeted_pairs": 0, "added_edges_nnz": 0}

    # 3. Median Scale Calibration
    if scale_calibration:
        w1_pos = W_1.data[W_1.data > 0]
        q_pos = Q.data[Q.data > 0]
        if len(w1_pos) and len(q_pos) and np.median(q_pos) > 0:
            kappa = float(np.median(w1_pos) / np.median(q_pos))
        else:
            kappa = 1.0
    else:
        kappa = 1.0

    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"Invalid scale calibration factor kappa: {kappa}")

    Q_hat = (Q * kappa).tocsr()

    # 4. Endpoint Mass Limiter
    d1 = np.asarray(W_1.sum(1)).ravel().astype(np.float64)
    r = np.asarray(Q_hat.sum(1)).ravel().astype(np.float64)

    cap = np.ones(n_items, dtype=np.float64)
    pos_r = r > 0
    np.divide(nu * d1, r, out=cap, where=pos_r)
    cap = np.minimum(1.0, cap)

    # (W_add)_ij = Q_hat_ij * min(u_i, u_j)
    coo = Q_hat.tocoo()
    multiplier = np.minimum(cap[coo.row], cap[coo.col])
    W_add = sp.csr_matrix(
        (coo.data * multiplier, (coo.row, coo.col)),
        shape=(n_items, n_items)
    )
    W_add.eliminate_zeros()
    W_add.sort_indices()

    # Invariant assertion: sum_j (W_add)_ij <= nu * d1_i + epsilon
    added_sums = np.asarray(W_add.sum(1)).ravel().astype(np.float64)
    if np.any(added_sums > nu * d1 + 1e-6):
        max_violation = float(np.max(added_sums - nu * d1))
        raise RuntimeError(f"Mass budget invariant violated! Max excess: {max_violation}")

    # Compute telemetry
    active = np.diff(W_add.indptr) > 0
    ratio = np.zeros_like(d1)
    np.divide(added_sums, d1, out=ratio, where=d1 > 0)
    cap_active = (cap < 1.0) & pos_r

    telemetry = {
        "scale_kappa": kappa,
        "mutual_k_add_nnz": mutual_nnz,
        "budgeted_pairs": int(upper.nnz),
        "added_edges_nnz": int(W_add.nnz),
        "added_mass_total": float(np.sum(W_add.data)),
        "base_mass_total": float(np.sum(d1)),
        "added_mass_ratio_median": float(np.median(ratio[active])) if np.any(active) else 0.0,
        "added_mass_ratio_max": float(np.max(ratio[active])) if np.any(active) else 0.0,
        "cap_active_fraction": float(np.mean(cap_active[active])) if np.any(active) else 0.0,
    }
    return W_add, telemetry


def build_calibrated_graph_v52(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    arm: str = "P-BPE",
    expansion_enabled: bool = True,
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    k_seed: int = 10,
    t_path: float = 1.0,
    k_add: int = 3,
    beta_edges: float = 0.25,
    nu: float = 0.5,
    cache_dir: Optional[str] = None,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
) -> GraphStateV52:
    """Builds the complete STAIR5-v5.2 blended operator S_5.2:

    W_star = W_1 + W_add
    S_star = SymNormIso(W_star)
    S_5.2 = (1 - eta) * S_0 + eta * S_star
    """
    if arm not in ARMS_V52:
        raise ValueError(f"Unknown arm: {arm}. Must be one of {ARMS_V52}")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite in [0, 1].")
    if tuple(baseline_normalized.shape) != (n_items, n_items):
        raise ValueError("Semantic graph shape differs from the catalog.")

    # Arm-specific hyperparameter overrides
    if arm == "C-V4":
        expansion_enabled = False
    elif arm == "P-BPE-low":
        nu = 0.2
    elif arm == "C-Radius":
        expansion_enabled = False
    scale_calibration = (arm != "C-NoScale")
    m_min = 2 if arm == "C-M2" else 1
    exclude_s0 = (arm != "C-Overlap")

    started = time.perf_counter()
    S0 = tensor_to_scipy(baseline_normalized).astype(np.float32)

    metadata: Dict[str, Any] = {
        "version": "5.2",
        "arm": arm,
        "expansion_enabled": expansion_enabled,
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "k_seed": int(k_seed),
        "t_path": float(t_path),
        "k_add": int(k_add),
        "beta_edges": float(beta_edges),
        "nu": float(nu),
        "m_min": int(m_min),
        "exclude_s0": exclude_s0,
        "scale_calibration": scale_calibration,
        "num_users": n_users,
        "num_items": n_items,
        "cache_hit": False,
    }

    # Off-path delegation function
    def off_path_v4(W_1_mat: sp.csr_matrix, reason: str) -> GraphStateV52:
        if W_1_mat.nnz == 0:
            # Fallback to pure S0
            metadata.update(active=False, effective_eta=0.0, s0_nnz=int(S0.nnz), s52_nnz=int(S0.nnz), note=reason)
            return GraphStateV52(baseline_normalized, metadata)
        cf_norm = normalize_cf_graph_with_fallback(W_1_mat)
        mixed = ((1.0 - eta) * S0 + eta * cf_norm).tocsr()
        mixed.eliminate_zeros()
        mixed.sort_indices()
        metadata.update(
            active=False,
            effective_eta=float(eta),
            s0_nnz=int(S0.nnz),
            w1_nnz=int(W_1_mat.nnz),
            s52_nnz=int(mixed.nnz),
            graph_fingerprint=sparse_fingerprint(mixed),
            note=reason,
        )
        return GraphStateV52(scipy_to_tensor(mixed).to(baseline_normalized.device), metadata)

    # 1. Binary train keys and signature
    pairs = binary_train_keys(train_edges, n_users, n_items)
    pair_hash = hashlib.sha256()
    _digest_array(pair_hash, pairs)
    train_fingerprint = pair_hash.hexdigest()

    signature = {
        "cache_version": CACHE_VERSION_V52,
        "train": train_fingerprint,
        "users": n_users,
        "items": n_items,
        "arm": arm,
        "k": k_cf,
        "cmin": c_min,
        "t": t_shrinkage,
        "k_seed": k_seed,
        "t_path": t_path,
        "k_add": k_add,
        "beta_edges": beta_edges,
        "nu": nu,
        "m_min": m_min,
        "exclude_s0": exclude_s0,
        "scale_calibration": scale_calibration,
        "eta": eta,
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"s52_{cache_key}.pt" if cache_dir else None

    # Check cache
    if cache_file and cache_file.is_file():
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            candidate = tensor_to_scipy(payload["operator"])
            if payload["signature"] != signature or payload["fingerprint"] != sparse_fingerprint(candidate):
                raise ValueError("Cache signature/content fingerprint mismatch.")
            if candidate.shape != (n_items, n_items):
                raise ValueError("Cache shape mismatch.")
            validate_csr_operator(candidate, n_items)
            metadata.update(payload["metadata"])
            metadata["cache_hit"] = True
            metadata["preprocessing_seconds"] = time.perf_counter() - started
            return GraphStateV52(scipy_to_tensor(candidate).to(baseline_normalized.device), metadata)
        except Exception as error:
            warnings.warn(f"Ignoring invalid S5.2 cache {cache_file.name}: {error}", RuntimeWarning)

    # 2. Build W_1
    return_reserve = (arm == "C-Direct")
    if return_reserve:
        W_1, reserve_direct = build_candidate_support_graph(
            train_edges, n_items, n_users, k_cf, c_min, t_shrinkage,
            block_size, memory_budget_mib, return_reserve=True, reserve_k=15
        )
    else:
        W_1 = build_candidate_support_graph(
            train_edges, n_items, n_users, k_cf, c_min, t_shrinkage,
            block_size, memory_budget_mib
        )
        reserve_direct = None

    metadata["train_fingerprint"] = train_fingerprint
    metadata["w1_nnz"] = int(W_1.nnz)
    w1_degrees = np.diff(W_1.indptr)
    metadata["w1_isolated_nodes"] = int(np.count_nonzero(w1_degrees == 0))
    metadata["w1_max_degree"] = int(np.max(w1_degrees)) if len(w1_degrees) else 0

    # 3. Off-path check
    if not expansion_enabled or nu <= 0 or beta_edges <= 0 or W_1.nnz == 0:
        return off_path_v4(W_1, "Exact v4 fallback path; BPE expansion disabled or zero budget.")

    # 4. Generate candidate additions
    if arm == "C-Direct":
        # Candidate pairs come from reserve direct co-occurrence outside W_1 and S_0
        cand_mat = reserve_direct.copy()
        cand_mat = cand_mat - cand_mat.multiply(W_1.sign())
        if exclude_s0:
            cand_mat = cand_mat - cand_mat.multiply(S0.sign())
        cand_mat.setdiag(0)
        cand_mat.eliminate_zeros()
        cand_mat.sort_indices()
        cand_stats = {"total_m1_candidates": cand_mat.nnz, "total_m2_candidates": 0, "excluded_w1_count": 0, "excluded_s0_count": 0}
    else:
        # BPE: Mutual seed graph T
        T = build_mutual_seed_graph(W_1, k_seed=k_seed)
        seed_degrees = np.diff(T.indptr)
        metadata["seed_nnz"] = int(T.nnz)
        metadata["seed_max_degree"] = int(np.max(seed_degrees)) if len(seed_degrees) else 0

        # Enumerate wedges and compute scores a_ij
        cand_mat, cand_stats = enumerate_path_candidates(
            T=T,
            W_1=W_1,
            S_0=S0,
            m_min=m_min,
            t_path=t_path,
            exclude_s0=exclude_s0,
            block_size=512,
        )

    metadata.update(cand_stats)
    metadata["raw_candidates_nnz"] = int(cand_mat.nnz)

    # 5. Apply edge and mass budgets
    W_add, budget_stats = apply_edge_and_mass_budgets(
        candidates=cand_mat,
        W_1=W_1,
        k_add=k_add,
        beta_edges=beta_edges,
        nu=nu,
        scale_calibration=scale_calibration,
    )
    metadata.update(budget_stats)

    # 6. Check if additions are empty
    if W_add.nnz == 0:
        return off_path_v4(W_1, "No valid candidate edges survived pruning and budget; delegated to v4.")

    # 7. W_star = W_1 + W_add, S_star = SymNormIso(W_star)
    W_star = (W_1 + W_add).tocsr()
    W_star.setdiag(0)
    W_star.eliminate_zeros()
    W_star.sort_indices()

    S_star = normalize_cf_graph_with_fallback(W_star)

    # 8. S_5.2 = (1 - eta) * S_0 + eta * S_star
    S_52 = ((1.0 - eta) * S0 + eta * S_star).tocsr()
    S_52.eliminate_zeros()
    S_52.sort_indices()

    validate_csr_operator(S_52, n_items)

    elapsed = time.perf_counter() - started
    metadata.update(
        active=True,
        effective_eta=float(eta),
        s0_nnz=int(S0.nnz),
        s_star_nnz=int(S_star.nnz),
        s52_nnz=int(S_52.nnz),
        graph_fingerprint=sparse_fingerprint(S_52),
        preprocessing_seconds=elapsed,
        note="STAIR5-v5.2 BPE blended operator active.",
    )

    # Save cache if requested
    if cache_file:
        atomic_torch_save(
            {
                "operator": scipy_to_tensor(S_52),
                "signature": signature,
                "fingerprint": metadata["graph_fingerprint"],
                "metadata": metadata,
            },
            cache_file,
        )

    return GraphStateV52(scipy_to_tensor(S_52).to(baseline_normalized.device), metadata)
