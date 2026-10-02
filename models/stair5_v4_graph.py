"""STAIR5-v4 Graph Module: Candidate Support Expansion (CSE) for BSC.

Constructs the memory-bounded collaborative top-k item-item graph, applies
symmetric degree normalization with explicit isolated-item identity fallback,
and performs convex blending with the original semantic graph S0:

    S4 = (1 - eta) * S0 + eta * S_bar_CF

Properties:
- Exact train-only binary co-occurrence counts.
- Evidence shrinkage q_ij = [c_ij / (c_ij + t)] * [c_ij / sqrt(n_i * n_j)].
- Memory-bounded block sparse multiplication (SciPy block GEMM) to avoid OOM on large catalogs.
- Exact off-path: if eta == 0 or k_cf == 0 or arm in ('B0', 'N0'), returns S0 directly.
- Convex blend ensures ||S4||_2 <= 1.0 and exact symmetry.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class GraphStateV4:
    operator: torch.Tensor          # Sparse CSR tensor on CPU/CUDA
    metadata: Dict[str, Union[int, float, str, bool, dict]]


def compute_evidence_score(c: float, n_i: float, n_j: float, t: float = 5.0) -> float:
    """Computes Ochiai score with evidence shrinkage."""
    if n_i <= 0 or n_j <= 0 or c <= 0:
        return 0.0
    shrinkage = c / (c + t)
    base = c / math.sqrt(n_i * n_j)
    return float(shrinkage * base)


def build_candidate_support_graph(
    train_edges: torch.Tensor,
    n_items: int,
    n_users: int,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    block_size: int = 64,
) -> sp.csr_matrix:
    """Constructs symmetric collaborative candidate graph W_CF using block-wise sparse computation.
    
    Args:
        train_edges: (2, E) tensor of (user, item) pairs from training set only.
        n_items: Total item count.
        n_users: Total user count.
        k_cf: Top-k collaborative neighbors per item (default: 5).
        c_min: Minimum co-occurrence count threshold (default: 2).
        t_shrinkage: Shrinkage parameter t (default: 5.0).
        block_size: Number of items per block to limit peak memory during co-occurrence mining.
        
    Returns:
        sp.csr_matrix of shape (n_items, n_items): Symmetric weighted candidate adjacency W_CF.
    """
    edges_np = train_edges.detach().cpu().numpy()
    u = edges_np[0].astype(np.int64)
    i = edges_np[1].astype(np.int64)

    # Deduplicate binary user-item interactions
    flat = u * n_items + i
    unique_flat = np.unique(flat)
    u_clean = unique_flat // n_items
    i_clean = unique_flat % n_items

    # Binary interaction matrix R: shape (n_users, n_items)
    val = np.ones(len(unique_flat), dtype=np.float32)
    R = sp.csc_matrix((val, (u_clean, i_clean)), shape=(n_users, n_items))

    # Item interaction counts: n_i = sum_u R_{ui}
    item_counts = np.asarray(R.sum(axis=0)).flatten().astype(np.float64)

    # Block-wise computation of top-k collaborative neighbors
    directed_rows: List[int] = []
    directed_cols: List[int] = []
    directed_data: List[float] = []

    # Transpose R once in CSR format for fast row slicing by items
    R_items = R.transpose().tocsr()  # shape: (n_items, n_users)

    for start_idx in range(0, n_items, block_size):
        end_idx = min(start_idx + block_size, n_items)
        # Block of items: shape (end_idx - start_idx, n_users)
        block_items = R_items[start_idx:end_idx]
        # Multiply block with full R: (block_size, n_users) x (n_users, n_items) -> (block_size, n_items)
        co_occ_block = block_items.dot(R).tocsr()

        for local_row, item_i in enumerate(range(start_idx, end_idx)):
            n_i = item_counts[item_i]
            if n_i <= 0:
                continue

            row_start = co_occ_block.indptr[local_row]
            row_end = co_occ_block.indptr[local_row + 1]
            if row_start >= row_end:
                continue

            cand_items = co_occ_block.indices[row_start:row_end]
            cand_counts = co_occ_block.data[row_start:row_end]

            # Filter diagonal (item_j == item_i) and c_ij < c_min
            valid_mask = (cand_items != item_i) & (cand_counts >= c_min)
            if not np.any(valid_mask):
                continue

            filtered_items = cand_items[valid_mask]
            filtered_counts = cand_counts[valid_mask]

            # Compute shrinkage scores
            n_j_arr = item_counts[filtered_items]
            shrinkage = filtered_counts / (filtered_counts + t_shrinkage)
            scores = shrinkage * (filtered_counts / np.sqrt(n_i * n_j_arr))

            # Select top-k_cf with deterministic tie-breaking (by item ID)
            if len(scores) > k_cf:
                # Sort primarily by -score, secondarily by item ID
                order = np.lexsort((filtered_items, -scores))[:k_cf]
                top_items = filtered_items[order]
                top_scores = scores[order]
            else:
                top_items = filtered_items
                top_scores = scores

            for item_j, q_val in zip(top_items, top_scores):
                if q_val > 0:
                    directed_rows.append(item_i)
                    directed_cols.append(int(item_j))
                    directed_data.append(float(q_val))

    if not directed_rows:
        # Empty CF graph
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)

    # Build directed sparse matrix
    W_dir = sp.csr_matrix(
        (directed_data, (directed_rows, directed_cols)),
        shape=(n_items, n_items),
        dtype=np.float32
    )

    # Symmetric union: W_CF(i, j) = max(W_dir(i, j), W_dir(j, i))
    W_dir_t = W_dir.transpose().tocsr()
    W_cf = W_dir.maximum(W_dir_t)
    return W_cf


def normalize_cf_graph_with_fallback(W_cf: sp.csr_matrix) -> sp.csr_matrix:
    """Applies symmetric degree normalization with explicit identity fallback for isolated items:
    
        S_bar_CF = D_CF^{-1/2} W_CF D_CF^{-1/2} + diag(1[d_i^CF == 0])
    """
    n_items = W_cf.shape[0]
    degrees = np.asarray(W_cf.sum(axis=1)).flatten().astype(np.float64)

    # Compute D^{-1/2}
    inv_sqrt = np.zeros(n_items, dtype=np.float64)
    pos_mask = degrees > 0
    inv_sqrt[pos_mask] = 1.0 / np.sqrt(degrees[pos_mask])

    D_inv = sp.diags(inv_sqrt, dtype=np.float64)
    S_cf = D_inv.dot(W_cf).dot(D_inv).astype(np.float32)

    # Add identity for isolated nodes only
    isolated_mask = (~pos_mask).astype(np.float32)
    if np.any(isolated_mask > 0):
        I_isolated = sp.diags(isolated_mask, dtype=np.float32)
        S_bar_cf = S_cf + I_isolated
    else:
        S_bar_cf = S_cf

    return S_bar_cf.tocsr()


def permute_cf_graph_degree_stratified(
    W_cf: sp.csr_matrix,
    train_edges: torch.Tensor,
    n_items: int,
    seed: int = 1,
) -> sp.csr_matrix:
    """Placebo control: permutes item nodes within train-degree strata."""
    rng = np.random.RandomState(seed)
    edges_np = train_edges.detach().cpu().numpy()
    u, i = edges_np[0], edges_np[1]
    unique_i, counts = np.unique(i, return_counts=True)
    item_degrees = np.zeros(n_items, dtype=np.int64)
    item_degrees[unique_i] = counts

    perm = np.arange(n_items)
    # Stratify by degree log-bins
    max_deg = max(1, item_degrees.max())
    bins = np.geomspace(1, max_deg + 1, num=min(20, max_deg + 1))
    bin_ids = np.digitize(item_degrees, bins)

    for b in np.unique(bin_ids):
        stratum = np.where(bin_ids == b)[0]
        if len(stratum) > 1:
            shuffled = rng.permutation(stratum)
            perm[stratum] = shuffled

    # Permute rows and columns
    perm_inv = np.empty_like(perm)
    perm_inv[perm] = np.arange(n_items)
    W_perm = W_cf[perm, :][:, perm].tocsr()
    return W_perm


def build_calibrated_graph_v4(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    arm: str = "N-CSE",
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    placebo_seed: int = 1,
    seed: int = 1,
    cache_dir: Optional[str] = None,
) -> GraphStateV4:
    """Builds the STAIR5-v4 blended operator S4.
    
    Args:
        raw_semantic: Raw semantic adjacency (sparse COO or CSR).
        baseline_normalized: Original S0 normalized semantic adjacency (sparse CSR).
        train_edges: (2, E) training interactions tensor.
        n_users: User count.
        n_items: Item count.
        arm: One of ('B0', 'N0', 'C0', 'N-CSE', 'N-CSE-placebo', 'v4b').
        eta: Convex blend weight for collaborative operator in [0, 1].
        k_cf: Collaborative top-k neighbors.
        c_min: Minimum co-occurrence count threshold.
        t_shrinkage: Evidence shrinkage parameter.
        placebo_seed: Seed for placebo permutation.
        seed: Seed.
        cache_dir: Optional disk cache directory.
        
    Returns:
        GraphStateV4 containing the blended sparse CSR operator and full audit metadata.
    """
    if arm in ("B0", "N0") or eta <= 0.0 or k_cf <= 0:
        # Exact off-path: returns original S0 buffer
        metadata = {
            "version": 4,
            "arm": arm,
            "active": False,
            "eta": 0.0,
            "k_cf": 0,
            "c_min": c_min,
            "t_shrinkage": t_shrinkage,
            "num_users": n_users,
            "num_items": n_items,
            "cache_hit": False,
            "nnz": int(baseline_normalized._nnz() if baseline_normalized.is_sparse else baseline_normalized.numel()),
            "note": "Exact S0 baseline operator (CF branch inactive).",
        }
        return GraphStateV4(operator=baseline_normalized, metadata=metadata)

    # Check cache if directory is provided
    cache_path = None
    if cache_dir is not None:
        cache_p = Path(cache_dir)
        cache_p.mkdir(parents=True, exist_ok=True)
        # Fingerprint computation
        hasher = hashlib.sha256()
        hasher.update(f"v4_{arm}_{eta}_{k_cf}_{c_min}_{t_shrinkage}_{placebo_seed}_{seed}_{n_users}_{n_items}".encode())
        hasher.update(train_edges.detach().cpu().numpy().tobytes()[:100000])
        cache_key = hasher.hexdigest()
        cache_path = cache_p / f"stair5_v4_graph_{cache_key}.pt"
        if cache_path.is_file():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                cached["metadata"]["cache_hit"] = True
                return GraphStateV4(operator=cached["operator"], metadata=cached["metadata"])
            except Exception:
                pass

    # Step 1: Mine collaborative candidate graph
    W_cf = build_candidate_support_graph(
        train_edges=train_edges,
        n_items=n_items,
        n_users=n_users,
        k_cf=k_cf,
        c_min=c_min,
        t_shrinkage=t_shrinkage,
    )

    is_placebo = (arm == "N-CSE-placebo")
    if is_placebo:
        W_cf = permute_cf_graph_degree_stratified(
            W_cf=W_cf,
            train_edges=train_edges,
            n_items=n_items,
            seed=placebo_seed,
        )

    # Step 2: Normalize CF graph with explicit identity fallback for isolated items
    S_bar_cf = normalize_cf_graph_with_fallback(W_cf)

    # Step 3: Convert S0 baseline to SciPy CSR
    s0_coo = baseline_normalized.to_sparse_coo().coalesce()
    s0_indices = s0_coo.indices().detach().cpu().numpy()
    s0_values = s0_coo.values().detach().cpu().numpy()
    S0_sp = sp.csr_matrix((s0_values, (s0_indices[0], s0_indices[1])), shape=(n_items, n_items), dtype=np.float32)

    # Step 4: Convex blend S4 = (1 - eta) * S0 + eta * S_bar_cf
    S4_sp = ((1.0 - eta) * S0_sp + eta * S_bar_cf).tocsr()

    # Step 5: Convert S4_sp to PyTorch sparse CSR
    crow_indices = torch.from_numpy(S4_sp.indptr.astype(np.int64))
    col_indices = torch.from_numpy(S4_sp.indices.astype(np.int64))
    values = torch.from_numpy(S4_sp.data.astype(np.float32))

    S4_tensor = torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        size=(n_items, n_items),
        dtype=torch.float32,
    )

    # Diagnostics
    cf_nnz = int(W_cf.nnz)
    s0_nnz = int(S0_sp.nnz)
    s4_nnz = int(S4_sp.nnz)
    cf_isolated = int(np.sum(np.asarray(W_cf.sum(axis=1)).flatten() == 0))

    # Overlap metrics
    cf_coo = W_cf.tocoo()
    cf_edges_set = set(zip(cf_coo.row.tolist(), cf_coo.col.tolist()))
    s0_edges_set = set(zip(s0_indices[0].tolist(), s0_indices[1].tolist()))
    intersect_count = len(cf_edges_set & s0_edges_set)
    union_count = len(cf_edges_set | s0_edges_set)
    jaccard = float(intersect_count / max(1, union_count))
    new_cf_edges = len(cf_edges_set - s0_edges_set)
    new_cf_fraction = float(new_cf_edges / max(1, len(cf_edges_set)))

    # Relative probe delta ||S4 - S0||_F / ||S0||_F
    diff = S4_sp - S0_sp
    fro_diff = float(np.sqrt(np.sum(diff.data ** 2)))
    fro_s0 = float(np.sqrt(np.sum(S0_sp.data ** 2)))
    rel_delta = float(fro_diff / max(1e-12, fro_s0))

    metadata = {
        "version": 4,
        "arm": arm,
        "active": True,
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "num_users": int(n_users),
        "num_items": int(n_items),
        "cache_hit": False,
        "s0_nnz": s0_nnz,
        "cf_nnz": cf_nnz,
        "s4_nnz": s4_nnz,
        "cf_isolated_nodes": cf_isolated,
        "cf_isolated_fraction": float(cf_isolated / n_items),
        "jaccard_semantic_cf": jaccard,
        "new_cf_edges_count": new_cf_edges,
        "new_cf_fraction": new_cf_fraction,
        "relative_operator_delta": rel_delta,
        "is_placebo": is_placebo,
    }

    if cache_path is not None:
        try:
            torch.save({"operator": S4_tensor, "metadata": metadata}, cache_path)
        except Exception:
            pass

    return GraphStateV4(operator=S4_tensor, metadata=metadata)
