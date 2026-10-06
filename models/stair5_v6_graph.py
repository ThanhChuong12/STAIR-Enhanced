"""models/stair5_v6_graph.py — STAIR5-v6 (NLGCL-BCSR) Graph Engine.
=================================================================
Behavior-Conditioned Semantic Retention (BCSR) for Item Update Smoothing.

Key Invariants & Mathematics:
1. Purely train-only: interactions R in {0, 1}^{M x N}, candidate-only intersections.
2. Candidate edges are strictly existing off-diagonal edges of raw semantic graph W_0.
3. Degree-null expected co-occurrence e_ij = (n_i * n_j) / M.
4. Under-support deficit h_ij = max(0, 1 - c_ij / e_ij) if e_ij > 0 else 0.
5. Opportunity shrinkage r_ij = e_ij / (e_ij + t_rel).
6. Suppression fraction a_ij = theta * r_ij * h_ij in [0, theta].
7. Weight reduction A_ij = W_{0, ij} * a_ij, removed row mass m_i = sum_j A_ij.
8. Raw retained graph W_R = W_0 - A + diag(m) preserves exact raw degrees: W_R 1 = W_0 1 = d^0.
9. Normalized retained operator S_R = D_0^{-1/2} W_R D_0^{-1/2} with isolated row zeros.
10. Final blended operator S_6 = (1 - eta) S_R + eta S_CF, spectral norm <= 1.0.
11. Exact theta=0 fast path returns bitwise v4 blended operator.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any, List
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v6_utils import atomic_torch_save, validate_csr_operator
from models import stair5_v4_graph as reference_graph

ARMS_V6 = (
    "P-BCSR",
    "C-V4",
    "C-Uniform",
    "C-NoShrink",
    "C-Shuffled",
    "C-NoRetention",
    "C-Renorm",
    "C-CF-Placebo",
)
ARM_ALIASES = {
    "bcsr": "P-BCSR",
    "p-bcsr": "P-BCSR",
    "v4": "C-V4",
    "theta0": "C-V4",
    "c-v4": "C-V4",
    "uniform": "C-Uniform",
    "c-uniform": "C-Uniform",
    "noshrink": "C-NoShrink",
    "c-noshrink": "C-NoShrink",
    "shuffled": "C-Shuffled",
    "c-shuffled": "C-Shuffled",
    "noretention": "C-NoRetention",
    "c-noretention": "C-NoRetention",
    "renorm": "C-Renorm",
    "c-renorm": "C-Renorm",
    "placebo": "C-CF-Placebo",
    "c-cf-placebo": "C-CF-Placebo",
}
CACHE_VERSION_V6 = 60


def canonicalize_arm_name(arm: str) -> str:
    key = str(arm).strip().lower()
    if key in ARM_ALIASES:
        return ARM_ALIASES[key]
    if arm in ARMS_V6:
        return arm
    raise ValueError(f"Unknown v6 arm: '{arm}'. Must be one of {ARMS_V6}")


@dataclass
class GraphStateV6:
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
            shape=tuple(matrix.shape),
        )
    if matrix.layout == torch.sparse_coo:
        matrix = matrix.coalesce()
        indices = matrix.indices().numpy()
        return sp.csr_matrix(
            (matrix.values().numpy(), (indices[0], indices[1])),
            shape=tuple(matrix.shape),
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
        size=matrix.shape,
    )


def compute_evidence_score(c: float, n_i: float, n_j: float, t: float = 5.0) -> float:
    """Unchanged direct CF evidence scoring from v4 baseline."""
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Evidence shrinkage must be finite and positive.")
    if n_i <= 0 or n_j <= 0 or c <= 0:
        return 0.0
    return float(c / (c + t) * c / math.sqrt(n_i * n_j))


def _count_sorted_intersection(a: np.ndarray, b: np.ndarray) -> int:
    """Exact two-pointer common user count between two sorted 1D arrays."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0 or a[0] > b[-1] or b[0] > a[-1]:
        return 0
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i, j, count = 0, 0, 0
    while i < la and j < lb:
        va, vb = a[i], b[j]
        if va == vb:
            count += 1
            i += 1
            j += 1
        elif va < vb:
            i += 1
        else:
            j += 1
    return count


def compute_candidate_intersections(
    R_T: sp.csr_matrix,
    cand_i: np.ndarray,
    cand_j: np.ndarray,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Computes exact common-user counts c_ij for candidate pairs (cand_i, cand_j) in bounded chunks.

    R_T is the CSR matrix of shape (n_items, n_users) with sorted user indices per item.
    Returns 1D int64 array of common user counts of length len(cand_i).
    """
    total = len(cand_i)
    if total == 0:
        return np.empty(0, dtype=np.int64)
    counts = np.empty(total, dtype=np.int64)
    indptr = R_T.indptr
    indices = R_T.indices

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        for k in range(start, stop):
            i = cand_i[k]
            j = cand_j[k]
            u_i = indices[indptr[i] : indptr[i + 1]]
            u_j = indices[indptr[j] : indptr[j + 1]]
            counts[k] = _count_sorted_intersection(u_i, u_j)
    return counts


def compute_under_support_scores(
    n_i: np.ndarray,
    n_j: np.ndarray,
    c_ij: np.ndarray,
    M: int,
    theta: float = 0.25,
    t_rel: float = 5.0,
    arm: str = "P-BCSR",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes degree-null expected counts e, under-support deficits h, shrinkage r, and fractions a.

    All computations are in float64 for numerical precision.
    Returns (e_ij, h_ij, r_ij, a_ij).
    """
    if M <= 0:
        raise ValueError("Universe user count M must be positive.")
    if theta < 0 or theta > 0.5:
        raise ValueError(f"theta must be in [0, 0.5], got {theta}")
    if t_rel <= 0 or not math.isfinite(t_rel):
        raise ValueError(f"t_rel must be finite and > 0, got {t_rel}")

    n_i = n_i.astype(np.float64, copy=False)
    n_j = n_j.astype(np.float64, copy=False)
    c_ij = c_ij.astype(np.float64, copy=False)

    # Expected co-occurrence under degree-null model
    e_ij = (n_i * n_j) / float(M)

    # Deficit h_ij = [1 - c_ij / e_ij]_+ for e_ij > 0
    h_ij = np.zeros_like(e_ij)
    valid_e = e_ij > 0.0
    if np.any(valid_e):
        ratio = c_ij[valid_e] / e_ij[valid_e]
        h_ij[valid_e] = np.maximum(0.0, 1.0 - ratio)

    # Opportunity shrinkage r_ij = e_ij / (e_ij + t_rel)
    r_ij = np.zeros_like(e_ij)
    if np.any(valid_e):
        r_ij[valid_e] = e_ij[valid_e] / (e_ij[valid_e] + float(t_rel))

    # Suppression fraction a_ij
    arm_canonical = canonicalize_arm_name(arm)
    if arm_canonical == "C-NoShrink":
        a_ij = float(theta) * h_ij
    else:
        a_ij = float(theta) * r_ij * h_ij

    # Clamp to [0, theta] to eliminate any tiny FP roundoff
    a_ij = np.clip(a_ij, 0.0, float(theta))
    return e_ij, h_ij, r_ij, a_ij


def build_candidate_support_graph(
    train_edges: torch.Tensor,
    n_items: int,
    n_users: int,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
) -> sp.csr_matrix:
    """Exact delegation to v4 CF graph builder for direct collaborative support."""
    return reference_graph.build_candidate_support_graph(
        train_edges=train_edges,
        n_items=n_items,
        n_users=n_users,
        k_cf=k_cf,
        c_min=c_min,
        t_shrinkage=t_shrinkage,
        block_size=block_size,
        memory_budget_mib=memory_budget_mib,
    )


def normalize_cf_graph_with_fallback(W_cf: sp.csr_matrix) -> sp.csr_matrix:
    """Exact delegation to v4 CF normalizer with isolated node identity fallback."""
    return reference_graph.normalize_cf_graph_with_fallback(W_cf)


def build_retained_semantic_graph(
    W_0: sp.csr_matrix,
    d_0: np.ndarray,
    cand_i: np.ndarray,
    cand_j: np.ndarray,
    a_ij: np.ndarray,
    n_items: int,
    arm: str = "P-BCSR",
    seed: int = 1,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Builds retained raw graph W_R and weight reduction matrix A.

    Preserves exact raw row sums: W_R 1 = W_0 1 = d^0.
    Returns (W_R, A_matrix, m_diagonal, diagnostics).
    """
    arm_canonical = canonicalize_arm_name(arm)
    total_candidates = len(cand_i)
    diagnostics = {}

    if arm_canonical == "C-Uniform":
        # Uniform retention with matched total removed mass
        # Total removed mass if primary a_ij were used
        cand_w0 = np.asarray(W_0[cand_i, cand_j]).ravel()
        primary_removed_mass = np.sum(cand_w0 * a_ij) * 2.0
        total_semantic_offdiag_mass = float(W_0.sum())  # W_0 has zero diagonal
        a_bar = primary_removed_mass / max(1e-12, total_semantic_offdiag_mass)
        a_bar = min(a_bar, 0.5)
        effective_a = np.full(total_candidates, a_bar, dtype=np.float64)
        diagnostics["uniform_a_bar"] = float(a_bar)
    elif arm_canonical == "C-Shuffled":
        # Symmetric shuffled suppression fractions across candidate edges
        rng = np.random.RandomState(seed)
        effective_a = a_ij.copy()
        rng.shuffle(effective_a)
        diagnostics["shuffled_seed"] = int(seed)
    else:
        effective_a = a_ij

    # Extract W_0 values for candidate pairs
    cand_w0 = np.asarray(W_0[cand_i, cand_j]).ravel().astype(np.float64)
    cand_A = cand_w0 * effective_a

    # Build symmetric weight reduction matrix A
    # Symmetrize candidates: both (i, j) and (j, i)
    sym_rows = np.concatenate([cand_i, cand_j])
    sym_cols = np.concatenate([cand_j, cand_i])
    sym_data = np.concatenate([cand_A, cand_A])

    A_mat = sp.csr_matrix((sym_data, (sym_rows, sym_cols)), shape=(n_items, n_items), dtype=np.float64)
    A_mat.sum_duplicates()
    A_mat.eliminate_zeros()

    # Removed row mass m_i = sum_j A_ij
    m_i = np.asarray(A_mat.sum(axis=1)).ravel().astype(np.float64)

    # Build W_R:
    if arm_canonical in ("C-NoRetention", "C-Renorm"):
        # Without diagonal retention: W_R = W_0 - A
        W_R = (W_0.astype(np.float64) - A_mat).tocsr()
        W_R.eliminate_zeros()
        retained_diagonal = False
    else:
        # Default & primary: W_R = W_0 - A + diag(m)
        W_diff = (W_0.astype(np.float64) - A_mat).tocsr()
        diag_m = sp.diags(m_i, offsets=0, shape=(n_items, n_items), format="csr", dtype=np.float64)
        W_R = (W_diff + diag_m).tocsr()
        W_R.eliminate_zeros()
        retained_diagonal = True

    W_R.sort_indices()
    # Check degree preservation when diagonal retention is active
    if retained_diagonal:
        actual_degrees = np.asarray(W_R.sum(axis=1)).ravel()
        max_deg_error = float(np.max(np.abs(actual_degrees - d_0)))
        diagnostics["degree_conservation_max_error"] = max_deg_error
        if max_deg_error > 1e-6:
            raise RuntimeError(f"Degree conservation invariant violated! Max error: {max_deg_error}")

    diagnostics.update({
        "retained_diagonal": retained_diagonal,
        "total_removed_edge_mass": float(np.sum(cand_A) * 2.0),
        "total_retained_diagonal_mass": float(np.sum(m_i)),
        "affected_edges_count": int(np.count_nonzero(effective_a > 0)),
        "affected_edges_fraction": float(np.mean(effective_a > 0)) if total_candidates > 0 else 0.0,
        "mean_effective_a": float(np.mean(effective_a[effective_a > 0])) if np.any(effective_a > 0) else 0.0,
    })
    return W_R, A_mat, m_i, diagnostics


def normalize_retained_semantic_graph(
    W_R: sp.csr_matrix,
    d_0: np.ndarray,
    arm: str = "P-BCSR",
) -> sp.csr_matrix:
    """Normalizes retained semantic graph W_R.

    For primary and standard arms: S_R = D_0^{-1/2} W_R D_0^{-1/2} (isolated rows zero).
    For C-Renorm arm: S_R = D_R^{-1/2} W_R D_R^{-1/2} with new degrees.
    """
    arm_canonical = canonicalize_arm_name(arm)
    if arm_canonical == "C-Renorm":
        deg = np.asarray(W_R.sum(axis=1)).ravel().astype(np.float64)
    else:
        deg = d_0.astype(np.float64)

    inv_sqrt = np.zeros(len(deg), dtype=np.float64)
    pos = deg > 0.0
    inv_sqrt[pos] = 1.0 / np.sqrt(deg[pos])
    scale = sp.diags(inv_sqrt, offsets=0, shape=W_R.shape, format="csr", dtype=np.float64)

    S_R = (scale @ W_R.astype(np.float64) @ scale).astype(np.float32).tocsr()
    S_R.eliminate_zeros()
    S_R.sort_indices()
    return S_R


def build_calibrated_graph_v6(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    arm: str = "P-BCSR",
    theta: float = 0.25,
    t_rel: float = 5.0,
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    placebo_seed: int = 1,
    seed: int = 1,
    cache_dir: Optional[str] = None,
    candidate_chunk_size: int = 65536,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
) -> GraphStateV6:
    """Builds the complete STAIR5-v6 blended graph operator S_6 = (1 - eta) S_R + eta S_CF.

    Fully cached, content-addressed, mathematically rigorous, with exact theta=0 v4 parity.
    """
    arm_canonical = canonicalize_arm_name(arm)
    if not math.isfinite(eta) or not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    if theta < 0.0 or theta > 0.5:
        raise ValueError(f"theta must be in [0, 0.5], got {theta}")
    if t_rel <= 0.0 or not math.isfinite(t_rel):
        raise ValueError(f"t_rel must be finite and > 0, got {t_rel}")
    if tuple(baseline_normalized.shape) != (n_items, n_items):
        raise ValueError("Semantic graph shape differs from the catalog count.")

    metadata = {
        "version": 6,
        "arm": arm_canonical,
        "theta": float(theta),
        "t_rel": float(t_rel),
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "num_users": int(n_users),
        "num_items": int(n_items),
        "cache_hit": False,
    }

    # FAST-PATH FOR THETA = 0 OR EXACT V4 CONTROL:
    if arm_canonical == "C-V4" or theta == 0.0:
        # Exact bitwise v4 graph recovery
        v4_state = reference_graph.build_calibrated_graph_v4(
            raw_semantic=raw_semantic,
            baseline_normalized=baseline_normalized,
            train_edges=train_edges,
            n_users=n_users,
            n_items=n_items,
            arm="N-CSE",
            eta=eta,
            k_cf=k_cf,
            c_min=c_min,
            t_shrinkage=t_shrinkage,
            placebo_seed=placebo_seed,
            seed=seed,
            cache_dir=cache_dir,
            block_size=block_size,
            memory_budget_mib=memory_budget_mib,
        )
        metadata.update(v4_state.metadata)
        metadata.update({
            "version": 6,
            "arm": arm_canonical,
            "theta": float(theta),
            "t_rel": float(t_rel),
            "fast_path_theta0": True,
            "graph_fingerprint": v4_state.metadata.get("graph_fingerprint", sparse_fingerprint(tensor_to_scipy(v4_state.operator))),
        })
        return GraphStateV6(v4_state.operator, metadata)

    started = time.perf_counter()

    # Deduplicate train keys and build binary R
    pairs = binary_train_keys(train_edges, n_users, n_items)
    pair_hash = hashlib.sha256()
    _digest_array(pair_hash, pairs)
    train_hash = pair_hash.hexdigest()

    # Canonicalize raw semantic W_0 from raw_semantic tensor
    W_0 = tensor_to_scipy(raw_semantic).astype(np.float64).tocsr()
    W_0.sum_duplicates()
    W_0.eliminate_zeros()
    W_0.sort_indices()
    # Mask diagonal if any
    W_0.setdiag(0)
    W_0.eliminate_zeros()
    validate_csr_operator(W_0, n_items)

    d_0 = np.asarray(W_0.sum(axis=1)).ravel().astype(np.float64)

    # Content-addressed cache signature
    signature = {
        "cache_version": CACHE_VERSION_V6,
        "train": train_hash,
        "users": int(n_users),
        "items": int(n_items),
        "arm": arm_canonical,
        "theta": float(theta),
        "t_rel": float(t_rel),
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "placebo_seed": int(placebo_seed) if arm_canonical == "C-CF-Placebo" else 0,
        "w0_fingerprint": sparse_fingerprint(W_0),
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"s6_{cache_key}.pt" if cache_dir else None

    if cache_file and cache_file.is_file():
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=True)
            cached_csr = tensor_to_scipy(payload["operator"])
            if payload["signature"] != signature or payload["fingerprint"] != sparse_fingerprint(cached_csr):
                raise ValueError("Cache signature or fingerprint mismatch.")
            if cached_csr.shape != (n_items, n_items):
                raise ValueError("Cache shape mismatch.")
            cached_meta = payload.get("metadata", {})
            metadata.update(cached_meta)
            metadata["cache_hit"] = True
            metadata["bcsr_preprocessing_seconds"] = time.perf_counter() - started
            return GraphStateV6(scipy_to_tensor(cached_csr).to(baseline_normalized.device), metadata)
        except (OSError, RuntimeError, ValueError, KeyError, EOFError) as err:
            warnings.warn(f"Ignoring invalid STAIR5-v6 cache {cache_file.name}: {err}", RuntimeWarning)

    # 1. Build binary interaction matrix R and CSC transposed items R_T
    R = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.int64), (pairs // n_items, pairs % n_items)),
        shape=(n_users, n_items),
    )
    R_T = R.T.tocsr()
    R_T.sort_indices()
    item_degrees = np.diff(R_T.indptr).astype(np.float64)

    # 2. Extract strictly off-diagonal candidate edges (i < j) from W_0
    coo = sp.triu(W_0, k=1).tocoo()
    cand_i = coo.row.astype(np.int64)
    cand_j = coo.col.astype(np.int64)
    num_candidates = len(cand_i)

    # 3. Compute common-user intersection counts c_ij
    c_ij = compute_candidate_intersections(R_T, cand_i, cand_j, chunk_size=candidate_chunk_size)

    # 4. Compute under-support statistics
    n_i = item_degrees[cand_i]
    n_j = item_degrees[cand_j]
    e_ij, h_ij, r_ij, a_ij = compute_under_support_scores(
        n_i=n_i,
        n_j=n_j,
        c_ij=c_ij,
        M=n_users,
        theta=theta,
        t_rel=t_rel,
        arm=arm_canonical,
    )

    # 5. Build retained raw semantic graph W_R and weight reduction A
    W_R, A_mat, m_i, retention_diagnostics = build_retained_semantic_graph(
        W_0=W_0,
        d_0=d_0,
        cand_i=cand_i,
        cand_j=cand_j,
        a_ij=a_ij,
        n_items=n_items,
        arm=arm_canonical,
        seed=seed,
    )

    # 6. Normalize retained semantic graph S_R
    S_R = normalize_retained_semantic_graph(W_R, d_0, arm=arm_canonical)

    # 7. Build direct collaborative support operator S_CF
    W_cf = build_candidate_support_graph(
        train_edges=train_edges,
        n_items=n_items,
        n_users=n_users,
        k_cf=k_cf,
        c_min=c_min,
        t_shrinkage=t_shrinkage,
        block_size=block_size,
        memory_budget_mib=memory_budget_mib,
    )
    if arm_canonical == "C-CF-Placebo":
        W_cf, placebo_meta = reference_graph.permute_cf_graph_degree_stratified(
            W_cf, train_edges, n_items, placebo_seed, return_metadata=True
        )
        metadata.update(placebo_meta)

    S_CF = normalize_cf_graph_with_fallback(W_cf)

    # 8. Blend final operator: S_6 = (1 - eta) S_R + eta S_CF
    S_6 = ((1.0 - eta) * S_R.astype(np.float64) + eta * S_CF.astype(np.float64)).astype(np.float32).tocsr()
    S_6.eliminate_zeros()
    S_6.sort_indices()
    validate_csr_operator(S_6, n_items)

    elapsed = time.perf_counter() - started
    s6_fingerprint = sparse_fingerprint(S_6)

    # Compile comprehensive metadata & telemetry
    metadata.update(retention_diagnostics)
    metadata.update({
        "train_fingerprint": train_hash,
        "s6_cache_key": cache_key,
        "bcsr_preprocessing_seconds": float(elapsed),
        "num_candidate_semantic_pairs": int(num_candidates),
        "mean_expected_overlap_e": float(np.mean(e_ij)) if num_candidates > 0 else 0.0,
        "mean_deficit_h": float(np.mean(h_ij)) if num_candidates > 0 else 0.0,
        "mean_shrinkage_r": float(np.mean(r_ij)) if num_candidates > 0 else 0.0,
        "mean_raw_a": float(np.mean(a_ij)) if num_candidates > 0 else 0.0,
        "s0_nnz": int(baseline_normalized._nnz()),
        "w0_nnz": int(W_0.nnz),
        "sr_nnz": int(S_R.nnz),
        "cf_nnz": int(W_cf.nnz),
        "s6_nnz": int(S_6.nnz),
        "s6_spectral_norm_bound": 1.0,
        "graph_fingerprint": s6_fingerprint,
    })

    if cache_file:
        atomic_torch_save(
            {
                "operator": scipy_to_tensor(S_6),
                "signature": signature,
                "fingerprint": s6_fingerprint,
                "metadata": metadata,
            },
            cache_file,
        )

    return GraphStateV6(scipy_to_tensor(S_6).to(baseline_normalized.device), metadata)
