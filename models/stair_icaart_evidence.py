"""models/stair_icaart_evidence.py — Full-activity pooled evidence and legacy replicated controls.
====================================================================================================
Implements:
1. Deterministic activity stratification b(u) = floor(log2(h_u)) on deduplicated TRAIN data.
2. Disjoint user splitting f in {0, 1} with deterministic seeded hash (no Python randomized hash).
3. Candidate-only observed co-interactions c_ijf and activity-null expectations mu_ijf.
4. Full-activity pooled excess with kappa=2*tau, preserving all legacy half-based controls.

Contracts:
- Runs strictly on CPU in bounded blocks (default candidate_block = 8192).
- Never allocates dense N x N matrices or full R.T @ R.
- Zero gradients, zero learned parameters.
"""

import hashlib
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp


def deterministic_user_hash(evidence_seed: int, user_id: int) -> int:
    """Compute a deterministic 64-bit unsigned integer hash for (evidence_seed, user_id).

    Uses SHA-256 to guarantee bitwise reproducibility across platforms and Python runs,
    completely independent of Python's randomized hash() seed or PyTorch global RNG.
    """
    token = f"{evidence_seed}:{user_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def build_activity_replication(
    R_csr: sp.csr_matrix,
    evidence_seed: int = 20261016,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratifies users by train activity and partitions active users into disjoint halves.

    Mathematical formulation (Eq 3-4):
    - h_u = sum_i R_{ui} (train activity only)
    - Zero-history users (h_u == 0): excluded from strata, assigned user_bin = -1, half = -1.
    - Active users (h_u >= 1): b(u) = floor(log2(h_u)).
    - Within each non-empty stratum b: sort active users by (deterministic_hash, user_id),
      then alternate assignment f in {0, 1}.
    - Singleton strata (|U_b| == 1): deterministically assigned to half 0.

    Args:
        R_csr: Deduplicated binary interaction matrix [M, N] in CSR format (CPU).
        evidence_seed: Integer seed for user hashing. Default 20261016.

    Returns:
        user_bin: int64 array [M] of activity strata (-1 for inactive).
        half: int64 array [M] of user halves {0, 1} (-1 for inactive).
        group_sizes: int64 array [2, B] of user counts per (half, stratum).
    """
    if not sp.isspmatrix_csr(R_csr):
        raise TypeError("R_csr must be a scipy.sparse.csr_matrix on CPU")
    if isinstance(evidence_seed, bool) or not isinstance(evidence_seed, (int, np.integer)) or evidence_seed < 0:
        raise ValueError("evidence_seed must be a non-negative integer")
    R_csr = _canonical_presence(R_csr, "csr")

    M, N = R_csr.shape
    if M <= 0 or N <= 0:
        raise ValueError(f"Invalid matrix shape: ({M}, {N})")

    # h_u: user interaction count (train activity)
    h_u = np.diff(R_csr.indptr).astype(np.int64)

    user_bin = np.full(M, -1, dtype=np.int64)
    half = np.full(M, -1, dtype=np.int64)

    active_mask = h_u > 0
    active_users = np.flatnonzero(active_mask)

    if len(active_users) == 0:
        return user_bin, half, np.zeros((2, 0), dtype=np.int64)

    # b(u) = floor(log2(max(1, h_u)))
    active_bins = np.floor(np.log2(h_u[active_users])).astype(np.int64)
    user_bin[active_users] = active_bins

    B = int(np.max(active_bins)) + 1
    group_sizes = np.zeros((2, B), dtype=np.int64)

    for b in range(B):
        u_in_b = active_users[active_bins == b]
        num_u = len(u_in_b)
        if num_u == 0:
            continue
        elif num_u == 1:
            u_single = u_in_b[0]
            half[u_single] = 0
            group_sizes[0, b] += 1
        else:
            # Deterministic seeded hash with user_id tie-breaker
            hashes = np.fromiter(
                (deterministic_user_hash(evidence_seed, int(u)) for u in u_in_b),
                dtype=np.uint64, count=num_u,
            )
            order = np.lexsort((u_in_b, hashes))
            sorted_u = u_in_b[order]
            half[sorted_u] = np.arange(num_u, dtype=np.int64) % 2
            group_sizes[:, b] = ((num_u + 1) // 2, num_u // 2)

    return user_bin, half, group_sizes


def _canonical_presence(matrix: sp.spmatrix, format: str) -> sp.spmatrix:
    """Return private binary, sorted storage without changing caller training data."""
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Interaction counts must be finite and non-negative")
    # Boolean duplicate reduction avoids int8 count overflow inside this conversion.
    result = matrix.astype(bool).asformat(format, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def build_activity_strata(R_csr: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """Full-user log2 activity strata; inactive users are excluded, with no RNG or split."""
    if not sp.isspmatrix_csr(R_csr):
        raise TypeError("R_csr must be a scipy.sparse.csr_matrix")
    if min(R_csr.shape) <= 0:
        raise ValueError("Interaction dimensions must be positive")
    matrix = _canonical_presence(R_csr, "csr")
    activity = np.diff(matrix.indptr).astype(np.int64)
    bins = np.full(matrix.shape[0], -1, dtype=np.int64)
    active = activity > 0
    bins[active] = np.floor(np.log2(activity[active])).astype(np.int64)
    sizes = np.bincount(bins[active]).astype(np.int64)
    return bins, sizes


def candidate_pooled_statistics(
    R_csc: sp.csc_matrix,
    E_sem: np.ndarray,
    user_bin: np.ndarray,
    group_sizes: np.ndarray,
    candidate_block: int = 8192,
    *, null_scope: str = "full_activity",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute full-user c[E], mu[E] on semantic candidates, without half-conditioned nulls.

    mu_ij = sum_b n_ib*n_jb/M_b, using binary TRAIN presence. With
    null_scope='global_full', one pool contains all and only active TRAIN users,
    giving mu_ij=n_i*n_j/M_active. Observed overlap counting is identical.
    Integer counters
    are promoted before multiplication. Temporary expectation arrays are block-by-bin,
    never item-by-item. Posting searches execute in NumPy, not per-history pair loops.
    """
    if not sp.isspmatrix_csc(R_csc):
        raise TypeError("R_csc must be a scipy.sparse.csc_matrix")
    if (isinstance(candidate_block, bool)
            or not isinstance(candidate_block, (int, np.integer)) or candidate_block < 1):
        raise ValueError("candidate_block must be a positive integer")
    matrix = _canonical_presence(R_csc, "csc")
    n_users, n_items = matrix.shape
    if min(matrix.shape) <= 0:
        raise ValueError("Interaction dimensions must be positive")
    pairs = np.asarray(E_sem)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0) or np.any(pairs >= n_items)
            or np.any(pairs[:, 0] >= pairs[:, 1])):
        raise ValueError("Candidate pairs must be integer [E,2], in catalog, with i < j")
    pairs = pairs.astype(np.int64, copy=False)
    if len(np.unique(pairs[:, 0] * n_items + pairs[:, 1])) != len(pairs):
        raise ValueError("Candidate pairs must be unique")
    if null_scope not in ("full_activity", "global_full"):
        raise ValueError("Unknown full-user null scope")
    bins, sizes = np.asarray(user_bin), np.asarray(group_sizes)
    activity = np.bincount(matrix.indices, minlength=n_users)
    active = activity > 0
    expected_bins = np.full(n_users, -1, dtype=np.int64)
    if null_scope == "global_full":
        expected_bins[active] = 0
        expected_sizes = np.array([np.count_nonzero(active)], dtype=np.int64)
    else:
        expected_bins[active] = np.floor(np.log2(activity[active])).astype(np.int64)
        expected_sizes = np.bincount(expected_bins[active])
    if (bins.shape != (n_users,) or bins.dtype.kind not in "iu"
            or sizes.ndim != 1 or sizes.dtype.kind not in "iu"
            or not np.array_equal(bins, expected_bins)
            or not np.array_equal(sizes, expected_sizes)):
        raise ValueError("Null groups must match the selected full-user TRAIN scope exactly")

    counts = np.zeros(len(pairs), dtype=np.int64)
    expected = np.zeros(len(pairs), dtype=np.float64)
    if not len(pairs) or not active.any():
        return counts, expected
    n_bins = len(sizes)
    item_ids = np.repeat(np.arange(n_items, dtype=np.int64), np.diff(matrix.indptr))
    flat = item_ids * n_bins + bins[matrix.indices]
    item_counts = np.bincount(flat, minlength=n_items * n_bins).reshape(n_items, n_bins)
    del flat, item_ids
    inverse = np.zeros(n_bins, dtype=np.float64)
    np.divide(1.0, sizes, out=inverse, where=sizes > 0)
    for start in range(0, len(pairs), candidate_block):
        stop = min(start + candidate_block, len(pairs))
        left_ids, right_ids = pairs[start:stop].T
        for offset, (i, j) in enumerate(zip(left_ids, right_ids)):
            left = matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]
            right = matrix.indices[matrix.indptr[j]:matrix.indptr[j + 1]]
            if len(left) > len(right):
                left, right = right, left
            positions = np.searchsorted(right, left)
            valid = positions < len(right)
            selected = left[valid]
            counts[start + offset] = np.count_nonzero(right[positions[valid]] == selected)
        products = item_counts[left_ids].astype(np.float64)
        products *= item_counts[right_ids].astype(np.float64)
        expected[start:stop] = products @ inverse
    if np.any((counts > 0) & (expected <= 0)):
        raise ArithmeticError("Positive overlap must imply positive null expectation")
    return counts, expected


def degree_stratified_permutation(
    scores: np.ndarray, degrees: np.ndarray, pairs: np.ndarray, seed: int,
) -> np.ndarray:
    """Shuffle the supplied score family on unordered endpoint-degree quintile pairs."""
    scores, degrees, pairs = map(np.asarray, (scores, degrees, pairs))
    if (scores.ndim != 1 or not np.isfinite(scores).all()
            or degrees.ndim != 1 or not np.isfinite(degrees).all() or np.any(degrees < 0)
            or pairs.shape != (len(scores), 2) or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0) or np.any(pairs >= len(degrees))
            or np.any(pairs[:, 0] >= pairs[:, 1])):
        raise ValueError("Invalid scores, train degrees or unordered candidate pairs")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Permutation seed must be a non-negative integer")
    if not len(scores):
        return scores.astype(np.float64, copy=True)
    cuts = np.unique(np.quantile(degrees, [.2, .4, .6, .8]))
    strata = np.searchsorted(cuts, degrees, side="right")
    left, right = strata[pairs[:, 0]], strata[pairs[:, 1]]
    groups = np.minimum(left, right) * 5 + np.maximum(left, right)
    rng = np.random.default_rng(seed)
    result = scores.astype(np.float64, copy=True)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        result[indices] = rng.permutation(scores[indices])
    return result


def pooled_activity_score(
    c: np.ndarray, mu: np.ndarray, tau: float = 2.0,
    mode: str = "pooled_activity_excess", *, degrees: Optional[np.ndarray] = None,
    E_sem: Optional[np.ndarray] = None, placebo_seed: int = 20261016,
) -> np.ndarray:
    """PEEC-full: [c-mu]+/(c+mu+2*tau), or its explicitly labeled count/placebo controls.

    The vectors are full-stratum statistics, not sums of half-specific expectations.
    tau retains its legacy count-scale convention; pooled kappa is exactly 2*tau.
    """
    from stair_icaart_config import FULL_ACTIVITY_MODES
    if mode not in FULL_ACTIVITY_MODES:
        raise ValueError(f"Unknown full-activity evidence mode: {mode}")
    if isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0 or not math.isfinite(2 * tau):
        raise ValueError("tau and pooled kappa must be finite and positive")
    c, mu = np.asarray(c), np.asarray(mu)
    if (c.ndim != 1 or c.dtype.kind not in "iu" or mu.shape != c.shape
            or np.any(c < 0) or not np.isfinite(mu).all() or np.any(mu < 0)):
        raise ValueError("Full counts must be integer [E]; expectations must be finite nonnegative [E]")
    if np.any((c > 0) & (mu <= 0)):
        raise ArithmeticError("Positive overlap must imply positive null expectation")
    counts = c.astype(np.float64)
    expected = mu.astype(np.float64)
    if mode == "pooled_activity_count":
        scores = counts / (counts + 2.0 * tau)
    else:
        scores = np.maximum(0.0, counts - expected) / (counts + expected + 2.0 * tau)
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores >= 1):
        raise ArithmeticError("Pooled evidence score must lie in [0,1)")
    if mode == "pooled_activity_shuffle":
        if degrees is None or E_sem is None:
            raise ValueError("Pooled placebo requires TRAIN item degrees and candidate pairs")
        scores = degree_stratified_permutation(scores, degrees, E_sem, placebo_seed)
    return scores


def candidate_excess_statistics(
    R_csc: sp.csc_matrix,
    E_sem: np.ndarray,
    half: np.ndarray,
    user_bin: np.ndarray,
    group_sizes: np.ndarray,
    candidate_block: int = 8192,
    mode: str = "replicated_excess",
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes observed co-interaction counts c_ijf and conditional null expectations mu_ijf.

    Only evaluates unique undirected candidate pairs (i, j) in E_sem with i < j.
    Uses bounded block processing (default candidate_block = 8192) to strictly limit memory.

    Mathematical formulation (Eq 4):
    - M_fb = |{u : f(u) = f, b(u) = b}|
    - n_ifb = sum_{u: f(u)=f, b(u)=b} R_{ui}
    - c_ijf = sum_{u: f(u)=f} R_{ui} R_{uj}
    - mu_ijf = sum_{b: M_fb > 0} (n_ifb * n_jfb) / M_fb

    Args:
        R_csc: Binary interaction matrix [M, N] in CSC format (CPU).
        E_sem: int64 array [E, 2] of candidate item pairs (i, j) with i < j.
        half: int64 array [M] of user halves {0, 1} (-1 for inactive).
        user_bin: int64 array [M] of user strata (-1 for inactive).
        group_sizes: int64 array [2, B] of user group sizes.
        candidate_block: Maximum candidate edges processed per chunk.
        mode: Evidence mode (supports 'unstratified_replication' ablation).

    Returns:
        c: int64 array [E, 2] of observed co-occurrence counts.
        mu: float64 array [E, 2] of null expectation values.
    """
    if not sp.isspmatrix_csc(R_csc):
        raise TypeError("R_csc must be a scipy.sparse.csc_matrix on CPU")
    if (isinstance(candidate_block, bool)
            or not isinstance(candidate_block, (int, np.integer)) or candidate_block < 1):
        raise ValueError("candidate_block must be a positive integer")
    if mode not in ("replicated_excess", "v8_exact", "pooled_excess",
                    "unstratified_replication", "replicated_count",
                    "degree_stratified_shuffle", "uniform_multiplier"):
        raise ValueError(f"Unknown evidence mode: {mode}")
    R = _canonical_presence(R_csc, "csc")
    M, N = R.shape
    E_sem = np.asarray(E_sem)
    if E_sem.ndim != 2 or E_sem.shape[1] != 2 or E_sem.dtype.kind not in "iu":
        raise ValueError("E_sem must be an integer array of shape [E, 2]")
    if np.any(E_sem < 0) or np.any(E_sem >= N) or np.any(E_sem[:, 0] >= E_sem[:, 1]):
        raise ValueError("Candidate pairs must be in catalog and strictly satisfy i < j")
    if N and len(np.unique(E_sem[:, 0].astype(np.int64) * N + E_sem[:, 1])) != len(E_sem):
        raise ValueError("Candidate pairs must be unique")
    half, user_bin, group_sizes = map(np.asarray, (half, user_bin, group_sizes))
    if (half.shape != (M,) or user_bin.shape != (M,)
            or half.dtype.kind not in "iu" or user_bin.dtype.kind not in "iu"
            or group_sizes.ndim != 2 or group_sizes.shape[0] != 2
            or group_sizes.dtype.kind not in "iu" or np.any(group_sizes < 0)):
        raise ValueError("Invalid user assignments or group_sizes shape/dtype")
    B = group_sizes.shape[1]
    active = np.bincount(R.indices, minlength=M) > 0
    if (np.any(~np.isin(half[active], (0, 1))) or np.any(user_bin[active] < 0)
            or np.any(user_bin[active] >= B)
            or np.any(half[~active] != -1) or np.any(user_bin[~active] != -1)):
        raise ValueError("Active users require valid groups; inactive users require group ID -1")
    actual_sizes = np.bincount(
        half[active] * B + user_bin[active], minlength=2 * B,
    ).reshape(2, B)
    if not np.array_equal(actual_sizes, group_sizes):
        raise ArithmeticError("group_sizes do not match the disjoint active-user assignments")

    E = len(E_sem)
    c = np.zeros((E, 2), dtype=np.int64)
    mu = np.zeros((E, 2), dtype=np.float64)
    if E == 0:
        return c, mu
    if not active.any():
        return c, mu

    # One streaming incidence pass; no split R0/R1 matrices and no M x block product.
    item_ids = np.repeat(np.arange(N, dtype=np.int64), np.diff(R.indptr))
    users = R.indices
    flat_groups = item_ids * (2 * B) + half[users] * B + user_bin[users]
    n_ifb = np.bincount(flat_groups, minlength=N * 2 * B).reshape(N, 2, B)
    del item_ids, flat_groups
    inv_sizes = np.zeros((2, B), dtype=np.float64)
    np.divide(1.0, group_sizes, out=inv_sizes, where=group_sizes > 0)
    n_if = n_ifb.sum(axis=2).astype(np.float64)
    inv_half = np.zeros(2, dtype=np.float64)
    half_sizes = group_sizes.sum(axis=1)
    np.divide(1.0, half_sizes, out=inv_half, where=half_sizes > 0)

    # NumPy searchsorted executes the posting-list search in compiled code. Python
    # dispatch is per eligible edge, never per user-history pair. Peak intersection
    # scratch is O(min(n_i,n_j)), independent of M*candidate_block or N*N.
    for start in range(0, E, candidate_block):
        stop = min(start + candidate_block, E)
        src, dst = E_sem[start:stop].T
        for offset, (i, j) in enumerate(zip(src, dst)):
            left = R.indices[R.indptr[i]:R.indptr[i + 1]]
            right = R.indices[R.indptr[j]:R.indptr[j + 1]]
            if len(left) > len(right):
                left, right = right, left
            if not len(left):
                continue
            positions = np.searchsorted(right, left)
            valid = positions < len(right)
            selected = left[valid]
            common = selected[right[positions[valid]] == selected]
            c[start + offset] = np.bincount(half[common], minlength=2)
        if mode == "unstratified_replication":
            mu[start:stop] = n_if[src] * n_if[dst] * inv_half
        else:
            # Cast each operand before multiplication; do not overflow int counters.
            left_counts = n_ifb[src].astype(np.float64)
            right_counts = n_ifb[dst].astype(np.float64)
            mu[start:stop] = np.sum(left_counts * right_counts * inv_sizes, axis=2)
        if np.any((c[start:stop] > 0) & (mu[start:stop] <= 0)):
            raise ArithmeticError("Positive overlap must imply positive null expectation")
    return c, mu


def replicated_excess_score(
    c: np.ndarray,
    mu: np.ndarray,
    tau: float = 2.0,
    mode: str = "replicated_excess",
    placebo_seed: int = 20261016,
    degrees: Optional[np.ndarray] = None,
    E_sem: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes bounded replicated excess score r in [0, 1) (Eq 5) and scientific controls.

    Primary formulation:
    - e_ijf = [c_ijf - mu_ijf]_+ / (c_ijf + mu_ijf + tau)
    - r_ij = min(e_ij0, e_ij1)

    Guarantees:
    - 0 <= r_ij < 1.0
    - Primary replicated_excess is zero if either half has zero count.
    - Pooled/uniform controls intentionally have different neutrality semantics.
    - No division by zero, no NaN/Inf.

    Ablation modes:
    - 'replicated_excess': Primary replicated excess score (Eq 5).
    - 'v8_exact': Exactly r = 0 (recovers native V8).
    - 'pooled_excess': r = [c_pool - mu_pool]_+ / (c_pool + mu_pool + 2*tau).
    - 'unstratified_replication': Unstratified null expectation replication.
    - 'replicated_count': r = min(c0 / (c0 + tau), c1 / (c1 + tau)).
    - 'degree_stratified_shuffle': Permute r among pairs stratified by endpoint degree.
    - 'uniform_multiplier': Constant r across all candidate edges.
    """
    from stair_icaart_config import ALLOWED_EVIDENCE_MODES, FULL_ACTIVITY_MODES
    if mode not in ALLOWED_EVIDENCE_MODES:
        raise ValueError(f"Unknown evidence mode: {mode}")
    if isinstance(tau, bool) or tau <= 0 or not math.isfinite(tau):
        raise ValueError(f"tau must be finite and positive; got {tau}")
    if mode in FULL_ACTIVITY_MODES:
        raise ValueError("Full-activity modes require pooled_activity_score, not half statistics")
    c, mu = np.asarray(c), np.asarray(mu)
    if c.ndim != 2 or c.shape[1] != 2 or c.dtype.kind not in "iu" or mu.shape != c.shape:
        raise ValueError("c must be integer [E, 2] and mu must have the same shape")
    if np.any(c < 0) or not np.isfinite(mu).all() or np.any(mu < 0):
        raise ValueError("Counts and expectations must be finite and non-negative")
    if np.any((c > 0) & (mu <= 0)):
        raise ArithmeticError("Positive overlap must imply positive null expectation")
    E = c.shape[0]
    if E == 0:
        return np.empty(0, dtype=np.float64)

    c0 = c[:, 0].astype(np.float64)
    c1 = c[:, 1].astype(np.float64)
    mu0 = mu[:, 0].astype(np.float64)
    mu1 = mu[:, 1].astype(np.float64)

    if mode == "v8_exact":
        return np.zeros(E, dtype=np.float64)

    elif mode in ("replicated_excess", "unstratified_replication"):
        excess0 = np.maximum(0.0, c0 - mu0)
        excess1 = np.maximum(0.0, c1 - mu1)
        denom0 = c0 + mu0 + tau
        denom1 = c1 + mu1 + tau
        e0 = excess0 / denom0
        e1 = excess1 / denom1
        r = np.minimum(e0, e1)

    elif mode == "pooled_excess":
        c_pool = c0 + c1
        mu_pool = mu0 + mu1
        excess_pool = np.maximum(0.0, c_pool - mu_pool)
        denom_pool = c_pool + mu_pool + 2.0 * tau
        r = excess_pool / denom_pool

    elif mode == "replicated_count":
        r0 = c0 / (c0 + tau)
        r1 = c1 / (c1 + tau)
        r = np.minimum(r0, r1)

    elif mode == "degree_stratified_shuffle":
        # First compute standard replicated excess
        excess0 = np.maximum(0.0, c0 - mu0)
        excess1 = np.maximum(0.0, c1 - mu1)
        e0 = excess0 / (c0 + mu0 + tau)
        e1 = excess1 / (c1 + mu1 + tau)
        r_base = np.minimum(e0, e1)

        if degrees is None or E_sem is None:
            raise ValueError("Degree-stratified shuffle requires train item degrees and candidate pairs")
        degrees, E_sem = np.asarray(degrees), np.asarray(E_sem)
        if (degrees.ndim != 1 or not np.isfinite(degrees).all() or np.any(degrees < 0)
                or E_sem.shape != (E, 2) or E_sem.dtype.kind not in "iu"
                or np.any(E_sem < 0) or np.any(E_sem >= len(degrees))):
            raise ValueError("Invalid train item degrees or candidate pairs for shuffle")
        cuts = np.unique(np.quantile(degrees, [0.2, 0.4, 0.6, 0.8]))
        strata = np.searchsorted(cuts, degrees, side="right")
        left, right = strata[E_sem[:, 0]], strata[E_sem[:, 1]]
        groups = np.minimum(left, right) * 5 + np.maximum(left, right)
        rng = np.random.default_rng(placebo_seed)
        r = r_base.copy()
        for group in np.unique(groups):
            positions = np.flatnonzero(groups == group)
            if len(positions) > 1:
                r[positions] = rng.permutation(r_base[positions])

    elif mode == "uniform_multiplier":
        excess0 = np.maximum(0.0, c0 - mu0)
        excess1 = np.maximum(0.0, c1 - mu1)
        e0 = excess0 / (c0 + mu0 + tau)
        e1 = excess1 / (c1 + mu1 + tau)
        r_base = np.minimum(e0, e1)
        # Set to mean of positive values, or 0.5
        pos = r_base[r_base > 0]
        val = float(np.mean(pos)) if len(pos) > 0 else 0.5
        r = np.full(E, val, dtype=np.float64)

    else:
        raise ValueError(f"Unknown evidence mode: {mode}")

    # Boundedness assertion: 0 <= r < 1
    if np.any(r < 0.0) or np.any(r >= 1.0) or not np.isfinite(r).all():
        raise ArithmeticError(f"Evidence score r outside [0, 1): min={r.min()}, max={r.max()}")

    return r


# Public alias for backward compatibility and verification scripts
compute_replicated_excess_evidence = replicated_excess_score

