"""models/stair_icaart_graph.py — Replicated Excess-Evidence Calibration (STAIR-REEC).
========================================================================================
Implements:
1. Recovery of exact native STAIR5-v8 raw semantic graph W^8 and degree normalization N(W).
2. Exponential calibration W^delta = W^8 * exp(delta * r) on undirected candidate support E_sem.
3. True actual-degree re-normalization S_delta = N(W^delta) and unchanged CF blend S_ICA.
4. Exact V8 bypass (at delta = 0 or when all r = 0).
5. Offline mechanism diagnostics: coverage, disagreement, endpoint contrast (Eq 10),
   operator dose, polynomial direction dose (Eq 11), and degree side-effect ratios.
6. Checkpoint manifest generation and graph fingerprinting.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v8_graph import (
    build_wmsg_semantic_graph,
)
from models.stair5_v4_graph import (
    normalize_cf_graph_with_fallback,
    binary_train_keys, CACHE_VERSION, _digest_array, build_candidate_support_graph,
    sparse_fingerprint,
    tensor_to_scipy,
    scipy_to_tensor,
)
from models.stair5_v4_utils import atomic_torch_save, file_sha256
from stair_icaart_config import validate_icaart_config
from models.stair_icaart_evidence import (
    build_activity_replication,
    candidate_excess_statistics,
    replicated_excess_score,
)


@dataclass
class GraphStateICAART:
    """Encapsulates the final calibrated S_ICA operator, audit manifest, and diagnostics."""
    operator: torch.Tensor
    metadata: Dict[str, Any]
    manifest: Dict[str, Any]


def normalize_degree_isolated(W: sp.csr_matrix, dtype: Any = np.float32) -> sp.csr_matrix:
    """Computes exact symmetric degree normalization N(W) with identity isolated blocks.

    Mathematical definition (Eq 2):
        N(W) = D^{-1/2} W D^{-1/2} + diag(1[d_i == 0]), where d = W 1.

    Contracts:
    - Computed in FP64 for numerical precision.
    - Preserves exact symmetry and zero diagonal on connected nodes.
    - Returns CSR matrix with sorted indices and eliminated zeros in requested dtype (default float32).
    """
    if W.shape[0] != W.shape[1]:
        raise ValueError("Graph matrix must be square")

    W_64 = W.astype(np.float64).tocsr(copy=True)
    W_64.sum_duplicates()
    W_64.eliminate_zeros()

    if not np.isfinite(W_64.data).all() or np.any(W_64.data < 0.0):
        raise ValueError("Matrix entries must be finite and non-negative")

    # Strict symmetry check
    diff = W_64 - W_64.T
    diff.eliminate_zeros()
    if diff.nnz > 0:
        raise ValueError("Raw graph must be exactly symmetric before normalization")

    n = W_64.shape[0]
    degrees = np.asarray(W_64.sum(axis=1)).ravel()

    inv_sqrt = np.zeros(n, dtype=np.float64)
    pos_mask = degrees > 0.0
    inv_sqrt[pos_mask] = 1.0 / np.sqrt(degrees[pos_mask])

    D_scale = sp.diags(inv_sqrt, format="csr")
    normalized = (D_scale @ W_64 @ D_scale).astype(dtype)

    # Isolated nodes receive identity diagonal
    isolated_mask = (degrees == 0.0).astype(dtype)
    if np.any(isolated_mask > 0.0):
        normalized = normalized + sp.diags(isolated_mask, format="csr")

    normalized = normalized.tocsr()
    normalized.sum_duplicates()
    normalized.eliminate_zeros()
    normalized.sort_indices()
    return normalized


def recover_v8_raw_semantic_graph(
    mfeats: List[torch.Tensor],
    knn_edges_list: List[torch.Tensor],
    n_items: int,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Recovers the exact native STAIR5-v8 raw semantic graph W^8 (Eq 1).

    Uses frozen text (5) and vision (1) candidate sets with f_p(a) = 0.05 + 0.95 * [max(0, a)]^2,
    B_ij = sum_m 1[j in K_m(i)] f_p(a_ij^m), W^8_ij = max(B_ij, B_ji), W^8_ii = 0.

    Returns:
        W8_csr: Native V8 symmetric raw semantic matrix [N, N].
        E_sem: Unique undirected candidate pairs [E, 2] with i < j.
        w8_values: Raw semantic weights [E] corresponding to E_sem.
        audit: Modality and graph construction audit statistics.
    """
    W8_csr, audit = build_wmsg_semantic_graph(
        mfeats=mfeats,
        knn_edges_list=knn_edges_list,
        n_items=n_items,
        semantic_mode="weighted",
        edge_power=2.0,
        edge_floor=0.05,
        relation_gate="off",
        relation_strength=0.0,
    )

    # Extract unique undirected pairs i < j
    upper = sp.triu(W8_csr, k=1).tocoo()
    E_sem = np.column_stack((upper.row, upper.col)).astype(np.int64)
    w8_values = upper.data.astype(np.float64)

    return W8_csr, E_sem, w8_values, audit


def compute_mechanism_diagnostics(
    W8_csr: sp.csr_matrix,
    E_sem: np.ndarray,
    w8_values: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    S_8: sp.csr_matrix,
    S_ICA: sp.csr_matrix,
    d8: np.ndarray,
    d_delta: np.ndarray,
    gamma: float = 0.2,
    embedding_dim: int = 64,
    probe_dim: int = 8,
    probe_seed: int = 20261016,
) -> Dict[str, Any]:
    """Computes comprehensive offline mechanism diagnostics before expensive GPU training.

    Diagnostics computed:
    1. Replicated coverage: frac(r > 0) and weighted coverage sum(W8 * 1[r > 0]) / sum(W8).
    2. Single-half disagreement: frac((e0 > 0) xor (e1 > 0)).
    3. Endpoint contrast (Eq 10): r_ij - 0.5 * (rbar_i + rbar_j).
    4. Operator relative dose: ||S_ICA - S_8||_F / ||S_8||_F.
    5. Polynomial direction dose (Eq 11): ||[P(S_ICA) - P(S_8)] Z||_F / ||P(S_8) Z||_F.
    6. Degree side effect: d_delta / d8 on edges with r = 0.
    """
    E = len(E_sem)
    N = W8_csr.shape[0]
    diag: Dict[str, Any] = {}

    if E == 0:
        return {"empty_support": True, "num_candidate_edges": 0,
                "num_edges_with_replicated_excess": 0,
                "frac_edges_with_replicated_excess": 0.0,
                "weighted_replicated_coverage": 0.0,
                "relative_operator_dose": 0.0, "relative_polynomial_dose": 0.0}

    # 1. Replicated coverage
    r_pos = r > 0.0
    num_pos = int(np.count_nonzero(r_pos))
    frac_pos = float(num_pos / E)
    total_w8 = float(np.sum(w8_values))
    weighted_pos = float(np.sum(w8_values[r_pos]) / max(1e-12, total_w8))

    diag["num_candidate_edges"] = E
    diag["num_edges_with_replicated_excess"] = num_pos
    diag["frac_edges_with_replicated_excess"] = frac_pos
    diag["weighted_replicated_coverage"] = weighted_pos
    diag["r_mean"] = float(np.mean(r))
    diag["r_max"] = float(np.max(r)) if E > 0 else 0.0
    diag["r_positive_mean"] = float(np.mean(r[r_pos])) if num_pos > 0 else 0.0

    # 2. Single-half disagreement
    if c.shape[0] == E and mu.shape[0] == E:
        e0_pos = (c[:, 0] - mu[:, 0]) > 0.0
        e1_pos = (c[:, 1] - mu[:, 1]) > 0.0
        disagreement = e0_pos ^ e1_pos
        diag["frac_single_half_disagreement"] = float(np.mean(disagreement))
        diag["num_single_half_disagreement"] = int(np.count_nonzero(disagreement))
        diag["num_both_halves_positive"] = int(np.count_nonzero(e0_pos & e1_pos))

    # 3. Endpoint contrast (Eq 10): r_ij - 0.5 * rbar_i - 0.5 * rbar_j
    # rbar_i = sum_j W8_ij * r_ij / d8_i
    # Build symmetric r matrix
    r_mat = sp.csr_matrix((r, (E_sem[:, 0], E_sem[:, 1])), shape=(N, N))
    r_mat = r_mat + r_mat.T
    weighted_r = W8_csr.multiply(r_mat)
    weighted_r_sum = np.asarray(weighted_r.sum(axis=1)).ravel()

    rbar = np.zeros(N, dtype=np.float64)
    pos_deg = d8 > 0.0
    rbar[pos_deg] = weighted_r_sum[pos_deg] / d8[pos_deg]

    u_idx = E_sem[:, 0]
    v_idx = E_sem[:, 1]
    contrast = r - 0.5 * (rbar[u_idx] + rbar[v_idx])

    diag["endpoint_contrast_mean"] = float(np.mean(contrast))
    diag["endpoint_contrast_std"] = float(np.std(contrast))
    diag["endpoint_contrast_min"] = float(np.min(contrast))
    diag["endpoint_contrast_max"] = float(np.max(contrast))
    diag["endpoint_contrast_p10"] = float(np.percentile(contrast, 10))
    diag["endpoint_contrast_p50"] = float(np.percentile(contrast, 50))
    diag["endpoint_contrast_p90"] = float(np.percentile(contrast, 90))

    # 4. Operator relative dose: ||S_ICA - S_8||_F / ||S_8||_F
    diff_S = (S_ICA - S_8).astype(np.float64)
    fro_diff = float(np.sqrt(np.dot(diff_S.data, diff_S.data)))
    fro_s8 = float(np.sqrt(np.dot(S_8.data.astype(np.float64), S_8.data.astype(np.float64))))
    rel_operator_dose = fro_diff / max(1e-12, fro_s8)
    diag["frobenius_operator_delta"] = fro_diff
    diag["relative_operator_dose"] = rel_operator_dose

    # 5. Polynomial direction dose (Eq 11): ||[P(S_ICA) - P(S_8)] Z||_F / ||P(S_8) Z||_F
    rng = np.random.default_rng(probe_seed)
    Z = rng.choice([-1.0, 1.0], size=(N, probe_dim)).astype(np.float64)

    # Evaluate L=3 polynomial for representative middle coordinate k = d // 2
    k_mid = embedding_dim // 2
    b_k = 0.1 + 0.9 * (k_mid / embedding_dim) ** gamma
    omega = np.array([1.0, b_k, b_k**2, b_k**3], dtype=np.float64)
    omega = omega / np.sum(omega)

    def eval_poly(S_mat: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
        # P(S) X = omega_0 X + omega_1 S X + omega_2 S^2 X + omega_3 S^3 X
        res = omega[0] * X
        cur = X
        for l_idx in range(1, 4):
            cur = S_mat @ cur
            res = res + omega[l_idx] * cur
        return res

    P_s8_Z = eval_poly(S_8, Z)
    P_sica_Z = eval_poly(S_ICA, Z)
    diff_P = P_sica_Z - P_s8_Z
    fro_diff_P = float(np.linalg.norm(diff_P))
    fro_s8_P = float(np.linalg.norm(P_s8_Z))
    rel_poly_dose = fro_diff_P / max(1e-12, fro_s8_P)
    diag["relative_polynomial_dose"] = rel_poly_dose
    diag["polynomial_probe_coordinate"] = int(k_mid)
    diag["polynomial_probe_dim"] = int(probe_dim)
    diag["polynomial_probe_seed"] = int(probe_seed)
    diag["polynomial_probe_scope"] = "middle-coordinate L3 response on fixed Rademacher probes"

    row_delta = np.asarray(abs(diff_S).sum(axis=1)).ravel()
    row_base = np.asarray(abs(S_8).sum(axis=1)).ravel()
    relative_rows = np.divide(row_delta, row_base, out=np.zeros(N), where=row_base > 0)
    diag["relative_row_l1_dose_quantiles"] = np.quantile(relative_rows, [0, .1, .5, .9, 1]).tolist()

    # 6. Degree side effect: d_delta / d8 on edges where r = 0
    zero_r_mask = ~r_pos
    if np.any(zero_r_mask):
        u_zero = u_idx[zero_r_mask]
        v_zero = v_idx[zero_r_mask]
        items_in_zero_r = np.unique(np.concatenate([u_zero, v_zero]))
        valid_items = items_in_zero_r[d8[items_in_zero_r] > 0]
        if len(valid_items) > 0:
            deg_ratios = d_delta[valid_items] / d8[valid_items]
            diag["degree_ratio_on_zero_r_mean"] = float(np.mean(deg_ratios))
            diag["degree_ratio_on_zero_r_max"] = float(np.max(deg_ratios))
            diag["degree_ratio_on_zero_r_min"] = float(np.min(deg_ratios))
        # With zero evidence on the edge, its normalized coefficient changes only
        # through endpoint degree growth. This is not a retained-weight guarantee.
        ratios = np.sqrt((d8[u_zero] / d_delta[u_zero]) * (d8[v_zero] / d_delta[v_zero]))
        diag["zero_score_semantic_edge_coefficient_ratio_quantiles"] = np.quantile(
            ratios, [0, .1, .5, .9, 1],
        ).tolist()

    return diag


def calibrate_semantic_operator(
    W8_csr: sp.csr_matrix,
    E_sem: np.ndarray,
    r: np.ndarray,
    delta: float,
    SCF_csr: Optional[sp.csr_matrix] = None,
    eta: float = 0.1,
    V8_operator_cached: Optional[sp.csr_matrix] = None,
    gamma: float = 0.2,
    embedding_dim: int = 64,
    c_counts: Optional[np.ndarray] = None,
    mu_expectations: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """Calibrates semantic operator with exponential reweighting and returns S_ICA (Eq 6).

    Mathematical formulation:
        W^delta_ij = W^8_ij * exp(delta * r_ij) on undirected candidate edges i < j,
        W^delta_ji = W^delta_ij, W^delta_ii = 0.
        S_delta = N(W^delta) with true actual new degrees d_delta = W^delta 1.
        S_ICA = (1 - eta) * S_delta + eta * S_CF (or S_delta if CF is empty/disabled).

    Exact Bypass:
        - When delta == 0: returns exact V8 cached operator directly.
        - When all r == 0: returns exact V8 cached operator with inactive_reason="no_replicated_excess".
    """
    if not math.isfinite(delta) or not 0 <= delta <= 2:
        raise ValueError("delta must be finite in [0, 2]")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite in [0, 1]")
    W8_csr = W8_csr.tocsr(copy=True)
    W8_csr.sum_duplicates()
    W8_csr.eliminate_zeros()
    W8_csr.sort_indices()
    N = W8_csr.shape[0]
    if (W8_csr.shape != (N, N) or not np.isfinite(W8_csr.data).all()
            or np.any(W8_csr.data < 0) or np.any(W8_csr.diagonal() != 0)
            or (W8_csr - W8_csr.T).nnz):
        raise ValueError("W8 must be square, symmetric, finite, nonnegative and zero-diagonal")
    E_sem, r = np.asarray(E_sem), np.asarray(r, dtype=np.float64)
    E = len(E_sem)
    if (E_sem.shape != (E, 2) or E_sem.dtype.kind not in "iu" or r.shape != (E,)
            or not np.isfinite(r).all() or np.any(r < 0) or np.any(r >= 1)
            or np.any(E_sem < 0) or np.any(E_sem >= N)
            or np.any(E_sem[:, 0] >= E_sem[:, 1])):
        raise ValueError("Invalid candidate pairs or scores")
    upper = sp.triu(W8_csr, k=1, format="coo")
    expected_keys = upper.row.astype(np.int64) * N + upper.col
    pair_keys = E_sem[:, 0].astype(np.int64) * N + E_sem[:, 1]
    if (len(pair_keys) != upper.nnz
            or not np.array_equal(np.sort(pair_keys), np.sort(expected_keys))):
        raise ValueError("E_sem must contain each raw semantic edge exactly once")
    # Explicit indexing binds weights to caller pair order; triu storage order is
    # not an API contract and must never silently permute evidence assignments.
    w8_values = np.asarray(W8_csr[E_sem[:, 0], E_sem[:, 1]]).ravel().astype(np.float64)
    d8 = np.asarray(W8_csr.astype(np.float64).sum(axis=1)).ravel()
    if SCF_csr is not None:
        if (SCF_csr.shape != (N, N) or not np.isfinite(SCF_csr.data).all()
                or np.any(SCF_csr.data < 0)):
            raise ValueError("Invalid normalized CF graph")
        diff_cf = SCF_csr - SCF_csr.T
        if diff_cf.nnz and np.max(np.abs(diff_cf.data)) > 1e-7:
            raise ValueError("Normalized CF graph must be symmetric within FP32 tolerance")
    effective_eta = float(eta if SCF_csr is not None and SCF_csr.nnz and eta > 0 else 0)
    if V8_operator_cached is not None:
        if V8_operator_cached.shape != (N, N):
            raise ValueError("Cached V8 operator shape mismatch")
        S_8 = V8_operator_cached
    else:
        semantic = normalize_degree_isolated(W8_csr)
        S_8 = (((1 - effective_eta) * semantic + effective_eta * SCF_csr).tocsr()
               if effective_eta else semantic)
        S_8.eliminate_zeros()
        S_8.sort_indices()

    audit = {"delta": float(delta), "effective_eta": effective_eta,
             "n_items": N, "n_candidate_edges": E,
             "exact_v8_bypass": False, "inactive_reason": None}
    d_delta = d8
    if delta == 0 or E == 0 or not np.any(r):
        reason = "delta_zero" if delta == 0 else "no_replicated_excess"
        audit.update(exact_v8_bypass=True, inactive_reason=reason)
        S_ICA = S_8
    else:
        weighted = w8_values * np.exp(delta * r)
        directed = sp.csr_matrix((weighted, (E_sem[:, 0], E_sem[:, 1])), shape=(N, N))
        W_delta = (directed + directed.T).tocsr()
        d_delta = np.asarray(W_delta.sum(axis=1)).ravel()
        # Detect exact component-constant evidence without treating small real
        # contrasts as zero via np.allclose's default relative tolerance.
        _, components = sp.csgraph.connected_components(W8_csr, directed=False)
        labels = components[E_sem[:, 0]]
        lo = np.full(N, np.inf)
        hi = np.full(N, -np.inf)
        np.minimum.at(lo, labels, r)
        np.maximum.at(hi, labels, r)
        present = np.isfinite(lo)
        neutral = bool(np.all(lo[present] == hi[present]))
        audit["neutral_calibration"] = neutral
        if neutral:
            S_ICA = S_8
            audit.update(exact_v8_bypass=True, inactive_reason="component_constant_evidence")
        else:
            semantic = normalize_degree_isolated(W_delta)
            S_ICA = (((1 - effective_eta) * semantic + effective_eta * SCF_csr).tocsr()
                     if effective_eta else semantic)
            S_ICA.sum_duplicates()
            S_ICA.eliminate_zeros()
            S_ICA.sort_indices()
    audit.update(compute_mechanism_diagnostics(
        W8_csr, E_sem, w8_values, r,
        c_counts if c_counts is not None else np.zeros((E, 2), dtype=np.int64),
        mu_expectations if mu_expectations is not None else np.zeros((E, 2), dtype=np.float64),
        S_8, S_ICA, d8, d_delta, gamma, embedding_dim,
    ))
    audit["s_ica_nnz"] = int(S_ICA.nnz)
    audit["s_ica_fingerprint"] = sparse_fingerprint(S_ICA)
    return S_ICA, audit


def _native_cf_operator(train_edges, n_users, n_items, cfg):
    """Reuse V8's raw-CF cache signature, checksum and exact native normalizer."""
    pairs = binary_train_keys(train_edges, n_users, n_items)
    digest = hashlib.sha256()
    _digest_array(digest, pairs)
    signature = {
        "cache_version": CACHE_VERSION, "train": digest.hexdigest(),
        "users": n_users, "items": n_items, "k": int(getattr(cfg, "k_cf", 5)),
        "cmin": int(getattr(cfg, "c_min", 2)), "t": float(getattr(cfg, "t_shrinkage", 5)),
        "source": file_sha256(Path(__file__).with_name("stair5_v4_graph.py")),
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_dir = getattr(cfg, "graph_cache_dir", "")
    cache = Path(cache_dir) / f"cf_{key}.pt" if cache_dir else None
    raw = None
    hit = False
    if cache is not None and cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        raw = tensor_to_scipy(payload["operator"])
        if (payload.get("signature") != signature or raw.shape != (n_items, n_items)
                or payload.get("fingerprint") != sparse_fingerprint(raw)):
            raise ValueError("Native CF cache signature or checksum mismatch")
        hit = True
    if raw is None:
        raw = build_candidate_support_graph(
            train_edges, n_items, n_users, signature["k"], signature["cmin"], signature["t"],
            int(getattr(cfg, "cf_block_size", 64)), float(getattr(cfg, "cf_memory_budget_mib", 128)),
        )
        if cache is not None:
            atomic_torch_save({"operator": scipy_to_tensor(raw), "signature": signature,
                               "fingerprint": sparse_fingerprint(raw)}, cache)
    raw_hash = sparse_fingerprint(raw)
    # A wholly empty raw graph must not become an identity CF contribution.
    normalized = normalize_cf_graph_with_fallback(raw) if raw.nnz and signature["k"] else None
    return normalized, {"native_cf_cache_hit": hit, "raw_cf_fingerprint": raw_hash,
                        "normalized_cf_fingerprint": (sparse_fingerprint(normalized)
                                                      if normalized is not None else None),
                        "cf_cache_key": key}


@torch.no_grad()
def build_stair_icaart_graph(
    mfeats: List[torch.Tensor],
    knn_edges_list: List[torch.Tensor],
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    cfg: Any,
    V8_operator_cached: Optional[sp.csr_matrix] = None,
    SCF_csr_cached: Optional[sp.csr_matrix] = None,
) -> GraphStateICAART:
    """End-to-end builder for STAIR-REEC (versionICAART) static item graph S_ICA.

    Pipeline:
    1. Recover exact native V8 raw semantic graph W^8.
    2. Extract deduplicated binary train-presence matrix R.
    3. Deterministic user activity stratification and 2-half splitting.
    4. Candidate-only co-interaction counting c and activity-null expectation mu.
    5. Replicated excess scoring r.
    6. Exponential calibration and actual-degree normalization S_ICA.
    7. Generate comprehensive audit manifest.
    """
    started = time.perf_counter()

    validate_icaart_config(cfg)
    evidence_mode = getattr(cfg, "evidence_mode", "replicated_excess")
    evidence_seed = int(getattr(cfg, "evidence_seed", 20261016))
    evidence_tau = float(getattr(cfg, "evidence_tau", 2.0))
    evidence_delta = float(getattr(cfg, "evidence_delta", 1.0))
    candidate_block = int(getattr(cfg, "evidence_candidate_block", 8192))
    placebo_seed = int(getattr(cfg, "evidence_placebo_seed", 20261016))
    eta = float(getattr(cfg, "eta", 0.1))
    gamma = float(getattr(cfg, "gamma", 0.2))
    embedding_dim = int(getattr(cfg, "embedding_dim", 64))

    # 1. Recover exact native W^8
    W8_csr, E_sem, w8_values, w8_audit = recover_v8_raw_semantic_graph(
        mfeats=mfeats,
        knn_edges_list=knn_edges_list,
        n_items=n_items,
    )

    # 2. Deduplicate train presence matrix R [M, N]
    unique_keys = binary_train_keys(train_edges, n_users, n_items)
    num_raw_interactions = int(train_edges.shape[1])
    num_unique_interactions = len(unique_keys)

    u_dedup = unique_keys // np.int64(n_items)
    i_dedup = unique_keys % np.int64(n_items)

    R_csr = sp.csr_matrix(
        (np.ones(num_unique_interactions, dtype=np.int8), (u_dedup, i_dedup)),
        shape=(n_users, n_items),
    )
    R_csc = R_csr.tocsc()

    # Presence hash for provenance
    presence_digest = hashlib.sha256(str((n_users, n_items)).encode())
    _digest_array(presence_digest, unique_keys)
    presence_hash = presence_digest.hexdigest()

    # 3. Activity stratification and user splitting
    user_bin, half, group_sizes = build_activity_replication(
        R_csr=R_csr,
        evidence_seed=evidence_seed,
    )

    # 4. Candidate co-interactions and null expectations
    c, mu = candidate_excess_statistics(
        R_csc=R_csc,
        E_sem=E_sem,
        half=half,
        user_bin=user_bin,
        group_sizes=group_sizes,
        candidate_block=candidate_block,
        mode=evidence_mode,
    )

    # 5. Replicated excess scoring
    degrees = np.diff(R_csc.indptr).astype(np.int64)
    r = replicated_excess_score(
        c=c,
        mu=mu,
        tau=evidence_tau,
        mode=evidence_mode,
        placebo_seed=placebo_seed,
        degrees=degrees,
        E_sem=E_sem,
    )

    # 6. Load or build CF operator S_CF
    cf_audit = {}
    if SCF_csr_cached is None:
        SCF_csr, cf_audit = _native_cf_operator(train_edges, n_users, n_items, cfg)
    else:
        SCF_csr = SCF_csr_cached
        cf_audit["normalized_cf_fingerprint"] = sparse_fingerprint(SCF_csr)

    if V8_operator_cached is not None:
        semantic_reference = normalize_degree_isolated(W8_csr)
        native_eta = eta if SCF_csr is not None and SCF_csr.nnz else 0.0
        reference = (((1 - native_eta) * semantic_reference + native_eta * SCF_csr).tocsr()
                     if native_eta else semantic_reference)
        if sparse_fingerprint(reference) != sparse_fingerprint(V8_operator_cached):
            raise ValueError("Reconstructed native graph differs from the cached V8 control")

    # 7. Calibrate semantic operator
    S_ICA, cal_audit = calibrate_semantic_operator(
        W8_csr=W8_csr,
        E_sem=E_sem,
        r=r,
        delta=evidence_delta,
        SCF_csr=SCF_csr,
        eta=eta,
        V8_operator_cached=V8_operator_cached,
        gamma=gamma,
        embedding_dim=embedding_dim,
        c_counts=c,
        mu_expectations=mu,
    )

    elapsed = time.perf_counter() - started

    # 8. Compile comprehensive manifest
    manifest = {
        "architecture": "STAIR-REEC-versionICAART",
        "evidence_mode": evidence_mode,
        "evidence_seed": evidence_seed,
        "evidence_tau": evidence_tau,
        "evidence_delta": evidence_delta,
        "evidence_candidate_block": candidate_block,
        "evidence_activity_bins": "log2",
        "evidence_halves": 2,
        "evidence_placebo_seed": placebo_seed,
        "training_seed": int(getattr(cfg, "seed", 1)),
        "evidence_dtype": "float64",
        "operator_dtype": "float32",
        "raw_semantic_fingerprint": sparse_fingerprint(W8_csr),
        "baseline_v8_fingerprint": sparse_fingerprint(V8_operator_cached) if V8_operator_cached is not None else None,
        "assignment_hash": hashlib.sha256(user_bin.astype("<i8").tobytes() + half.astype("<i8").tobytes()).hexdigest(),
        "group_sizes": group_sizes.tolist(),
        "evidence_score_hash": hashlib.sha256(r.astype("<f8").tobytes()).hexdigest(),
        "evidence_config": validate_icaart_config(cfg),
        "cf": {k: v for k, v in cf_audit.items() if k != "native_cf_cache_hit"},
        "num_raw_train_interactions": num_raw_interactions,
        "num_deduplicated_train_interactions": num_unique_interactions,
        "train_presence_hash": presence_hash,
        "num_users": n_users,
        "num_items": n_items,
        "num_activity_strata": int(group_sizes.shape[1]),
        "user_half_counts": [int(np.sum(half == 0)), int(np.sum(half == 1))],
        "graph_fingerprint": cal_audit["s_ica_fingerprint"],
    }

    metadata = {**w8_audit, **cal_audit, **cf_audit, **manifest, "build_elapsed_seconds": elapsed}

    device = train_edges.device
    operator_tensor = scipy_to_tensor(S_ICA).to(device)

    return GraphStateICAART(operator=operator_tensor, metadata=metadata, manifest=manifest)


# Public alias for backward compatibility and verification scripts
calibrate_semantic_support = calibrate_semantic_operator

