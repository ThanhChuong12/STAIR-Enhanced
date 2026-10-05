"""models/stair5_v51_graph.py — STAIR5-v5.1 Graph Operators: S4 and CAM.
========================================================================
Implements:
1. Exact reuse of STAIR5-v4 Candidate Support Expansion (CSE) and blended S4 operator:
       S4 = 0.9 * S0 + 0.1 * S_bar_CF
2. Conditional Contraction-Preserving Adaptive Mixing (CAM, Eq. 10 in Report):
       eta_i = eta_0 + delta * (r_i - r_bar)
       S_CAM = A @ S0 @ A + B @ S_bar_CF @ B
       A = diag(sqrt(1 - eta)), B = diag(sqrt(eta))
   When delta = 0, S_CAM strictly recovers S4.
   Since A^2 + B^2 = I and ||S0||_2 <= 1, ||S_bar_CF||_2 <= 1, ||S_CAM||_2 <= 1 is guaranteed.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v4_graph import (
    binary_train_keys,
    build_candidate_support_graph,
    compute_evidence_score,
    normalize_cf_graph_with_fallback,
    permute_cf_graph_degree_stratified,
    scipy_to_tensor,
    sparse_fingerprint,
    tensor_to_scipy,
)
from models.stair5_v51_utils import atomic_torch_save

ARMS_V51 = (
    "V51-KPE",       # Primary: S4 graph + NLGCL-KPE
    "V4",            # Control: S4 graph + original occurrence NLGCL
    "DEDUP",         # Ablation: S4 graph + unique-key CE
    "MP-MATCH",      # Ablation: S4 graph + uniform multi-positive CE
    "CAM",           # Factorial: S_CAM graph (delta=0.05) + original NLGCL
    "CAM-KPE",       # Factorial: S_CAM graph (delta=0.05) + NLGCL-KPE
    "B0",            # Baseline control: S0 semantic only, lambda_cl = 0
    "N0",            # Reference reproduction: S0 semantic only, lambda_cl = 0.01
    "C0",            # CSE graph only, lambda_cl = 0
    "N-CSE-placebo", # Degree-stratified placebo CF graph + NLGCL-KPE
    "MASK-PLACEBO",  # S4 graph + mask-placebo random negative reduction
)

CAM_ARMS = ("CAM", "CAM-KPE")
CACHE_VERSION_V51 = 51


@dataclass
class GraphStateV51:
    operator: torch.Tensor
    metadata: dict


def compute_cam_node_support(
    W_cf: sp.csr_matrix,
    n_items: int,
) -> Tuple[np.ndarray, float]:
    """Computes node-wise collaborative support r_i with active-node P95 clipping.

    Args:
        W_cf: Symmetric unnormalized CF co-occurrence evidence graph.
        n_items: Total item count.

    Returns:
        r: Normalized support array of shape (n_items,), in [0, 1].
        r_bar: All-node mean support scalar.
    """
    degrees = np.asarray((W_cf > 0).sum(axis=1)).ravel().astype(np.float64)
    active = degrees > 0
    sums = np.asarray(W_cf.sum(axis=1)).ravel().astype(np.float64)

    r_raw = np.zeros(n_items, dtype=np.float64)
    r_raw[active] = sums[active] / np.maximum(degrees[active], 1.0)

    if np.any(active):
        p95 = float(np.percentile(r_raw[active], 95))
        if p95 > 1e-12:
            r = np.clip(r_raw / p95, 0.0, 1.0)
        else:
            r = np.clip(r_raw, 0.0, 1.0)
    else:
        r = np.zeros(n_items, dtype=np.float64)

    r_bar = float(r.mean())
    return r, r_bar


def build_calibrated_graph_v51(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    arm: str = "V51-KPE",
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    cam_delta: float = 0.05,
    placebo_seed: int = 1,
    seed: int = 1,
    cache_dir: Optional[str] = None,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
) -> GraphStateV51:
    """Builds calibrated operator (S4 or S_CAM) for STAIR5-v5.1.

    Guarantees operator spectral bound ||S||_2 <= 1.0 for all configurations.
    """
    if arm not in ARMS_V51:
        raise ValueError(f"Unknown v5.1 arm: {arm}. Must be one of {ARMS_V51}")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite in [0, 1].")
    if not math.isfinite(cam_delta) or cam_delta < 0:
        raise ValueError("cam_delta must be finite and >= 0.")
    if k_cf < 0 or int(k_cf) != k_cf or c_min < 1 or int(c_min) != c_min:
        raise ValueError("Invalid CF top-k or minimum count.")

    metadata = {
        "version": 51,
        "arm": arm,
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "cam_delta": float(cam_delta),
        "num_users": n_users,
        "num_items": n_items,
        "cache_hit": False,
    }

    def off(reason: str) -> GraphStateV51:
        metadata.update(active=False, effective_eta=0.0, nnz=int(baseline_normalized._nnz()), note=reason)
        return GraphStateV51(baseline_normalized, metadata)

    if arm in ("B0", "N0") or eta == 0 or k_cf == 0:
        return off("Exact original S0; CF branch disabled.")

    started = time.perf_counter()
    pairs = binary_train_keys(train_edges, n_users, n_items)
    pair_hash = hashlib.sha256()
    pair_arr = np.ascontiguousarray(pairs)
    pair_hash.update(str((pair_arr.shape, pair_arr.dtype.str)).encode())
    pair_hash.update(memoryview(pair_arr).cast("B"))

    signature = {
        "cache_version": CACHE_VERSION_V51,
        "train": pair_hash.hexdigest(),
        "users": n_users,
        "items": n_items,
        "k": k_cf,
        "cmin": c_min,
        "t": t_shrinkage,
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"cf_v51_{cache_key}.pt" if cache_dir else None
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
        except Exception:
            W_cf = None

    if W_cf is None:
        W_cf = build_candidate_support_graph(
            train_edges, n_items, n_users, k_cf, c_min, t_shrinkage, block_size, memory_budget_mib
        )
        if cache:
            atomic_torch_save(
                {
                    "operator": scipy_to_tensor(W_cf),
                    "signature": signature,
                    "fingerprint": sparse_fingerprint(W_cf),
                },
                cache,
            )

    metadata.update(
        train_fingerprint=signature["train"],
        cf_cache_key=cache_key,
        cf_preprocessing_seconds=time.perf_counter() - started,
    )

    if W_cf.nnz == 0:
        return off("Exact original S0; CF graph has no eligible edges.")

    if arm == "N-CSE-placebo":
        W_cf, placebo = permute_cf_graph_degree_stratified(W_cf, train_edges, n_items, placebo_seed, True)
        metadata.update(placebo)

    S0 = tensor_to_scipy(baseline_normalized).astype(np.float32)
    S_bar_cf = normalize_cf_graph_with_fallback(W_cf)

    # Determine whether CAM is active
    use_cam = (arm in CAM_ARMS) and (cam_delta > 0.0)

    if use_cam:
        r, r_bar = compute_cam_node_support(W_cf, n_items)
        eta_nodes = np.clip(eta + cam_delta * (r - r_bar), 0.0, 1.0).astype(np.float32)
        A_diag = np.sqrt(np.clip(1.0 - eta_nodes, 0.0, 1.0))
        B_diag = np.sqrt(np.clip(eta_nodes, 0.0, 1.0))

        # S_CAM = A @ S0 @ A + B @ S_bar_cf @ B
        A_sp = sp.diags(A_diag)
        B_sp = sp.diags(B_diag)
        part_sem = (A_sp @ S0 @ A_sp).tocsr()
        part_cf = (B_sp @ S_bar_cf @ B_sp).tocsr()
        mixed = (part_sem + part_cf).tocsr()
        metadata.update(
            is_cam=True,
            cam_delta=float(cam_delta),
            cam_eta_min=float(eta_nodes.min()),
            cam_eta_max=float(eta_nodes.max()),
            cam_eta_mean=float(eta_nodes.mean()),
            r_bar=float(r_bar),
        )
    else:
        # Standard S4 convex blend
        mixed = ((1.0 - eta) * S0 + eta * S_bar_cf).tocsr()
        metadata.update(is_cam=False, effective_eta=float(eta))

    mixed.eliminate_zeros()
    mixed.sort_indices()

    overlap = int(W_cf.sign().multiply(S0.sign()).nnz)
    isolated = int(np.count_nonzero(np.asarray(W_cf.sum(1)).ravel() == 0))
    delta = mixed - S0
    ratio = float(
        np.sqrt(np.dot(delta.data.astype(np.float64), delta.data.astype(np.float64)))
        / max(1e-12, np.sqrt(np.dot(S0.data.astype(np.float64), S0.data.astype(np.float64))))
    )

    metadata.update(
        active=True,
        s0_nnz=int(S0.nnz),
        cf_nnz=int(W_cf.nnz),
        operator_nnz=int(mixed.nnz),
        cf_isolated_nodes=isolated,
        cf_isolated_fraction=isolated / n_items,
        jaccard_semantic_cf=overlap / max(1, W_cf.nnz + S0.nnz - overlap),
        new_cf_edges_count=int(W_cf.nnz - overlap),
        relative_operator_delta=ratio,
        graph_fingerprint=sparse_fingerprint(mixed),
    )

    tensor_op = scipy_to_tensor(mixed).to(baseline_normalized.device)
    return GraphStateV51(tensor_op, metadata)
