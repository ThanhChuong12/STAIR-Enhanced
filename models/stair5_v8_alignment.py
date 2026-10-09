"""models/stair5_v8_alignment.py — STAIR5-v8 Degree-Stratified Anchor-Procrustes (SAP).
=====================================================================================
Stage 1 Modality Initialization Alignment Module.

Implements:
1. Exact V4 MI control (alignment="off"):
   V^{(0)} = (5 * Z_t + Z_v) / 6
2. Degree-Stratified Anchor-Procrustes (SAP) extension (alignment="stratified_procrustes"):
   - Identifies eligible same-item pairs (Z_{v, i}, Z_{t, i}) with n_i > 0.
   - Partitions items into 5 near-equal degree strata by (n_i, item_id).
   - Samples <= anchor_per_stratum (default 1024) items per stratum with anchor_seed.
   - 80:20 fit/audit split with stratum weighting w_i = 1 / (5 * |A_{q, fit}|).
   - Weighted Procrustes fit in FP64: C = Z_{v, A}^T diag(w) Z_{t, A} = P Sigma H^T -> Q* = P H^T.
   - Strict engineering guards:
     a. All strata non-empty.
     b. Total fitting rows >= d.
     c. Finite SVD and sigma_min(C) / sigma_max(C) >= 1e-6.
     d. Weighted audit residual reduction >= 1% relative to identity.
     e. No single degree stratum has > 2% residual regression relative to identity.
   - Fallback to identity Q* = I_d if any guard fails.
   - Refits on all selected anchors upon guard success.
   - Computes aligned MI: V^{(0)}_{SAP} = (5 * Z_t + Z_v Q*) / 6.

Zero-parameter, no autograd, FP64 CPU fitting, strict audit logging.
"""

from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import math
import numpy as np
import torch


def compute_weighted_procrustes(
    Z_v: np.ndarray,
    Z_t: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes Weighted Orthogonal Procrustes Q* = P H^T in FP64.

    Args:
        Z_v: Source visual features [A, d], float64.
        Z_t: Target text features [A, d], float64.
        weights: Nonnegative sample weights [A], float64, sum to 1.

    Returns:
        Q: Optimal orthogonal transform [d, d], float64.
        singular_values: Singular values of cross-covariance C, float64 [d].
        cond_ratio: sigma_min / sigma_max.
    """
    Z_v = np.asarray(Z_v, dtype=np.float64)
    Z_t = np.asarray(Z_t, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if Z_v.ndim != 2 or Z_t.shape != Z_v.shape or weights.shape != (len(Z_v),):
        raise ValueError("Procrustes inputs must have matching [A, d] features and [A] weights.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Procrustes weights must be finite, nonnegative and have positive mass.")
    # C = Z_v^T @ diag(w) @ Z_t  shape: [d, d]
    weighted_Zv = Z_v * weights[:, np.newaxis]
    C = weighted_Zv.T @ Z_t

    if not np.isfinite(C).all():
        raise FloatingPointError("Cross-covariance matrix C contains nonfinite values.")

    P, Sigma, Ht = np.linalg.svd(C, full_matrices=False)
    Q = P @ Ht  # Orthogonal matrix Q*

    sigma_max = float(Sigma[0])
    sigma_min = float(Sigma[-1])
    cond_ratio = sigma_min / sigma_max if sigma_max > 0.0 else 0.0

    return Q, Sigma, cond_ratio


def evaluate_weighted_residual(
    Z_v: np.ndarray,
    Z_t: np.ndarray,
    Q: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Computes weighted L2 residual: sum_i w_i || Z_{v, i} Q - Z_{t, i} ||_2^2."""
    diff = (np.asarray(Z_v, dtype=np.float64) @ Q) - np.asarray(Z_t, dtype=np.float64)
    squared_norms = np.sum(diff ** 2, axis=1)
    return float(np.sum(weights * squared_norms))


@torch.no_grad()
def align_modality_features(
    Z_t: torch.Tensor,
    Z_v: torch.Tensor,
    train_degrees: torch.Tensor,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Degree-Stratified Anchor-Procrustes (SAP) alignment or exact V4 bypass.

    Args:
        Z_t: Whitened text features [N, d], FP32.
        Z_v: Whitened visual features [N, d], FP32.
        train_degrees: Item degrees in training bipartite graph [N], int64.
        cfg: Configuration object containing:
             - alignment: 'off' or 'stratified_procrustes'
             - anchor_seed: int (default 2718)
             - anchor_per_stratum: int (default 1024)
             - embedding_dim: int d (default 64)

    Returns:
        fused_MI: Aligned multimodal representation [N, d], FP32.
        Q_matrix: Alignment transform [d, d], FP64 tensor on CPU.
        audit_record: Structured diagnostic dictionary for provenance/manifest.
    """
    if Z_t.ndim != 2 or Z_v.shape != Z_t.shape or train_degrees.shape != (len(Z_t),):
        raise ValueError("SAP requires matching [N, d] modalities and [N] degrees.")
    if Z_t.device.type != "cpu" or Z_v.device.type != "cpu":
        raise ValueError("SAP fitting and MI preprocessing must run on CPU.")
    N, d = Z_t.shape
    alignment_mode = str(getattr(cfg, "alignment", "off")).lower()
    
    # -------------------------------------------------------------------------
    # Arm 1: Exact V4 Control Bypass (alignment='off')
    # -------------------------------------------------------------------------
    if alignment_mode == "off":
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        Q_identity = torch.eye(d, dtype=torch.float64)
        audit_record = {
            "alignment_mode": "off",
            "status": "BYPASS_V4_PARITY",
            "passed": True,
            "fallback_reason": None,
            "num_items": N,
            "embedding_dim": d,
            "fused_formula": "V^{(0)} = (5*Z_t + Z_v) / 6",
        }
        return fused_MI, Q_identity, audit_record

    if alignment_mode != "stratified_procrustes":
        raise ValueError(f"Unknown alignment mode: {alignment_mode}. Expected 'off' or 'stratified_procrustes'.")

    # -------------------------------------------------------------------------
    # Arm 2: Degree-Stratified Anchor-Procrustes (SAP)
    # -------------------------------------------------------------------------
    anchor_seed = int(getattr(cfg, "anchor_seed", 2718))
    anchor_per_stratum = int(getattr(cfg, "anchor_per_stratum", 1024))
    if anchor_seed < 0 or anchor_per_stratum < 2:
        raise ValueError("SAP requires a nonnegative seed and at least two anchors per stratum.")

    degrees_np = train_degrees.detach().cpu().numpy().astype(np.int64)
    # Promote selected anchors only; avoid two full-catalog FP64 copies.
    Z_t_np = Z_t.detach().numpy()
    Z_v_np = Z_v.detach().numpy()

    # 1. Eligible items: n_i > 0 and finite features
    finite_mask = np.isfinite(Z_t_np).all(axis=1) & np.isfinite(Z_v_np).all(axis=1)
    eligible_indices = np.where((degrees_np > 0) & finite_mask)[0]
    num_eligible = len(eligible_indices)

    # Audit accumulator
    audit_record: Dict[str, Any] = {
        "alignment_mode": "stratified_procrustes",
        "num_items": N,
        "num_eligible": num_eligible,
        "anchor_seed": anchor_seed,
        "anchor_per_stratum": anchor_per_stratum,
        "embedding_dim": d,
        "status": "INITIALIZED",
        "passed": False,
        "fallback_reason": None,
    }

    # Guard 0: Minimum eligible items check
    if num_eligible < 10:
        audit_record["status"] = "FALLBACK_INSUFFICIENT_ITEMS"
        audit_record["fallback_reason"] = f"Eligible items ({num_eligible}) cannot support five nonempty fit/audit strata."
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    # 2. Sort eligible items by (n_i, item_id) to guarantee deterministic stratum partitioning
    # Pack into structured key or sort stably:
    sort_keys = np.lexsort((eligible_indices, degrees_np[eligible_indices]))
    sorted_eligible = eligible_indices[sort_keys]

    # 3. Partition into 5 equal/near-equal strata Q_1, ..., Q_5
    strata = np.array_split(sorted_eligible, 5)
    audit_record["strata_sizes_full"] = [len(s) for s in strata]
    audit_record["strata_degree_bounds"] = [
        {"min": int(degrees_np[s[0]]), "max": int(degrees_np[s[-1]])} for s in strata
    ]

    # Dedicated fixed RNG
    rng = np.random.default_rng(anchor_seed)

    fit_indices_by_stratum: List[np.ndarray] = []
    audit_indices_by_stratum: List[np.ndarray] = []

    # 4. Stratum sampling (<= anchor_per_stratum) and 80:20 fit/audit split
    guards_failed = False
    failure_reason = None

    for q_idx, stratum_items in enumerate(strata):
        if len(stratum_items) == 0:
            guards_failed = True
            failure_reason = f"Stratum {q_idx + 1} is empty."
            break

        # Uniform sample at most anchor_per_stratum
        if len(stratum_items) > anchor_per_stratum:
            sampled_items = rng.choice(stratum_items, size=anchor_per_stratum, replace=False)
        else:
            sampled_items = stratum_items.copy()
            rng.shuffle(sampled_items)

        # 80:20 split within stratum
        n_sampled = len(sampled_items)
        n_fit = int(round(0.8 * n_sampled))
        # Ensure at least 1 fit and 1 audit if n_sampled >= 2
        n_fit = max(1, min(n_fit, n_sampled - 1)) if n_sampled >= 2 else n_fit

        fit_part = sampled_items[:n_fit]
        audit_part = sampled_items[n_fit:]

        if len(fit_part) == 0 or len(audit_part) == 0:
            guards_failed = True
            failure_reason = f"Stratum {q_idx + 1} fit or audit split is empty (fit={len(fit_part)}, audit={len(audit_part)})."
            break

        fit_indices_by_stratum.append(fit_part)
        audit_indices_by_stratum.append(audit_part)

    if guards_failed:
        audit_record["status"] = "FALLBACK_STRATA_SPLIT_FAILED"
        audit_record["fallback_reason"] = failure_reason
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    # 5. Build weights: each stratum contributes total fitting weight 1/5
    fit_items_concat = np.concatenate(fit_indices_by_stratum)
    audit_items_concat = np.concatenate(audit_indices_by_stratum)
    audit_record["fit_anchor_ids"] = fit_items_concat.tolist()
    audit_record["audit_anchor_ids"] = audit_items_concat.tolist()

    if len(fit_items_concat) < d:
        audit_record["status"] = "FALLBACK_INSUFFICIENT_FIT_ROWS"
        audit_record["fallback_reason"] = f"Total fit anchors ({len(fit_items_concat)}) < d ({d})."
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    fit_weights = np.zeros(len(fit_items_concat), dtype=np.float64)
    audit_weights = np.zeros(len(audit_items_concat), dtype=np.float64)

    fit_offset = 0
    audit_offset = 0
    for q_idx in range(5):
        s_fit_len = len(fit_indices_by_stratum[q_idx])
        s_audit_len = len(audit_indices_by_stratum[q_idx])
        fit_weights[fit_offset : fit_offset + s_fit_len] = 1.0 / (5.0 * s_fit_len)
        audit_weights[audit_offset : audit_offset + s_audit_len] = 1.0 / (5.0 * s_audit_len)
        fit_offset += s_fit_len
        audit_offset += s_audit_len

    # Normalize weights to exactly sum to 1.0
    fit_weights /= np.sum(fit_weights)
    audit_weights /= np.sum(audit_weights)

    # 6. Fit candidate Q_cand on fit anchors
    Z_v_fit = Z_v_np[fit_items_concat]
    Z_t_fit = Z_t_np[fit_items_concat]

    try:
        Q_cand, Sigma, cond_ratio = compute_weighted_procrustes(Z_v_fit, Z_t_fit, fit_weights)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        audit_record["status"] = "FALLBACK_SVD_ERROR"
        audit_record["fallback_reason"] = f"SVD fitting failed: {str(exc)}"
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    audit_record["fit_singular_values"] = [float(s) for s in Sigma]
    audit_record["fit_condition_ratio"] = float(cond_ratio)

    # Guard a: Condition ratio >= 1e-6
    if not np.isfinite(Q_cand).all() or not np.isfinite(cond_ratio) or cond_ratio < 1e-6:
        audit_record["status"] = "FALLBACK_ILL_CONDITIONED"
        audit_record["fallback_reason"] = f"Cross-covariance condition ratio ({cond_ratio:.3e}) < 1e-6."
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    # 7. Audit evaluation against Identity matrix I_d
    Z_v_audit = Z_v_np[audit_items_concat]
    Z_t_audit = Z_t_np[audit_items_concat]
    I_d = np.eye(d, dtype=np.float64)

    res_ident_total = evaluate_weighted_residual(Z_v_audit, Z_t_audit, I_d, audit_weights)
    res_Q_total = evaluate_weighted_residual(Z_v_audit, Z_t_audit, Q_cand, audit_weights)

    total_reduction = (res_ident_total - res_Q_total) / max(res_ident_total, 1e-12)
    audit_record["audit_res_identity"] = float(res_ident_total)
    audit_record["audit_res_candidate"] = float(res_Q_total)
    audit_record["audit_relative_reduction"] = float(total_reduction)

    # Guard b: At least 1% reduction in weighted audit residual
    if not np.isfinite(total_reduction) or total_reduction < 0.01:
        audit_record["status"] = "FALLBACK_INSUFFICIENT_AUDIT_GAIN"
        audit_record["fallback_reason"] = (
            f"Audit residual reduction ({total_reduction * 100:.2f}%) < 1.00% minimum required."
        )
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    # Guard c: Per-stratum regression check (no stratum regression > 2%)
    per_stratum_audit = []
    regressed_strata = []
    audit_cur = 0

    for q_idx in range(5):
        s_audit_len = len(audit_indices_by_stratum[q_idx])
        s_items = audit_items_concat[audit_cur : audit_cur + s_audit_len]
        s_weights = np.ones(s_audit_len, dtype=np.float64) / s_audit_len
        s_Zv = Z_v_np[s_items]
        s_Zt = Z_t_np[s_items]

        s_res_id = evaluate_weighted_residual(s_Zv, s_Zt, I_d, s_weights)
        s_res_q = evaluate_weighted_residual(s_Zv, s_Zt, Q_cand, s_weights)
        # Ratio of candidate residual to identity residual
        rel_change = (s_res_q - s_res_id) / max(s_res_id, 1e-12)

        per_stratum_audit.append({
            "stratum": q_idx + 1,
            "res_identity": float(s_res_id),
            "res_candidate": float(s_res_q),
            "relative_change": float(rel_change),
        })

        # Regression > 2% means rel_change > +0.02
        if not np.isfinite(rel_change) or rel_change > 0.02:
            regressed_strata.append(q_idx + 1)

        audit_cur += s_audit_len

    audit_record["per_stratum_audit"] = per_stratum_audit

    if len(regressed_strata) > 0:
        audit_record["status"] = "FALLBACK_STRATUM_REGRESSION"
        audit_record["fallback_reason"] = (
            f"Stratum regression > 2% observed in strata {regressed_strata}."
        )
        fused_MI = (5.0 * Z_t + Z_v) / 6.0
        return fused_MI, torch.eye(d, dtype=torch.float64), audit_record

    # -------------------------------------------------------------------------
    # 8. All Guards Passed! Refit on all selected anchors (fit + audit)
    # -------------------------------------------------------------------------
    all_selected_by_stratum = [
        np.concatenate([fit_indices_by_stratum[q], audit_indices_by_stratum[q]])
        for q in range(5)
    ]
    all_selected_concat = np.concatenate(all_selected_by_stratum)
    all_weights = np.zeros(len(all_selected_concat), dtype=np.float64)

    all_offset = 0
    for q_idx in range(5):
        s_total = len(all_selected_by_stratum[q_idx])
        all_weights[all_offset : all_offset + s_total] = 1.0 / (5.0 * s_total)
        all_offset += s_total
    all_weights /= np.sum(all_weights)

    Z_v_all = Z_v_np[all_selected_concat]
    Z_t_all = Z_t_np[all_selected_concat]

    try:
        Q_final, Sigma_final, cond_ratio_final = compute_weighted_procrustes(
            Z_v_all, Z_t_all, all_weights
        )
        if not np.isfinite(Q_final).all() or not np.isfinite(cond_ratio_final) or cond_ratio_final < 1e-6:
            raise FloatingPointError("Final anchor refit is nonfinite or ill-conditioned.")
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        audit_record["status"] = "FALLBACK_REFIT_FAILED"
        audit_record["fallback_reason"] = str(exc)
        return (5.0 * Z_t + Z_v) / 6.0, torch.eye(d, dtype=torch.float64), audit_record

    # Convert Q_final to torch
    Q_final_tensor = torch.from_numpy(Q_final).to(dtype=torch.float64)
    Q_final_fp32 = Q_final_tensor.to(dtype=Z_v.dtype, device=Z_v.device)

    # 9. Form Aligned MI: V^{(0)}_{SAP} = (5 * Z_t + Z_v @ Q*) / 6
    Z_v_aligned = Z_v @ Q_final_fp32
    fused_MI = (5.0 * Z_t + Z_v_aligned) / 6.0

    # Fingerprint and finalize audit record
    q_bytes = Q_final.tobytes()
    q_sha256 = hashlib.sha256(q_bytes).hexdigest()

    audit_record.update({
        "status": "SAP_ALIGNED_SUCCESS",
        "passed": True,
        "final_singular_values": [float(s) for s in Sigma_final],
        "final_condition_ratio": float(cond_ratio_final),
        "total_selected_anchors": len(all_selected_concat),
        "transform_sha256": q_sha256,
        "transform_values": Q_final.tolist(),
        "selected_anchor_ids": all_selected_concat.tolist(),
        "fused_formula": "V^{(0)}_{SAP} = (5*Z_t + Z_v @ Q*) / 6",
    })

    return fused_MI, Q_final_tensor, audit_record
