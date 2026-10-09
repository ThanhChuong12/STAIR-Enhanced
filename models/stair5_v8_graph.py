"""models/stair5_v8_graph.py — Continuous WMSG and Conditional Relation-Proxy Graph.
========================================================================================
Implements Stage 2 and Stage 3 of STAIR5-v8 (WMSG-CSE):
- Stage 2: Continuous Weighted Multimodal Semantic Graph (WMSG) on frozen candidate
  support, with optional Relation-Proxy Gate (RPG).
- Stage 3: Normalization and unchanged Candidate Support Expansion (CSE) blend.

Mathematical Formulation:
--------------------------
1. Selected-edge Cosine Similarities on frozen V4 kNN candidates (Eq 5):
   a_{ij}^m = clamp( (X_{m,i}^T X_{m,j}) / (max(||X_{m,i}||_2, eps_x) * max(||X_{m,j}||_2, eps_x)), -1, 1 )
   for j in K_m(i).

2. Power-scaled Continuous Magnitude Weights (Eq 6):
   f_p(a) = eps_w + (1 - eps_w) * [max(0, a)]^p,   eps_w = 0.05, p in {1.0, 1.5, 2.0, 3.0}
   When semantic_mode in ('reference', 'binary'): f_p(a) = 1.0 (exact V4 unit occurrence).

3. Multimodal Fusion & Symmetrization (Eq 7):
   B_{ij} = sum_{m in {t, v}} 1[j in K_m(i)] f_p(a_{ij}^m)
   W_{ij} = max(B_{ij}, B_{ji}), with zero diagonal: W_{ii} = 0.

4. Conditional Relation-Proxy Gate (RPG, Eq 8-9):
   r_{ij}^{price} = min(p_i, p_j) / max(p_i, p_j)
   r_{ij}^{cat} = Jaccard(C_i, C_j)
   h_{ij} = r_{ij}^{price} * r_{ij}^{cat}
   g_{ij} = 1 - kappa * m_{ij} * sigmoid( (h_{ij} - tau_s) / T_s )
   W^g = W * g (elementwise). Default kappa = 0 (gate neutral, g_{ij} = 1).

5. Symmetric Normalization with Isolated-Node Identity Fallback (Section 6.4):
   D_i = sum_j W_{ij}^g
   S_{mag}[i, j] = W_{ij}^g / sqrt(D_i * D_j) for positive-degree nodes
   S_{mag}[i, i] = 1.0 for isolated nodes (D_i == 0).

6. Blended Operator Construction (Eq 10):
   S_{sem,8} = (1 - epsilon) * S_0 + epsilon * S_{mag}   (primary uses epsilon = 1.0)
   S_8 = 0.9 * S_{sem,8} + 0.1 * S_bar_{CF}

Theorem 1 Guarantee:
   ||S_8||_2 <= 1.0 for all epsilon in [0, 1].

Memory Contract:
   All graph construction, cosine evaluation, and symmetrization take place on the CPU
   or in bounded chunked buffers. Only the final coalesced sparse CSR tensor S_8 is transferred
   to the GPU.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.stair5_v4_graph import (
    CACHE_VERSION,
    _digest_array,
    build_candidate_support_graph,
    normalize_cf_graph_with_fallback,
    sparse_fingerprint,
    tensor_to_scipy,
    scipy_to_tensor,
    binary_train_keys,
)
from models.stair5_v4_utils import atomic_torch_save, file_sha256

ARMS_V8 = (
    "WMSG-core",
    "V4-control",
    "SAP-only",
    "WMSG-SAP",
    "WMSG-RPG",
    "WMSG-residual",
    "Full-stack",
    "WMSG-placebo",
    "WMSG-full",
)
CACHE_VERSION_V8 = 1


def resolve_v8_config(cfg):
    """Resolve named experiment arms to an explicit, idempotent configuration."""
    arm = getattr(cfg, "v8_arm", "WMSG-core")
    if arm not in ARMS_V8:
        raise ValueError(f"Unknown V8 arm: {arm}")
    if getattr(cfg, "semantic_mode", "weighted") not in ("weighted", "reference"):
        raise ValueError("Unknown semantic mode.")
    if getattr(cfg, "alignment", "off") not in ("off", "stratified_procrustes"):
        raise ValueError("Unknown alignment mode.")
    if getattr(cfg, "relation_gate", "off") not in ("off", "metadata_proxy"):
        raise ValueError("Unknown relation gate mode.")
    aligned = arm in ("SAP-only", "WMSG-SAP", "Full-stack")
    gated = arm in ("WMSG-RPG", "Full-stack")
    residual = arm in ("WMSG-residual", "Full-stack")
    cfg.alignment = "stratified_procrustes" if aligned else "off"
    cfg.relation_gate = "metadata_proxy" if gated else "off"
    strength = float(getattr(cfg, "relation_strength", 0.0))
    omega = float(getattr(cfg, "bsc_residual", 0.0))
    if not math.isfinite(strength) or not 0 <= strength <= 0.25:
        raise ValueError("Relation strength must be finite in [0, 0.25].")
    if not math.isfinite(omega) or not 0 <= omega <= 0.1:
        raise ValueError("BSC residual must be finite in [0, 0.1].")
    if float(getattr(cfg, "lambda_dirichlet", 0.0)) != 0:
        raise ValueError("V8 excludes a Dirichlet loss.")
    cfg.relation_strength = (strength if strength != 0 else 0.1) if gated else 0.0
    cfg.bsc_residual = (omega if omega != 0 else 0.05) if residual else 0.0
    if arm in ("V4-control", "SAP-only"):
        cfg.semantic_mode, cfg.semantic_mix = "reference", 0.0
    else:
        cfg.semantic_mode = "weighted"
    return cfg


@dataclass
class GraphStateV8:
    """Encapsulates the final coalesced sparse operator S_8 and its audit metadata."""
    operator: torch.Tensor
    metadata: dict


@torch.no_grad()
def compute_selected_edge_cosines(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int = 16384,
    eps_x: float = 1e-12,
    memory_budget_mib: float = 64.0,
) -> np.ndarray:
    """Computes exact cosine similarities for selected directed candidate edges in chunks.

    Args:
        features: FP32 tensor [N, D] of raw modality features.
        edge_index: LongTensor [2, E] of directed candidate edges (src, dst).
        chunk_size: Maximum number of edges to evaluate per vectorized chunk.
        eps_x: Numerical floor for L2 feature norms.

    Returns:
        cosines: 1D NumPy float32 array of length E, clamped to [-1.0, 1.0].
    """
    if features.device.type != "cpu" or edge_index.device.type != "cpu":
        raise ValueError("Selected-edge preprocessing requires CPU tensors.")
    if features.dtype != torch.float32 or features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Modality features must be a finite 2D tensor.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype != torch.int64:
        raise ValueError("Candidate IDs must be int64.")
    if chunk_size < 1 or not math.isfinite(eps_x) or eps_x <= 0:
        raise ValueError("Invalid cosine chunk size or norm floor.")
    if not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("Cosine memory budget must be finite and positive.")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= len(features)):
        raise ValueError("Candidate IDs are outside the catalog.")

    n_edges = edge_index.shape[1]
    if n_edges == 0:
        return np.empty(0, dtype=np.float32)

    # Compute row L2 norms with eps floor
    norms = torch.linalg.vector_norm(features, dim=1).clamp_min(eps_x)
    safe_chunk = int(memory_budget_mib * 2**20 // (3 * features.shape[1] * features.element_size()))
    if safe_chunk < 1:
        raise MemoryError("Cosine workspace cannot fit one edge.")
    chunk_size = min(chunk_size, safe_chunk)
    cosines = np.empty(n_edges, dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]

    for start in range(0, n_edges, chunk_size):
        stop = min(start + chunk_size, n_edges)
        s_chunk = src[start:stop]
        d_chunk = dst[start:stop]
        # Dot product of normalized rows
        left = features.index_select(0, s_chunk)
        right = features.index_select(0, d_chunk)
        chunk_cos = torch.einsum("ed,ed->e", left, right)
        chunk_cos.div_(norms[s_chunk] * norms[d_chunk]).clamp_(-1.0, 1.0)
        cosines[start:stop] = chunk_cos.numpy()
        del left, right, chunk_cos

    return cosines


def power_scale_similarities(
    cosines: np.ndarray,
    edge_power: float = 2.0,
    edge_floor: float = 0.05,
    semantic_mode: str = "weighted",
    p: Optional[float] = None,
    floor: Optional[float] = None,
) -> np.ndarray:
    """Applies power scaling f_p(a) = eps_w + (1 - eps_w) * [max(0, a)]^p (Eq 6).

    Args:
        cosines: 1D array of cosine similarities in [-1, 1].
        edge_power: Exponent p in {1.0, 1.5, 2.0, 3.0}.
        edge_floor: Minimum weight floor eps_w (default 0.05).
        semantic_mode: 'weighted' for continuous scaling, or 'reference'/'binary' for unit weights.
        p: Optional alias for edge_power.
        floor: Optional alias for edge_floor.

    Returns:
        1D float32 array of edge weights.
    """
    power = float(p if p is not None else edge_power)
    fl = float(floor if floor is not None else edge_floor)

    if semantic_mode not in ("weighted", "reference", "binary"):
        raise ValueError("Unknown semantic mode.")
    if not np.isfinite(cosines).all() or np.any(np.abs(cosines) > 1.000001):
        raise ValueError("Cosines must be finite and bounded.")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("edge_power must be finite and positive.")
    if not math.isfinite(fl) or not 0 < fl < 1:
        raise ValueError("edge_floor must be finite in (0, 1).")
    if semantic_mode in ("reference", "binary"):
        return np.ones_like(cosines, dtype=np.float32)

    # Rectified cosine: max(0, a)
    rectified = np.maximum(0.0, cosines).astype(np.float64)
    # Scaled magnitude: eps_w + (1 - eps_w) * rectified^p
    scaled = fl + (1.0 - fl) * np.power(rectified, power)
    return scaled.astype(np.float32)


def compute_relation_proxy_gate(
    undirected_pairs: Optional[np.ndarray] = None,
    metadata_dict: Optional[dict] = None,
    kappa: float = 0.0,
    tau_s: float = 0.7,
    T_s: float = 0.1,
    u_indices: Optional[np.ndarray] = None,
    v_indices: Optional[np.ndarray] = None,
    n_items: Optional[int] = None,
) -> np.ndarray:
    """Computes the symmetric retention gate g_ij = 1 - kappa * m_ij * sigmoid((h_ij - tau_s) / T_s) (Eq 9).

    Args:
        undirected_pairs: Array of shape [E, 2] containing item ID pairs (i, j).
        metadata_dict: Dict optionally containing 'prices' (array [N]) and 'categories' (dict/list of sets).
        kappa: Retention attenuation strength in [0.0, 0.25]. Default 0.0 (neutral).
        tau_s: Similarity threshold parameter (default 0.7).
        T_s: Sigmoid temperature parameter (default 0.1).
        u_indices, v_indices: Optional 1D arrays of endpoints instead of undirected_pairs.
        n_items: Total item count.

    Returns:
        1D float32 array of retention coefficients g_ij in [1 - kappa, 1.0].
    """
    if undirected_pairs is None:
        if u_indices is not None and v_indices is not None:
            undirected_pairs = np.stack([u_indices, v_indices], axis=1)
        else:
            return np.empty(0, dtype=np.float32)

    n_pairs = len(undirected_pairs)
    if not math.isfinite(kappa) or not 0 <= kappa <= 0.25:
        raise ValueError("Relation strength must be in [0, 0.25].")
    if not math.isfinite(T_s) or T_s <= 0 or not math.isfinite(tau_s):
        raise ValueError("Invalid relation gate temperature or threshold.")
    if n_pairs == 0 or kappa == 0.0 or metadata_dict is None:
        return np.ones(n_pairs, dtype=np.float32)

    prices = metadata_dict.get("prices")
    categories = metadata_dict.get("categories")

    if prices is None or categories is None:
        # Neutral fallback when metadata is unavailable
        return np.ones(n_pairs, dtype=np.float32)

    g = np.ones(n_pairs, dtype=np.float32)
    u_idx = undirected_pairs[:, 0]
    v_idx = undirected_pairs[:, 1]

    # Vectorized price ratios
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 1 or (n_items is not None and len(prices) != n_items):
        raise ValueError("Prices must match the item mapping.")
    if np.any(undirected_pairs < 0) or np.any(undirected_pairs >= len(prices)):
        raise ValueError("Metadata endpoints are outside the catalog.")
    price_mask = np.asarray(metadata_dict.get("price_mask", np.ones(len(prices), dtype=bool)), dtype=bool)
    category_mask = np.asarray(metadata_dict.get("category_mask", np.ones(len(prices), dtype=bool)), dtype=bool)
    if price_mask.shape != prices.shape or category_mask.shape != prices.shape:
        raise ValueError("Metadata validity masks must match prices.")
    p_u = prices[u_idx]
    p_v = prices[v_idx]
    valid_price = (p_u > 0) & (p_v > 0) & np.isfinite(p_u) & np.isfinite(p_v)
    valid_price &= price_mask[u_idx] & price_mask[v_idx]
    currencies = metadata_dict.get("currencies")
    if currencies is not None:
        currencies = np.asarray(currencies)
        if currencies.shape != prices.shape:
            raise ValueError("Currency IDs must match the catalog.")
        valid_price &= currencies[u_idx] == currencies[v_idx]
    roots = set(metadata_dict.get("generic_category_roots", ()))

    p_min = np.minimum(p_u, p_v)
    p_max = np.maximum(p_u, p_v)
    r_price = np.zeros(n_pairs, dtype=np.float64)
    r_price[valid_price] = p_min[valid_price] / p_max[valid_price]

    # Category Jaccard
    r_cat = np.zeros(n_pairs, dtype=np.float64)
    valid_cat = np.zeros(n_pairs, dtype=bool)

    for idx, (i, j) in enumerate(undirected_pairs):
        if not valid_price[idx]:
            continue
        if not (category_mask[i] and category_mask[j]):
            continue
        c_i = categories.get(int(i), categories.get(str(i))) if isinstance(categories, dict) else (categories[i] if i < len(categories) else None)
        c_j = categories.get(int(j), categories.get(str(j))) if isinstance(categories, dict) else (categories[j] if j < len(categories) else None)
        if c_i and c_j and len(c_i) > 0 and len(c_j) > 0:
            set_i = set(c_i) - roots
            set_j = set(c_j) - roots
            if not set_i or not set_j:
                continue
            union = len(set_i | set_j)
            if union > 0:
                r_cat[idx] = len(set_i & set_j) / union
                valid_cat[idx] = True

    m_ij = valid_price & valid_cat
    if not np.any(m_ij):
        return np.ones(n_pairs, dtype=np.float32)

    h_ij = r_price * r_cat
    # Numerically stable sigmoid: 1 / (1 + exp(-x))
    x = (h_ij[m_ij] - tau_s) / T_s
    x = np.clip(x, -50.0, 50.0)
    sig = 1.0 / (1.0 + np.exp(-x))
    g[m_ij] = (1.0 - kappa * sig).astype(np.float32)

    return g


def normalize_graph_with_isolated_fallback(W: sp.csr_matrix) -> sp.csr_matrix:
    """Computes symmetric normalization N(W) = D^{-1/2} W D^{-1/2} with isolated identity blocks.

    Isolated nodes (d_i == 0) receive an identity block (entry 1.0 on diagonal) to guarantee
    ||N(W)||_2 <= 1.0 under Theorem 1.
    """
    if W.shape[0] != W.shape[1]:
        raise ValueError("Graph matrix must be square.")
    W = W.astype(np.float64).tocsr(copy=True)
    W.sum_duplicates()
    W.eliminate_zeros()
    if not np.isfinite(W.data).all() or np.any(W.data < 0):
        raise ValueError("Graph weights must be finite and nonnegative.")
    n = W.shape[0]

    # Symmetrize check
    diff = W - W.T
    diff.eliminate_zeros()
    if diff.nnz:
        raise ValueError("Matrix must be strictly symmetric before normalization.")

    degrees = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    inv_sqrt = np.zeros(n, dtype=np.float64)
    pos_mask = degrees > 0
    inv_sqrt[pos_mask] = 1.0 / np.sqrt(degrees[pos_mask])

    scale = sp.diags(inv_sqrt)
    normalized = (scale @ W @ scale).astype(np.float32)

    # Insert identity for isolated items
    isolated_mask = (degrees == 0).astype(np.float32)
    if np.any(isolated_mask > 0):
        normalized = normalized + sp.diags(isolated_mask)

    normalized = normalized.tocsr()
    normalized.sum_duplicates()
    normalized.eliminate_zeros()
    normalized.sort_indices()
    return normalized


def build_wmsg_semantic_graph(
    mfeats: List[torch.Tensor],
    knn_edges_list: List[torch.Tensor],
    n_items: int,
    semantic_mode: str = "weighted",
    edge_power: float = 2.0,
    edge_floor: float = 0.05,
    relation_gate: str = "off",
    relation_strength: float = 0.0,
    relation_threshold: float = 0.7,
    relation_temperature: float = 0.1,
    metadata_dict: Optional[dict] = None,
) -> Tuple[sp.csr_matrix, dict]:
    """Constructs the WMSG continuous semantic graph W^g and its audit statistics (Stage 2).

    Steps:
    1. Compute selected-edge cosines for each modality.
    2. Power scale magnitudes: f_p(a) = eps_w + (1 - eps_w) * [max(0, a)]^p.
    3. Sum directed occurrences across modalities.
    4. Max-symmetrize W_ij = max(B_ij, B_ji) with zero diagonal.
    5. Optionally apply symmetric relation-proxy gate g_ij.
    """
    if not mfeats or len(mfeats) != len(knn_edges_list):
        raise ValueError("Number of modality features must match number of kNN edge sets.")
    if relation_gate not in ("off", "metadata_proxy"):
        raise ValueError("Unknown relation gate mode.")

    all_src = []
    all_dst = []
    all_weights = []
    audit_stats = {}

    for idx, (feat, edges) in enumerate(zip(mfeats, knn_edges_list)):
        mod_name = f"modality_{idx}"
        edges_np = edges.detach().cpu().numpy().astype(np.int64)
        cosines = compute_selected_edge_cosines(feat, edges)
        weights = power_scale_similarities(
            cosines,
            edge_power=edge_power,
            edge_floor=edge_floor,
            semantic_mode=semantic_mode,
        )

        nonpos_fraction = float(np.mean(cosines <= 0.0)) if len(cosines) else 0.0
        audit_stats[f"{mod_name}_edges_count"] = int(edges.shape[1])
        audit_stats[f"{mod_name}_cosine_mean"] = float(np.mean(cosines)) if len(cosines) else 0.0
        audit_stats[f"{mod_name}_cosine_std"] = float(np.std(cosines)) if len(cosines) else 0.0
        audit_stats[f"{mod_name}_nonpositive_cosine_fraction"] = nonpos_fraction
        audit_stats[f"{mod_name}_weight_mean"] = float(np.mean(weights)) if len(weights) else 0.0
        audit_stats[f"{mod_name}_weight_min"] = float(np.min(weights)) if len(weights) else 0.0
        audit_stats[f"{mod_name}_weight_max"] = float(np.max(weights)) if len(weights) else 0.0

        all_src.append(edges_np[0])
        all_dst.append(edges_np[1])
        all_weights.append(weights)

    flat_src = np.concatenate(all_src)
    flat_dst = np.concatenate(all_dst)
    flat_weights = np.concatenate(all_weights)

    # Coalesce directed occurrence weights by summation: B_ij = sum_m 1[j in K_m(i)] f_p(a_{ij}^m)
    B = sp.csr_matrix((flat_weights, (flat_src, flat_dst)), shape=(n_items, n_items))
    B.sum_duplicates()

    # Symmetrize by maximum: W_ij = max(B_ij, B_ji) with zero diagonal: W_ii = 0
    W = B.maximum(B.T).tocsr()
    W.setdiag(0.0)
    W.eliminate_zeros()

    audit_stats["wmsg_symmetric_nnz"] = int(W.nnz)
    audit_stats["wmsg_max_weight"] = float(W.data.max()) if W.nnz else 0.0
    audit_stats["wmsg_min_weight"] = float(W.data.min()) if W.nnz else 0.0

    # Stage 2 RPG extension (default: off)
    if relation_gate == "metadata_proxy" and relation_strength > 0.0 and metadata_dict is not None:
        coo = W.tocoo()
        pairs = np.column_stack((coo.row, coo.col))
        gate = compute_relation_proxy_gate(
            pairs,
            metadata_dict=metadata_dict,
            kappa=relation_strength,
            tau_s=relation_threshold,
            T_s=relation_temperature,
            n_items=n_items,
        )
        W.data = (W.data * gate).astype(np.float32)
        audit_stats["rpg_active"] = bool(np.any(gate < 1.0))
        audit_stats["rpg_affected_fraction"] = float(np.mean(gate < 1.0)) if len(gate) else 0.0
        audit_stats["rpg_mean_retention"] = float(np.mean(gate)) if len(gate) else 1.0
        audit_stats["rpg_min_retention"] = float(np.min(gate)) if len(gate) else 1.0
    else:
        audit_stats["rpg_active"] = False
        if relation_gate == "metadata_proxy" and relation_strength > 0:
            warnings.warn("RPG is neutral because item-aligned metadata is unavailable.", RuntimeWarning)
            audit_stats["rpg_fallback_reason"] = "metadata_unavailable"

    return W, audit_stats


def build_calibrated_graph_v8(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    mfeats: List[torch.Tensor],
    knn_edges_list: List[torch.Tensor],
    v8_arm: str = "WMSG-core",
    semantic_mode: str = "weighted",
    edge_power: float = 2.0,
    edge_floor: float = 0.05,
    semantic_mix: float = 1.0,
    relation_gate: str = "off",
    relation_strength: float = 0.0,
    relation_threshold: float = 0.7,
    relation_temperature: float = 0.1,
    metadata_dict: Optional[dict] = None,
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    cache_dir: Optional[str] = None,
    block_size: int = 64,
    memory_budget_mib: int = 128,
    placebo_seed: int = 1,
) -> GraphStateV8:
    """Builds the final coalesced STAIR5-v8 operator S_8 (Stages 2 & 3).

    Pipeline:
    1. Exact V4-control bypass if arm == 'V4-control' or semantic_mode == 'reference' with semantic_mix == 0.0.
    2. Builds WMSG graph W^g and normalizes: S_{mag} = N(W^g).
    3. Semantic blend: S_{sem,8} = (1 - semantic_mix) * S_0 + semantic_mix * S_{mag}.
    4. Computes or loads cached exact V4 CF operator S_bar_{CF}.
    5. Final CSE blend: S_8 = (1 - eta) * S_{sem,8} + eta * S_bar_{CF} (default eta = 0.1).
    6. Returns GraphStateV8 with single CSR tensor and comprehensive audit metadata.
    """
    if v8_arm not in ARMS_V8:
        raise ValueError(f"Unknown v8_arm: {v8_arm}. Must be one of {ARMS_V8}")
    if not (0.0 <= semantic_mix <= 1.0):
        raise ValueError("semantic_mix (epsilon) must be in [0.0, 1.0].")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0.0, 1.0].")

    metadata = {
        "version": 8,
        "v8_arm": v8_arm,
        "semantic_mode": semantic_mode,
        "edge_power": float(edge_power),
        "edge_floor": float(edge_floor),
        "semantic_mix": float(semantic_mix),
        "relation_gate": relation_gate,
        "relation_strength": float(relation_strength),
        "eta": float(eta),
        "k_cf": int(k_cf),
        "c_min": int(c_min),
        "t_shrinkage": float(t_shrinkage),
        "num_users": n_users,
        "num_items": n_items,
        "cache_hit": False,
    }

    device = baseline_normalized.device
    S0_scipy = tensor_to_scipy(baseline_normalized).astype(np.float32)

    # 1. Exact V4-control bypass
    if v8_arm in ("V4-control", "SAP-only") or semantic_mix == 0.0:
        metadata.update(active=False, note="Exact STAIR5-v4 control reference bypass.")
        from models.stair5_v4_graph import build_calibrated_graph_v4
        v4_state = build_calibrated_graph_v4(
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
            cache_dir=cache_dir,
            block_size=block_size,
            memory_budget_mib=memory_budget_mib,
        )
        metadata.update(v4_state.metadata)
        metadata["version"] = 8
        metadata["active"] = False
        metadata["v8_active"] = False
        metadata["v4_bypass"] = True
        metadata["graph_fingerprint"] = sparse_fingerprint(tensor_to_scipy(v4_state.operator))
        return GraphStateV8(v4_state.operator, metadata)

    started = time.perf_counter()

    # 2. Build WMSG graph W^g
    W_g, wmsg_audit = build_wmsg_semantic_graph(
        mfeats=mfeats,
        knn_edges_list=knn_edges_list,
        n_items=n_items,
        semantic_mode=semantic_mode,
        edge_power=edge_power,
        edge_floor=edge_floor,
        relation_gate=relation_gate,
        relation_strength=relation_strength,
        relation_threshold=relation_threshold,
        relation_temperature=relation_temperature,
        metadata_dict=metadata_dict,
    )
    metadata.update(wmsg_audit)

    if v8_arm == "WMSG-placebo":
        # Shuffle undirected magnitudes within modality-membership and train-degree strata.
        unique_items = binary_train_keys(train_edges, n_users, n_items) % n_items
        degree = np.bincount(unique_items, minlength=n_items)
        order = np.lexsort((np.arange(n_items), degree))
        strata = np.empty(n_items, dtype=np.int64)
        for index, ids in enumerate(np.array_split(order, 5)):
            strata[ids] = index
        upper = sp.triu(W_g, k=1).tocoo()
        membership = np.zeros(upper.nnz, dtype=np.int64)
        for index, edges in enumerate(knn_edges_list):
            ids = edges.detach().cpu().numpy()
            support = sp.csr_matrix((np.ones(ids.shape[1]), (ids[0], ids[1])), shape=W_g.shape)
            support = support.maximum(support.T)
            membership |= (np.asarray(support[upper.row, upper.col]).ravel() > 0).astype(np.int64) << index
        lo = np.minimum(strata[upper.row], strata[upper.col])
        hi = np.maximum(strata[upper.row], strata[upper.col])
        groups = membership * 25 + lo * 5 + hi
        rng = np.random.default_rng(placebo_seed)
        weights = upper.data.copy()
        changed_groups = 0
        for group in np.unique(groups):
            positions = np.flatnonzero(groups == group)
            if len(positions) > 1:
                weights[positions] = rng.permutation(weights[positions])
                changed_groups += 1
        shuffled = sp.csr_matrix((weights, (upper.row, upper.col)), shape=W_g.shape)
        W_g = shuffled + shuffled.T
        metadata.update(placebo_seed=int(placebo_seed), placebo=True,
                        placebo_shuffle_groups=changed_groups,
                        placebo_changed_edges=int(np.count_nonzero(weights != upper.data)),
                        placebo_degree_preservation="stratified; not exact weighted degrees")

    # Normalize S_{mag} = N(W^g)
    S_mag = normalize_graph_with_isolated_fallback(W_g)

    # 3. Semantic blend: S_{sem,8} = (1 - epsilon) * S_0 + epsilon * S_{mag}
    if semantic_mix == 1.0:
        S_sem_8 = S_mag
    elif semantic_mix == 0.0:
        S_sem_8 = S0_scipy
    else:
        S_sem_8 = ((1.0 - semantic_mix) * S0_scipy + semantic_mix * S_mag).tocsr()
        S_sem_8.eliminate_zeros()

    # 4. Candidate Support Expansion (CF branch - exact V4 cached Ochiai builder)
    W_cf = None
    pairs = binary_train_keys(train_edges, n_users, n_items)
    pair_hash = hashlib.sha256()
    _digest_array(pair_hash, pairs)
    signature_cf = {
        "cache_version": CACHE_VERSION,
        "train": pair_hash.hexdigest(),
        "users": n_users,
        "items": n_items,
        "k": k_cf,
        "cmin": c_min,
        "t": t_shrinkage,
        "source": file_sha256(Path(__file__).with_name("stair5_v4_graph.py")),
    }
    cf_cache_key = hashlib.sha256(json.dumps(signature_cf, sort_keys=True).encode()).hexdigest()
    cf_cache_file = Path(cache_dir) / f"cf_{cf_cache_key}.pt" if cache_dir else None

    if cf_cache_file and cf_cache_file.is_file():
        try:
            payload = torch.load(cf_cache_file, map_location="cpu", weights_only=True)
            candidate = tensor_to_scipy(payload["operator"])
            if candidate.shape != (n_items, n_items) or payload.get("signature") != signature_cf:
                raise ValueError("CF cache shape/signature mismatch.")
            if payload.get("fingerprint") != sparse_fingerprint(candidate):
                raise ValueError("CF cache content checksum mismatch.")
            W_cf = candidate
            metadata["cf_cache_hit"] = True
        except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
            warnings.warn(f"Ignoring invalid CF cache: {exc}", RuntimeWarning)
            W_cf = None

    if W_cf is None:
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
        if cf_cache_file:
            atomic_torch_save(
                {
                    "operator": scipy_to_tensor(W_cf),
                    "signature": signature_cf,
                    "fingerprint": sparse_fingerprint(W_cf),
                },
                cf_cache_file,
            )

    effective_eta = eta if W_cf.nnz and k_cf > 0 else 0.0
    # 5. Final CSE blend: S_8 = (1 - eta) * S_{sem,8} + eta * S_bar_{CF} (Eq 10)
    if effective_eta == 0.0:
        S_8 = S_sem_8.copy()
    else:
        S_bar_cf = normalize_cf_graph_with_fallback(W_cf)
        S_8 = ((1.0 - effective_eta) * S_sem_8 + effective_eta * S_bar_cf).tocsr()
    S_8.sum_duplicates()
    S_8.eliminate_zeros()
    S_8.sort_indices()

    # Operator Delta vs V4 baseline S4
    delta_s0 = S_8 - S0_scipy
    fro_s0 = np.sqrt(np.dot(S0_scipy.data.astype(np.float64), S0_scipy.data.astype(np.float64)))
    fro_delta = np.sqrt(np.dot(delta_s0.data.astype(np.float64), delta_s0.data.astype(np.float64)))
    relative_perturbation = float(fro_delta / max(1e-12, fro_s0))

    elapsed = time.perf_counter() - started
    final_fp = sparse_fingerprint(S_8)

    metadata.update(
        active=True,
        effective_eta=float(effective_eta),
        s8_nnz=int(S_8.nnz),
        s0_nnz=int(S0_scipy.nnz),
        cf_nnz=int(W_cf.nnz),
        relative_operator_delta_vs_s0=relative_perturbation,
        graph_build_seconds=elapsed,
        graph_fingerprint=final_fp,
    )

    final_tensor = scipy_to_tensor(S_8).to(device)
    return GraphStateV8(final_tensor, metadata)
