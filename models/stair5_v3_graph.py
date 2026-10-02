"""STAIR5-v3 Degree-Preserving Preference-Compatible Graph (DP-PC-BSC).

Implements candidate-restricted co-occurrence statistics with a convex dual
symmetric KL projection solver that strictly preserves the original baseline
weighted node degrees while redistributing edge mass.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Optional

import numpy as np
from scipy.optimize import minimize
import torch


@dataclass
class GraphCalibrationV3:
    operator: torch.Tensor
    support_scores: torch.Tensor
    item_degrees: torch.Tensor
    common_counts: torch.Tensor
    metadata: dict


def _coo(graph: torch.Tensor) -> torch.Tensor:
    if graph.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise ValueError("Expected a sparse COO/CSR graph.")
    graph = graph.detach().to(device="cpu").to_sparse_coo().coalesce()
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("The graph must be square.")
    if not torch.isfinite(graph.values()).all() or (graph.values() < 0).any():
        raise ValueError("Graph weights must be finite and nonnegative.")
    transpose = graph.transpose(0, 1).coalesce()
    if not torch.equal(graph.indices(), transpose.indices()) or not torch.allclose(
        graph.values(), transpose.values(), rtol=1e-6, atol=1e-8
    ):
        raise ValueError("The graph must be symmetric before calibration.")
    return graph


def normalize_graph(raw: torch.Tensor) -> torch.Tensor:
    """Symmetric degree normalization; zero-degree vertices remain isolated."""
    graph = _coo(raw)
    rows, cols = graph.indices()
    degree = torch.zeros(graph.shape[0], dtype=graph.dtype)
    degree.scatter_add_(0, rows, graph.values())
    inv = torch.zeros_like(degree)
    valid = degree > 0
    inv[valid] = degree[valid].rsqrt()
    values = graph.values() * inv[rows] * inv[cols]
    return torch.sparse_coo_tensor(graph.indices(), values, graph.shape).coalesce().to_sparse_csr()


def _binary_edges(edges: torch.Tensor, num_users: int, num_items: int) -> torch.Tensor:
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("train_edges must be an int64 tensor of shape [2, E].")
    edges = edges.detach().cpu()
    if num_users <= 0 or num_items <= 0:
        raise ValueError("Node counts must be positive.")
    if edges.numel() and ((edges < 0).any() or (edges[0] >= num_users).any()
                         or (edges[1] >= num_items).any()):
        raise ValueError("Train interaction IDs are outside the declared node range.")
    keys = torch.unique(edges[0] * num_items + edges[1], sorted=True)
    return torch.stack((keys // num_items, keys % num_items))


def candidate_statistics(raw: torch.Tensor, train_edges: torch.Tensor, num_users: int,
                         shrinkage: float = 5.0):
    """Return (q, train item degrees, common-user counts), aligned to COO edges.

    Intersect sorted item-user lists for each undirected candidate. Intersection
    uses binary search in the longer list; cost depends on actual degrees.
    """
    if not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("evidence_shrinkage must be finite and positive.")
    graph = _coo(raw)
    n_items = graph.shape[0]
    edges = _binary_edges(train_edges, num_users, n_items).numpy()
    order = np.lexsort((edges[0], edges[1]))
    users, items = edges[0, order], edges[1, order]
    degrees = np.bincount(items, minlength=n_items)
    offsets = np.concatenate(([0], np.cumsum(degrees)))
    row, col = graph.indices().numpy()
    counts = np.zeros(len(row), dtype=np.int64)
    keys = row * n_items + col
    for pos in np.flatnonzero(row <= col):
        i, j = row[pos], col[pos]
        left = users[offsets[i]:offsets[i + 1]]
        right = users[offsets[j]:offsets[j + 1]]
        if len(left) > len(right):
            left, right = right, left
        if len(left) and len(right):
            indexes = np.searchsorted(right, left)
            valid = indexes < len(right)
            count = np.count_nonzero(right[indexes[valid]] == left[valid])
        else:
            count = 0
        counts[pos] = count
        reverse = np.searchsorted(keys, j * n_items + i)
        counts[reverse] = count
    denom = np.sqrt(degrees[row].astype(np.float64) * degrees[col])
    ochiai = np.divide(counts, denom, out=np.zeros(len(row), dtype=np.float64), where=denom > 0)
    scores = counts / (counts + shrinkage) * ochiai
    return (torch.from_numpy(scores).to(graph.dtype), torch.from_numpy(degrees),
            torch.from_numpy(counts))


def _placebo(scores: torch.Tensor, raw: torch.Tensor, degrees: torch.Tensor, seed: int):
    """Permute undirected edge scores within log2 endpoint-degree strata."""
    row, col = raw.indices()
    unique = torch.nonzero(row <= col).flatten()
    bins = torch.floor(torch.log2(degrees.float().clamp_min(1))).long()
    low = torch.minimum(bins[row[unique]], bins[col[unique]])
    high = torch.maximum(bins[row[unique]], bins[col[unique]])
    strata = low * (int(bins.max()) + 1) + high
    result = scores.clone()
    generator = torch.Generator().manual_seed(seed)
    shuffled = 0
    for group in torch.unique(strata):
        indexes = unique[strata == group]
        perm = torch.randperm(len(indexes), generator=generator)
        result[indexes] = scores[indexes[perm]]
        shuffled += int((perm != torch.arange(len(indexes))).sum())
    keys = row * raw.shape[0] + col
    reverse = torch.searchsorted(keys, col[unique] * raw.shape[0] + row[unique])
    result[reverse] = result[unique]
    return result, shuffled


def project_to_original_degrees(raw: torch.Tensor, scores: torch.Tensor, strength: float = 0.25,
                                max_iter: int = 200, tol: float = 1e-6) -> tuple[torch.Tensor, dict]:
    """Solve the symmetric KL projection to preserve original row sums d^0.

    Minimizes:
        F(v) = 0.5 * sum_{(i,j)} K_{ij} * exp(v_i + v_j) - sum_i d_i^0 * v_i
    subject to v_i = 0 on isolated vertices.

    Returns:
        projected: torch.sparse_coo_tensor of same shape and support as raw.
        diagnostics: dict with solver statistics and residuals.
    """
    graph = _coo(raw)
    n_items = graph.shape[0]
    row, col = graph.indices().numpy()
    w0_vals = graph.values().numpy().astype(np.float64)
    q_vals = scores.detach().cpu().numpy().astype(np.float64)
    k_vals = w0_vals * (1.0 + strength * q_vals)

    # Initial degrees d^0
    d0 = np.bincount(row, weights=w0_vals, minlength=n_items)
    active_mask = d0 > 0
    active_indices = np.flatnonzero(active_mask)
    n_active = len(active_indices)

    if n_active == 0 or strength == 0 or np.all(q_vals == 0):
        # Fast path: W* = W0
        diagnostics = {
            "converged": True, "iterations": 0, "max_relative_residual_fp64": 0.0,
            "max_relative_residual_fp32": 0.0, "status": "exact_identity_fastpath",
            "max_weight_change": 0.0, "mean_weight_change": 0.0
        }
        return graph, diagnostics

    # Map full node indices to active sub-indices
    node_map = np.full(n_items, -1, dtype=np.int64)
    node_map[active_indices] = np.arange(n_active, dtype=np.int64)

    active_row = node_map[row]
    active_col = node_map[col]
    active_d0 = d0[active_indices]

    def objective_and_grad(v_active):
        v_row = v_active[active_row]
        v_col = v_active[active_col]
        # Bounded exponent to prevent overflow in line search
        v_sum = np.clip(v_row + v_col, -50.0, 50.0)
        exp_term = np.exp(v_sum)
        w_current = k_vals * exp_term
        val = 0.5 * np.sum(w_current) - np.dot(active_d0, v_active)
        row_sums = np.bincount(active_row, weights=w_current, minlength=n_active)
        grad = row_sums - active_d0
        return val, grad

    # Optimize using L-BFGS-B in FP64
    init_v = np.zeros(n_active, dtype=np.float64)
    res = minimize(
        objective_and_grad,
        init_v,
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": max_iter}
    )

    v_opt = np.zeros(n_items, dtype=np.float64)
    v_opt[active_indices] = res.x

    w_star_fp64 = k_vals * np.exp(v_opt[row] + v_opt[col])
    d_star_fp64 = np.bincount(row, weights=w_star_fp64, minlength=n_items)

    abs_res_fp64 = np.abs(d_star_fp64[active_indices] - d0[active_indices])
    rel_res_fp64 = float(np.max(abs_res_fp64 / d0[active_indices]))

    w_star_fp32 = torch.from_numpy(w_star_fp64).to(graph.dtype)
    projected = torch.sparse_coo_tensor(graph.indices(), w_star_fp32, graph.shape).coalesce()

    # Check FP32 cast residual
    d_star_fp32 = torch.zeros(n_items, dtype=graph.dtype)
    d_star_fp32.scatter_add_(0, projected.indices()[0], projected.values())
    d0_tensor = torch.from_numpy(d0).to(graph.dtype)
    abs_res_fp32 = (d_star_fp32[active_indices] - d0_tensor[active_indices]).abs()
    rel_res_fp32 = float((abs_res_fp32 / d0_tensor[active_indices]).max().item())

    weight_diff = (w_star_fp64 - w0_vals)
    diagnostics = {
        "converged": bool(res.success or rel_res_fp64 <= tol),
        "iterations": int(res.nit),
        "max_relative_residual_fp64": rel_res_fp64,
        "max_relative_residual_fp32": rel_res_fp32,
        "status": str(res.message),
        "max_weight_change": float(np.max(np.abs(weight_diff))),
        "mean_weight_change": float(np.mean(np.abs(weight_diff))),
    }

    if rel_res_fp64 > 1e-3:
        raise RuntimeError(
            f"Degree-preserving KL solver failed to converge: relative residual {rel_res_fp64:.2e} > 1e-3. "
            f"Diagnostics: {diagnostics}"
        )

    return projected, diagnostics


def _fingerprint(raw, baseline, edges, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for tensor in (raw.indices(), raw.values(), baseline.indices(), baseline.values(), edges):
        array = tensor.contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_calibrated_graph_v3(raw_graph, baseline_graph, train_edges, num_users,
                              arm="DP-ref", strength=0.25, mix=0.25, shrinkage=5.0,
                              placebo_seed=1, seed=1, cache_dir: Optional[str] = None) -> GraphCalibrationV3:
    """Build a frozen sparse operator according to the specified STAIR5-v3 arm.

    Arms supported:
      - 'B0': Exact fast-path baseline S0.
      - 'ET-ref': v2 comparator (degree normalization of boosted K, then blended).
      - 'DP-ref': v3a primary (degree-preserving symmetric KL projection, then blended).
      - 'DP-placebo': DP projection on stratified degree-shuffled scores.
      - 'DP-zero': Negative control (scores set to 0, mathematically recovers S0).
    """
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("edge_strength must be >= 0 and edge_mix must be in [0, 1].")
    if not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("evidence_shrinkage must be positive.")

    # Fast-path for baseline B0, DP-zero or zero-mix
    if arm in ("B0", "DP-zero") or strength == 0 or mix == 0:
        empty = torch.empty(0)
        return GraphCalibrationV3(baseline_graph, empty, empty.long(), empty.long(),
                                  {"active": False, "arm": arm, "strength": strength, "mix": mix})

    raw, baseline = _coo(raw_graph), _coo(baseline_graph)
    if raw.shape != baseline.shape or not torch.equal(raw.indices(), baseline.indices()):
        raise ValueError("Raw and baseline graphs must have matching support.")
    expected = normalize_graph(raw).to_sparse_coo().coalesce()
    if not torch.allclose(expected.values(), baseline.values(), rtol=2e-5, atol=1e-7):
        raise ValueError("baseline_graph is not the normalized supplied raw graph.")

    edges = _binary_edges(train_edges, num_users, raw.shape[0])
    settings = dict(version=3, arm=arm, strength=strength, mix=mix, shrinkage=shrinkage,
                    placebo_seed=placebo_seed, seed=seed, num_users=num_users,
                    num_items=raw.shape[0], torch=str(torch.__version__))
    key = _fingerprint(raw, baseline, edges, settings)
    cache = Path(cache_dir) / f"dp_pc_bsc_{key}.pt" if cache_dir else None

    if cache and cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if payload["metadata"]["fingerprint"] != key:
            raise ValueError("Graph cache fingerprint mismatch.")
        payload["metadata"]["cache_hit"] = True
        return GraphCalibrationV3(**payload)

    scores, degrees, counts = candidate_statistics(raw, edges, num_users, shrinkage)
    permuted = 0

    if arm == "DP-zero":
        scores = torch.zeros_like(scores)
    elif arm == "DP-placebo":
        scores, permuted = _placebo(scores, raw, degrees, placebo_seed)

    solver_diag = {}
    if arm in ("DP-ref", "DP-placebo", "DP-zero"):
        projected, solver_diag = project_to_original_degrees(raw, scores, strength=strength)
        # S_* = D_0^(-1/2) W^* D_0^(-1/2)
        s_star = normalize_graph(projected).to_sparse_coo().coalesce()
        values = (1 - mix) * baseline.values() + mix * s_star.values()
    elif arm == "ET-ref":
        # v2 normalization comparator
        boosted = torch.sparse_coo_tensor(raw.indices(), raw.values() * (1 + strength * scores), raw.shape)
        plus = normalize_graph(boosted).to_sparse_coo().coalesce()
        values = (1 - mix) * baseline.values() + mix * plus.values()
    else:
        raise ValueError(f"Unknown graph arm: {arm}")

    operator = torch.sparse_coo_tensor(raw.indices(), values, raw.shape).coalesce().to_sparse_csr()

    # Diagnostics probe
    probes = torch.randn((raw.shape[0], 4), generator=torch.Generator().manual_seed(0), dtype=raw.dtype)
    baseline_action = baseline @ probes
    op_action = operator @ probes
    probe_delta = float((op_action - baseline_action).norm() / baseline_action.norm().clamp_min(1e-12))

    metadata = dict(
        settings, active=True, fingerprint=key, cache_hit=False, nnz=raw._nnz(),
        nonzero_score_fraction=float((scores > 0).float().mean()) if len(scores) else 0.,
        score_mean=float(scores.mean()) if len(scores) else 0.,
        score_max=float(scores.max()) if len(scores) else 0.,
        placebo_permuted=permuted,
        relative_operator_probe_delta=probe_delta,
        solver=solver_diag,
    )

    result = GraphCalibrationV3(operator, scores, degrees, counts, metadata)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".dp_bsc_", suffix=".pt", dir=cache.parent)
        os.close(fd)
        try:
            torch.save(vars(result), temporary)
            os.replace(temporary, cache)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    return result


# Public aliases for backwards-compatibility with test suites and specifications
candidate_statistics_v3 = candidate_statistics
solve_degree_preserving_projection = project_to_original_degrees
