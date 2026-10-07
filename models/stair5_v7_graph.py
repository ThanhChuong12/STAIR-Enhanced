"""models/stair5_v7_graph.py — STAIR5-v7 (UCR-D) Graph & Operator Architecture.
=============================================================================
Update-Compatible Retention with Dose Control (UCR-D) on Item BSC Semantic Branch.

Mathematical & Architectural Principles (§3 of STAIR5_v7_Report.md):
1. Intercepts actual Adam update direction D_t (detached view).
2. Computes row magnitude r_i = ||D_{t, i:}||_2 and threshold tau_mag = max(1e-12, zeta * median_{r_i > 0}(r_i)).
3. Evaluates pair conflict q_ij = max(0, -v_ij) on eligible edges (r_i >= tau_mag, r_j >= tau_mag) in chunks (C=4096).
4. Calibrates dose:
     rho_t = rho * min(1.0, epoch_progress / 10.0)
     H = sum_{i != j} W_{0, ij}
     Z_t = H^{-1} sum_{i != j} W_{0, ij} q_{ij}^{(t)}
     theta_t = min(theta_max, rho_t / Z_t) if Z_t > 0 else 0.0
     a_ij = theta_t * q_ij
     A_t = W_0 * a_t
5. Returns suppressed off-diagonal mass to endpoint diagonals:
     h_{t, i} = sum_{j != i} A_{t, ij}
     W_R^{(t)} = W_0 - A_t + diag(h_t) => sum_j W_{R, ij} = d_{0, i}
6. Blended composite operator with pre-allocated unified CSR topology:
     S_7^{(t)} = S_4 - 0.9 * D_0^{-1/2} A_t D_0^{-1/2} + 0.9 * diag(h_{t, i} / d_{0, i})
7. Zero-allocation fast-path delegation when rho=0 or arm in ('V4-control', 'V7-recovery').
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import scipy.sparse as sp
import torch

from models.stair5_v7_utils import (
    atomic_torch_save,
    sparse_fingerprint,
    validate_csr_operator,
)

ARMS_V7 = (
    "UCR-D",
    "V4-control",
    "V7-recovery",
    "Uniform-retention",
    "Shuffled-UCR",
    "IHP-NCER-only",
)

ARM_ALIASES_V7 = {
    "ucr-d": "UCR-D",
    "ucr": "UCR-D",
    "v4-control": "V4-control",
    "v4": "V4-control",
    "v7-recovery": "V7-recovery",
    "recovery": "V7-recovery",
    "rho0": "V7-recovery",
    "uniform-retention": "Uniform-retention",
    "uniform": "Uniform-retention",
    "shuffled-ucr": "Shuffled-UCR",
    "shuffled": "Shuffled-UCR",
    "ihp-ncer-only": "IHP-NCER-only",
    "ihp-ncer": "IHP-NCER-only",
    "ncer": "IHP-NCER-only",
}

CACHE_VERSION_V7 = 70


def canonicalize_arm_name_v7(arm: str) -> str:
    key = str(arm).strip().lower()
    if key in ARM_ALIASES_V7:
        return ARM_ALIASES_V7[key]
    if arm in ARMS_V7:
        return arm
    raise ValueError(f"Unknown v7 arm: '{arm}'. Must be one of {ARMS_V7}")


def binary_train_keys(train_edges: torch.Tensor, n_users: int, n_items: int) -> np.ndarray:
    """Validates IDs and deduplicates all train interaction pairs."""
    if n_users <= 0 or n_items <= 0 or n_users * n_items > np.iinfo(np.int64).max:
        raise ValueError("Invalid entity counts or int64 pair-key overflow.")
    if train_edges.ndim != 2 or train_edges.shape[0] != 2 or train_edges.dtype not in (torch.int32, torch.int64):
        raise ValueError("train_edges must be a [2, E] integer tensor.")
    edges = train_edges.detach().cpu().numpy().astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges[0].max() >= n_users or edges[1].max() >= n_items):
        raise ValueError("Train interaction IDs are outside the declared entity counts.")
    return np.unique(edges[0] * n_items + edges[1])


def _digest_array(hasher: Any, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    hasher.update(str((array.shape, array.dtype.str)).encode())
    hasher.update(memoryview(array).cast("B"))


def tensor_to_scipy(matrix: torch.Tensor) -> sp.csr_matrix:
    """Converts a PyTorch sparse CSR or COO tensor to a SciPy CSR matrix."""
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
    raise ValueError("Expected a sparse COO or CSR tensor.")


def scipy_to_tensor(matrix: sp.csr_matrix) -> torch.Tensor:
    """Converts a SciPy CSR matrix to a PyTorch sparse CSR tensor."""
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
    )


def compute_evidence_score(c: float, n_i: float, n_j: float, t: float = 5.0) -> float:
    if not math.isfinite(t) or t <= 0:
        raise ValueError("Evidence shrinkage must be finite and positive.")
    if n_i <= 0 or n_j <= 0 or c <= 0:
        return 0.0
    return float(c / (c + t) * c / math.sqrt(n_i * n_j))


def _deterministic_topk(ids: np.ndarray, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions top-k candidates deterministically with lexicographical tie-breaking."""
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
) -> sp.csr_matrix:
    """Exact binary R.T @ R in bounded CSR blocks for direct CF candidate support."""
    if k_cf < 0 or int(k_cf) != k_cf or c_min < 1 or int(c_min) != c_min:
        raise ValueError("Require integer k_cf >= 0 and c_min >= 1.")
    if block_size < 1 or int(block_size) != block_size or not math.isfinite(memory_budget_mib) or memory_budget_mib <= 0:
        raise ValueError("Require a positive integer block size and finite memory budget.")
    compute_evidence_score(0, 0, 0, t_shrinkage)
    flat = binary_train_keys(train_edges, n_users, n_items)
    if k_cf == 0 or not len(flat):
        return sp.csr_matrix((n_items, n_items), dtype=np.float32)
    R = sp.csr_matrix(
        (np.ones(len(flat), dtype=np.int64), (flat // n_items, flat % n_items)),
        shape=(n_users, n_items),
    )
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
    directed = sp.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items),
    )
    result = directed.maximum(directed.T).tocsr()
    result.sort_indices()
    return result


def normalize_cf_graph_with_fallback(W_cf: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalization of direct CF graph with isolated diagonal fallback."""
    if W_cf.shape[0] != W_cf.shape[1] or not np.isfinite(W_cf.data).all() or (W_cf.data < 0).any():
        raise ValueError("CF graph must be square, finite and nonnegative.")
    diff = W_cf - W_cf.T
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-7:
        raise ValueError("CF graph must be symmetric before normalization.")
    degrees = np.asarray(W_cf.sum(1)).ravel().astype(np.float64)
    inv = np.zeros(len(degrees), dtype=np.float64)
    inv[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    scale = sp.diags(inv)
    normalized = (scale @ W_cf @ scale).astype(np.float32)
    return (normalized + sp.diags((degrees == 0).astype(np.float32))).tocsr()


def compute_ihp_ncer_scores(
    W_cf: sp.csr_matrix,
    rho: float = 0.01,
    theta_max: float = 0.25,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """Intermediate-Hub-Penalized Structural CF Retention (§3.7 of STAIR5_v7_Report.md).

    Evaluates bounded Gram cosine s_{ij} = K_{ij} / sqrt(K_{ii} * K_{jj}) where
    K = X X^T with X_{ik} = W_{CF, ik} / sqrt(d_k^{CF}).
    Applies dose calibration and mass return to diagonal:
        W_{CF, R} = W_{CF} - A_{CF} + diag(h_{CF})
    """
    n_items = W_cf.shape[0]
    degrees = np.asarray(W_cf.sum(1)).ravel().astype(np.float64)
    inv_sqrt_d = np.zeros(n_items, dtype=np.float64)
    inv_sqrt_d[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])

    # X = W_cf @ diags(inv_sqrt_d)
    X = (W_cf @ sp.diags(inv_sqrt_d)).tocsr()
    # Row energy K_ii = ||X_i||_2^2
    K_diag = np.asarray((X.multiply(X)).sum(axis=1)).ravel().astype(np.float64)

    # For each candidate edge (i, j) in W_cf with i < j
    W_coo = sp.triu(W_cf, k=1).tocoo()
    rows = W_coo.row
    cols = W_coo.col
    weights = W_coo.data.astype(np.float64)

    if len(weights) == 0:
        return W_cf.copy(), {"H_cf": 0.0, "Z_cf": 0.0, "rho_eff": 0.0}

    # Vectorized sparse dot product on candidate edges
    # K_ij = sum_k X_ik * X_jk
    K_ij = np.asarray((X[rows].multiply(X[cols])).sum(axis=1)).ravel()
    denom = np.sqrt(K_diag[rows] * K_diag[cols])
    s_ij = np.divide(K_ij, denom, out=np.zeros_like(K_ij), where=denom > 0)
    s_ij = np.clip(s_ij, 0.0, 1.0)
    q_ij = 1.0 - s_ij

    H = np.sum(weights)  # unordered sum
    Z = np.sum(weights * q_ij) / H if H > 0 else 0.0
    theta = min(theta_max, rho / Z) if Z > 0 else 0.0
    a_ij = theta * q_ij
    A_ij = weights * a_ij
    rho_eff = float(theta * Z)

    # Build symmetric attenuated graph and diagonal return
    A_sym = sp.csr_matrix((np.concatenate([A_ij, A_ij]), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n_items, n_items))
    h = np.asarray(A_sym.sum(axis=1)).ravel()
    W_retained = (W_cf - A_sym + sp.diags(h)).tocsr()
    W_retained.eliminate_zeros()
    W_retained.sort_indices()

    metadata = {
        "H_cf": float(2.0 * H),
        "Z_cf": float(Z),
        "theta_cf": float(theta),
        "rho_eff_cf": rho_eff,
        "mean_s_struct": float(np.mean(s_ij)),
    }
    return W_retained, metadata


class STAIR5V7GraphAdapter:
    """Adapter managing fixed CSR topology, in-place value mutation and UCR-D algebra.

    Topology:
        Maintains fixed indices (crow_indices, col_indices) containing all edges of S_4
        plus explicit entries on diagonals where d_{0, i} > 0.
        Maintains immutable base_values and mutable active_values buffers.
    """

    def __init__(
        self,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        base_values: torch.Tensor,
        n_items: int,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        w0_pairs: torch.Tensor,
        w_norm_pairs: torch.Tensor,
        slot_fwd: torch.Tensor,
        slot_bwd: torch.Tensor,
        diag_nodes: torch.Tensor,
        diag_slots: torch.Tensor,
        diag_coeff: torch.Tensor,
        d0: torch.Tensor,
        arm: str = "UCR-D",
        rho: float = 0.01,
        theta_max: float = 0.25,
        zeta: float = 0.01,
        pair_chunk_size: int = 4096,
        warmup_epochs: float = 10.0,
        shuffled_seed: int = 1,
    ):
        self.n_items = int(n_items)
        self.arm = canonicalize_arm_name_v7(arm)
        self.rho = float(rho)
        self.theta_max = float(theta_max)
        self.zeta = float(zeta)
        self.pair_chunk_size = int(pair_chunk_size)
        self.warmup_epochs = float(warmup_epochs)
        self.shuffled_seed = int(shuffled_seed)

        self.crow_indices = crow_indices
        self.col_indices = col_indices
        self.base_values = base_values.float()
        self.active_values = self.base_values.clone()

        self.base_operator = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, self.base_values, size=(self.n_items, self.n_items)
        )
        self.active_operator = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, self.active_values, size=(self.n_items, self.n_items)
        )

        self.pair_i = pair_i.long()
        self.pair_j = pair_j.long()
        self.w0_pairs = w0_pairs.float()
        self.w_norm_pairs = w_norm_pairs.float()
        self.slot_fwd = slot_fwd.long()
        self.slot_bwd = slot_bwd.long()
        self.num_pairs = len(self.pair_i)
        self.H_unord = float(self.w0_pairs.sum().item()) if self.num_pairs > 0 else 0.0

        self.diag_nodes = diag_nodes.long()
        self.diag_slots = diag_slots.long()
        self.diag_coeff = diag_coeff.float()
        self.d0 = d0.float()

        # Preallocated GPU buffers
        self.q_buf = torch.zeros(self.num_pairs, dtype=torch.float32)
        self.h_buf = torch.zeros(self.n_items, dtype=torch.float32)

        # Fast path detection: zero dose or exact v4 fallback
        self.is_fast_path = (self.arm in ("V4-control", "V7-recovery")) or (self.rho <= 0.0)

        # Degree strata for Shuffled-UCR placebo
        if self.num_pairs > 0:
            pair_deg = torch.sqrt(self.d0[self.pair_i] * self.d0[self.pair_j]).clamp(min=1.0)
            self.pair_strata = torch.floor(torch.log2(pair_deg)).long()
        else:
            self.pair_strata = torch.zeros(0, dtype=torch.long)

        self._step_diagnostics: Optional[Dict[str, Any]] = None
        self._current_snapshot: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "STAIR5V7GraphAdapter":
        """Transfers all internal CSR arrays and index maps to the target device."""
        dev = torch.device(device)
        self.crow_indices = self.crow_indices.to(dev)
        self.col_indices = self.col_indices.to(dev)
        self.base_values = self.base_values.to(dev)
        self.active_values = self.active_values.to(dev)

        self.base_operator = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, self.base_values, size=(self.n_items, self.n_items), device=dev
        )
        self.active_operator = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, self.active_values, size=(self.n_items, self.n_items), device=dev
        )

        self.pair_i = self.pair_i.to(dev)
        self.pair_j = self.pair_j.to(dev)
        self.w0_pairs = self.w0_pairs.to(dev)
        self.w_norm_pairs = self.w_norm_pairs.to(dev)
        self.slot_fwd = self.slot_fwd.to(dev)
        self.slot_bwd = self.slot_bwd.to(dev)
        self.diag_nodes = self.diag_nodes.to(dev)
        self.diag_slots = self.diag_slots.to(dev)
        self.diag_coeff = self.diag_coeff.to(dev)
        self.d0 = self.d0.to(dev)
        self.q_buf = self.q_buf.to(dev)
        self.h_buf = self.h_buf.to(dev)
        if len(self.pair_strata):
            self.pair_strata = self.pair_strata.to(dev)
        return self

    @torch.no_grad()
    def score_pairs_chunked(
        self,
        D: torch.Tensor,
        r: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates pair antagonism q = max(0, -cos) in scratchpad chunks (C=4096)."""
        K = self.num_pairs
        C = self.pair_chunk_size
        q_buf = self.q_buf

        for start in range(0, K, C):
            end = min(start + C, K)
            pi = self.pair_i[start:end]
            pj = self.pair_j[start:end]

            mask = eligible[pi] & eligible[pj]
            if not mask.any():
                q_buf[start:end].zero_()
                continue

            # Dot products in chunk scratchpad
            dots = (D[pi] * D[pj]).sum(dim=-1)
            denom = (r[pi] * r[pj]).clamp(min=1e-12)
            cos = torch.clamp(dots / denom, -1.0, 1.0)
            q_chunk = torch.where(mask, torch.clamp(-cos, min=0.0), torch.zeros_like(cos))
            q_buf[start:end].copy_(q_chunk)

        return q_buf

    @torch.no_grad()
    def update_step_snapshot(
        self,
        D: torch.Tensor,
        epoch_progress: float = 10.0,
    ) -> torch.Tensor:
        """Computes dynamic UCR-D attenuation and updates mutable CSR values in place.

        Args:
            D: Detached Adam direction tensor of shape (N, d).
            epoch_progress: Completed epochs + batch fraction.

        Returns:
            active_operator: Sparse CSR operator S_7^{(t)} for the current BSC step.
        """
        if not torch.isfinite(D).all():
            raise ValueError("[STAIR5-v7] Nonfinite Adam direction encountered; numerical failure.")

        # Exact Fast-path short-circuit
        if self.is_fast_path or self.H_unord == 0.0 or self.num_pairs == 0:
            self._step_diagnostics = {
                "fast_path": True,
                "arm": self.arm,
                "Z_t": 0.0,
                "theta_t": 0.0,
                "rho_eff": 0.0,
            }
            return self.base_operator

        # Target dose ramp (linear warmup over 10 epochs)
        ramp = min(1.0, max(0.0, float(epoch_progress) / max(1e-6, self.warmup_epochs)))
        rho_t = self.rho * ramp
        if rho_t <= 0.0:
            self._step_diagnostics = {
                "fast_path": True,
                "arm": self.arm,
                "Z_t": 0.0,
                "theta_t": 0.0,
                "rho_eff": 0.0,
            }
            return self.base_operator

        # Row norms and magnitude eligibility threshold (§3.3)
        r = torch.norm(D, dim=-1)
        r_pos = r[r > 0]
        if len(r_pos) == 0:
            return self.base_operator

        tau_mag = max(1e-12, float(self.zeta * torch.median(r_pos).item()))
        eligible = (r >= tau_mag)

        # Arm routing
        if self.arm == "Uniform-retention":
            # Constant attenuation independent of direction: a_k = min(theta_max, rho_t)
            theta_const = min(self.theta_max, rho_t)
            a = torch.full((self.num_pairs,), theta_const, dtype=torch.float32, device=D.device)
            Z_t = 1.0
            theta_t = theta_const
            rho_eff = theta_const
            A = self.w0_pairs * a
        else:
            # Pair scoring
            q = self.score_pairs_chunked(D, r, eligible)

            if self.arm == "Shuffled-UCR":
                # Permute q within degree strata to break location-specific conflict
                q_perm = q.clone()
                rng = torch.Generator(device=D.device).manual_seed(self.shuffled_seed)
                unique_strata = torch.unique(self.pair_strata)
                for s in unique_strata:
                    idx = torch.nonzero(self.pair_strata == s, as_tuple=False).squeeze(-1)
                    if len(idx) > 1:
                        perm = torch.randperm(len(idx), generator=rng, device=D.device)
                        q_perm[idx] = q[idx[perm]]
                q = q_perm

            # Conflict load Z_t (§3.4)
            weighted_conflict = (self.w0_pairs * q).sum().item()
            Z_t = weighted_conflict / self.H_unord if self.H_unord > 0 else 0.0

            if Z_t <= 0.0:
                self._step_diagnostics = {
                    "fast_path": False,
                    "arm": self.arm,
                    "Z_t": 0.0,
                    "theta_t": 0.0,
                    "rho_eff": 0.0,
                    "eligible_edges": int(eligible[self.pair_i].sum().item()),
                }
                return self.base_operator

            # Dose calibration (§3.4)
            theta_t = min(self.theta_max, rho_t / Z_t)
            rho_eff = theta_t * Z_t
            a = theta_t * q
            A = self.w0_pairs * a

        # Off-diagonal reduction: delta_S = w_norm_pairs * a
        delta_S = self.w_norm_pairs * a

        # Diagonal mass return (§3.3 & §3.4): h_i = sum_j A_ij
        self.h_buf.zero_()
        self.h_buf.scatter_add_(0, self.pair_i, A)
        self.h_buf.scatter_add_(0, self.pair_j, A)
        delta_S_diag = self.h_buf[self.diag_nodes] * self.diag_coeff

        # In-place mutable CSR values update
        self.active_values.copy_(self.base_values)
        self.active_values.scatter_add_(0, self.slot_fwd, -delta_S)
        self.active_values.scatter_add_(0, self.slot_bwd, -delta_S)
        self.active_values.scatter_add_(0, self.diag_slots, delta_S_diag)

        self._step_diagnostics = {
            "fast_path": False,
            "arm": self.arm,
            "tau_mag": tau_mag,
            "rho_t": rho_t,
            "Z_t": float(Z_t),
            "theta_t": float(theta_t),
            "rho_eff": float(rho_eff),
            "cap_hit": bool(theta_t >= self.theta_max - 1e-7),
            "moved_mass": float(A.sum().item()),
        }
        self._current_snapshot = self.active_operator
        return self.active_operator

    def clear_step_snapshot(self) -> None:
        """Clears transient step references to prevent any GPU memory leakage."""
        self._current_snapshot = None
        self._step_diagnostics = None

    def get_last_step_diagnostics(self) -> Optional[Dict[str, Any]]:
        return self._step_diagnostics


@dataclass
class GraphStateV7:
    adapter: STAIR5V7GraphAdapter
    metadata: Dict[str, Any]


def build_calibrated_graph_v7(
    raw_semantic: torch.Tensor,
    baseline_normalized: torch.Tensor,
    train_edges: torch.Tensor,
    n_users: int,
    n_items: int,
    arm: str = "UCR-D",
    rho: float = 0.01,
    theta_max: float = 0.25,
    zeta: float = 0.01,
    pair_chunk_size: int = 4096,
    warmup_epochs: float = 10.0,
    eta: float = 0.1,
    k_cf: int = 5,
    c_min: int = 2,
    t_shrinkage: float = 5.0,
    shuffled_seed: int = 1,
    cache_dir: Optional[str] = None,
    block_size: int = 64,
    memory_budget_mib: float = 128.0,
) -> GraphStateV7:
    """Constructs fixed CSR topology, index mappings and STAIR5V7GraphAdapter."""
    arm = canonicalize_arm_name_v7(arm)

    # 1. Extract raw semantic W_0 (zero diagonal, symmetric)
    W0_csr = tensor_to_scipy(raw_semantic)
    W0_coo = sp.triu(W0_csr, k=1).tocoo()
    pair_i_np = W0_coo.row.astype(np.int64)
    pair_j_np = W0_coo.col.astype(np.int64)
    w0_pairs_np = W0_coo.data.astype(np.float32)

    # Degree vector d_0
    d0_np = np.asarray(W0_csr.sum(axis=1)).ravel().astype(np.float32)

    # 2. Build or load Direct CF candidate support graph
    cache_path = None
    W_cf = None
    if cache_dir and k_cf > 0:
        pairs = binary_train_keys(train_edges, n_users, n_items)
        pair_hash = hashlib.sha256()
        _digest_array(pair_hash, pairs)
        signature = {
            "cache_version": CACHE_VERSION_V7,
            "train": pair_hash.hexdigest(),
            "users": n_users,
            "items": n_items,
            "k": k_cf,
            "cmin": c_min,
            "t": t_shrinkage,
        }
        cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        cache_path = Path(cache_dir) / f"cf_v7_{cache_key}.pt"
        if cache_path.is_file():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                cand = tensor_to_scipy(payload["operator"])
                if payload.get("signature") == signature and cand.shape == (n_items, n_items):
                    W_cf = cand
            except Exception as e:
                warnings.warn(f"Ignoring corrupt CF cache: {e}")

    if W_cf is None:
        W_cf = build_candidate_support_graph(
            train_edges, n_items, n_users, k_cf, c_min, t_shrinkage, block_size, memory_budget_mib
        )
        if cache_path:
            atomic_torch_save(
                {
                    "operator": scipy_to_tensor(W_cf),
                    "signature": signature,
                    "fingerprint": sparse_fingerprint(W_cf),
                },
                cache_path,
            )

    # 3. Handle IHP-NCER-only structural alternative arm (§3.7)
    ihp_meta = {}
    if arm == "IHP-NCER-only":
        W_cf_retained, ihp_meta = compute_ihp_ncer_scores(W_cf, rho=rho, theta_max=theta_max)
        S_cf = normalize_cf_graph_with_fallback(W_cf_retained)
    else:
        S_cf = normalize_cf_graph_with_fallback(W_cf)

    # S_0 normalized semantic graph
    S0_csr = tensor_to_scipy(baseline_normalized)

    # Combined baseline operator S_4 = 0.9 * S_0 + 0.1 * S_CF
    S4_csr = (1.0 - eta) * S0_csr + eta * S_cf
    S4_csr.sort_indices()

    # 4. Construct Pre-allocated Unified CSR Topology (§4.1)
    # Ensure explicit diagonal slots (i, i) exist for all nodes where d_{0, i} > 0
    diag_needed = np.flatnonzero(d0_np > 0).astype(np.int64)
    missing_diags = [
        i for i in diag_needed if i not in S4_csr.indices[S4_csr.indptr[i]:S4_csr.indptr[i + 1]]
    ]
    if missing_diags:
        missing_rows = np.array(missing_diags, dtype=np.int64)
        missing_cols = missing_rows.copy()
        missing_vals = np.zeros(len(missing_diags), dtype=np.float32)
        coo = S4_csr.tocoo()
        comb_rows = np.concatenate([coo.row, missing_rows])
        comb_cols = np.concatenate([coo.col, missing_cols])
        comb_data = np.concatenate([coo.data, missing_vals])
        S4_ext = sp.csr_matrix((comb_data, (comb_rows, comb_cols)), shape=(n_items, n_items))
        S4_ext.sum_duplicates()
        S4_ext.sort_indices()
    else:
        S4_ext = S4_csr.copy()
        S4_ext.sort_indices()

    # 5. Map forward/backward off-diagonal pairs and diagonal slots
    slot_fwd = np.empty(len(pair_i_np), dtype=np.int64)
    slot_bwd = np.empty(len(pair_i_np), dtype=np.int64)
    indptr = S4_ext.indptr
    indices = S4_ext.indices

    for k in range(len(pair_i_np)):
        u, v = pair_i_np[k], pair_j_np[k]
        u_slice = indices[indptr[u]:indptr[u + 1]]
        slot_fwd[k] = indptr[u] + np.searchsorted(u_slice, v)
        v_slice = indices[indptr[v]:indptr[v + 1]]
        slot_bwd[k] = indptr[v] + np.searchsorted(v_slice, u)

    diag_slots = np.empty(len(diag_needed), dtype=np.int64)
    for idx, node in enumerate(diag_needed):
        row_slice = indices[indptr[node]:indptr[node + 1]]
        diag_slots[idx] = indptr[node] + np.searchsorted(row_slice, node)

    # Precomputed weight factors
    w_norm_pairs_np = (1.0 - eta) * w0_pairs_np / np.sqrt(d0_np[pair_i_np] * d0_np[pair_j_np])
    diag_coeff_np = (1.0 - eta) / d0_np[diag_needed]

    adapter = STAIR5V7GraphAdapter(
        crow_indices=torch.from_numpy(S4_ext.indptr.astype(np.int64)),
        col_indices=torch.from_numpy(S4_ext.indices.astype(np.int64)),
        base_values=torch.from_numpy(S4_ext.data.astype(np.float32)),
        n_items=n_items,
        pair_i=torch.from_numpy(pair_i_np),
        pair_j=torch.from_numpy(pair_j_np),
        w0_pairs=torch.from_numpy(w0_pairs_np),
        w_norm_pairs=torch.from_numpy(w_norm_pairs_np.astype(np.float32)),
        slot_fwd=torch.from_numpy(slot_fwd),
        slot_bwd=torch.from_numpy(slot_bwd),
        diag_nodes=torch.from_numpy(diag_needed),
        diag_slots=torch.from_numpy(diag_slots),
        diag_coeff=torch.from_numpy(diag_coeff_np.astype(np.float32)),
        d0=torch.from_numpy(d0_np),
        arm=arm,
        rho=rho,
        theta_max=theta_max,
        zeta=zeta,
        pair_chunk_size=pair_chunk_size,
        warmup_epochs=warmup_epochs,
        shuffled_seed=shuffled_seed,
    )

    metadata = {
        "version": 7,
        "arm": arm,
        "rho": float(rho),
        "theta_max": float(theta_max),
        "zeta": float(zeta),
        "eta": float(eta),
        "n_items": int(n_items),
        "num_unordered_pairs": len(pair_i_np),
        "total_offdiag_mass": float(2.0 * adapter.H_unord),
        "csr_nnz": int(S4_ext.nnz),
        "ihp_meta": ihp_meta,
    }

    return GraphStateV7(adapter=adapter, metadata=metadata)
