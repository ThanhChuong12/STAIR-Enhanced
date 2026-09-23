"""STAIR4-v2.1 BCCR Contrastive Heads (§6.2, §6.3, §6.4, §6.5 of STAIR4_v2_1_Report.md).

Architecture & Mathematical Contracts:
---------------------------------------
1. Bounded Phase Encoder:
   r(x) = x / max(||x||_2, eps)
   ψ_k(x) = (1 / sqrt(d)) * [cos(kappa * sqrt(d) * r_k(x)) + i * sin(kappa * sqrt(d) * r_k(x))]
   Guarantees unit row norm ||ψ(x)||_2 = 1.0; no arctan/tanh gradient saturation.

2. Online Stop-Gradient & Key-only Rotation:
   Query: q = ψ(X0)
   Target (pre-rotation): t = ψ(X1).detach()   <-- STOP-GRADIENT eliminates gradient conflict
   Key: k = G_θ(t)                              <-- G_θ receives gradients for θ
   G_θ uses θ_k = (π/4) * tanh(ω_k) with ω_k initialized to 0 (identity at step 0).

3. Hybrid Complex Similarity Kernel:
   h(q, k) = q^H k
   s_η(q, k) = (1 - η) * Re(h) + η * |h|^2
   Logits = s_η / T
   Computed via 2-D GEMM operations without 3-D tensor materialization.

4. Train-only Multi-Positive Supervised CE:
   Uniform target distribution over train-positive candidate pairs.
   Filters rows to valid subset V_u having ≥ 1 positive, ≥ 1 non-positive,
   and valid (non-near-zero) views.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.stair4_v2_utils import (
    CandidateBatch,
    TrainPositiveIndex,
    bounded_phase_encoder,
    bounded_real_encoder,
    build_positive_mask,
    check_view_validity,
    complex_hybrid_similarity,
)


# ---------------------------------------------------------------------------
# Diagnostics NamedTuple
# ---------------------------------------------------------------------------

class BCCRDiagnostics(NamedTuple):
    """Detached per-step diagnostics from CrossLayerContrastiveHead."""
    cl_raw: float              # raw loss before lambda weighting
    pos_count_ui: float        # mean positive count per query row (u→i direction)
    pos_count_iu: float        # mean positive count per query row (i→u direction)
    sim_q25: float             # 25th percentile of similarity scores
    sim_q75: float             # 75th percentile of similarity scores
    softmax_entropy_ui: float
    softmax_entropy_iu: float
    theta_grad_norm: float     # gradient norm of Givens omega (0 if identity)
    theta_norm: float          # current L2 norm of theta
    near_zero_view_rate: float # fraction of query/key views with norm < eps


# Alias for backward compatibility
POCLDiagnostics = BCCRDiagnostics


# ---------------------------------------------------------------------------
# Rotation Modules (§6.3)
# ---------------------------------------------------------------------------

class IdentityRotation(nn.Module):
    """Zero-parameter identity rotation (default pilot mode)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x_real: Tensor,
        x_imag: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if x_imag is None:
            return x_real
        return x_real, x_imag


class PairwiseGivens(nn.Module):
    """Block-diagonal Givens rotation on key branch.

    d/2 independent 2×2 blocks.
    Parameter ω ∈ ℝ^{d/2}, initialized to zero.
    Angle θ_k = (π/4) * tanh(ω_k) ∈ (-π/4, π/4).
    At init, ω = 0 → θ = 0 → exact identity rotation G = I.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and ≥ 2, got {d}")
        self.d = d
        # Trainable omega parameter, initialized to 0
        self.omega = nn.Parameter(torch.zeros(d // 2))

    @property
    def theta(self) -> Tensor:
        return (math.pi / 4.0) * torch.tanh(self.omega)

    def forward(
        self,
        x_real: Tensor,
        x_imag: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        B, d = x_real.shape
        if d != self.d:
            raise ValueError(f"Expected d={self.d} columns, got {d}")

        theta = self.theta  # [d/2]
        c = torch.cos(theta)  # [d/2]
        s = torch.sin(theta)  # [d/2]

        # Reshape to [B, d/2, 2]
        r_blocks = x_real.reshape(B, self.d // 2, 2)
        r0, r1 = r_blocks[..., 0], r_blocks[..., 1]
        r0_new = c * r0 - s * r1
        r1_new = s * r0 + c * r1
        r_out = torch.stack([r0_new, r1_new], dim=-1).reshape(B, self.d)

        if x_imag is None:
            return r_out

        i_blocks = x_imag.reshape(B, self.d // 2, 2)
        i0, i1 = i_blocks[..., 0], i_blocks[..., 1]
        i0_new = c * i0 - s * i1
        i1_new = s * i0 + c * i1
        i_out = torch.stack([i0_new, i1_new], dim=-1).reshape(B, self.d)
        return r_out, i_out


# ---------------------------------------------------------------------------
# Multi-positive Cross-Entropy Loss (§6.5)
# ---------------------------------------------------------------------------

def multi_positive_ce(
    logits: Tensor,
    mask: Tensor,
    temperature: float = 0.2,
    valid_query_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Supervised in-batch CE with uniform target over train positives.

    ``logits`` must already be temperature-scaled.  ``temperature`` remains
    a validated compatibility argument and is not applied a second time.

    L = -(1 / |V_u|) sum_{u in V_u} sum_i Y_ui * log_softmax(logits)_i
    where Y_ui = M_ui / sum_j M_uj.

    Valid rows V_u must have:
      - at least 1 positive: mask.sum(dim=-1) >= 1
      - at least 1 non-positive: (~mask).sum(dim=-1) >= 1
      - query view is valid (not near-zero) if valid_query_mask provided.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D Tensor")
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a bool Tensor with the same shape as logits")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    # Row filtering for valid supervision (§6.5)
    row_pos = mask.sum(dim=1)
    row_neg = (~mask).sum(dim=1)
    valid_rows = (row_pos >= 1) & (row_neg >= 1)
    if valid_query_mask is not None:
        valid_rows = valid_rows & valid_query_mask

    n_valid = int(valid_rows.sum().item())
    if n_valid == 0:
        # Fallback to empty loss with zero gradient
        zero_loss = 0.0 * logits.sum()
        return zero_loss, {
            "pos_count_mean": float(row_pos.float().mean().item()) if row_pos.numel() > 0 else 0.0,
            "valid_rows": 0.0,
            "softmax_entropy": 0.0,
        }

    v_logits = logits[valid_rows]
    v_mask = mask[valid_rows]
    v_pos = row_pos[valid_rows].float()

    log_probs = F.log_softmax(v_logits, dim=1)
    target = v_mask.float() / v_pos[:, None]
    loss = -(target * log_probs).sum(dim=1).mean()

    with torch.no_grad():
        probs = F.softmax(v_logits, dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean().item()
        diag = {
            "pos_count_mean": float(v_pos.mean().item()),
            "valid_rows": float(n_valid),
            "softmax_entropy": entropy,
        }
    return loss, diag


# ---------------------------------------------------------------------------
# CrossLayerContrastiveHead
# ---------------------------------------------------------------------------

class CrossLayerContrastiveHead(nn.Module):
    """STAIR4-v2.1 BCCR contrastive learning head.

    Supports:
      - kernel_mode: 'hybrid' (default), 'signed', 'fidelity', 'cosine', 'none'
      - rotation_mode: 'identity' (default), 'learned_givens'
      - Stop-gradient on key (target) branch before rotation
      - Near-zero view detection & exclusion (§6.2, §6.5)
    """

    def __init__(
        self,
        d: int,
        kernel: str = "hybrid",
        rotation: str = "identity",
        eta: float = 0.25,
        kappa: float = 0.5,
        temperature: float = 0.2,
        query_chunk_size: int = 256,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if kernel not in ("hybrid", "signed", "fidelity", "cosine", "none"):
            raise ValueError(f"Unsupported kernel: {kernel!r}")
        if rotation not in ("identity", "learned_givens"):
            raise ValueError(f"Unsupported rotation: {rotation!r}")
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and >= 2, got {d}")
        if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be finite and in [0, 1]")
        if not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.d = d
        self.kernel_mode = kernel
        self.rotation_mode = rotation
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.temperature = float(temperature)
        self.query_chunk_size = int(query_chunk_size)
        self.eps = float(eps)

        if rotation == "learned_givens":
            self.rotation = PairwiseGivens(d)
        else:
            self.rotation = IdentityRotation()

    @property
    def has_trainable_parameters(self) -> bool:
        return (
            self.rotation_mode == "learned_givens"
            and isinstance(self.rotation, PairwiseGivens)
        )

    def _compute_logits(
        self,
        q_raw: Tensor,  # [Bq, d]
        t_raw: Tensor,  # [Bk, d]
    ) -> Tensor:
        """Compute [Bq, Bk] logits with stop-gradient on target before rotation."""
        if self.kernel_mode in ("hybrid", "signed", "fidelity"):
            # Determine effective eta
            if self.kernel_mode == "signed":
                eff_eta = 0.0
            elif self.kernel_mode == "fidelity":
                eff_eta = 1.0
            else:
                eff_eta = self.eta

            # Query: bounded complex phase feature
            q_real, q_imag = bounded_phase_encoder(q_raw, kappa=self.kappa, eps=self.eps)

            # Stop-gradient at the target source before encoding.
            t_real_sg, t_imag_sg = bounded_phase_encoder(
                t_raw.detach(), kappa=self.kappa, eps=self.eps
            )

            # Key: Givens rotation applied to detached target (omega receives grad)
            k_real, k_imag = self.rotation(t_real_sg, t_imag_sg)

            # Complex hybrid similarity GEMM
            return complex_hybrid_similarity(
                q_real, q_imag, k_real, k_imag,
                eta=eff_eta, temperature=self.temperature
            )

        elif self.kernel_mode == "cosine":
            # Real-space cosine control
            q_norm = bounded_real_encoder(q_raw, eps=self.eps)
            t_norm = bounded_real_encoder(t_raw.detach(), eps=self.eps)  # stop-gradient
            k_norm = self.rotation(t_norm)
            k_norm = bounded_real_encoder(k_norm, eps=self.eps)
            sim = torch.mm(q_norm, k_norm.T)
            return sim / self.temperature

        else:
            raise RuntimeError(f"Unknown kernel_mode: {self.kernel_mode!r}")

    def _directional_loss(
        self,
        queries: Tensor,
        targets: Tensor,
        query_ids: Tensor,
        key_ids: Tensor,
        train_index: TrainPositiveIndex,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute one directional multi-positive loss in query chunks (§6.5).

        Excludes near-zero views (norm < eps) per §6.2.
        """
        total_loss = queries.new_zeros(())
        valid_rows = 0
        positive_total = 0.0
        entropy_total = 0.0
        score_count = 0
        q25_total = 0.0
        q75_total = 0.0

        # Check key view validity
        key_valid = check_view_validity(targets, eps=self.eps)
        near_zero_keys = int((~key_valid).sum().item())
        near_zero_queries = 0

        for start in range(0, queries.shape[0], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, queries.shape[0])
            chunk_queries = queries[start:stop]
            chunk_query_valid = check_view_validity(chunk_queries, eps=self.eps)
            near_zero_queries += int((~chunk_query_valid).sum().item())

            logits = self._compute_logits(chunk_queries, targets)
            mask = build_positive_mask(
                query_ids[start:stop], key_ids, train_index
            ).to(device=queries.device, non_blocking=True)

            # Mask out near-zero keys if any
            if not key_valid.all():
                mask = mask & key_valid[None, :]
                logits = logits.masked_fill(~key_valid[None, :], -1e9)

            chunk_loss, chunk_diag = multi_positive_ce(
                logits, mask, valid_query_mask=chunk_query_valid
            )
            chunk_valid = int(chunk_diag["valid_rows"])
            if chunk_valid:
                total_loss = total_loss + chunk_loss * chunk_valid
                valid_rows += chunk_valid
                positive_total += chunk_diag["pos_count_mean"] * chunk_valid
                entropy_total += chunk_diag["softmax_entropy"] * chunk_valid

            with torch.no_grad():
                scores = (logits * self.temperature).detach().reshape(-1)
                if scores.numel():
                    score_count += scores.numel()
                    q25_total += float(torch.quantile(scores, 0.25).item()) * scores.numel()
                    q75_total += float(torch.quantile(scores, 0.75).item()) * scores.numel()

        total_views = max(1, queries.shape[0] + targets.shape[0])
        near_zero_rate = float((near_zero_queries + near_zero_keys) / total_views)

        if valid_rows == 0:
            return total_loss, {
                "pos_count_mean": 0.0,
                "valid_rows": 0.0,
                "softmax_entropy": 0.0,
                "sim_q25": 0.0,
                "sim_q75": 0.0,
                "near_zero_rate": near_zero_rate,
            }
        return total_loss / valid_rows, {
            "pos_count_mean": positive_total / valid_rows,
            "valid_rows": float(valid_rows),
            "softmax_entropy": entropy_total / valid_rows,
            "sim_q25": q25_total / score_count if score_count else 0.0,
            "sim_q75": q75_total / score_count if score_count else 0.0,
            "near_zero_rate": near_zero_rate,
        }

    def forward(
        self,
        X0_all: Tensor,          # [U+I, d] full embedding table at layer 0
        X1_all: Tensor,          # [U+I, d] unscaled first-hop embeddings
        candidates: CandidateBatch,
        train_index: TrainPositiveIndex,
        reverse_train_index: TrainPositiveIndex,
        n_users: int,
    ) -> Tuple[Tensor, BCCRDiagnostics]:
        """Compute BCCR auxiliary loss for a minibatch (§6.5)."""
        if self.kernel_mode == "none":
            raise RuntimeError("CrossLayerContrastiveHead called with kernel_mode='none'")

        u_ids = candidates.user_ids.to(X0_all.device)
        i_ids = candidates.item_ids.to(X0_all.device)

        # Slice embedding tables
        U0 = X0_all[u_ids]
        I0 = X0_all[n_users + i_ids]
        U1 = X1_all[u_ids]
        I1 = X1_all[n_users + i_ids]

        loss_ui, diag_ui = self._directional_loss(
            U0, I1, candidates.user_ids, candidates.item_ids, train_index
        )
        loss_iu, diag_iu = self._directional_loss(
            I0, U1, candidates.item_ids, candidates.user_ids,
            reverse_train_index,
        )

        # Combine losses (§6.5): 1/2 factor only applied if both directions valid
        has_ui = diag_ui["valid_rows"] > 0
        has_iu = diag_iu["valid_rows"] > 0
        if has_ui and has_iu:
            loss = 0.5 * (loss_ui + loss_iu)
        elif has_ui:
            loss = loss_ui
        elif has_iu:
            loss = loss_iu
        else:
            loss = 0.0 * (loss_ui + loss_iu)

        avg_near_zero = 0.5 * (diag_ui.get("near_zero_rate", 0.0) + diag_iu.get("near_zero_rate", 0.0))
        theta_norm = float(self.rotation.theta.detach().norm().item()) if self.has_trainable_parameters else 0.0

        diag = BCCRDiagnostics(
            cl_raw=float(loss.detach().item()),
            pos_count_ui=diag_ui["pos_count_mean"],
            pos_count_iu=diag_iu["pos_count_mean"],
            sim_q25=diag_ui["sim_q25"],
            sim_q75=diag_ui["sim_q75"],
            softmax_entropy_ui=diag_ui["softmax_entropy"],
            softmax_entropy_iu=diag_iu["softmax_entropy"],
            theta_grad_norm=0.0,
            theta_norm=theta_norm,
            near_zero_view_rate=avg_near_zero,
        )
        return loss, diag
