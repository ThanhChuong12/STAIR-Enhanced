# -*- coding: utf-8 -*-
"""
models/stair5_v1_objectives.py — Contrastive Objectives & Self-Return Removal Module
====================================================================================
Senior AI Research Engineer Implementation:
1. Self-Return Removal with Degree-1 Safety (Norm Protection):
   H~(1)_{target|source} = H(1)_{target} - (E_{source} / sqrt(d~_{target} * d~_{source})) * beta
   where d~ = max(d - 1, 1). If ||H~||_2 < eps_floor, clamps to eps_floor along safe direction.
2. Adaptive Spectral Re-weighting:
   H^(1) / max(beta, eps_beta) with optional quantile-based norm clamping M_norm.
3. Fixed Scale Normalization:
   q_{t, l} = max(1e-3, Median_{n in V_t} ||H_n^(l)(0)||_2).
4. Multi-Positive InfoNCE Loss (Bidirectional: U0 -> I1 and I0 -> U1):
   Utilizes train-positive ground truth interaction matrix P in-batch.
   Replaces positive entries with self-return removed representations.
   Computes alignment and uniformity diagnostics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stair5_v1_geometry import LorentzGeometryModule


def remove_self_return(
    H1_target: torch.Tensor,
    E_source: torch.Tensor,
    deg_source: torch.Tensor,
    deg_target: torch.Tensor,
    beta: torch.Tensor,
    eps_floor: float = 1e-2,
) -> torch.Tensor:
    """
    Removes the 1-hop self-return shortcut from the target node representation:
        H~(1)_{target|source} = H(1)_{target} - (E_{source} / sqrt(d~_{target} * d~_{source})) * beta
    where d~ = max(d - 1, 1).

    Degree-1 Safety (Norm Protection):
        If the remaining vector magnitude falls below eps_floor (e.g. for degree-1 leaf items),
        its norm is bounded from below at eps_floor along the safe directional unit vector,
        preventing zero-vector collapse and degenerate geometry.

    Args:
        H1_target: (N, D) target node layer 1 representations.
        E_source: (N, D) source node layer 0 representations.
        deg_source: (N,) or (N, 1) degrees of source nodes.
        deg_target: (N,) or (N, 1) degrees of target nodes.
        beta: (D,) spectral decay factor beta = 1 - beta3.
        eps_floor: minimum L2 norm floor for degree-1 protection.

    Returns:
        H_corrected: (N, D) self-return removed representation.
    """
    # Ensure correct shapes
    if deg_source.dim() == 1:
        deg_source = deg_source.unsqueeze(-1)
    if deg_target.dim() == 1:
        deg_target = deg_target.unsqueeze(-1)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)

    # Effective degree excluding the ego interaction
    d_src_eff = torch.clamp(deg_source.float() - 1.0, min=1.0)
    d_tgt_eff = torch.clamp(deg_target.float() - 1.0, min=1.0)
    norm_factor = torch.sqrt(d_src_eff * d_tgt_eff)

    self_return = (E_source / norm_factor) * beta
    H_corrected = H1_target - self_return

    # Norm Protection (Peer Review Round 2 Requirement)
    corrected_norm = torch.norm(H_corrected, p=2, dim=-1, keepdim=True)
    need_protection = (corrected_norm < eps_floor).float()

    # Direction: if H_corrected is non-zero, use it; otherwise fallback to H1_target
    target_norm = torch.norm(H1_target, p=2, dim=-1, keepdim=True)
    fallback_direction = H1_target / (target_norm + 1e-8)
    safe_direction = torch.where(corrected_norm > 1e-7, H_corrected / (corrected_norm + 1e-8), fallback_direction)
    
    # In case both were 0, fallback to standard basis e_0 = [1, 0, ...]
    dir_norm = torch.norm(safe_direction, p=2, dim=-1, keepdim=True)
    basis_direction = torch.zeros_like(safe_direction)
    basis_direction[..., 0] = 1.0
    safe_direction = torch.where(dir_norm > 1e-7, safe_direction, basis_direction)

    H_corrected = (1.0 - need_protection) * H_corrected + need_protection * (eps_floor * safe_direction)

    return H_corrected


def spectral_reweight(
    H1: torch.Tensor,
    beta: torch.Tensor,
    eps_beta: float = 0.05,
    max_norm: Optional[float] = None,
) -> torch.Tensor:
    """
    Compensates for the high-dimensional spectral attenuation beta_j in layer H^(1):
        H^(1)_reweighted = H^(1) / max(beta_j, eps_beta)

    Optionally clips the L2 norm to max_norm (the 95th quantile of ||H^(0)|| established at epoch 0).

    Args:
        H1: (..., D) layer 1 representations.
        beta: (D,) spectral propagation vector.
        eps_beta: numerical lower bound for beta_j (default: 0.05).
        max_norm: optional adaptive norm threshold M_norm.

    Returns:
        H1_reweighted: (..., D) restored representation.
    """
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)
    beta_safe = torch.clamp(beta, min=eps_beta)

    H1_reweighted = H1 / beta_safe

    if max_norm is not None and max_norm > 0:
        norm = torch.norm(H1_reweighted, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(float(max_norm) / (norm + 1e-8), max=1.0)
        H1_reweighted = H1_reweighted * scale

    return H1_reweighted


class MultiPositiveInfoNCELoss(nn.Module):
    """
    Bidirectional Multi-Positive InfoNCE Loss Module with Lorentz/Euclidean Geodesic Scoring.

    Direction 1: U_0 -> I_1 (User layer 0 queries Item layer 1 with Self-Return Removal on positive keys).
    Direction 2: I_0 -> U_1 (Item layer 0 queries User layer 1 with Self-Return Removal on positive keys).

    Args:
        geo (LorentzGeometryModule): Lorentz/Euclidean geometry module.
        radius_cap (float): Maximum radius R for scaling (default: 2.0).
        tau (float): Softmax temperature (default: 0.3).
        eps_floor (float): Floor for degree-1 norm protection (default: 1e-2).
        eps_beta (float): Clamp floor for spectral re-weighting (default: 0.05).
        disable_self_return (bool): Ablation flag to bypass self-return removal (H0-noself).
        disable_reweight (bool): Ablation flag to bypass spectral re-weighting (H0-reweight).
    """

    def __init__(
        self,
        geo: LorentzGeometryModule,
        radius_cap: float = 2.0,
        tau: float = 0.3,
        eps_floor: float = 1e-2,
        eps_beta: float = 0.05,
        disable_self_return: bool = False,
        disable_reweight: bool = False,
    ) -> None:
        super().__init__()
        self.geo = geo
        self.radius_cap = float(radius_cap)
        self.tau = float(tau)
        self.eps_floor = float(eps_floor)
        self.eps_beta = float(eps_beta)
        self.disable_self_return = bool(disable_self_return)
        self.disable_reweight = bool(disable_reweight)

    def forward(
        self,
        u_0: torch.Tensor,
        i_0: torch.Tensor,
        u_1: torch.Tensor,
        i_1: torch.Tensor,
        P: torch.Tensor,
        deg_u: torch.Tensor,
        deg_i: torch.Tensor,
        beta: torch.Tensor,
        q_u0: float,
        q_i0: float,
        q_u1: float,
        q_i1: float,
        M_norm: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the bidirectional multi-positive InfoNCE loss.

        Args:
            u_0: (B_u, D) user representations at layer 0 in current batch.
            i_0: (B_i, D) item representations at layer 0 in current batch.
            u_1: (B_u, D) user representations at layer 1 in current batch.
            i_1: (B_i, D) item representations at layer 1 in current batch.
            P: (B_u, B_i) binary mask: P[u, i] = 1 if (u, i) in E_train else 0.
            deg_u: (B_u,) degrees of batch users in train graph.
            deg_i: (B_i,) degrees of batch items in train graph.
            beta: (D,) spectral vector.
            q_u0, q_i0, q_u1, q_i1: float scaling constants (frozen medians).
            M_norm: optional float 95th quantile norm threshold.

        Returns:
            total_loss: scalar tensor for backward.
            diagnostics: dictionary with alignment, uniformity, loss_u2i, loss_i2u, raw_cl_loss.
        """
        device = u_0.device
        B_u = u_0.size(0)
        B_i = i_0.size(0)

        # Scale denominator factor: 2 * R^2 * tau
        denom_scale = 2.0 * (self.radius_cap ** 2) * self.tau

        # ─────────────────────────────────────────────────────────────────────
        # 1. Scale Normalization & Radius Capping for Standard Embeddings
        # ─────────────────────────────────────────────────────────────────────
        # Direction 1 queries: U_0
        v_u0 = self.geo.cap_radius(u_0 / max(q_u0, 1e-4))

        # Direction 1 standard candidate keys: I_1
        i1_cand = i_1
        if not self.disable_reweight:
            i1_cand = spectral_reweight(i_1, beta, self.eps_beta, M_norm)
        v_i1 = self.geo.cap_radius(i1_cand / max(q_i1, 1e-4))

        # Direction 2 queries: I_0
        v_i0 = self.geo.cap_radius(i_0 / max(q_i0, 1e-4))

        # Direction 2 standard candidate keys: U_1
        u1_cand = u_1
        if not self.disable_reweight:
            u1_cand = spectral_reweight(u_1, beta, self.eps_beta, M_norm)
        v_u1 = self.geo.cap_radius(u1_cand / max(q_u1, 1e-4))

        # Standard pairwise distance matrices (B_u x B_i and B_i x B_u)
        D1 = self.geo.compute_distance_matrix(v_u0, v_i1)  # (B_u, B_i)
        D2 = self.geo.compute_distance_matrix(v_i0, v_u1)  # (B_i, B_u)

        # ─────────────────────────────────────────────────────────────────────
        # 2. Self-Return Removal on True Positive Pairs (P == 1)
        # ─────────────────────────────────────────────────────────────────────
        pos_u_idx, pos_i_idx = torch.where(P > 0.5)
        num_pos_pairs = pos_u_idx.numel()

        if num_pos_pairs > 0 and not self.disable_self_return:
            # --- Direction 1 Positive Keys: I_1 conditioned on U_0 ---
            h1_i_pos = i_1[pos_i_idx]
            e_u_pos = u_0[pos_u_idx]
            deg_u_pos = deg_u[pos_u_idx]
            deg_i_pos = deg_i[pos_i_idx]

            i1_pos_corrected = remove_self_return(
                H1_target=h1_i_pos,
                E_source=e_u_pos,
                deg_source=deg_u_pos,
                deg_target=deg_i_pos,
                beta=beta,
                eps_floor=self.eps_floor,
            )
            if not self.disable_reweight:
                i1_pos_corrected = spectral_reweight(i1_pos_corrected, beta, self.eps_beta, M_norm)
            v_i1_pos = self.geo.cap_radius(i1_pos_corrected / max(q_i1, 1e-4))
            v_u0_pos = v_u0[pos_u_idx]

            # Compute paired distance for positive keys
            D1_pos = self.geo.compute_paired_distance(v_u0_pos, v_i1_pos)
            D1 = D1.clone()
            D1[pos_u_idx, pos_i_idx] = D1_pos

            # --- Direction 2 Positive Keys: U_1 conditioned on I_0 ---
            h1_u_pos = u_1[pos_u_idx]
            e_i_pos = i_0[pos_i_idx]

            u1_pos_corrected = remove_self_return(
                H1_target=h1_u_pos,
                E_source=e_i_pos,
                deg_source=deg_i_pos,
                deg_target=deg_u_pos,
                beta=beta,
                eps_floor=self.eps_floor,
            )
            if not self.disable_reweight:
                u1_pos_corrected = spectral_reweight(u1_pos_corrected, beta, self.eps_beta, M_norm)
            v_u1_pos = self.geo.cap_radius(u1_pos_corrected / max(q_u1, 1e-4))
            v_i0_pos = v_i0[pos_i_idx]

            D2_pos = self.geo.compute_paired_distance(v_i0_pos, v_u1_pos)
            D2 = D2.clone()
            D2[pos_i_idx, pos_u_idx] = D2_pos

        # ─────────────────────────────────────────────────────────────────────
        # 3. Multi-Positive InfoNCE Computation
        # ─────────────────────────────────────────────────────────────────────
        # Direction 1: Logits = -D1 / denom_scale
        Z1 = -D1 / denom_scale  # (B_u, B_i)
        lse1 = torch.logsumexp(Z1, dim=-1, keepdim=True)  # (B_u, 1)

        # Log probability matrix
        log_prob1 = Z1 - lse1  # (B_u, B_i)

        pos_counts1 = P.sum(dim=-1)  # (B_u,)
        # Valid users: at least 1 positive and at least 1 negative
        valid_u = (pos_counts1 >= 1.0) & (pos_counts1 < float(B_i))

        if valid_u.any():
            pos_log_prob1 = (log_prob1 * P).sum(dim=-1)  # (B_u,)
            mean_pos_log_prob1 = pos_log_prob1[valid_u] / pos_counts1[valid_u]
            loss_u2i = -mean_pos_log_prob1.mean()
        else:
            loss_u2i = torch.tensor(0.0, device=device, requires_grad=True)

        # Direction 2: Logits = -D2 / denom_scale
        Z2 = -D2 / denom_scale  # (B_i, B_u)
        lse2 = torch.logsumexp(Z2, dim=-1, keepdim=True)  # (B_i, 1)

        log_prob2 = Z2 - lse2  # (B_i, B_u)
        P_t = P.t()  # (B_i, B_u)
        pos_counts2 = P_t.sum(dim=-1)  # (B_i,)
        valid_i = (pos_counts2 >= 1.0) & (pos_counts2 < float(B_u))

        if valid_i.any():
            pos_log_prob2 = (log_prob2 * P_t).sum(dim=-1)  # (B_i,)
            mean_pos_log_prob2 = pos_log_prob2[valid_i] / pos_counts2[valid_i]
            loss_i2u = -mean_pos_log_prob2.mean()
        else:
            loss_i2u = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = 0.5 * loss_u2i + 0.5 * loss_i2u

        # ─────────────────────────────────────────────────────────────────────
        # 4. Telemetry & Health Diagnostics (Peer Review Round 2 Requirement)
        # ─────────────────────────────────────────────────────────────────────
        with torch.no_grad():
            pos_mask = (P > 0.5)
            neg_mask = (P < 0.5)

            if pos_mask.any():
                alignment = D1[pos_mask].mean().item()
            else:
                alignment = 0.0

            if neg_mask.any():
                # Uniformity: log E_{neg}[exp(-2 * d_neg)]
                d_neg = D1[neg_mask]
                uniformity = torch.log(torch.exp(-2.0 * d_neg).mean() + 1e-8).item()
            else:
                uniformity = 0.0

        diagnostics = {
            "loss_u2i": loss_u2i.item() if isinstance(loss_u2i, torch.Tensor) else float(loss_u2i),
            "loss_i2u": loss_i2u.item() if isinstance(loss_i2u, torch.Tensor) else float(loss_i2u),
            "raw_cl_loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss),
            "alignment": alignment,
            "uniformity": uniformity,
        }

        return total_loss, diagnostics
