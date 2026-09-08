# -*- coding: utf-8 -*-
"""
models/stair_ne_nlgcl_plus.py — STAIR-NE-NLGCL+ (v3) Module
=============================================================
Selective Synergy Architecture:
- Backbone: Layer-wise Neighborhood-Enriched Graph Contrastive (H^(0) <-> H^(1))
- Perturbation: Spectral-Decayed True Sign-Preserving Noise (|noise| >= 0)
- Decoupling: Contrastive MLP Projection Head (Linear + LayerNorm + LeakyReLU)
- False Negative Protection: Thresholded Dynamic MFNA with Dynamic Slicing [B x B]
- Adaptive Scheduling: Hybrid Dynamic HANS Scheduler (Cosine Ceiling Cap + Loss-Gated Feedback)
- Modality Projection: Regularized Diagonal Spectral Projector (0-rotation + L2 Anchoring)
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'STAIR_NE_NLGCL_Plus',
    'RegularizedDiagonalSpectralProjector',
    'DiagonalSpectralProjector',
]


class RegularizedDiagonalSpectralProjector(nn.Module):
    """
    Zero-rotation diagonal spectral projector with L2 anchoring regularization:
        E_proj = E_svd * w
    Locks coordinate axes to preserve orthogonal SVD basis and monotonic spectral decay.
    """
    def __init__(self, dim: int = 64, reg_weight: float = 1e-4):
        super().__init__()
        self.dim = dim
        self.reg_weight = float(reg_weight)
        self.w = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w

    def get_anchoring_loss(self) -> torch.Tensor:
        if self.reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.w.device)
        return self.reg_weight * torch.sum((self.w - 1.0) ** 2)


# Backward compatibility alias
DiagonalSpectralProjector = RegularizedDiagonalSpectralProjector


class STAIR_NE_NLGCL_Plus(nn.Module):
    """
    STAIR-NE-NLGCL+ (v3) Hybrid Contrastive Learning Module.
    Combines Phase 2 v5 graph neighborhood contrast with Phase 3 v2.1 mathematical pillars:
    Absolute Sign-Preserving Noise, Contrastive MLP Projection Head,
    Thresholded Dynamic MFNA with Dynamic Slicing, and Hybrid Dynamic HANS Scheduler.
    """

    def __init__(
        self,
        dim: int = 64,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        tau: float = 0.20,
        alpha_dir: float = 0.50,
        eps: float = 0.10,
        tau_thresh: float = 0.85,
        lambda_max: float = 0.010,
        lambda_min: float = 0.002,
        gamma_max: float = 0.35,
        gamma_min: float = 0.05,
        warmup_epochs: int = 50,
        total_epochs: int = 500,
    ):
        super().__init__()
        self.dim = dim
        self.n_users = n_users
        self.n_items = n_items
        self.tau = tau
        self.alpha_dir = alpha_dir
        self.eps = eps
        self.tau_thresh = tau_thresh

        # Scheduler bounds
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # Pillar 3: Contrastive MLP Projection Head (Decoupling GNN Representation)
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2)
        )

        # Dynamic state trackers
        self.current_epoch = 0
        self.current_lambda = lambda_min
        self.current_gamma_h = gamma_min
        self.loss_history: List[float] = []

    def update_scheduler(self, current_cl_loss: float, window: int = 10, threshold: float = 0.99):
        """
        Pillar 5: Hybrid Dynamic HANS Scheduler.
        Combines Cosine-Annealed Ceiling Cap with a Loss-Gated Feedback Loop.
        """
        self.current_epoch += 1
        epoch = self.current_epoch

        # 1. Warmup stage: linear ramp-up of lambda, constant gamma_min
        if epoch <= self.warmup_epochs:
            ratio = float(epoch) / float(max(1, self.warmup_epochs))
            self.current_lambda = self.lambda_min + ratio * (self.lambda_max - self.lambda_min)
            self.current_gamma_h = self.gamma_min
            return

        # 2. Cooling stage: Cosine Annealing dynamic ceiling for lambda and gamma_h
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        dynamic_lambda_cap = self.lambda_min + (self.lambda_max - self.lambda_min) * cosine_decay
        dynamic_gamma_cap = self.gamma_min + (self.gamma_max - self.gamma_min) * cosine_decay

        self.current_lambda = max(self.lambda_min, dynamic_lambda_cap)

        # 3. Loss-Gated Feedback Loop: Adapt gamma_h below dynamic ceiling cap
        self.loss_history.append(float(current_cl_loss))
        if len(self.loss_history) > window * 2:
            self.loss_history.pop(0)
            loss_curr = sum(self.loss_history[-window:]) / float(window)
            loss_prev = sum(self.loss_history[-window*2:-window]) / float(window)

            # Increase hard-negative pressure if CL loss plateaus
            if loss_curr >= threshold * loss_prev:
                self.current_gamma_h = min(self.current_gamma_h + 0.015, dynamic_gamma_cap)
            else:
                self.current_gamma_h = max(self.current_gamma_h - 0.010, self.gamma_min)
        else:
            # Maintain gamma within ceiling before window fills up
            self.current_gamma_h = min(self.current_gamma_h, dynamic_gamma_cap)

    def get_current_hans_params(self) -> Tuple[float, float]:
        """Returns (current_gamma_h, current_lambda)."""
        return self.current_gamma_h, self.current_lambda

    def inject_spectral_noise(self, h: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Pillar 2: True Sign-Preserving Spectral Perturbation.
        Using absolute Gaussian noise |eta| guarantees:
            sign(h_tilde_d) = sign(h_d) * sign(|h_d| + c_d) == sign(h_d) 100% of the time.
        """
        if not self.training or self.eps <= 0.0:
            return h

        noise = torch.randn_like(h).abs()
        noise = F.normalize(noise, p=2, dim=-1)
        
        beta_weight = beta.unsqueeze(0) if beta.dim() == 1 else beta
        h_perturbed = h + self.eps * (beta_weight * torch.sign(h) * noise)
        return h_perturbed

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        positives: torch.Tensor,
        beta: torch.Tensor,
        item_modals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Forward pass for STAIR-NE-NLGCL+ contrastive loss.

        Args:
            layer_embeds: List of intermediate representations [H^(0), H^(1), ...]
            users:        (B,) user indices in mini-batch
            positives:    (B,) positive item indices in mini-batch
            beta:         (D,) spectral propagation decay curve (1 - beta3)
            item_modals:  Optional (N_i, D) or (B, D) modal feature matrix

        Returns:
            Tuple of:
            - weighted_loss: Scalar loss tensor (scaled by current_lambda) for backprop
            - raw_loss:      Raw InfoNCE loss value (float) for scheduler tracking
            - lambda_val:    Current contrastive weight lambda (float)
        """
        users = users.view(-1)
        positives = positives.view(-1)
        device = layer_embeds[0].device
        batch_size = users.size(0)

        # 1. Layer-wise Representation Extraction (H^(0) and H^(1))
        if self.n_users is not None and self.n_items is not None:
            U_0, I_0 = torch.split(layer_embeds[0], [self.n_users, self.n_items])
            U_1, I_1 = torch.split(layer_embeds[1], [self.n_users, self.n_items])
            u_0 = U_0[users]
            i_1 = I_1[positives]
            i_0 = I_0[positives]
            u_1 = U_1[users]
        else:
            num_u = layer_embeds[0].size(0) - (item_modals.size(0) if (item_modals is not None and item_modals.size(0) != batch_size) else 0)
            u_0 = layer_embeds[0][users]
            i_1 = layer_embeds[1][num_u + positives]
            i_0 = layer_embeds[0][num_u + positives]
            u_1 = layer_embeds[1][users]

        # 2. Inject Spectral-Decayed True Sign-Preserving Noise
        u_0_tilde = self.inject_spectral_noise(u_0, beta)
        i_1_tilde = self.inject_spectral_noise(i_1, beta)
        i_0_tilde = self.inject_spectral_noise(i_0, beta)
        u_1_tilde = self.inject_spectral_noise(u_1, beta)

        # 3. Decouple via Contrastive MLP Projection Head
        u_0_proj = self.proj_head(u_0_tilde)
        i_1_proj = self.proj_head(i_1_tilde)
        i_0_proj = self.proj_head(i_0_tilde)
        u_1_proj = self.proj_head(u_1_tilde)

        # L2-normalization onto hypersphere
        u_0_norm = F.normalize(u_0_proj, p=2, dim=-1)
        i_1_norm = F.normalize(i_1_proj, p=2, dim=-1)
        i_0_norm = F.normalize(i_0_proj, p=2, dim=-1)
        u_1_norm = F.normalize(u_1_proj, p=2, dim=-1)

        # 4. Pillar 4: Thresholded Dynamic MFNA with Dynamic Slicing [B x B]
        if item_modals is not None and self.tau_thresh < 1.0:
            with torch.no_grad():
                # Dynamic Slicing: only slice batch positives if full matrix is provided
                i_modal_batch = item_modals[positives] if item_modals.size(0) != batch_size else item_modals
                i_modal_norm = F.normalize(i_modal_batch, p=2, dim=-1)
                sim_modal = torch.matmul(i_modal_norm, i_modal_norm.t())  # [B, B]
                
                # Soft attenuation on suspected false negatives (sim > tau_thresh)
                excess_sim = torch.clamp(
                    (sim_modal - self.tau_thresh) / max(1e-5, (1.0 - self.tau_thresh)),
                    0.0, 1.0
                )
                mfna_alpha = 1.0 - excess_sim  # [B, B]
        else:
            mfna_alpha = torch.ones((batch_size, batch_size), device=device)

        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 5. Direction 1: User-to-Item Neighborhood CL (U_0 -> I_1)
        # ─────────────────────────────────────────────────────────────────
        pos_u2i = torch.sum(u_0_norm * i_1_norm, dim=-1) / self.tau  # [B]
        sim_u2i = torch.matmul(u_0_norm, i_1_norm.t()) / self.tau    # [B, B]

        # Graph HANS Negative Hardness Weighting: Psi = exp(gamma_h * sim)
        hans_u2i = torch.exp(torch.clamp(self.current_gamma_h * sim_u2i, max=5.0))
        exp_u2i = torch.exp(sim_u2i)
        
        weighted_neg_u2i = (mfna_alpha * hans_u2i * exp_u2i).masked_fill(~diag_mask, 0.0)
        sum_neg_u2i = weighted_neg_u2i.sum(dim=-1)
        loss_u2i = -(pos_u2i - torch.log(torch.exp(pos_u2i) + sum_neg_u2i + 1e-8)).mean()

        # ─────────────────────────────────────────────────────────────────
        # 6. Direction 2: Item-to-User Neighborhood CL (I_0 -> U_1)
        # ─────────────────────────────────────────────────────────────────
        pos_i2u = torch.sum(i_0_norm * u_1_norm, dim=-1) / self.tau  # [B]
        sim_i2u = torch.matmul(i_0_norm, u_1_norm.t()) / self.tau    # [B, B]

        hans_i2u = torch.exp(torch.clamp(self.current_gamma_h * sim_i2u, max=5.0))
        exp_i2u = torch.exp(sim_i2u)
        
        # Symmetrical transposed MFNA mask
        weighted_neg_i2u = (mfna_alpha.t() * hans_i2u * exp_i2u).masked_fill(~diag_mask, 0.0)
        sum_neg_i2u = weighted_neg_i2u.sum(dim=-1)
        loss_i2u = -(pos_i2u - torch.log(torch.exp(pos_i2u) + sum_neg_i2u + 1e-8)).mean()

        # 7. Total Multi-Task Contrastive Loss
        raw_loss = self.alpha_dir * loss_u2i + (1.0 - self.alpha_dir) * loss_i2u
        weighted_loss = self.current_lambda * raw_loss

        return weighted_loss, raw_loss.item(), self.current_lambda
