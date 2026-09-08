# -*- coding: utf-8 -*-
"""
models/stair_sre_ans_v2_1.py
======================================================================
STAIR-SRE-ANS v2.1: Decoupled Multi-Scale Representation,
Thresholded Topology-Driven MFNA, Full Partition Conservation,
and Cosine-Annealed HANS Scheduler.

Phase 3 -- Batch 2 Upgrade (Giai đoạn 3 -- Đợt 2.1)

Key Architectural Fixes over v2:
  1. Decoupled Representation: InfoNCE operates on Layer-0 raw embeddings
     via an MLP Projection Head, liberating H^(L) for pure BPR clustering.
  2. Thresholded Dynamic MFNA: Fixes the sigmoid(0)=0.5 bug by only gating
     when cosine > 0.25, preserving 100% repulsive force for orthogonal negatives.
  3. Full Partition InfoNCE: Preserves all Q negatives in the denominator,
     restoring hyperspherical uniformity while modulating hard negatives.
  4. Cosine-Annealed HANS: Dynamic ceiling cools down in the final 150 epochs
     for smooth BPR decision boundary fine-tuning.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'RegularizedDiagonalSpectralProjector',
    'DiagonalSpectralProjector',
    'StepwiseSREANSLoss_v21',
]


class RegularizedDiagonalSpectralProjector(nn.Module):
    """
    Zero-rotation diagonal spectral projector with L2 anchoring regularization.
    Preserves orthogonal SVD coordinate system and monotonic spectral decay:
        E_proj = E_svd * w
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


class StepwiseSREANSLoss_v21(nn.Module):
    """
    Stepwise SRE-ANS Loss v2.1.

    Args:
        dim:            Embedding dimension D (default: 64).
        tau:            InfoNCE temperature hyperparameter (default: 0.25).
        queue_size:     Capacity of cross-batch FIFO Memory Bank Q (default: 4096).
        warmup_epochs:  Number of initial bootstrap epochs (default: 50).
        total_epochs:   Total training epochs for cosine schedule (default: 500).
        gamma_max:      Ceiling cap for hard negative penalty coefficient (default: 0.30).
        hn_ratio_max:   Ceiling cap for hard negative selection ratio (default: 0.35).
        subspace_alpha: Balance between low-frequency and high-frequency bands (default: 0.35).
        beta:           Precomputed continuous spectral decay profile.
        gamma:          Exponent for beta profile fallback (default: 0.10).
    """

    def __init__(
        self,
        dim: int = 64,
        tau: float = 0.25,
        queue_size: int = 4096,
        warmup_epochs: int = 50,
        total_epochs: int = 500,
        gamma_max: float = 0.30,
        hn_ratio_max: float = 0.35,
        subspace_alpha: float = 0.35,
        beta: Optional[torch.Tensor] = None,
        gamma: float = 0.10,
    ):
        super().__init__()
        self.dim = dim
        self.tau = float(tau)
        self.queue_size = int(queue_size)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.gamma_max = float(gamma_max)
        self.hn_ratio_max = float(hn_ratio_max)
        self.subspace_alpha = float(subspace_alpha)

        # Spectral continuous decay curves
        if beta is not None:
            beta_curve = beta.detach().clone().to(dtype=torch.float32)
        else:
            d_indices = torch.arange(dim, dtype=torch.float32)
            beta_curve = 0.9 * (1.0 - torch.pow(d_indices / float(dim), float(gamma)))
        self.register_buffer('beta', beta_curve)
        self.register_buffer('beta_high', 1.0 - beta_curve)

        # 1-Layer MLP Projection Head to decouple CL from GNN backbone
        self.proj_head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
        )

        # Cross-Batch Memory Bank FIFO Queue
        init_queue = F.normalize(torch.randn(queue_size, dim), p=2, dim=-1)
        self.register_buffer('neg_queue', init_queue)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # HANS Adaptive Dynamic Scheduler
        self.hn_ratio = 0.10
        self.gamma_h = 0.05
        self.current_epoch = 0
        self.loss_history = []

    @torch.no_grad()
    def enqueue_negatives(self, pos_emb: torch.Tensor):
        batch_size = pos_emb.size(0)
        norm_emb = F.normalize(pos_emb.detach(), p=2, dim=-1)
        if batch_size > self.queue_size:
            norm_emb = norm_emb[-self.queue_size:]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.neg_queue[ptr:ptr + batch_size] = norm_emb
            ptr = (ptr + batch_size) % self.queue_size
        else:
            first = self.queue_size - ptr
            self.neg_queue[ptr:] = norm_emb[:first]
            remain = batch_size - first
            self.neg_queue[:remain] = norm_emb[first:]
            ptr = remain
        self.queue_ptr[0] = ptr

    def update_scheduler(
        self,
        current_cl_loss: float,
        window: int = 10,
        threshold: float = 0.99,
    ):
        self.current_epoch += 1
        if self.current_epoch < self.warmup_epochs:
            self.gamma_h = 0.05
            self.hn_ratio = 0.10
            return

        # Cosine-Annealed Ceiling: relaxes HN pressure in late epochs for fine-tuning
        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, (self.total_epochs - self.warmup_epochs)
        )
        dynamic_gamma_cap = self.gamma_max * 0.5 * (
            1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()
        )
        dynamic_gamma_cap = max(0.08, dynamic_gamma_cap)

        self.loss_history.append(float(current_cl_loss))
        if len(self.loss_history) > window * 2:
            self.loss_history.pop(0)
            loss_curr = sum(self.loss_history[-window:]) / float(window)
            loss_prev = sum(self.loss_history[-window * 2:-window]) / float(window)

            if loss_curr >= threshold * loss_prev:
                self.gamma_h = min(self.gamma_h + 0.015, dynamic_gamma_cap)
                self.hn_ratio = min(self.hn_ratio + 0.015, self.hn_ratio_max)
            else:
                self.gamma_h = max(self.gamma_h - 0.01, 0.05)
                self.hn_ratio = max(self.hn_ratio - 0.01, 0.10)

    def forward(
        self,
        u_raw: torch.Tensor,
        i_raw: torch.Tensor,
        batch_users: Optional[torch.Tensor] = None,
        batch_items: Optional[torch.Tensor] = None,
        metadata_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = u_raw.device

        # 1. Mini-batch Extraction & Decoupled Projection onto Contrastive Manifold
        # Ensure 2D tensor shapes [B, D] by flattening input index vectors
        if batch_users is not None:
            u_in = u_raw[batch_users.view(-1)]
        else:
            u_in = u_raw.view(-1, self.dim)

        if batch_items is not None:
            pos_in = i_raw[batch_items.view(-1)]
        else:
            pos_in = i_raw.view(-1, self.dim)

        u_batch = self.proj_head(u_in)       # [B, D]
        pos_batch = self.proj_head(pos_in)   # [B, D]

        neg_pool = self.neg_queue.detach().to(device) # [Q, D]
        Q = neg_pool.size(0)

        # 2. Continuous Spectral Subspace Decoupling
        beta = self.beta.to(device)
        beta_high = self.beta_high.to(device)

        u_low = F.normalize(u_batch * beta, p=2, dim=-1)
        u_high = F.normalize(u_batch * beta_high, p=2, dim=-1)
        neg_low = F.normalize(neg_pool * beta, p=2, dim=-1)
        neg_high = F.normalize(neg_pool * beta_high, p=2, dim=-1)

        cos_low = torch.matmul(u_low, neg_low.T)     # [B, Q]
        cos_high = torch.matmul(u_high, neg_high.T)   # [B, Q]
        difficulty = self.subspace_alpha * cos_low + (1.0 - self.subspace_alpha) * cos_high # [B, Q]

        # 3. Thresholded Dynamic MFNA Attenuation
        u_norm = F.normalize(u_batch, p=2, dim=-1)
        pos_norm = F.normalize(pos_batch, p=2, dim=-1)
        neg_norm = F.normalize(neg_pool, p=2, dim=-1)

        cos_all = torch.matmul(u_norm, neg_norm.T)   # [B, Q]
        sim_all = cos_all / self.tau                 # [B, Q]

        # Gated threshold: only attenuate when cosine > 0.25 (True False-Negative Risk)
        active_mask = (cos_all > 0.25).float()
        W = torch.sigmoid(sim_all) * active_mask
        if metadata_mask is not None:
            gated_meta = metadata_mask.to(device) * torch.clamp(cos_all, min=0.0)
            W = W * (1.0 + 0.5 * gated_meta)
        attenuation = torch.clamp(1.0 - W, min=0.1, max=1.0) # [B, Q]

        # 4. Gated Top-K Selection
        selection_score = difficulty * attenuation   # [B, Q]
        k_hn = max(1, min(int(self.hn_ratio * Q), Q))
        _, hn_indices = torch.topk(selection_score, k=k_hn, dim=-1) # [B, k_hn]

        # 5. Stratified Weights & Full Partition InfoNCE
        diff_norm = (difficulty + 1.0) / 2.0
        psi_HN = 1.0 + self.gamma_h * diff_norm
        psi_EN = 1.0 - self.gamma_h

        psi_all = torch.full_like(sim_all, psi_EN)
        hn_mask = torch.zeros_like(sim_all, dtype=torch.bool)
        hn_mask.scatter_(-1, hn_indices, True)
        psi_all = torch.where(hn_mask, psi_HN, psi_all)

        final_weights = psi_all * attenuation # [B, Q]

        # Full Partition Conservation: sum over all Q negatives
        pos_sim = torch.sum(u_norm * pos_norm, dim=-1) / self.tau # [B]
        pos_exp = torch.exp(pos_sim)                              # [B]
        exp_all = torch.exp(sim_all)                              # [B, Q]
        neg_weighted_sum = (exp_all * final_weights).sum(dim=-1)  # [B]

        loss = -torch.log(pos_exp / (pos_exp + neg_weighted_sum + 1e-8)).mean()

        if self.training:
            with torch.no_grad():
                self.enqueue_negatives(pos_batch)

        return loss
