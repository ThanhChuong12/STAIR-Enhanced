# -*- coding: utf-8 -*-
"""
models/stair_sre_ans_v2.py
======================================================================
STAIR-SRE-ANS v2: Stepwise Spectral-Refined Contrastive Learning with
Adaptive Negative Scheduling & Continuous Spectral Difficulty Decoupling.

Phase 3 -- Batch 2 (Giai đoạn 3 -- Đợt 2)

Architecture Highlights:
  Pillar 1: Regularized Diagonal Spectral Projector (0-rotation)
     E_proj = E_svd * w   where w in R^D (initialized to 1.0)
     Anchoring loss: L_reg_w = lambda_w * ||w - 1||_2^2
     Preserves orthogonal SVD coordinates and monotonic FSC filter ordering.

  Pillar 2: Continuous Spectral Difficulty Decoupling (No Hard Dim-32 Boundary)
     Decouples low-frequency (graph collaborative) and high-frequency
     (multimodal detail) signals using STAIR's continuous beta(d) function:
       beta(d) = 0.9 * (1 - (d/D)^0.1)
     Applies independent L2 normalization to each spectral band, eliminating
     collaborative magnitude dominance without arbitrary binary slicing.

  Pillar 3: Gated Top-K Hard Negative Selection
     Selection_Score = Difficulty * Attenuation
     Filters out False Negatives BEFORE Top-K allocation, eliminating the
     Top-K Budget Bottleneck and ensuring 100% budget for True Hard Negatives.

  Pillar 4: Thresholded Cosine-Gated False Negative Attenuation (MFNA)
     M_gated = Metadata * max(0, cos)
     W = sigmoid(sim_all) * (1.0 + 0.5 * M_gated)
     Attenuation = clamp(1.0 - W, min=0.0, max=1.0)
     Preserves fine-grained intra-category ranking for orthogonal items.

  Pillar 5: Decoupled HANS Scheduler & Training-Guarded Memory Bank FIFO
     Monitors contrastive loss (loss_ans) independently from BPR.
     Loss-Gated Trigger with exponential smoothing update (step <= 0.02,
     cap <= 0.35) harmonizes smoothly with BSC gradients.
     FIFO Queue Q in R^(4096 x 64) with training guard (if self.training)
     eliminates evaluation distribution leak.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'RegularizedDiagonalSpectralProjector',
    'DiagonalSpectralProjector',
    'StepwiseSREANSLoss',
]


class RegularizedDiagonalSpectralProjector(nn.Module):
    """
    Zero-rotation diagonal spectral projector with L2 anchoring regularization.

    Preserves the orthogonal SVD coordinate system and monotonic spectral decay:
        E_proj = E_svd * w

    Args:
        dim: Embedding dimension (default: 64).
        reg_weight: L2 anchoring penalty coefficient lambda_w (default: 1e-4).
    """

    def __init__(self, dim: int = 64, reg_weight: float = 1e-4):
        super(RegularizedDiagonalSpectralProjector, self).__init__()
        self.dim = dim
        self.reg_weight = float(reg_weight)
        # Initialized to 1.0 so epoch 0 matches SVD coordinates identically
        self.w = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Element-wise Hadamard scaling: E_proj = E_svd * w"""
        return x * self.w

    def get_anchoring_loss(self) -> torch.Tensor:
        """
        L2 anchoring penalty: L_reg_w = lambda_w * ||w - 1||_2^2.
        Prevents coordinate drift away from original SVD manifold.
        """
        if self.reg_weight <= 0.0:
            return torch.tensor(0.0, device=self.w.device)
        return self.reg_weight * torch.sum((self.w - 1.0) ** 2)


# Alias for backward compatibility
DiagonalSpectralProjector = RegularizedDiagonalSpectralProjector


class StepwiseSREANSLoss(nn.Module):
    """
    Stepwise Spectral-Refined Contrastive Loss with Adaptive Negative Scheduling
    and Continuous Spectral Decoupling (STAIR-SRE-ANS v2).

    Args:
        dim:            Embedding dimension D (default: 64).
        tau:            InfoNCE temperature hyperparameter (default: 0.20).
        queue_size:     Capacity of cross-batch FIFO Memory Bank Q (default: 4096).
        warmup_epochs:  Number of initial bootstrap epochs with gamma_h = 0.05 (default: 50).
        gamma_max:      Ceiling cap for hard negative penalty coefficient (default: 0.35).
        hn_ratio_max:   Ceiling cap for hard negative selection ratio (default: 0.40).
        subspace_alpha: Balance between low-frequency and high-frequency bands (default: 0.50).
    """

    def __init__(
        self,
        dim: int = 64,
        tau: float = 0.20,
        queue_size: int = 4096,
        warmup_epochs: int = 50,
        gamma_max: float = 0.35,
        hn_ratio_max: float = 0.40,
        subspace_alpha: float = 0.50,
    ):
        super(StepwiseSREANSLoss, self).__init__()
        self.dim = dim
        self.tau = float(tau)
        self.queue_size = int(queue_size)
        self.warmup_epochs = int(warmup_epochs)
        self.gamma_max = float(gamma_max)
        self.hn_ratio_max = float(hn_ratio_max)
        self.subspace_alpha = float(subspace_alpha)

        # ------------------------------------------------------------------
        # Pillar 2: Continuous Spectral Decay Vector beta(d)
        # beta(d) = 0.9 * (1 - (d / D)^0.10)
        # Completely continuous across all 64 dimensions -- NO hard dim-32 boundary!
        # ------------------------------------------------------------------
        d_indices = torch.arange(dim, dtype=torch.float32)
        beta_curve = 0.9 * (1.0 - torch.pow(d_indices / float(dim), 0.10))
        self.register_buffer('beta', beta_curve)             # [D] Low-frequency band (Graph Collaborative)
        self.register_buffer('beta_high', 1.0 - beta_curve)  # [D] High-frequency band (Multimodal Invariant)

        # ------------------------------------------------------------------
        # Pillar 5: Cross-Batch FIFO Memory Bank Queue (Detached, no gradients)
        # ------------------------------------------------------------------
        self.register_buffer('neg_queue', torch.randn(queue_size, dim))
        self.neg_queue = F.normalize(self.neg_queue, p=2, dim=-1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # HANS Adaptive Scheduling state
        self.hn_ratio = 0.10
        self.gamma_h = 0.05
        self.current_epoch = 0
        self.loss_history = []

    @torch.no_grad()
    def enqueue_negatives(self, pos_emb: torch.Tensor):
        """
        Updates FIFO Memory Bank with positive item representations from current batch.
        Uses circular buffer pointer to overwrite oldest entries.
        """
        batch_size = pos_emb.size(0)
        norm_emb = F.normalize(pos_emb.detach(), p=2, dim=-1)

        ptr = int(self.queue_ptr.item())
        if ptr + batch_size <= self.queue_size:
            self.neg_queue[ptr:ptr + batch_size] = norm_emb
            ptr = (ptr + batch_size) % self.queue_size
        else:
            first_chunk = self.queue_size - ptr
            self.neg_queue[ptr:] = norm_emb[:first_chunk]
            remain = batch_size - first_chunk
            self.neg_queue[:remain] = norm_emb[first_chunk:]
            ptr = remain

        self.queue_ptr[0] = ptr

    def update_scheduler(
        self,
        current_cl_loss: float,
        window: int = 10,
        threshold: float = 0.99,
    ):
        """
        HANS (Hardness-Aware Negative Scheduling) Decoupled Scheduler.

        Monitors CONTRASTIVE LOSS (loss_ans) independently from BPR ranking loss.
        Applies Loss-Gated Trigger when CL loss improvement plateaus.
        """
        self.current_epoch += 1

        # Bootstrap Warmup Phase
        if self.current_epoch < self.warmup_epochs:
            self.gamma_h = 0.05
            self.hn_ratio = 0.10
            return

        self.loss_history.append(float(current_cl_loss))
        if len(self.loss_history) > window * 2:
            self.loss_history.pop(0)
            loss_curr = sum(self.loss_history[-window:]) / float(window)
            loss_prev = sum(self.loss_history[-window * 2:-window]) / float(window)

            # Loss-Gated Trigger: activate when CL loss plateaus (reduction < 1%)
            if loss_curr >= threshold * loss_prev:
                self.gamma_h = min(self.gamma_h + 0.02, self.gamma_max)
                self.hn_ratio = min(self.hn_ratio + 0.02, self.hn_ratio_max)
            else:
                self.gamma_h = max(self.gamma_h - 0.01, 0.05)
                self.hn_ratio = max(self.hn_ratio - 0.01, 0.10)

    def forward(
        self,
        u_embed: torch.Tensor,
        i_pos_embed: torch.Tensor,
        batch_users: Optional[torch.Tensor] = None,
        batch_items: Optional[torch.Tensor] = None,
        metadata_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes Stepwise SRE-ANS v2 Loss.

        Args:
            u_embed:        (N_u, D) or (B, D) user representations.
            i_pos_embed:    (N_i, D) or (B, D) positive item representations.
            batch_users:    (B,) user indices, or None if u_embed is already sliced.
            batch_items:    (B,) positive item indices, or None if i_pos_embed is sliced.
            metadata_mask:  (B, Q) optional binary mask for brand/category match.

        Returns:
            Scalar InfoNCE loss tensor.
        """
        device = u_embed.device

        # 1. Mini-batch extraction
        if batch_users is not None:
            u_batch = u_embed[batch_users.view(-1)]
        else:
            u_batch = u_embed

        if batch_items is not None:
            pos_batch = i_pos_embed[batch_items.view(-1)]
        else:
            pos_batch = i_pos_embed

        B = u_batch.size(0)

        # 2. Negative pool from FIFO Queue (detached)
        neg_pool = self.neg_queue.clone().to(device)  # [Q, D]
        Q = neg_pool.size(0)

        # ------------------------------------------------------------------
        # Pillar 2: Continuous Spectral Difficulty Decoupling
        # Continuous Hadamard spectral decomposition across all 64 dimensions
        # ------------------------------------------------------------------
        beta = self.beta.to(device)
        beta_high = self.beta_high.to(device)

        # Separate L2 Normalization on each spectral band
        u_low = F.normalize(u_batch * beta, p=2, dim=-1)
        u_high = F.normalize(u_batch * beta_high, p=2, dim=-1)

        neg_low = F.normalize(neg_pool * beta, p=2, dim=-1)
        neg_high = F.normalize(neg_pool * beta_high, p=2, dim=-1)

        cos_low = torch.matmul(u_low, neg_low.T)     # [B, Q] - Low-frequency similarity
        cos_high = torch.matmul(u_high, neg_high.T) # [B, Q] - High-frequency similarity

        # Difficulty Score combines both spectral bands fairly (50/50)
        difficulty = self.subspace_alpha * cos_low + (1.0 - self.subspace_alpha) * cos_high  # [B, Q]

        # ------------------------------------------------------------------
        # Full Hyperspherical Cosine Similarity & Scaled Logits
        # ------------------------------------------------------------------
        u_norm = F.normalize(u_batch, p=2, dim=-1)
        pos_norm = F.normalize(pos_batch, p=2, dim=-1)
        neg_norm = F.normalize(neg_pool, p=2, dim=-1)

        cos_all = torch.matmul(u_norm, neg_norm.T)                 # [B, Q]
        sim_all = cos_all / self.tau                              # [B, Q]

        # ------------------------------------------------------------------
        # Pillar 4: Thresholded Cosine-Gated False Negative Attenuation (MFNA)
        # Only activates metadata mask when cosine similarity is positive
        # ------------------------------------------------------------------
        if metadata_mask is not None:
            gated_metadata = metadata_mask.to(device) * torch.clamp(cos_all, min=0.0)
            W = torch.sigmoid(sim_all) * (1.0 + 0.5 * gated_metadata)
        else:
            W = torch.sigmoid(sim_all)
        attenuation = torch.clamp(1.0 - W, min=0.0, max=1.0)       # [B, Q]

        # ------------------------------------------------------------------
        # Pillar 3: Gated Top-K Selection (Eliminates Budget Bottleneck)
        # Selection_Score = Difficulty * Attenuation
        # Filters False Negatives BEFORE Top-K hard negative allocation
        # ------------------------------------------------------------------
        selection_score = difficulty * attenuation                 # [B, Q]
        k_hn = max(1, int(self.hn_ratio * Q))
        _, hn_indices = torch.topk(selection_score, k=k_hn, dim=1)  # [B, k_hn]

        # ------------------------------------------------------------------
        # Stratified Negative Penalty (Strict Order: Psi_HN >= 1.0 > Psi_EN)
        # ------------------------------------------------------------------
        diff_norm = (difficulty + 1.0) / 2.0                       # Normalized to [0, 1]
        psi_HN = 1.0 + self.gamma_h * diff_norm                    # >= 1.0
        psi_EN = 1.0 - self.gamma_h                                # <= 1.0

        psi_all = torch.full_like(sim_all, psi_EN)                 # Default: Easy Negatives
        hn_mask = torch.zeros_like(sim_all, dtype=torch.bool)
        hn_mask.scatter_(1, hn_indices, True)
        psi_all = torch.where(hn_mask, psi_HN, psi_all)            # Inject Hard Negatives

        # Modulate penalty by attenuation
        final_weights = psi_all * attenuation                      # [B, Q]

        # ------------------------------------------------------------------
        # Vectorized InfoNCE Loss with Gather on Top-K Hard Negatives
        # ------------------------------------------------------------------
        pos_sim = torch.sum(u_norm * pos_norm, dim=-1) / self.tau  # [B]
        pos_exp = torch.exp(pos_sim)                               # [B]
        exp_all = torch.exp(sim_all)                               # [B, Q]

        hn_exp = torch.gather(exp_all, dim=1, index=hn_indices)           # [B, k_hn]
        hn_weights = torch.gather(final_weights, dim=1, index=hn_indices) # [B, k_hn]

        neg_weighted_sum = (hn_exp * hn_weights).sum(dim=1)               # [B]

        loss = -torch.log(pos_exp / (pos_exp + neg_weighted_sum + 1e-8)).mean()

        # ------------------------------------------------------------------
        # Training Guard: Prevent Evaluation Leak into FIFO Queue
        # ------------------------------------------------------------------
        if self.training:
            with torch.no_grad():
                self.enqueue_negatives(pos_batch)

        return loss
