"""
models/stair_nlgcl.py — STAIR-NLGCL v4
========================================
Core Architecture: STAIR Baseline + Neighborhood-enriched Contrastive Learning (NLGCL).

Design Philosophy (from STAIRE2_v4 Task Prompt):
─────────────────────────────────────────────────
1. **NO hard split** of the 64-D embedding space.
   STAIR's spectral decay β₃(d) = 0.1 + 0.9·(d/D)^γ is a *continuous* weighting curve.
   Slicing the space into "collaborative / modal" halves would destroy this distribution.

2. **Zero-cost Augmentation via FSC Intermediates.**
   Instead of expensive graph dropout or edge perturbation, we extract the *intermediate
   representations* {H⁰, H¹, …, Hᴸ} produced by Forward Stepwise Convolution (FSC).
   Each Hˡ = Ã^l · E · β^l captures l-hop neighborhood structure — a natural "view"
   for contrastive learning, obtained at zero additional forward-pass cost.

3. **In-batch Heterogeneous GCL (VRAM-safe).**
   Following NLGCL/NLGCL-Plus, we contrast *across entity types*:
     • User(Layer g) ↔ Item(Layer g+1)   — "How well does a user's ego embedding
                                              predict her neighbor items' 1-hop view?"
     • Item(Layer g) ↔ User(Layer g+1)   — symmetric counterpart.
   Crucially, the denominator of InfoNCE is computed **only over the in-batch negatives**,
   avoiding the O(N²) global contrast that would OOM on Kaggle T4.

Mathematical Formulation
────────────────────────
Given L=3 FSC layers producing layer_embeds = [H⁰, H¹, H², H³]:

  For g = 0, 1, …, G-1 (default G=1, contrasting Layer 0 vs Layer 1):
    U_g, I_g = split(H^g)          — ego user/item embeddings at layer g
    U_{g+1}, I_{g+1} = split(H^{g+1}) — 1-hop propagated embeddings

    L_u = -log [ exp(sim(I_{g+1}[pos], U_g[u]) / τ)
                 / Σ_k exp(sim(I_{g+1}[pos], U_g[k]) / τ) ]

    L_i = -log [ exp(sim(U_{g+1}[u], I_g[pos]) / τ)
                 / Σ_k exp(sim(U_{g+1}[u], I_g[k]) / τ) ]

    L_NLGCL = Σ_g (α · L_u + (1-α) · L_i)

  Total Loss:
    L = L_BPR + λ_nlgcl · L_NLGCL

References:
  - NLGCL: "Neighborhood-enriched Contrastive Learning for Graph Collaborative Filtering" (2024)
  - NLGCL-Plus: lgmrec_plus.py — neighbor_cl_loss, InfoNCE
  - NEGCL: negcl.py — ssl_triple_loss, modality graph embedding
  - STAIR: main.py — Forward Stepwise Convolution (FSC), BSC Smoother, AdamWSEvo
  - main_v1.py — Previous v1 attempt (graph dropout + hard-split CL, abandoned)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLGCL_Module(nn.Module):
    """
    Standalone module for Neighborhood-enriched Graph Contrastive Learning.
    
    Can be composed with any GNN encoder that produces per-layer embeddings.
    Separated from the model class for clean modularity and testability.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
    ):
        """
        Args:
            n_users: Number of users (for splitting joint embeddings).
            n_items: Number of items.
            G:       Number of contrastive gaps. G=1 contrasts Layer_g vs Layer_{g+1}
                     for g=0 only. G=2 would also contrast Layer_1 vs Layer_2.
            tau:     Temperature for InfoNCE softmax.
            alpha:   Balance between user-side and item-side CL loss.
                     L = alpha * L_u + (1-alpha) * L_i
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha

    def info_nce_in_batch(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        In-batch InfoNCE loss.
        
        Given:
            anchor:    (B, D) — the query vectors
            positive:  (B, D) — the positive key for each query
            negatives: (B, D) — the negative pool (same batch entities)
        
        Computes:
            L = -mean[ log( exp(sim(a, p)/τ) / Σ_k exp(sim(a, n_k)/τ) ) ]
        
        where sim(a, b) = a·b (dot product on L2-normalized vectors).
        
        Uses logsumexp for numerical stability (avoids exp overflow).
        """
        # L2 normalize all vectors to unit sphere
        anchor    = F.normalize(anchor,    p=2, dim=-1)  # (B, D)
        positive  = F.normalize(positive,  p=2, dim=-1)  # (B, D)
        negatives = F.normalize(negatives, p=2, dim=-1)  # (B, D)

        # Positive similarity: (B,)
        pos_sim = (anchor * positive).sum(dim=-1) / self.tau

        # Negative similarities: anchor (B, D) × negatives^T (D, B) → (B, B)
        neg_sim = torch.mm(anchor, negatives.t()) / self.tau

        # Numerically stable InfoNCE via logsumexp
        # log[ exp(pos) / (exp(pos) + Σ exp(neg)) ]
        # = pos - logsumexp(cat(pos, all_neg))
        # Here neg_sim already includes the positive pair (when anchor_i aligns with negatives_i),
        # which is standard in InfoNCE (the positive is part of the denominator).
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=-1)

        return loss.mean()

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the In-batch Heterogeneous NLGCL loss.
        
        Args:
            layer_embeds: List of (N_users + N_items, D) tensors from FSC.
                          layer_embeds[0] = H⁰ (initial embeddings)
                          layer_embeds[l] = Ã^l · H^{l-1} · β (l-hop propagated)
            users:        (B,) or (B, 1) — user indices in the current batch.
            pos_items:    (B,) or (B, 1) — positive item indices in the current batch.
        
        Returns:
            Scalar loss tensor.
        """
        users = users.view(-1)          # (B,)
        pos_items = pos_items.view(-1)   # (B,)

        total_loss = torch.tensor(0.0, device=layer_embeds[0].device)

        # G contrastive gaps: contrast layer g against layer g+1
        num_gaps = min(self.G, len(layer_embeds) - 1)

        for g in range(num_gaps):
            # Split joint embeddings into user / item parts
            U_g, I_g = torch.split(
                layer_embeds[g], [self.n_users, self.n_items]
            )
            U_g1, I_g1 = torch.split(
                layer_embeds[g + 1], [self.n_users, self.n_items]
            )

            # ─────────────────────────────────────────────────────────
            # User-side CL: L_u
            #   anchor   = I_{g+1}[pos_items]  — item's 1-hop view
            #   positive = U_g[users]           — user's ego view
            #   negatives = U_g[users]          — all in-batch users as negatives
            # Intuition: "Can the propagated item embedding recognize its
            #             interacting user among all in-batch users?"
            # ─────────────────────────────────────────────────────────
            cl_u = self.info_nce_in_batch(
                anchor    = I_g1[pos_items],   # (B, D)
                positive  = U_g[users],        # (B, D)
                negatives = U_g[users],        # (B, D) — in-batch user pool
            )

            # ─────────────────────────────────────────────────────────
            # Item-side CL: L_i
            #   anchor   = U_{g+1}[users]      — user's 1-hop view
            #   positive = I_g[pos_items]       — item's ego view
            #   negatives = I_g[pos_items]      — all in-batch items as negatives
            # Intuition: "Can the propagated user embedding recognize its
            #             interacting item among all in-batch items?"
            # ─────────────────────────────────────────────────────────
            cl_i = self.info_nce_in_batch(
                anchor    = U_g1[users],       # (B, D)
                positive  = I_g[pos_items],    # (B, D)
                negatives = I_g[pos_items],    # (B, D) — in-batch item pool
            )

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        return total_loss
