"""STAIR5-v4 Objectives: Faithful Neighborhood-enriched Contrastive Learning (NLGCL).

Implements:
1. Standard NLGCL in-batch InfoNCE loss (faithful to reference STAIR-NLGCL v4).
   - Zero-cost view generation using FSC intermediate representations.
   - Cross-entity contrastive pairs:
     * User-side: Item_{g+1}[pos_item] (anchor) vs User_g[user] (positive) vs all in-batch User_g (negatives).
     * Item-side: User_{g+1}[user] (anchor) vs Item_g[pos_item] (positive) vs all in-batch Item_g (negatives).
   - Numerically stable logsumexp.
   - Optional anchor-chunking for memory-bounded execution on large batches (e.g. B=4096 on Electronics).
2. Positive-Aware NLGCL loss (v4b conditional ablation):
   - Mitigates false-negative penalty by identifying in-batch collaborative positives from training interactions.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLGCL_Module(nn.Module):
    """Faithful Standalone NLGCL Module.
    
    Reconstructs the exact neighborhood contrastive loss from STAIR-NLGCL reference
    with mathematical identity and verified gradient backpropagation.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            n_users: Total user count.
            n_items: Total item count.
            G: Number of contrastive gaps (default 1: contrast Layer 0 vs Layer 1).
            tau: InfoNCE temperature (default 0.2).
            alpha: Balance between user-side and item-side CL loss (default 0.5).
            chunk_size: Optional chunk size for anchor batching (e.g. 1024) to reduce peak memory.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha
        self.chunk_size = chunk_size

    def info_nce_in_batch(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """In-batch InfoNCE loss.
        
        Args:
            anchor: (B, D) — query vectors
            positive: (B, D) — positive key for each query
            negatives: (B, D) — in-batch negative pool
            
        Returns:
            Scalar loss tensor.
        """
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        B = anchor.size(0)
        chunk = self.chunk_size

        if chunk is not None and chunk > 0 and B > chunk:
            # Memory-bounded chunked computation
            total_loss = 0.0
            neg_t = negatives.t()  # (D, B)
            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                a_chunk = anchor[start:end]      # (c, D)
                p_chunk = positive[start:end]    # (c, D)

                pos_sim = (a_chunk * p_chunk).sum(dim=-1) / self.tau  # (c,)
                neg_sim = torch.mm(a_chunk, neg_t) / self.tau         # (c, B)
                loss_chunk = -pos_sim + torch.logsumexp(neg_sim, dim=-1)
                total_loss = total_loss + loss_chunk.sum()
            return total_loss / B
        else:
            # Exact unchunked 2D GEMM
            pos_sim = (anchor * positive).sum(dim=-1) / self.tau  # (B,)
            neg_sim = torch.mm(anchor, negatives.t()) / self.tau  # (B, B)
            loss = -pos_sim + torch.logsumexp(neg_sim, dim=-1)
            return loss.mean()

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """Computes heterogeneous in-batch contrastive loss across FSC intermediate layers.
        
        Args:
            layer_embeds: List of joint embeddings [H^0, H^1, ..., H^L], each of shape (N_users + N_items, D).
            users: (B,) user indices.
            pos_items: (B,) positive item indices.
            
        Returns:
            Scalar loss tensor.
        """
        users = users.view(-1)
        pos_items = pos_items.view(-1)
        device = layer_embeds[0].device
        total_loss = torch.tensor(0.0, device=device)

        num_gaps = min(self.G, len(layer_embeds) - 1)
        if num_gaps <= 0:
            return total_loss

        for g in range(num_gaps):
            U_g, I_g = torch.split(layer_embeds[g], [self.n_users, self.n_items])
            U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])

            # User-side CL (L_u):
            # anchor: I_{g+1}[pos_items] (propagated item)
            # positive: U_g[users] (ego user)
            # negatives: U_g[users] (all batch users)
            cl_u = self.info_nce_in_batch(
                anchor=I_g1[pos_items],
                positive=U_g[users],
                negatives=U_g[users],
            )

            # Item-side CL (L_i):
            # anchor: U_{g+1}[users] (propagated user)
            # positive: I_g[pos_items] (ego item)
            # negatives: I_g[pos_items] (all batch items)
            cl_i = self.info_nce_in_batch(
                anchor=U_g1[users],
                positive=I_g[pos_items],
                negatives=I_g[pos_items],
            )

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        return total_loss / num_gaps


class NLGCL_PositiveAware_Module(nn.Module):
    """Positive-Aware NLGCL Module (v4b conditional ablation).
    
    When an in-batch user has interacted with other in-batch items during training,
    treating those items as strict negatives introduces false-negative penalties.
    This module identifies known train interactions within the batch and averages
    positive log-probabilities over all verified positives in the batch.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        train_pair_keys: torch.Tensor,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha
        self.register_buffer("train_pair_keys", train_pair_keys.long(), persistent=False)

    def _get_positive_mask(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Determines binary membership of user-item pairs in the train interaction set.
        
        Returns:
            (B_u, B_i) boolean tensor where mask[u, i] is True iff (users[u], items[i]) is a train pair.
        """
        device = users.device
        keys = self.train_pair_keys.to(device)
        queries = users[:, None] * self.n_items + items[None, :]
        if keys.numel() == 0:
            return torch.zeros_like(queries, dtype=torch.bool)
        positions = torch.searchsorted(keys, queries)
        valid = positions < len(keys)
        return valid & (keys[positions.clamp_max(len(keys) - 1)] == queries)

    def positive_aware_info_nce(
        self,
        anchor: torch.Tensor,
        keys: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Positive-aware InfoNCE:
        For query a, L_a = - 1/|P(a)| sum_{k in P(a)} log [ exp(sim(a,k)/tau) / sum_j exp(sim(a,j)/tau) ]
        = logsumexp(sim(a, :)/tau) - 1/|P(a)| sum_{k in P(a)} (sim(a, k)/tau).
        """
        anchor = F.normalize(anchor, p=2, dim=-1)
        keys = F.normalize(keys, p=2, dim=-1)

        sim_matrix = torch.mm(anchor, keys.t()) / self.tau  # (B, B)
        log_denom = torch.logsumexp(sim_matrix, dim=-1)      # (B,)

        # Compute positive sum
        pos_sim_sum = (sim_matrix * pos_mask.float()).sum(dim=-1)  # (B,)
        pos_count = pos_mask.sum(dim=-1).clamp_min(1.0)           # (B,)

        loss = log_denom - (pos_sim_sum / pos_count)
        return loss.mean()

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        users = users.view(-1)
        pos_items = pos_items.view(-1)
        device = layer_embeds[0].device
        total_loss = torch.tensor(0.0, device=device)

        num_gaps = min(self.G, len(layer_embeds) - 1)
        if num_gaps <= 0:
            return total_loss

        # Mask: shape (B, B) — mask[b, r] = True if (users[r], pos_items[b]) is in train
        # User-side: anchor is I_g1[pos_items], keys are U_g[users].
        # anchor b is item pos_items[b], key r is user users[r].
        u_pos_mask = self._get_positive_mask(users, pos_items).t()  # (B, B) where [b, r] is item b vs user r
        # Ensure diagonal is always True (the sampled pair is always positive)
        diag_idx = torch.arange(users.size(0), device=device)
        u_pos_mask[diag_idx, diag_idx] = True

        # Item-side: anchor is U_g1[users], keys are I_g[pos_items].
        # anchor b is user users[b], key r is item pos_items[r].
        i_pos_mask = self._get_positive_mask(users, pos_items)      # (B, B) where [b, r] is user b vs item r
        i_pos_mask[diag_idx, diag_idx] = True

        for g in range(num_gaps):
            U_g, I_g = torch.split(layer_embeds[g], [self.n_users, self.n_items])
            U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])

            cl_u = self.positive_aware_info_nce(
                anchor=I_g1[pos_items],
                keys=U_g[users],
                pos_mask=u_pos_mask,
            )
            cl_i = self.positive_aware_info_nce(
                anchor=U_g1[users],
                keys=I_g[pos_items],
                pos_mask=i_pos_mask,
            )

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        return total_loss / num_gaps
