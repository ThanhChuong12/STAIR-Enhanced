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
import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
        if n_users <= 0 or n_items <= 0 or G < 1 or int(G) != G:
            raise ValueError("Require positive entity counts and integer G >= 1.")
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Require finite tau > 0 and alpha in [0, 1].")
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha
        if chunk_size is not None and (chunk_size < 0 or int(chunk_size) != chunk_size):
            raise ValueError("chunk_size must be None or a nonnegative integer (0 disables it).")
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
        if anchor.ndim != 2 or anchor.shape != positive.shape or negatives.ndim != 2:
            raise ValueError("Expected matching nonempty 2D query/positive tensors and 2D keys.")
        if anchor.size(0) == 0 or negatives.size(0) == 0 or anchor.size(1) != negatives.size(1):
            raise ValueError("InfoNCE requires nonempty compatible query/key dimensions.")
        # Reuse normalized keys only when positive and negatives are the same object.
        shared_keys = positive is negatives
        anchor = F.normalize(anchor, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)
        positive = negatives if shared_keys else F.normalize(positive, p=2, dim=-1)
        batch = anchor.size(0)
        chunk = self.chunk_size or batch
        if batch <= chunk:
            return _info_nce_sum(anchor, positive, negatives, self.tau) / batch
        total = anchor.new_zeros(())
        for start in range(0, batch, chunk):
            inputs = (anchor[start:start + chunk], positive[start:start + chunk], negatives, self.tau)
            if torch.is_grad_enabled() and any(x.requires_grad for x in inputs[:3]):
                # Plain chunk loops retain every logits block until backward. Recompute
                # each block instead, retaining both query and key gradient paths.
                value = checkpoint(_info_nce_sum, *inputs, use_reentrant=False, preserve_rng_state=False)
            else:
                value = _info_nce_sum(*inputs)
            total = total + value
        return total / batch

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
        if users.numel() == 0 or users.numel() != pos_items.numel():
            raise ValueError("NLGCL requires equally sized, nonempty sampled pairs.")
        total_loss = layer_embeds[0].new_zeros(())

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
            user_keys = U_g[users]
            cl_u = self.info_nce_in_batch(I_g1[pos_items], user_keys, user_keys)

            # Item-side CL (L_i):
            # anchor: U_{g+1}[users] (propagated user)
            # positive: I_g[pos_items] (ego item)
            # negatives: I_g[pos_items] (all batch items)
            item_keys = I_g[pos_items]
            cl_i = self.info_nce_in_batch(U_g1[users], item_keys, item_keys)

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        return total_loss / num_gaps


def _info_nce_sum(anchor, positive, keys, tau):
    positive_logits = (anchor * positive).sum(dim=-1) / tau
    logits = anchor @ keys.t() / tau
    return (torch.logsumexp(logits, dim=-1) - positive_logits).sum()


def _membership(train_keys, pair_keys):
    if train_keys.numel() == 0:
        return torch.zeros_like(pair_keys, dtype=torch.bool)
    positions = torch.searchsorted(train_keys, pair_keys.contiguous())
    return (positions < train_keys.numel()) & (train_keys[positions.clamp_max(train_keys.numel() - 1)] == pair_keys)


def _positive_sum(anchor, keys, mask, tau):
    logits = anchor @ keys.t() / tau
    count = mask.sum(-1).to(logits.dtype)
    return (torch.logsumexp(logits, -1) - (logits * mask).sum(-1) / count).sum()


class NLGCL_PositiveAware_Module(NLGCL_Module):
    """Conditional v4b: uniform unique queries/keys and mean positive log probability.

    Only deduplicated known train relations define positives. The sampled pair is
    verified rather than silently forced positive. Membership/logits are generated
    per anchor chunk and recomputed in backward; no full B-by-B int64 lookup table.
    """

    def __init__(self, n_users, n_items, train_pair_keys, G=1, tau=0.2, alpha=0.5, chunk_size=1024):
        super().__init__(n_users, n_items, G, tau, alpha, chunk_size)
        if train_pair_keys.dtype != torch.long or train_pair_keys.ndim != 1:
            raise ValueError("train_pair_keys must be a 1D int64 tensor.")
        keys = torch.unique(train_pair_keys.detach(), sorted=True)
        if keys.numel() and (keys.min() < 0 or keys.max() >= n_users * n_items):
            raise ValueError("Train pair keys contain out-of-range IDs.")
        self.register_buffer("train_pair_keys", keys, persistent=False)

    def _get_positive_mask(self, users, items):
        pairs = users[:, None] * self.n_items + items[None, :]
        return _membership(self.train_pair_keys.to(users.device), pairs)

    def positive_aware_info_nce(self, anchor, keys, pos_mask):
        if pos_mask.shape != (len(anchor), len(keys)) or not len(anchor) or not len(keys):
            raise ValueError("Positive mask must match nonempty query/key pools.")
        if not pos_mask.any(-1).all():
            raise ValueError("Every contrastive query must have a known positive.")
        anchor, keys = F.normalize(anchor, dim=-1), F.normalize(keys, dim=-1)
        total = anchor.new_zeros(())
        chunk = self.chunk_size or len(anchor)
        for start in range(0, len(anchor), chunk):
            args = (anchor[start:start + chunk], keys, pos_mask[start:start + chunk], self.tau)
            value = checkpoint(_positive_sum, *args, use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else _positive_sum(*args)
            total = total + value
        return total / len(anchor)

    def _direction(self, anchor, keys, query_ids, key_ids, query_is_item):
        anchor, keys = F.normalize(anchor, dim=-1), F.normalize(keys, dim=-1)
        train_keys = self.train_pair_keys.to(anchor.device)

        def loss_block(a, k, ids):
            if query_is_item:
                pairs = key_ids[None, :] * self.n_items + ids[:, None]
            else:
                pairs = ids[:, None] * self.n_items + key_ids[None, :]
            return _positive_sum(a, k, _membership(train_keys, pairs), self.tau)

        total = anchor.new_zeros(())
        chunk = self.chunk_size or len(anchor)
        for start in range(0, len(anchor), chunk):
            args = (anchor[start:start + chunk], keys, query_ids[start:start + chunk])
            value = checkpoint(loss_block, *args, use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else loss_block(*args)
            total = total + value
        return total / len(anchor)

    def forward(self, layer_embeds, users, pos_items):
        users, pos_items = users.reshape(-1), pos_items.reshape(-1)
        if users.numel() == 0 or users.numel() != pos_items.numel():
            raise ValueError("Expected nonempty sampled user/item pairs of equal length.")
        pairs = users * self.n_items + pos_items
        if not _membership(self.train_pair_keys.to(users.device), pairs).all():
            raise ValueError("Sampled positives must belong to the train split.")
        users, items = torch.unique(users, sorted=True), torch.unique(pos_items, sorted=True)
        total = layer_embeds[0].new_zeros(())
        gaps = min(self.G, len(layer_embeds) - 1)
        for g in range(gaps):
            u0, i0 = torch.split(layer_embeds[g], (self.n_users, self.n_items))
            u1, i1 = torch.split(layer_embeds[g + 1], (self.n_users, self.n_items))
            lu = self._direction(i1[items], u0[users], items, users, True)
            li = self._direction(u1[users], i0[items], users, items, False)
            total = total + self.alpha * lu + (1 - self.alpha) * li
        return total / gaps if gaps else total
