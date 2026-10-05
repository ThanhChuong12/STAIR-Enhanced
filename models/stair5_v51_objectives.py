"""models/stair5_v51_objectives.py — STAIR5-v5.1 Objectives: NLGCL-KPE.
========================================================================
Faithful implementation of Known-Positive Exclusion (KPE), Deduplicated
InfoNCE (DEDUP), Multi-Positive Match (MP-MATCH), Grouped Occurrence Reference (REF),
and Mask-Placebo controls for Neighborhood-enriched Graph Contrastive Learning.

Key mathematical properties:
1. Two-sided contrastive directions:
   - User-side: query I_{g+1}[pos_items[b]] vs unique keys U_g[K_U]
   - Item-side: query U_{g+1}[users[b]] vs unique keys I_g[K_I]
2. Query occurrences are NOT deduplicated (preserves full B sampled occurrences).
3. Keys are deduplicated to unique entities in-batch (reduces GEMM from BxB to Bxm).
4. KPE Denominator: excludes all in-batch known train-positives other than the
   designated target from the InfoNCE denominator:
     A_{bk} = 1[k == t_b] or not M_{bk}
     ell_b^{KPE} = log sum_{k: A_{bk}=1} exp(z_{bk}) - z_{b, t_b}
5. Exact zero loss and zero direct gradient when a row has no admissible negatives.
6. Gradient checkpointing on anchor chunks: recomputes chunk GEMM and membership
   on the fly during backward; both query and key embeddings receive valid gradients.
"""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.stair5_v51_utils import gpu_membership


OBJECTIVE_MODES = ("KPE", "DEDUP", "MP-MATCH", "V4", "REF", "MASK-PLACEBO")


def _info_nce_sum(anchor: torch.Tensor, positive: torch.Tensor, keys: torch.Tensor, tau: float) -> torch.Tensor:
    """Original V4 un-deduplicated occurrence InfoNCE sum."""
    pos_logits = (anchor * positive).sum(dim=-1) / tau
    logits = anchor @ keys.t() / tau
    return (torch.logsumexp(logits, dim=-1) - pos_logits).sum()


def _chunk_v4_forward(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    keys: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    return _info_nce_sum(anchor, positive, keys, tau)


def _chunk_grouped_ref_forward(
    anchor_chunk: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    log_counts: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Grouped occurrence reference (Eq. 7 in STAIR5-v5.1 Report).

    ell_b^{ref} = LSE_k(z_{bk} + log n_k) - z_{b, t_b}
    Mathematically identical to occurrence InfoNCE on shared embeddings.
    """
    logits = anchor_chunk @ keys.t() / tau
    pos_logits = logits[torch.arange(logits.size(0), device=logits.device), targets]
    weighted_logits = logits + log_counts[None, :]
    lse = torch.logsumexp(weighted_logits, dim=-1)
    return (lse - pos_logits).sum()


def _chunk_dedup_forward(
    anchor_chunk: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """DEDUP control: Unique-key InfoNCE without occurrence weighting or positive exclusion."""
    logits = anchor_chunk @ keys.t() / tau
    pos_logits = logits[torch.arange(logits.size(0), device=logits.device), targets]
    lse = torch.logsumexp(logits, dim=-1)
    return (lse - pos_logits).sum()


def _chunk_mp_match_forward(
    anchor_chunk: torch.Tensor,
    keys: torch.Tensor,
    query_ids: torch.Tensor,
    unique_key_ids: torch.Tensor,
    train_index: torch.Tensor,
    n_items: int,
    is_user_side: bool,
    tau: float,
) -> torch.Tensor:
    """MP-MATCH control (Eq. 1 in Report): Uniform multi-positive attraction on unique keys.

    ell_b^{MP} = log sum_k exp(z_{bk}) - (1 / |P_b|) sum_{p in P_b} z_{bp}
    """
    logits = anchor_chunk @ keys.t() / tau
    if is_user_side:
        pairs = unique_key_ids[None, :] * n_items + query_ids[:, None]
    else:
        pairs = query_ids[:, None] * n_items + unique_key_ids[None, :]
    is_pos = gpu_membership(train_index, pairs)
    lse = torch.logsumexp(logits, dim=-1)
    pos_count = is_pos.sum(dim=-1).clamp(min=1).to(logits.dtype)
    pos_mean = (logits * is_pos.to(logits.dtype)).sum(dim=-1) / pos_count
    return (lse - pos_mean).sum()


def _chunk_kpe_forward(
    anchor_chunk: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    query_ids: torch.Tensor,
    unique_key_ids: torch.Tensor,
    train_index: torch.Tensor,
    n_items: int,
    is_user_side: bool,
    tau: float,
) -> torch.Tensor:
    """KPE forward on an anchor chunk (Eq. 5 in Report).

    Recomputes chunk logits and train-pair membership dynamically.
    Excludes other known positives from the denominator while strictly
    preserving the designated positive target.
    """
    logits = anchor_chunk @ keys.t() / tau
    batch_c = logits.size(0)
    row_idx = torch.arange(batch_c, device=logits.device)

    # Compute pair keys for the chunk
    if is_user_side:
        pairs = unique_key_ids[None, :] * n_items + query_ids[:, None]
    else:
        pairs = query_ids[:, None] * n_items + unique_key_ids[None, :]

    is_known_pos = gpu_membership(train_index, pairs)  # (C, m) bool

    # Admissible mask: target is always allowed; other known positives are excluded
    admissible = ~is_known_pos
    admissible[row_idx, targets] = True

    pos_logits = logits[row_idx, targets]

    # Mask out excluded positives with -1e9 (clean FP32 underflow with zero direct gradient)
    masked_logits = logits.masked_fill(~admissible, -1e9)
    lse = torch.logsumexp(masked_logits, dim=-1)

    # For any row with no admissible negatives (only target admissible),
    # masked_logits has exactly one non-masked entry pos_logits[c],
    # so lse == pos_logits[c] and (lse - pos_logits) == 0.0.
    return (lse - pos_logits).sum()


def _chunk_mask_placebo_forward(
    anchor_chunk: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    query_ids: torch.Tensor,
    unique_key_ids: torch.Tensor,
    train_index: torch.Tensor,
    n_items: int,
    is_user_side: bool,
    tau: float,
    seed: int,
) -> torch.Tensor:
    """MASK-PLACEBO control: Masks the exact same number of non-target keys per row
    as KPE, but selected at pseudo-random without using true positive identities.
    """
    logits = anchor_chunk @ keys.t() / tau
    batch_c, m = logits.shape
    row_idx = torch.arange(batch_c, device=logits.device)

    if is_user_side:
        pairs = unique_key_ids[None, :] * n_items + query_ids[:, None]
    else:
        pairs = query_ids[:, None] * n_items + unique_key_ids[None, :]

    is_known_pos = gpu_membership(train_index, pairs)
    # Number of excluded keys per row in true KPE
    n_excluded = (is_known_pos.sum(dim=-1) - 1).clamp(min=0)  # (C,)

    # Create random scores to select placebo excluded keys
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rand_scores = torch.rand((batch_c, m), generator=gen, device="cpu").to(logits.device)
    # Protect target from being masked
    rand_scores[row_idx, targets] = 2.0

    admissible = torch.ones((batch_c, m), dtype=torch.bool, device=logits.device)
    for c in range(batch_c):
        k_exc = int(n_excluded[c].item())
        if k_exc > 0 and k_exc < m:
            # Mask out the k_exc keys with lowest random scores
            _, mask_indices = torch.topk(rand_scores[c], k_exc, largest=False)
            admissible[c, mask_indices] = False

    admissible[row_idx, targets] = True
    pos_logits = logits[row_idx, targets]
    masked_logits = logits.masked_fill(~admissible, -1e9)
    lse = torch.logsumexp(masked_logits, dim=-1)
    return (lse - pos_logits).sum()


class STAIR5_v51_NLGCL_Module(nn.Module):
    """Production STAIR5-v5.1 NLGCL Module supporting NLGCL-KPE, DEDUP, MP-MATCH,
    Grouped Reference (REF), MASK-PLACEBO, and V4 occurrence baseline.

    Memory bounded via chunked gradient checkpointing.
    No full B x B x d allocations or persistent B x m lookup matrices.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        train_pair_keys: torch.Tensor,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
        rho: float = 1.0,
        mode: str = "KPE",
        chunk_size: Optional[int] = 1024,
        placebo_seed: int = 1,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0:
            raise ValueError("Entity counts must be positive.")
        if G < 1 or int(G) != G:
            raise ValueError("G must be an integer >= 1.")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and > 0.")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1].")
        if mode not in OBJECTIVE_MODES:
            raise ValueError(f"mode must be one of {OBJECTIVE_MODES}")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.G = int(G)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.mode = mode
        self.chunk_size = chunk_size if (chunk_size and chunk_size > 0) else None
        self.placebo_seed = int(placebo_seed)

        # Register train pair keys as a non-persistent buffer on CPU/GPU
        keys = torch.unique(train_pair_keys.detach(), sorted=True).long()
        if keys.numel() and (keys.min() < 0 or keys.max() >= n_users * n_items):
            raise ValueError("Train pair keys contain out-of-range IDs.")
        self.register_buffer("train_pair_keys", keys, persistent=False)

    def _execute_chunk(
        self,
        anchor: torch.Tensor,
        keys: torch.Tensor,
        targets: torch.Tensor,
        query_ids: torch.Tensor,
        unique_key_ids: torch.Tensor,
        log_counts: torch.Tensor,
        is_user_side: bool,
    ) -> torch.Tensor:
        """Executes loss computation on an anchor chunk using gradient checkpointing."""
        batch = anchor.size(0)
        chunk = self.chunk_size or batch

        def _run_forward(a_chunk, q_ids, tgts):
            if self.mode == "KPE":
                if self.rho == 1.0:
                    return _chunk_kpe_forward(
                        a_chunk, keys, tgts, q_ids, unique_key_ids,
                        self.train_pair_keys, self.n_items, is_user_side, self.tau
                    )
                elif self.rho == 0.0:
                    return _chunk_grouped_ref_forward(a_chunk, keys, tgts, log_counts, self.tau)
                else:
                    kpe_loss = _chunk_kpe_forward(
                        a_chunk, keys, tgts, q_ids, unique_key_ids,
                        self.train_pair_keys, self.n_items, is_user_side, self.tau
                    )
                    ref_loss = _chunk_grouped_ref_forward(a_chunk, keys, tgts, log_counts, self.tau)
                    return (1.0 - self.rho) * ref_loss + self.rho * kpe_loss
            elif self.mode == "DEDUP":
                return _chunk_dedup_forward(a_chunk, keys, tgts, self.tau)
            elif self.mode == "MP-MATCH":
                return _chunk_mp_match_forward(
                    a_chunk, keys, q_ids, unique_key_ids,
                    self.train_pair_keys, self.n_items, is_user_side, self.tau
                )
            elif self.mode == "REF":
                return _chunk_grouped_ref_forward(a_chunk, keys, tgts, log_counts, self.tau)
            elif self.mode == "MASK-PLACEBO":
                return _chunk_mask_placebo_forward(
                    a_chunk, keys, tgts, q_ids, unique_key_ids,
                    self.train_pair_keys, self.n_items, is_user_side, self.tau, self.placebo_seed
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        if batch <= chunk:
            return _run_forward(anchor, query_ids, targets)

        total = anchor.new_zeros(())
        for start in range(0, batch, chunk):
            stop = min(start + chunk, batch)
            a_sub = anchor[start:stop]
            q_sub = query_ids[start:stop]
            t_sub = targets[start:stop]

            if torch.is_grad_enabled() and (a_sub.requires_grad or keys.requires_grad):
                val = checkpoint(
                    _run_forward, a_sub, q_sub, t_sub,
                    use_reentrant=False, preserve_rng_state=False
                )
            else:
                val = _run_forward(a_sub, q_sub, t_sub)
            total = total + val
        return total

    def _direction_loss(
        self,
        anchor_embeds: torch.Tensor,
        key_embeds: torch.Tensor,
        query_ids: torch.Tensor,
        unique_key_ids: torch.Tensor,
        targets: torch.Tensor,
        log_counts: torch.Tensor,
        is_user_side: bool,
    ) -> torch.Tensor:
        """Computes contrastive loss for one direction (user-side or item-side).

        Args:
            anchor_embeds: (B, D) L2-normalized query representations.
            key_embeds: (m, D) L2-normalized unique key representations.
            query_ids: (B,) entity IDs of the queries.
            unique_key_ids: (m,) entity IDs of the unique keys.
            targets: (B,) indices of the positive keys in unique_key_ids.
            log_counts: (m,) log(count) of each key in the batch.
            is_user_side: True if user-side CL, False if item-side CL.

        Returns:
            Mean loss scalar over the B query rows.
        """
        B = anchor_embeds.size(0)
        if B == 0:
            return anchor_embeds.new_zeros(())

        loss_sum = self._execute_chunk(
            anchor=anchor_embeds,
            keys=key_embeds,
            targets=targets,
            query_ids=query_ids,
            unique_key_ids=unique_key_ids,
            log_counts=log_counts,
            is_user_side=is_user_side,
        )
        return loss_sum / B

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        pos_items: torch.Tensor,
    ) -> torch.Tensor:
        """Computes STAIR5-v5.1 NLGCL contrastive loss across FSC intermediate layers.

        Args:
            layer_embeds: List of joint representations [H^0, H^1, ..., H^L],
                          each of shape (N_users + N_items, D).
            users: (B,) user indices of sampled train pairs.
            pos_items: (B,) positive item indices of sampled train pairs.

        Returns:
            Scalar loss tensor.
        """
        users = users.view(-1)
        pos_items = pos_items.view(-1)
        B = users.numel()
        if B == 0 or B != pos_items.numel():
            raise ValueError("NLGCL requires equally sized, nonempty sampled pairs.")

        total_loss = layer_embeds[0].new_zeros(())
        num_gaps = min(self.G, len(layer_embeds) - 1)
        if num_gaps <= 0:
            return total_loss

        # V4 fast-path exact fallback
        if self.mode == "V4":
            for g in range(num_gaps):
                U_g, I_g = torch.split(layer_embeds[g], [self.n_users, self.n_items])
                U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])

                # User-side: anchor I_{g+1}[pos_items], keys U_g[users]
                anc_u = F.normalize(I_g1[pos_items], p=2, dim=-1)
                keys_u = F.normalize(U_g[users], p=2, dim=-1)
                chunk = self.chunk_size or B
                if B <= chunk:
                    cl_u = _info_nce_sum(anc_u, keys_u, keys_u, self.tau) / B
                else:
                    tot_u = anc_u.new_zeros(())
                    for s in range(0, B, chunk):
                        e = min(s + chunk, B)
                        args = (anc_u[s:e], keys_u[s:e], keys_u, self.tau)
                        v = checkpoint(_chunk_v4_forward, *args, use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else _chunk_v4_forward(*args)
                        tot_u = tot_u + v
                    cl_u = tot_u / B

                # Item-side: anchor U_{g+1}[users], keys I_g[pos_items]
                anc_i = F.normalize(U_g1[users], p=2, dim=-1)
                keys_i = F.normalize(I_g[pos_items], p=2, dim=-1)
                if B <= chunk:
                    cl_i = _info_nce_sum(anc_i, keys_i, keys_i, self.tau) / B
                else:
                    tot_i = anc_i.new_zeros(())
                    for s in range(0, B, chunk):
                        e = min(s + chunk, B)
                        args = (anc_i[s:e], keys_i[s:e], keys_i, self.tau)
                        v = checkpoint(_chunk_v4_forward, *args, use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else _chunk_v4_forward(*args)
                        tot_i = tot_i + v
                    cl_i = tot_i / B

                total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i
            return total_loss / num_gaps

        # Deduplicate keys while strictly preserving B query rows
        unique_users, u_inverse, u_counts = torch.unique(users, return_inverse=True, return_counts=True)
        unique_items, i_inverse, i_counts = torch.unique(pos_items, return_inverse=True, return_counts=True)

        log_counts_u = torch.log(u_counts.float())
        log_counts_i = torch.log(i_counts.float())

        for g in range(num_gaps):
            U_g, I_g = torch.split(layer_embeds[g], [self.n_users, self.n_items])
            U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])

            # Normalize representations once per gap
            # User-side CL:
            # Query anchor: propagated item I_{g+1}[pos_items] (size B x D)
            # Keys: ego user representations U_g[unique_users] (size m_U x D)
            # Target for query b: index of users[b] in unique_users, which is u_inverse[b]
            query_u = F.normalize(I_g1[pos_items], p=2, dim=-1)
            keys_u = F.normalize(U_g[unique_users], p=2, dim=-1)
            cl_u = self._direction_loss(
                anchor_embeds=query_u,
                key_embeds=keys_u,
                query_ids=pos_items,
                unique_key_ids=unique_users,
                targets=u_inverse,
                log_counts=log_counts_u,
                is_user_side=True,
            )

            # Item-side CL:
            # Query anchor: propagated user U_{g+1}[users] (size B x D)
            # Keys: ego item representations I_g[unique_items] (size m_I x D)
            # Target for query b: index of pos_items[b] in unique_items, which is i_inverse[b]
            query_i = F.normalize(U_g1[users], p=2, dim=-1)
            keys_i = F.normalize(I_g[unique_items], p=2, dim=-1)
            cl_i = self._direction_loss(
                anchor_embeds=query_i,
                key_embeds=keys_i,
                query_ids=users,
                unique_key_ids=unique_items,
                targets=i_inverse,
                log_counts=log_counts_i,
                is_user_side=False,
            )

            total_loss = total_loss + self.alpha * cl_u + (1.0 - self.alpha) * cl_i

        return total_loss / num_gaps
