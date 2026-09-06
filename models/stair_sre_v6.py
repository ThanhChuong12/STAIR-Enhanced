"""
models/stair_sre_v6.py — STAIR-SRE v6 Module
==============================================
Stepwise Spectral-Refined Contrastive Learning

Phase 3 — Dot 1: Core Architecture with 4 Pillars

Pillar 1: Diagonal Spectral-scaling Projector (0-rotation)
   E_proj = E_svd ⊙ w   where w ∈ ℝᴰ is a learnable parameter.
   Equivalent to diag(w) — only rescales per-dimension variance,
   absolutely NO rotation of the SVD spectral coordinate system.
   Initialised as ones so epoch 0 is identical to baseline.

Pillar 2: Soft Spectral Swapping (Hard Negative Generation)
   Instead of hard-splitting [0:32] / [32:64], we draw a per-dimension
   Bernoulli swap mask with probability p_swap[j] = 1 - β[j]:
     - Low-freq CF dims (j~0, β~0.9) -> p_swap ~ 0.1 (almost never swapped)
     - High-freq MM dims (j~63, β~0.0) -> p_swap ~ 1.0 (almost always swapped)
   Hard negative: i_hard = i⁺ ⊙ (1 - m) + i_rolled ⊙ m

Pillar 3: Adaptive False Negative Attenuation
   Replaces v5 binary hard mask with smooth attenuation coefficient
   placed OUTSIDE the exp() in the InfoNCE denominator:
     (1 - W_{u,k}) · exp(sim(u, i_k) / τ)
   Supports optional threshold τ_atten (default: 0.0 = pure 1-W, 0.35 = selective).
   When W -> 1 (false negative), attenuation -> 0 (no repulsion).
   When W -> 0 (true negative), attenuation -> 1 (full repulsion).

Pillar 4: Hierarchical Layer-wise Contrastive Alignment
   Preserves the NLGCL multi-layer natural contrastive framework from v4/v5.
   Contrasts layer g (user) <-> layer g+1 (item) for g ∈ {0, ..., G-1}.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DiagonalSpectralProjector', 'StepwiseSRELoss', 'StepwiseSREv2Loss']


class DiagonalSpectralProjector(nn.Module):
    """
    Zero-rotation spectral projector.

    Applies element-wise learned scaling to the SVD-whitened embeddings:
        E_proj = E_svd ⊙ w

    The Jacobian is strictly diagonal: J_{jk} = 0 for j != k.
    This preserves the monotonic spectral ordering of STAIR's FSC/BSC filters.

    Args:
        dim: Embedding dimension (default: 64)
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        # Initialise as identity scaling — epoch 0 matches baseline exactly
        self.w = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hadamard product: E_proj = E_svd ⊙ w"""
        return x * self.w


class StepwiseSRELoss(nn.Module):
    """
    Stepwise Spectral-Refined Contrastive Loss (STAIR-SRE v6).

    Integrates:
      - Soft Spectral Swapping for hard negative generation
      - Adaptive False Negative Attenuation (smooth, outside exp)
      - Layer-wise natural contrastive alignment (NLGCL heritage)
      - Spectral noise injection (inherited from v5)

    Args:
        n_users:    Number of users in the dataset.
        n_items:    Number of items in the dataset.
        beta:       (D,) spectral propagation vector (1.0 - beta3).
        G:          Number of contrastive layer gaps (default: 1).
        tau:        InfoNCE temperature (default: 0.2).
        alpha:      Balance between user-CL and item-CL (default: 0.5).
        eps:        Spectral noise amplitude epsilon (default: 0.1).
        tau_atten:  Attenuation threshold (default: 0.0 = pure 1-W, 0.35 = selective).
        debug:      Enable diagnostic logging for first few batches.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        beta: torch.Tensor,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
        eps: float = 0.1,
        tau_atten: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha
        self.eps = eps
        self.tau_atten = tau_atten
        self.debug = debug
        self._debug_step = 0

        # Register spectral vectors as non-trainable buffers
        self.register_buffer('beta_buf', beta.clone().detach())
        # Swap probability: inversely proportional to collaborative weight
        self.register_buffer('prob_swap', torch.clamp(1.0 - beta, 0.0, 1.0))

    # --- Spectral Noise Injection (inherited from v5) ---

    def inject_spectral_noise(
        self, h: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral-decayed sign-preserving noise injection.
        h_tilde = h + eps * (beta * sign(h) * (eta / ||eta||_2))
        """
        if not self.training or self.eps <= 0.0:
            return h
        noise = torch.randn_like(h)
        noise = F.normalize(noise, p=2, dim=-1)
        beta_w = beta.to(h.device)
        beta_w = beta_w.unsqueeze(0) if beta_w.dim() == 1 else beta_w
        return h + self.eps * (beta_w * torch.sign(h) * noise)

    # --- Soft Spectral Swapping ---

    def create_spectral_hard_negatives(
        self, i_norm: torch.Tensor, beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate hard negatives via Soft Spectral Swapping.

        For each item in the batch, creates a challenging negative by
        mixing its collaborative dimensions (low-freq, low swap prob)
        with multimodal dimensions (high-freq, high swap prob) from
        a different item obtained via circular roll.

        Args:
            i_norm: (B, D) L2-normalised item representations
            beta:   Optional (D,) spectral propagation vector
        Returns:
            i_hard_neg: (B, D) hard negative representations (re-normalised)
        """
        if i_norm.size(0) <= 1:
            return i_norm

        # Circular shift to obtain a different item for each position
        i_rolled = torch.roll(i_norm, shifts=1, dims=0)

        # Draw per-dimension Bernoulli mask from spectral swap probabilities
        # prob_swap[j] = 1 - beta[j]: high for multimodal dims, low for CF dims
        if beta is not None:
            prob_swap = torch.clamp(1.0 - beta, 0.0, 1.0).to(i_norm.device)
        else:
            prob_swap = self.prob_swap.to(i_norm.device)

        swap_mask = torch.bernoulli(
            prob_swap.expand(i_norm.size(0), -1)
        )  # (B, D)

        # Composite: keep CF dims from i+, swap MM dims from i_rolled
        i_hard_neg = i_norm * (1.0 - swap_mask) + i_rolled * swap_mask

        # Re-normalise to unit sphere after mixing
        return F.normalize(i_hard_neg, p=2, dim=-1)

    # --- Core Forward: Layer-wise SRE Contrastive Loss ---

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        positives: torch.Tensor,
        beta: torch.Tensor,
        user_profiles: Optional[torch.Tensor] = None,
        item_modals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the STAIR-SRE contrastive loss across layer pairs.

        Args:
            layer_embeds:  List of (N_u + N_i, D) tensors for layers 0..L.
            users:         (B,) user indices in current mini-batch.
            positives:     (B,) positive item indices in current mini-batch.
            beta:          (D,) spectral propagation vector (1.0 - beta3).
            user_profiles: (B, D) optional user interaction-weighted profiles.
            item_modals:   (B, D) optional item whitened modal features.

        Returns:
            Scalar contrastive loss tensor.
        """
        users = users.view(-1)
        positives = positives.view(-1)

        total_loss = torch.tensor(0.0, device=layer_embeds[0].device)
        num_gaps = min(self.G, len(layer_embeds) - 1)
        if num_gaps <= 0:
            return total_loss

        batch_size = users.size(0)
        device = layer_embeds[0].device

        # -----------------------------------------------------------------
        # 1. Compute Adaptive False Negative Attenuation Weights
        # -----------------------------------------------------------------
        # W_multi[b, k] = cosine(user_profile_b, item_modal_k)
        # If tau_atten > 0:
        #   W_eff = clamp((W_multi - tau_atten) / (1 - tau_atten), 0, 1)
        # Else:
        #   W_eff = clamp(W_multi, 0, 1)
        # Attenuation[b, k] = 1 - W_eff
        #   -> 0 for false negatives (W->1): suppress repulsion
        #   -> 1 for true negatives  (W->0): full repulsion

        if user_profiles is not None and item_modals is not None:
            with torch.no_grad():
                u_prof_norm = F.normalize(user_profiles, p=2, dim=-1)
                i_mod_norm = F.normalize(item_modals, p=2, dim=-1)
                W_multi = torch.matmul(u_prof_norm, i_mod_norm.t())  # (B, B)
                if self.tau_atten > 0.0:
                    W_eff = torch.clamp(
                        (W_multi - self.tau_atten) / (1.0 - self.tau_atten + 1e-8),
                        0.0, 1.0
                    )
                else:
                    W_eff = torch.clamp(W_multi, 0.0, 1.0)
                attenuation = 1.0 - W_eff  # (B, B)

                # Diagnostic logging
                if self.debug and self._debug_step < 3 and self.training:
                    self._debug_step += 1
                    off_diag = ~torch.eye(
                        batch_size, dtype=torch.bool, device=device
                    )
                    off_w = W_multi[off_diag]
                    thresh_val = self.tau_atten if self.tau_atten > 0.0 else 0.35
                    high_w_cnt = (off_w > thresh_val).sum().item()
                    total_off = off_diag.sum().item()
                    print(
                        f"[SRE Debug Batch {self._debug_step}] "
                        f"W_multi: min={off_w.min():.4f}, "
                        f"max={off_w.max():.4f}, "
                        f"mean={off_w.mean():.4f} | "
                        f"High-W (>{thresh_val:.2f}): {high_w_cnt}/{total_off} "
                        f"({high_w_cnt/max(total_off, 1)*100:.3f}%)"
                    )
        else:
            # No FN attenuation — all negatives treated equally
            attenuation = torch.ones(
                (batch_size, batch_size), device=device
            )

        # Diagonal mask: exclude positive pair from negative sum
        diag_mask = 1.0 - torch.eye(batch_size, device=device)

        # -----------------------------------------------------------------
        # 2. Multi-Gap Cross-Entity Contrastive Loop
        # -----------------------------------------------------------------
        for g in range(num_gaps):
            # Extract user/item representations at layers g and g+1
            U_g, I_g = torch.split(
                layer_embeds[g], [self.n_users, self.n_items]
            )
            U_g1, I_g1 = torch.split(
                layer_embeds[g + 1], [self.n_users, self.n_items]
            )

            u_g = U_g[users]          # (B, D) users at layer g
            i_g1 = I_g1[positives]    # (B, D) pos items at layer g+1
            i_g = I_g[positives]      # (B, D) pos items at layer g
            u_g1 = U_g1[users]        # (B, D) users at layer g+1

            # --- User-side CL: U_g[u] <-> I_{g+1}[i] ---
            u_g_tilde = self.inject_spectral_noise(u_g, beta)
            i_g1_tilde = self.inject_spectral_noise(i_g1, beta)

            u_g_norm = F.normalize(u_g_tilde, p=2, dim=-1)
            i_g1_norm = F.normalize(i_g1_tilde, p=2, dim=-1)

            # Positive similarity: (B,)
            pos_sim_u = (u_g_norm * i_g1_norm).sum(dim=-1) / self.tau
            pos_exp_u = torch.exp(pos_sim_u)

            # Hard negative via Soft Spectral Swapping: (B, D)
            i_hard_neg = self.create_spectral_hard_negatives(i_g1_norm, beta)
            hard_sim_u = (u_g_norm * i_hard_neg).sum(dim=-1) / self.tau
            hard_exp_u = torch.exp(hard_sim_u)

            # All in-batch item similarities: (B, B)
            all_sim_u = torch.matmul(u_g_norm, i_g1_norm.t()) / self.tau
            all_exp_u = torch.exp(all_sim_u)

            # Apply attenuation weights (1 - W) OUTSIDE exp, mask diagonal
            neg_sum_u = (all_exp_u * attenuation * diag_mask).sum(dim=1)

            # InfoNCE denominator
            denom_u = pos_exp_u + hard_exp_u + neg_sum_u + 1e-8
            loss_u = -torch.log(pos_exp_u / denom_u).mean()

            # --- Item-side CL: I_g[i] <-> U_{g+1}[u] ---
            i_g_tilde = self.inject_spectral_noise(i_g, beta)
            u_g1_tilde = self.inject_spectral_noise(u_g1, beta)

            i_g_norm = F.normalize(i_g_tilde, p=2, dim=-1)
            u_g1_norm = F.normalize(u_g1_tilde, p=2, dim=-1)

            pos_sim_i = (i_g_norm * u_g1_norm).sum(dim=-1) / self.tau
            pos_exp_i = torch.exp(pos_sim_i)

            # Hard negative for item-side (swap user dims)
            u_hard_neg = self.create_spectral_hard_negatives(u_g1_norm, beta)
            hard_sim_i = (i_g_norm * u_hard_neg).sum(dim=-1) / self.tau
            hard_exp_i = torch.exp(hard_sim_i)

            all_sim_i = torch.matmul(i_g_norm, u_g1_norm.t()) / self.tau
            all_exp_i = torch.exp(all_sim_i)

            # Transpose attenuation for item->user direction
            neg_sum_i = (all_exp_i * attenuation.t() * diag_mask).sum(dim=1)

            denom_i = pos_exp_i + hard_exp_i + neg_sum_i + 1e-8
            loss_i = -torch.log(pos_exp_i / denom_i).mean()

            # Symmetric combination
            total_loss = total_loss + self.alpha * loss_u + (1.0 - self.alpha) * loss_i

        # Normalise by number of layer gaps
        return total_loss / float(num_gaps)


# Alias for compatibility with STAIR3_v1_Report naming convention
StepwiseSREv2Loss = StepwiseSRELoss
