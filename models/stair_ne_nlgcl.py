"""
models/stair_ne_nlgcl.py — STAIR-NE-NLGCL v5 Module
=====================================================
Spectral-Guided Noise-Enhanced Neighborhood-Enriched Graph Contrastive Learning

Core Innovations:
─────────────────
1. Spectral-Decayed Sign-Preserving Noise (from NEGCL & STAIR):
   Injects direction-preserving unit-normalized noise modulated by STAIR's
   spectral propagation curve beta = 1 - beta3:
     h_tilde = h + eps * (beta * sign(h) * (eta / ||eta||_2))
   - Low-frequency CF dimensions (d=0, beta approx 0.9) receive strong perturbation
     to counteract oversmoothing / representations collapse on ultra-sparse graphs.
   - High-frequency multimodal SVD dimensions (d=63, beta approx 0.0) remain pristine,
     safeguarding the modality coordinate anchor.

2. In-batch False Negative Attenuation (from DSCSC / CLID):
   Calculates user-item semantic similarity matrix S in-batch:
     S_{b, k} = Cosine(user_profile_{u_b}, item_modal_{i_k})
   Masks out false negatives where similarity > tau_thresh:
     M_{b, k} = 0 if S_{b, k} > tau_thresh else 1
   Prevents detrimental repulsive forces on semantically compatible unobserved items.

3. Numerically Stable & Zero-Overhead:
   - 100% vectorized (no per-element Python loops).
   - InfoNCE computed via logsumexp with masked fill (-1e9) to prevent NaN/overflow.
   - In-batch computation requires only O(B^2) memory (~16 MB for B=2048), eliminating OOM risks.

References:
  - STAIR: Forward Stepwise Convolution (FSC) & spectral decay beta3 (2024)
  - NEGCL: Knowledge-Based Systems 2025 (sign-preserving noise injection)
  - NLGCL / NLGCL-Plus: In-batch cross-entity contrastive learning
  - DSCSC: False negative attenuation via similarity thresholds
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAIR_NE_NLGCL(nn.Module):
    """
    STAIR-NE-NLGCL v5 contrastive learning module.
    """

    def __init__(
        self,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        G: int = 1,
        tau: float = 0.2,
        alpha: float = 0.5,
        eps: float = 0.1,
        tau_thresh: float = 1.0,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.G = G
        self.tau = tau
        self.alpha = alpha
        self.eps = eps
        self.tau_thresh = tau_thresh

    def inject_spectral_noise(self, h: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Injects spectral-decayed, sign-preserving noise into representation h.
        Args:
            h:    (B, D) node representation
            beta: (D,) spectral decay factor beta = 1 - beta3
        Returns:
            h_perturbed: (B, D) perturbed representation
        """
        if not self.training or self.eps <= 0.0:
            return h

        # 1. Generate standard Gaussian noise and L2-normalize to unit sphere
        noise = torch.randn_like(h)
        noise = F.normalize(noise, p=2, dim=-1)

        # 2. Modulate amplitude with spectral curve beta and preserve quadrant via sign(h)
        beta_weight = beta.unsqueeze(0) if beta.dim() == 1 else beta
        h_perturbed = h + self.eps * (beta_weight * torch.sign(h) * noise)
        return h_perturbed

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
        Forward pass for STAIR-NE-NLGCL contrastive loss.

        Args:
            layer_embeds:  List of per-layer representations [H^0, H^1, ..., H^L]
                           from FSC, each of shape (N_u + N_i, D).
            users:         (B,) user indices in current mini-batch.
            positives:     (B,) positive item indices in current mini-batch.
            beta:          (D,) spectral propagation vector (1.0 - beta3).
            user_profiles: (B, D) optional user interaction-weighted modal profiles.
            item_modals:   (B, D) optional item whitened modal features.

        Returns:
            Scalar InfoNCE contrastive loss tensor.
        """
        users = users.view(-1)
        positives = positives.view(-1)

        total_loss = torch.tensor(0.0, device=layer_embeds[0].device)
        num_gaps = min(self.G, len(layer_embeds) - 1)
        if num_gaps <= 0:
            return total_loss

        batch_size = users.size(0)
        device = layer_embeds[0].device

        # ─────────────────────────────────────────────────────────────────
        # 1. In-batch Semantic Attenuation Mask (Phase 2 False Negative Masking)
        # ─────────────────────────────────────────────────────────────────
        if user_profiles is not None and item_modals is not None and self.tau_thresh < 1.0:
            with torch.no_grad():
                u_norm = F.normalize(user_profiles, p=2, dim=-1)
                i_norm = F.normalize(item_modals, p=2, dim=-1)
                # Cosine similarity matrix: (B, B) where S[b, k] = Cosine(u_b, i_k)
                sim_matrix = torch.matmul(u_norm, i_norm.t())
                # Mask: 0 if similarity > tau_thresh (suspected false negative), 1 otherwise
                mask_u = (sim_matrix <= self.tau_thresh).float()
        else:
            # Phase 1: Keep all in-batch negatives
            mask_u = torch.ones((batch_size, batch_size), device=device)

        diag_idx = torch.arange(batch_size, device=device)

        # ─────────────────────────────────────────────────────────────────
        # 2. Multi-Gap Cross-Entity Contrastive Loop
        # ─────────────────────────────────────────────────────────────────
        for g in range(num_gaps):
            # Extract representations for users and items at layers g and g+1
            if self.n_users is not None and self.n_items is not None:
                U_g, I_g = torch.split(layer_embeds[g], [self.n_users, self.n_items])
                U_g1, I_g1 = torch.split(layer_embeds[g + 1], [self.n_users, self.n_items])
                u_g = U_g[users]
                i_g1 = I_g1[positives]
                i_g = I_g[positives]
                u_g1 = U_g1[users]
            else:
                num_u = layer_embeds[0].size(0) - (item_modals.size(0) if item_modals is not None else 0)
                u_g = layer_embeds[g][users]
                i_g1 = layer_embeds[g + 1][num_u + positives]
                i_g = layer_embeds[g][num_u + positives]
                u_g1 = layer_embeds[g + 1][users]

            # ─────────────────────────────────────────────────────────────
            # User-side CL: Query U_g[u] <-> Positive Key I_{g+1}[i]
            # ─────────────────────────────────────────────────────────────
            # Inject spectral noise
            u_g_tilde = self.inject_spectral_noise(u_g, beta)
            i_g1_tilde = self.inject_spectral_noise(i_g1, beta)

            # L2 normalize onto unit sphere
            u_g_norm = F.normalize(u_g_tilde, p=2, dim=-1)
            i_g1_norm = F.normalize(i_g1_tilde, p=2, dim=-1)

            # Positive similarity: (B,)
            pos_score_u = (u_g_norm * i_g1_norm).sum(dim=-1) / self.tau

            # All pair similarities: (B, B)
            all_score_u = torch.matmul(u_g_norm, i_g1_norm.t()) / self.tau

            # Apply attenuation mask: set masked false negatives to large negative value
            all_score_u_masked = all_score_u.masked_fill(mask_u == 0, -1e9)

            # Ensure diagonal is precisely positive similarity for numerical stability
            all_score_u_masked[diag_idx, diag_idx] = pos_score_u

            # Numerically stable InfoNCE: -(pos - logsumexp(all))
            loss_u = -(pos_score_u - torch.logsumexp(all_score_u_masked, dim=-1)).mean()

            # ─────────────────────────────────────────────────────────────
            # Item-side CL: Query I_g[i] <-> Positive Key U_{g+1}[u]
            # ─────────────────────────────────────────────────────────────
            i_g_tilde = self.inject_spectral_noise(i_g, beta)
            u_g1_tilde = self.inject_spectral_noise(u_g1, beta)

            i_g_norm = F.normalize(i_g_tilde, p=2, dim=-1)
            u_g1_norm = F.normalize(u_g1_tilde, p=2, dim=-1)

            pos_score_i = (i_g_norm * u_g1_norm).sum(dim=-1) / self.tau
            all_score_i = torch.matmul(i_g_norm, u_g1_norm.t()) / self.tau

            # Transpose mask for Item-to-User contrast
            all_score_i_masked = all_score_i.masked_fill(mask_u.t() == 0, -1e9)
            all_score_i_masked[diag_idx, diag_idx] = pos_score_i

            loss_i = -(pos_score_i - torch.logsumexp(all_score_i_masked, dim=-1)).mean()

            # Symmetric combination
            total_loss = total_loss + self.alpha * loss_u + (1.0 - self.alpha) * loss_i

        # 1/G gap normalization
        return total_loss / float(num_gaps)
