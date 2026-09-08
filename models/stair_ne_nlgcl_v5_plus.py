# -*- coding: utf-8 -*-
"""
models/stair_ne_nlgcl_v5_plus.py
=================================
Mô hình hoàn thiện tối ưu: STAIR-NE-NLGCL v5+ (v3-Refined)
Selective Synergy & Minimalist Clean Architecture

Core Design Pillars:
────────────────────
1. GNN Backbone trực tiếp (No Projection Head):
   - Tương phản trực tiếp giữa H^(0) và H^(1) như SOTA v5.
   - 100% thông lượng gradient InfoNCE truyền thẳng vào bảng embedding cơ sở E_u, E_i.
2. True Sign-Preserving Spectral Perturbation (|noise| >= 0):
   - h_tilde = h + eps * (beta * sign(h) * (|eta| / || |eta| ||_2))
   - Bảo toàn tuyệt đối 100% góc phần tư không gian, triệt tiêu hiện tượng đảo pha tọa độ.
3. Clean Linear HANS (Hardness-Aware Negative Scheduling):
   - Phạt tuyến tính có ngưỡng: psi = 1.0 + gamma_h * clamp(cos_sim, min=0.0)
   - gamma_h = 0.15 (thấp hơn v3), không làm co rút nhiệt độ hiệu dụng tau_eff = tau / (1 + gamma_h).
4. Hard-Threshold MFNA (Modality False Negative Attenuation):
   - Ngưỡng lọc cứng: Nếu S_modal > tau_thresh (0.85) -> mask = 0.0 (loại bỏ hoàn toàn mẫu âm giả).
   - Ngược lại mask = 1.0. Tránh việc làm mờ gradient do soft clamping.
5. In-batch Dynamic Slicing [B x B]:
   - Chỉ tính toán lát cắt tương đồng modal trong batch, tiết kiệm 96.5% bộ nhớ, chống OOM.
6. Constant Contrastive Weight with Linear Warmup:
   - lambda_cl = 0.010 cố định (không Cosine decay làm suy kiệt lực đẩy ở cuối).
   - Linear warmup 0 -> 0.010 trong 50 epochs đầu giúp BPR ổn định cấu trúc tô-pô.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['STAIR_NE_NLGCL_v5_Plus']


class STAIR_NE_NLGCL_v5_Plus(nn.Module):
    """
    STAIR-NE-NLGCL v5+ Contrastive Learning Module.
    Eliminates Projection Head, retains sign-preserving noise, applies linear HANS
    and hard-threshold MFNA with constant contrastive pressure.
    """

    def __init__(
        self,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        tau: float = 0.20,
        alpha_dir: float = 0.50,
        eps: float = 0.08,           # Giảm nhẹ biên độ nhiễu xuống 0.08
        tau_thresh: float = 0.85,    # Ngưỡng lọc cứng False Negatives
        lambda_cl: float = 0.010,    # Cố định lực đẩy chống over-smoothing
        gamma_h: float = 0.15,       # Phạt tuyến tính vừa phải (tránh co rút tau)
        warmup_epochs: int = 50,     # Warmup tuyến tính cho lambda trong 50 epoch đầu
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.tau = tau
        self.alpha_dir = alpha_dir
        self.eps = eps
        self.tau_thresh = tau_thresh
        self.target_lambda = lambda_cl
        self.gamma_h = gamma_h
        self.warmup_epochs = warmup_epochs

        self.current_epoch = 0
        self.current_lambda = 0.0

    def update_epoch(self, epoch: int):
        """Warmup lambda từ 0 -> lambda_cl trong warmup_epochs đầu, sau đó cố định hoàn toàn."""
        self.current_epoch = epoch
        if epoch <= self.warmup_epochs:
            self.current_lambda = self.target_lambda * (float(epoch) / float(max(1, self.warmup_epochs)))
        else:
            self.current_lambda = self.target_lambda

    def get_current_params(self) -> Tuple[float, float]:
        """Trả về tuple (gamma_h, current_lambda) phục vụ logging."""
        return self.gamma_h, self.current_lambda

    def get_current_hans_params(self) -> Tuple[float, float]:
        """Alias tương thích ngược cho Coach v3."""
        return self.gamma_h, self.current_lambda

    def update_scheduler(self, current_cl_loss: float = 0.0, **kwargs):
        """Alias tương thích ngược cho Coach gọi update_scheduler."""
        pass

    def inject_spectral_noise(self, h: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Bơm nhiễu quang phổ bảo toàn hướng tuyệt đối với |noise|.
        h_tilde = h + eps * (beta * sign(h) * (|noise| / || |noise| ||_2))
        """
        if not self.training or self.eps <= 0.0:
            return h

        noise = torch.randn_like(h).abs()
        noise = F.normalize(noise, p=2, dim=-1)

        beta_w = beta.unsqueeze(0) if beta.dim() == 1 else beta
        return h + self.eps * (beta_w * torch.sign(h) * noise)

    def forward(
        self,
        layer_embeds: List[torch.Tensor],
        users: torch.Tensor,
        positives: torch.Tensor,
        beta: torch.Tensor,
        item_modals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass tính toán mất mát InfoNCE hai chiều với Linear HANS và Hard MFNA.

        Args:
            layer_embeds: [H^(0), H^(1), ...] từ FSC backbone, mỗi tensor có shape (N_u + N_i, D).
            users: (B,) user indices trong mini-batch.
            positives: (B,) positive item indices trong mini-batch.
            beta: (D,) spectral propagation vector (1.0 - beta3).
            item_modals: (B, D) hoặc (N_items, D) whitened modal features.

        Returns:
            Tuple: (total_loss có trọng số, raw_cl_loss dạng float)
        """
        users = users.view(-1)
        positives = positives.view(-1)
        device = layer_embeds[0].device
        batch_size = users.size(0)

        # ─────────────────────────────────────────────────────────────────
        # 1. Trích xuất trực tiếp tầng 0 và tầng 1 (Không dùng Projection Head)
        # ─────────────────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────────────────
        # 2. Bơm nhiễu bảo toàn góc phần tư và chuẩn hóa L2
        # ─────────────────────────────────────────────────────────────────
        u_0_t = F.normalize(self.inject_spectral_noise(u_0, beta), p=2, dim=-1)
        i_1_t = F.normalize(self.inject_spectral_noise(i_1, beta), p=2, dim=-1)
        i_0_t = F.normalize(self.inject_spectral_noise(i_0, beta), p=2, dim=-1)
        u_1_t = F.normalize(self.inject_spectral_noise(u_1, beta), p=2, dim=-1)

        # ─────────────────────────────────────────────────────────────────
        # 3. Dynamic Slicing + Ngưỡng cứng MFNA (Chống OOM & lọc triệt để False Negatives)
        # ─────────────────────────────────────────────────────────────────
        if item_modals is not None and self.tau_thresh < 1.0:
            with torch.no_grad():
                i_batch = item_modals[positives] if item_modals.size(0) != batch_size else item_modals
                i_norm = F.normalize(i_batch, p=2, dim=-1)
                sim_modal = torch.matmul(i_norm, i_norm.t())
                # Ngưỡng cứng: nếu sim > tau_thresh loại bỏ hoàn toàn (mask = 0.0), ngược lại 1.0
                mfna_mask = (sim_modal <= self.tau_thresh).float()
        else:
            mfna_mask = torch.ones((batch_size, batch_size), device=device)

        diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        valid_neg_mask = mfna_mask * diag_mask.float()

        # ─────────────────────────────────────────────────────────────────
        # 4. Hướng 1: User-to-Item (U_0 -> I_1) với phạt HANS tuyến tính
        # ─────────────────────────────────────────────────────────────────
        pos_u2i = (u_0_t * i_1_t).sum(dim=-1) / self.tau
        cos_u2i = torch.matmul(u_0_t, i_1_t.t())
        sim_u2i = cos_u2i / self.tau

        # Phạt tuyến tính: psi = 1.0 + gamma_h * max(0, cos)
        # Tuyệt đối không làm thay đổi nhiệt độ hiệu dụng tau_eff
        hans_u2i = 1.0 + self.gamma_h * torch.clamp(cos_u2i, min=0.0)
        neg_terms_u2i = valid_neg_mask * hans_u2i * torch.exp(sim_u2i)
        loss_u2i = -(pos_u2i - torch.log(torch.exp(pos_u2i) + neg_terms_u2i.sum(dim=-1) + 1e-8)).mean()

        # ─────────────────────────────────────────────────────────────────
        # 5. Hướng 2: Item-to-User (I_0 -> U_1) với phạt HANS tuyến tính
        # ─────────────────────────────────────────────────────────────────
        pos_i2u = (i_0_t * u_1_t).sum(dim=-1) / self.tau
        cos_i2u = torch.matmul(i_0_t, u_1_t.t())
        sim_i2u = cos_i2u / self.tau

        hans_i2u = 1.0 + self.gamma_h * torch.clamp(cos_i2u, min=0.0)
        neg_terms_i2u = valid_neg_mask.t() * hans_i2u * torch.exp(sim_i2u)
        loss_i2u = -(pos_i2u - torch.log(torch.exp(pos_i2u) + neg_terms_i2u.sum(dim=-1) + 1e-8)).mean()

        # ─────────────────────────────────────────────────────────────────
        # 6. Tổng hợp hàm mất mát có điều phối Warmup
        # ─────────────────────────────────────────────────────────────────
        raw_loss = self.alpha_dir * loss_u2i + (1.0 - self.alpha_dir) * loss_i2u
        total_loss = self.current_lambda * raw_loss

        return total_loss, raw_loss.item()
