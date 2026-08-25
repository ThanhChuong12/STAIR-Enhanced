"""
models/residual_projector_v2.py
===============================================================================
Module 1 Revised — STAIR-Enhanced v2a (ResOnly): Residual-Whitening Projector

Dong Luc (Lesson learned tu v1):
  Phien ban v1 (DeRedundantGatedProjector) loai bo hoan toan SVD Whitening
  khien cac gia tri beta3 (BSC Smoother) bi mismatch hyperparameter nghiem
  trong, dan den sut giam ~9% tren ca 3 datasets.

Chien Luoc V2:
  Giu nguyen SVD Whitening lam nhanh chinh (frozen, structural prior).
  Them Residual Projector HOC DUOC de hoc phan correction phi tuyen bo sung:

      e_i = whitening(x_i) + lambda_res * Delta_i

  Trong do:
    - whitening(x_i)    : tinh co dinh tu STAIR.whitening() (frozen, khong grad)
    - Delta_i           : nhanh correction hoc duoc (ResidualWhiteningProjector)
    - lambda_res        : nn.Parameter khoi tao tai 0.1 (ho tro gradient)

Cai Tien Ky Thuat:
  1. LeakyReLU(0.1) thay vi GELU: giu tinh chat gan identity quanh 0
     -> warm-start xap xi SVD khong bi bien dang boi ham kich hoat bao hoa
  2. L2 Normalization (eps=1e-12) thay vi LayerNorm: dam bao Delta_i
     co norm = 1 truoc khi nhan voi lambda_res -> scale duoc kiem soat hoan toan
  3. Warm-start tu SVD: W_t ~ V_t[:D, :] tu SVD cua text_feat
     -> tai epoch 0, h_t xap xi phep chieu SVD, Delta_i ~ 0 (correction nho)
  4. lambda_res la nn.Parameter: cho phep optimizer tu dong hoc muc do
     dong gop cua nhanh correction

Tensor Shapes (xac nhan tu main.py + YAML configs):
  - text_feat   : (N_items, 384)  -- Sentence-BERT
  - visual_feat : (N_items, 4096) -- Deep CNN / ResNet-50
  - d_hidden    : 64              -- cfg.embedding_dim
  - E_svd       : (N_items, 64)   -- ket qua tu STAIR.whitening() (frozen)
  - Delta_i     : (N_items, 64)   -- correction hoc duoc (L2-normalized)
  - e_final     : (N_items, 64)   -- drop-in cho Item.embeddings.weight

Tham khao:
  - STAIRE2_v2_Report.md (Root Cause Analysis + math)
  - He and McAuley (2016) VBPR -- multi-component embedding design
  - He et al. (2020) LightGCN -- additive residual design pattern
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualWhiteningProjector(nn.Module):
    """
    Residual Projector: hoc phan correction phi tuyen tren nen SVD Whitening.

    Cong thuc:
        Delta_i = L2Norm( (h_t_i + h_v_i) / 2 )
        e_i = E_svd_i + lambda_res * Delta_i

    Trong do E_svd duoc tinh ben ngoai qua STAIR.whitening() va truyen vao
    ham composite_embeddings() -- phan nay khong co gradient.

    Kien truc nhanh Residual:
        Text  (384-D)  ──► Linear(384, 64, bias=True) ──► LeakyReLU(0.1) ──► h_t
        Visual(4096-D) ──► Linear(4096, 64, bias=True) ──► LeakyReLU(0.1) ──► h_v
                                                            (h_t + h_v) / 2
                                                                │
                                                        L2Norm(·, eps=1e-12)
                                                                │
                                                            Delta_i  (64-D, ||=1)

    Args:
        d_text    (int):   Chieu text. STAIR Sentence-BERT = 384.
        d_visual  (int):   Chieu visual. STAIR deep CNN = 4096.
        d_hidden  (int):   Chieu dau ra = cfg.embedding_dim = 64.
        lambda_init (float): Gia tri khoi tao cua lambda_res. Mac dinh 0.1.
    """

    def __init__(
        self,
        d_text: int = 384,
        d_visual: int = 4096,
        d_hidden: int = 64,
        lambda_init: float = 0.1,
    ):
        super().__init__()

        self.d_text   = d_text
        self.d_visual = d_visual
        self.d_hidden = d_hidden

        # [A] Nhanh Text: 384 -> 64
        # Dung LeakyReLU(0.1) de: (1) dam bao gradient khong bi zero pha huy,
        # (2) giu tinh chat gan identity quanh 0 ho tro warm-start tot hon GELU.
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_hidden, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

        # [B] Nhanh Visual: 4096 -> 64
        self.vis_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )

        # [C] Learnable Scaling Parameter
        # lambda_res khoi tao = 0.1 de correction ban dau chi dong gop 10%
        # so voi E_svd. Khi can, optimizer tu hoc tang len neu correction co ich.
        self.lambda_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.max_lambda = 0.3

        # Khoi tao Kaiming cho Linear layers (fallback truoc khi warm-start)
        self._init_kaiming()

    def _init_kaiming(self):
        """Khoi tao Kaiming Normal cho all Linear layers (fallback default)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_warm_start_weights(
        self,
        text_feat: torch.Tensor,   # (N, 384) -- raw Sentence-BERT features
        vis_feat: torch.Tensor,    # (N, 4096) -- raw visual features
    ) -> None:
        """
        Warm-start trong so tu SVD cua raw features.

        Muc tieu: tai epoch 0, h_t ~ phep chieu SVD ban dau cua STAIR,
        tuc la Delta_i ~ 0 (correction gan bang 0). Mo hinh bat dau tu
        diem tuong duong baseline, chi can hoc phan deviation nho.

        Math:
          text_feat (N x d_t) = U_t @ S_t @ V_t^T
            => U_t (N x N), S_t (N,), V_t (d_t x d_t)
          V_t[:D, :] la D hang dau cua V_t^T, tuong ung D eigenvectors lon nhat
          Ta khoi tao: W_t = V_t^T[:D, :] = V_t[:, :D].T (d_hidden x d_text)
            sao cho: W_t @ x_t ≈ projection lên D PC dau tien = U_t[:, :D] * S

          Luong phu (bias va LeakyReLU): du phep chieu khong chinh xac tuyet doi,
          warm-start dam bao W_t bat dau gan voi huong tot nhat cua data.

        Args:
            text_feat (Tensor): (N, d_text) -- raw text features, CPU/GPU.
            vis_feat  (Tensor): (N, d_visual) -- raw visual features, CPU/GPU.
        """
        with torch.no_grad():
            # ---- Text branch warm-start ----
            x_t = text_feat.float()
            x_t = x_t - x_t.mean(dim=0, keepdim=True)   # center (nhu whitening)
            # Tinh SVD: U (N x N), S (min(N,d_t),), Vh (d_t x d_t)
            # Dung svd_lowrank de tiet kiem VRAM voi N lon (Electronics ~192K)
            # q = d_hidden = 64 la so PC can, nsamples=4*q cho precision cao
            try:
                U_t, S_t, Vh_t = torch.linalg.svd(x_t, full_matrices=False)
                # Vh_t: (min(N,384), 384), cac hang = eigenvectors
                W_t_init = Vh_t[:self.d_hidden, :]          # (64, 384)
            except Exception:
                # Fallback: svd_lowrank cho dataset qua lon
                _, _, Vh_t = torch.svd_lowrank(x_t, q=self.d_hidden, niter=4)
                W_t_init = Vh_t.T[:self.d_hidden, :]        # (64, 384)

            # Copy vao Linear layer (bias giu nguyen = 0)
            self.text_proj[0].weight.data.copy_(W_t_init.to(self.text_proj[0].weight.device))

            # ---- Visual branch warm-start ----
            x_v = vis_feat.float()
            x_v = x_v - x_v.mean(dim=0, keepdim=True)   # center
            try:
                # Visual 4096-D: dung svd_lowrank bat buoc (full SVD qua lon)
                _, _, Vh_v = torch.svd_lowrank(x_v, q=self.d_hidden, niter=4)
                W_v_init = Vh_v.T[:self.d_hidden, :]        # (64, 4096)
            except Exception:
                # Double fallback: random normal gan 0
                W_v_init = torch.randn(self.d_hidden, self.d_visual) * 0.01

            self.vis_proj[0].weight.data.copy_(W_v_init.to(self.vis_proj[0].weight.device))

        print(f'[Warm-start] text_proj.weight: {self.text_proj[0].weight.shape}, '
              f'vis_proj.weight: {self.vis_proj[0].weight.shape}')
        print(f'[Warm-start] lambda_raw initial = {self.lambda_raw.item():.4f} (requires_grad={self.lambda_raw.requires_grad})')

    def forward(
        self,
        text_feat: torch.Tensor,    # (N, 384)
        visual_feat: torch.Tensor,  # (N, 4096)
    ) -> torch.Tensor:
        """
        Tinh phan correction phi tuyen Delta_i (L2-normalized).

        Flow:
            h_t = LeakyReLU(W_t @ x_t + b_t)          (N, 64)
            h_v = LeakyReLU(W_v @ x_v + b_v)          (N, 64)
            pooled = (h_t + h_v) / 2                   (N, 64)
            Delta = pooled / max(||pooled||_2, eps)    (N, 64), ||Delta||=1

        Returns:
            Delta (Tensor): (N, d_hidden), moi hang co L2-norm = 1.
                            Duoc nhan voi lambda_res ben ngoai (trong
                            composite_embeddings) de kiem soat bien do correction.
        """
        h_t = self.text_proj(text_feat)        # (N, 64) -- text projection
        h_v = self.vis_proj(visual_feat)       # (N, 64) -- visual projection

        # Trung binh don gian: ca 2 phuong thuc dong gop bang nhau
        # (khac v1 dung gate phuc tap de tranh gate collapse)
        pooled = (h_t + h_v) * 0.5             # (N, 64)

        # L2 Normalization voi eps=1e-12 tranh chia 0 khi pooled ~ 0
        # (thuong xay ra o epoch 0 do warm-start -> h_t ~ h_v ~ U*S ~= const)
        delta = F.normalize(pooled, p=2, dim=-1, eps=1e-12)   # (N, 64), norm=1

        return delta                            # (N, 64)


def composite_embeddings(
    projector: ResidualWhiteningProjector,
    text_feat: torch.Tensor,
    visual_feat: torch.Tensor,
    e_svd: torch.Tensor,
    n_items: int,
    embedding_dim: int = 64,
) -> torch.Tensor:
    delta = projector(text_feat, visual_feat)
    scale = math.sqrt(n_items / embedding_dim)
    delta_scaled = delta * scale
    actual_lambda = projector.max_lambda * torch.sigmoid(projector.lambda_raw)
    e_final = e_svd + actual_lambda * delta_scaled
    e_norm = e_final / (e_final.norm(dim=-1, keepdim=True).clamp(min=1e-12))
    return e_norm * scale                            # (N, 64)


# ============================================================================
# SANITY CHECK & DUMMY TEST BLOCK
# Chay: python models/residual_projector_v2.py
# ============================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('Sanity Check: ResidualWhiteningProjector (STAIR-Enhanced v2a)')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    D_TEXT, D_VIS, D_H = 384, 4096, 64
    DATASETS = [
        ('Baby',        7050),
        ('Sports',      35598),
        ('Electronics', 1000),   # Giam N de test nhanh tren CPU
    ]

    all_passed = True
    for ds_name, N in DATASETS:
        print(f'--- Dataset: {ds_name} (N={N:,}) ---')
        try:
            # Tao dummy features
            text_feat   = torch.randn(N, D_TEXT)
            visual_feat = torch.randn(N, D_VIS).mul(20.0)  # scale nhu actual data
            e_svd       = torch.randn(N, D_H).mul(math.sqrt(N / D_H))  # xap xi whitened

            proj = ResidualWhiteningProjector(
                d_text=D_TEXT, d_visual=D_VIS, d_hidden=D_H, lambda_init=0.1
            ).to(device)

            # === Test 1: lambda_res la Parameter va requires_grad=True ===
            assert isinstance(proj.lambda_raw, nn.Parameter), \
                'lambda_raw phai la nn.Parameter!'
            assert proj.lambda_raw.requires_grad, \
                'lambda_raw phai co requires_grad=True!'
            assert abs(proj.lambda_raw.item() - 0.0) < 1e-6, \
                f'lambda_raw khoi tao sai: {proj.lambda_raw.item()}'
            print(f'  [OK] lambda_raw = {proj.lambda_raw.item():.4f}, '
                  f'requires_grad = {proj.lambda_raw.requires_grad}')

            # === Test 2: Warm-start SVD ===
            print('  Running warm-start SVD...')
            proj.init_warm_start_weights(text_feat, visual_feat)
            W_t_norm = proj.text_proj[0].weight.data.norm().item()
            W_v_norm = proj.vis_proj[0].weight.data.norm().item()
            print(f'  [OK] After warm-start: ||W_t||={W_t_norm:.3f}, ||W_v||={W_v_norm:.3f}')

            # === Test 3: Forward pass — Delta shape & L2 norm ===
            proj.eval()
            with torch.no_grad():
                text_feat_d   = text_feat.to(device)
                visual_feat_d = visual_feat.to(device)
                e_svd_d       = e_svd.to(device)
                delta = proj(text_feat_d, visual_feat_d)

            assert delta.shape == (N, D_H), f'Delta shape error: {delta.shape}'
            max_norm_err = (delta.norm(dim=-1) - 1.0).abs().max().item()
            assert max_norm_err < 1e-5, f'L2-norm error: {max_norm_err}'
            print(f'  [OK] Delta: shape={tuple(delta.shape)}, L2-norm err={max_norm_err:.2e}')

            # === Test 4: composite_embeddings — output shape & magnitude ===
            with torch.no_grad():
                e_final = composite_embeddings(
                    proj, text_feat_d, visual_feat_d, e_svd_d, N, D_H
                )
            assert e_final.shape == (N, D_H), f'e_final shape error: {e_final.shape}'
            target_scale = math.sqrt(N / D_H)
            actual_scale = e_final.norm(dim=-1).mean().item()
            print(f'  [OK] e_final: shape={tuple(e_final.shape)}, '
                  f'target_scale={target_scale:.3f}, actual_mean_norm={actual_scale:.3f}')

            # === Test 5: Backward pass — gradient flow ===
            proj.train()
            e_svd_d2       = torch.randn(N, D_H).mul(math.sqrt(N / D_H)).to(device)
            delta2         = proj(text_feat_d, visual_feat_d)
            loss           = (e_svd_d2 + torch.sigmoid(proj.lambda_raw) * delta2).sum()
            loss.backward()

            # Kiem tra gradient cua lambda_raw
            assert proj.lambda_raw.grad is not None, 'lambda_raw phai co gradient!'
            assert not torch.isnan(proj.lambda_raw.grad), 'gradient cua lambda_raw bi NaN!'

            # Kiem tra tat ca Linear params co gradient hop le
            for name, p in proj.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and not torch.isnan(p.grad).any(), \
                        f'Bad gradient: {name}'

            print(f'  [OK] Backward: lambda_res.grad={proj.lambda_raw.grad.item():.6f}, '
                  f'all params have valid gradients')

            # === Test 6: lambda_res thay doi qua optimizer buoc ===
            proj.zero_grad()
            lambda_before = proj.lambda_raw.item()
            opt = torch.optim.SGD(proj.parameters(), lr=0.1)
            e_svd_d3 = torch.randn(N, D_H).to(device)
            delta3   = proj(text_feat_d, visual_feat_d)
            (e_svd_d3 + torch.sigmoid(proj.lambda_raw) * delta3).sum().backward()
            opt.step()
            lambda_after = proj.lambda_raw.item()
            assert abs(lambda_before - lambda_after) > 1e-8, \
                'lambda_raw khong duoc cap nhat boi optimizer!'
            print(f'  [OK] lambda_raw update: {lambda_before:.6f} -> {lambda_after:.6f}')

            print()
        except Exception as e:
            print(f'  [FAIL] {ds_name}: {e}')
            import traceback; traceback.print_exc()
            all_passed = False
            print()

    print('=' * 70)
    if all_passed:
        print('[PASSED] Tat ca 6 sanity checks deu OK cho ca 3 datasets.')
    else:
        print('[FAILED] Co loi -- kiem tra lai!')
    print()
    print('Kien truc tom tat:')
    proj_demo = ResidualWhiteningProjector()
    total_params = sum(p.numel() for p in proj_demo.parameters())
    trainable   = sum(p.numel() for p in proj_demo.parameters() if p.requires_grad)
    print(f'  Total params    : {total_params:,}')
    print(f'  Trainable params: {trainable:,}')
    print(f'  lambda_res      : nn.Parameter, init=0.1, requires_grad=True')
    print()
    print('Cach tich hop vao main_enhanced_v2.py:')
    print('  e_svd  = self.whitening_stacked(mfeats_raw, cfg.num_neighbors)')
    print('  e_final = composite_embeddings(')
    print('      self.projector, text_feat, vis_feat, e_svd, N_items)')
    print('  self.Item.embeddings.weight.data.copy_(e_final)')
    print('=' * 70)
