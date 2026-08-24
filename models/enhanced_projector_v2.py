"""
models/enhanced_projector_v2.py - Module 1: De-redundant Gated Projector
Enhancement Stage 2 cho mo hinh STAIR (SIGIR 2025).

Muc tieu: Thay the SVD Whitening tinh trong ham whitening() cua STAIR bang
module phi tuyen, tu thich ung voi:
  1. EMA Null-space Projection de khu du thua thong ke lien-modality.
  2. Gated Fusion de hoc trong so thich ung giua text va visual.

Tensor shapes xac nhan tu main.py va configs yaml:
  - text_feat   : (N_items, 384)  -- Sentence-BERT
  - visual_feat : (N_items, 4096) -- Deep CNN (ResNet-50)
  - d_hidden    : 64              -- cfg.embedding_dim
  - Dau ra      : (N_items, 64)   -- drop-in cho Item.embeddings.weight

Dieu chinh tu ban nhap EnhancedSTAIRV2.md:
  - d_text=384, d_visual=4096, d_hidden=64 xac nhan tu STAIR yaml/main.py
  - Them EMA covariance tracking dung register_buffer (khong co trong ban nhap)
  - Them Null-space Projection thay the SVD Whitening tinh
  - .detach() de ngat computation graph trong EMA update
  - L2-norm cuoi de khop voi normalize trong get_knn_graph() STAIR

Tham khao: yhhe2004/STAIR (SIGIR 2025), EnhancedSTAIRV2.md, MENTOR model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeRedundantGatedProjector(nn.Module):
    """
    De-redundant Gated Projector thay the SVD Whitening tinh trong STAIR.

    Kien truc 3 buoc:
      [A] Independent Projection: text 384->64, visual 4096->64 (GELU phi tuyen)
      [B] EMA Null-space Projection: loai bo top-r PC du thua giua 2 modality
      [C] Gated Fusion: z * h_t + (1-z) * h_v, z hoc tu gate network

    Args:
        d_text    (int):   Chieu text. STAIR Sentence-BERT = 384.
        d_visual  (int):   Chieu visual. STAIR deep CNN = 4096.
        d_hidden  (int):   Chieu dau ra = cfg.embedding_dim = 64.
        ema_decay (float): He so EMA alpha. Mac dinh 0.99.
        null_rank (int):   So PC bi loai bo. Mac dinh 16.
    """

    def __init__(
        self,
        d_text: int = 384,
        d_visual: int = 4096,
        d_hidden: int = 64,
        ema_decay: float = 0.99,
        null_rank: int = 16,
    ):
        super().__init__()

        assert d_hidden >= null_rank * 2, (
            f"null_rank ({null_rank}) qua lon so voi d_hidden ({d_hidden}). "
            f"Can d_hidden >= null_rank * 2."
        )

        self.d_hidden = d_hidden
        self.ema_decay = ema_decay
        self.null_rank = null_rank

        # [A] Independent Projection
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_hidden, bias=True),
            nn.GELU(),
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden, bias=True),
            nn.GELU(),
        )

        # [B] EMA Covariance Buffer
        # register_buffer: tu luu/load cung checkpoint, khong co gradient,
        # tu chuyen device khi .to(device)
        self.register_buffer("C_global", torch.eye(d_hidden, dtype=torch.float32))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

        # [C] Gated Fusion
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden, bias=True),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init cho Linear layers (phu hop voi GELU activation)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _update_ema_covariance(self, h: torch.Tensor):
        """
        Cap nhat EMA covariance tu batch hien tai.

        Math:
          x = h - mean(h)                 -- center
          C_batch = x.T @ x / (N-1)       -- batch covariance
          C_global <- alpha*C_global + (1-alpha)*C_batch

        Args:
            h (Tensor): (N, d_hidden). PHAI la .detach() -- ngat gradient!
        """
        with torch.no_grad():
            N = h.size(0)
            x = h - h.mean(dim=0, keepdim=True)    # (N, D) center
            C_batch = (x.T @ x) / max(N - 1, 1)   # (D, D) batch cov

            self.num_updates += 1
            if self.num_updates == 1:
                # Batch dau tien: set truc tiep (I chua co y nghia thong ke)
                self.C_global.copy_(C_batch)
            else:
                alpha = self.ema_decay
                # In-place EMA: tranh cap phat bo nho moi
                self.C_global.mul_(alpha).add_(C_batch, alpha=(1.0 - alpha))

    def _null_space_projection(self, h: torch.Tensor) -> torch.Tensor:
        """
        Chieu h vao null-space cua top-r PC cua C_global.

        Math:
          eigvals, eigvecs = eigh(C_global)    -- O(D^3), D=64: rat nhanh
          U_top = eigvecs[:, -r:]              -- top-r eigenvectors
          P_null = I - U_top @ U_top.T         -- null-space projector
          h_proj = h @ P_null                  -- chieu bo du thua

        Gradient chay qua h, KHONG chay qua P_null (with no_grad).
        """
        with torch.no_grad():
            try:
                # eigh: cho symmetric matrix, nhanh/on dinh hon svd
                _, eigvecs = torch.linalg.eigh(self.C_global)
            except torch.linalg.LinAlgError:
                # Fallback: regularize neu khong pos-semi-def
                C_reg = self.C_global + 1e-5 * torch.eye(
                    self.d_hidden, device=self.C_global.device,
                    dtype=self.C_global.dtype
                )
                _, eigvecs = torch.linalg.eigh(C_reg)

            # eigh tra ve eigenvalues tang dan -> top-r la r cuoi cung
            U_top = eigvecs[:, -self.null_rank:]               # (D, r)
            P_null = (
                torch.eye(self.d_hidden, device=h.device, dtype=h.dtype)
                - U_top @ U_top.T
            )                                                   # (D, D)

        return h @ P_null                                       # (N, D)

    def forward(
        self,
        text_feat: torch.Tensor,    # (N, 384)
        visual_feat: torch.Tensor,  # (N, 4096)
    ) -> torch.Tensor:
        """
        Args:
            text_feat   (Tensor): (N_items, 384)  -- raw Sentence-BERT
            visual_feat (Tensor): (N_items, 4096) -- raw deep CNN

        Returns:
            Tensor: (N_items, 64), L2-normalized.
                    Drop-in cho Item.embeddings.weight.data.copy_().
        """
        # [A] Project ve d_hidden
        h_t = self.text_proj(text_feat)     # (N, 64)
        h_v = self.vis_proj(visual_feat)    # (N, 64)

        # [B] EMA + Null-space Projection
        if self.training:
            # .detach() BAT BUOC: ngat computation graph, tranh bung no do thi
            h_pooled = (h_t + h_v).detach() * 0.5
            self._update_ema_covariance(h_pooled)

        h_t_proj = self._null_space_projection(h_t)    # (N, 64)
        h_v_proj = self._null_space_projection(h_v)    # (N, 64)

        # [C] Gated Fusion
        gate_input = torch.cat([h_t_proj, h_v_proj], dim=-1)   # (N, 128)
        z = self.gate(gate_input)                                # (N, 64)
        fused = z * h_t_proj + (1.0 - z) * h_v_proj            # (N, 64)

        # L2-normalize: khop voi F.normalize trong get_knn_graph() STAIR goc
        return F.normalize(fused, p=2, dim=-1)                   # (N, 64)


def prepare_item_embeddings(
    projector: DeRedundantGatedProjector,
    text_feat: torch.Tensor,
    visual_feat: torch.Tensor,
    n_items: int,
    embedding_dim: int = 64,
) -> torch.Tensor:
    """
    Drop-in replacement cho doan code trong prepare() cua STAIR goc:

        # STAIR goc (KHONG CHINH SUA FILE GOC):
        mfeats = [self.whitening(mfeat)*k for mfeat,k in zip(mfeats, num_neighbors)]
        mfeats = sum(mfeats).div(sum(num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        # Dung trong main_v5.py (FILE MOI):
        from models.enhanced_projector_v2 import (
            DeRedundantGatedProjector, prepare_item_embeddings
        )
        proj = DeRedundantGatedProjector(d_text=384, d_visual=4096, d_hidden=64).to(dev)
        item_embeds = prepare_item_embeddings(proj, text_feat, visual_feat, n_items)
        self.Item.embeddings.weight.data.copy_(item_embeds)

    Returns:
        Tensor: (N_items, 64), scale nhu whitening goc: x sqrt(N_items/D)
    """
    projector.eval()
    with torch.no_grad():
        out = projector(text_feat, visual_feat)      # (N, 64), L2-normalized

    # Scale giong whitening() goc:
    # return feats[:, :D] * math.sqrt(self.Item.count / cfg.embedding_dim)
    scale = math.sqrt(n_items / embedding_dim)
    return out * scale                               # (N, 64)


# ===========================================================================
# SANITY CHECK: python models/enhanced_projector_v2.py
# ===========================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Sanity Check: DeRedundantGatedProjector")
    print("STAIR tensor shapes: Baby, Sports, Electronics")
    print("=" * 65)

    TEST_CASES = [
        ("Baby",        7050,   384, 4096, 64),
        ("Sports",      35598,  384, 4096, 64),
        ("Electronics", 192403, 384, 4096, 64),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Shapes : text=(N,384), visual=(N,4096), out=(N,64)\n")

    all_passed = True
    for ds_name, n_items, d_text, d_vis, d_hidden in TEST_CASES:
        print(f"--- Dataset: {ds_name} (N_items={n_items:,}) ---")
        try:
            text_feat   = torch.randn(n_items, d_text).to(device)
            visual_feat = torch.randn(n_items, d_vis).mul(50.0).to(device)

            proj = DeRedundantGatedProjector(
                d_text=d_text, d_visual=d_vis,
                d_hidden=d_hidden, ema_decay=0.99, null_rank=16,
            ).to(device)

            # Test 1: Training forward + EMA update
            proj.train()
            out_train = proj(text_feat, visual_feat)
            assert out_train.shape == (n_items, d_hidden)
            assert proj.num_updates.item() == 1, "EMA phai update 1 lan khi train"

            # Test 2: Eval forward (EMA frozen)
            proj.eval()
            with torch.no_grad():
                out_eval = proj(text_feat, visual_feat)
            assert out_eval.shape == (n_items, d_hidden)
            assert proj.num_updates.item() == 1, "EMA KHONG duoc update khi eval"

            # Test 3: L2-norm = 1.0
            max_norm_err = (out_eval.norm(dim=-1) - 1.0).abs().max().item()
            assert max_norm_err < 1e-5, f"L2-norm error: {max_norm_err:.2e}"

            # Test 4: prepare_item_embeddings (drop-in replacement)
            item_embeds = prepare_item_embeddings(
                proj, text_feat, visual_feat, n_items, d_hidden
            )
            assert item_embeds.shape == (n_items, d_hidden)

            # Test 5: Backward pass -- gradient flow check
            proj.train()
            proj(text_feat, visual_feat).sum().backward()
            for name, p in proj.named_parameters():
                assert p.grad is not None and not torch.isnan(p.grad).any(), \
                    f"Bad grad: {name}"
            assert proj.C_global.grad is None, "C_global KHONG duoc co gradient!"

            scale = math.sqrt(n_items / d_hidden)
            print(f"  [OK] train/eval: {out_train.shape}, L2-err={max_norm_err:.1e}")
            print(f"  [OK] item_embeds: {item_embeds.shape}, scale={scale:.2f}")
            print(f"  [OK] EMA: {proj.num_updates.item()} update, C_global no grad")
            print(f"  [OK] Backward: all params have valid gradients")

        except Exception as e:
            print(f"  [FAIL] {e}")
            all_passed = False
        print()

    print("=" * 65)
    if all_passed:
        print("[PASSED] Tat ca 5 sanity checks OK cho ca 3 datasets.")
    else:
        print("[FAILED] Co loi -- kiem tra lai!")
    print("  Output (N,64): drop-in cho Item.embeddings.weight.data.copy_()")
    print("  EMA frozen khi eval; C_global khong co grad; gradient flow dung")
    print("=" * 65)
