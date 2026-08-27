"""
preprocess_stair_lia.py
=======================
STAIR-LIA v3 - Offline Preprocessing Pipeline

Phase 1 - PyTorch ZCA Whitening (from CFDTBD/Model-beauty/model-beauty.py TF->PyTorch):
    text (384-D) -> 64-D, visual (4096-D) -> 64-D
    Zero-phase: giữ nguyên hướng trục tọa độ, chỉ decorrelate + rescale.

Phase 2 - Offline Bidirectional Cross-Modal Attention ROI Extraction:
    Source: CLID-master/CLID/src/Cross_attention.py + models/CLID.py
    Virtual Token Segmentation (STAIRE2_v3.md):
        Text  384-D -> (6,  64) virtual tokens
        Visual 4096-D -> (64, 64) virtual patches
    Trích xuất E_roi_t (64-D) và E_roi_v (64-D) offline.

Phase 3 - Fusion (CLID fuweight):
    E_fused = fuweight * E_global + (1-fuweight) * E_roi

Export: E_global_t, E_global_v, E_roi_t, E_roi_v, E_fused_t, E_fused_v
        -> .pt + .npy

Cach dung:
    python preprocess_stair_lia.py \
        --dataset Baby \
        --text_feat  /path/to/text_feat.npy \
        --visual_feat /path/to/image_feat.npy \
        --output_dir preprocessed_lia \
        --sanity_check

TUYET DOI KHONG chay script nay trong vong lap training.
"""

import math, argparse, logging, time
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: ZCA Whitening
# ============================================================================
# Source: CFDTBD/Model-beauty/model-beauty.py  zca_whitening() [TF1.x -> PyTorch]
#
# TF goc:
#   cov = X_batch^T X_batch / n
#   s, u, _ = tf.linalg.svd(cov)
#   zca_matrix = u @ diag(1/sqrt(s+eps)) @ u^T
#   whitened = X_batch @ zca_matrix
#
# PyTorch: thay SVD bang eigh (on dinh hon voi ma tran doi xung)
# ============================================================================

class ZCAWhiteningProjector(nn.Module):
    """
    ZCA Whitening + Dimensionality Reduction.

    Cho X (N, D_in):
        1. Khử trung bình:  X_c = X - mean(X)
        2. Hiệp phương sai: C   = X_c^T X_c / (N-1)
        3. Phân rã eigen:   C   = V @ diag(lam) @ V^T
        4. ZCA matrix:      W   = V @ diag(1/sqrt(lam+eps)) @ V^T
        5. Whitened:        X_w = X_c @ W  -> lấy D_out cột cuối
        6. Scale:           E   = X_w * sqrt(N/D_out)   [STAIR mag convention]

    Khac voi SVD Whitening cua STAIR goc (U[:, :D] * scale):
        - SVD xoay truc toa do -> pha SVD khong con zero-phase
        - ZCA giu nguyen huong -> bao ve cau truc dimension-wise cua BSC Smoother
    """

    def __init__(self, d_in: int, d_out: int = 64, eps: float = 1e-5):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.eps   = eps
        # Buffer (not nn.Parameter) - chi tinh 1 lan offline
        self.register_buffer("zca_matrix", torch.zeros(d_in, d_in))
        self.register_buffer("mean_vec",   torch.zeros(d_in))
        self.register_buffer("scale",      torch.tensor(1.0))
        self._fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> None:
        """Tinh ZCA matrix tu toan bo item features X (N, d_in)."""
        logger.info(f"  [ZCA fit] shape={tuple(X.shape)}")
        t0 = time.time()

        mean  = X.mean(dim=0)
        X_c   = X - mean                          # (N, d_in)
        N     = X.shape[0]

        # Tinh covariance  C = X_c^T X_c / (N-1)
        # Visual 4096-D: C la (4096, 4096) - chi can ~128MB fp32
        cov   = X_c.t().mm(X_c) / (N - 1)        # (d_in, d_in)

        # Phan ra Eigendecomposition (eigh: doi xung, eigenvalue tang dan)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # ZCA matrix: W = V @ diag(1/sqrt(lam+eps)) @ V^T
        inv_sqrt = 1.0 / torch.sqrt(eigenvalues.clamp(min=0.0) + self.eps)
        zca_mat  = eigenvectors.mm(torch.diag(inv_sqrt)).mm(eigenvectors.t())

        # Scale theo convention STAIR: sqrt(N/D_out)
        scale = math.sqrt(N / self.d_out)

        self.mean_vec.copy_(mean)
        self.zca_matrix.copy_(zca_mat)
        self.scale.copy_(torch.tensor(scale))
        self._fitted = True

        logger.info(
            f"  [ZCA fit] done {time.time()-t0:.1f}s | "
            f"lam=[{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]"
        )

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Ap dung ZCA va chieu ve d_out chieu.  X: (N, d_in) -> (N, d_out)"""
        if not self._fitted:
            raise RuntimeError("Goi fit() truoc khi transform()!")
        X_c  = X - self.mean_vec
        X_w  = X_c.mm(self.zca_matrix)           # (N, d_in) ZCA whitened
        # Lay d_out cot CUOI (eigenvalue lon nhat o cuoi khi dung eigh)
        E    = X_w[:, -self.d_out:] * self.scale  # (N, d_out)
        return E

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)


# ============================================================================
# Phase 2: Bidirectional Cross-Modal Attention ROI Extraction
# ============================================================================
# Source:
#   CLID/src/Cross_attention.py: ContextSaver.process_block()
#   CLID/src/models/CLID.py:    cross_modal_attention(), fuweight fusion
#   STAIRE2_v3.md:              Virtual Token Segmentation design
#
# CLID goc (Cross_attention.py) thuc hien dimension-wise attention:
#   t_q = text.unsqueeze(2)   # (N, 384, 1)
#   i_k = image.unsqueeze(2)  # (N, 4096, 1)
#   scores = t_q @ i_k^T      # (N, 384, 4096)
#   context = softmax(scores) @ i_v  # (N, 384, 1) -> mean -> (N, 384)
#
# STAIRE2_v3 cai tien: Virtual Token Segmentation -> Scaled Dot-Product Attention
# ============================================================================

class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention ROI Extraction (CLID-style, STAIRE2_v3 design).

    Thay vi dimension-wise attention cua CLID goc, dung Virtual Token Segmentation:
        Text  (N, 384)  -> (N, N_t=6,  d_sub=64)  virtual tokens
        Visual (N, 4096) -> (N, N_v=64, d_sub=64)  virtual patches

    Bidirectional Scaled Dot-Product Attention:
        Text-guided Visual ROI:
            Q = X_t @ W_q_t              (N, N_t, d_k)
            K = X_v @ W_k_v              (N, N_v, d_k)
            V = X_v @ W_v_v              (N, N_v, d_sub)
            attn_v = softmax(QK^T/sqrt(d_k))  (N, N_t, N_v)
            ctx    = attn_v @ V          (N, N_t, d_sub)
            E_roi_v = LN(X_t + ctx).mean(dim=1)   (N, d_sub)

        Visual-guided Text ROI (doi xung) -> E_roi_t (N, d_sub)

    Toan bo chay @no_grad() - KHONG train, KHONG backward.
    """

    def __init__(
        self,
        d_t:   int = 384,
        d_v:   int = 4096,
        d_sub: int = 64,
        d_k:   int = 32,
    ):
        super().__init__()
        self.d_sub = d_sub
        self.d_k   = d_k
        assert d_t % d_sub == 0, f"d_t={d_t} phai chia het cho d_sub={d_sub}"
        assert d_v % d_sub == 0, f"d_v={d_v} phai chia het cho d_sub={d_sub}"
        self.N_t = d_t // d_sub   # 6  (so virtual text tokens)
        self.N_v = d_v // d_sub   # 64 (so virtual visual patches)
        self._scale = math.sqrt(d_k)

        # Text-guided Visual ROI
        self.W_q_t = nn.Linear(d_sub, d_k,  bias=False)
        self.W_k_v = nn.Linear(d_sub, d_k,  bias=False)
        self.W_v_v = nn.Linear(d_sub, d_sub, bias=False)

        # Visual-guided Text ROI
        self.W_q_v = nn.Linear(d_sub, d_k,  bias=False)
        self.W_k_t = nn.Linear(d_sub, d_k,  bias=False)
        self.W_v_t = nn.Linear(d_sub, d_sub, bias=False)

        self.ln_v = nn.LayerNorm(d_sub)
        self.ln_t = nn.LayerNorm(d_sub)

        # Xavier init (tranh attention collapse ngay tu dau)
        for layer in [self.W_q_t, self.W_k_v, self.W_v_v,
                      self.W_q_v, self.W_k_t, self.W_v_t]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        feat_t: torch.Tensor,    # (N, d_t)
        feat_v: torch.Tensor,    # (N, d_v)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            E_roi_t: (N, d_sub) - text ROI guided by visual
            E_roi_v: (N, d_sub) - visual ROI guided by text
        """
        N = feat_t.size(0)

        # Virtual Token Segmentation (STAIRE2_v3.md)
        X_t = feat_t.view(N, self.N_t, self.d_sub)   # (N,  6, 64)
        X_v = feat_v.view(N, self.N_v, self.d_sub)   # (N, 64, 64)

        # ----- Text-guided Visual ROI (Text=Query, Visual=Key/Value) -----------
        Q_t = self.W_q_t(X_t)                          # (N, 6, d_k)
        K_v = self.W_k_v(X_v)                          # (N, 64, d_k)
        V_v = self.W_v_v(X_v)                          # (N, 64, d_sub)
        # (N,6,d_k) x (N,d_k,64) -> (N, 6, 64)
        attn_v = F.softmax(
            torch.bmm(Q_t, K_v.transpose(1, 2)) / self._scale, dim=-1
        )
        ctx_v   = torch.bmm(attn_v, V_v)               # (N, 6, d_sub)
        E_roi_v = self.ln_v(X_t + ctx_v).mean(dim=1)   # (N, d_sub)

        # ----- Visual-guided Text ROI (Visual=Query, Text=Key/Value) -----------
        Q_v = self.W_q_v(X_v)                          # (N, 64, d_k)
        K_t = self.W_k_t(X_t)                          # (N, 6, d_k)
        V_t = self.W_v_t(X_t)                          # (N, 6, d_sub)
        # (N,64,d_k) x (N,d_k,6) -> (N, 64, 6)
        attn_t = F.softmax(
            torch.bmm(Q_v, K_t.transpose(1, 2)) / self._scale, dim=-1
        )
        ctx_t   = torch.bmm(attn_t, V_t)               # (N, 64, d_sub)
        E_roi_t = self.ln_t(X_v + ctx_t).mean(dim=1)   # (N, d_sub)

        return E_roi_t, E_roi_v


# ============================================================================
# Utilities
# ============================================================================

def load_raw_features(path: str, device: torch.device) -> torch.Tensor:
    """Load features tu .pt / .npy / .npz"""
    logger.info(f"  Loading: {path}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Khong tim thay: {path}")
    ext = p.suffix.lower()
    if ext == ".pt":
        feat = torch.load(path, map_location="cpu")
    elif ext == ".npy":
        feat = torch.from_numpy(np.load(path))
    elif ext == ".npz":
        npz  = np.load(path)
        feat = torch.from_numpy(npz[list(npz.keys())[0]])
    elif ext in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        feat = torch.from_numpy(d) if isinstance(d, np.ndarray) else d
    else:
        raise ValueError(f"Dinh dang chua ho tro: {ext}")
    feat = feat.float().to(device)
    logger.info(f"  -> shape={tuple(feat.shape)}")
    return feat


def sanity_check(E: torch.Tensor, name: str, tol: float = 0.15) -> None:
    """Kiem tra decorrelation va mean_norm."""
    N, D = E.shape
    mean = E.mean(dim=0)
    Ec   = E - mean
    cov  = Ec.t().mm(Ec) / (N - 1)
    diag_mask    = torch.eye(D, dtype=torch.bool)
    mean_off     = cov[~diag_mask].abs().mean().item()
    mean_norm    = E.norm(dim=-1).mean().item()
    status = "OK" if mean_off < tol else "WARN"
    logger.info(
        f"  [Sanity {name}] {tuple(E.shape)} | "
        f"mean_norm={mean_norm:.3f} | "
        f"off_diag_cov={mean_off:.4f} [{status}]"
    )


# ============================================================================
# Main Pipeline
# ============================================================================

def process_dataset(
    dataset_name:     str,
    text_feat_path:   str,
    visual_feat_path: str,
    output_dir:       str,
    d_sub:            int   = 64,
    d_k:              int   = 32,
    zca_eps:          float = 1e-5,
    fuweight:         float = 0.6,
    batch_size:       int   = 1024,
    device:           torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Pipeline day du:
      0. Load raw features
      1. ZCA Whitening: text 384->64, visual 4096->64  (E_global)
      2. Bidirectional ROI Attention tren raw features  (E_roi)
      3. Fusion: E_fused = fuweight * E_global + (1-fuweight) * E_roi
      4. Export: .pt + .npy
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    # --- 0. Load ---
    logger.info("[Phase 0] Loading features...")
    text_raw   = load_raw_features(text_feat_path,   device)   # (N, 384)
    visual_raw = load_raw_features(visual_feat_path, device)   # (N, 4096)
    N = text_raw.size(0)
    assert visual_raw.size(0) == N
    logger.info(f"  N_items={N}")

    # --- 1. ZCA Whitening ---
    logger.info("[Phase 1] ZCA Whitening...")
    zca_t = ZCAWhiteningProjector(d_in=text_raw.size(1),   d_out=d_sub, eps=zca_eps)
    zca_t.fit(text_raw)
    E_global_t = zca_t.transform(text_raw).cpu()            # (N, 64)

    zca_v = ZCAWhiteningProjector(d_in=visual_raw.size(1), d_out=d_sub, eps=zca_eps)
    zca_v.fit(visual_raw)
    E_global_v = zca_v.transform(visual_raw).cpu()          # (N, 64)

    logger.info(
        f"  E_global_t {tuple(E_global_t.shape)} "
        f"mean_norm={E_global_t.norm(dim=-1).mean():.3f}"
    )
    logger.info(
        f"  E_global_v {tuple(E_global_v.shape)} "
        f"mean_norm={E_global_v.norm(dim=-1).mean():.3f}"
    )

    # --- 2. ROI Extraction ---
    logger.info("[Phase 2] Bidirectional Cross-Modal ROI Attention...")
    logger.info(
        f"  Tokens: text={text_raw.size(1)//d_sub}x{d_sub}, "
        f"visual={visual_raw.size(1)//d_sub}x{d_sub}"
    )
    attn = BidirectionalCrossModalAttention(
        d_t=text_raw.size(1), d_v=visual_raw.size(1),
        d_sub=d_sub, d_k=d_k,
    ).to(device).eval()

    roi_t_list, roi_v_list = [], []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            rt, rv = attn(text_raw[s:e], visual_raw[s:e])
            roi_t_list.append(rt.cpu())
            roi_v_list.append(rv.cpu())
            if (s // batch_size) % 5 == 0:
                logger.info(f"  ROI: {e}/{N}")

    E_roi_t = torch.cat(roi_t_list)   # (N, d_sub)
    E_roi_v = torch.cat(roi_v_list)   # (N, d_sub)
    logger.info(
        f"  E_roi_t {tuple(E_roi_t.shape)} "
        f"mean_norm={E_roi_t.norm(dim=-1).mean():.3f}"
    )
    logger.info(
        f"  E_roi_v {tuple(E_roi_v.shape)} "
        f"mean_norm={E_roi_v.norm(dim=-1).mean():.3f}"
    )

    # --- 3. Fusion (CLID fuweight) ---
    # CLID.py: fu_image = fuweight * image_emb + (1-fuweight) * roi_image
    logger.info(f"[Phase 3] Fusion: {fuweight}*Global + {1-fuweight}*ROI ...")
    E_fused_t = fuweight * E_global_t + (1 - fuweight) * E_roi_t
    E_fused_v = fuweight * E_global_v + (1 - fuweight) * E_roi_v

    # --- 4. Export ---
    logger.info("[Phase 4] Saving...")
    results = {
        "E_global_t": E_global_t,
        "E_global_v": E_global_v,
        "E_roi_t":    E_roi_t,
        "E_roi_v":    E_roi_v,
        "E_fused_t":  E_fused_t,
        "E_fused_v":  E_fused_v,
    }
    for name, tensor in results.items():
        pt_path = out / f"{name}.pt"
        torch.save(tensor, pt_path)
        logger.info(f"  Saved {pt_path}  {tuple(tensor.shape)}")

    np.save(out / "E_fused_t.npy", E_fused_t.numpy())
    np.save(out / "E_fused_v.npy", E_fused_v.numpy())
    logger.info(f"  [Done] Output: {out}")
    return results


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="STAIR-LIA Offline Preprocessing: ZCA + ROI Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",      required=True,
                   choices=["Baby", "Sports", "Electronics"])
    p.add_argument("--text_feat",    required=True,
                   help="Path to text features (N, 384): .pt/.npy")
    p.add_argument("--visual_feat",  required=True,
                   help="Path to visual features (N, 4096): .pt/.npy")
    p.add_argument("--output_dir",   default="preprocessed_lia")
    p.add_argument("--d_sub",        type=int,   default=64)
    p.add_argument("--d_k",          type=int,   default=32)
    p.add_argument("--zca_eps",      type=float, default=1e-5)
    p.add_argument("--fuweight",     type=float, default=0.6,
                   help="Weight of Global in fusion (CLID default=0.6)")
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--device",       default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--sanity_check", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = __import__("torch").device(args.device)

    logger.info("=" * 60)
    logger.info("STAIR-LIA Offline Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Dataset  : {args.dataset}")
    logger.info(f"Device   : {device}")
    logger.info(f"d_sub    : {args.d_sub}  |  d_k={args.d_k}")
    logger.info(f"fuweight : {args.fuweight}  (Global vs ROI)")
    logger.info(f"batch    : {args.batch_size}")
    logger.info(f"Output   : {args.output_dir}")

    t0 = time.time()
    results = process_dataset(
        dataset_name     = args.dataset,
        text_feat_path   = args.text_feat,
        visual_feat_path = args.visual_feat,
        output_dir       = args.output_dir,
        d_sub            = args.d_sub,
        d_k              = args.d_k,
        zca_eps          = args.zca_eps,
        fuweight         = args.fuweight,
        batch_size       = args.batch_size,
        device           = device,
    )

    if args.sanity_check:
        logger.info("\n[Sanity Checks]")
        for name, tensor in results.items():
            sanity_check(tensor, name)

    logger.info(f"\nDone in {time.time()-t0:.1f}s")
    logger.info(f"Files saved to: {args.output_dir}/{args.dataset}/")
    logger.info("")
    logger.info("=== Tich hop vao STAIR training ===")
    logger.info("Trong prepare() cua EnhancedSTAIR:")
    logger.info("  E_t = torch.load('preprocessed_lia/{ds}/E_fused_t.pt')")
    logger.info("  E_v = torch.load('preprocessed_lia/{ds}/E_fused_v.pt')")
    logger.info("  # Thay the mfeats_raw = [E_t, E_v]")
    logger.info("  # Bo qua buoc whitening() hoac dung E_fused truc tiep lam E_svd")


if __name__ == "__main__":
    main()
