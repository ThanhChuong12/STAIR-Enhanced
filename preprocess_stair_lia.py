"""
preprocess_stair_lia.py
=======================
STAIR-LIA v3 — Offline Preprocessing Pipeline

Phase 1 — PyTorch ZCA Whitening (from CFDTBD/Model-beauty/model-beauty.py TF->PyTorch):
    text (384-D) -> 64-D, visual (4096-D) -> 64-D
    Zero-phase: giữ nguyên hướng trục tọa độ, chỉ decorrelate + rescale.

Phase 2 — Offline Bidirectional Cross-Modal Attention ROI Extraction:
    Source: CLID-master/CLID/src/Cross_attention.py + models/CLID.py
    Virtual Token Segmentation (STAIRE2_v3.md):
        Text  384-D -> (6,  64) virtual tokens
        Visual 4096-D -> (64, 64) virtual patches
    Trích xuất E_roi_t (64-D) và E_roi_v (64-D) offline.

Phase 3 — Fusion (CLID fuweight):
    E_fused = fuweight * E_global + (1-fuweight) * E_roi

Export: E_global_t, E_global_v, E_roi_t, E_roi_v, E_fused_t, E_fused_v
        -> .pt + .npy
"""

import math, argparse, logging, time, os, pickle
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
class ZCAWhiteningProjector(nn.Module):
    """
    ZCA Whitening + Dimensionality Reduction.
    Zero-phase whitening giữ nguyên cấu trúc tọa độ ban đầu.
    """

    def __init__(self, d_in: int, d_out: int = 64, eps: float = 1e-5):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.eps   = eps
        self.register_buffer("zca_matrix", torch.zeros(d_in, d_in))
        self.register_buffer("mean_vec",   torch.zeros(d_in))
        self.register_buffer("scale",      torch.tensor(1.0))
        self._fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> None:
        logger.info(f"  [ZCA fit] shape={tuple(X.shape)}")
        t0 = time.time()
        mean = X.mean(dim=0)
        X_c  = X - mean
        N    = X.shape[0]

        cov  = X_c.t().mm(X_c) / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        inv_sqrt = 1.0 / torch.sqrt(eigenvalues.clamp(min=0.0) + self.eps)
        zca_mat  = eigenvectors.mm(torch.diag(inv_sqrt)).mm(eigenvectors.t())
        scale    = math.sqrt(N / self.d_out)

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
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước khi transform()!")
        X_c  = X - self.mean_vec
        X_w  = X_c.mm(self.zca_matrix)
        E    = X_w[:, -self.d_out:] * self.scale
        return E

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)


# ============================================================================
# Phase 2: Bidirectional Cross-Modal Attention ROI Extraction
# ============================================================================
class BidirectionalCrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention ROI Extraction (CLID-style, STAIRE2_v3 design).
    Virtual Token Segmentation:
        Text  (N, 384)  -> (N, N_t=6,  d_sub=64)
        Visual (N, 4096) -> (N, N_v=64, d_sub=64)
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
        assert d_t % d_sub == 0, f"d_t={d_t} phải chia hết cho d_sub={d_sub}"
        assert d_v % d_sub == 0, f"d_v={d_v} phải chia hết cho d_sub={d_sub}"
        self.N_t = d_t // d_sub
        self.N_v = d_v // d_sub
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

        for layer in [self.W_q_t, self.W_k_v, self.W_v_v,
                      self.W_q_v, self.W_k_t, self.W_v_t]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        feat_t: torch.Tensor,
        feat_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = feat_t.size(0)
        X_t = feat_t.view(N, self.N_t, self.d_sub)
        X_v = feat_v.view(N, self.N_v, self.d_sub)

        # Text-guided Visual ROI
        Q_t = self.W_q_t(X_t)
        K_v = self.W_k_v(X_v)
        V_v = self.W_v_v(X_v)
        attn_v = F.softmax(
            torch.bmm(Q_t, K_v.transpose(1, 2)) / self._scale, dim=-1
        )
        ctx_v   = torch.bmm(attn_v, V_v)
        E_roi_v = self.ln_v(X_t + ctx_v).mean(dim=1)

        # Visual-guided Text ROI
        Q_v = self.W_q_v(X_v)
        K_t = self.W_k_t(X_t)
        V_t = self.W_v_t(X_t)
        attn_t = F.softmax(
            torch.bmm(Q_v, K_t.transpose(1, 2)) / self._scale, dim=-1
        )
        ctx_t   = torch.bmm(attn_t, V_t)
        E_roi_t = self.ln_t(X_v + ctx_t).mean(dim=1)

        return E_roi_t, E_roi_v


# ============================================================================
# Utilities
# ============================================================================
def load_raw_features(path: str, device: torch.device) -> torch.Tensor:
    """Load features tu .pt / .npy / .npz / .pkl (tuong thich moi format)."""
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
        with open(path, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, torch.Tensor):
            feat = d.cpu()
        elif isinstance(d, np.ndarray):
            feat = torch.from_numpy(d)
        else:
            feat = torch.tensor(d)
    else:
        raise ValueError(f"Dinh dang chua ho tro: {ext}")
    feat = feat.float().to(device)
    logger.info(f"  -> shape={tuple(feat.shape)}")
    return feat


def sanity_check(E: torch.Tensor, name: str, tol: float = 0.15) -> None:
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
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[Phase 0] Loading raw features...")
    text_raw   = load_raw_features(text_feat_path,   device)
    visual_raw = load_raw_features(visual_feat_path, device)
    N = text_raw.size(0)
    assert visual_raw.size(0) == N, f"Mismatch item count: text={N} vs vis={visual_raw.size(0)}"
    logger.info(f"  N_items={N}")

    logger.info("[Phase 1] ZCA Whitening...")
    zca_t = ZCAWhiteningProjector(d_in=text_raw.size(1),   d_out=d_sub, eps=zca_eps)
    zca_t.fit(text_raw)
    E_global_t = zca_t.transform(text_raw).cpu()

    zca_v = ZCAWhiteningProjector(d_in=visual_raw.size(1), d_out=d_sub, eps=zca_eps)
    zca_v.fit(visual_raw)
    E_global_v = zca_v.transform(visual_raw).cpu()

    logger.info("[Phase 2] Bidirectional Cross-Modal ROI Attention...")
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

    E_roi_t = torch.cat(roi_t_list)
    E_roi_v = torch.cat(roi_v_list)

    logger.info(f"[Phase 3] Fusion: {fuweight}*Global + {1-fuweight}*ROI ...")
    E_fused_t = fuweight * E_global_t + (1 - fuweight) * E_roi_t
    E_fused_v = fuweight * E_global_v + (1 - fuweight) * E_roi_v

    logger.info("[Phase 4] Saving tensors...")
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
    logger.info(f"  [Done] Output saved to: {out}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="STAIR-LIA Offline Preprocessing: ZCA + ROI")
    p.add_argument("--dataset",      required=True)
    p.add_argument("--text_feat",    required=True)
    p.add_argument("--visual_feat",  required=True)
    p.add_argument("--output_dir",   default="preprocessed_lia")
    p.add_argument("--d_sub",        type=int,   default=64)
    p.add_argument("--d_k",          type=int,   default=32)
    p.add_argument("--zca_eps",      type=float, default=1e-5)
    p.add_argument("--fuweight",     type=float, default=0.6)
    p.add_argument("--batch_size",   type=int,   default=1024)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sanity_check", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    logger.info("=" * 60)
    logger.info("STAIR-LIA Offline Preprocessing")
    logger.info("=" * 60)
    logger.info(f"Dataset  : {args.dataset}")
    logger.info(f"Device   : {device}")

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


if __name__ == "__main__":
    main()
