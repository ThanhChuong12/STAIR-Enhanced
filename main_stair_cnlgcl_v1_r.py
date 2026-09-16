# -*- coding: utf-8 -*-
"""
main_stair_cnlgcl_v1_r.py — STAIR-CNLGCL v1-R Training Script (Giai đoạn 4)
==============================================================================
Cross-Component Synergy: NLGCL Loss-Level Contrastive Regularization
& Reweighted BSC Graph-Level Optimization.

Tích hợp trực giao (Orthogonal Integration) 2 cơ chế cải tiến:
  Cơ chế 1 — CNLGCL InfoNCE (Tầng Forward Pass / Loss-level):
    - Tương phản trực tiếp H^(0) ↔ H^(1), Fused Ops [4, B, D].
    - Percentile AMM, In-batch FNF Mask, Sign-Preserving Spectral Noise.
    - λ_cl = 0.008 (giảm từ 0.010, bù trừ gradient boost từ BSC).
  Cơ chế 2 — BSC-Reweight Engine (Tầng Backward Pass / Optimizer):
    - Multiplicative Consensus Boost: W_ij = W_base * (1 + α*q_m + β*q_b).
    - SPSD Symmetric Laplacian, Zero Edge Pruning, W ∈ [1.0, 3.6].
    - Smoother(mAdj_boosted) trong AdamWSEvo.

Cấu hình Dataset-Adaptive (Giai đoạn 4):
  Amazon Sports : full_ssb α=0.40 β=0.20 λ=0.008 FNF=Off  AMM=On
  Amazon Baby   : modal_only α=0.50 β=0.00 λ=0.008 τ_thresh=0.35 FNF=On AMM=On
  Amazon Elec   : full_ssb α=0.40 β=0.20 λ=0.010 FNF=Off  AMM=On

Usage:
    python main_stair_cnlgcl_v1_r.py --config configs/Amazon2014Sports_550_MMRec.yaml
    python main_stair_cnlgcl_v1_r.py --config configs/Amazon2014Baby_550_MMRec.yaml \\
        --ssb-mode modal_only --ssb-alpha 0.50 --ssb-beta 0.00 \\
        --tau-thresh 0.35 --use-fn-mask 1
    python main_stair_cnlgcl_v1_r.py --config configs/Amazon2014Electronics_550_MMRec.yaml \\
        --lambda-cl 0.010
"""

import gc
import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# Fix Windows console utf-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tối ưu CUDA memory fragmentation cho Electronics 63K items
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch Internal Compatibility Patches (Kaggle / Python 3.12 / PyTorch 2.x)
# ═════════════════════════════════════════════════════════════════════════════

# ── Patch 1: torch._utils & _get_device_index (Dynamo / DataParallel) ──
try:
    import torch._utils
except Exception:
    pass

if not hasattr(torch, '_utils'):
    try:
        import torch._utils_internal as _utils
        torch._utils = _utils
    except Exception:
        pass

if hasattr(torch, '_utils') and not hasattr(torch._utils, '_get_device_index'):
    def _get_device_index(device=None, optional=False, allow_cpu=False):
        if device is None:
            return torch.cuda.current_device() if torch.cuda.is_available() else 0
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except Exception:
                return 0
        return device.index if hasattr(device, 'index') and device.index is not None else 0
    torch._utils._get_device_index = _get_device_index

# ── Patch 2: torchdata.datapipes (FreeRec dependency) ──
try:
    import torchdata
    import torchdata.datapipes as dp
except Exception:
    dp = None

if dp is None or 'torchdata.datapipes' not in sys.modules:
    if 'torchdata' not in sys.modules:
        td = types.ModuleType('torchdata')
        sys.modules['torchdata'] = td
    else:
        td = sys.modules['torchdata']

    dp = types.ModuleType('torchdata.datapipes')
    td.datapipes = dp
    sys.modules['torchdata.datapipes'] = dp

# Ensure dp.iter exists
if not hasattr(dp, 'iter'):
    iter_mod = types.ModuleType('torchdata.datapipes.iter')
    dp.iter = iter_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod
else:
    iter_mod = dp.iter

# Ensure IterDataPipe exists
if not hasattr(iter_mod, 'IterDataPipe'):
    try:
        from torch.utils.data import IterDataPipe as _IDP
    except Exception:
        class _IDP(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter([])
    iter_mod.IterDataPipe = _IDP
    if 'torchdata.datapipes.iter' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.iter'], 'IterDataPipe', _IDP)
else:
    _IDP = getattr(iter_mod, 'IterDataPipe')

# Ensure IterableWrapper exists (FreeRec RowFilter requires dp.iter.IterableWrapper)
if not hasattr(iter_mod, 'IterableWrapper'):
    try:
        from torch.utils.data.datapipes.iter import IterableWrapper as _IW
    except Exception:
        _IW = None
    if _IW is None:
        class _IW(_IDP):
            def __init__(self, iterable=None):
                super().__init__()
                self.iterable = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self.iterable)
            def __len__(self):
                try:
                    return len(self.iterable)
                except Exception:
                    return 0
            def __getitem__(self, idx):
                if hasattr(self.iterable, '__getitem__'):
                    return self.iterable[idx]
                raise NotImplementedError
    iter_mod.IterableWrapper = _IW
    setattr(dp, 'IterableWrapper', _IW)
    if 'torchdata.datapipes.iter' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.iter'], 'IterableWrapper', _IW)
    if 'torchdata.datapipes' in sys.modules:
        setattr(sys.modules['torchdata.datapipes'], 'IterableWrapper', _IW)

# Ensure dp.map and MapDataPipe exist
if not hasattr(dp, 'map'):
    map_mod = types.ModuleType('torchdata.datapipes.map')
    dp.map = map_mod
    sys.modules['torchdata.datapipes.map'] = map_mod
else:
    map_mod = dp.map

if not hasattr(map_mod, 'MapDataPipe'):
    try:
        from torch.utils.data import MapDataPipe as _MDP
    except Exception:
        class _MDP(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                raise NotImplementedError
            def __len__(self):
                return 0
    map_mod.MapDataPipe = _MDP
    if 'torchdata.datapipes.map' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.map'], 'MapDataPipe', _MDP)

# Ensure functional_datapipe decorator exists on dp
if not hasattr(dp, 'functional_datapipe'):
    def functional_datapipe(name, enable_df_datapipes_support=False):
        def decorator(cls):
            def method(self, *args, **kwargs):
                return cls(self, *args, **kwargs)
            if hasattr(dp, 'iter') and hasattr(dp.iter, 'IterDataPipe'):
                setattr(dp.iter.IterDataPipe, name, method)
            if hasattr(dp, 'map') and hasattr(dp.map, 'MapDataPipe'):
                setattr(dp.map.MapDataPipe, name, method)
            try:
                if hasattr(torch.utils.data, 'IterDataPipe'):
                    setattr(torch.utils.data.IterDataPipe, name, method)
                if hasattr(torch.utils.data, 'MapDataPipe'):
                    setattr(torch.utils.data.MapDataPipe, name, method)
            except Exception:
                pass
            return cls
        return decorator
    dp.functional_datapipe = functional_datapipe

# ── Patch 3: torch_geometric (freerec.graph dependency) ──
if 'torch_geometric' not in sys.modules or not hasattr(sys.modules.get('torch_geometric', None), 'utils'):
    try:
        import torch_geometric
        import torch_geometric.utils
    except Exception:
        tg = types.ModuleType('torch_geometric')
        tg_utils = types.ModuleType('torch_geometric.utils')
        def _stub(*args, **kwargs):
            raise NotImplementedError("freerec.graph requires working torch-geometric")
        for _fn in ['coalesce', 'scatter', 'spmm', 'to_undirected', 'to_edge_index']:
            setattr(tg_utils, _fn, _stub)
        tg.utils = tg_utils
        sys.modules['torch_geometric'] = tg
        sys.modules['torch_geometric.utils'] = tg_utils


# ═════════════════════════════════════════════════════════════════════════════
# FreeRec & Optimizer & Model Imports
# ═════════════════════════════════════════════════════════════════════════════
# Idempotent DataPipe registration patch for FreeRec / Kaggle notebook reloads
try:
    from torch.utils.data.datapipes.datapipe import IterDataPipe as _NativeIterDP, MapDataPipe as _NativeMapDP
    for _cls in [_NativeIterDP, _NativeMapDP]:
        if hasattr(_cls, "register_datapipe_as_function"):
            _orig_reg = _cls.register_datapipe_as_function
            def _make_safe_reg(orig_fn):
                def _safe_reg(cls, function_name, cls_to_register, *args, **kwargs):
                    if hasattr(cls, "functions") and function_name in cls.functions:
                        try:
                            del cls.functions[function_name]
                        except Exception:
                            pass
                    return orig_fn.__func__(cls, function_name, cls_to_register, *args, **kwargs)
                return _safe_reg
            _cls.register_datapipe_as_function = classmethod(_make_safe_reg(_orig_reg))
except Exception:
    pass

import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.GD4.stair_cnlgcl_v1_r import BSC_Reweight_Engine, CNLGCL_Loss_v1R

freerec.declare(version='0.8.5')


# ═════════════════════════════════════════════════════════════════════════════
# Configuration Setup: STAIR-CNLGCL v1-R (NLGCL + BSC-Reweight)
# ═════════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()

# ── STAIR Baseline Parameters ──
cfg.add_argument("--embedding-dim", type=int, default=64,
                 help="Latent vector embedding dimension D (default: 64)")
cfg.add_argument("--num-layers", type=int, default=3,
                 help="Number of layers for FSC/BSC (default: 3)")
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated modality feature files")
cfg.add_argument("--num-neighbors", type=str, default='5-1',
                 help="kNN counts per modality, e.g. '5-1'")
cfg.add_argument("--gamma", type=float, default=0.2,
                 help="Spectral decay exponent for beta3 (default: 0.2)")

# ── CNLGCL v1-R Contrastive Parameters ──
cfg.add_argument("--tau", type=float, default=0.20,
                 help="Temperature tau for InfoNCE softmax (default: 0.20)")
cfg.add_argument("--alpha-dir", type=float, default=0.50,
                 help="Direction balance: alpha*L_{u->i} + (1-alpha)*L_{i->u} (default: 0.50)")
cfg.add_argument("--eps", type=float, default=0.08,
                 help="Sign-preserving noise amplitude epsilon (default: 0.08)")
cfg.add_argument("--tau-thresh", type=float, default=0.85,
                 help="FNF threshold: sim_modal > tau_thresh → masked (default: 0.85, 1.0=off)")
cfg.add_argument("--lambda-cl", type=float, default=0.008,
                 help="Constant CL weight after warmup (default: 0.008, giảm từ 0.010)")
cfg.add_argument("--warmup-epochs", type=int, default=50,
                 help="Linear warmup epochs for lambda_cl (default: 50)")
cfg.add_argument("--margin-max", type=float, default=0.02,
                 help="AMM hard upper bound for margin (default: 0.02 = 10%% of tau)")
cfg.add_argument("--use-amm", type=int, default=1,
                 help="Enable Adaptive Multimodal Margin (1=on, 0=off, default: 1)")
cfg.add_argument("--use-fn-mask", type=int, default=1,
                 help="Enable In-batch FNF Mask (1=on, 0=off, default: 1)")
cfg.add_argument("--use-fused-ops", type=int, default=1,
                 help="Enable Fused Tensor Operations [4,B,D] (1=on, 0=off, default: 1)")

# ── BSC-Reweight Engine Parameters (v5) with Dual-Alias Support ──
cfg.add_argument("--ssb-mode", "--mode", dest="ssb_mode", type=str, default="full_ssb",
                 choices=["full_ssb", "modal_only", "behavior_only", "baseline"],
                 help="Reweighting mode (default: full_ssb)")
cfg.add_argument("--ssb-alpha", "--alpha", dest="ssb_alpha", type=float, default=0.40,
                 help="Modal consensus boost weight alpha (default: 0.40)")
cfg.add_argument("--ssb-beta", "--beta", dest="ssb_beta", type=float, default=0.20,
                 help="Behavioral Ochiai boost weight beta (default: 0.20)")
cfg.add_argument("--ssb-tau-text", "--tau-text", "--tau_text", dest="ssb_tau_text", type=float, default=0.10,
                 help="Text similarity threshold tau_t (default: 0.10)")
cfg.add_argument("--ssb-tau-visual", "--tau-visual", "--tau_visual", dest="ssb_tau_visual", type=float, default=0.10,
                 help="Visual similarity threshold tau_v (default: 0.10)")
cfg.add_argument("--ssb-min-weight", "--min-weight", dest="ssb_min_weight", type=float, default=1.00,
                 help="Edge weight lower bound (default: 1.00)")
cfg.add_argument("--ssb-max-weight", "--max-weight", dest="ssb_max_weight", type=float, default=3.60,
                 help="Edge weight upper bound (default: 3.60)")
cfg.add_argument("--eval-chunk-size", type=int, default=512,
                 help="User chunk size for full-ranking evaluation to prevent peak VRAM spike (default: 512)")

cfg.set_defaults(
    description="STAIR-CNLGCL-v1-R",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=1024,
    optimizer='adamwsevo',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
cfg.compile()

# =========================================================================
# YAML Configuration Loader & Normalizer (CLI > YAML > Defaults)
# =========================================================================
# Ghi nhận các tham số được chỉ định tường minh trên CLI để ưu tiên tuyệt đối
cli_specified = set()
for a in sys.argv[1:]:
    if a.startswith("--"):
        clean_a = a.lstrip("-").split("=")[0].replace("-", "_").lower()
        cli_specified.add(clean_a)
        if clean_a.startswith("ssb_"):
            cli_specified.add(clean_a[4:])
        else:
            cli_specified.add(f"ssb_{clean_a}")

if getattr(cfg, "config", None) is not None:
    config_path = cfg.config
    if not os.path.isabs(config_path):
        if not os.path.exists(config_path):
            repo_root = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(repo_root, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}

        for raw_key, val in yaml_cfg.items():
            norm_key = raw_key.replace("-", "_").lower()

            if norm_key in cli_specified or f"ssb_{norm_key}" in cli_specified:
                print(f"[Config] Giữ nguyên giá trị CLI cho '{norm_key}' (không bị YAML ghi đè)")
                continue

            matched = False
            if hasattr(cfg, norm_key):
                setattr(cfg, norm_key, val)
                matched = True
            elif hasattr(cfg, f"ssb_{norm_key}"):
                setattr(cfg, f"ssb_{norm_key}", val)
                matched = True
            elif norm_key.startswith("ssb_") and hasattr(cfg, norm_key[4:]):
                setattr(cfg, norm_key[4:], val)
                matched = True
            elif hasattr(cfg, raw_key):
                setattr(cfg, raw_key, val)
                matched = True

            if matched:
                print(f"[Config] Nạp {norm_key} = {val} từ YAML")
            else:
                setattr(cfg, norm_key, val)
                print(f"[Config] Nạp {norm_key} = {val} từ YAML")
    else:
        print(f"[Config Cảnh báo] Không tìm thấy file config: {config_path}")

# Đồng bộ các alias sang thuộc tính chuẩn
if hasattr(cfg, "ssb_alpha") and not hasattr(cfg, "alpha"):
    cfg.alpha = cfg.ssb_alpha
if hasattr(cfg, "alpha") and not hasattr(cfg, "ssb_alpha"):
    cfg.ssb_alpha = cfg.alpha

if hasattr(cfg, "ssb_beta") and not hasattr(cfg, "beta"):
    cfg.beta = cfg.ssb_beta
if hasattr(cfg, "beta") and not hasattr(cfg, "ssb_beta"):
    cfg.ssb_beta = cfg.beta

if hasattr(cfg, "ssb_mode") and not hasattr(cfg, "mode"):
    cfg.mode = cfg.ssb_mode

# ── Dataset-Adaptive Smart Defaults (Giai đoạn 4 v1-R Tuned) ──
ds_name = getattr(cfg, 'dataset', '')
if 'Baby' in ds_name:
    if 'lambda_cl' not in cli_specified:
        cfg.lambda_cl = 0.005       # Giảm 37.5% lực InfoNCE để bảo toàn cấu trúc đa tạp
    if 'warmup_epochs' not in cli_specified:
        cfg.warmup_epochs = 100     # Kéo dài 100 epochs cho đồ thị mật độ cao
    if 'tau_thresh' not in cli_specified:
        cfg.tau_thresh = 0.50       # Tăng ngưỡng FNF để lọc nhiều cặp âm tính giả ngữ nghĩa
    if 'ssb_mode' not in cli_specified and 'mode' not in cli_specified:
        cfg.ssb_mode = "modal_only"
        cfg.mode = "modal_only"
    if 'ssb_beta' not in cli_specified and 'beta' not in cli_specified:
        cfg.ssb_beta = 0.00
        cfg.beta = 0.00
    if 'ssb_alpha' not in cli_specified and 'alpha' not in cli_specified:
        cfg.ssb_alpha = 0.50
        cfg.alpha = 0.50
elif 'Sports' in ds_name:
    if 'lambda_cl' not in cli_specified:
        cfg.lambda_cl = 0.008
    if 'warmup_epochs' not in cli_specified:
        cfg.warmup_epochs = 50
    if 'ssb_mode' not in cli_specified and 'mode' not in cli_specified:
        cfg.ssb_mode = "full_ssb"
        cfg.mode = "full_ssb"
elif 'Electronics' in ds_name:
    if 'lambda_cl' not in cli_specified:
        cfg.lambda_cl = 0.005       # Giảm xuống 0.005 để tránh nhiễu InfoNCE trên 63K items
    if 'warmup_epochs' not in cli_specified:
        cfg.warmup_epochs = 50
    if 'ssb_mode' not in cli_specified and 'mode' not in cli_specified:
        cfg.ssb_mode = "full_ssb"
        cfg.mode = "full_ssb"
    if 'eval_chunk_size' not in cli_specified:
        cfg.eval_chunk_size = 512

if isinstance(cfg.mfiles, str):
    cfg.mfiles = cfg.mfiles.split(',')
if isinstance(cfg.num_neighbors, str):
    cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# BSC Smoother spectral decay beta3 (tính SAU khi nạp gamma từ YAML)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═════════════════════════════════════════════════════════════════════════════
# STAIR-CNLGCL v1-R Model Architecture (GenRecArch — FreeRec Compatible)
# ═════════════════════════════════════════════════════════════════════════════
class STAIR_CNLGCL_v1R_Model(freerec.models.GenRecArch):
    """
    STAIR-CNLGCL v1-R Model Architecture (Giai đoạn 4):
    Tích hợp trực giao CNLGCL InfoNCE (Loss-level) và BSC-Reweight (Graph-level).

    Pipeline prepare():
      1. SVD Whitening & Modal Consistency Percentile Calibration.
      2. kNN Graph Construction (raw, coalesced).
      3. BSC-Reweight Engine → Boosted SPSD mAdj.
      4. Embedding Initialization (Items: whitened fusion 5:1, Users: R @ items).

    Runtime fit():
      L_total = L_BPR + λ_cl(t) * L_CNLGCL
      ∇ smoothed by BSC Smoother(mAdj_boosted) in AdamWSEvo.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module(
            'embeddings', nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            'embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )
        self.register_buffer('beta3', cfg.beta3)

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        # CNLGCL v1-R Contrastive Module
        self.cnlgcl_loss = CNLGCL_Loss_v1R(
            n_users       = self.User.count,
            n_items       = self.Item.count,
            tau           = cfg.tau,
            alpha_dir     = cfg.alpha_dir,
            eps           = cfg.eps,
            tau_thresh    = cfg.tau_thresh,
            lambda_cl     = cfg.lambda_cl,
            warmup_epochs = cfg.warmup_epochs,
            margin_max    = cfg.margin_max,
            use_amm       = bool(cfg.use_amm),
            use_fn_mask   = bool(cfg.use_fn_mask),
            use_fused_ops = bool(cfg.use_fused_ops),
        )
        self.last_cl_loss: Optional[float] = None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def marked_params(self):
        """Đăng ký Smoother với mAdj ĐÃ ĐƯỢC tăng cường từ BSC-Reweight Engine."""
        return [
            {
                'params': self.User.parameters(),
                'smoother': None,
            },
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(
                    self.mAdj, beta=cfg.beta3,
                    L=cfg.num_layers, aggr='neumann'
                ),
            },
        ]

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """SVD Whitening chuẩn tắc: U[:, :d] * sqrt(N/d), Cov = 1/d * I_d."""
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, dtype=torch.float32)
        else:
            feats = feats.float()
        feats = feats - feats.mean(0, keepdim=True)
        U, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return U[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """kNN graph trên không gian cosine — identical to STAIR baseline."""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def prepare(self, path: str):
        """
        Tiền xử lý offline (1 lần duy nhất trước Epoch 1):
          1. Nạp đặc trưng đa phương thức.
          2. SVD Whitening (tính 1 lần, tái sử dụng).
          3. Precompute Modal Consistency (Percentile 5%-95%).
          4. Xây dựng kNN Graph gốc (coalesced, chưa normalized).
          5. Trích xuất train_R (user-item interaction matrix).
          6. BSC-Reweight Engine → mAdj SPSD tăng cường.
          7. Khởi tạo Item & User Embeddings.
        """
        from freerec.utils import import_pickle

        # ─────────────────────────────────────────────────────────────────
        # 1. Nạp đặc trưng đa phương thức (kèm tìm kiếm dự phòng Kaggle)
        # ─────────────────────────────────────────────────────────────────
        mfeats = []
        for mfile in cfg.mfiles:
            mpath = os.path.join(path, mfile)
            if not os.path.exists(mpath):
                for cand in [
                    os.path.join(cfg.root, cfg.dataset, mfile),
                    os.path.join("/kaggle/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/data/Processed", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR/data", cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset, mfile),
                    os.path.join("data", cfg.dataset, mfile),
                    os.path.join("data/Processed", cfg.dataset, mfile),
                ]:
                    if os.path.exists(cand):
                        mpath = cand
                        break
            mfeats.append(import_pickle(mpath))

        # ─────────────────────────────────────────────────────────────────
        # 2. SVD Whitening (tính 1 lần, cache cho cả consistency lẫn init)
        # ─────────────────────────────────────────────────────────────────
        whitened_feats = [self.whitening(mfeat) for mfeat in mfeats]

        # ─────────────────────────────────────────────────────────────────
        # 3. Precompute Modal Consistency (Percentile 5%-95% Calibration)
        # ─────────────────────────────────────────────────────────────────
        if len(whitened_feats) >= 2:
            t_wn = F.normalize(whitened_feats[0], p=2, dim=-1)
            v_wn = F.normalize(whitened_feats[1], p=2, dim=-1)
            raw_consistency = (t_wn * v_wn).sum(dim=-1)  # [N_items], ∈ [-1, 1]

            c_lo = torch.quantile(raw_consistency, 0.05).item()
            c_hi = torch.quantile(raw_consistency, 0.95).item()
            cons_span = c_hi - c_lo + 1e-8
            modal_consistency = torch.clamp(
                (raw_consistency - c_lo) / cons_span, 0.0, 1.0
            )
            self.register_buffer('modal_consistency', modal_consistency)

            print(f"  [v1-R] Modal consistency (Raw): "
                  f"mean={raw_consistency.mean():.4f}, std={raw_consistency.std():.4f}, "
                  f"min={raw_consistency.min():.4f}, max={raw_consistency.max():.4f}")
            print(f"  [v1-R] Percentile Calibration (5%-95%): "
                  f"c_lo={c_lo:.4f}, c_hi={c_hi:.4f}, span={cons_span:.4f} "
                  f"-> Calibrated [0, 1]: mean={modal_consistency.mean():.4f}")
        else:
            self.register_buffer('modal_consistency', torch.zeros(1))
            self._has_modal_consistency = False

        # ─────────────────────────────────────────────────────────────────
        # 4. Xây dựng kNN Graph gốc (coalesced, chưa to_undirected/to_normalized)
        # ─────────────────────────────────────────────────────────────────
        # Phân giải phương thức text và visual cho BSC Engine
        text_feat = None
        vis_feat = None
        for mf, feat in zip(cfg.mfiles, mfeats):
            mf_lower = str(mf).lower()
            if 'text' in mf_lower:
                text_feat = feat
            elif 'vis' in mf_lower or 'image' in mf_lower:
                vis_feat = feat
        if text_feat is None:
            text_feat = mfeats[0]
        if vis_feat is None:
            vis_feat = mfeats[1] if len(mfeats) > 1 else mfeats[0]

        # Chuyển sang Tensor nếu cần (BSC Engine yêu cầu Tensor)
        if not isinstance(text_feat, torch.Tensor):
            text_feat = torch.tensor(text_feat, dtype=torch.float32)
        if not isinstance(vis_feat, torch.Tensor):
            vis_feat = torch.tensor(vis_feat, dtype=torch.float32)

        print(f"  [v1-R] Đang xây dựng đồ thị kNN gốc: k_neighbors={cfg.num_neighbors}...")
        raw_edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(raw_edge_index[0], dtype=torch.float)
        raw_edge_index, raw_edge_weight = freerec.graph.coalesce(
            raw_edge_index, edge_weight, reduce='sum'
        )

        # Loại bỏ triệt để các cạnh self-loop sau coalesce giữa các modality
        mask_self_loop = (raw_edge_index[0] != raw_edge_index[1])
        if not mask_self_loop.all():
            num_loops = (~mask_self_loop).sum().item()
            raw_edge_index = raw_edge_index[:, mask_self_loop]
            raw_edge_weight = raw_edge_weight[mask_self_loop]
            print(f"  [v1-R] Đã loại bỏ {num_loops:,} cạnh self-loop sau coalesce. Còn lại: {raw_edge_index.size(1):,} cạnh.")

        print(f"  [v1-R] kNN gốc hoàn tất: {raw_edge_index.size(1):,} cạnh "
              f"(W: {raw_edge_weight.min().item():.1f} - {raw_edge_weight.max().item():.1f})")

        # ─────────────────────────────────────────────────────────────────
        # 5. Trích xuất train_R (user-item interaction matrix scipy CSR)
        # ─────────────────────────────────────────────────────────────────
        edge_index_ui = self.dataset.train().to_bigraph(
            edge_type='u2i'
        )['u2i'].edge_index
        row_ui = edge_index_ui[0].cpu().numpy()
        col_ui = edge_index_ui[1].cpu().numpy()
        data_ui = np.ones(len(row_ui), dtype=np.float32)
        train_R = sp.csr_matrix(
            (data_ui, (row_ui, col_ui)),
            shape=(self.User.count, self.Item.count)
        )

        # ─────────────────────────────────────────────────────────────────
        # 6. BSC-Reweight Engine → mAdj SPSD tăng cường trọng số
        # ─────────────────────────────────────────────────────────────────
        ssb_mode = getattr(cfg, 'ssb_mode', 'full_ssb')
        engine = BSC_Reweight_Engine(
            mode=ssb_mode,
            alpha=cfg.ssb_alpha,
            beta=cfg.ssb_beta,
            tau_t=cfg.ssb_tau_text,
            tau_v=cfg.ssb_tau_visual,
            min_weight=cfg.ssb_min_weight,
            max_weight=cfg.ssb_max_weight,
            verbose=True,
        )

        mAdj_boosted = engine.build_boosted_mAdj(
            text_feats=text_feat,
            vis_feats=vis_feat,
            train_user_item_matrix=train_R,
            raw_knn_adj=raw_edge_index,
            raw_edge_weight=raw_edge_weight,
            num_items=self.Item.count,
            target_device=cfg.device,
            all_modal_feats=[
                (f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32))
                for f in mfeats
            ],
        )
        self.register_buffer('mAdj', mAdj_boosted)
        assert self.mAdj.is_sparse_csr, "mAdj phải ở định dạng sparse CSR!"
        assert self.mAdj._nnz() > 0, "mAdj không được rỗng!"
        print(f"  [v1-R] mAdj SPSD đã đăng ký: nnz={self.mAdj._nnz():,}, "
              f"device={self.mAdj.device}, mode={ssb_mode}")

        # ─────────────────────────────────────────────────────────────────
        # 7. Khởi tạo Item & User Embeddings từ SVD Whitening
        # ─────────────────────────────────────────────────────────────────
        mfeats_init = sum(
            w * k for w, k in zip(whitened_feats, cfg.num_neighbors)
        ).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        # User Embeddings: R_norm @ item_embeddings (identical to STAIR)
        edge_index_norm, edge_weight_norm = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R_torch = torch.sparse_coo_tensor(
            edge_index_norm, edge_weight_norm,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        user_profiles_init = R_torch @ mfeats_init
        self.User.embeddings.weight.data.copy_(user_profiles_init)
        print("  [v1-R] Item & User embeddings initialized from SVD Whitening.")

        # Register whitened modal features cho MFNA (FNF Mask)
        self.register_buffer('item_modals_raw', mfeats_init.detach().clone())

        # ─── Cleanup ──────────────────────────────────────────────────
        del (raw_edge_index, raw_edge_weight, edge_weight, mAdj_boosted,
             train_R, whitened_feats, mfeats, mfeats_init,
             R_torch, edge_index_norm, edge_weight_norm, user_profiles_init)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    # ═════════════════════════════════════════════════════════════════════
    # encode(): Forward Stepwise Convolution + Layer Embeds Capture
    # ═════════════════════════════════════════════════════════════════════
    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        FSC with intermediate layer capture.
        Returns:
            userEmbds:    (N_u, D) final aggregated user representations.
            itemEmbds:    (N_i, D) final aggregated item representations.
            layer_embeds: [H^0, H^1, ..., H^L] per-layer intermediates.
        """
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )
        layer_embeds = [allEmbds]
        features = allEmbds
        smoothed = allEmbds

        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
            layer_embeds.append(features)

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation encode (zero layer_embeds overhead)."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight),
            dim=0,
        )
        features = allEmbds
        smoothed = allEmbds
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    # ═════════════════════════════════════════════════════════════════════
    # fit(): BPR Loss + CNLGCL InfoNCE Loss
    # ═════════════════════════════════════════════════════════════════════
    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Training step:
        L_total = L_BPR + λ_cl(t) * L_CNLGCL_v1R
        """
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users     = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # 1. Pairwise BPR Ranking Loss
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]),
        )

        # 2. CNLGCL v1-R Contrastive Loss (with AMM + FNF + Fused Ops)
        if self.training:
            beta = (1.0 - self.beta3).to(userEmbds.device)
            i_mod = self.item_modals_raw if hasattr(self, 'item_modals_raw') else None

            # Modal consistency cho AMM
            m_cons = self.modal_consistency if (
                hasattr(self, 'modal_consistency')
                and not hasattr(self, '_has_modal_consistency')
            ) else None

            weighted_cl_loss, raw_cl_loss = self.cnlgcl_loss(
                layer_embeds      = layer_embeds,
                users             = users,
                positives         = positives,
                beta              = beta,
                item_modals       = i_mod,
                modal_consistency = m_cons,
            )
            self.last_cl_loss = raw_cl_loss
            return rec_loss + weighted_cl_loss

        return rec_loss

    # ═════════════════════════════════════════════════════════════════════
    # Evaluation Methods
    # ═════════════════════════════════════════════════════════════════════
    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        B = userEmbds.size(0)
        N = itemEmbds.size(0)
        chunk_size = getattr(cfg, 'eval_chunk_size', 512)
        if B > chunk_size and N > 15000:
            scores_list = []
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                u_chunk = userEmbds[start:end]
                scores_list.append(torch.einsum('BKD,ND->BN', u_chunk, itemEmbds))
            return torch.cat(scores_list, dim=0)
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ═════════════════════════════════════════════════════════════════════════════
# Coach Class for STAIR-CNLGCL v1-R
# ═════════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_CNLGCL_v1R(freerec.launcher.Coach):

    def set_optimizer(self):
        opt_name = getattr(self.cfg, 'optimizer', 'adamwsevo').lower()
        if opt_name == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"CoachForSTAIR_CNLGCL_v1R does not support '{opt_name}' optimizer"
            )

    def train_per_epoch(self, epoch: int):
        self.model.train()
        self.model.cnlgcl_loss.update_epoch(epoch + 1)
        total_cl_loss = 0.0
        cl_batches = 0

        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction="mean", mode='train', pool=['LOSS'],
            )

            if hasattr(self.model, 'last_cl_loss') and self.model.last_cl_loss is not None:
                total_cl_loss += self.model.last_cl_loss
                cl_batches += 1

        # Logging mỗi 10 epoch
        if cl_batches > 0:
            avg_cl_loss = total_cl_loss / float(cl_batches)
            curr_lambda, margin_max = self.model.cnlgcl_loss.get_current_params()
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.cfg.epochs:
                ssb_mode = getattr(self.cfg, 'ssb_mode', 'full_ssb')
                print(
                    f"  [v1-R Epoch {epoch + 1:03d}] "
                    f"lambda: {curr_lambda:.5f} | "
                    f"avg_cl_loss: {avg_cl_loss:.6f} | "
                    f"margin_max: {margin_max:.4f} | "
                    f"ssb_mode: {ssb_mode}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# Main Execution Entry Point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ssb_mode = getattr(cfg, 'ssb_mode', 'full_ssb')

    print("=" * 90)
    print("🚀 STAIR-CNLGCL v1-R (Giai đoạn 4) — Cross-Component Integration")
    print(f"   * Dataset         : {cfg.dataset}")
    print(f"   * BSC-Reweight    : mode={ssb_mode}, α={cfg.ssb_alpha}, β={cfg.ssb_beta}")
    print(f"   * CNLGCL          : λ_cl={cfg.lambda_cl}, τ={cfg.tau}, ε={cfg.eps}")
    print(f"   * AMM             : {'ON' if cfg.use_amm else 'OFF'}, margin_max={cfg.margin_max}")
    print(f"   * FNF Mask        : {'ON' if cfg.use_fn_mask else 'OFF'}, τ_thresh={cfg.tau_thresh}")
    print(f"   * Fused Ops       : {'ON' if cfg.use_fused_ops else 'OFF'}")
    print(f"   * STAIR Config    : L={cfg.num_layers}, D={cfg.embedding_dim}, γ={cfg.gamma}")
    print(f"   * Optimizer       : {cfg.optimizer}, LR={cfg.lr}, WD={cfg.weight_decay}")
    print(f"   * Warmup          : {cfg.warmup_epochs} epochs")
    print("=" * 90)

    # ── Auto-bridge dataset cho FreeRec/Kaggle ──
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (
        os.path.isdir(processed_dir) and not os.listdir(processed_dir)
    ):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join("data", cfg.dataset),
            os.path.join("data/Processed", cfg.dataset),
        ]
        for cand in candidates:
            if (os.path.exists(cand) and os.path.isdir(cand)
                    and os.path.abspath(cand) != os.path.abspath(processed_dir)
                    and len(os.listdir(cand)) > 0):
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                    print(f"[DataSet] >>> Auto-bridged symlink: {cand} -> {processed_dir}")
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                    print(f"[DataSet] >>> Auto-bridged copied: {cand} -> {processed_dir}")
                break

    # ── Robust dataset loading ──
    tasktag = getattr(cfg, 'tasktag', None) or getattr(freerec.data.tags, 'MATCHING', None)
    if hasattr(freerec.data.datasets, 'RecDataSet'):
        freerec.data.datasets.RecDataSet.TASK = tasktag
    if hasattr(freerec.data.datasets, 'base') and hasattr(freerec.data.datasets.base, 'BaseSet'):
        freerec.data.datasets.base.BaseSet.TASK = tasktag

    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            try:
                from freerec.data.datasets.base import MatchingRecDataSet
                dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
            except Exception:
                dataset = freerec.data.datasets.RecDataSet(
                    cfg.root, cfg.dataset, tasktag=tasktag
                )
    else:
        try:
            from freerec.data.datasets.base import MatchingRecDataSet
            dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(
                cfg.root, cfg.dataset, tasktag=tasktag
            )

    if not hasattr(dataset, 'TASK') or dataset.TASK is None:
        dataset.TASK = tasktag

    # ── Build Model & Pipelines ──
    model = STAIR_CNLGCL_v1R_Model(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_CNLGCL_v1R(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )

    # ── Training ──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    coach.fit()

    # ── VRAM Telemetry ──
    if torch.cuda.is_available():
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print("=" * 90)
        print("[VRAM TELEMETRY — PYTORCH ALLOCATOR (AUTHOR PAPER METHOD)]")
        print(f"  * Pure Tensor Peak (max_memory_allocated) : {max_alloc_mb:.2f} MB")
        print(f"  * Peak Reserved Memory (max_memory_reserved): {max_res_mb:.2f} MB")
        print("=" * 90)


if __name__ == '__main__':
    main()
