# -*- coding: utf-8 -*-
"""
main_stair_ne_nlgcl_3v1.py — STAIR-NE-NLGCL v3.1 Training Script
==================================================================
Kế thừa toàn bộ từ v5+ (v3-Refined) và bổ sung 2 cải tiến:

1. Adaptive Multimodal Margin (AMM):
   - Precompute text-visual consistency trước whitening.
   - Thêm margin thích ứng vào InfoNCE hướng u→i.
   - m₀ = 0.05, m_max = 0.02 (10% của τ = 0.20).

2. Fused Tensor Operations:
   - Gộp 4x noise injection + 4x normalize thành 1 batch [4, B, D].

Usage:
    python main_stair_ne_nlgcl_3v1.py --config configs/Amazon2014Baby_550_MMRec.yaml
    python main_stair_ne_nlgcl_3v1.py --config configs/Amazon2014Sports_550_MMRec.yaml
    python main_stair_ne_nlgcl_3v1.py --config configs/Amazon2014Electronics_550_MMRec.yaml
"""

import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# ── PyTorch Internal Compatibility Patch (_utils & _get_device_index for Dynamo) ──
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

# ── Compatibility Patch for torchdata in PyTorch 2.x / Python 3.12 / Kaggle ──
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

# Ensure IterableWrapper exists (CRITICAL: FreeRec RowFilter type annotation requires dp.iter.IterableWrapper)
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

# Ensure torch_geometric exists (required by freerec.graph)
try:
    import torch_geometric
except Exception:
    import subprocess
    print("  [Auto-Provision] Đang cài đặt thư viện phụ thuộc: torch-geometric...")
    try:
        TORCH_VER = torch.__version__.split('+')[0]
        CUDA_TAG  = 'cu' + torch.version.cuda.replace('.', '') if (torch.cuda.is_available() and torch.version.cuda) else 'cpu'
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q', 'torch-geometric',
            '-f', f'https://data.pyg.org/whl/torch-{TORCH_VER}+{CUDA_TAG}.html'
        ], check=False)
    except Exception:
        pass
    try:
        import torch_geometric
    except Exception:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-geometric'], check=False)
        try:
            import torch_geometric
        except Exception:
            pass

# If torch_geometric or torch_geometric.utils is still missing, provide stub for freerec.graph
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

import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

from models.stair_ne_nlgcl_3v1 import STAIR_NE_NLGCL_3v1

freerec.declare(version='0.8.5')

# ═════════════════════════════════════════════════════════════════════════════
# Configuration Setup: STAIR Baseline + STAIR-NE-NLGCL v3.1
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

# ── STAIR-NE-NLGCL v5+ Parameters (inherited) ──
cfg.add_argument("--tau", type=float, default=0.20,
                 help="Temperature tau for InfoNCE softmax (default: 0.20)")
cfg.add_argument("--alpha-dir", type=float, default=0.50,
                 help="Direction balance: alpha*L_{u->i} + (1-alpha)*L_{i->u} (default: 0.50)")
cfg.add_argument("--eps", type=float, default=0.08,
                 help="Sign-preserving noise amplitude epsilon (default: 0.08)")
cfg.add_argument("--tau-thresh", type=float, default=0.85,
                 help="Semantic similarity threshold for false negative masking (default: 0.85)")
cfg.add_argument("--lambda-cl", type=float, default=0.010,
                 help="Constant contrastive loss weight (default: 0.010)")
cfg.add_argument("--gamma-h", type=float, default=0.15,
                 help="Linear HANS hardness penalty coefficient (default: 0.15)")
cfg.add_argument("--warmup-epochs", type=int, default=50,
                 help="Warmup epochs for lambda (default: 50)")

# ── v3.1 NEW: Adaptive Multimodal Margin Parameters ──
cfg.add_argument("--margin-coef", type=float, default=0.05,
                 help="[v3.1] AMM base margin coefficient (default: 0.05)")
cfg.add_argument("--margin-max", type=float, default=0.02,
                 help="[v3.1] AMM hard upper bound for margin (default: 0.02, 10%% of tau)")

cfg.set_defaults(
    description="STAIR-NE-NLGCL-v3.1",
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

cfg.mfiles        = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# BSC Smoother spectral decay beta3
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═════════════════════════════════════════════════════════════════════════════
# STAIR-NE-NLGCL v3.1 Model Architecture
# ═════════════════════════════════════════════════════════════════════════════
class STAIR_NE_NLGCL_3v1_Model(freerec.models.GenRecArch):
    """
    STAIR-NE-NLGCL v3.1 Model Architecture:
    Kế thừa toàn bộ v5+ và bổ sung:
    1. Adaptive Multimodal Margin (AMM) trên hướng u→i
    2. Fused Tensor Operations cho noise injection + normalization

    Pipeline prepare() bổ sung:
    - Precompute modal_consistency = cosine(whiten(text), whiten(visual)) per item
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

        # STAIR-NE-NLGCL v3.1 Contrastive Module
        self.ne_nlgcl_3v1 = STAIR_NE_NLGCL_3v1(
            n_users       = self.User.count,
            n_items       = self.Item.count,
            tau           = cfg.tau,
            alpha_dir     = cfg.alpha_dir,
            eps           = cfg.eps,
            tau_thresh    = cfg.tau_thresh,
            lambda_cl     = cfg.lambda_cl,
            gamma_h       = cfg.gamma_h,
            warmup_epochs = cfg.warmup_epochs,
            margin_coef   = cfg.margin_coef,    # [v3.1]
            margin_max    = cfg.margin_max,      # [v3.1]
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
        params = [
            {
                'params': self.User.parameters(),
                'smoother': None
            },
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')
            },
        ]
        return params

    def whitening(self, feats: torch.Tensor):
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, dtype=torch.float32)
        else:
            feats = feats.float()
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
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
        from freerec.utils import import_pickle

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
        # [v3.1] Precompute Modal Consistency TRƯỚC khi whitening
        # Cosine similarity giữa text và visual features sau whitening riêng lẻ
        # ─────────────────────────────────────────────────────────────────
        # [v3.1] Precompute Robust Percentile-Calibrated Modal Consistency
        # Cosine similarity giữa text và visual features sau whitening riêng lẻ
        # Áp dụng chuẩn hóa bách phân vị 5% - 95% (Percentile Calibration)
        # ─────────────────────────────────────────────────────────────────
        if len(mfeats) >= 2:
            t_w = self.whitening(mfeats[0])    # (N_items, D) whitened text
            v_w = self.whitening(mfeats[1])    # (N_items, D) whitened visual
            t_wn = F.normalize(t_w, p=2, dim=-1)
            v_wn = F.normalize(v_w, p=2, dim=-1)
            raw_consistency = (t_wn * v_wn).sum(dim=-1)   # [N_items], ∈ [-1, 1]
            
            # Robust Percentile (5th - 95th) Calibration:
            c_lo = torch.quantile(raw_consistency, 0.05).item()
            c_hi = torch.quantile(raw_consistency, 0.95).item()
            cons_span = c_hi - c_lo + 1e-8
            
            # Chuẩn hóa về [0, 1] và khóa chặt 2 đầu (loại bỏ hoàn toàn rủi ro outlier)
            modal_consistency = torch.clamp((raw_consistency - c_lo) / cons_span, 0.0, 1.0)
            self.register_buffer('modal_consistency', modal_consistency)
            
            print(f"  [v3.1] Modal consistency stats (Raw): "
                  f"mean={raw_consistency.mean():.4f}, std={raw_consistency.std():.4f}, "
                  f"min={raw_consistency.min():.4f}, max={raw_consistency.max():.4f}")
            print(f"  [v3.1] Percentile Calibration (5%-95%): "
                  f"c_lo={c_lo:.4f}, c_hi={c_hi:.4f}, span={cons_span:.4f} "
                  f"-> Calibrated [0, 1]: mean={modal_consistency.mean():.4f}")
        else:
            self.register_buffer('modal_consistency', torch.zeros(1))
            self._has_modal_consistency = False

        # ─────────────────────────────────────────────────────────────────
        # kNN Graph Construction (identical to v5+)
        # ─────────────────────────────────────────────────────────────────
        edge_index = torch.cat(
            [self.get_knn_graph(feats, k)
             for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(
            edge_index, edge_weight, reduce='sum'
        )
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight, reduce='max'
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight, normalization='sym'
        )
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Whitened modal feature initialization (identical to v5+)
        mfeats_w = [
            self.whitening(mfeat) * k
            for mfeat, k in zip(mfeats, cfg.num_neighbors)
        ]
        mfeats_init = sum(mfeats_w).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        edge_index_ui = self.dataset.train().to_bigraph(
            edge_type='u2i'
        )['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        user_profiles_init = R @ mfeats_init
        self.User.embeddings.weight.data.copy_(user_profiles_init)

        # Register raw modal features for MFNA (identical to v5+)
        self.register_buffer('item_modals_raw', mfeats_init.detach().clone())

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward Stepwise Convolution with Layer Intermediates capture.
        Returns:
            userEmbds:    (N_u, D) final aggregated user representations
            itemEmbds:    (N_i, D) final aggregated item representations
            layer_embeds: [H^0, H^1, ..., H^L] per-layer representations
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
        """Evaluation encode function (zero overhead)."""
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

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """
        Training step:
        L_total = L_BPR + lambda_cl(t) * L_NE-NLGCL_v3.1
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

        # 2. STAIR-NE-NLGCL v3.1 Contrastive Loss (with AMM)
        if self.training:
            beta = (1.0 - self.beta3).to(userEmbds.device)
            i_mod = self.item_modals_raw if hasattr(self, 'item_modals_raw') else None

            # [v3.1] Pass modal_consistency to enable Adaptive Multimodal Margin
            m_cons = self.modal_consistency if (
                hasattr(self, 'modal_consistency')
                and not hasattr(self, '_has_modal_consistency')
            ) else None

            weighted_cl_loss, raw_cl_loss = self.ne_nlgcl_3v1(
                layer_embeds      = layer_embeds,
                users             = users,
                positives         = positives,
                beta              = beta,
                item_modals       = i_mod,
                modal_consistency = m_cons,     # [v3.1]
            )
            self.last_cl_loss = raw_cl_loss
            return rec_loss + weighted_cl_loss

        return rec_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ═════════════════════════════════════════════════════════════════════════════
# Coach Class for STAIR-NE-NLGCL v3.1
# ═════════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_NE_NLGCL_3v1(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"CoachForSTAIR_NE_NLGCL_3v1 does not support {self.cfg.optimizer} optimizer"
            )

    def train_per_epoch(self, epoch: int):
        self.model.train()
        self.model.ne_nlgcl_3v1.update_epoch(epoch + 1)
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

        if cl_batches > 0:
            avg_cl_loss = total_cl_loss / float(cl_batches)
            gamma_h, curr_lambda = self.model.ne_nlgcl_3v1.get_current_params()
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.cfg.epochs:
                print(
                    f"  [v3.1 Epoch {epoch + 1:03d}] gamma_h: {gamma_h:.4f} | "
                    f"lambda: {curr_lambda:.5f} | avg_cl_loss: {avg_cl_loss:.6f} | "
                    f"margin_coef: {self.model.ne_nlgcl_3v1.margin_coef:.4f}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# Main Execution Entry Point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Auto-bridge dataset for FreeRec:
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (os.path.isdir(processed_dir) and not os.listdir(processed_dir)):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join("data", cfg.dataset),
        ]
        for cand in candidates:
            if os.path.exists(cand) and os.path.isdir(cand) and os.path.abspath(cand) != os.path.abspath(processed_dir) and len(os.listdir(cand)) > 0:
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                    print(f"[DataSet] >>> Auto-bridged symlink: {cand} -> {processed_dir}")
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                    print(f"[DataSet] >>> Auto-bridged copied: {cand} -> {processed_dir}")
                break

    # Robust dataset loading:
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

    model = STAIR_NE_NLGCL_3v1_Model(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR_NE_NLGCL_3v1(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    coach.fit()

    if torch.cuda.is_available():
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_res_mb   = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print("=" * 80)
        print("[VRAM TELEMETRY — PYTORCH ALLOCATOR (AUTHOR PAPER METHOD)]")
        print(f"  * Pure Tensor Peak (max_memory_allocated) : {max_alloc_mb:.2f} MB")
        print(f"  * Peak Reserved Memory (max_memory_reserved): {max_res_mb:.2f} MB")
        print("=" * 80)


if __name__ == '__main__':
    main()
