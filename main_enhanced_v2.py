"""
main_enhanced_v2.py — STAIR-Enhanced v2a (ResOnly): Residual-Whitening Projector
================================================================================
Chien luoc: Giu nguyen SVD Whitening (Structural Prior, frozen) + them nhanh
Residual Projector hoc phan correction phi tuyen.

    e_i = whitening(x_i)  +  lambda_res * Delta_i
          (frozen baseline)      (learned, ||Delta||=1)

Cai tien ky thuat:
  - whitening() duoc giu nguyen nhu STAIR goc
  - res_projector hoc phan deviation nho tren nen warm-start SVD
  - Dual-Optimizer: AdamWSEvo cho User/Item embeddings, Adam rieng cho projector (lr=5e-3)
  - AMP (autocast + GradScaler) de tiet kiem VRAM tren T4/P100 Kaggle
  - init_warm_start_weights() khoi tao trong so projector tu SVD
  - Log lambda_res moi epoch de giam sat convergence cua projector
  - sure_trainpipe(self, batch_size) duoc cai dat day du
"""

from typing import Dict, Tuple

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import freerec

from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother
from models.residual_projector_v2 import ResidualWhiteningProjector, composite_embeddings

freerec.declare(version='1.0.1')

# ============================================================================
# Config (dong nhat voi main.py; them --lr-proj va --lambda-init)
# ============================================================================
cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3)
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1')
cfg.add_argument("--gamma",          type=float, default=0.2)
# V2-specific: learning rate cho residual projector
cfg.add_argument("--lr-proj",        type=float, default=5e-3,
                 help="Learning rate danh rieng cho ResidualWhiteningProjector (default: 5e-3)")
cfg.add_argument("--lambda-init",    type=float, default=0.1,
                 help="Gia tri khoi tao cua lambda_res (default: 0.1)")

cfg.set_defaults(
    description="EnhancedSTAIR-v2a",
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

# beta3: spectral decay parameter cho BSC Smoother (dong nhat voi main.py)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ============================================================================
# Model: EnhancedSTAIR_v2a (Residual-Whitening Architecture)
# ============================================================================
class EnhancedSTAIR_v2a(freerec.models.GenRecArch):
    """
    STAIR voi Residual Projector (v2a — ResOnly).

    FSC / BSC / BPR core KHONG THAY DOI tu bai bao goc.
    SVD Whitening DUOC GIU NGUYEN lam Structural Prior.
    ResidualWhiteningProjector chi them phan correction nho ben canh.

    e_i  = whitening(x_i)  +  lambda_res * Delta_i
    e_u  = R @ e_i    (giong STAIR goc)
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module('embeddings', nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim))

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )

        # Module v2a: Residual Projector (learnable correction)
        # Text=384-D, Visual=4096-D (xac nhan tu main.py + YAML)
        self.res_projector = ResidualWhiteningProjector(
            d_text=384,
            d_visual=4096,
            d_hidden=cfg.embedding_dim,
            lambda_init=cfg.lambda_init,
        )

        self.reset_parameters()
        self.prepare(dataset.path)      # goi whitening goc + warm-start + composite
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

    def reset_parameters(self):
        """Kaiming init cho moi linear layer; tiny normal cho embeddings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)

    def marked_params(self):
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {'params': self.Item.parameters(), 'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')},
            {'params': self.res_projector.parameters(), 'smoother': None},
        ]

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Ham whitening goc cua STAIR (copy y nguyen tu main.py, KHONG CHINH SUA).

        Buoc: center -> SVD -> lay top-D singular vectors -> scale.
        Ket qua: ma tran U_top * sqrt(N/D), moi hang co norm xap xi 1,
        cac cot truc giao (decorrelated spectrum) phu hop cho BSC Smoother.
        """
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """Tinh kNN graph tu cosine similarity (dong nhat voi main.py)."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """
        Ket hop SVD whitening (frozen) va residual correction (learnable).

        Quy trinh:
          1. Load raw features (text + visual)
          2. Xay mAdj (kNN graph) — y het main.py
          3. Tinh e_svd bang whitening() — y het main.py
          4. Warm-start projector tu raw features
          5. Tinh e_final = composite_embeddings(e_svd, Delta) — DIEM MOI
          6. Gan e_final vao Item.embeddings va User.embeddings
        """
        from freerec.utils import import_pickle

        mfeats_raw = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]
        text_feat   = mfeats_raw[0]   # (N, 384)  -- Sentence-BERT
        visual_feat = mfeats_raw[1]   # (N, 4096) -- deep CNN

        # ---- Buoc 1: Xay mAdj (kNN graph) — dong nhat voi main.py ----
        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats_raw, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # ---- Buoc 2: Tinh e_svd (Structural Prior, FROZEN) — giong main.py ----
        e_svd_parts = [self.whitening(feat) * k
                       for feat, k in zip(mfeats_raw, cfg.num_neighbors)]
        e_svd = sum(e_svd_parts) / sum(cfg.num_neighbors)   # (N, 64)
        print(f'[prepare] e_svd: shape={tuple(e_svd.shape)}, '
              f'mean_norm={e_svd.norm(dim=-1).mean():.3f}')

        # ---- Buoc 3: Warm-start Projector tu raw features ----
        print('[prepare] Running warm-start SVD for residual projector...')
        self.res_projector.init_warm_start_weights(text_feat, visual_feat)

        # ---- Buoc 4: Tinh e_final = e_svd + lambda_res * Delta ----
        with torch.no_grad():
            e_final = composite_embeddings(
                self.res_projector,
                text_feat, visual_feat, e_svd,
                n_items=self.Item.count,
                embedding_dim=cfg.embedding_dim,
            )   # (N, 64), mean_norm ~ sqrt(N/D)
        print(f'[prepare] e_final: shape={tuple(e_final.shape)}, '
              f'mean_norm={e_final.norm(dim=-1).mean():.3f}')
        self.Item.embeddings.weight.data.copy_(e_final)

        # ---- Buoc 5: User init theo R @ e_final (giong main.py voi mfeats) ----
        edge_index_ui = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui,
            size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ e_final)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """FSC/BSC smoothing (y het main.py)."""
        allEmbds  = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        features  = allEmbds
        smoothed  = allEmbds
        beta      = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """BPR loss (y het main.py)."""
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users]
        iposEmbds = itemEmbds[positives]
        inegEmbds = itemEmbds[negatives]
        return self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds, iposEmbds),
            torch.einsum('BKD,BKD->BK', userEmbds, inegEmbds),
        )

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)


# ============================================================================
# Coach: Dual-Optimizer + AMP + lambda_res logging
# ============================================================================
class CoachForEnhancedSTAIR_v2a(freerec.launcher.Coach):
    """
    Custom Coach cho v2a voi:
      - Dual-Optimizer: AdamWSEvo cho embeddings, Adam rieng cho projector
      - AMP: autocast + GradScaler de tiet kiem VRAM (T4/P100 Kaggle)
      - Logging lambda_res sau moi epoch de theo doi gradient flow
    """

    def set_optimizer(self):
        # === Optimizer 1: AdamWSEvo cho User + Item embeddings ===
        # Giu nguyen hyperparams cua baseline de BSC Smoother on dinh
        self.optimizer = AdamWSEvo(
            [
                {'params': self.model.User.parameters(), 'smoother': None},
                {
                    'params': self.model.Item.parameters(),
                    'smoother': Smoother(
                        self.model.mAdj, beta=cfg.beta3,
                        L=cfg.num_layers, aggr='neumann'
                    ),
                },
            ],
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

        # === Optimizer 2: Adam doc lap cho ResidualWhiteningProjector ===
        # lr cao hon 5x so voi embedding optimizer de projector hoc nhanh
        # weight_decay nho (1e-4) de tranh regularize qua manh lambda_res
        self.optimizer_proj = torch.optim.Adam(
            self.model.res_projector.parameters(),
            lr=self.cfg.lr_proj,        # default 5e-3 (5x embedding lr)
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            eps=1e-8,
        )

        # === GradScaler duy nhat cho ca 2 optimizer (AMP) ===
        use_amp = torch.cuda.is_available()
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)

        print(f'[Optimizer] AdamWSEvo lr={self.cfg.lr}, wd={self.cfg.weight_decay}')
        print(f'[Optimizer] Adam(proj)  lr={self.cfg.lr_proj}, wd=1e-4')
        print(f'[Optimizer] AMP enabled: {use_amp}')

    def train_per_epoch(self, epoch: int):
        """
        Training loop mot epoch voi:
          - torch.cuda.amp.autocast() cho moi forward pass
          - GradScaler.step() cho ca 2 optimizer
          - Log gia tri lambda_res trung binh va cuoi epoch
        """
        self.model.res_projector.train()
        lambda_vals = []

        for data in self.dataloader:
            data = self.dict_to_device(data)

            # Zero gradients cho ca 2 optimizer truoc moi batch
            self.optimizer.zero_grad()
            self.optimizer_proj.zero_grad()

            # Forward pass voi AMP
            with autocast(enabled=self.use_amp):
                loss = self.model(data)

            # Backward pass qua GradScaler
            self.scaler.scale(loss).backward()

            # Step ca 2 optimizer voi scaled gradients
            self.scaler.step(self.optimizer)
            self.scaler.step(self.optimizer_proj)
            self.scaler.update()

            # Ghi nhan lambda_res sau moi batch de theo doi xu huong
            lambda_vals.append(self.model.res_projector.lambda_res.item())

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS'],
            )

        # Log lambda_res summary sau khi het epoch
        if lambda_vals:
            lam_mean = sum(lambda_vals) / len(lambda_vals)
            lam_last = lambda_vals[-1]
            print(
                f'  [lambda_res @epoch {epoch:3d}]: '
                f'mean={lam_mean:.6f}, last={lam_last:.6f}',
                flush=True,
            )


# ============================================================================
# Entry Point
# ============================================================================
def main():
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = EnhancedSTAIR_v2a(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe  = model.sure_testpipe(cfg.ranking)

    coach = CoachForEnhancedSTAIR_v2a(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == '__main__':
    main()
