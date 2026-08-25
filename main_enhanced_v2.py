"""
main_enhanced_v2.py — STAIR-Enhanced v2a (ResOnly): Residual-Whitening Projector
================================================================================
Chien luoc: Giu nguyen SVD Whitening (Structural Prior, frozen) + them nhanh
Residual Projector hoc phan correction phi tuyen.

    e_i = whitening(x_i)  +  lambda_res * Delta_i
          (frozen baseline)      (learned, ||Delta||=1)

Cai tien ky thuat:
  - whitening() duoc giu nguyen nhu STAIR goc (structural prior)
  - res_projector hoc phan deviation nho tren nen warm-start SVD
  - Param groups: User/Item dung lr=1e-3 voi Smoother, res_projector dung lr=5e-3
  - Optimizer: AdamWSEvo on dinh tuyet doi tren FreeRec pipeline
  - init_warm_start_weights() khoi tao trong so projector tu SVD
  - Log lambda_res sau moi epoch de giam sat convergence
  - sure_trainpipe() cai dat day du cho BPR training
"""

from typing import Dict, Tuple

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
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
        """
        Param groups cho AdamWSEvo:
          - User embeddings: standard lr, no smoother
          - Item embeddings: standard lr, with Smoother on mAdj
          - res_projector: custom lr (5e-3), no smoother
        """
        return [
            {
                'params': self.User.parameters(),
                'smoother': None,
            },
            {
                'params': self.Item.parameters(),
                'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann'),
            },
            {
                'params': self.res_projector.parameters(),
                'smoother': None,
                'lr': cfg.lr_proj,
            },
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
# Coach: AdamWSEvo + Param Groups + lambda_res logging
# ============================================================================
class CoachForEnhancedSTAIR_v2a(freerec.launcher.Coach):
    """
    Coach on dinh cho v2a:
      - AdamWSEvo quan ly param groups voi per-group learning rate
      - Logging lambda_res sau moi epoch de theo doi convergence
    """

    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )
        print(f'[Optimizer] AdamWSEvo initialized with marked_params (base lr={self.cfg.lr}, lr_proj={self.cfg.lr_proj})')

    def train_per_epoch(self, epoch: int):
        self.model.res_projector.train()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction='mean', mode='train', pool=['LOSS'],
            )

        lam_val = self.model.res_projector.lambda_res.item()
        print(f'  [lambda_res @epoch {epoch:3d}]: {lam_val:.6f}', flush=True)


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
