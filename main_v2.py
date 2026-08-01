"""
main_v2.py — STAIR + Modality-Adaptive kNN Graph Reweighting (STAIR-DyFuse)
==============================================================================
Cải tiến 2: STAIR-DyFuse (Modality-Adaptive Graph Reweighting)
  - Thay vì dùng reduce='max' để gộp textual + visual kNN graph (STAIR gốc),
    STAIR-DyFuse tính confidence score per-item dựa trên L2 norm của feature
    vector sau SVD whitening.
  - Mỗi item i nhận trọng số c_i^v (visual confidence) và c_i^t (textual
    confidence) thích nghi với đặc trưng riêng của item đó.
  - Trọng số cạnh được reweight: s_ij = c_i^v * s_ij^v + c_i^t * s_ij^t
  - Hoàn toàn parameter-free — không thêm tham số học, không đổi forward pass,
    không đổi hàm mất mát. Chỉ thay đổi cách xây dựng mAdj trong prepare().

Nguồn cảm hứng:
  - TAMER: dùng weighted sum (alpha*s_v + beta*s_t) với alpha tĩnh toàn cục.
  - NLGCL+: dùng norm vector feature làm confidence signal cho adaptive weighting.
  STAIR-DyFuse kết hợp: confidence signal từ norm (NLGCL+) + weighted sum (TAMER),
  nhưng tính per-item thay vì toàn cục.
"""

from typing import Dict, Tuple, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3, help="the number of layers for FSC/BSC")

cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl",
                 help="the files saving modality (textual first, visual second)")
cfg.add_argument("--num-neighbors", type=str, default='5-1', help="for kNN graph (text-visual)")
cfg.add_argument("--gamma", type=float, default=0.2)

# STAIR-DyFuse specific args
cfg.add_argument("--conf-temp", type=float, default=1.0,
                 help="temperature for softmax-scaled confidence score (1.0 = linear norm ratio)")
cfg.add_argument("--conf-mode", type=str, default='norm',
                 choices=['norm', 'softmax'],
                 help="confidence score mode: 'norm' (linear ratio) or 'softmax' (temperature-scaled)")

cfg.set_defaults(
    description="STAIR-v2 (Modality-Adaptive kNN Graph Reweighting — STAIR-DyFuse)",
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

cfg.mfiles = cfg.mfiles.split(',')
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split('-')))

# beta3 here is the 1 - beta_j for BSC (same as STAIR gốc)
cfg.beta3 = (0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)


class STAIR(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

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
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph(self, features: torch.Tensor, k: int = 5):
        """Standard kNN graph — returns edge_index only (binary edges)."""
        features = F.normalize(features, dim=-1)  # (N, D)
        sim = features @ features.t()             # (N, N)
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, k, symmetric=False
        )
        return edge_index

    def get_knn_graph_with_weights(self, features: torch.Tensor, k: int = 5):
        """
        kNN graph with actual cosine similarity as edge weights.
        Returns (edge_index, edge_weight) where edge_weight = cosine sim (clipped >= 0).
        """
        features_norm = F.normalize(features, dim=-1)  # (N, D)
        sim = features_norm @ features_norm.t()        # (N, N)
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        row, col = edge_index[0], edge_index[1]
        # Retrieve the actual similarity value for each selected edge
        edge_weight = sim[row, col].clamp(min=0.0)
        return edge_index, edge_weight

    def compute_confidence(self, feat_t: torch.Tensor, feat_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-item modality confidence scores.

        Two modes:
          'norm'    : c_v = ||f_v|| / (||f_v|| + ||f_t|| + eps)  [linear ratio]
          'softmax' : c_v = exp(||f_v||/tau) / (exp(||f_v||/tau) + exp(||f_t||/tau))
                      [temperature-scaled — amplifies difference between modalities]

        Returns:
            c_v  (N,) — visual confidence per item
            c_t  (N,) — textual confidence per item  (c_t = 1 - c_v)
        """
        eps = 1e-7
        norm_v = feat_v.norm(p=2, dim=-1)   # (N,)
        norm_t = feat_t.norm(p=2, dim=-1)   # (N,)

        if cfg.conf_mode == 'softmax':
            tau = cfg.conf_temp
            exp_v = torch.exp(norm_v / tau)
            exp_t = torch.exp(norm_t / tau)
            c_v = exp_v / (exp_v + exp_t + eps)
        else:  # 'norm' — default, linear ratio
            c_v = norm_v / (norm_v + norm_t + eps)

        c_t = 1.0 - c_v
        return c_v, c_t

    def prepare(self, path: str):
        from freerec.utils import import_pickle

        # Load raw (unwhitened) features for confidence computation
        raw_mfeats = [
            import_pickle(os.path.join(path, mfile))
            for mfile in cfg.mfiles
        ]
        feat_t_raw, feat_v_raw = raw_mfeats[0], raw_mfeats[1]  # textual, visual

        # ── Step 1: Compute per-item confidence scores ──────────────────────────
        # NOTE: Confidence computed on RAW features before whitening so the norm
        #       reflects natural signal strength of each modality.
        c_v, c_t = self.compute_confidence(feat_t_raw, feat_v_raw)
        # Log confidence distribution for diagnostics
        print(f"[DyFuse] Visual confidence — mean: {c_v.mean():.4f}, std: {c_v.std():.4f}, "
              f"min: {c_v.min():.4f}, max: {c_v.max():.4f}")
        print(f"[DyFuse] Text  confidence — mean: {c_t.mean():.4f}, std: {c_t.std():.4f}")

        # ── Step 2: Build per-modality kNN graphs with cosine sim weights ────────
        ei_t, ew_t = self.get_knn_graph_with_weights(feat_t_raw, cfg.num_neighbors[0])  # textual
        ei_v, ew_v = self.get_knn_graph_with_weights(feat_v_raw, cfg.num_neighbors[1])  # visual

        # ── Step 3: Item-level edge reweighting ──────────────────────────────────
        # For each directed edge (i -> j), scale weight by source-item's confidence
        row_t = ei_t[0]  # source items of textual edges
        row_v = ei_v[0]  # source items of visual edges

        ew_t = c_t[row_t] * ew_t  # text-dominant items upweight textual edges
        ew_v = c_v[row_v] * ew_v  # visual-dominant items upweight visual edges

        # ── Step 4: Merge and normalize ─────────────────────────────────────────
        edge_index = torch.cat([ei_t, ei_v], dim=1)
        edge_weight = torch.cat([ew_t, ew_v], dim=0)

        # Sum weights when same edge appears in both graphs (vs. max in baseline)
        edge_index, edge_weight = freerec.graph.coalesce(
            edge_index, edge_weight, reduce='sum'
        )
        # Symmetrize: max of (i->j) and (j->i) to keep graph undirected
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight, reduce='max'
        )
        # Symmetric Laplacian normalization (same as STAIR baseline)
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight,
            normalization='sym'
        )

        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.Item.count, self.Item.count)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # ── Step 5: Item / User embedding initialization (same as STAIR gốc) ────
        mfeats = [self.whitening(mfeat) * k for mfeat, k in zip(raw_mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward stepwise convolution (FSC) — identical to STAIR baseline."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )  # (N_users + N_items, D)

        features = allEmbds
        smoothed = allEmbds

        # FSC: dimension-wise step-weight smoothing
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """Training step — pure BPR loss (same as STAIR baseline, no extra loss)."""
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        userEmbds = userEmbds[users]      # (B, 1, D)
        iposEmbds = itemEmbds[positives]  # (B, 1, D)
        inegEmbds = itemEmbds[negatives]  # (B, K, D)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )
        return rec_loss

    def reset_ranking_buffers(self):
        """Called before evaluation."""
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]  # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]    # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForSTAIR(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=len(data[self.User]), reduction="mean",
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = STAIR(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()
