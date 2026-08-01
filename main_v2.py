"""
main_v2.py — STAIR-DyFuse: Modality-Adaptive kNN Graph Reweighting

Enhancement over STAIR baseline:
  - Per-item confidence scores c_v and c_t are derived from the L2-norm of raw
    modal feature vectors, reflecting natural signal strength of each modality.
  - Edge weights in the multimodal item graph (mAdj) are scaled by the source
    item's modality confidence: w_ij = c_i^v * s_ij^v + c_i^t * s_ij^t
  - Fully parameter-free: no learnable parameters added; only mAdj construction
    in prepare() is modified. Forward pass and loss remain identical to baseline.

Design reference:
  - TAMER  : global weighted multi-graph fusion (static alpha per modality)
  - NLGCL+ : L2-norm as per-sample confidence signal for adaptive weighting
  STAIR-DyFuse applies norm-based confidence at item level, not batch level.
"""

from typing import Dict, Tuple

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3,   help="number of FSC/BSC layers")
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl",
                                                 help="modal feature files (textual first, visual second)")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1', help="kNN neighbors per modality")
cfg.add_argument("--gamma",          type=float, default=0.2)
cfg.add_argument("--conf-mode",      type=str,   default='norm', choices=['norm', 'softmax'],
                                                 help="confidence mode: linear norm ratio or softmax-scaled")
cfg.add_argument("--conf-temp",      type=float, default=1.0,
                                                 help="temperature for softmax confidence (ignored in norm mode)")

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

# Dimension-wise step weight for BSC: beta_j = 1 - (0.1 + 0.9*(j/D)^gamma)
cfg.beta3 = (0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)


class STAIR(freerec.models.GenRecArch):

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.User.add_module("embeddings", nn.Embedding(self.User.count, cfg.embedding_dim))
        self.Item.add_module("embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim))

        # Symmetric-normalized user-item interaction graph for FSC
        self.register_buffer("Adj", self.dataset.train().to_normalized_adj(normalization='sym'))

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
        """Attach BSC smoother to item parameters for the custom optimizer."""
        return [
            {'params': self.User.parameters(), 'smoother': None},
            {'params': self.Item.parameters(),
             'smoother': Smoother(self.mAdj, beta=cfg.beta3, L=cfg.num_layers, aggr='neumann')},
        ]

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """SVD whitening: zero-mean, project to embedding_dim, rescale variance."""
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)

    def get_knn_graph_with_weights(self, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a kNN graph using cosine similarity as edge weights.
        Returns edge_index (2, E) and edge_weight (E,) with values clipped to [0, 1].
        """
        features_norm = F.normalize(features, dim=-1)
        sim = features_norm @ features_norm.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        row, col = edge_index[0], edge_index[1]
        edge_weight = sim[row, col].clamp(min=0.0)
        return edge_index, edge_weight

    def compute_confidence(self, feat_t: torch.Tensor, feat_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-item modality confidence scores from raw L2 norms.

        norm mode    : c_v = ||f_v|| / (||f_v|| + ||f_t|| + eps)
        softmax mode : c_v = softmax([||f_v||, ||f_t||] / tau)[0]

        Returns c_v (N,) and c_t (N,) where c_t = 1 - c_v.
        """
        eps   = 1e-7
        norm_v = feat_v.norm(p=2, dim=-1)
        norm_t = feat_t.norm(p=2, dim=-1)

        if cfg.conf_mode == 'softmax':
            exp_v = torch.exp(norm_v / cfg.conf_temp)
            exp_t = torch.exp(norm_t / cfg.conf_temp)
            c_v = exp_v / (exp_v + exp_t + eps)
        else:
            c_v = norm_v / (norm_v + norm_t + eps)

        return c_v, 1.0 - c_v

    def prepare(self, path: str):
        """
        Build DyFuse multimodal item graph (mAdj) and initialize embeddings.

        Pipeline:
          1. Compute per-item confidence scores from raw modal feature norms.
          2. Build cosine-similarity-weighted kNN graphs per modality.
          3. Scale edge weights by source-item confidence.
          4. Merge, symmetrize, and sym-normalize into mAdj.
          5. Initialize item/user embeddings from whitened modal features.
        """
        from freerec.utils import import_pickle

        raw_mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
        feat_t, feat_v = raw_mfeats[0], raw_mfeats[1]

        # Step 1: per-item confidence (computed on raw features to reflect true norm)
        c_v, c_t = self.compute_confidence(feat_t, feat_v)
        print(f"[DyFuse] c_v — mean: {c_v.mean():.4f}, std: {c_v.std():.4f}, "
              f"min: {c_v.min():.4f}, max: {c_v.max():.4f}")

        # Step 2: weighted kNN graphs (cosine similarity as edge weight)
        ei_t, ew_t = self.get_knn_graph_with_weights(feat_t, cfg.num_neighbors[0])
        ei_v, ew_v = self.get_knn_graph_with_weights(feat_v, cfg.num_neighbors[1])

        # Step 3: scale each edge by the source item's modality confidence
        ew_t = c_t[ei_t[0]] * ew_t
        ew_v = c_v[ei_v[0]] * ew_v

        # Step 4: merge, symmetrize, and normalize
        edge_index = torch.cat([ei_t, ei_v], dim=1)
        edge_weight = torch.cat([ew_t, ew_v], dim=0)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')

        mAdj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(self.Item.count, self.Item.count))
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Step 5: initialize embeddings (identical to STAIR baseline)
        mfeats = [self.whitening(feat) * k for feat, k in zip(raw_mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ mfeats)

    def sure_trainpipe(self, batch_size: int):
        return (self.dataset.train()
                .shuffled_pairs_source()
                .gen_train_sampling_neg_(num_negatives=1)
                .batch_(batch_size)
                .tensor_())

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward stepwise convolution (FSC) — identical to STAIR baseline."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )  # (N_users + N_items, D)

        features = allEmbds
        smoothed = allEmbds
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features

        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        """BPR training step — no additional loss terms."""
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]

        return self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds[users], itemEmbds[positives]),
            torch.einsum("BKD,BKD->BK", userEmbds[users], itemEmbds[negatives]),
        )

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        return torch.einsum(
            "BKD,ND->BN",
            self.ranking_buffer[self.User][data[self.User]],
            self.ranking_buffer[self.Item],
        )

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        return torch.einsum(
            "BKD,BKD->BK",
            self.ranking_buffer[self.User][data[self.User]],
            self.ranking_buffer[self.Item][data[self.IUnseen]],
        )


class CoachForSTAIR(freerec.launcher.Coach):

    def set_optimizer(self):
        opt    = self.cfg.optimizer.lower()
        kwargs = dict(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        betas  = (self.cfg.beta1, self.cfg.beta2)

        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov, **kwargs)
        elif opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.marked_params(), betas=betas, **kwargs)
        elif opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.marked_params(), betas=betas, **kwargs)
        elif opt == 'adamsevo':
            self.optimizer = AdamSEvo(self.model.marked_params(), betas=betas, **kwargs)
        elif opt == 'adamwsevo':
            self.optimizer = AdamWSEvo(self.model.marked_params(), betas=betas, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.cfg.optimizer}")

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.monitor(loss.item(), n=len(data[self.User]),
                         reduction="mean", mode='train', pool=['LOSS'])


def main():
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = STAIR(dataset)
    coach = CoachForSTAIR(
        dataset=dataset,
        trainpipe=model.sure_trainpipe(cfg.batch_size),
        validpipe=model.sure_validpipe(cfg.ranking),
        testpipe=model.sure_testpipe(cfg.ranking),
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
