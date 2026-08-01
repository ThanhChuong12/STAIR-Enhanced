"""
main_v1.py — STAIR-GCL: Stepwise Graph Contrastive Learning

Enhancement over STAIR baseline:
  - Two augmented graph views via edge dropout on the user-item interaction graph (Adj).
  - InfoNCE contrastive loss applied only to the collaborative subspace (first D/2 dims),
    leaving the multimodal subspace intact to avoid over-smoothing modal features.
  - Total loss: L = L_BPR + lambda * (L_CL_user + L_CL_item)
"""

from typing import Dict, Tuple, Optional

import torch, os, math
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='0.9.7')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers",    type=int, default=3, help="number of FSC/BSC layers")
cfg.add_argument("--mfiles",        type=str, default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors", type=str, default='5-1', help="kNN neighbors per modality")
cfg.add_argument("--gamma",         type=float, default=0.2)
cfg.add_argument("--cl-weight",     type=float, default=1e-3,  help="contrastive loss weight")
cfg.add_argument("--edge-drop",     type=float, default=0.2,   help="edge dropout rate for graph augmentation")
cfg.add_argument("--cl-temp",       type=float, default=0.2,   help="InfoNCE temperature")

cfg.set_defaults(
    description="STAIR-v1 (Stepwise Graph Contrastive Learning)",
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

    def get_knn_graph(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """Build a kNN graph from cosine similarity; returns edge_index (2, E)."""
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def prepare(self, path: str):
        """
        Build the multimodal item-item graph (mAdj) and initialize embeddings.
        mAdj fuses textual and visual kNN graphs with sym-normalized Laplacian.
        """
        from freerec.utils import import_pickle

        mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]

        # Fuse per-modality kNN graphs; coalesce duplicate edges by summing weights
        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, cfg.num_neighbors)],
            dim=1
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce='sum')
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce='max')
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization='sym')

        mAdj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(self.Item.count, self.Item.count))
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Initialize item embeddings from whitened modal features (weighted by kNN count)
        mfeats = [self.whitening(feat) * k for feat, k in zip(mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        # Initialize user embeddings as left-normalized interaction-weighted item features
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

    def graph_dropout(self, adj: torch.Tensor, drop_rate: float) -> torch.Tensor:
        """
        Stochastic edge dropout for graph augmentation (training only).
        Surviving edges are rescaled by 1/(1-p) to preserve expected activation.
        """
        if drop_rate <= 0.0 or not self.training:
            return adj

        adj_coo = adj.to_sparse_coo().coalesce()
        indices, values = adj_coo.indices(), adj_coo.values()
        mask = torch.rand(values.size(0), device=adj.device) >= drop_rate
        return torch.sparse_coo_tensor(
            indices[:, mask], values[mask] / (1.0 - drop_rate), adj.shape
        ).coalesce().to_sparse_csr()

    def calc_contrastive_loss(self, view1: torch.Tensor, view2: torch.Tensor, temp: float) -> torch.Tensor:
        """
        InfoNCE loss restricted to the collaborative subspace (first D/2 dimensions).
        The multimodal subspace (last D/2 dims) is excluded to preserve modal signal.
        """
        d = cfg.embedding_dim // 2
        v1 = F.normalize(view1[:, :d], dim=-1)
        v2 = F.normalize(view2[:, :d], dim=-1)

        pos_score = torch.exp((v1 * v2).sum(dim=-1) / temp)
        all_score  = torch.exp((v1 @ v2.t()) / temp).sum(dim=-1)
        return -torch.log(pos_score / (all_score + 1e-8)).mean()

    def encode(self, adj: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward stepwise convolution (FSC) over the user-item graph."""
        if adj is None:
            adj = self.Adj

        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )  # (N_users + N_items, D)

        features = allEmbds
        smoothed = allEmbds
        beta = 1 - cfg.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = adj @ features * beta
            smoothed = smoothed + features

        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds[users],     itemEmbds[positives]),
            torch.einsum("BKD,BKD->BK", userEmbds[users],     itemEmbds[negatives]),
        )

        if self.training and cfg.cl_weight > 0:
            # Build two independently augmented graph views
            u1, i1 = self.encode(self.graph_dropout(self.Adj, cfg.edge_drop))
            u2, i2 = self.encode(self.graph_dropout(self.Adj, cfg.edge_drop))

            u_idx = users.view(-1)
            i_idx = positives.view(-1)
            cl_loss = cfg.cl_weight * (
                self.calc_contrastive_loss(u1[u_idx], u2[u_idx], cfg.cl_temp) +
                self.calc_contrastive_loss(i1[i_idx], i2[i_idx], cfg.cl_temp)
            )
            return rec_loss + cl_loss

        return rec_loss

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
        opt = self.cfg.optimizer.lower()
        kwargs = dict(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        # freerec 0.9.x does not expose beta1/beta2 by default; use Adam defaults
        betas  = (getattr(self.cfg, 'beta1', 0.9), getattr(self.cfg, 'beta2', 0.999))

        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(),
                momentum=getattr(self.cfg, 'momentum', 0.9),
                nesterov=getattr(self.cfg, 'nesterov', False),
                **kwargs)
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
