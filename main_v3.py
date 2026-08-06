"""
main_v3.py — STAIR-v3.1 (ClipFuse-Consensus): Prior-Preserving Adaptive Fusion with Cross-Modal Consensus Boosting

Enhancements over STAIR-v2 & STAIR-v3.0 (Senior Researcher Design):
  Root Cause Analysis of v3.0:
    - In Amazon Baby/Sports, Text (title/description) is significantly cleaner and more
      discriminative than Visual (product thumbnails). STAIR baseline used k_t=5, k_v=1,
      establishing a 5:1 (83.3% Text : 16.7% Visual) structural prior.
    - STAIR-v3.0 san-equalized modality weights to ~51% Visual : 49% Text, diluting the
      high-quality Text graph with 50% Visual noise, causing a drop vs baseline (0.0933 vs 0.1034).
    - STAIR-v3.0 binarized all edges to 1.0, destroying the 2.0x weight boost that STAIR
      baseline gave to Cross-Modal Consensus edges (edges present in BOTH Text & Visual kNN).

  STAIR-v3.1 Solutions:
    1. Prior-Preserving Structural Weighting:
       Edge weights incorporate the k_t : k_v structural prior (5:1).
       Text edge weight pool = k_t * c_t (~83%), Visual edge weight pool = k_v * c_v (~17%).
       c_t, c_v are derived adaptively via kNN-discriminability (1 - mean_knn_sim).
    2. Cross-Modal Consensus Boosting (alpha):
       Edges present in BOTH Text and Visual kNN receive a consensus multiplier (1 + alpha).
       This reinforces high-confidence multi-modal alignment edges for the BSC smoother.
"""

import math
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother

freerec.declare(version='0.9.7')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim",    type=int,   default=64)
cfg.add_argument("--num-layers",       type=int,   default=3,     help="number of FSC/BSC layers")
cfg.add_argument("--mfiles",           type=str,   default="textual_modality.pkl,visual_modality.pkl",
                                                   help="modal feature files (textual first, visual second)")
cfg.add_argument("--num-neighbors",    type=str,   default='5-1', help="kNN neighbors per modality")
cfg.add_argument("--gamma",            type=float, default=0.2)

# STAIR-v3.1 Hyperparameters
cfg.add_argument("--conf-delta",       type=float, default=0.3,   help="min confidence floor in [0.2, 0.4]")
cfg.add_argument("--conf-temp",        type=float, default=1.0,   help="softmax temperature tau")
cfg.add_argument("--alpha-consensus",  type=float, default=0.5,   help="consensus boost multiplier for overlapping edges")

cfg.set_defaults(
    description="STAIR-v3.1 (ClipFuse-Consensus — Prior-Preserving Fusion + Cross-Modal Consensus Boosting)",
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

# Dimension-wise step weight for BSC: beta_j = 1 - (0.1 + 0.9*(j/D)^gamma)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


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

    def build_knn_weighted(
        self, features: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build kNN graph from cosine similarity on L2-normalized features.
        Returns CPU tensors for device safety.
        """
        feat_n = F.normalize(features.float(), p=2, dim=-1)
        sim    = feat_n @ feat_n.t()
        sim.fill_diagonal_(-10.)

        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        row, col = edge_index[0], edge_index[1]
        edge_weight = sim[row, col].clamp(min=0.0)
        return edge_index, edge_weight

    def compute_confidence(
        self,
        ew_t: torch.Tensor,
        ew_v: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute clipped-softmax modality confidence from kNN discriminability.
        disc_m = 1 - mean_knn_sim_m
        """
        eps   = 1e-7
        delta = cfg.conf_delta
        tau   = cfg.conf_temp

        mean_sim_t = ew_t.mean().item()
        mean_sim_v = ew_v.mean().item()

        disc_t = max(0.0, 1.0 - mean_sim_t)
        disc_v = max(0.0, 1.0 - mean_sim_v)

        exp_v   = math.exp(disc_v / (tau + eps))
        exp_t   = math.exp(disc_t / (tau + eps))
        c_v_raw = exp_v / (exp_v + exp_t + eps)

        c_v = max(delta, min(1.0 - delta, c_v_raw))
        c_t = 1.0 - c_v

        print(
            f"[ClipFuse-v3.1] mean_knn_sim: visual={mean_sim_v:.4f}, text={mean_sim_t:.4f} | "
            f"disc_v={disc_v:.4f}, disc_t={disc_t:.4f} | "
            f"raw_c_v={c_v_raw:.4f} → clipped c_v={c_v:.4f}, c_t={c_t:.4f}"
        )
        return c_v, c_t

    def prepare(self, path: str):
        """
        Build STAIR-v3.1 (ClipFuse-Consensus) multimodal item graph (mAdj).

        Pipeline (v3.1):
          1. Load raw modal features (CPU).
          2. Build kNN graphs: Text (k_t=5), Visual (k_v=1).
          3. Compute kNN discriminability confidence: c_t, c_v.
          4. Prior-Preserving Scaling:
             Text edge weight = c_t, Visual edge weight = c_v.
             Total edge weight pool maintains ~5:1 ratio (k_t * c_t : k_v * c_v).
          5. Coalesce with Cross-Modal Consensus Boosting:
             - Single-modality edge: weight = c_t (or c_v)
             - Consensus edge (present in BOTH kNN graphs):
               weight = (c_t + c_v) * (1 + alpha_consensus)
          6. Symmetrize + Symmetric Laplacian normalization → mAdj.
          7. Initialize item/user embeddings (prior-weighted 5:1 whitening).
        """
        from freerec.utils import import_pickle

        raw_mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
        feat_t_raw = raw_mfeats[0]   # textual (N, D_t)
        feat_v_raw = raw_mfeats[1]   # visual  (N, D_v)
        N = feat_t_raw.shape[0]

        k_t, k_v = cfg.num_neighbors[0], cfg.num_neighbors[1]
        ei_t, ew_t = self.build_knn_weighted(feat_t_raw, k_t)
        ei_v, ew_v = self.build_knn_weighted(feat_v_raw, k_v)

        c_v, c_t = self.compute_confidence(ew_t, ew_v)

        # Total structural weight pool:
        total_t_weight = k_t * c_t
        total_v_weight = k_v * c_v
        pct_t = total_t_weight / (total_t_weight + total_v_weight) * 100
        pct_v = total_v_weight / (total_t_weight + total_v_weight) * 100

        print(
            f"[ClipFuse-v3.1] Prior-Preserving Structural Weights (k_t={k_t}, k_v={k_v}):\n"
            f"  Textual total weight: {total_t_weight:.4f} ({pct_t:.1f}%)\n"
            f"  Visual  total weight: {total_v_weight:.4f} ({pct_v:.1f}%)\n"
            f"  Consensus Boost Alpha: {cfg.alpha_consensus}"
        )

        # Assign base confidence weights
        score_t = torch.full_like(ew_t, fill_value=c_t)
        score_v = torch.full_like(ew_v, fill_value=c_v)

        # Merge candidate edges
        edge_index_all = torch.cat([ei_t, ei_v], dim=1)
        score_all      = torch.cat([score_t, score_v], dim=0)

        # Coalesce with sum: overlapping edges will get (c_t + c_v)
        edge_index_coalesced, score_coalesced = freerec.graph.coalesce(
            edge_index_all, score_all, reduce='sum'
        )

        # Apply Cross-Modal Consensus Boost:
        # If score_coalesced > max(c_t, c_v), it means the edge appeared in BOTH graphs!
        eps_threshold = max(c_t, c_v) + 1e-5
        consensus_mask = score_coalesced >= eps_threshold
        num_consensus  = consensus_mask.sum().item()

        # Boost consensus edges by (1 + alpha)
        score_coalesced[consensus_mask] = score_coalesced[consensus_mask] * (1.0 + cfg.alpha_consensus)

        print(
            f"[ClipFuse-v3.1] Graph Coalesced: {edge_index_coalesced.size(1)} unique directed edges | "
            f"Consensus edges (in both kNN): {num_consensus} ({num_consensus/N:.2f} per item) "
            f"[Boosted by x{1.0 + cfg.alpha_consensus:.2f}]"
        )

        # To-undirected + symmetric Laplacian normalization
        edge_index_final, edge_weight_final = freerec.graph.to_undirected(
            edge_index_coalesced, score_coalesced, reduce='max'
        )
        edge_index_final, edge_weight_final = freerec.graph.to_normalized(
            edge_index_final, edge_weight_final, normalization='sym'
        )

        mAdj = torch.sparse_coo_tensor(
            edge_index_final, edge_weight_final, size=(N, N)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        # Initialize item/user embeddings (prior-weighted whitening)
        mfeats = [self.whitening(feat) * k for feat, k in zip(raw_mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index_u2i = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index_u2i, edge_weight_u2i = freerec.graph.to_normalized(
            edge_index_u2i, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index_u2i, edge_weight_u2i,
            size=(self.User.count, self.Item.count)
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
