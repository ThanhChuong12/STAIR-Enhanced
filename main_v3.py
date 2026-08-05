"""
main_v3.py — STAIR-ClipFuse: Clipped Softmax Fusion with Topology-Preserving Binary Graph

Enhancement over STAIR-v2 (DyFuse):
  Root causes identified in v2 experiments:
    L1 — Text features pre-L2-normalized (norm=1), visual not → raw std imbalance (0.04 vs 0.94)
         leads to c_v=95.6%, effectively suppressing the text graph.
    L2 — Continuous cosine edge weights create skewed distribution → unstable BSC diffusion.
    L3 — No connectivity floor → one modality can be almost entirely suppressed.

  STAIR-v3 (ClipFuse) fixes all three:
    Step 0 — L2-normalize both modalities before computing confidence (removes scale bias).
    Step 1 — Dimension-wise std after normalization measures true information richness.
    Step 2 — Softmax(sigma/tau) then clip to [delta, 1-delta] (connectivity floor).
    Step 3 — TopK combined scores, then BINARIZE → binary adjacency (stable BSC topology).
    Step 4 — Symmetric Laplacian normalization (identical to STAIR baseline).

Design reference:
  - STAIR-DyFuse (v2) : per-item norm-based confidence (failed due to scale bias)
  - STAIR-v3 (ClipFuse): dataset-level dim-wise-std confidence + clip + binary TopK fusion
"""

import math
import os
import pickle
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
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3,   help="number of FSC/BSC layers")
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl",
                                                 help="modal feature files (textual first, visual second)")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1', help="kNN neighbors per modality")
cfg.add_argument("--gamma",          type=float, default=0.2)

# ClipFuse-specific arguments
cfg.add_argument("--conf-delta",     type=float, default=0.3,
                                                 help="connectivity floor: minimum confidence per modality in [0.2, 0.4]")
cfg.add_argument("--conf-temp",      type=float, default=1.0,
                                                 help="softmax temperature tau for confidence scoring")

cfg.set_defaults(
    description="STAIR-v3 (ClipFuse — Clipped Softmax Fusion with Topology-Preserving Binary Graph)",
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

    def compute_confidence(
        self,
        feat_t: torch.Tensor,
        feat_v: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute clipped softmax modality confidence from dimension-wise std.

        Key fix over v2:
          v2 used raw L2-norm ratio, which collapsed due to text pre-normalization
          (text norm = 1.0 constant, visual norm = ~76.8) → c_v ≈ 95.6%.

          v3 fixes this by:
            1. L2-normalizing both modalities to unit sphere (removes scale bias).
            2. Computing dimension-wise std across items (captures information richness).
            3. Applying softmax with temperature tau (smooth, differentiable confidence).
            4. Clipping to [delta, 1-delta] (connectivity floor — prevents text collapse).

        Args:
            feat_t: raw textual features (N, D_t) — before whitening
            feat_v: raw visual  features (N, D_v) — before whitening
            delta:  minimum confidence per modality (cfg.conf_delta, default 0.3)
            tau:    softmax temperature             (cfg.conf_temp,  default 1.0)

        Returns:
            c_v, c_t: global dataset-level confidence floats in [delta, 1-delta]
        """
        eps = 1e-7
        delta = cfg.conf_delta
        tau   = cfg.conf_temp

        # Step 0 — L2-normalize to unit sphere (eliminates preprocessing scale bias)
        feat_v_n = F.normalize(feat_v.float(), p=2, dim=-1)   # (N, D_v)
        feat_t_n = F.normalize(feat_t.float(), p=2, dim=-1)   # (N, D_t)

        # Step 1 — Dimension-wise std across items, then average over dims
        sigma_v = feat_v_n.std(dim=0).mean().item()   # scalar
        sigma_t = feat_t_n.std(dim=0).mean().item()   # scalar

        # Step 2 — Softmax confidence with temperature tau
        score_v = sigma_v / (tau + eps)
        score_t = sigma_t / (tau + eps)
        exp_v   = math.exp(score_v)
        exp_t   = math.exp(score_t)
        c_v_raw = exp_v / (exp_v + exp_t + eps)

        # Step 3 — Clip to [delta, 1-delta] (connectivity floor)
        c_v = max(delta, min(1.0 - delta, c_v_raw))
        c_t = 1.0 - c_v

        print(
            f"[ClipFuse] L2-normalized sigma_v={sigma_v:.4f}, sigma_t={sigma_t:.4f} | "
            f"raw c_v={c_v_raw:.4f} → clipped c_v={c_v:.4f}, c_t={c_t:.4f} "
            f"(delta={delta}, tau={tau})"
        )
        return c_v, c_t

    def build_knn_weighted(self, features: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a kNN graph from cosine similarity (on L2-normalized features).

        Returns:
            edge_index: (2, N*k) — directed source→neighbor edges
            edge_weight: (N*k,)  — cosine similarity, clipped to [0, 1]
        """
        feat_n = F.normalize(features.float(), p=2, dim=-1)
        sim    = feat_n @ feat_n.t()          # (N, N) cosine similarity matrix
        sim.fill_diagonal_(-10.)              # exclude self-loops

        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        row, col = edge_index[0], edge_index[1]
        edge_weight = sim[row, col].clamp(min=0.0)
        return edge_index, edge_weight

    def prepare(self, path: str):
        """
        Build ClipFuse multimodal item graph (mAdj) and initialize embeddings.

        Pipeline:
          1. Load raw modal features.
          2. Compute clipped-softmax confidence (global, dataset-level).
          3. Build cosine-weighted kNN graphs per modality.
          4. Compute combined edge scores: score_ij = c_v * s_v + c_t * s_t.
          5. Coalesce (sum duplicate edges from both graphs).
          6. TopK selection: keep top (k_t + k_v) edges per source node.
          7. Binarize: set all kept edge weights to 1.0 (topology-preserving).
          8. To-undirected + symmetric Laplacian normalization → mAdj.
          9. Initialize item/user embeddings from whitened modal features.
        """
        from freerec.utils import import_pickle

        raw_mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
        feat_t_raw = raw_mfeats[0]   # textual (N, D_t)
        feat_v_raw = raw_mfeats[1]   # visual  (N, D_v)

        N = feat_t_raw.shape[0]

        # Step 2 — Compute clipped-softmax confidence on raw features
        c_v, c_t = self.compute_confidence(feat_t_raw, feat_v_raw)

        # Step 3 — Build cosine-weighted kNN graphs
        k_t, k_v = cfg.num_neighbors[0], cfg.num_neighbors[1]
        ei_t, ew_t = self.build_knn_weighted(feat_t_raw, k_t)   # textual graph
        ei_v, ew_v = self.build_knn_weighted(feat_v_raw, k_v)   # visual  graph

        # Step 4 — Scale edge weights by global confidence
        ew_t_scored = c_t * ew_t    # (E_t,) combined score for textual edges
        ew_v_scored = c_v * ew_v    # (E_v,) combined score for visual edges

        # Step 5 — Merge all candidate edges (sum scores for overlapping edges)
        edge_index_all = torch.cat([ei_t, ei_v], dim=1)              # (2, E_t + E_v)
        score_all      = torch.cat([ew_t_scored, ew_v_scored], dim=0) # (E_t + E_v,)
        edge_index_all, score_all = freerec.graph.coalesce(
            edge_index_all, score_all, reduce='sum'
        )

        # Step 6 & 7 — TopK per source node, then BINARIZE to 0/1
        # Memory-efficient: sort by (row, -score) → take top k_total per row
        k_total = k_t + k_v
        row, col = edge_index_all[0], edge_index_all[1]

        # Build a (N, k_total) mask using per-row top-k selection
        # Strategy: for each row, keep only the top-k_total scoring neighbors
        # Use scatter-based approach to avoid dense N×N matrix
        edge_index_binary, _ = self._topk_binarize(row, col, score_all, k_total, N)

        # Binary weights: 1.0 for all kept edges
        edge_weight_binary = torch.ones(edge_index_binary.size(1), device=cfg.device)

        # Step 8 — To-undirected + symmetric Laplacian normalization
        edge_index_binary, edge_weight_binary = freerec.graph.to_undirected(
            edge_index_binary, edge_weight_binary, reduce='max'
        )
        edge_index_binary, edge_weight_binary = freerec.graph.to_normalized(
            edge_index_binary, edge_weight_binary, normalization='sym'
        )

        mAdj = torch.sparse_coo_tensor(
            edge_index_binary, edge_weight_binary, size=(N, N)
        )
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        print(
            f"[ClipFuse] mAdj built: {edge_index_binary.size(1)} edges "
            f"(after undirected + sym-norm) | "
            f"c_v={c_v:.4f}, c_t={c_t:.4f} | k_total={k_total}"
        )

        # Step 9 — Initialize embeddings (identical to STAIR baseline)
        mfeats = [self.whitening(feat) * k for feat, k in zip(raw_mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats)

        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ mfeats)

    @staticmethod
    def _topk_binarize(
        row: torch.Tensor,
        col: torch.Tensor,
        score: torch.Tensor,
        k: int,
        N: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient TopK selection per source node without building dense NxN matrix.

        For each source node i, keep only the top-k neighbors by combined score.
        All kept edges receive binary weight 1.0.

        Args:
            row:   (E,) source node indices
            col:   (E,) destination node indices
            score: (E,) combined edge scores
            k:     number of top neighbors to keep per source node
            N:     total number of nodes

        Returns:
            edge_index: (2, E_kept) filtered edges
            mask:       (E_kept,) boolean mask (all True — kept for readability)
        """
        device = row.device

        # Sort edges by (row ASC, score DESC) for grouped top-k extraction
        # Trick: sort by row * (max_score + 1) - score to get row-major desc-score order
        max_score = score.max().item() + 1.0
        sort_key  = row.float() * max_score - score
        order     = torch.argsort(sort_key, stable=True)

        row_sorted   = row[order]
        col_sorted   = col[order]
        score_sorted = score[order]

        # Count edges per source node and build a cumulative index
        # Then keep only the first k entries per group
        ones   = torch.ones(len(row_sorted), dtype=torch.long, device=device)
        # running count of how many edges we've seen so far for each row
        # Compute rank within each group (how many edges of the same row precede this one)
        # E.g. for row [0,0,0,1,1,2], rank = [0,1,2,0,1,0]
        cumcount = torch.zeros(len(row_sorted), dtype=torch.long, device=device)
        # Vectorized cumcount using scatter_add trick
        # count[i] = number of edges with row_sorted[j] < row_sorted[i] for j<i
        # We use a shifted cumsum approach
        row_change   = torch.cat([torch.tensor([1], device=device),
                                  (row_sorted[1:] != row_sorted[:-1]).long()])
        group_starts = torch.cumsum(row_change, dim=0) - 1   # group index 0,0,0,1,1,...
        # positions within group
        global_idx   = torch.arange(len(row_sorted), device=device)
        # first occurrence position of each group
        first_in_grp = torch.zeros(len(row_sorted), dtype=torch.long, device=device)
        first_in_grp[row_change.bool()] = global_idx[row_change.bool()]
        # broadcast first position to all members in group
        # (simple scan — works because row_sorted is sorted)
        for i in range(1, len(first_in_grp)):
            if row_change[i] == 0:
                first_in_grp[i] = first_in_grp[i - 1]
        rank_in_group = global_idx - first_in_grp   # 0,1,2,0,1,0,...

        keep_mask = rank_in_group < k

        row_kept   = row_sorted[keep_mask]
        col_kept   = col_sorted[keep_mask]
        edge_index_kept = torch.stack([row_kept, col_kept], dim=0)

        return edge_index_kept, keep_mask

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
