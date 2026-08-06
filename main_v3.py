"""
main_v3.py — STAIR-ClipFuse: Clipped Softmax Fusion with Topology-Preserving Binary Graph

Enhancement over STAIR-v2 (DyFuse) — fixes 3 root causes identified in experiments:
  L1 — Scale Bias: L2-normalize both modalities before computing confidence (removes
       the text-norm=1 vs visual-norm=76.8 imbalance that collapsed c_v to 95.6%).
  L2 — Skewed Edge Weights: TopK selection + BINARIZE edges → restores stable binary
       adjacency, preserving BSC diffusion properties.
  L3 — Modality Collapse: Clip confidence to [delta, 1-delta] → each modality always
       contributes at least delta to the fused graph.

Confidence signal (v3):
  Uses kNN-discriminability: for each modality, compute mean of top-k cosine similarities
  among items. Lower mean similarity → items are more spread out → modality is more
  discriminative → higher confidence.
    disc_m = 1 - mean_knn_sim_m
    c_v_raw = softmax([disc_v/tau, disc_t/tau])[0]
    c_v = clip(c_v_raw, delta, 1-delta)
  This is computed AFTER building kNN graphs (meaningful signal, device-safe).

Design reference:
  - TAMER  : global weighted multi-graph fusion (static alpha per modality)
  - NLGCL+ : L2-norm as per-sample confidence signal for adaptive weighting
  - STAIR-v2: per-item std-based confidence (collapsed due to scale bias)
  - STAIR-v3: kNN-discriminability confidence + binary TopK fusion
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
cfg.add_argument("--embedding-dim",  type=int,   default=64)
cfg.add_argument("--num-layers",     type=int,   default=3,   help="number of FSC/BSC layers")
cfg.add_argument("--mfiles",         type=str,   default="textual_modality.pkl,visual_modality.pkl",
                                                 help="modal feature files (textual first, visual second)")
cfg.add_argument("--num-neighbors",  type=str,   default='5-1',
                                                 help="kNN neighbors per modality (e.g. 5-1)")
cfg.add_argument("--gamma",          type=float, default=0.2)

# ClipFuse-specific hyperparameters
cfg.add_argument("--conf-delta",     type=float, default=0.3,
                                                 help="connectivity floor: min confidence per modality, in [0.2, 0.4]")
cfg.add_argument("--conf-temp",      type=float, default=1.0,
                                                 help="softmax temperature tau for confidence scoring")

cfg.set_defaults(
    description="STAIR-v3 (ClipFuse — kNN-Discriminability Confidence + Topology-Preserving Binary Graph)",
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

cfg.mfiles       = cfg.mfiles.split(',')
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

        Args:
            features: raw item features (N, D) — any device
            k:        number of neighbors per item

        Returns:
            edge_index: (2, N*k) directed source→neighbor edges  [CPU]
            edge_weight: (N*k,)  cosine similarity clipped to [0, 1] [CPU]
        """
        feat_n = F.normalize(features.float(), p=2, dim=-1)   # unit sphere
        sim    = feat_n @ feat_n.t()                           # (N, N) cosine sim [CPU]
        sim.fill_diagonal_(-10.)

        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        row, col = edge_index[0], edge_index[1]
        edge_weight = sim[row, col].clamp(min=0.0)
        return edge_index, edge_weight   # both CPU tensors

    def compute_confidence(
        self,
        ew_t: torch.Tensor,
        ew_v: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute clipped-softmax modality confidence from kNN discriminability.

        Why kNN discriminability (instead of dim-wise std after L2-normalize):
          After L2-normalizing, visual features (4096-dim, original norm ~76.8) have
          very small per-dim std (~0.012) compared to text (384-dim, pre-normalized,
          std ~0.044). This is a dimension-count artifact, not a measure of quality.

          Using mean kNN cosine similarity avoids this bias:
            low mean kNN sim → items are spread out in this modality's space
                             → modality is more discriminative → higher confidence

        Steps:
          1. disc_m = 1 - mean(top-k cosine similarities) for each modality m
          2. Softmax([disc_v, disc_t] / tau) → raw confidence ratio
          3. Clip to [delta, 1-delta]         → connectivity floor

        Args:
            ew_t: textual edge weights (N*k_t,) — cosine similarities [CPU]
            ew_v: visual  edge weights (N*k_v,) — cosine similarities [CPU]

        Returns:
            c_v, c_t: global confidence floats in [delta, 1-delta]
        """
        eps   = 1e-7
        delta = cfg.conf_delta
        tau   = cfg.conf_temp

        mean_sim_t = ew_t.mean().item()
        mean_sim_v = ew_v.mean().item()

        disc_t = max(0.0, 1.0 - mean_sim_t)   # lower sim → more discriminative
        disc_v = max(0.0, 1.0 - mean_sim_v)

        exp_v   = math.exp(disc_v / (tau + eps))
        exp_t   = math.exp(disc_t / (tau + eps))
        c_v_raw = exp_v / (exp_v + exp_t + eps)

        c_v = max(delta, min(1.0 - delta, c_v_raw))
        c_t = 1.0 - c_v

        print(
            f"[ClipFuse] mean_knn_sim: visual={mean_sim_v:.4f}, text={mean_sim_t:.4f} | "
            f"disc_v={disc_v:.4f}, disc_t={disc_t:.4f} | "
            f"raw_c_v={c_v_raw:.4f} → clipped c_v={c_v:.4f}, c_t={c_t:.4f} "
            f"(delta={delta}, tau={tau})"
        )
        return c_v, c_t

    @staticmethod
    def _topk_binarize(
        row: torch.Tensor,
        col: torch.Tensor,
        score: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Fully-vectorized TopK selection per source node — no Python loops, no dense NxN.

        For each source node, keeps the top-k neighbors by combined score.
        Returns binary edge_index (caller sets weights to 1.0).

        Algorithm:
          1. Sort edges by (row ASC, score DESC)
          2. Compute rank within each row group via vectorized cumsum trick
          3. Keep edges with rank < k

        Args:
            row:   (E,) source node indices  [CPU]
            col:   (E,) destination indices  [CPU]
            score: (E,) combined scores      [CPU]
            k:     max neighbors to keep per source node

        Returns:
            edge_index_kept: (2, E_kept) filtered binary edge_index [CPU]
        """
        device = row.device

        # Step 1: Sort by (row ASC, score DESC)
        max_score = score.max().item() + 1.0
        sort_key  = row.float() * max_score - score       # row-major, score minor desc
        order     = torch.argsort(sort_key, stable=True)

        row_s = row[order]
        col_s = col[order]

        # Step 2: Vectorized rank within each row group
        n = row_s.size(0)
        global_pos = torch.arange(n, device=device)

        # Positions where the row changes (start of new group)
        row_change = torch.cat([
            torch.tensor([True], device=device),
            row_s[1:] != row_s[:-1],
        ])  # bool, shape (n,)

        # group_idx[i] = which group edge i belongs to (0-indexed)
        group_idx = torch.cumsum(row_change.long(), dim=0) - 1   # (n,)

        # Start position (in sorted array) of each group
        # change_positions[g] = index of first edge in group g
        change_positions = torch.where(row_change)[0]   # (n_groups,)

        # Rank = global position − start position of my group
        start_of_my_group = change_positions[group_idx]   # (n,)
        rank = global_pos - start_of_my_group             # (n,)

        # Step 3: Keep only top-k per row
        keep = rank < k
        edge_index_kept = torch.stack([row_s[keep], col_s[keep]], dim=0)
        return edge_index_kept   # CPU tensor

    def prepare(self, path: str):
        """
        Build ClipFuse multimodal item graph (mAdj) and initialize embeddings.

        Pipeline:
          1. Load raw modal features (CPU).
          2. Build cosine-weighted kNN graphs per modality (CPU).
          3. Compute kNN-discriminability confidence (no scale bias).
          4. Scale edge scores by confidence: score_ij = c_v*s_v + c_t*s_t.
          5. Coalesce (sum overlapping edges from both graphs).
          6. TopK binarize: keep top (k_t+k_v) neighbors per item → binary 0/1.
          7. To-undirected + symmetric Laplacian normalization → mAdj.
          8. Initialize embeddings (identical to STAIR baseline).

        All graph operations are on CPU tensors to avoid device-mismatch errors.
        register_buffer() moves mAdj to cfg.device automatically.
        """
        from freerec.utils import import_pickle

        # Step 1 — Load raw features (CPU tensors)
        raw_mfeats  = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
        feat_t_raw  = raw_mfeats[0]   # textual (N, D_t)
        feat_v_raw  = raw_mfeats[1]   # visual  (N, D_v)
        N = feat_t_raw.shape[0]

        # Step 2 — Build cosine-weighted kNN graphs (CPU)
        k_t, k_v = cfg.num_neighbors[0], cfg.num_neighbors[1]
        ei_t, ew_t = self.build_knn_weighted(feat_t_raw, k_t)
        ei_v, ew_v = self.build_knn_weighted(feat_v_raw, k_v)

        # Step 3 — kNN-discriminability confidence (computed from kNN edge weights)
        c_v, c_t = self.compute_confidence(ew_t, ew_v)

        # Step 4 — Scale edge scores by global confidence
        score_t = c_t * ew_t   # (E_t,) confidence-weighted text scores
        score_v = c_v * ew_v   # (E_v,) confidence-weighted visual scores

        # Step 5 — Coalesce: merge and sum overlapping edges from both graphs
        edge_index_all = torch.cat([ei_t, ei_v], dim=1)              # (2, E_t+E_v)
        score_all      = torch.cat([score_t, score_v], dim=0)        # (E_t+E_v,)
        edge_index_all, score_all = freerec.graph.coalesce(
            edge_index_all, score_all, reduce='sum'
        )

        # Step 6 — TopK binary fusion: select top (k_t+k_v) neighbors, then binarize
        k_total = k_t + k_v
        row, col = edge_index_all[0], edge_index_all[1]
        edge_index_bin = self._topk_binarize(row, col, score_all, k_total)

        # Binary weights: 1.0 — keep on CPU (same device as edge_index_bin)
        edge_weight_bin = torch.ones(edge_index_bin.size(1))   # CPU float tensor

        # Step 7 — To-undirected + symmetric Laplacian normalization (all CPU)
        edge_index_bin, edge_weight_bin = freerec.graph.to_undirected(
            edge_index_bin, edge_weight_bin, reduce='max'
        )
        edge_index_bin, edge_weight_bin = freerec.graph.to_normalized(
            edge_index_bin, edge_weight_bin, normalization='sym'
        )

        mAdj = torch.sparse_coo_tensor(
            edge_index_bin, edge_weight_bin, size=(N, N)
        )
        # register_buffer moves tensor to cfg.device and keeps it in sync
        self.register_buffer('mAdj', mAdj.to_sparse_csr())

        print(
            f"[ClipFuse] mAdj built: {edge_index_bin.size(1)} edges "
            f"(after undirected + sym-norm) | "
            f"k_total={k_total}, c_v={c_v:.4f}, c_t={c_t:.4f}"
        )

        # Step 8 — Initialize embeddings (identical to STAIR baseline)
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
