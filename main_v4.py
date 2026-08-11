"""
main_v4.py — STAIR-v4 (SGInit): SG-URInit Integration
        STAIR-v3.1 (ClipFuse-Consensus) + Semantically Guaranteed User Representation Initialization

Enhancements over STAIR-v3.1:
  Root Cause (remaining gap on Baby):
    - v3.1 solved item-item graph (ClipFuse-Consensus) but user initialization
      still relies on sparse left-normalized R @ mfeats:
        * Only captures LOCAL semantics (5-7 clicked items per user on Baby)
        * No global context: user embedding doesn't know which semantic cluster
          they belong to
        * Asymmetric initialization: item embeddings anchored by rich FSC+whitening,
          user embeddings anchored by sparse interaction averaging only

  STAIR-v4 Solution — SG-URInit (SIGIR'26, Xu et al., arXiv:2604.14839):
    Training-free, model-agnostic user initialization using two complementary signals:
      1. Local Item-Level (sym-normalized, LightGCN-style):
           U_local = D_u^{-1/2} R D_i^{-1/2} @ F_whitened
      2. Global Cluster-Level (K-Means semantic fallback):
           U_global = D_u^{-1} R @ F_cluster
           F_cluster[i] = centroid of item i's K-Means cluster
      3. Mix: U_init = lambda * U_local + (1-lambda) * U_global  (lambda=0.1)
      4. Rescale U_init to match item embedding norm at epoch 0

  Key engineering fixes over naive SG-URInit:
    - Scale alignment: rescale user init to match mean item embedding norm
    - Same interaction graph: reuse edge_index_u2i from prepare() (no extra API call)
    - Cluster diagnostics: print cluster size distribution to catch degenerate configs
    - Zero-interaction user warning
    - Explicit int64 casting throughout
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

# STAIR-v3.1 ClipFuse hyperparameters (inherited, unchanged)
cfg.add_argument("--conf-delta",       type=float, default=0.3,   help="min confidence floor in [0.2, 0.4]")
cfg.add_argument("--conf-temp",        type=float, default=1.0,   help="softmax temperature tau")
cfg.add_argument("--alpha-consensus",  type=float, default=0.5,   help="consensus boost multiplier for overlapping edges")

# STAIR-v4 SG-URInit hyperparameters
cfg.add_argument("--sg-num-clusters",  type=int,   default=8,     help="K for K-Means in SG-URInit {4,8}")
cfg.add_argument("--sg-lambda",        type=float, default=0.1,   help="local mixing weight: 0=cluster-only, 1=local-only")

cfg.set_defaults(
    description="STAIR-v4 (SGInit — SG-URInit: Semantically Guaranteed User Initialization)",
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
            f"  v3 (ClipFuse): mean_knn_sim_v={mean_sim_v:.4f}, mean_knn_sim_t={mean_sim_t:.4f}\n"
            f"    disc_v={disc_v:.4f}, disc_t={disc_t:.4f} => raw_c_v={c_v_raw:.4f} -> clipped c_v={c_v:.4f}\n"
            f"  => v3: Visual {c_v*100:.1f}%, Text {c_t*100:.1f}%"
        )
        return c_v, c_t

    def sg_urinit(
        self,
        mfeats_whitened: torch.Tensor,
        edge_index_u2i: torch.Tensor,
        n_clusters: int = 8,
        lambda_mix: float = 0.1,
    ) -> torch.Tensor:
        """
        SG-URInit: Semantically Guaranteed User Representation Initialization.

        Computes user embeddings from two complementary signals:
          1. Local (sym-normalized item aggregation): captures specific preferences
          2. Global (cluster-level aggregation): semantic fallback for sparse users

        Args:
            mfeats_whitened : (N_I, D) prior-weighted whitened item features — same tensor
                              used for Item.embeddings initialization, so spaces are aligned.
            edge_index_u2i  : (2, E) LongTensor — user-item interaction edges (row=user,col=item).
                              Reuses the same edge_index already built in prepare() to guarantee
                              exact graph consistency with the rest of the pipeline.
            n_clusters      : K for K-Means (default=8, try 4 for very small datasets).
            lambda_mix      : local mixing weight; 0=cluster-only, 1=local-only (default=0.1).

        Returns:
            user_init : (N_U, D) float32 CPU tensor, rescaled to match item embedding norm.

        Engineering notes:
          - All ops on CPU to avoid VRAM pressure during preprocessing.
          - Scale alignment: U_init rescaled to mean item embedding norm → BPR scores balanced.
          - Cluster diagnostics: prints cluster size distribution to detect degenerate configs.
          - Cold users (0 interactions): fall back to global mean of item embeddings.
        """
        from sklearn.cluster import KMeans
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        N_I = mfeats_whitened.shape[0]
        N_U = self.User.count
        D   = mfeats_whitened.shape[1]

        # Guarantee int64 throughout
        users_idx = edge_index_u2i[0].long()  # (E,)
        items_idx = edge_index_u2i[1].long()  # (E,)

        # ── Compute user and item degrees ───────────────────────────────────
        deg_u = torch.zeros(N_U, dtype=torch.float32)
        deg_i = torch.zeros(N_I, dtype=torch.float32)
        deg_u.scatter_add_(0, users_idx, torch.ones(len(users_idx), dtype=torch.float32))
        deg_i.scatter_add_(0, items_idx, torch.ones(len(items_idx), dtype=torch.float32))

        # Warn about cold (zero-interaction) users
        cold_users = (deg_u == 0).sum().item()
        if cold_users > 0:
            print(f"  [SG-URInit] WARNING: {cold_users}/{N_U} users have 0 training interactions."
                  f" They will receive the global mean item embedding as initialization.")

        deg_u_safe = deg_u.clamp(min=1.0)
        deg_i_safe = deg_i.clamp(min=1.0)

        # ── Bước 1: Local — Sym-normalized item aggregation ─────────────────
        # ew_sym = 1 / (sqrt(deg_u[u]) * sqrt(deg_i[i]))  — LightGCN-style
        ew_sym = 1.0 / (deg_u_safe[users_idx].sqrt() * deg_i_safe[items_idx].sqrt())

        F_white = mfeats_whitened.cpu().float()  # (N_I, D)

        weighted_F = F_white[items_idx] * ew_sym.unsqueeze(-1)   # (E, D)
        U_local = torch.zeros(N_U, D, dtype=torch.float32)
        U_local.scatter_add_(0, users_idx.unsqueeze(-1).expand(-1, D), weighted_F)

        # ── Bước 2: Global — K-Means cluster aggregation ────────────────────
        print(f"  [SG-URInit] Running K-Means (K={n_clusters}, seed={cfg.seed}) on {N_I} items ...")
        feats_np = F_white.numpy()
        km = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=10, max_iter=300)
        km.fit(feats_np)

        clust_idx     = torch.from_numpy(km.labels_).long()         # (N_I,)
        cluster_cents = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # (K, D)

        # Diagnostics: cluster size distribution
        counts = torch.bincount(clust_idx, minlength=n_clusters)
        max_pct = counts.max().item() / N_I * 100
        print(f"  [SG-URInit] Cluster sizes: {counts.tolist()}")
        if max_pct > 70:
            print(f"  [SG-URInit] WARNING: largest cluster contains {max_pct:.1f}% of items. "
                  f"Consider reducing K or checking feature quality.")

        F_cluster = cluster_cents[clust_idx]   # (N_I, D) — centroid per item

        # Left-normalized aggregation for cluster: U_cluster = D_u^{-1} R @ F_cluster
        ew_left = 1.0 / deg_u_safe[users_idx]
        weighted_Fc = F_cluster[items_idx] * ew_left.unsqueeze(-1)  # (E, D)
        U_cluster = torch.zeros(N_U, D, dtype=torch.float32)
        U_cluster.scatter_add_(0, users_idx.unsqueeze(-1).expand(-1, D), weighted_Fc)

        # Cold users: fallback to global item mean
        if cold_users > 0:
            global_mean = F_white.mean(0, keepdim=True)  # (1, D)
            cold_mask = (deg_u == 0)
            U_local[cold_mask]   = global_mean.expand(cold_mask.sum(), D)
            U_cluster[cold_mask] = global_mean.expand(cold_mask.sum(), D)

        # ── Bước 3: Mix ─────────────────────────────────────────────────────
        U_init = lambda_mix * U_local + (1.0 - lambda_mix) * U_cluster  # (N_U, D)

        # ── Bước 4: Scale alignment ─────────────────────────────────────────
        # Rescale U_init so its per-user mean norm matches item embedding mean norm.
        # This ensures BPR scores u·i are not imbalanced at epoch 0.
        item_norm_mean = F_white.norm(dim=-1).mean()
        init_norm_mean = U_init.norm(dim=-1).mean().clamp(min=1e-8)
        scale_factor   = item_norm_mean / init_norm_mean
        U_init = U_init * scale_factor

        print(
            f"  [SG-URInit] Norms — item: {item_norm_mean:.4f} | "
            f"local: {U_local.norm(dim=-1).mean():.4f} | "
            f"cluster: {U_cluster.norm(dim=-1).mean():.4f} | "
            f"init (before scale): {(U_init / scale_factor).norm(dim=-1).mean():.4f} | "
            f"init (after scale):  {U_init.norm(dim=-1).mean():.4f} [x{scale_factor:.3f}]"
        )
        return U_init

    def prepare(self, path: str):
        """
        Build STAIR-v4 (SGInit) pipeline.

        Phases:
          A. Item-item graph (identical to STAIR-v3.1 ClipFuse-Consensus):
             kNN → discriminability confidence → prior-preserving weights
             → consensus boosting → mAdj
          B. Item embedding: prior-weighted whitened features (identical to v3.1)
          C. User embedding: SG-URInit (v4 new):
             sym-normalized local + K-Means cluster global → mixed + scale-aligned
        """
        from freerec.utils import import_pickle

        raw_mfeats = [import_pickle(os.path.join(path, f)) for f in cfg.mfiles]
        feat_t_raw = raw_mfeats[0]   # textual (N, D_t)
        feat_v_raw = raw_mfeats[1]   # visual  (N, D_v)
        N = feat_t_raw.shape[0]

        k_t, k_v = cfg.num_neighbors[0], cfg.num_neighbors[1]

        # ── Phase A: Item-item graph (v3.1 ClipFuse-Consensus, unchanged) ──
        print(f"Building kNN graphs (k_t={k_t}, k_v={k_v})...")
        raw_norms = (
            feat_t_raw.float().norm(dim=-1).mean().item(),
            feat_v_raw.float().norm(dim=-1).mean().item(),
        )
        print(f"  Raw norms  : text={raw_norms[0]:.4f}, visual={raw_norms[1]:.4f}")

        ei_t, ew_t = self.build_knn_weighted(feat_t_raw, k_t)
        ei_v, ew_v = self.build_knn_weighted(feat_v_raw, k_v)

        c_v, c_t = self.compute_confidence(ew_t, ew_v)

        total_t_weight = k_t * c_t
        total_v_weight = k_v * c_v
        pct_t = total_t_weight / (total_t_weight + total_v_weight) * 100
        pct_v = 100 - pct_t

        print(
            f"  => v4: Prior-Preserving Pool: Text {pct_t:.1f}%, Visual {pct_v:.1f}%"
            f" [Consensus α={cfg.alpha_consensus}]"
        )

        score_t = torch.full_like(ew_t, fill_value=c_t)
        score_v = torch.full_like(ew_v, fill_value=c_v)

        edge_index_all = torch.cat([ei_t, ei_v], dim=1)
        score_all      = torch.cat([score_t, score_v], dim=0)

        edge_index_coalesced, score_coalesced = freerec.graph.coalesce(
            edge_index_all, score_all, reduce='sum'
        )

        eps_threshold  = max(c_t, c_v) + 1e-5
        consensus_mask = score_coalesced >= eps_threshold
        num_consensus  = consensus_mask.sum().item()
        score_coalesced[consensus_mask] = score_coalesced[consensus_mask] * (1.0 + cfg.alpha_consensus)

        print(
            f"  [mAdj] {edge_index_coalesced.size(1)} directed edges | "
            f"Consensus edges: {num_consensus} ({num_consensus/N:.2f}/item)"
        )

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

        # ── Phase B: Item embedding (v3.1, unchanged) ───────────────────────
        mfeats = [self.whitening(feat) * k for feat, k in zip(raw_mfeats, cfg.num_neighbors)]
        mfeats = sum(mfeats).div(sum(cfg.num_neighbors))    # (N_I, D) — fused, whitened
        self.Item.embeddings.weight.data.copy_(mfeats)

        # ── Phase C: User embedding — SG-URInit (v4) ────────────────────────
        # Build user-item interaction edge_index once (left-normalized form)
        edge_index_u2i, edge_weight_u2i = freerec.graph.to_normalized(
            self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index,
            normalization='left'
        )
        # We only need raw edge_index_u2i (unweighted topology) for sg_urinit;
        # the function computes its own sym/left weights from scratch for correctness.
        print(f"\n[SG-URInit] Computing semantically guaranteed user embeddings ...")
        print(f"  K={cfg.sg_num_clusters}, λ={cfg.sg_lambda} (local weight), seed={cfg.seed}")
        user_init = self.sg_urinit(
            mfeats_whitened=mfeats,
            edge_index_u2i=edge_index_u2i,
            n_clusters=cfg.sg_num_clusters,
            lambda_mix=cfg.sg_lambda,
        )
        self.User.embeddings.weight.data.copy_(user_init.to(cfg.device))
        print(f"[SG-URInit] User embedding initialized successfully.\n")

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
