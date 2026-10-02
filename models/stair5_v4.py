"""models/stair5_v4.py — STAIR5-v4 / NLGCL-CSE Architecture.
============================================================
Neighborhood-enriched Graph Contrastive Learning with Candidate Support Expansion.

Combines:
1. Exact STAIR-NLGCL backbone and heterogeneous in-batch InfoNCE loss.
2. Candidate Support Expansion (CSE) for item-item BSC smoothing operator:
   S4 = (1 - eta) * S0 + eta * S_bar_CF
3. Clean factorial arms:
   - 'B0': STAIR Baseline control (NLGCL off, S0 semantic graph only).
   - 'N0': NLGCL reference reproduction (NLGCL lambda=0.01, S0 semantic graph only).
   - 'C0': Candidate Support Expansion graph only (NLGCL off, S4 blended graph).
   - 'N-CSE': Primary STAIR5-v4 proposed model (NLGCL lambda=0.01 + S4 blended graph).
   - 'N-CSE-placebo': Placebo control (NLGCL lambda=0.01 + degree-stratified randomized CF graph).
   - 'v4b': Conditional ablation with Positive-Aware NLGCL loss + S4 blended graph.
"""
from typing import Dict, List, Optional, Tuple, Union
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import models.freerec_compat
except Exception:
    try:
        import freerec_compat
    except Exception:
        pass

import freerec

from models.stair5_v4_objectives import NLGCL_Module, NLGCL_PositiveAware_Module
from models.stair5_v4_graph import build_calibrated_graph_v4, GraphStateV4
from optimizers.stair5_v4_smoother import STAIR5V4Smoother

ARMS_V4 = ("B0", "N0", "C0", "N-CSE", "N-CSE-placebo", "v4b")
NLGCL_ARMS_V4 = ("N0", "N-CSE", "N-CSE-placebo", "v4b")
CSE_ARMS_V4 = ("C0", "N-CSE", "N-CSE-placebo", "v4b")


class STAIR5_v4_Model(freerec.models.GenRecArch):
    """STAIR5-v4 / NLGCL-CSE Model Architecture."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet, cfg) -> None:
        super().__init__(dataset)
        self.cfg = copy.copy(cfg)
        self.num_layers = cfg.num_layers

        arm = getattr(cfg, "v4_arm", "N-CSE")
        if arm not in ARMS_V4:
            raise ValueError(f"Unknown v4 arm: {arm}. Must be one of {ARMS_V4}")
        self.v4_arm = arm

        # Embedding tables
        self.User.add_module(
            "embeddings", nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        # Bipartite User-Item graph
        try:
            adj = self.dataset.train().to_normalized_adj(normalization="sym")
        except Exception:
            from freerec.graph import to_adjacency, to_normalized
            User = self.User
            Item = self.Item
            data = self.dataset.train()[(User, Item)]
            u = torch.as_tensor(data[User], dtype=torch.long)
            i = torch.as_tensor(data[Item], dtype=torch.long) + User.count
            row = torch.cat([u, i], dim=0)
            col = torch.cat([i, u], dim=0)
            edge_index = torch.stack([row, col], dim=0)
            edge_index, edge_weight = to_normalized(edge_index, normalization="sym")
            adj = to_adjacency(edge_index, edge_weight, num_nodes=User.count + Item.count)

        self.register_buffer("Adj", adj)
        self.register_buffer("beta3", cfg.beta3)

        # Effective lambda_nlgcl
        if arm in ("B0", "C0"):
            self.lambda_nlgcl = 0.0
        else:
            self.lambda_nlgcl = float(getattr(cfg, "lambda_nlgcl", 0.01))

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # Initialize NLGCL module
        if arm == "v4b":
            self.nlgcl = NLGCL_PositiveAware_Module(
                n_users=self.User.count,
                n_items=self.Item.count,
                train_pair_keys=self.train_pair_keys,
                G=getattr(cfg, "nlgcl_G", 1),
                tau=getattr(cfg, "nlgcl_tau", 0.2),
                alpha=getattr(cfg, "nlgcl_alpha", 0.5),
            )
        else:
            self.nlgcl = NLGCL_Module(
                n_users=self.User.count,
                n_items=self.Item.count,
                G=getattr(cfg, "nlgcl_G", 1),
                tau=getattr(cfg, "nlgcl_tau", 0.2),
                alpha=getattr(cfg, "nlgcl_alpha", 0.5),
                chunk_size=getattr(cfg, "cl_chunk_size", 1024),
            )

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def marked_params(self) -> List[Dict]:
        """Provides parameter groups with STAIR5V4Smoother on item embeddings."""
        return [
            {"params": self.User.parameters(), "smoother": None},
            {
                "params": self.Item.parameters(),
                "smoother": STAIR5V4Smoother(
                    lambda x: torch.sparse.mm(self.mAdj, x),
                    beta=self.beta3,
                    L=self.num_layers,
                ),
            },
        ]

    def whitening(self, feats: torch.Tensor) -> torch.Tensor:
        """SVD Whitening — identical to STAIR baseline."""
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, dtype=torch.float32)
        else:
            feats = feats.float()
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, : self.cfg.embedding_dim] * math.sqrt(
            self.Item.count / self.cfg.embedding_dim
        )

    def get_knn_graph(self, features: torch.Tensor, k: int = 5) -> torch.Tensor:
        """kNN graph computation — identical to STAIR baseline."""
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = features.float()
        features = F.normalize(features, dim=-1)
        sim = features @ features.t()
        sim.fill_diagonal_(-10.0)
        edge_index, _ = freerec.graph.get_knn_graph(sim, k, symmetric=False)
        return edge_index

    def _build_train_interaction_table(self, edge_index_ui: torch.Tensor) -> None:
        """Builds sorted train interaction keys for O(log N) lookup."""
        edges = edge_index_ui.detach().cpu().long()
        flat_keys = torch.unique(edges[0] * self.Item.count + edges[1], sorted=True)
        self.register_buffer("train_pair_keys", flat_keys, persistent=False)

    @torch.no_grad()
    def prepare(self, path: str) -> None:
        """Constructs semantic kNN graph S0, collaborative candidate graph, and blended S4."""
        from freerec.utils import import_pickle

        cfg = self.cfg
        mfeats = []
        feature_manifest = []

        for mfile in cfg.mfiles:
            candidates = [
                os.path.join(path, mfile),
                os.path.join(cfg.root, cfg.dataset, mfile),
                os.path.join(cfg.root, "Processed", cfg.dataset, mfile),
                os.path.join("/kaggle/data", cfg.dataset, mfile),
                os.path.join("/kaggle/data/Processed", cfg.dataset, mfile),
                os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset, mfile),
                os.path.join("data", cfg.dataset, mfile),
                os.path.join("data", "Processed", cfg.dataset, mfile),
            ]
            mpath = next((c for c in candidates if os.path.exists(c)), None)
            if mpath is None:
                raise FileNotFoundError(f"Missing modality feature file {mfile} in {path}")
            
            feat = import_pickle(mpath)
            mfeats.append(feat)

            hasher = hashlib.sha256()
            with open(mpath, "rb") as h:
                for chunk in iter(lambda: h.read(1024 * 1024), b""):
                    hasher.update(chunk)
            feature_manifest.append({"file": mfile, "sha256": hasher.hexdigest()})

        # Build S0: exact STAIR baseline semantic graph
        edge_index = torch.cat(
            [self.get_knn_graph(f, k) for f, k in zip(mfeats, cfg.num_neighbors)],
            dim=1,
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce="sum")
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce="max")
        raw_semantic = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        ).coalesce()

        edge_index_norm, edge_weight_norm = freerec.graph.to_normalized(
            edge_index, edge_weight, normalization="sym"
        )
        baseline_s0 = torch.sparse_coo_tensor(
            edge_index_norm, edge_weight_norm, size=(self.Item.count, self.Item.count)
        ).to_sparse_csr()

        # Training interactions
        train_edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        self._build_train_interaction_table(train_edges)

        # Build blended S4 operator via Candidate Support Expansion
        graph_state: GraphStateV4 = build_calibrated_graph_v4(
            raw_semantic=raw_semantic,
            baseline_normalized=baseline_s0,
            train_edges=train_edges.long(),
            n_users=self.User.count,
            n_items=self.Item.count,
            arm=self.v4_arm,
            eta=getattr(cfg, "eta", 0.1),
            k_cf=getattr(cfg, "k_cf", 5),
            c_min=getattr(cfg, "c_min", 2),
            t_shrinkage=getattr(cfg, "t_shrinkage", 5.0),
            placebo_seed=getattr(cfg, "placebo_seed", getattr(cfg, "seed", 1)),
            seed=getattr(cfg, "seed", 1),
            cache_dir=getattr(cfg, "graph_cache_dir", None) or None,
        )

        self.register_buffer("mAdj", graph_state.operator)
        self.graph_metadata = graph_state.metadata
        self.feature_manifest = feature_manifest

        # Modality Interaction (MI) Initialization
        mfeats_w = [self.whitening(f) * k for f, k in zip(mfeats, cfg.num_neighbors)]
        mfeats_init = sum(mfeats_w).div(sum(cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            train_edges, normalization="left"
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R @ mfeats_init)

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward Stepwise Convolution (FSC) with intermediate view capture."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
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
        """Fast evaluation forward pass without layer intermediate list overhead."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
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

    def fit(self, data: Dict) -> torch.Tensor:
        """Computes combined BPR ranking loss + NLGCL contrastive loss."""
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # Robust index tensor shapes for einsum (supports (B,) and (B, K))
        u_idx = users.unsqueeze(1) if users.ndim == 1 else users
        p_idx = positives.unsqueeze(1) if positives.ndim == 1 else positives
        n_idx = negatives.unsqueeze(1) if negatives.ndim == 1 else negatives

        # Pairwise BPR ranking loss
        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[p_idx]),
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[n_idx]),
        )

        # In-batch NLGCL contrastive loss
        if self.training and self.lambda_nlgcl > 0.0:
            nlgcl_loss = self.nlgcl(layer_embeds, users, positives)
            return rec_loss + self.lambda_nlgcl * nlgcl_loss

        return rec_loss

    def reset_ranking_buffers(self) -> None:
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self) -> torch.Tensor:
        return self.ranking_buffer[self.User] @ self.ranking_buffer[self.Item].t()

    def recommend_from_pool(self, pool: torch.Tensor) -> torch.Tensor:
        user = self.ranking_buffer[self.User]
        item = self.ranking_buffer[self.Item][pool]
        return torch.einsum("ND,NKD->NK", user, item)
