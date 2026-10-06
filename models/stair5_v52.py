"""models/stair5_v52.py — STAIR5-v5.2 / NLGCL-BPE Architecture.
=============================================================
Budgeted Path Expansion (BPE) for Item Update Smoothing in RecSys.

Key Components:
1. Exact STAIR-NLGCL v4 backbone:
   - Modality Interaction (MI) SVD whitening initialization.
   - Forward Stepwise Convolution (FSC, L=3) on normalized train UI graph.
   - BPR ranking loss + heterogeneous in-batch InfoNCE NLGCL objective (tau=0.2, alpha=0.5, G=1).
   - AdamWSEvo optimizer: Item embeddings smoothed with S_5.2; user embeddings unsmoothed.
2. Budgeted Path Expansion (BPE) for item-item BSC smoothing operator:
   - W_star = W_1 + W_add
   - S_star = SymNormIso(W_star)
   - S_5.2  = (1 - eta) * S_0 + eta * S_star (eta=0.1)
3. Factorial & Ablation Arms:
   - 'P-BPE': Proposed primary model (nu=0.5, beta_edges=0.25).
   - 'P-BPE-low': Conservative mass budget (nu=0.2, beta_edges=0.25).
   - 'C-V4': Exact v4 fallback baseline.
   - 'C-Direct': Reserve direct co-occurrence outside top-5 matching budgets.
   - 'C-Radius': v4 with BSC L=4, FSC L=3.
   - 'C-NoScale': BPE with kappa=1.0.
   - 'C-M2': BPE with m_ij >= 2.
   - 'C-Overlap': BPE with S_0 candidate overlap permitted.
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

import models.freerec_compat  # Install compatibility before importing FreeRec.
import freerec

from models.stair5_v4_objectives import NLGCL_Module
from models.stair5_v52_graph import (
    ARMS_V52,
    build_calibrated_graph_v52,
    GraphStateV52,
    sparse_fingerprint,
    tensor_to_scipy,
)
from models.stair5_v52_utils import atomic_torch_save, file_sha256
from optimizers.stair5_v52_smoother import STAIR5V52Smoother


class STAIR5_v52_Model(freerec.models.GenRecArch):
    """STAIR5-v5.2 (NLGCL-BPE) Model Architecture."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet, cfg) -> None:
        super().__init__(dataset)
        self.cfg = copy.copy(cfg)
        self.num_layers = cfg.num_layers

        arm = getattr(cfg, "v52_arm", "P-BPE")
        if arm not in ARMS_V52:
            raise ValueError(f"Unknown v5.2 arm: {arm}. Must be one of {ARMS_V52}")
        self.v52_arm = arm
        self.bsc_num_layers = 4 if arm == "C-Radius" else self.num_layers

        # Embedding tables
        self.User.add_module(
            "embeddings", nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        # Preserve the native normalized train UI graph (FSC)
        self.register_buffer("Adj", self.dataset.train().to_normalized_adj(normalization="sym"))
        self.register_buffer("beta3", cfg.beta3)

        # Effective lambda_nlgcl
        self.lambda_nlgcl = float(getattr(cfg, "lambda_nlgcl", 0.01))
        if not math.isfinite(self.lambda_nlgcl) or self.lambda_nlgcl < 0:
            raise ValueError("lambda_nlgcl must be finite and nonnegative.")

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # Standard NLGCL Module (heterogeneous InfoNCE)
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
        """Provides parameter groups with STAIR5V52Smoother on item embeddings."""
        groups = [
            {"params": list(self.User.parameters()), "smoother": None},
            {
                "params": list(self.Item.parameters()),
                "smoother": STAIR5V52Smoother(
                    lambda x: torch.sparse.mm(self.mAdj, x),
                    beta=self.beta3,
                    L=self.bsc_num_layers,
                ),
            },
        ]
        ids = [id(p) for group in groups for p in group["params"]]
        if len(set(ids)) != len(ids) or set(ids) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("Every trainable parameter must occur in exactly one optimizer group.")
        return groups

    def sure_trainpipe(self, batch_size: int):
        """Native reference sampler: shuffled pairs, one negative, fixed batch size."""
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

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

    @torch.no_grad()
    def get_knn_graph(self, features: torch.Tensor, k: int = 5, cache_key=None) -> torch.Tensor:
        """FP32 cosine top-k in anchor blocks with diagonal exclusion."""
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2 or len(features) != self.Item.count or not torch.isfinite(features).all():
            raise ValueError("Modality features must be finite [n_items, feature_dim] tensors.")
        if not 0 < k < len(features):
            raise ValueError("Semantic k must be between 1 and n_items - 1.")
        chunk = int(getattr(self.cfg, "knn_chunk_size", 1024))
        requested = getattr(self.cfg, "knn_device", "auto")
        if chunk <= 0 or requested not in ("cpu", "cuda", "auto"):
            raise ValueError("Invalid kNN chunk size or device.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA kNN requested but no CUDA device is available.")

        device = torch.device("cpu")
        if requested != "cpu" and torch.cuda.is_available():
            configured = torch.device(self.cfg.device)
            device = configured if configured.type == "cuda" else torch.device("cuda")
            free, _ = torch.cuda.mem_get_info(device)
            budget = int(0.60 * free) - 3 * features.numel() * features.element_size()
            if budget < features.size(0) * 4:
                if requested == "cuda":
                    raise MemoryError("Insufficient GPU kNN headroom; reduce preprocessing memory or use CPU.")
                device = torch.device("cpu")
            else:
                chunk = min(chunk, budget // (features.size(0) * 4))

        signature = {
            "features": cache_key,
            "k": k,
            "chunk": chunk,
            "algorithm": "cosine_fp32_diagonal_minus10_torch_topk_v1",
            "torch": str(torch.__version__),
            "device": str(device),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "source": file_sha256(Path(__file__)),
        }
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        directory = getattr(self.cfg, "graph_cache_dir", None)
        cache = Path(directory) / f"semantic_{fingerprint}.pt" if directory and cache_key else None
        record = {**signature, "fingerprint": fingerprint, "cache_hit": False}
        if not hasattr(self, "knn_metadata"):
            self.knn_metadata = []

        if cache and cache.is_file():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            result = payload["edges"]
            if payload["signature"] != signature or result.shape != (2, len(features) * k):
                raise ValueError("Semantic cache signature or edge shape mismatch.")
            digest = hashlib.sha256(result.numpy().tobytes()).hexdigest()
            if digest != payload["sha256"]:
                raise ValueError("Semantic cache content checksum mismatch.")
            record["cache_hit"] = True
        else:
            normalized = F.normalize(features.to(device), dim=-1)
            rows, columns = [], []
            for start in range(0, len(features), chunk):
                stop = min(start + chunk, len(features))
                similarity = normalized[start:stop] @ normalized.t()
                local = torch.arange(stop - start, device=device)
                similarity[local, start + local] = -10.0
                neighbors = similarity.topk(k, dim=1).indices
                rows.append(torch.arange(start, stop)[:, None].expand(-1, k).reshape(-1))
                columns.append(neighbors.reshape(-1).cpu())
                del similarity
            result = torch.stack((torch.cat(rows), torch.cat(columns)))
            if cache:
                atomic_torch_save(
                    {
                        "edges": result,
                        "signature": signature,
                        "sha256": hashlib.sha256(result.numpy().tobytes()).hexdigest(),
                    },
                    cache,
                )
        self.knn_metadata.append(record)
        return result

    @torch.no_grad()
    def prepare(self, path: str) -> None:
        """Constructs multimodal semantic graph S_0 and BPE blended operator S_5.2."""
        from freerec.utils import import_pickle

        cfg = self.cfg
        if len(cfg.mfiles) != len(cfg.num_neighbors) or not cfg.mfiles or sum(cfg.num_neighbors) <= 0:
            raise ValueError("Each modality requires a positive semantic neighbor count.")
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

            feat = torch.as_tensor(import_pickle(mpath), dtype=torch.float32)
            if feat.ndim != 2 or feat.size(0) != self.Item.count or min(feat.shape) < cfg.embedding_dim:
                raise ValueError(f"Feature shape cannot initialize the declared catalog/dimension: {mpath}")
            if not torch.isfinite(feat).all():
                raise ValueError(f"Nonfinite modality features: {mpath}")
            mfeats.append(feat)

            hasher = hashlib.sha256()
            with open(mpath, "rb") as h:
                for chunk in iter(lambda: h.read(1024 * 1024), b""):
                    hasher.update(chunk)
            feature_manifest.append({"file": mfile, "sha256": hasher.hexdigest()})

        # Build S0: exact STAIR baseline multimodal semantic graph
        edge_index = torch.cat(
            [self.get_knn_graph(f, k, cache_key=m["sha256"])
             for f, k, m in zip(mfeats, cfg.num_neighbors, feature_manifest)],
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

        # Build S_5.2 operator via Budgeted Path Expansion (BPE)
        expansion_enabled = getattr(cfg, "expansion_enabled", True) and (self.v52_arm != "C-V4")
        graph_state: GraphStateV52 = build_calibrated_graph_v52(
            raw_semantic=raw_semantic,
            baseline_normalized=baseline_s0,
            train_edges=train_edges.long(),
            n_users=self.User.count,
            n_items=self.Item.count,
            arm=self.v52_arm,
            expansion_enabled=expansion_enabled,
            eta=getattr(cfg, "eta", 0.1),
            k_cf=getattr(cfg, "k_cf", 5),
            c_min=getattr(cfg, "c_min", 2),
            t_shrinkage=getattr(cfg, "t_shrinkage", 5.0),
            k_seed=getattr(cfg, "k_seed", 10),
            t_path=getattr(cfg, "t_path", 1.0),
            k_add=getattr(cfg, "k_add", 3),
            beta_edges=getattr(cfg, "beta_edges", 0.25),
            nu=getattr(cfg, "nu", 0.5),
            cache_dir=getattr(cfg, "graph_cache_dir", None) or None,
            block_size=getattr(cfg, "cf_block_size", 64),
            memory_budget_mib=getattr(cfg, "cf_memory_budget_mib", 128.0),
        )

        self.register_buffer("mAdj", graph_state.operator)
        self.graph_metadata = graph_state.metadata
        self.feature_manifest = feature_manifest
        self.data_fingerprint = hashlib.sha256(train_edges.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        self.graph_fingerprint = sparse_fingerprint(tensor_to_scipy(self.mAdj))

        # Split and source manifests
        self.split_manifest = {p.name: file_sha256(p) for p in sorted(Path(path).glob("*.txt"))}
        self.split_manifest.update({p.name: file_sha256(p) for p in sorted(Path(path).glob("*.csv"))})
        sources = [
            Path(__file__),
            Path(__file__).with_name("stair5_v52_graph.py"),
            Path(__file__).with_name("stair5_v52_utils.py"),
            Path(__file__).with_name("stair5_v4_objectives.py"),
            Path(__file__).with_name("stair5_v4_graph.py"),
            Path(__file__).resolve().parents[1] / "optimizers/stair5_v52_smoother.py",
            Path(__file__).resolve().parents[1] / "optimizers/AdamW.py",
            Path(__file__).resolve().parents[1] / "main_stair5_v52.py",
        ]
        self.source_manifest = {p.name: file_sha256(p) for p in sources if p.is_file()}

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
        """Forward Stepwise Convolution (FSC, L=3) with intermediate view capture."""
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

    def training_objective(self, data: Dict):
        """Returns total loss, BPR loss, and raw NLGCL loss without host-GPU barriers."""
        userEmbds, itemEmbds, layers = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        u_idx = users.unsqueeze(1) if users.ndim == 1 else users
        p_idx = positives.unsqueeze(1) if positives.ndim == 1 else positives
        n_idx = negatives.unsqueeze(1) if negatives.ndim == 1 else negatives
        rec = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[p_idx]),
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[n_idx]),
        )
        cl = self.nlgcl(layers, users, positives) if self.training and self.lambda_nlgcl > 0 else rec.new_zeros(())
        return rec + self.lambda_nlgcl * cl, rec, cl

    def fit(self, data: Dict) -> torch.Tensor:
        return self.training_objective(data)[0]

    def get_extra_state(self):
        """Immutable compatibility contract for model and training checkpoints."""
        names = (
            "embedding_dim", "num_layers", "gamma", "num_neighbors", "mfiles",
            "lr", "weight_decay", "beta1", "beta2", "batch_size", "seed",
            "eval_freq", "ranking", "which4best", "num_workers",
            "nlgcl_G", "nlgcl_tau", "nlgcl_alpha", "eta", "k_cf", "c_min",
            "t_shrinkage", "k_seed", "t_path", "k_add", "beta_edges", "nu", "expansion_enabled",
            "cl_chunk_size",
        )
        return {
            "version": "5.2",
            "arm": self.v52_arm,
            "lambda_nlgcl": self.lambda_nlgcl,
            "config": {k: getattr(self.cfg, k, None) for k in names},
            "data_fingerprint": self.data_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "features": self.feature_manifest,
            "splits": self.split_manifest,
            "sources": self.source_manifest,
            "torch": str(torch.__version__),
            "freerec": freerec.__version__,
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("Checkpoint provenance mismatch; model state has not been modified.")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint architecture/configuration/data provenance does not match this model.")
        self.ranking_buffer = {}

    @torch.no_grad()
    def reset_ranking_buffers(self) -> None:
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data: Dict) -> torch.Tensor:
        users = data[self.User]
        if users.ndim == 1:
            users = users[:, None]
        user = self.ranking_buffer[self.User][users]
        return torch.einsum("BKD,ND->BN", user, self.ranking_buffer[self.Item])

    def recommend_from_pool(self, data: Dict) -> torch.Tensor:
        users = data[self.User]
        if users.ndim == 1:
            users = users[:, None]
        user = self.ranking_buffer[self.User][users]
        item = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", user, item)
