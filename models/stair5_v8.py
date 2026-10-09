"""models/stair5_v8.py — STAIR5-v8 / WMSG-CSE Architecture.
============================================================
Weighted Multimodal Semantic Graph with Candidate Support Expansion (WMSG-CSE).

Integrates the four-stage specification from STAIR5_v8_Report.md:
1. Stage 1: Degree-Stratified Anchor-Procrustes (SAP) MI initialization (stair5_v8_alignment.py)
   or exact V4 control (alignment="off").
2. Stage 2: Continuous WMSG on frozen V4 support with power scaling f_p(a) = 0.05 + 0.95 * [max(0, a)]^p
   and conditional Relation-Proxy Gate (RPG) retention (stair5_v8_graph.py).
3. Stage 3: Current-degree symmetric normalization with isolated-node identity fallback,
   semantic blend S_{sem,8} = (1 - epsilon)*S_0 + epsilon*S_{mag}, and CSE blend S_8 = 0.9*S_{sem,8} + 0.1*S_bar_CF.
   (Theorem 1: ||S_8||_2 <= 1.0).
4. Stage 4: Matrix-free polynomial smoother P_8 with optional Dirichlet-floor bypass DF-BSC
   D_tilde = omega * D + (1 - omega) * P_8 D (stair5_v8_smoother.py).
   (Theorem 2: ||D_tilde||_F <= ||D||_F and omega^2 * E_8(D) <= E_8(D_tilde) <= E_8(D)).
5. Forward Representation, Loss, and Evaluator:
   Preserves immutable V4 FSC forward pass, NLGCL heterogeneous InfoNCE loss, and BPR objective.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
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
from models.stair5_v4_utils import atomic_torch_save, file_sha256
from models.stair5_v8_graph import (
    build_calibrated_graph_v8,
    GraphStateV8,
    sparse_fingerprint,
    tensor_to_scipy,
    ARMS_V8,
    resolve_v8_config,
)
from models.stair5_v8_alignment import align_modality_features
from optimizers.stair5_v8_smoother import STAIR5V8Smoother


class STAIR5_v8_Model(freerec.models.GenRecArch):
    """STAIR5-v8 / WMSG-CSE Model Architecture."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet, cfg: Any) -> None:
        super().__init__(dataset)
        self.cfg = resolve_v8_config(copy.copy(cfg))
        cfg = self.cfg
        self.num_layers = int(cfg.num_layers)

        arm = getattr(cfg, "v8_arm", "WMSG-core")
        if arm not in ARMS_V8:
            raise ValueError(f"Unknown v8_arm: {arm}. Must be one of {ARMS_V8}")
        self.v8_arm = arm

        # Embedding tables
        self.User.add_module(
            "embeddings", nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        # Preserve the native normalized train UI graph; do not mask unrelated
        # data/import errors with a different fallback graph construction.
        self.register_buffer("Adj", self.dataset.train().to_normalized_adj(normalization="sym"))
        self.register_buffer("beta3", cfg.beta3)

        # Effective lambda_nlgcl (V8 preserves V4 NLGCL InfoNCE loss with lambda=0.01)
        self.lambda_nlgcl = float(getattr(cfg, "lambda_nlgcl", 0.01))
        if not math.isfinite(self.lambda_nlgcl) or self.lambda_nlgcl < 0:
            raise ValueError("lambda_nlgcl must be finite and nonnegative.")

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # Initialize NLGCL module (exact V4 structure)
        self.nlgcl = NLGCL_Module(
            n_users=self.User.count,
            n_items=self.Item.count,
            G=getattr(cfg, "nlgcl_G", 1),
            tau=getattr(cfg, "nlgcl_tau", 0.2),
            alpha=getattr(cfg, "nlgcl_alpha", 0.5),
            chunk_size=getattr(cfg, "cl_chunk_size", 1024),
        )

    def reset_parameters(self) -> None:
        """Initializes weights using standard PyTorch / FreeRec conventions."""
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
        """Provides parameter groups with STAIR5V8Smoother on item embeddings."""
        omega = float(getattr(self.cfg, "bsc_residual", 0.0))
        groups = [
            {"params": list(self.User.parameters()), "smoother": None},
            {
                "params": list(self.Item.parameters()),
                "smoother": STAIR5V8Smoother(
                    lambda x: torch.sparse.mm(self.mAdj, x),
                    beta=self.beta3,
                    L=self.num_layers,
                    omega=omega,
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
        """SVD Whitening — centered SVD U[:, :d] * sqrt(N / d). Identical to V4 baseline."""
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
    def get_knn_graph(self, features: torch.Tensor, k: int = 5, cache_key: Optional[str] = None) -> torch.Tensor:
        """FP32 cosine top-k in anchor blocks with reference diagonal policy.

        No approximate neighbors. Ties use torch.topk as in FreeRec.
        """
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2 or len(features) != self.Item.count or not torch.isfinite(features).all():
            raise ValueError("Modality features must be finite [n_items, feature_dim] tensors.")
        if not 0 < k < len(features):
            raise ValueError("Semantic k must be between 1 and n_items - 1.")
        chunk = int(getattr(self.cfg, "knn_chunk_size", 1024))
        requested = getattr(self.cfg, "knn_device", "cpu")
        if chunk <= 0 or requested != "cpu" or features.device.type != "cpu":
            raise ValueError("Invalid kNN chunk size or device.")
        device = torch.device("cpu")

        safe_rows = int(128 * 2**20 // (3 * len(features) * 4))
        if safe_rows < 1:
            raise MemoryError("CPU kNN workspace cannot fit one catalog row.")
        chunk = min(chunk, 1024, safe_rows)
        signature = {
            "features": cache_key,
            "k": k,
            "chunk": chunk,
            "algorithm": "cosine_fp32_diagonal_minus10_torch_topk_v1",
            "torch": str(torch.__version__),
            "device": str(device),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "source": file_sha256(Path(__file__).with_name("stair5_v4.py")),
        }
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        directory = getattr(self.cfg, "graph_cache_dir", None)
        cache = Path(directory) / f"semantic_{fingerprint}.pt" if directory and cache_key else None
        record = {**signature, "fingerprint": fingerprint, "cache_hit": False}
        if not hasattr(self, "knn_metadata"):
            self.knn_metadata = []

        frozen_files = getattr(self.cfg, "v4_support_files", ())
        feature_position = len(self.knn_metadata)
        if isinstance(frozen_files, str):
            frozen_files = [p.strip() for p in frozen_files.split(",") if p.strip()]
        if frozen_files and len(frozen_files) != len(self.cfg.mfiles):
            raise ValueError("Frozen V4 support files must match the modality count.")
        frozen = Path(frozen_files[feature_position]) if frozen_files else None
        if frozen is not None or (cache and cache.is_file()):
            if frozen is not None:
                cache = frozen
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            result = payload["edges"]
            recorded = payload["signature"]
            compatible = recorded == signature if frozen is None else (
                recorded.get("features") == cache_key and recorded.get("k") == k
                and recorded.get("algorithm") == signature["algorithm"]
            )
            if not compatible or result.shape != (2, len(features) * k):
                raise ValueError("Semantic cache signature or edge shape mismatch.")
            if result.dtype != torch.int64 or result.min() < 0 or result.max() >= len(features):
                raise ValueError("Cached candidate IDs are invalid.")
            if (result[0] == result[1]).any() or not torch.equal(torch.bincount(result[0], minlength=len(features)), torch.full((len(features),), k)):
                raise ValueError("Cached candidates contain self-edges or invalid row counts.")
            if torch.unique(result[0] * len(features) + result[1]).numel() != result.size(1):
                raise ValueError("Cached candidates contain duplicate directed pairs.")
            digest = hashlib.sha256(result.numpy().tobytes()).hexdigest()
            if digest != payload["sha256"]:
                raise ValueError("Semantic cache content checksum mismatch.")
            record["cache_hit"] = True
            record["loaded_signature"] = recorded
            record["candidate_sha256"] = digest
            record["frozen_support"] = frozen is not None
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
            record["candidate_sha256"] = hashlib.sha256(result.numpy().tobytes()).hexdigest()
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

    def _build_train_interaction_table(self, edge_index_ui: torch.Tensor) -> None:
        """Builds sorted train interaction keys for O(log N) lookup."""
        edges = edge_index_ui.detach().cpu().long()
        flat_keys = torch.unique(edges[0] * self.Item.count + edges[1], sorted=True)
        self.register_buffer("train_pair_keys", flat_keys, persistent=False)

    @torch.no_grad()
    def prepare(self, path: str) -> None:
        """Constructs MI initialization and the final blended S8 operator."""
        from freerec.utils import import_pickle

        cfg = self.cfg
        if len(cfg.mfiles) != len(cfg.num_neighbors) or not cfg.mfiles or sum(cfg.num_neighbors) <= 0:
            raise ValueError("Each modality requires a positive semantic neighbor count.")
        mfeats: List[torch.Tensor] = []
        feature_manifest: List[Dict[str, str]] = []

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

        # Discover directed kNN neighbor IDs for each modality
        knn_edges_list = [
            self.get_knn_graph(f, k, cache_key=m["sha256"])
            for f, k, m in zip(mfeats, cfg.num_neighbors, feature_manifest)
        ]

        # Build baseline S0: exact STAIR baseline unweighted semantic graph
        edge_index = torch.cat(knn_edges_list, dim=1)
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

        # Compute train item degrees for stratification
        # Deduplicated edges for degrees and CF statistics
        i_idx = self.train_pair_keys.remainder(self.Item.count)
        train_degrees = torch.bincount(i_idx, minlength=self.Item.count).long()

        # ---------------------------------------------------------------------
        # Stage 1: Modality Initialization & Alignment (SAP)
        # ---------------------------------------------------------------------
        # Compute whitened features: Z_t = mfeats_whitened[0], Z_v = mfeats_whitened[1]
        mfeats_whitened = [self.whitening(f) for f in mfeats]
        Z_t = mfeats_whitened[0]
        Z_v = mfeats_whitened[1]

        fused_MI, Q_matrix, sap_audit = align_modality_features(
            Z_t=Z_t,
            Z_v=Z_v,
            train_degrees=train_degrees,
            cfg=cfg,
        )
        self.sap_audit = sap_audit
        self.sap_transform = Q_matrix

        # Copy aligned MI into item embeddings
        self.Item.embeddings.weight.data.copy_(fused_MI)

        # User MI initialization via left-normalized training interaction operator
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            train_edges, normalization="left"
        )
        R_ui = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()
        self.User.embeddings.weight.data.copy_(R_ui @ fused_MI)

        # ---------------------------------------------------------------------
        # Stage 2 & 3: WMSG Continuous Graph Construction & Blend S_8
        # ---------------------------------------------------------------------
        # Optional metadata dictionary for relation proxy gate (RPG)
        metadata_dict = getattr(cfg, "metadata_dict", None)
        if metadata_dict is not None:
            if not isinstance(metadata_dict, dict):
                raise ValueError("Item metadata must be a dictionary.")
            if cfg.relation_gate != "off" and not (
                metadata_dict.get("mapping_verified") is True
                and metadata_dict.get("comparable_snapshot") is True
            ):
                raise ValueError("RPG requires verified item mapping and comparable price snapshots.")
            if getattr(cfg, "metadata_fingerprint", None) is None:
                encoded = json.dumps(metadata_dict, sort_keys=True, default=lambda value: (
                    value.tolist() if hasattr(value, "tolist") else sorted(value)
                )).encode()
                cfg.metadata_fingerprint = hashlib.sha256(encoded).hexdigest()

        graph_state: GraphStateV8 = build_calibrated_graph_v8(
            raw_semantic=raw_semantic,
            baseline_normalized=baseline_s0,
            train_edges=train_edges.long(),
            n_users=self.User.count,
            n_items=self.Item.count,
            mfeats=mfeats,
            knn_edges_list=knn_edges_list,
            v8_arm=self.v8_arm,
            semantic_mode=getattr(cfg, "semantic_mode", "weighted"),
            edge_power=float(getattr(cfg, "edge_power", 2.0)),
            edge_floor=float(getattr(cfg, "edge_floor", 0.05)),
            semantic_mix=float(getattr(cfg, "semantic_mix", 1.0)),
            relation_gate=getattr(cfg, "relation_gate", "off"),
            relation_strength=float(getattr(cfg, "relation_strength", 0.0)),
            relation_threshold=float(getattr(cfg, "relation_threshold", 0.7)),
            relation_temperature=float(getattr(cfg, "relation_temperature", 0.1)),
            metadata_dict=metadata_dict,
            eta=float(getattr(cfg, "eta", 0.1)),
            k_cf=int(getattr(cfg, "k_cf", 5)),
            c_min=int(getattr(cfg, "c_min", 2)),
            t_shrinkage=float(getattr(cfg, "t_shrinkage", 5.0)),
            cache_dir=getattr(cfg, "graph_cache_dir", None) or None,
            block_size=int(getattr(cfg, "cf_block_size", 64)),
            memory_budget_mib=int(getattr(cfg, "cf_memory_budget_mib", 128)),
            placebo_seed=int(getattr(cfg, "placebo_seed", 1)),
        )

        self.register_buffer("mAdj", graph_state.operator)
        self.graph_metadata = graph_state.metadata
        self.feature_manifest = feature_manifest
        self.data_fingerprint = hashlib.sha256(
            train_edges.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        self.graph_fingerprint = sparse_fingerprint(tensor_to_scipy(self.mAdj))

        # Hashes of dataset files and source implementations
        self.split_manifest = {p.name: file_sha256(p) for p in sorted(Path(path).glob("*.txt"))}
        self.split_manifest.update({p.name: file_sha256(p) for p in sorted(Path(path).glob("*.csv"))})

        sources = [
            Path(__file__),
            Path(__file__).with_name("stair5_v8_alignment.py"),
            Path(__file__).with_name("stair5_v8_graph.py"),
            Path(__file__).with_name("stair5_v4_graph.py"),
            Path(__file__).with_name("stair5_v4.py"),
            Path(__file__).with_name("stair5_v4_objectives.py"),
            Path(__file__).with_name("stair5_v4_utils.py"),
            Path(__file__).with_name("freerec_compat.py"),
            Path(__file__).resolve().parents[1] / "optimizers/stair5_v8_smoother.py",
            Path(__file__).resolve().parents[1] / "optimizers/AdamW.py",
            Path(__file__).resolve().parents[1] / "main_stair5_v8.py",
        ]
        self.source_manifest = {p.name: file_sha256(p) for p in sources if p.is_file()}

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

    def training_objective(self, data: Dict):
        """Return total, BPR and raw NLGCL tensors without synchronizing the GPU."""
        userEmbds, itemEmbds, layers = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        u_idx = users.unsqueeze(1) if users.ndim == 1 else users
        p_idx = positives.unsqueeze(1) if positives.ndim == 1 else positives
        n_idx = negatives.unsqueeze(1) if negatives.ndim == 1 else negatives
        rec = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[p_idx]),
            torch.einsum("BKD,BKD->BK", userEmbds[u_idx], itemEmbds[n_idx]),
        )
        cl = (
            self.nlgcl(layers, users, positives)
            if self.training and self.lambda_nlgcl > 0
            else rec.new_zeros(())
        )
        return rec + self.lambda_nlgcl * cl, rec, cl

    def fit(self, data: Dict) -> torch.Tensor:
        return self.training_objective(data)[0]

    def get_extra_state(self):
        """Immutable compatibility contract for model and training checkpoints."""
        names = (
            "embedding_dim",
            "num_layers",
            "gamma",
            "num_neighbors",
            "mfiles",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "batch_size",
            "seed",
            "eval_freq",
            "ranking",
            "which4best",
            "num_workers",
            "nlgcl_G",
            "nlgcl_tau",
            "nlgcl_alpha",
            "eta",
            "k_cf",
            "c_min",
            "t_shrinkage",
            "v8_arm",
            "semantic_mode",
            "edge_power",
            "edge_floor",
            "semantic_mix",
            "alignment",
            "anchor_seed",
            "anchor_per_stratum",
            "relation_gate",
            "relation_strength",
            "relation_threshold",
            "relation_temperature",
            "bsc_residual",
            "lambda_dirichlet",
            "placebo_seed",
            "knn_device",
            "knn_chunk_size",
            "cl_chunk_size",
            "metadata_fingerprint",
        )
        return {
            "version": 8,
            "arm": self.v8_arm,
            "lambda_nlgcl": self.lambda_nlgcl,
            "config": {k: getattr(self.cfg, k, None) for k in names},
            "data_fingerprint": self.data_fingerprint,
            "graph_fingerprint": self.graph_fingerprint,
            "sap_audit": self.sap_audit,
            "sap_transform_sha256": self.sap_audit.get("transform_sha256"),
            "candidate_hashes": [m["candidate_sha256"] for m in self.knn_metadata],
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
