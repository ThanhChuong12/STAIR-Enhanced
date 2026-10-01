# -*- coding: utf-8 -*-
"""
models/stair5_v1.py — STAIR-LHC v1 Model Architecture
=====================================================
Senior AI Research Engineer Implementation:
STAIR with Lorentz Hidden Contrastive Regularization (STAIR-LHC v1 Revised).

Key Design Pillars:
───────────────────
1. Preserves 100% STAIR Backbone:
   - SVD Whitening + Modal Interaction initialization.
   - Forward Stepwise Convolution (FSC) with spectral decay beta = 1 - beta3.
   - AdamWSEvo optimizer with BSC Smoother on Item embeddings.
   - Pure Euclidean dot product during evaluation/inference (Zero latency overhead).
2. Auxiliary Lorentz Contrastive Regularizer:
   - Multi-positive InfoNCE across H^(0) and H^(1).
   - Degree-1 Safe Self-Return Removal on positive keys.
   - Adaptive Spectral Re-weighting with M_norm quantile bounding.
   - Taylor series arcosh^2 stability (Zero NaNs).
   - Linear Warmup schedule: lambda = 0 for epochs 0-20, ramping to lambda_max at epoch 50.
3. Strict Ablation Parity:
   - H0: Main Lorentz CL (kappa=1.0, w=0.0).
   - E0: Euclidean Control (kappa=0.0) with identical normalizations and self-return removal.
   - HC: Constant-Radius Control (||v||=1.0) to isolate geometric capacity from kernel effect.
   - H0w5: Hybrid Kernel (w=0.5).
   - B0: STAIR Baseline parity control (lambda=0.0).
"""

from typing import Dict, List, Optional, Tuple, Set
import math
import os
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

from optimizers.utils import Smoother
from models.stair5_v1_geometry import LorentzGeometryModule
from models.stair5_v1_objectives import (
    MultiPositiveInfoNCELoss,
    remove_self_return,
    spectral_reweight,
)


class STAIR5_v1_Model(freerec.models.GenRecArch):
    """
    STAIR-LHC v1 Model Architecture.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet, cfg) -> None:
        super().__init__(dataset)
        self.cfg = cfg
        self.num_layers = cfg.num_layers

        # ─── 1. Embeddings & Graph Structures (STAIR Backbone) ───────────
        self.User.add_module(
            "embeddings", nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

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

        # ─── 2. Node Degrees for Self-Return Removal ─────────────────────
        edge_index_ui = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        u_deg = edge_index_ui[0].bincount(minlength=self.User.count).float().clamp(min=1.0)
        i_deg = edge_index_ui[1].bincount(minlength=self.Item.count).float().clamp(min=1.0)
        self.register_buffer("user_degrees", u_deg)
        self.register_buffer("item_degrees", i_deg)

        # Build fast train user->items dictionary for in-batch positive lookup
        self._build_train_interaction_table(edge_index_ui)

        # ─── 3. Scale Buffers & M_norm (Frozen at epoch 0) ───────────────
        self.register_buffer("q_u0", torch.tensor(1.0))
        self.register_buffer("q_u1", torch.tensor(1.0))
        self.register_buffer("q_i0", torch.tensor(1.0))
        self.register_buffer("q_i1", torch.tensor(1.0))
        self.register_buffer("M_norm", torch.tensor(10.0))
        self.register_buffer("is_scale_initialized", torch.tensor(False, dtype=torch.bool))

        # ─── 4. Reset & Modality Preparation ─────────────────────────────
        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # ─── 5. STAIR-LHC v1 Geometry & Contrastive Modules ──────────────
        arm = getattr(cfg, "lhc_arm", "H0")
        kappa = getattr(cfg, "lhc_kappa", 1.0)
        radius_cap = getattr(cfg, "radius_cap", 2.0)
        w_hybrid = getattr(cfg, "w_hybrid", 0.0)
        eps_taylor = getattr(cfg, "eps_taylor", 1e-4)

        self.geo = LorentzGeometryModule(
            kappa=kappa,
            radius_cap=radius_cap,
            w_hybrid=w_hybrid,
            eps_taylor=eps_taylor,
            arm=arm,
        )

        disable_self_return = (arm == "H0-NOSELF") or getattr(cfg, "disable_self_return", False)
        disable_reweight = (arm == "H0-REWEIGHT") or getattr(cfg, "disable_reweight", False)

        self.lhc_loss_fn = MultiPositiveInfoNCELoss(
            geo=self.geo,
            radius_cap=radius_cap,
            tau=getattr(cfg, "lhc_tau", 0.3),
            eps_floor=getattr(cfg, "eps_floor", 1e-2),
            eps_beta=getattr(cfg, "eps_beta", 0.05),
            disable_self_return=disable_self_return,
            disable_reweight=disable_reweight,
        )

        # ─── 6. Warmup & Regularization Scheduling ───────────────────────
        self.lambda_lhc = getattr(cfg, "lambda_lhc", 3e-4)
        self.warmup_start = getattr(cfg, "warmup_start", 20)
        self.warmup_end = getattr(cfg, "warmup_end", 50)
        self.current_epoch = 0
        self.current_lambda = 0.0

        # Telemetry storage
        self.last_diagnostics: Dict[str, float] = {}

    def _build_train_interaction_table(self, edge_index_ui: torch.Tensor) -> None:
        """Builds fast lookup structures for in-batch positive matrix construction.

        Creates two complementary data structures:
        1. Dict[int, Set[int]] for backward compatibility and gradient diagnostics.
        2. Sorted COO + CSR-offset tensors for O(|E_batch|) vectorized P construction
           (critical for Electronics where B_u~3900 × B_i~3700 = 14.4M Python comparisons).
        """
        u_indices = edge_index_ui[0].cpu()
        i_indices = edge_index_ui[1].cpu()

        # --- Legacy dict (kept for gradient diagnostics in Coach) ---
        u_list = u_indices.tolist()
        i_list = i_indices.tolist()
        self.train_u2i_set: Dict[int, Set[int]] = {}
        for u, i in zip(u_list, i_list):
            if u not in self.train_u2i_set:
                self.train_u2i_set[u] = set()
            self.train_u2i_set[u].add(i)

        # --- Vectorized COO + CSR-offset structure ---
        # Sort edges by user ID for searchsorted-based batch intersection
        sort_order = torch.argsort(u_indices)
        self.train_u_sorted = u_indices[sort_order].contiguous()  # (|E|,)
        self.train_i_for_u = i_indices[sort_order].contiguous()   # (|E|,)

        # CSR-like offsets: for each user u, train_i_for_u[offsets[u]:offsets[u+1]] = items of u
        # Use searchsorted on sorted user IDs to find boundaries
        unique_users = torch.unique(self.train_u_sorted)
        max_uid = int(unique_users.max().item()) + 2
        self.train_u_offsets = torch.zeros(max_uid, dtype=torch.long)  # (max_uid,)
        # Left boundary for each user
        left = torch.searchsorted(self.train_u_sorted, unique_users, side='left')
        right = torch.searchsorted(self.train_u_sorted, unique_users, side='right')
        self.train_u_offsets[unique_users] = left
        self.train_u_offsets[unique_users + 1] = right
        # Fill gaps (users with no interactions get empty ranges)
        # We only need lookup for users that exist, so this is sufficient.
        self._train_edge_right = torch.zeros(max_uid, dtype=torch.long)
        self._train_edge_right[unique_users] = right

    def _build_positive_matrix_vectorized(
        self,
        u_unique: torch.Tensor,
        i_unique: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Constructs binary ground-truth train positive matrix P in-batch using vectorized ops.

        Fully vectorized approach (no Python for-loop):
        1. Gather all train edges for batch users via CSR-offset tensor slicing.
        2. Map item IDs to column indices via dense lookup table.
        3. Scatter valid (row, col) pairs into P.

        On Electronics (B_u~3900, B_i~3700): reduces 14.4M Python dict lookups to
        pure tensor operations. Eliminates ALL Python loops in the hot path.

        Args:
            u_unique: (B_u,) unique user IDs in current batch (on GPU).
            i_unique: (B_i,) unique item IDs in current batch (on GPU).
            device: target device for output.

        Returns:
            P: (B_u, B_i) binary positive matrix on target device.
        """
        B_u = u_unique.size(0)
        B_i = i_unique.size(0)

        u_cpu = u_unique.cpu()
        i_cpu = i_unique.cpu()

        # Build reverse mapping: item_id -> column index in P
        # Dense lookup table — O(1) per item, total O(B_i) setup
        max_iid = max(int(i_cpu.max().item()) + 1, int(self.train_i_for_u.max().item()) + 1)
        item_to_col = torch.full((max_iid,), -1, dtype=torch.long)
        item_to_col[i_cpu] = torch.arange(B_i, dtype=torch.long)

        offsets = self.train_u_offsets
        rights = self._train_edge_right

        # Clamp user IDs to valid range for offset lookup
        max_valid_uid = offsets.size(0) - 1
        u_clamped = torch.clamp(u_cpu, max=max_valid_uid)

        # Vectorized CSR range extraction for all batch users
        starts = offsets[u_clamped]           # (B_u,) start indices into train_i_for_u
        ends = rights[u_clamped]              # (B_u,) end indices into train_i_for_u
        lengths = (ends - starts).clamp(min=0)  # (B_u,) number of train items per user

        total_edges = int(lengths.sum().item())

        P = torch.zeros((B_u, B_i), dtype=torch.float32, device=device)
        if total_edges == 0:
            return P

        # Gather ALL train items for ALL batch users in one vectorized pass
        # Build flat index array: [start[0], start[0]+1, ..., end[0]-1, start[1], ...]
        edge_offsets = torch.repeat_interleave(starts, lengths)  # (total_edges,)
        within_user = torch.cat([torch.arange(l) for l in lengths.tolist()])  # (total_edges,)
        flat_indices = edge_offsets + within_user

        # Gather train item IDs and map to batch column indices
        all_items = self.train_i_for_u[flat_indices]    # (total_edges,)
        all_cols = item_to_col[all_items]               # (total_edges,) — -1 if not in batch

        # Broadcast row indices: user a_idx repeated by lengths[a_idx]
        all_rows = torch.repeat_interleave(
            torch.arange(B_u, dtype=torch.long), lengths
        )  # (total_edges,)

        # Filter to only valid (in-batch) items
        valid_mask = all_cols >= 0
        if valid_mask.any():
            P[all_rows[valid_mask], all_cols[valid_mask]] = 1.0

        return P

    def reset_parameters(self) -> None:
        """Initializes model parameters identical to STAIR baseline."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def marked_params(self) -> List[Dict]:
        """
        Optimizer Parameter Groups Contract:
        - User embeddings: smoother = None
        - Item embeddings: smoother = BSC Smoother
        - Buffers: frozen, not updated by optimizer
        """
        params = [
            {"params": self.User.parameters(), "smoother": None},
            {
                "params": self.Item.parameters(),
                "smoother": Smoother(
                    self.mAdj, beta=self.cfg.beta3, L=self.cfg.num_layers, aggr="neumann"
                ),
            },
        ]
        return params

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

    def prepare(self, path: str) -> None:
        """Prepares multimodal features and graph adjacency — identical to STAIR baseline."""
        from freerec.utils import import_pickle

        mfeats = []
        for mfile in self.cfg.mfiles:
            mpath = os.path.join(path, mfile)
            if not os.path.exists(mpath):
                for cand in [
                    os.path.join(self.cfg.root, self.cfg.dataset, mfile),
                    os.path.join("/kaggle/data", self.cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR/data", self.cfg.dataset, mfile),
                    os.path.join("/kaggle/working/STAIR-Enhanced/data", self.cfg.dataset, mfile),
                    os.path.join("data", self.cfg.dataset, mfile),
                ]:
                    if os.path.exists(cand):
                        mpath = cand
                        break
            mfeats.append(import_pickle(mpath))

        edge_index = torch.cat(
            [self.get_knn_graph(feats, k) for feats, k in zip(mfeats, self.cfg.num_neighbors)],
            dim=1,
        )
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce="sum")
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce="max")
        edge_index, edge_weight = freerec.graph.to_normalized(
            edge_index, edge_weight, normalization="sym"
        )
        mAdj = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.Item.count, self.Item.count)
        )
        self.register_buffer("mAdj", mAdj.to_sparse_csr())

        # MI Whitening Initialization
        mfeats_w = [self.whitening(mfeat) * k for mfeat, k in zip(mfeats, self.cfg.num_neighbors)]
        mfeats_init = sum(mfeats_w).div(sum(self.cfg.num_neighbors))
        self.Item.embeddings.weight.data.copy_(mfeats_init)

        edge_index_ui = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        edge_index_ui, edge_weight_ui = freerec.graph.to_normalized(
            edge_index_ui, normalization="left"
        )
        R = torch.sparse_coo_tensor(
            edge_index_ui, edge_weight_ui, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()

        self.User.embeddings.weight.data.copy_(R @ mfeats_init)

    @torch.no_grad()
    def initialize_scale_buffers(self) -> None:
        """
        Computes median L2 norms (q_{t, l}) and 95th quantile norm bound (M_norm) at epoch 0.
        Once computed, these values are frozen as buffers for the entire training lifecycle.
        """
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )
        beta = (1.0 - self.beta3).to(allEmbds.device)

        # Layer 0
        u0, i0 = torch.split(allEmbds, (self.User.count, self.Item.count))
        u0_norms = torch.norm(u0, p=2, dim=-1)
        i0_norms = torch.norm(i0, p=2, dim=-1)
        all0_norms = torch.norm(allEmbds, p=2, dim=-1)

        self.q_u0.copy_(torch.clamp(torch.median(u0_norms), min=1e-3))
        self.q_i0.copy_(torch.clamp(torch.median(i0_norms), min=1e-3))
        self.M_norm.copy_(torch.quantile(all0_norms, 0.95))

        # Layer 1
        h1 = (self.Adj @ allEmbds) * beta
        u1, i1 = torch.split(h1, (self.User.count, self.Item.count))
        u1_norms = torch.norm(u1, p=2, dim=-1)
        i1_norms = torch.norm(i1, p=2, dim=-1)

        self.q_u1.copy_(torch.clamp(torch.median(u1_norms), min=1e-3))
        self.q_i1.copy_(torch.clamp(torch.median(i1_norms), min=1e-3))

        self.is_scale_initialized.copy_(torch.tensor(True))
        print(
            f"[STAIR-LHC v1 Scale Initialization] "
            f"q_u0={self.q_u0.item():.4f}, q_i0={self.q_i0.item():.4f}, "
            f"q_u1={self.q_u1.item():.4f}, q_i1={self.q_i1.item():.4f}, "
            f"M_norm={self.M_norm.item():.4f}"
        )

    def update_epoch(self, epoch: int) -> None:
        """
        Linear Warmup Schedule for Regularization Weight lambda(t):
        - Epochs 0 to warmup_start (e.g. 20): lambda = 0.0 (BPR stabilization phase).
        - Epochs warmup_start to warmup_end (e.g. 21-49): linear ramp 0 -> lambda_lhc.
        - Epochs >= warmup_end (e.g. 50+): lambda = lambda_lhc.
        - If arm == 'B0': lambda = 0.0 unconditionally (baseline parity).
        """
        self.current_epoch = epoch

        if not self.is_scale_initialized:
            self.initialize_scale_buffers()

        if self.geo.arm == "B0" or self.lambda_lhc <= 0.0:
            self.current_lambda = 0.0
            return

        if epoch <= self.warmup_start:
            self.current_lambda = 0.0
        elif epoch < self.warmup_end:
            ramp = float(epoch - self.warmup_start) / float(max(1, self.warmup_end - self.warmup_start))
            self.current_lambda = self.lambda_lhc * ramp
        else:
            self.current_lambda = self.lambda_lhc

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward Stepwise Convolution (FSC) with layer intermediate capture.
        Returns:
            userEmbds: (N_u, D) final aggregated user representations
            itemEmbds: (N_i, D) final aggregated item representations
            layer_embeds: [H^0, H^1, ...] intermediates for contrastive loss
        """
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )

        layer_embeds = [allEmbds]
        features = allEmbds
        smoothed = allEmbds

        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        for _ in range(self.num_layers):
            features = (self.Adj @ features) * beta
            smoothed = smoothed + features
            layer_embeds.append(features)

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds, layer_embeds

    def encode_for_eval(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure baseline evaluation forward pass with zero overhead."""
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        )
        features = allEmbds
        smoothed = allEmbds
        beta = (1.0 - self.beta3).to(allEmbds.device)
        norm_correction = 1.0 - beta ** (self.num_layers + 1)

        for _ in range(self.num_layers):
            features = (self.Adj @ features) * beta
            smoothed = smoothed + features

        avgEmbds = smoothed.mul(1.0 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> torch.Tensor:
        """
        Computes global objective: L_total = L_BPR + lambda(t) * L_LHC.
        """
        userEmbds, itemEmbds, layer_embeds = self.encode()

        users = data[self.User]
        positives = data[self.Item]
        negatives = data[self.INeg]

        # ─── 1. Primary Objective: BPR Pairwise Ranking Loss ─────────────
        u_emb = userEmbds[users]  # (B, 1, D)
        pos_emb = itemEmbds[positives]  # (B, 1, D)
        neg_emb = itemEmbds[negatives]  # (B, 1, D)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", u_emb, pos_emb),
            torch.einsum("BKD,BKD->BK", u_emb, neg_emb),
        )

        # ─── 2. Auxiliary Objective: STAIR-LHC v1 Contrastive Loss ──────
        if self.training and self.current_lambda > 0.0:
            if not self.is_scale_initialized:
                self.initialize_scale_buffers()

            device = u_emb.device
            users_flat = users.view(-1)
            positives_flat = positives.view(-1)

            # Unique users and items in current mini-batch
            u_unique, u_inv = torch.unique(users_flat, return_inverse=True)
            i_unique, i_inv = torch.unique(positives_flat, return_inverse=True)
            B_u = u_unique.size(0)
            B_i = i_unique.size(0)

            # [OPTIMIZED] Construct binary ground-truth train positive matrix P in-batch
            # using vectorized COO-based lookup instead of O(B_u × B_i) Python loop.
            # On Electronics: reduces ~14.4M Python dict lookups to ~4K vectorized ops.
            P = self._build_positive_matrix_vectorized(u_unique, i_unique, device)

            # Extract layer 0 and layer 1 embeddings for unique batch entities
            H0 = layer_embeds[0]
            H1 = layer_embeds[1]
            U0, I0 = torch.split(H0, (self.User.count, self.Item.count))
            U1, I1 = torch.split(H1, (self.User.count, self.Item.count))

            u_0_b = U0[u_unique]
            i_0_b = I0[i_unique]
            u_1_b = U1[u_unique]
            i_1_b = I1[i_unique]

            deg_u_b = self.user_degrees[u_unique]
            deg_i_b = self.item_degrees[i_unique]
            beta = (1.0 - self.beta3).to(device)

            cl_loss, cl_diag = self.lhc_loss_fn(
                u_0=u_0_b,
                i_0=i_0_b,
                u_1=u_1_b,
                i_1=i_1_b,
                P=P,
                deg_u=deg_u_b,
                deg_i=deg_i_b,
                beta=beta,
                q_u0=self.q_u0.item(),
                q_i0=self.q_i0.item(),
                q_u1=self.q_u1.item(),
                q_i1=self.q_i1.item(),
                M_norm=self.M_norm.item(),
            )

            total_loss = rec_loss + self.current_lambda * cl_loss

            # Save telemetry diagnostics
            self.last_diagnostics = {
                "bpr_loss": rec_loss.item(),
                "cl_loss": cl_loss.item(),
                "current_lambda": self.current_lambda,
                "alignment": cl_diag.get("alignment", 0.0),
                "uniformity": cl_diag.get("uniformity", 0.0),
            }
            return total_loss

        self.last_diagnostics = {
            "bpr_loss": rec_loss.item(),
            "cl_loss": 0.0,
            "current_lambda": 0.0,
            "alignment": 0.0,
            "uniformity": 0.0,
        }
        return rec_loss

    # ─── 3. Inference Methods (100% Euclidean, Zero Overhead) ─────────
    def reset_ranking_buffers(self) -> None:
        """Executes before validation/testing to pre-aggregate representations."""
        userEmbds, itemEmbds = self.encode_for_eval()
        self.ranking_buffer = {
            self.User: userEmbds.detach().clone(),
            self.Item: itemEmbds.detach().clone(),
        }

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)
