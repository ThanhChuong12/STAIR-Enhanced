# -*- coding: utf-8 -*-
"""
main_stair_ne_nlgcl_v5_plus.py — STAIR-NE-NLGCL v5+ (v3-Refined) Training Script
================================================================================
Kế thừa trọn vẹn sự tinh gọn tối ưu của v5 (100% Direct Gradient Flow):
1. Bỏ hoàn toàn Projection Head -> InfoNCE tác động trực tiếp vào H^(0) và H^(1).
2. Bỏ hoàn toàn Regularized Diagonal Spectral Projector -> Loại bỏ ma sát tối ưu.
3. Giữ nguyên Sign-Preserving Spectral Perturbation (|noise| >= 0) -> Bảo toàn góc phần tư 64D.
4. Linear HANS (Hardness-Aware Negative Scheduling):
   psi = 1.0 + gamma_h * clamp(cos_sim, min=0.0) với gamma_h = 0.15 (không làm méo tau_eff).
5. Hard-Threshold MFNA (Modality False Negative Attenuation):
   Nếu sim_modal > 0.85 -> mask = 0.0 (loại bỏ hoàn toàn near-duplicates), ngược lại 1.0.
6. Constant Contrastive Pressure:
   lambda_cl = 0.010 (warmup 0 -> 0.010 trong 50 epoch đầu, sau đó cố định 100%).

Usage:
    python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Baby_550_MMRec.yaml
    python main_stair_ne_nlgcl_v5_plus.py --config configs/Amazon2014Sports_550_MMRec.yaml
"""

import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# ── Import freerec framework ──
import freerec
from freerec.data.postprocessing import (
    FieldSourceFilter,
    PostProcessorComposer,
    ToDevice,
)
from freerec.data.tags import USER, ID, POSITIVE

# ── Local optimizers and model ──
from optimizers.adamsevo import AdamSEvo
from optimizers.adamwsevo import AdamWSEvo
from models.stair_ne_nlgcl_v5_plus import STAIR_NE_NLGCL_v5_Plus


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Setup
# ═══════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension (default: 64)")
cfg.add_argument("--num-layers", type=int, default=3, help="GNN propagation layers (default: 3)")
cfg.add_argument("--gamma", type=float, default=0.2, help="Spectral decay exponent for beta3 (default: 0.2)")

# ── STAIR-NE-NLGCL v5+ Hyperparameters ──
cfg.add_argument("--tau", type=float, default=0.20, help="Temperature tau for InfoNCE softmax (default: 0.20)")
cfg.add_argument("--alpha-dir", type=float, default=0.50, help="Direction balance: alpha*L_{u->i} + (1-alpha)*L_{i->u} (default: 0.50)")
cfg.add_argument("--eps", type=float, default=0.08, help="Sign-preserving noise amplitude epsilon (default: 0.08)")
cfg.add_argument("--tau-thresh", type=float, default=0.85, help="Semantic similarity threshold for false negative masking (default: 0.85)")
cfg.add_argument("--lambda-cl", type=float, default=0.010, help="Constant contrastive loss weight (default: 0.010)")
cfg.add_argument("--gamma-h", type=float, default=0.15, help="Linear HANS hardness penalty coefficient (default: 0.15)")
cfg.add_argument("--warmup-epochs", type=int, default=50, help="Warmup epochs for lambda (default: 50)")

cfg.set_defaults(
    description="STAIR-NE-NLGCL-v5-Plus",
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

# BSC Smoother spectral decay beta3
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═══════════════════════════════════════════════════════════════════════════
# STAIR-NE-NLGCL v5+ Model Class
# ═══════════════════════════════════════════════════════════════════════════
class STAIR_NE_NLGCL_v5_Plus_Model(freerec.models.GenRecArch):
    """
    STAIR-NE-NLGCL v5+ Model:
    Integrates STAIR Forward Stepwise Convolution with Clean Direct Contrastive Learning:
    - No Projection Head (100% Direct InfoNCE Gradient Flow)
    - No Diagonal Projector (Zero optimization friction)
    - Sign-Preserving Spectral Perturbation (|noise| >= 0)
    - Linear HANS (psi = 1 + gamma_h * max(0, cos))
    - Hard-Threshold MFNA (mask = I(sim_modal <= 0.85))
    - Constant Lambda (0.010 with 50-epoch linear warmup)
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)
        self.num_layers = cfg.num_layers

        self.User.add_module(
            'embeddings', nn.Embedding(self.User.count, cfg.embedding_dim)
        )
        self.Item.add_module(
            'embeddings', nn.Embedding(self.Item.count, cfg.embedding_dim)
        )

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj(normalization='sym')
        )
        self.register_buffer('beta3', cfg.beta3)

        self.reset_parameters()
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        # Clean STAIR-NE-NLGCL v5+ Module
        self.ne_nlgcl_v5_plus = STAIR_NE_NLGCL_v5_Plus(
            n_users       = self.User.count,
            n_items       = self.Item.count,
            tau           = cfg.tau,
            alpha_dir     = cfg.alpha_dir,
            eps           = cfg.eps,
            tau_thresh    = cfg.tau_thresh,
            lambda_cl     = cfg.lambda_cl,
            gamma_h       = cfg.gamma_h,
            warmup_epochs = cfg.warmup_epochs,
        )
        self.last_cl_loss: Optional[float] = None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1e-4)

    def prepare(self, path: str):
        """Prepare multimodal whitened features (SVD Whitening 64D)."""
        import numpy as np

        def _svd_whiten(features: np.ndarray, target_dim: int = 64) -> torch.Tensor:
            X = features.astype(np.float64)
            X = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            X_white = U[:, :target_dim] * np.sqrt(X.shape[0])
            norms = np.linalg.norm(X_white, axis=1, keepdims=True)
            X_white = X_white / np.maximum(norms, 1e-12)
            return torch.from_numpy(X_white.astype(np.float32))

        features_list = []
        for mfile in cfg.mfiles:
            feat = np.load(os.path.join(path, mfile))
            features_list.append(feat)

        if len(features_list) == 1:
            combined = features_list[0]
        else:
            normed = [f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12) for f in features_list]
            combined = np.concatenate(normed, axis=-1)

        whitened = _svd_whiten(combined, cfg.embedding_dim).to(cfg.device)
        self.register_buffer('item_modals_raw', whitened)

    def marked_params(self):
        """Direct backbone parameters only (No Projector, No ProjHead)."""
        return [
            {'params': self.User.parameters()},
            {'params': self.Item.parameters()},
        ]

    def forward(self, data: Dict) -> torch.Tensor:
        users = data[USER]
        positives = data[POSITIVE]

        # ── 1. Forward Stepwise Convolution (FSC) ──
        items = self.Item.embeddings.weight
        users_emb = self.User.embeddings.weight

        # Layer 0
        h_0 = torch.cat([users_emb, items], dim=0)
        layer_embeds = [h_0]

        # Stepwise convolution layers
        h = h_0
        for _ in range(self.num_layers):
            h = torch.sparse.mm(self.Adj, h)
            # Backward Stepwise Correction (BSC) Smoother
            h = h * (1.0 - self.beta3) + h_0 * self.beta3
            layer_embeds.append(h)

        # Multi-scale average representation for final BPR
        final_embeds = torch.stack(layer_embeds, dim=1).mean(dim=1)
        u_final, i_final = torch.split(final_embeds, [self.User.count, self.Item.count])

        # ── 2. BPR Recommendation Loss ──
        rec_loss = self.criterion(
            u_final[users],
            i_final[positives],
            i_final[data[freerec.data.tags.NEGATIVE]],
        )

        # ── 3. Clean InfoNCE Contrastive Loss (v5+) ──
        if self.training and self.ne_nlgcl_v5_plus.current_lambda > 0.0:
            beta = 1.0 - self.beta3
            i_mod = self.item_modals_raw if hasattr(self, 'item_modals_raw') else None

            cl_loss, raw_cl = self.ne_nlgcl_v5_plus(
                layer_embeds = layer_embeds,
                users        = users,
                positives    = positives,
                beta         = beta,
                item_modals  = i_mod,
            )
            self.last_cl_loss = raw_cl
            return rec_loss + cl_loss

        return rec_loss

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        users_emb = self.User.embeddings.weight
        items_emb = self.Item.embeddings.weight

        h_0 = torch.cat([users_emb, items_emb], dim=0)
        layer_embeds = [h_0]
        h = h_0
        for _ in range(self.num_layers):
            h = torch.sparse.mm(self.Adj, h)
            h = h * (1.0 - self.beta3) + h_0 * self.beta3
            layer_embeds.append(h)

        final_embeds = torch.stack(layer_embeds, dim=1).mean(dim=1)
        u_final, i_final = torch.split(final_embeds, [self.User.count, self.Item.count])
        return (u_final[users] * i_final[items]).sum(dim=-1)

    def recommend_from_full(self) -> torch.Tensor:
        users_emb = self.User.embeddings.weight
        items_emb = self.Item.embeddings.weight

        h_0 = torch.cat([users_emb, items_emb], dim=0)
        layer_embeds = [h_0]
        h = h_0
        for _ in range(self.num_layers):
            h = torch.sparse.mm(self.Adj, h)
            h = h * (1.0 - self.beta3) + h_0 * self.beta3
            layer_embeds.append(h)

        final_embeds = torch.stack(layer_embeds, dim=1).mean(dim=1)
        u_final, i_final = torch.split(final_embeds, [self.User.count, self.Item.count])
        return torch.matmul(u_final, i_final.t())


# ═══════════════════════════════════════════════════════════════════════════
# Coach Class for STAIR-NE-NLGCL v5+
# ═══════════════════════════════════════════════════════════════════════════
class CoachForSTAIR_NE_NLGCL_v5_Plus(freerec.launcher.Coach):
    """
    Coach for STAIR-NE-NLGCL v5+:
    - Optimizer with User and Item embeddings.
    - Tracks per-epoch warmup of lambda_cl.
    - Real-time diagnostic logging.
    """

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.marked_params(), lr=self.cfg.lr,
                momentum=self.cfg.momentum, nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamsevo':
            self.optimizer = AdamSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == 'adamwsevo':
            self.optimizer = AdamWSEvo(
                self.model.marked_params(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"CoachForSTAIR_NE_NLGCL_v5_Plus does not support {self.cfg.optimizer} optimizer"
            )

    def train_per_epoch(self, epoch: int):
        self.model.train()
        # Cập nhật epoch cho hàm điều phối warmup lambda
        self.model.ne_nlgcl_v5_plus.update_epoch(epoch)
        total_cl_loss = 0.0
        cl_batches = 0

        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(), n=len(data[self.User]),
                reduction="mean", mode='train', pool=['LOSS'],
            )

            if hasattr(self.model, 'last_cl_loss') and self.model.last_cl_loss is not None:
                total_cl_loss += self.model.last_cl_loss
                cl_batches += 1

        if cl_batches > 0:
            avg_cl_loss = total_cl_loss / float(cl_batches)
            gamma_h, curr_lambda = self.model.ne_nlgcl_v5_plus.get_current_params()
            if epoch % 10 == 0 or epoch == 1 or epoch == self.cfg.epochs:
                print(
                    f"  [v5+ Epoch {epoch:3d}] gamma_h: {gamma_h:.4f} | "
                    f"lambda: {curr_lambda:.5f} | avg_cl_loss: {avg_cl_loss:.6f}"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Main Execution Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def main():
    dataset = freerec.data.datasets.RecDataSet(
        cfg.root, cfg.dataset,
        tasktag=cfg.tasktag,
    )

    # Training pipeline with FieldSourceFilter and ToDevice
    trainpipe = freerec.data.postprocessing.SourceFieldFilter(
        dataset.train().to_bigraph().to_row_indices(
            dataset.field(USER),
            dataset.field(POSITIVE),
        )
    ).wrap(
        freerec.data.postprocessing.UniformSampler(
            dataset.train().to_bigraph().to_negative_sampler(
                dataset.field(USER),
                dataset.field(POSITIVE),
            )
        )
    ).batch(cfg.batch_size).wrap(ToDevice(cfg.device))

    # Validation pipeline
    validpipe = freerec.data.postprocessing.SourceFieldFilter(
        dataset.valid().to_bigraph().to_row_indices(
            dataset.field(USER),
            dataset.field(POSITIVE),
        )
    ).batch(cfg.batch_size).wrap(ToDevice(cfg.device))

    # Test pipeline
    testpipe = freerec.data.postprocessing.SourceFieldFilter(
        dataset.test().to_bigraph().to_row_indices(
            dataset.field(USER),
            dataset.field(POSITIVE),
        )
    ).batch(cfg.batch_size).wrap(ToDevice(cfg.device))

    model = STAIR_NE_NLGCL_v5_Plus_Model(dataset)

    coach = CoachForSTAIR_NE_NLGCL_v5_Plus(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == '__main__':
    main()
