# -*- coding: utf-8 -*-
"""
main_stair5_v1.py — STAIR-LHC v1 Training & Evaluation Script
=============================================================
Senior AI Research Engineer Implementation:
Training launcher for STAIR with Lorentz Hidden Contrastive Regularization (STAIR-LHC v1 Revised).

Usage Examples:
    # 1. Main Proposed Architecture (H0) on Amazon Sports:
    python main_stair5_v1.py --config configs/Amazon2014Sports_STAIR5_v1.yaml

    # 2. Baseline Parity Control (B0, lambda=0.0):
    python main_stair5_v1.py --config configs/Amazon2014Sports_STAIR5_v1.yaml --lhc-arm B0

    # 3. Euclidean Control (E0, kappa=0.0):
    python main_stair5_v1.py --config configs/Amazon2014Sports_STAIR5_v1.yaml --lhc-arm E0

    # 4. Constant-Radius Control (HC, ||v||=1.0):
    python main_stair5_v1.py --config configs/Amazon2014Sports_STAIR5_v1.yaml --lhc-arm HC
"""

import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# ── Compatibility Patch for torchdata in PyTorch 2.x / Python 3.12 ───────────
try:
    import torchdata
    import torchdata.datapipes as dp
except Exception:
    dp = None

if dp is None or "torchdata.datapipes" not in sys.modules:
    if "torchdata" not in sys.modules:
        td = types.ModuleType("torchdata")
        sys.modules["torchdata"] = td
    else:
        td = sys.modules["torchdata"]

    dp = types.ModuleType("torchdata.datapipes")
    td.datapipes = dp
    sys.modules["torchdata.datapipes"] = dp

if not hasattr(dp, "iter"):
    iter_mod = types.ModuleType("torchdata.datapipes.iter")
    dp.iter = iter_mod
    sys.modules["torchdata.datapipes.iter"] = iter_mod
if not hasattr(dp.iter, "IterDataPipe"):
    class IterDataPipe(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter([])
    dp.iter.IterDataPipe = IterDataPipe

if not hasattr(dp, "map"):
    map_mod = types.ModuleType("torchdata.datapipes.map")
    dp.map = map_mod
    sys.modules["torchdata.datapipes.map"] = map_mod
if not hasattr(dp.map, "MapDataPipe"):
    class MapDataPipe(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            raise NotImplementedError
        def __len__(self):
            return 0
    dp.map.MapDataPipe = MapDataPipe

if not hasattr(dp, "functional_datapipe"):
    def functional_datapipe(name, enable_df_datapipes_support=False):
        def decorator(cls):
            def method(self, *args, **kwargs):
                return cls(self, *args, **kwargs)
            if hasattr(dp, "iter") and hasattr(dp.iter, "IterDataPipe"):
                setattr(dp.iter.IterDataPipe, name, method)
            if hasattr(dp, "map") and hasattr(dp.map, "MapDataPipe"):
                setattr(dp.map.MapDataPipe, name, method)
            try:
                if hasattr(torch.utils.data, "IterDataPipe"):
                    setattr(torch.utils.data.IterDataPipe, name, method)
                if hasattr(torch.utils.data, "MapDataPipe"):
                    setattr(torch.utils.data.MapDataPipe, name, method)
            except Exception:
                pass
            return cls
        return decorator
    dp.functional_datapipe = functional_datapipe

import freerec

from optimizers.Adam import AdamSEvo
from optimizers.AdamW import AdamWSEvo
from optimizers.utils import Smoother
from models.stair5_v1 import STAIR5_v1_Model

freerec.declare(version="0.8.5")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration Setup: STAIR Baseline + STAIR-LHC v1 Parameters
# ═══════════════════════════════════════════════════════════════════════════
cfg = freerec.parser.Parser()

# ── STAIR Baseline Parameters ──
cfg.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension D (default: 64)")
cfg.add_argument("--num-layers", type=int, default=3, help="Number of layers for FSC/BSC (default: 3)")
cfg.add_argument("--mfiles", type=str, default="textual_modality.pkl,visual_modality.pkl",
                 help="Comma-separated modality feature files")
cfg.add_argument("--num-neighbors", type=str, default="5-1", help="kNN counts per modality (default: '5-1')")
cfg.add_argument("--gamma", type=float, default=0.2, help="Spectral decay exponent for beta3 (default: 0.2)")

# ── STAIR-LHC v1 Parameters ──
cfg.add_argument("--lhc-arm", type=str, default="H0",
                 choices=["H0", "B0", "E0", "HC", "H0W5", "H0-REWEIGHT", "H0-NOSELF"],
                 help="Ablation arm: H0 (proposed), B0 (baseline), E0 (Euclidean), HC (const-radius), H0w5, etc.")
cfg.add_argument("--lambda-lhc", type=float, default=3e-4,
                 help="Maximum regularization weight lambda_max (default: 3e-4)")
cfg.add_argument("--lhc-tau", type=float, default=0.3,
                 help="InfoNCE temperature tau (default: 0.3)")
cfg.add_argument("--lhc-kappa", type=float, default=1.0,
                 help="Lorentz curvature kappa (default: 1.0)")
cfg.add_argument("--radius-cap", type=float, default=2.0,
                 help="Maximum L2 radius R for hyperbolic projection (default: 2.0)")
cfg.add_argument("--w-hybrid", type=float, default=0.0,
                 help="Weight for hybrid distance kernel (default: 0.0)")
cfg.add_argument("--eps-taylor", type=float, default=1e-4,
                 help="Taylor expansion threshold for arcosh^2 (default: 1e-4)")
cfg.add_argument("--eps-floor", type=float, default=1e-2,
                 help="Degree-1 norm protection floor (default: 1e-2)")
cfg.add_argument("--eps-beta", type=float, default=0.05,
                 help="Spectral re-weighting clamp floor (default: 0.05)")
cfg.add_argument("--warmup-start", type=int, default=20,
                 help="Epoch to begin ramping lambda (default: 20)")
cfg.add_argument("--warmup-end", type=int, default=50,
                 help="Epoch where lambda reaches lambda_max (default: 50)")

# ── Diagnostics & Safety Controls ──
cfg.add_argument("--grad-diag-every", type=int, default=10,
                 help="Interval in epochs to compute BPR vs CL gradient cosine on Item embeddings (0=disabled)")
cfg.add_argument("--grad-cosine-thresh", type=float, default=-0.1,
                 help="Threshold below which warning is issued for gradient conflict")

cfg.set_defaults(
    description="STAIR-LHC-v1",
    root="data",
    dataset="Amazon2014Sports_550_MMRec",
    epochs=500,
    batch_size=1024,
    optimizer="adamwsevo",
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
cfg.compile()

cfg.mfiles = cfg.mfiles.split(",")
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split("-")))

# BSC Smoother spectral decay beta3: beta3 = 0.1 + 0.9 * (j/D)^gamma
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ═══════════════════════════════════════════════════════════════════════════
# Coach Implementation with Diagnostics & Telemetry
# ═══════════════════════════════════════════════════════════════════════════
class CoachForSTAIR5_v1(freerec.launcher.Coach):

    def set_optimizer(self):
        marked = self.model.marked_params()
        opt_name = self.cfg.optimizer.lower()

        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                marked, lr=self.cfg.lr,
                momentum=self.cfg.momentum, nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                marked, lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                marked, lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == "adamsevo":
            self.optimizer = AdamSEvo(
                marked, lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        elif opt_name == "adamwsevo":
            self.optimizer = AdamWSEvo(
                marked, lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.cfg.optimizer}")

    def compute_gradient_cosine_diagnostic(self, data: Dict) -> Optional[float]:
        """
        Computes cosine similarity between BPR loss gradient and LHC loss gradient
        on Item embeddings: rho_grad = <g_BPR, g_LHC> / (||g_BPR|| * ||g_LHC||).
        """
        if self.model.current_lambda <= 0.0:
            return None

        self.model.zero_grad()
        userEmbds, itemEmbds, layer_embeds = self.model.encode()
        users = data[self.model.User]
        positives = data[self.model.Item]
        negatives = data[self.model.INeg]

        # 1. BPR loss gradient
        u_emb = userEmbds[users]
        pos_emb = itemEmbds[positives]
        neg_emb = itemEmbds[negatives]
        rec_loss = self.model.criterion(
            torch.einsum("BKD,BKD->BK", u_emb, pos_emb),
            torch.einsum("BKD,BKD->BK", u_emb, neg_emb),
        )
        rec_loss.backward(retain_graph=True)
        g_bpr = self.model.Item.embeddings.weight.grad
        if g_bpr is None:
            return None
        g_bpr_flat = g_bpr.clone().flatten()

        # 2. LHC loss gradient
        self.model.zero_grad()
        users_flat = users.view(-1)
        positives_flat = positives.view(-1)
        u_unique = torch.unique(users_flat)
        i_unique = torch.unique(positives_flat)
        B_u = u_unique.size(0)
        B_i = i_unique.size(0)

        device = u_emb.device
        P = torch.zeros((B_u, B_i), dtype=torch.float32, device=device)
        u_list = u_unique.cpu().tolist()
        i_list = i_unique.cpu().tolist()
        for a_idx, u_id in enumerate(u_list):
            u_pos_set = self.model.train_u2i_set.get(u_id, set())
            for b_idx, i_id in enumerate(i_list):
                if i_id in u_pos_set:
                    P[a_idx, b_idx] = 1.0

        H0 = layer_embeds[0]
        H1 = layer_embeds[1]
        U0, I0 = torch.split(H0, (self.model.User.count, self.model.Item.count))
        U1, I1 = torch.split(H1, (self.model.User.count, self.model.Item.count))

        deg_u_b = self.model.user_degrees[u_unique]
        deg_i_b = self.model.item_degrees[i_unique]
        beta = (1.0 - self.model.beta3).to(device)

        cl_loss, _ = self.model.lhc_loss_fn(
            u_0=U0[u_unique],
            i_0=I0[i_unique],
            u_1=U1[u_unique],
            i_1=I1[i_unique],
            P=P,
            deg_u=deg_u_b,
            deg_i=deg_i_b,
            beta=beta,
            q_u0=self.model.q_u0.item(),
            q_i0=self.model.q_i0.item(),
            q_u1=self.model.q_u1.item(),
            q_i1=self.model.q_i1.item(),
            M_norm=self.model.M_norm.item(),
        )
        cl_loss.backward()
        g_lhc = self.model.Item.embeddings.weight.grad
        if g_lhc is None:
            return None
        g_lhc_flat = g_lhc.clone().flatten()

        self.model.zero_grad()

        norm_bpr = torch.norm(g_bpr_flat)
        norm_lhc = torch.norm(g_lhc_flat)
        if norm_bpr < 1e-8 or norm_lhc < 1e-8:
            return 0.0

        cos_sim = torch.dot(g_bpr_flat, g_lhc_flat) / (norm_bpr * norm_lhc)
        return cos_sim.item()

    def train_per_epoch(self, epoch: int):
        self.model.update_epoch(epoch)
        grad_diag_every = getattr(self.cfg, "grad_diag_every", 10)

        epoch_bpr_sum = 0.0
        epoch_cl_sum = 0.0
        epoch_align_sum = 0.0
        epoch_unif_sum = 0.0
        batch_count = 0

        for batch_idx, data in enumerate(self.dataloader):
            data = self.dict_to_device(data)

            # Gradient Cosine Diagnostics (Baseline-Calibrated) on first batch of period
            if grad_diag_every > 0 and (epoch % grad_diag_every == 0) and batch_idx == 0:
                rho = self.compute_gradient_cosine_diagnostic(data)
                if rho is not None:
                    thresh = getattr(self.cfg, "grad_cosine_thresh", -0.1)
                    status = "OK" if rho >= thresh else "ALERT (Opposing Gradients!)"
                    print(
                        f"[Gradient Diagnostics @ Epoch {epoch}] "
                        f"rho_grad(BPR, LHC) = {rho:+.4f} | Status: {status}"
                    )

            loss = self.model(data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record telemetry
            diag = getattr(self.model, "last_diagnostics", {})
            epoch_bpr_sum += diag.get("bpr_loss", 0.0)
            epoch_cl_sum += diag.get("cl_loss", 0.0)
            epoch_align_sum += diag.get("alignment", 0.0)
            epoch_unif_sum += diag.get("uniformity", 0.0)
            batch_count += 1

            self.monitor(
                loss.item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )

        if batch_count > 0:
            avg_bpr = epoch_bpr_sum / batch_count
            avg_cl = epoch_cl_sum / batch_count
            avg_align = epoch_align_sum / batch_count
            avg_unif = epoch_unif_sum / batch_count
            curr_lam = self.model.current_lambda
            print(
                f"[Epoch {epoch:03d} Telemetry] "
                f"BPR: {avg_bpr:.4f} | LHC: {avg_cl:.4f} (lambda={curr_lam:.6f}) | "
                f"Align: {avg_align:.4f} | Unif: {avg_unif:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Auto-Bridging and Main Execution
# ═══════════════════════════════════════════════════════════════════════════
def main():
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (os.path.isdir(processed_dir) and not os.listdir(processed_dir)):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join("data", cfg.dataset),
        ]
        for cand in candidates:
            if (
                os.path.exists(cand)
                and os.path.isdir(cand)
                and os.path.abspath(cand) != os.path.abspath(processed_dir)
                and len(os.listdir(cand)) > 0
            ):
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                    print(f"[DataSet] Auto-bridged symlink: {cand} -> {processed_dir}")
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                    print(f"[DataSet] Auto-bridged copied: {cand} -> {processed_dir}")
                break

    tasktag = getattr(cfg, "tasktag", None) or getattr(freerec.data.tags, "MATCHING", None)
    if hasattr(freerec.data.datasets, "RecDataSet"):
        freerec.data.datasets.RecDataSet.TASK = tasktag
    if hasattr(freerec.data.datasets, "base") and hasattr(freerec.data.datasets.base, "BaseSet"):
        freerec.data.datasets.base.BaseSet.TASK = tasktag

    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            try:
                from freerec.data.datasets.base import MatchingRecDataSet
                dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
            except Exception:
                dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
    else:
        try:
            from freerec.data.datasets.base import MatchingRecDataSet
            dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)

    if not hasattr(dataset, "TASK") or dataset.TASK is None:
        dataset.TASK = tasktag

    print(f"============================================================")
    print(f"           STAIR-LHC v1 (REVISED) EXPERIMENTAL RUN          ")
    print(f"============================================================")
    print(f"Dataset      : {cfg.dataset}")
    print(f"Arm          : {cfg.lhc_arm}")
    print(f"Curvature    : kappa={cfg.lhc_kappa}")
    print(f"Lambda Max   : {cfg.lambda_lhc}")
    print(f"Tau          : {cfg.lhc_tau}")
    print(f"Radius Cap   : R={cfg.radius_cap}")
    print(f"Warmup Range : Epoch {cfg.warmup_start} -> Epoch {cfg.warmup_end}")
    print(f"============================================================")

    model = STAIR5_v1_Model(dataset, cfg)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR5_v1(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
