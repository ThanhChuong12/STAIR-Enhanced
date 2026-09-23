"""STAIR4-v2.1: Baseline-preserving Complex Cross-layer Regularization (BCCR).

FreeRec GenRecArch adapter implementing STAIR4-v2.1 (§6, §8, §11 of STAIR4_v2_1_Report.md).

Core Invariants & Contracts:
----------------------------
1. Gate 0 Baseline Equivalence:
   - FSC correctly uses self.beta (b_j):
     Z = smoothed.mul(self.beta).div(1 - (1 - self.beta)**(L + 1))
   - With identical static MI and kNN graph inputs, auxiliary_kernel='none'
     (or lambda_cl=0), and smoother_mode='baseline', forward, backward,
     optimizer step and scoring match STAIR baseline (main.py).

2. BCCR Auxiliary Regularization:
   - Unscaled first-hop X1 = Adj @ X0.
   - Key branch has stop-gradient on target view before rotation.
   - Primary rotation is 'identity' (zero parameters); optional 'learned_givens'.
   - Hybrid similarity s_η(q, k) = (1 - η) Re(h) + η |h|^2 with η=0.25, T=0.2.
   - Train-only multi-positive supervised cross-entropy.

3. Smoother:
   - Original STAIR BSC Neumann smoother by default (xi=0).
   - Optional residual spectral correction T_ξ(Δ) = (I - ξ H^2) P(Δ).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import models.freerec_compat
except Exception:
    pass
import freerec
import torch
import torch.nn as nn
from torch import Tensor

from models.stair4_v2_utils import (
    TrainPositiveIndex,
    baseline_whitening,
    build_train_positive_index,
    exact_cosine_knn,
    make_batch_candidates,
    transpose_train_positive_index,
    validate_static_bundle,
)
from models.stair4_heads import CrossLayerContrastiveHead
from optimizers.mhd_smoother import MHDSmoother
from optimizers.stair4_v2_smoother import STAIR4V21Smoother


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class STAIR4V2Options:
    # --- Auxiliary kernel ---
    auxiliary_kernel: str = "hybrid"           # 'none'|'hybrid'|'signed'|'fidelity'|'cosine'
    rotation_mode: str = "identity"            # 'identity'|'learned_givens'
    smoother_mode: str = "baseline"            # 'baseline'|'residual_spectral'

    # --- Kernel parameters (§6.2, §6.4) ---
    eta: float = 0.25                          # hybrid kernel mixing: (1-η) Re(h) + η |h|^2
    kappa: float = 0.5                         # phase scaling factor: exp(i * kappa * sqrt(d) * r(x))
    contrastive_temperature: float = 0.2       # τ_c

    # --- Loss schedule (§6.6) ---
    pocl_weight_target: float = 0.0001        # λ_max (pilot default: 1e-4)
    warmup_epochs: int = 10                    # W: epochs before any auxiliary loss
    ramp_epochs: int = 20                      # R: epochs to linearly reach λ_max

    # --- Residual spectral smoother (§7) ---
    xi: float = 0.0                            # ξ: residual spectral correction factor (0.0 = exact BSC)

    # --- Optimizer groups (§8) ---
    aux_lr_ratio: float = 0.1                  # Givens omega lr = 0.1 * baseline lr
    aux_weight_decay: float = 0.0

    # --- Diagnostics ---
    diagnostic_interval_steps: int = 100
    query_chunk_size: int = 256

    # --- Modality / graph ---
    modality_weights: tuple = (5 / 6, 1 / 6)
    knn_block_size: int = 256

    def __post_init__(self):
        valid_kernels = ("none", "hybrid", "signed", "fidelity", "cosine")
        if self.auxiliary_kernel not in valid_kernels:
            raise ValueError(f"auxiliary_kernel must be in {valid_kernels}, got {self.auxiliary_kernel!r}")
        if self.rotation_mode not in ("identity", "learned_givens"):
            raise ValueError(f"rotation_mode must be identity/learned_givens, got {self.rotation_mode!r}")
        if self.smoother_mode not in ("baseline", "residual_spectral"):
            raise ValueError(f"smoother_mode must be baseline/residual_spectral, got {self.smoother_mode!r}")
        if not math.isfinite(self.eta) or not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be finite and in [0, 1]")
        if not math.isfinite(self.kappa) or self.kappa < 0.0:
            raise ValueError("kappa must be finite and non-negative")
        if self.pocl_weight_target < 0:
            raise ValueError("pocl_weight_target must be non-negative")
        if self.contrastive_temperature <= 0 or not math.isfinite(self.contrastive_temperature):
            raise ValueError("contrastive_temperature must be positive and finite")
        if self.warmup_epochs < 0 or self.ramp_epochs <= 0:
            raise ValueError("warmup_epochs >= 0 and ramp_epochs > 0 required")
        if not math.isfinite(self.xi) or not (0.0 <= self.xi <= 0.1):
            raise ValueError("xi must be in [0, 0.1]")
        if (
            isinstance(self.query_chunk_size, bool)
            or not isinstance(self.query_chunk_size, int)
            or self.query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        if len(self.modality_weights) != 2 or min(self.modality_weights) < 0:
            raise ValueError("two non-negative modality_weights required")
        if not math.isclose(sum(self.modality_weights), 1.0, abs_tol=1e-7):
            raise ValueError("modality_weights must sum to 1")

    @classmethod
    def from_config(cls, cfg):
        known = set(cls.__dataclass_fields__)
        args = {}
        for key in known:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                if key == "modality_weights" and isinstance(val, str):
                    val = tuple(map(float, val.split(",")))
                args[key] = val
        return cls(**args)


# Backward-compatible alias
STAIR4V21Options = STAIR4V2Options


# ---------------------------------------------------------------------------
# Helper: tensor hash
# ---------------------------------------------------------------------------

def _tensor_hash(tensor: Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(t.shape), t.dtype)).encode())
    digest.update(t.numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class STAIR4V2(freerec.models.GenRecArch):
    """STAIR4-v2.1 BCCR FreeRec model adapter."""

    SCHEMA_VERSION = 2

    def __init__(self, dataset, cfg):
        super().__init__(dataset)
        self.cfg = cfg
        self.options = STAIR4V2Options.from_config(cfg)
        self.num_layers = int(cfg.num_layers)
        self.embedding_dim = int(cfg.embedding_dim)

        if self.num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {self.num_layers}")
        if self.embedding_dim < 2 or self.embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even and >= 2, got {self.embedding_dim}")
        if cfg.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {cfg.gamma}")

        self.current_epoch: int = 0
        self._global_step: int = 0

        # --- Baseline embedding tables ---
        self.User.add_module("embeddings", nn.Embedding(self.User.count, self.embedding_dim))
        self.Item.add_module("embeddings", nn.Embedding(self.Item.count, self.embedding_dim))

        # Baseline FSC coefficient: beta_j = 0.1 + 0.9 * (j/d)^gamma
        self.register_buffer(
            "beta",
            0.1 + 0.9 * (torch.arange(self.embedding_dim) / self.embedding_dim).pow(cfg.gamma),
        )

        # Baseline full-graph normalized adjacency (U+I × U+I)
        self.register_buffer("Adj", dataset.train().to_normalized_adj(normalization="sym"))

        # Buffers populated in prepare()
        self.data_hashes: Dict[str, str] = {}
        self._train_index: Optional[TrainPositiveIndex] = None
        self._reverse_train_index: Optional[TrainPositiveIndex] = None

        # --- Baseline parameter initialization ---
        self._reset_baseline_parameters()

        # --- Static graph preparation ---
        self.prepare(dataset.path)

        # Baseline BPR loss
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # --- Auxiliary head (isolated RNG) ---
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self._init_auxiliary()

        # --- Baseline smoother ---
        self.smoother = MHDSmoother(self._apply_bsc_operator, self.beta, self.num_layers)

        # --- v2.1 Direction smoother (exact BSC fast path when xi=0) ---
        self.bsf_smoother = STAIR4V21Smoother(
            self.smoother,
            lambda: self.mAdj,
            default_xi=self.options.xi,
        )

        self.last_diagnostics: Dict = {}

    def _reset_baseline_parameters(self) -> None:
        """Match reset_parameters() in main.py exactly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.0e-4)

    def _init_auxiliary(self) -> None:
        """Create BCCR head inside forked RNG; no trainable parameters when identity."""
        opts = self.options
        if opts.auxiliary_kernel == "none":
            self.head: Optional[CrossLayerContrastiveHead] = None
        else:
            self.head = CrossLayerContrastiveHead(
                d=self.embedding_dim,
                kernel=opts.auxiliary_kernel,
                rotation=opts.rotation_mode,
                eta=opts.eta,
                kappa=opts.kappa,
                temperature=opts.contrastive_temperature,
                query_chunk_size=opts.query_chunk_size,
            )

    @torch.no_grad()
    def prepare(self, path) -> None:
        """Build static train-only buffers using baseline MI and kNN logic."""
        opts = self.options
        mfiles = (
            self.cfg.mfiles.split(",")
            if isinstance(self.cfg.mfiles, str)
            else list(self.cfg.mfiles)
        )
        ks = (
            list(map(int, self.cfg.num_neighbors.split("-")))
            if isinstance(self.cfg.num_neighbors, str)
            else list(self.cfg.num_neighbors)
        )
        if len(mfiles) != 2 or len(ks) != 2 or min(ks) < 1:
            raise ValueError("Requires exactly two modalities and positive neighbor counts")

        mfeats = []
        for mfile in mfiles:
            candidates = [
                Path(path) / mfile,
                Path(self.cfg.root) / self.cfg.dataset / mfile,
                Path("/kaggle/data") / self.cfg.dataset / mfile,
                Path("/kaggle/working/STAIR/data") / self.cfg.dataset / mfile,
                Path("/kaggle/working/STAIR-Enhanced/data") / self.cfg.dataset / mfile,
                Path("data") / self.cfg.dataset / mfile,
            ]
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"missing modality file {mfile!r} for dataset {self.cfg.dataset!r}"
                )
            raw = torch.as_tensor(
                freerec.utils.import_pickle(str(resolved)), dtype=torch.float32
            )
            if raw.shape[0] != self.Item.count:
                raise ValueError(
                    f"feature rows ({raw.shape[0]}) must match item count ({self.Item.count})"
                )
            mfeats.append(raw)

        edge_index = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        canonical = torch.unique(edge_index.detach().cpu().T, dim=0).T.contiguous()
        self.data_hashes["train"] = _tensor_hash(canonical)

        directed_edges = []
        for m, (features, k) in enumerate(zip(mfeats, ks)):
            neighbors = exact_cosine_knn(features, k, opts.knn_block_size)
            self.data_hashes[f"features_{m}"] = _tensor_hash(features)
            self.data_hashes[f"neighbors_{m}"] = _tensor_hash(neighbors)
            centers = torch.arange(self.Item.count)[:, None].expand_as(neighbors)
            directed_edges.append(torch.stack((centers.reshape(-1), neighbors.reshape(-1))))

        indices = torch.cat(directed_edges, dim=1)
        weights = torch.ones_like(indices[0], dtype=torch.float)
        indices, weights = freerec.graph.coalesce(indices, weights, reduce="sum")
        indices, weights = freerec.graph.to_undirected(indices, weights, reduce="max")
        indices, weights = freerec.graph.to_normalized(indices, weights, normalization="sym")
        self.register_buffer(
            "mAdj",
            torch.sparse_coo_tensor(
                indices, weights, size=(self.Item.count, self.Item.count)
            ).to_sparse_csr(),
        )

        mi = sum(
            baseline_whitening(f, self.embedding_dim) * k
            for f, k in zip(mfeats, ks)
        ) / sum(ks)
        self.Item.embeddings.weight.copy_(mi)

        r_indices, r_weights = freerec.graph.to_normalized(edge_index, normalization="left")
        R = torch.sparse_coo_tensor(
            r_indices, r_weights,
            size=(self.User.count, self.Item.count),
        ).to_sparse_csr()
        self.User.embeddings.weight.copy_(R @ mi)

        self._train_index = build_train_positive_index(
            canonical, self.User.count, self.Item.count
        )
        self._reverse_train_index = transpose_train_positive_index(self._train_index)

        validate_static_bundle(
            self.Adj,
            self.mAdj,
            self.User.count,
            self.Item.count,
            self.embedding_dim,
        )

        if any(v.grad_fn is not None or v.requires_grad for v in self.buffers()):
            raise RuntimeError("prepare() created a non-static buffer with grad_fn")

    def _apply_bsc_operator(self, features: Tensor, snapshot) -> Tensor:
        return self.mAdj @ features

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def effective_coefficients(self) -> Tuple[float, float, str]:
        """Return (lambda_cl, xi, active_smoother_mode) based on ramp schedule."""
        elapsed = self.current_epoch - self.options.warmup_epochs
        ramp = min(1.0, max(0.0, elapsed / max(1, self.options.ramp_epochs)))
        lambda_cl = ramp * self.options.pocl_weight_target
        xi = ramp * self.options.xi

        if self.options.auxiliary_kernel == "none":
            lambda_cl = 0.0

        if self.options.smoother_mode == "baseline" or xi == 0.0:
            active_smoother = "baseline"
            xi = 0.0
        else:
            active_smoother = self.options.smoother_mode

        return lambda_cl, xi, active_smoother

    def parameter_groups(self) -> List[dict]:
        """Return optimizer parameter groups: users, items, and aux (if any)."""
        user_params = list(self.User.parameters())
        item_params = list(self.Item.parameters())

        user_ids = {id(p) for p in user_params}
        item_ids = {id(p) for p in item_params}
        aux_params = [
            p for p in self.parameters()
            if id(p) not in user_ids and id(p) not in item_ids
        ]

        seen = [id(p) for g in [user_params, item_params, aux_params] for p in g]
        if len(seen) != len(set(seen)) or set(seen) != {id(p) for p in self.parameters()}:
            raise RuntimeError("parameter_groups: parameter coverage check failed")

        groups = [
            {
                "params": user_params,
                "smoother": None,
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
                "role": "users",
            },
            {
                "params": item_params,
                "smoother": self.bsf_smoother,
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
                "role": "items",
            },
        ]
        if aux_params:
            groups.append(
                {
                    "params": aux_params,
                    "smoother": None,
                    "lr": self.options.aux_lr_ratio * self.cfg.lr,
                    "weight_decay": self.options.aux_weight_decay,
                    "role": "aux",
                }
            )
        return groups

    # ------------------------------------------------------------------
    # Forward: corrected baseline FSC + clean X0, X1 views
    # ------------------------------------------------------------------

    def _encode_baseline_fsc(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Baseline FSC forward with corrected normalization (§2 of STAIR4_v2_1_Report.md).

        b_j = self.beta
        a_j = 1 - self.beta = beta_complement
        F^{(0)} = E
        F^{(l+1)} = Adj @ F^{(l)} * a_j
        Z = smoothed * b_j / (1 - a_j^{L+1})   <-- CORRECTED (fixes 81x distortion)

        Returns
        -------
        (userZ, itemZ, X0_all, X1_all)
        """
        E_u = self.User.embeddings.weight   # [U, d]
        E_i = self.Item.embeddings.weight   # [I, d]
        X0_all = torch.cat([E_u, E_i], dim=0)   # [U+I, d]

        features = X0_all
        smoothed = X0_all
        beta_complement = 1 - self.beta                  # [d] (a_j)
        norm_correction = 1 - beta_complement ** (self.num_layers + 1)  # [d]

        # The L=0 contract deliberately uses X0 as the auxiliary target view.
        X1_all = None
        for layer in range(self.num_layers):
            hop = self.Adj @ features
            if layer == 0:
                X1_all = hop
            features = hop * beta_complement
            smoothed = smoothed + features

        if X1_all is None:
            X1_all = X0_all

        # Core Bug Fix: Multiply by self.beta (b_j), exactly matching baseline main.py line 207:
        # avgEmbds = smoothed.mul(1 - beta).div(norm_correction) where (1 - beta) == cfg.beta3 == self.beta!
        Z = smoothed.mul(self.beta).div(norm_correction)   # [U+I, d]

        userZ, itemZ = torch.split(Z, (self.User.count, self.Item.count))
        return userZ, itemZ, X0_all, X1_all

    def encode(self) -> Tuple[Tensor, Tensor]:
        """Baseline-compatible encode: returns (userZ, itemZ)."""
        userZ, itemZ, _, _ = self._encode_baseline_fsc()
        return userZ, itemZ

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def fit(self, data) -> Tensor:
        """Compute BPR + optional BCCR auxiliary loss; arm smoother; return total loss."""
        self._global_step += 1
        lambda_cl, xi, active_smoother = self.effective_coefficients()

        userZ, itemZ, X0_all, X1_all = self._encode_baseline_fsc()

        users = data[self.User]          # [B, 1]
        positives = data[self.Item]      # [B, 1]
        negatives = data[self.INeg]      # [B, K]

        # BPR loss
        bpr_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userZ[users], itemZ[positives]),
            torch.einsum("BKD,BKD->BK", userZ[users], itemZ[negatives]),
        )

        diag = {
            "bpr": bpr_loss.detach().item(),
            "lambda_cl": lambda_cl,
            "xi": xi,
            "smoother_mode": active_smoother,
            "epoch": self.current_epoch,
            "step": self._global_step,
        }

        total_loss = bpr_loss

        # BCCR auxiliary loss: only executed when lambda_cl > 0
        if (
            lambda_cl > 0
            and self.head is not None
            and self._train_index is not None
            and self._reverse_train_index is not None
        ):
            cands = make_batch_candidates(users, positives)
            cl_loss, cl_diag = self.head(
                X0_all,
                X1_all,
                cands,
                self._train_index,
                self._reverse_train_index,
                n_users=self.User.count,
            )
            total_loss = total_loss + lambda_cl * cl_loss

            diag["cl_raw"] = cl_diag.cl_raw
            diag["cl_weighted"] = lambda_cl * cl_diag.cl_raw
            diag["pos_count_ui"] = cl_diag.pos_count_ui
            diag["pos_count_iu"] = cl_diag.pos_count_iu
            diag["sim_q25"] = cl_diag.sim_q25
            diag["sim_q75"] = cl_diag.sim_q75
            diag["entropy_ui"] = cl_diag.softmax_entropy_ui
            diag["entropy_iu"] = cl_diag.softmax_entropy_iu
            diag["theta_norm"] = cl_diag.theta_norm
            diag["near_zero_view_rate"] = cl_diag.near_zero_view_rate

            # Gradient conflict diagnostics (§6.6): computed only at diagnostic intervals
            if (
                self.options.diagnostic_interval_steps > 0
                and self._global_step % self.options.diagnostic_interval_steps == 0
            ):
                with torch.enable_grad():
                    params = [self.User.embeddings.weight, self.Item.embeddings.weight]
                    g_bpr = torch.autograd.grad(bpr_loss, params, retain_graph=True, allow_unused=True)
                    g_cl = torch.autograd.grad(cl_loss, params, retain_graph=True, allow_unused=True)

                    for role, gb, gc in zip(["u", "i"], g_bpr, g_cl):
                        if gb is not None and gc is not None:
                            gb_flat = gb.detach().reshape(-1)
                            gc_flat = gc.detach().reshape(-1)
                            norm_b = float(gb_flat.norm().item())
                            norm_c = float(gc_flat.norm().item())
                            dot = float(torch.dot(gb_flat, gc_flat).item())
                            cos = dot / (norm_b * norm_c + 1e-8)
                            ratio = (lambda_cl * norm_c) / (norm_b + 1e-8)
                            diag[f"grad_bpr_norm_{role}"] = norm_b
                            diag[f"grad_cl_norm_{role}"] = norm_c
                            diag[f"grad_cos_{role}"] = cos
                            diag[f"weighted_grad_ratio_{role}"] = ratio
                        else:
                            diag[f"grad_bpr_norm_{role}"] = 0.0
                            diag[f"grad_cl_norm_{role}"] = 0.0
                            diag[f"grad_cos_{role}"] = 0.0
                            diag[f"weighted_grad_ratio_{role}"] = 0.0
        else:
            diag["cl_raw"] = 0.0
            diag["cl_weighted"] = 0.0
            diag["near_zero_view_rate"] = 0.0

        # Arm smoother before backward
        self.bsf_smoother.arm_step(
            mode=active_smoother,
            xi=xi,
        )

        self.last_diagnostics = diag
        return total_loss

    def update_post_backward_diagnostics(self) -> None:
        """Record gradients that are only defined after ``loss.backward()`` (§6.6)."""
        if self.head is None or not self.head.has_trainable_parameters:
            self.last_diagnostics["theta_grad_norm"] = 0.0
            self.last_diagnostics["theta_norm"] = 0.0
            return
        omega = self.head.rotation.omega
        grad = omega.grad
        self.last_diagnostics["theta_grad_norm"] = (
            float(grad.detach().norm().item()) if grad is not None else 0.0
        )
        self.last_diagnostics["theta_norm"] = (
            float(self.head.rotation.theta.detach().norm().item())
        )
        self._theta_before_step = self.head.rotation.theta.detach().clone()

    def update_post_step_diagnostics(self) -> None:
        """Record theta parameter step updates after ``optimizer.step()`` (§6.6)."""
        if self.head is None or not self.head.has_trainable_parameters:
            self.last_diagnostics["theta_update_norm"] = 0.0
            return
        if hasattr(self, "_theta_before_step"):
            delta = (self.head.rotation.theta.detach() - self._theta_before_step).norm().item()
            self.last_diagnostics["theta_update_norm"] = float(delta)
        else:
            self.last_diagnostics["theta_update_norm"] = 0.0

    # ------------------------------------------------------------------
    # Ranking / evaluation (baseline scorer)
    # ------------------------------------------------------------------

    def reset_ranking_buffers(self) -> None:
        userZ, itemZ = self.encode()
        self.ranking_buffer = {
            self.User: userZ.detach().clone(),
            self.Item: itemZ.detach().clone(),
        }

    def recommend_from_full(self, data) -> Tensor:
        userZ = self.ranking_buffer[self.User][data[self.User]]
        itemZ = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userZ, itemZ)

    def recommend_from_pool(self, data) -> Tensor:
        userZ = self.ranking_buffer[self.User][data[self.User]]
        itemZ = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", userZ, itemZ)

    def get_extra_state(self) -> dict:
        return {
            "schema": self.SCHEMA_VERSION,
            "epoch": self.current_epoch,
            "global_step": self._global_step,
            "data_hashes": self.data_hashes,
            "options": asdict(self.options),
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "gamma": float(self.cfg.gamma),
        }

    def set_extra_state(self, state: dict) -> None:
        if state.get("schema") != self.SCHEMA_VERSION:
            raise ValueError(f"checkpoint schema mismatch: {state.get('schema')}")
        self.current_epoch = int(state["epoch"])
        self._global_step = int(state.get("global_step", 0))
        saved_hashes = state.get("data_hashes")
        if saved_hashes is not None and saved_hashes != self.data_hashes:
            raise ValueError(
                "checkpoint data hashes do not match the currently prepared "
                "train split, features, or kNN graph"
            )
        if self.bsf_smoother._armed:
            self.bsf_smoother.clear_step_snapshot()
        if self.smoother._step_armed:
            self.smoother.clear_step_snapshot()

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def marked_params(self) -> List[dict]:
        return self.parameter_groups()


# Aliases
STAIR4V21 = STAIR4V2
STAIR4V2Model = STAIR4V2
STAIR4V21Model = STAIR4V2
