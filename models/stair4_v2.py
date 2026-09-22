"""STAIR4-v2: Bounded Spectral Filtering with Phase-Overlap Contrastive Learning.

FreeRec GenRecArch adapter.  Baseline contract (§3 of STAIR4_v2_Report.md):
  - MI initialization, FSC, BPR scorer and AdamWSEvo are inherited unchanged.
  - BSF callback and POCL head are added as auxiliary components.
  - Inference uses Z_u · Z_i; no phase head at evaluation time.
  - Auxiliary initialization is isolated inside torch.random.fork_rng(devices=[])
    so that it does not consume from the baseline RNG stream.

Ablation selector mapping (§14.6):
  auxiliary_kernel : 'none' | 'cosine' | 'phase_fidelity'
  rotation_mode    : 'identity' | 'learned_givens' | 'frozen_random'
  smoother_mode    : 'baseline' | 'identity_mix' | 'bsf_mix'

Baseline recovery: auxiliary_kernel='none' AND smoother_mode='baseline'
(i.e. ablation A0) must reproduce MI/FSC/BPR/optimizer exactly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
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
    hash_data_manifest,
    make_batch_candidates,
    validate_static_bundle,
)
from models.stair4_heads import CrossLayerContrastiveHead, POCLDiagnostics
from optimizers.mhd_smoother import MHDSmoother
from optimizers.stair4_v2_smoother import BSFDirectionSmoother


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class STAIR4V2Options:
    # --- Auxiliary kernel ---
    auxiliary_kernel: str = "phase_fidelity"   # 'none'|'cosine'|'phase_fidelity'
    rotation_mode: str = "learned_givens"      # 'identity'|'learned_givens'|'frozen_random'
    smoother_mode: str = "baseline"            # 'baseline'|'identity_mix'|'bsf_mix'
    aux_view: str = "raw_first_hop"            # 'raw_first_hop' only in pilot
    kernel_backend: str = "complex"            # 'complex'|'real_imag' (phase only)

    # --- Loss schedule ---
    pocl_weight_target: float = 0.001         # λ*
    contrastive_temperature: float = 0.2       # τ_c
    phase_scale: float = 1.0                   # arctan scale s (fixed in pilot)
    warmup_epochs: int = 10                    # W: epochs before any auxiliary
    ramp_epochs: int = 20                      # R: epochs to reach target λ

    # --- BSF ---
    spectral_time: float = 0.5                 # t ∈ [0,1]
    spectral_mix_target: float = 0.1          # ζ*

    # --- Optimizer groups ---
    aux_lr_ratio: float = 0.1                  # Givens lr = 0.1 * baseline lr
    aux_weight_decay: float = 0.0

    # --- Diagnostics ---
    diagnostic_interval_steps: int = 100

    # --- Modality / graph ---
    modality_weights: tuple = (5 / 6, 1 / 6)
    knn_block_size: int = 256

    def __post_init__(self):
        if self.auxiliary_kernel not in ("none", "cosine", "phase_fidelity"):
            raise ValueError(f"auxiliary_kernel must be none/cosine/phase_fidelity, got {self.auxiliary_kernel!r}")
        if self.rotation_mode not in ("identity", "learned_givens", "frozen_random"):
            raise ValueError(f"rotation_mode must be identity/learned_givens/frozen_random, got {self.rotation_mode!r}")
        if self.smoother_mode not in ("baseline", "identity_mix", "bsf_mix"):
            raise ValueError(f"smoother_mode must be baseline/identity_mix/bsf_mix, got {self.smoother_mode!r}")
        if self.aux_view != "raw_first_hop":
            raise ValueError("v2 currently supports aux_view='raw_first_hop' only")
        if self.kernel_backend not in ("complex", "real_imag"):
            raise ValueError("kernel_backend must be 'complex' or 'real_imag'")
        if self.pocl_weight_target < 0:
            raise ValueError("pocl_weight_target must be non-negative")
        if self.contrastive_temperature <= 0 or not math.isfinite(self.contrastive_temperature):
            raise ValueError("contrastive_temperature must be positive and finite")
        if self.phase_scale <= 0 or not math.isfinite(self.phase_scale):
            raise ValueError("phase_scale must be positive and finite")
        if self.warmup_epochs < 0 or self.ramp_epochs <= 0:
            raise ValueError("warmup_epochs >= 0 and ramp_epochs > 0 required")
        if not math.isfinite(self.spectral_time) or not (0 <= self.spectral_time <= 1):
            raise ValueError("spectral_time must be in [0, 1]")
        if not math.isfinite(self.spectral_mix_target) or not (0 <= self.spectral_mix_target <= 1):
            raise ValueError("spectral_mix_target must be in [0, 1]")
        if self.aux_lr_ratio != 0.1 or self.aux_weight_decay != 0.0:
            raise ValueError("v2 requires aux_lr_ratio=0.1 and aux_weight_decay=0.0")
        if len(self.modality_weights) != 2 or min(self.modality_weights) < 0:
            raise ValueError("two non-negative modality_weights required")
        if not all(math.isfinite(float(v)) for v in self.modality_weights):
            raise ValueError("modality_weights must be finite")
        if not math.isclose(sum(self.modality_weights), 1.0, abs_tol=1e-7):
            raise ValueError("modality_weights must sum to 1")
        if self.diagnostic_interval_steps < 0:
            raise ValueError("diagnostic_interval_steps must be non-negative")
        if self.knn_block_size < 1:
            raise ValueError("knn_block_size must be positive")

    @classmethod
    def from_config(cls, cfg):
        """Build from FreeRec config; only recognized keys are pulled."""
        known = set(cls.__dataclass_fields__)
        args = {}
        for key in known:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                if key == "modality_weights" and isinstance(val, str):
                    val = tuple(map(float, val.split(",")))
                args[key] = val
        return cls(**args)


# ---------------------------------------------------------------------------
# Helper: tensor hash (mirrors stair_mhd_v3._tensor_hash)
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
    """STAIR4-v2 FreeRec model adapter.

    All baseline MI/FSC/BPR/scorer logic is reproduced exactly.
    The auxiliary head and smoother are added without touching baseline paths.
    """

    # Schema version for checkpoint compatibility.
    SCHEMA_VERSION = 1

    def __init__(self, dataset, cfg):
        super().__init__(dataset)
        self.cfg = cfg
        self.options = STAIR4V2Options.from_config(cfg)
        self.num_layers = int(cfg.num_layers)
        self.embedding_dim = int(cfg.embedding_dim)

        # Validate baseline parameters
        if self.num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {self.num_layers}")
        if self.embedding_dim < 2 or self.embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even and >= 2 for v2 Givens blocks, got {self.embedding_dim}"
            )
        if cfg.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {cfg.gamma}")
        if self.options.auxiliary_kernel != "none" and self.num_layers < 1:
            raise ValueError("num_layers must be >= 1 when the auxiliary cross-layer view is enabled")

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

        # Placeholder; populated in prepare()
        self.data_hashes: Dict[str, str] = {}
        self._train_index: Optional[TrainPositiveIndex] = None

        # --- Baseline parameter initialization ---
        self._reset_baseline_parameters()

        # --- Static graph preparation ---
        self.prepare(dataset.path)

        # GenRecArch does not create the task criterion for custom adapters.
        # Keep the baseline BPR objective explicit and use the same reduction
        # as main.py so A0 remains comparable to STAIR.
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")

        # --- Auxiliary head (isolated RNG) ---
        # torch.random.fork_rng with devices=[] → CPU-only fork; does not
        # consume from CUDA stream, preserving baseline sampling RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)  # deterministic auxiliary init
            self._init_auxiliary()

        # --- Baseline smoother ---
        self.smoother = MHDSmoother(self._apply_bsc_operator, self.beta, self.num_layers)

        # --- BSF direction smoother (wraps baseline smoother) ---
        self.bsf_smoother = BSFDirectionSmoother(
            self.smoother,
            lambda: self.mAdj,          # S_getter: always reads live model buffer
            spectral_time=self.options.spectral_time,
        )

        # Step-level diagnostics (never stored as tensors on the model)
        self.last_diagnostics: Dict = {}

    # ------------------------------------------------------------------
    # Parameter initialization (baseline contract §3.1)
    # ------------------------------------------------------------------

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
        """Create head inside forked RNG; no trainable params when kernel='none'."""
        opts = self.options
        if opts.auxiliary_kernel == "none":
            self.head: Optional[CrossLayerContrastiveHead] = None
        else:
            self.head = CrossLayerContrastiveHead(
                d=self.embedding_dim,
                kernel=opts.auxiliary_kernel,
                rotation=opts.rotation_mode,
                scale=opts.phase_scale,
                temperature=opts.contrastive_temperature,
            )

    # ------------------------------------------------------------------
    # Static graph preparation (baseline contract §3.1 + §14.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare(self, path) -> None:
        """Build only static, train-only buffers using baseline MI logic.

        Mirrors STAIR_MHD_v3.prepare() but for two-modality weighted MI
        without hypergraph incidence (no MHD gates needed for v2).
        """
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
            raise ValueError("v2 requires exactly two modalities and positive neighbor counts")

        # Load and whiten modality features
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

        # Train edge index (unique, canonical)
        edge_index = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        canonical = torch.unique(edge_index.detach().cpu().T, dim=0).T.contiguous()
        self.data_hashes["train"] = _tensor_hash(canonical)

        # Build item-item normalized adjacency S via kNN (same pipeline as baseline)
        from models.stair4_v2_utils import (
            baseline_whitening as _whitening,
        )
        try:
            from models.stair_mhd_v3_utils import exact_knn
        except ImportError:
            raise ImportError(
                "stair_mhd_v3_utils.exact_knn is required for v2 graph preparation"
            )

        directed_edges = []
        for m, (features, k) in enumerate(zip(mfeats, ks)):
            neighbors = exact_knn(features, k, opts.knn_block_size)
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

        # MI initialization (weighted average of whitened modalities)
        mi = sum(
            _whitening(f, self.embedding_dim) * k
            for f, k in zip(mfeats, ks)
        ) / sum(ks)
        self.Item.embeddings.weight.copy_(mi)

        # User embedding: row-normalized R @ mi
        r_indices, r_weights = freerec.graph.to_normalized(edge_index, normalization="left")
        R = torch.sparse_coo_tensor(
            r_indices, r_weights,
            size=(self.User.count, self.Item.count),
        ).to_sparse_csr()
        self.User.embeddings.weight.copy_(R @ mi)

        # Train positive index for POCL mask
        self._train_index = build_train_positive_index(
            canonical, self.User.count, self.Item.count
        )

        # Sanity: no buffer should have grad_fn after prepare
        if any(v.grad_fn is not None or v.requires_grad for v in self.buffers()):
            raise RuntimeError("prepare() created a non-static buffer with grad_fn")

    # ------------------------------------------------------------------
    # Baseline BSC operator (used by MHDSmoother callback)
    # ------------------------------------------------------------------

    def _apply_bsc_operator(self, features: Tensor, snapshot) -> Tensor:
        """Standard adjacency propagation for BSC Neumann smoother.

        When snapshot is None (use_baseline mode): pure mAdj propagation.
        v2 does not use the hypergraph gate snapshot from v3.
        """
        return self.mAdj @ features

    # ------------------------------------------------------------------
    # Epoch / schedule management
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def effective_coefficients(self) -> Tuple[float, float, str]:
        """Return (lambda_cl, zeta, active_smoother_mode).

        r(e) = clip((e - W) / R, 0, 1); schedule starts after warmup.
        """
        elapsed = self.current_epoch - self.options.warmup_epochs
        ramp = min(1.0, max(0.0, elapsed / max(1, self.options.ramp_epochs)))
        lambda_cl = ramp * self.options.pocl_weight_target
        zeta = ramp * self.options.spectral_mix_target

        # Determine active smoother mode
        if self.options.smoother_mode == "baseline" or zeta == 0.0:
            active_smoother = "baseline"
        else:
            active_smoother = self.options.smoother_mode

        # If auxiliary is off, CL weight is always zero
        if self.options.auxiliary_kernel == "none":
            lambda_cl = 0.0

        return lambda_cl, zeta, active_smoother

    # ------------------------------------------------------------------
    # Optimizer parameter groups (§14.4)
    # ------------------------------------------------------------------

    def parameter_groups(self) -> List[dict]:
        """Return three groups: users / items / aux (Givens theta only)."""
        user_params = list(self.User.parameters())
        item_params = list(self.Item.parameters())

        # Collect auxiliary parameters (Givens theta only; LayerNorm has no affine)
        user_ids = {id(p) for p in user_params}
        item_ids = {id(p) for p in item_params}
        aux_params = [
            p for p in self.parameters()
            if id(p) not in user_ids and id(p) not in item_ids
        ]

        # Validate: every parameter belongs to exactly one group
        seen = [id(p) for g in [user_params, item_params, aux_params] for p in g]
        if len(seen) != len(set(seen)) or set(seen) != {id(p) for p in self.parameters()}:
            raise RuntimeError(
                "parameter_groups: parameter coverage check failed — "
                "each parameter must appear in exactly one group"
            )

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
    # Forward: encode with baseline FSC + expose X0, X1 views
    # ------------------------------------------------------------------

    def _encode_baseline_fsc(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Baseline FSC forward.  Returns (userZ, itemZ, X0_all, X1_all).

        X0_all = [E_u; E_i]            — raw embeddings (layer 0)
        X1_all = Adj @ X0_all * (1-b)  — first-hop BEFORE beta attenuation
                                          (used as auxiliary view per §6.2)

        Z is the full Neumann polynomial average: same as baseline.
        """
        E_u = self.User.embeddings.weight   # [U, d]
        E_i = self.Item.embeddings.weight   # [I, d]
        X0_all = torch.cat([E_u, E_i], dim=0)   # [U+I, d]

        features = X0_all
        smoothed = X0_all
        beta_complement = 1 - self.beta                  # [d]
        norm_correction = 1 - beta_complement ** (self.num_layers + 1)  # [d]

        X1_all: Optional[Tensor] = None
        for layer in range(self.num_layers):
            features = self.Adj @ features * beta_complement   # [U+I, d]
            smoothed = smoothed + features
            if layer == 0:
                # Capture X1 BEFORE beta multiplication completes (design §6.2:
                # "X1 is selected before beta attenuation".
                # We store Adj @ X0 (scaled by beta_complement already).
                # To get pre-attenuation: divide back if nonzero, but the spec
                # says "before beta attenuation" == the hop BEFORE scaling.
                # Implementation: store Adj @ X0 (without the beta multiply).
                # Recompute cleanly: X1 = Adj @ X0_all (unscaled).
                X1_all = self.Adj @ X0_all          # [U+I, d] first-hop, unscaled

        Z = smoothed.mul(beta_complement).div(norm_correction)   # [U+I, d]
        if X1_all is None:
            # L=0 edge case: no hop, X1 = X0
            X1_all = X0_all

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
        """Compute BPR + optional POCL; arm smoother; return total loss."""
        self._global_step += 1
        lambda_cl, zeta, active_smoother = self.effective_coefficients()

        # --- Baseline FSC forward ---
        userZ, itemZ, X0_all, X1_all = self._encode_baseline_fsc()

        users = data[self.User]          # [B, 1]
        positives = data[self.Item]      # [B, 1]
        negatives = data[self.INeg]      # [B, K]

        # Baseline BPR loss (duplicates kept, matching baseline data contract)
        bpr_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userZ[users], itemZ[positives]),
            torch.einsum("BKD,BKD->BK", userZ[users], itemZ[negatives]),
        )

        diag = {
            "bpr": bpr_loss.detach().item(),
            "lambda_cl": lambda_cl,
            "zeta": zeta,
            "smoother_mode": active_smoother,
            "epoch": self.current_epoch,
            "step": self._global_step,
        }

        total_loss = bpr_loss

        # --- POCL auxiliary loss (only when lambda_cl > 0) ---
        if lambda_cl > 0 and self.head is not None and self._train_index is not None:
            # Deduplicate: POCL operates on unique users/positive items only
            cands = make_batch_candidates(users, positives)
            pocl_loss, pocl_diag = self.head(
                X0_all, X1_all, cands, self._train_index, n_users=self.User.count
            )
            total_loss = total_loss + lambda_cl * pocl_loss

            diag["cl_raw"] = pocl_diag.cl_raw
            diag["cl_weighted"] = lambda_cl * pocl_diag.cl_raw
            diag["pos_count_ui"] = pocl_diag.pos_count_ui
            diag["fidelity_q25"] = pocl_diag.fidelity_q25
            diag["fidelity_q75"] = pocl_diag.fidelity_q75
            if (
                self.options.diagnostic_interval_steps > 0
                and self._global_step % self.options.diagnostic_interval_steps == 0
            ):
                diag["theta_grad_norm"] = pocl_diag.theta_grad_norm
                diag["entropy_ui"] = pocl_diag.softmax_entropy_ui
        else:
            diag["cl_raw"] = 0.0
            diag["cl_weighted"] = 0.0

        # Arm BSF smoother for the item parameter group BEFORE backward.
        # The smoother will be called inside optimizer.step() via the callback.
        self.bsf_smoother.arm_step(
            mode=active_smoother,
            zeta=zeta,
            spectral_time=self.options.spectral_time,
        )

        # Store diagnostics (no tensor references kept)
        self.last_diagnostics = diag
        return total_loss

    # ------------------------------------------------------------------
    # Ranking / evaluation (baseline scorer, no phase head)
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

    # ------------------------------------------------------------------
    # Checkpoint helpers (get/set_extra_state)
    # ------------------------------------------------------------------

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
            raise ValueError(
                f"checkpoint schema {state.get('schema')!r} != model schema {self.SCHEMA_VERSION}; "
                "rebuild checkpoint with matching code"
            )
        current = self.get_extra_state()
        for key in ("data_hashes", "options", "embedding_dim", "num_layers", "gamma"):
            if state.get(key) != current[key]:
                raise ValueError(
                    f"checkpoint mismatch in {key!r}; rebuild with the same data/config.\n"
                    f"  checkpoint: {state.get(key)}\n"
                    f"  current:    {current[key]}"
                )
        self.current_epoch = int(state["epoch"])
        self._global_step = int(state.get("global_step", 0))
        # Disarm smoother in case checkpoint was saved mid-step
        if self.bsf_smoother._armed:
            self.bsf_smoother.clear_step_snapshot()
        if self.smoother._step_armed:
            self.smoother.clear_step_snapshot()

    # ------------------------------------------------------------------
    # Alias for FreeRec compatibility
    # ------------------------------------------------------------------

    def sure_trainpipe(self, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_pairs_source()
            .gen_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    # Expose for engine diagnostics
    @property
    def is_auxiliary_active(self) -> bool:
        return self.options.auxiliary_kernel != "none"

    @property
    def has_bsf(self) -> bool:
        return self.options.smoother_mode != "baseline"

    def marked_params(self) -> List[dict]:
        """Alias: freerec.launcher.Coach may call model.marked_params().

        Delegates to parameter_groups() so the baseline Coach interface
        is satisfied without duplicating group logic.
        """
        return self.parameter_groups()


# Alias for FreeRec model registry / CLI
STAIR4V2Model = STAIR4V2
