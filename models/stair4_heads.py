"""Phase encoder, Givens rotation, and contrastive loss heads for STAIR4-v2.

Public API
----------
- PhaseEncoder(d, scale=1.0)          real[B,d] → complex[B,d]
- PairwiseGivens(d)                   complex[B,d] → complex[B,d]  (target branch)
- phase_fidelity(query_c, target_c)   → real[Bq, Bk] in [0, 1]
- cosine_kernel(x0, x1, givens=None)  → real[Bq, Bk] in [-1, 1]
- multi_positive_ce(logits, mask, temperature) → scalar
- CrossLayerContrastiveHead(d, kernel, rotation)  POCL loss + diagnostics

Mathematical contracts (see §4.3 / §6.2 of STAIR4_v2_Report.md)
-----------------------------------------------------------------
- phase_fidelity uses coherent squared overlap: |ψ_u^† ψ_i|²
  implemented as c = query.conj() @ target.T then c.real² + c.imag²
  No [B, B, d] broadcast tensor is ever materialized.
- Row norms of PhaseEncoder output: ||ψ(x)||² = 1 (unit norm per row).
- PairwiseGivens: theta real[d/2], init zero → identity at epoch 0.
  Applied on target branch only; query is unrotated.
- multi_positive_ce: uniform target over positives per row (not masked_fill
  of positives to -inf); supports all-positive rows with log of uniform CE.

Kernel selector values: 'phase_fidelity', 'cosine', 'none'
Rotation selector values: 'learned_givens', 'identity', 'frozen_random'
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.stair4_v2_utils import CandidateBatch, build_positive_mask, TrainPositiveIndex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_complex(t: Tensor, name: str) -> None:
    if not isinstance(t, Tensor) or not t.is_complex() or t.ndim != 2:
        raise ValueError(f"{name} must be a 2-D complex Tensor, got shape={tuple(t.shape) if isinstance(t, Tensor) else '?'} dtype={getattr(t, 'dtype', '?')}")


def _check_real2d(t: Tensor, name: str) -> None:
    if not isinstance(t, Tensor) or t.is_complex() or not t.is_floating_point() or t.ndim != 2:
        raise ValueError(f"{name} must be a 2-D real float Tensor")


def _to_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


# ---------------------------------------------------------------------------
# PhaseEncoder
# ---------------------------------------------------------------------------

class PhaseEncoder(nn.Module):
    """Map real embeddings to unit-norm complex phase vectors.

    Forward
    -------
    1. LayerNorm(d, affine=False, eps=1e-5)  — per-row normalization
    2. phi = arctan(scale * x_normed)        — scale in (0, ∞), fixed in pilot
    3. psi = (cos(phi) + i*sin(phi)) / sqrt(d)

    Output row norms: ||psi||² = sum_k |psi_k|² = sum_k 1/d = 1.
    float32 input → complex64; float64 → complex128.
    """

    def __init__(self, d: int, scale: float = 1.0) -> None:
        super().__init__()
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and ≥ 2, got {d}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {scale}")
        self.d = d
        self.scale = float(scale)
        # affine=False: no learnable shift/scale; eps matches design spec
        self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:  # [B, d] real → [B, d] complex
        _check_real2d(x, "x")
        if x.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d} columns, got {x.shape[1]}")
        # LayerNorm operates in real domain
        x_norm = self.layer_norm(x)                      # [B, d] real
        phi = torch.arctan(self.scale * x_norm)          # [B, d] real, range (-π/2, π/2)
        scale_d = math.sqrt(self.d)
        # Build complex output: (cos+i*sin)/sqrt(d); avoids torch.complex() on scalars
        cos_phi = torch.cos(phi) / scale_d               # [B, d] real
        sin_phi = torch.sin(phi) / scale_d               # [B, d] real
        cdtype = _to_complex_dtype(x.dtype)
        return torch.complex(cos_phi, sin_phi).to(cdtype)  # [B, d] complex


# ---------------------------------------------------------------------------
# PairwiseGivens (target branch only)
# ---------------------------------------------------------------------------

class PairwiseGivens(nn.Module):
    """Block-diagonal Givens rotation on *target* branch only.

    d/2 independent 2×2 blocks; no dense d×d matrix is built.
    theta is real[d/2], initialized to zero (→ identity at epoch 0).

    For target complex vector x[..., 2k], x[..., 2k+1]:
        [x_even_k']   [cos(t_k)  -sin(t_k)] [x_even_k]
        [x_odd_k' ] = [sin(t_k)   cos(t_k)] [x_odd_k ]

    Applied in complex domain: pair (z_0, z_1) becomes
        z_0' = cos(t)*z_0 - sin(t)*z_1
        z_1' = sin(t)*z_0 + cos(t)*z_1

    Contract: G†G = I  (verified by tests; unit test in tests/test_stair4_v2_heads.py)
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and ≥ 2, got {d}")
        self.d = d
        # theta real [d/2]; zero init → identity rotation
        self.theta = nn.Parameter(torch.zeros(d // 2))

    def forward(self, x: Tensor) -> Tensor:  # [B, d] complex → [B, d] complex
        _check_complex(x, "x")
        if x.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d} columns, got {x.shape[1]}")
        # Reshape to blocks: [B, d/2, 2]
        B = x.shape[0]
        x_blocks = x.reshape(B, self.d // 2, 2)          # [B, d/2, 2] complex
        c = torch.cos(self.theta)                         # [d/2]
        s = torch.sin(self.theta)                         # [d/2]
        z0 = x_blocks[..., 0]                             # [B, d/2] complex
        z1 = x_blocks[..., 1]                             # [B, d/2] complex
        # Givens rotation on each pair
        z0_new = c * z0 - s * z1                          # [B, d/2] complex
        z1_new = s * z0 + c * z1                          # [B, d/2] complex
        result = torch.stack([z0_new, z1_new], dim=-1)    # [B, d/2, 2] complex
        return result.reshape(B, self.d)                  # [B, d] complex


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def phase_fidelity(query_c: Tensor, target_c: Tensor) -> Tensor:
    """Coherent squared fidelity K(u,i) = |ψ_u† ψ_i|² ∈ [0, 1].

    Implements via GEMM in complex domain: c = query.conj() @ target.T
    then K = c.real² + c.imag².  Never materializes [Bq, Bk, d].

    Parameters
    ----------
    query_c  : complex[Bq, d]
    target_c : complex[Bk, d]

    Returns
    -------
    real[Bq, Bk] in [0, 1]
    """
    _check_complex(query_c, "query_c")
    _check_complex(target_c, "target_c")
    if query_c.shape[1] != target_c.shape[1]:
        raise ValueError(
            f"query and target must have the same d; got {query_c.shape[1]} vs {target_c.shape[1]}"
        )
    # complex GEMM: [Bq, d] × [d, Bk] → [Bq, Bk]
    c = query_c.conj() @ target_c.T                      # [Bq, Bk] complex
    return c.real.square() + c.imag.square()             # [Bq, Bk] real ∈ [0, 1]


def cosine_kernel(
    x0: Tensor,
    x1: Tensor,
    givens: Optional[PairwiseGivens] = None,
    layer_norm: Optional[nn.LayerNorm] = None,
) -> Tensor:
    """Cosine similarity control kernel for ablation A1/A7.

    Uses the same per-row LayerNorm normalization as PhaseEncoder,
    then L2-normalize and compute dot products.  Optional Givens
    applied to target only (for A1-G capacity control).

    Parameters
    ----------
    x0 : real[Bq, d]  query features
    x1 : real[Bk, d]  target features
    givens : optional PairwiseGivens for target branch (real mode via re/im channels)
    layer_norm : optional shared LayerNorm; if None, a temporary affine=False norm is used

    Returns
    -------
    real[Bq, Bk] in [-1, 1]
    """
    _check_real2d(x0, "x0")
    _check_real2d(x1, "x1")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError("x0 and x1 must have the same d")
    d = x0.shape[1]
    if layer_norm is None:
        layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-5).to(x0.device)
    q = F.normalize(layer_norm(x0), dim=-1)              # [Bq, d]
    k = F.normalize(layer_norm(x1), dim=-1)              # [Bk, d]
    if givens is not None:
        # Apply Givens in real by treating real[d] as complex[d/2] re+im channels
        # This is a capacity-matched control: d/2 rotation parameters, same as phase branch
        half = d // 2
        if givens.d != half:
            raise ValueError(
                "Cosine-kernel Givens rotation must operate on d/2 complex "
                f"channels; got givens.d={givens.d}, expected {half}"
            )
        k_complex = torch.complex(k[:, :half], k[:, half:])  # [Bk, d/2]
        k_rotated = givens(k_complex)                         # [Bk, d/2] complex
        k = torch.cat([k_rotated.real, k_rotated.imag], dim=-1)  # [Bk, d]
        k = F.normalize(k, dim=-1)
    return q @ k.T                                        # [Bq, Bk]


# ---------------------------------------------------------------------------
# Multi-positive contrastive CE loss
# ---------------------------------------------------------------------------

def multi_positive_ce(
    logits: Tensor,
    mask: Tensor,
    temperature: float = 0.2,
) -> Tuple[Tensor, Dict[str, float]]:
    """Supervised in-batch CE with uniform target over positives.

    L = -mean_u [ sum_i Y_ui * log_softmax(logits/tau)_i ]
    where Y_ui = M_ui / sum_j M_uj  (uniform over positives per row).

    Constraints enforced:
    - No masked_fill of positives to -inf.
    - Rows with zero positives → rejected (violates CandidateBatch contract).
    - All-positive rows → uniform CE; logged but not discarded.

    Parameters
    ----------
    logits : real[B_q, B_k]
    mask   : bool[B_q, B_k]  True where (u,i) is a train positive
    temperature : float > 0

    Returns
    -------
    (loss scalar, diagnostics dict with detached Python scalars)
    """
    _check_real2d(logits, "logits")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError("mask must be a bool tensor with the same shape as logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")

    # Validate rows have at least one positive
    row_pos_count = mask.sum(dim=1).float()               # [B_q]
    if (row_pos_count == 0).any():
        raise ValueError(
            "Every query row must have at least one positive; "
            "check make_batch_candidates and build_positive_mask"
        )

    # Compute log-softmax over keys
    log_probs = F.log_softmax(logits / temperature, dim=1)  # [B_q, B_k]

    # Uniform target distribution over positives
    target = mask.float() / row_pos_count[:, None]        # [B_q, B_k], sums to 1 per row

    # Cross-entropy: -sum_i Y_ui * log_prob_i  per row, then mean over rows
    loss = -(target * log_probs).sum(dim=1).mean()

    # Diagnostics — all detached Python scalars; never call .item() in a loop per edge
    with torch.no_grad():
        all_positive_rows = int((row_pos_count == logits.shape[1]).sum().item())
        probs = F.softmax(logits / temperature, dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean().item()
        diag: Dict[str, float] = {
            "pos_count_mean": float(row_pos_count.mean().item()),
            "all_positive_rows": float(all_positive_rows),
            "softmax_entropy": entropy,
        }
    return loss, diag


# ---------------------------------------------------------------------------
# CrossLayerContrastiveHead
# ---------------------------------------------------------------------------

class POCLDiagnostics(NamedTuple):
    """Detached per-step diagnostics from CrossLayerContrastiveHead."""
    cl_raw: float           # raw loss before lambda weighting
    pos_count_ui: float     # mean positive count per query row (u→i direction)
    pos_count_iu: float     # mean positive count per query row (i→u direction)
    fidelity_q25: float     # 25th percentile of fidelity scores
    fidelity_q75: float     # 75th percentile of fidelity scores
    softmax_entropy_ui: float
    softmax_entropy_iu: float
    theta_grad_norm: float  # gradient norm of Givens theta (0 if no Givens)


class CrossLayerContrastiveHead(nn.Module):
    """POCL: phase-overlap contrastive learning on cross-layer views.

    Architecture
    ------------
    - X0 = E^(0) (raw item embeddings, before FSC)
    - X1 = A @ E^(0) (first-hop, before beta attenuation)
    - POCL uses X0 as query and G(X1) as target (G = Givens on target only)
    - Two directions computed independently:
        K^{u→i} = |Ψ(U0)* [G Ψ(I1)]^T|²   [B_u, B_i]
        K^{i→u} = |Ψ(I0)* [G Ψ(U1)]^T|²   [B_i, B_u]
    - Loss = 0.5 * (L_ui + L_iu)

    The head returns raw loss (without lambda weighting).
    Lambda is applied in the model/engine, not here.

    Parameters
    ----------
    d : int  embedding dimension (must be even, ≥ 2)
    kernel : str  'phase_fidelity' | 'cosine' | 'none'
    rotation : str  'learned_givens' | 'identity' | 'frozen_random'
    scale : float  arctan scale for PhaseEncoder (fixed in pilot)
    temperature : float  contrastive temperature τ_c
    """

    def __init__(
        self,
        d: int,
        kernel: str = "phase_fidelity",
        rotation: str = "learned_givens",
        scale: float = 1.0,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel not in ("phase_fidelity", "cosine", "none"):
            raise ValueError(f"kernel must be one of 'phase_fidelity', 'cosine', 'none'; got {kernel!r}")
        if rotation not in ("learned_givens", "identity", "frozen_random"):
            raise ValueError(f"rotation must be one of 'learned_givens', 'identity', 'frozen_random'; got {rotation!r}")
        if d < 2 or d % 2 != 0:
            raise ValueError(f"d must be even and ≥ 2, got {d}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive and finite, got {temperature}")

        self.d = d
        self.kernel_mode = kernel
        self.rotation_mode = rotation
        self.temperature = temperature

        if kernel == "phase_fidelity":
            self.encoder = PhaseEncoder(d, scale=scale)
        elif kernel == "cosine":
            # Shared LayerNorm for cosine control; no PhaseEncoder
            self.layer_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-5)
            self.encoder = None  # type: ignore[assignment]
        else:
            self.encoder = None  # type: ignore[assignment]

        # Givens rotation on target branch
        if rotation == "learned_givens":
            # Phase vectors have d complex channels.  The cosine control
            # packs d real channels into d/2 complex channels, so its
            # capacity-matched rotation must be sized independently.
            self.givens = PairwiseGivens(d if kernel == "phase_fidelity" else d // 2)
        elif rotation == "frozen_random":
            self.givens = PairwiseGivens(d if kernel == "phase_fidelity" else d // 2)
            # Freeze: no gradient, sample random rotation
            nn.init.uniform_(self.givens.theta, -math.pi, math.pi)
            self.givens.theta.requires_grad_(False)
        else:
            self.givens = None  # type: ignore[assignment]

    @property
    def has_trainable_parameters(self) -> bool:
        return (
            self.rotation_mode == "learned_givens"
            and self.givens is not None
        )

    def _encode_phase(self, x: Tensor, apply_givens: bool = False) -> Tensor:
        """Encode real embeddings to complex; optionally apply Givens."""
        psi = self.encoder(x)                             # [B, d] complex
        if apply_givens and self.givens is not None:
            psi = self.givens(psi)
        return psi

    def _kernel_scores(
        self,
        query: Tensor,    # [Bq, d] real
        target: Tensor,   # [Bk, d] real
        apply_givens_to_target: bool = True,
    ) -> Tensor:
        """Compute score matrix [Bq, Bk] for the configured kernel."""
        if self.kernel_mode == "phase_fidelity":
            psi_q = self._encode_phase(query, apply_givens=False)
            psi_t = self._encode_phase(target, apply_givens=apply_givens_to_target)
            return phase_fidelity(psi_q, psi_t)           # [Bq, Bk] in [0, 1]
        elif self.kernel_mode == "cosine":
            givens = self.givens if (apply_givens_to_target and self.givens is not None) else None
            return cosine_kernel(query, target, givens=givens, layer_norm=self.layer_norm)
        else:
            raise RuntimeError("kernel_mode='none': forward should not be called")

    def forward(
        self,
        X0_all: Tensor,          # [U+I, d] full embedding table at layer 0
        X1_all: Tensor,          # [U+I, d] first-hop embeddings (before beta)
        candidates: CandidateBatch,
        train_index: TrainPositiveIndex,
        n_users: int,
    ) -> Tuple[Tensor, POCLDiagnostics]:
        """Compute POCL loss for a minibatch.

        Parameters
        ----------
        X0_all : real[U+I, d]  full embedding table at layer 0 (with grad)
        X1_all : real[U+I, d]  A @ E^(0) first-hop (with grad)
        candidates : CandidateBatch  deduplicated user/item IDs for this batch
        train_index : TrainPositiveIndex  prebuilt positive lookup
        n_users : int  number of users (offset for item IDs in global table)

        Returns
        -------
        (loss, diagnostics)  — loss carries grad_fn; diagnostics are detached
        """
        if self.kernel_mode == "none":
            raise RuntimeError("CrossLayerContrastiveHead called with kernel_mode='none'")

        u_ids = candidates.user_ids.to(X0_all.device)   # [B_u]
        i_ids = candidates.item_ids.to(X0_all.device)   # [B_i]

        # Select rows from global embedding table
        # Users occupy rows [0, n_users); items occupy rows [n_users, n_users+n_items)
        U0 = X0_all[u_ids]                               # [B_u, d]
        I0 = X0_all[n_users + i_ids]                     # [B_i, d]
        U1 = X1_all[u_ids]                               # [B_u, d]
        I1 = X1_all[n_users + i_ids]                     # [B_i, d]

        # Build positive mask on CPU then transfer
        mask_ui = build_positive_mask(
            candidates.user_ids, candidates.item_ids, train_index
        ).to(X0_all.device)                              # [B_u, B_i] bool

        mask_iu = mask_ui.T.contiguous()                  # [B_i, B_u] bool

        # Validate mask contract: every row must have ≥ 1 positive
        if (mask_ui.sum(dim=1) == 0).any():
            raise RuntimeError(
                "POCL: some query users have no train-positive items in the candidate batch; "
                "check make_batch_candidates uniqueness and train_index provenance"
            )
        if (mask_iu.sum(dim=1) == 0).any():
            raise RuntimeError(
                "POCL: some query items have no train-positive users in the candidate batch"
            )

        # Score matrices — Givens applied to target branch only
        K_ui = self._kernel_scores(U0, I1, apply_givens_to_target=True)   # [B_u, B_i]
        K_iu = self._kernel_scores(I0, U1, apply_givens_to_target=True)   # [B_i, B_u]

        loss_ui, diag_ui = multi_positive_ce(K_ui, mask_ui, self.temperature)
        loss_iu, diag_iu = multi_positive_ce(K_iu, mask_iu, self.temperature)
        loss = 0.5 * (loss_ui + loss_iu)

        # Fidelity quantiles for diagnostics (detached)
        with torch.no_grad():
            all_scores = K_ui.detach().reshape(-1)
            q25 = float(torch.quantile(all_scores, 0.25).item())
            q75 = float(torch.quantile(all_scores, 0.75).item())

            # Givens theta grad norm (0 if no learned rotation or no backward yet)
            theta_grad_norm = 0.0
            if self.givens is not None and self.givens.theta.grad is not None:
                theta_grad_norm = float(self.givens.theta.grad.norm().item())

        diag = POCLDiagnostics(
            cl_raw=float(loss.detach().item()),
            pos_count_ui=diag_ui["pos_count_mean"],
            pos_count_iu=diag_iu["pos_count_mean"],
            fidelity_q25=q25,
            fidelity_q75=q75,
            softmax_entropy_ui=diag_ui["softmax_entropy"],
            softmax_entropy_iu=diag_iu["softmax_entropy"],
            theta_grad_norm=theta_grad_norm,
        )
        return loss, diag
