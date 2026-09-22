"""BSF direction smoother for STAIR4-v2: blends Neumann BSC with Q_t(H).

Public API
----------
- BSFDirectionSmoother(baseline_smoother, S_getter, spectral_time)

Mathematical contract (§6.1, §14.3 of STAIR4_v2_Report.md)
-----------------------------------------------------------
H = (I - S) / 2,  σ(H) ⊆ [0, 1]  (given S symmetric, spectral norm ≤ 1)
Q_t(x) = x - (t²/2) * H(H(x))    eigenvalues of Q in [1 - t²/2, 1]

apply_H(x) = 0.5 * (x - S @ x)   — two SpMMs, no dense I or H²

Blended smoother:
  T_j = (1 - ζ) * P_{b_j,L}(S) [baseline] + ζ * Q_t

mode dispatch:
  'baseline'     or ζ=0 → exact baseline MHDSmoother (fast path)
  'identity_mix'         → (1-ζ)*P(Δ) + ζ*Δ
  'bsf_mix'              → (1-ζ)*P(Δ) + ζ*Q_t(Δ)

Optimizer contract:
  - Operates on the Adam direction Δ AFTER bias correction and before the
    parameter update (i.e., inside AdamWSEvo.update_embeddings as the
    'smoother' callback).
  - Does NOT recompute moments, apply weight decay, or call optimizer.step().
  - Does NOT hold trainable parameters.
  - S_getter is a callable () → sparse Tensor so that live model state is
    always accessed (avoids stale reference after .to(device) / checkpoint load).

Step lifecycle (enforced):
  1. arm_step(mode, zeta, spectral_time=None)   before optimizer.step()
  2. __call__(delta)                             inside optimizer callback
  3. clear_step_snapshot()                      in finally block after step
  Calling __call__ before arm_step → RuntimeError.
  Calling arm_step twice without clearing → RuntimeError.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Union

import torch
from torch import Tensor

from optimizers.mhd_smoother import MHDSmoother

# Modes accepted by arm_step
_VALID_MODES = ("baseline", "identity_mix", "bsf_mix")


class BSFDirectionSmoother:
    """Callback that transforms the Adam direction Δ ∈ ℝ^{I×d}.

    This is a plain callable (not nn.Module) with no learnable parameters.
    The optimizer param group stores it by reference; exclude it from
    checkpoint state and rebind after load.

    Parameters
    ----------
    baseline_smoother : MHDSmoother
        The existing Neumann BSC smoother; used for the 'baseline' fast path
        and as the P(Δ) component in blended modes.
    S_getter : Callable[[], Tensor]
        Zero-argument callable returning the current sparse item-item adjacency
        S on the correct device.  Must return a sparse Tensor with spectral norm ≤ 1
        and symmetric structure (validated at arm time for small graphs).
    spectral_time : float
        Default spectral time t ∈ [0, 1].  Can be overridden at arm_step.
    """

    def __init__(
        self,
        baseline_smoother: MHDSmoother,
        S_getter: Callable[[], Tensor],
        spectral_time: float = 0.5,
    ) -> None:
        if not isinstance(baseline_smoother, MHDSmoother):
            raise TypeError("baseline_smoother must be an MHDSmoother instance")
        if not callable(S_getter):
            raise TypeError("S_getter must be callable")
        if not math.isfinite(spectral_time) or not (0.0 <= spectral_time <= 1.0):
            raise ValueError(f"spectral_time must be in [0, 1], got {spectral_time}")
        self._baseline = baseline_smoother
        self._S_getter = S_getter
        self._default_t = float(spectral_time)
        # Step state (armed per step)
        self._armed: bool = False
        self._mode: Optional[str] = None
        self._zeta: Optional[float] = None
        self._t: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def arm_step(
        self,
        mode: str,
        zeta: float,
        spectral_time: Optional[float] = None,
    ) -> None:
        """Capture immutable step coefficients before optimizer.step().

        Must be called exactly once per step; cleared in finally block by
        clear_step_snapshot().  Calling twice without clearing raises.

        Parameters
        ----------
        mode : str  'baseline' | 'identity_mix' | 'bsf_mix'
        zeta : float  ζ ∈ [0, 1]; if 0, forced to 'baseline' fast path
        spectral_time : float | None  overrides default if provided
        """
        if self._armed:
            raise RuntimeError(
                "BSFDirectionSmoother.arm_step() called while already armed. "
                "Call clear_step_snapshot() in a finally block after optimizer.step()."
            )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if not math.isfinite(zeta) or not (0.0 <= zeta <= 1.0):
            raise ValueError(f"zeta must be in [0, 1], got {zeta}")
        t = self._default_t if spectral_time is None else float(spectral_time)
        if not math.isfinite(t) or not (0.0 <= t <= 1.0):
            raise ValueError(f"spectral_time must be in [0, 1], got {t}")

        # Normalize: zeta=0 always uses baseline fast path
        if zeta == 0.0:
            mode = "baseline"

        # Capture immutable detached scalars (no tensors to prevent grad_fn)
        self._armed = True
        self._mode = mode
        self._zeta = float(zeta)
        self._t = float(t)

    def clear_step_snapshot(self) -> None:
        """Disarm the smoother; must be called in a finally block after step."""
        self._armed = False
        self._mode = None
        self._zeta = None
        self._t = None

    # ------------------------------------------------------------------
    # Sparse spectral operators (no dense matrices)
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _apply_H(x: Tensor, S: Tensor) -> Tensor:
        """H = (I - S)/2; apply_H(x) = 0.5*(x - S@x).  Two SpMMs."""
        if S.layout not in (torch.sparse_coo, torch.sparse_csr):
            raise TypeError(f"S must be sparse (COO or CSR), got layout={S.layout}")
        if S.shape[0] != x.shape[0]:
            raise ValueError(
                f"S.shape[0]={S.shape[0]} must equal x.shape[0]={x.shape[0]}"
            )
        Sx = torch.sparse.mm(S, x)                       # [I, d]
        return 0.5 * (x - Sx)

    @classmethod
    @torch.no_grad()
    def _apply_Q(cls, x: Tensor, S: Tensor, t: float) -> Tensor:
        """Q_t(x) = x - (t²/2) * H(H(x)).  Four SpMMs total.  t=0 → identity."""
        if t == 0.0:
            return x.clone()
        Hx = cls._apply_H(x, S)                          # [I, d]
        HHx = cls._apply_H(Hx, S)                        # [I, d]
        return x - (0.5 * t * t) * HHx

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, delta: Tensor) -> Tensor:
        """Transform the Adam direction Δ ∈ ℝ^{I×d}.

        Called inside AdamWSEvo.update_embeddings as the 'smoother' callable.
        Must be armed first via arm_step().

        Parameters
        ----------
        delta : Tensor[I, d]  float32 or float64  (Adam direction, bias-corrected)

        Returns
        -------
        Tensor[I, d]  same dtype and device as delta
        """
        if not self._armed:
            raise RuntimeError(
                "BSFDirectionSmoother.__call__() invoked before arm_step(). "
                "Call arm_step(mode, zeta) before optimizer.step()."
            )
        # Shape/dtype/finite checks
        if not isinstance(delta, Tensor) or delta.ndim != 2:
            raise ValueError("delta must be a 2-D Tensor")
        if not delta.is_floating_point():
            raise TypeError(f"delta must be floating-point, got {delta.dtype}")
        if not torch.isfinite(delta).all():
            raise ValueError("delta contains non-finite values; skipping step should happen upstream")

        mode = self._mode
        zeta = self._zeta
        t = self._t

        # --- Fast path: exact baseline smoother arithmetic ---
        if mode == "baseline":
            # Arm baseline smoother and delegate
            self._baseline.use_baseline()
            result = self._baseline(delta)
            self._baseline.clear_step_snapshot()
            return result

        # --- Shared: compute P(Δ) via baseline smoother ---
        self._baseline.use_baseline()
        p_delta = self._baseline(delta)
        self._baseline.clear_step_snapshot()

        if mode == "identity_mix":
            # T(Δ) = (1-ζ)*P(Δ) + ζ*Δ
            return (1.0 - zeta) * p_delta + zeta * delta

        elif mode == "bsf_mix":
            # T(Δ) = (1-ζ)*P(Δ) + ζ*Q_t(Δ)
            S = self._S_getter()
            # Ensure S is on the same device as delta
            if S.device != delta.device:
                S = S.to(delta.device)
            if S.dtype != delta.dtype:
                S = S.to(dtype=delta.dtype)
            q_delta = self._apply_Q(delta, S, t)          # [I, d]
            return (1.0 - zeta) * p_delta + zeta * q_delta

        else:
            raise RuntimeError(f"Unknown mode {mode!r} — should have been caught at arm_step")

    # ------------------------------------------------------------------
    # Diagnostics (detached, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_smoothing_ratio(self, delta: Tensor) -> float:
        """||T(Δ)||_F / ||Δ||_F for the current step configuration.

        Useful for diagnostic logging; must be called when armed.
        Does NOT advance the step (does not consume the arm).
        """
        if not self._armed:
            raise RuntimeError("compute_smoothing_ratio requires an armed step")
        t_delta = self(delta)
        denom = delta.norm().item()
        if denom == 0.0:
            return float("nan")
        return float(t_delta.norm().item() / denom)

    def __repr__(self) -> str:
        state = f"armed={self._armed}"
        if self._armed:
            state += f", mode={self._mode!r}, zeta={self._zeta:.4f}, t={self._t:.4f}"
        return f"BSFDirectionSmoother({state}, default_t={self._default_t:.4f})"
