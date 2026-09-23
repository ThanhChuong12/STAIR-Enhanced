"""STAIR4-v2.1 Direction Smoother (§7 of STAIR4_v2_1_Report.md).

Architecture:
-------------
- Baseline: P(S)Δ (Neumann polynomial smoother of STAIR via MHDSmoother)
- Optional Residual Spectral Correction (v2.1-S):
    T_ξ(Δ) = (I - ξ H^2) P(S)Δ,  where H = (I - S) / 2
- Fast Path: When ξ = 0, exactly returns P(S)Δ with zero additional SpMM overhead.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from optimizers.mhd_smoother import MHDSmoother


class STAIR4V21Smoother:
    """Callback transforming the Adam update direction Δ ∈ ℝ^{I×d} for items.

    Parameters
    ----------
    baseline_smoother : MHDSmoother
        The original STAIR Neumann BSC polynomial smoother.
    S_getter : Callable[[], Tensor]
        Zero-argument callable returning the current sparse item-item adjacency S.
    default_xi : float
        Default residual spectral correction factor ξ ∈ [0, 0.1]. Default: 0.0.
    """

    def __init__(
        self,
        baseline_smoother: MHDSmoother,
        S_getter: Callable[[], Tensor],
        default_xi: float = 0.0,
    ) -> None:
        if not isinstance(baseline_smoother, MHDSmoother):
            raise TypeError("baseline_smoother must be an MHDSmoother instance")
        if not callable(S_getter):
            raise TypeError("S_getter must be callable")
        if not math.isfinite(default_xi) or not (0.0 <= default_xi <= 0.1):
            raise ValueError(f"default_xi must be in [0, 0.1], got {default_xi}")

        self._baseline = baseline_smoother
        self._S_getter = S_getter
        self._default_xi = float(default_xi)

        # Step state
        self._armed: bool = False
        self._mode: Optional[str] = None
        self._xi: float = 0.0

    def arm_step(
        self,
        mode: str = "baseline",
        xi: Optional[float] = None,
    ) -> None:
        """Arm step parameters before optimizer.step()."""
        if self._armed:
            raise RuntimeError(
                "STAIR4V21Smoother.arm_step() called while already armed. "
                "Call clear_step_snapshot() in a finally block after optimizer.step()."
            )
        if mode not in ("baseline", "residual_spectral"):
            raise ValueError(
                "mode must be 'baseline' or 'residual_spectral', "
                f"got {mode!r}"
            )
        eff_xi = self._default_xi if xi is None else float(xi)
        if not math.isfinite(eff_xi) or not 0.0 <= eff_xi <= 0.1:
            raise ValueError(f"xi must be finite and in [0, 0.1], got {eff_xi}")

        if eff_xi == 0.0:
            mode = "baseline"

        self._armed = True
        self._mode = mode
        self._xi = eff_xi

    def clear_step_snapshot(self) -> None:
        """Disarm the smoother; must be called after optimizer.step()."""
        self._armed = False
        self._mode = None
        self._xi = 0.0

    @staticmethod
    @torch.no_grad()
    def _apply_H(x: Tensor, S: Tensor) -> Tensor:
        """H = (I - S) / 2; apply_H(x) = 0.5 * (x - S @ x)."""
        Sx = torch.sparse.mm(S, x)
        return 0.5 * (x - Sx)

    @torch.no_grad()
    def __call__(self, delta: Tensor) -> Tensor:
        """Transform the Adam direction Δ ∈ ℝ^{I×d}."""
        if not self._armed:
            raise RuntimeError(
                "STAIR4V21Smoother.__call__() invoked before arm_step()."
            )
        if not isinstance(delta, Tensor) or delta.ndim != 2:
            raise ValueError("delta must be a 2-D Tensor")
        if not torch.isfinite(delta).all():
            raise ValueError("delta contains non-finite values")

        # Step 1: Compute baseline Neumann polynomial smoother P(Δ)
        self._baseline.use_baseline()
        p_delta = self._baseline(delta)
        self._baseline.clear_step_snapshot()

        # Step 2: Fast path if ξ == 0 (exact baseline recovery)
        if self._xi == 0.0 or self._mode == "baseline":
            return p_delta

        # Step 3: Optional residual spectral correction (v2.1-S)
        # T_ξ(Δ) = (I - ξ H^2) P(Δ) = P(Δ) - ξ H(H(P(Δ)))
        S = self._S_getter()
        if S.device != delta.device:
            S = S.to(delta.device)
        if S.dtype != delta.dtype:
            S = S.to(dtype=delta.dtype)

        Hp = self._apply_H(p_delta, S)
        HHp = self._apply_H(Hp, S)
        return p_delta - self._xi * HHp

    def __repr__(self) -> str:
        state = f"armed={self._armed}"
        if self._armed:
            state += f", mode={self._mode!r}, xi={self._xi:.4f}"
        return f"STAIR4V21Smoother({state}, default_xi={self._default_xi:.4f})"


# Backward-compatible alias
BSFDirectionSmoother = STAIR4V21Smoother
