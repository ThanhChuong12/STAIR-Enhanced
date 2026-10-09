"""optimizers/stair5_v8_smoother.py — Optimizer BSC and Residual Dirichlet-Floor Smoother.
=============================================================================================
Implements Stage 4 of STAIR5-v8 (DF-BSC):
- Polynomial Stepwise Smoother P_8(D_t) on Adam-preconditioned item update directions (Eq 14).
- Optional Residual Dirichlet-Floor Mechanism: \\widetilde{D}_t = \\omega D_t + (1 - \\omega) P_8 D_t (Eq 15).

Mathematical Formulation:
--------------------------
1. Coordinate-dependent attenuation coefficient (Eq 13):
   b_j = 0.1 + 0.9 * (j / d)^{gamma_{BSC}},   j = 0, ..., d - 1,   b_j in [0.1, 1.0)

2. Polynomial Smoother Response for L=3 (Eq 14):
   P_j(S_8) D_{t, :, j} = [ (1 - b_j) / (1 - b_j^{L+1}) ] * sum_{l=0}^L b_j^l S_8^l D_{t, :, j}

3. Residual Dirichlet-Floor Extension (Eq 15):
   \\widetilde{D}_t = \\omega D_t + (1 - \\omega) P_8 D_t,   0.0 <= \\omega <= 0.1
   Primary arm uses \\omega = 0.0 (exact V4 polynomial recovery without residual overhead).

Theorem 2 Guarantees (under ||S_8||_2 <= 1.0, L=3, and \\mathcal{L}_8 = I - S_8 >= 0):
   1. Frobenius Norm Bound: ||\\widetilde{D}_t||_F <= ||D_t||_F
   2. Dirichlet Energy Floor: \\omega^2 \\mathcal{E}_8(D_t) <= \\mathcal{E}_8(\\widetilde{D}_t) <= \\mathcal{E}_8(D_t)
   where \\mathcal{E}_8(D) = sum_j D_{:, j}^T (I - S_8) D_{:, j}.

Memory Contract:
   - Evaluated strictly within @torch.no_grad().
    - Matrix-free recurrence with a constant number of N x d work buffers.
   - When \\omega == 0.0, zero additional buffers or allocations beyond the baseline path.
"""

from typing import Any, Callable, Optional
import torch


class STAIR5V8Smoother:
    """Matrix-free Stepwise Smoother with optional Residual Dirichlet Floor for STAIR5-v8.

    Args:
        operator: Callable taking a tensor X [N, d] and returning S_8 @ X.
        beta: 1D Tensor or array of coordinate coefficients b_j in [0, 1)^d.
        L: Polynomial truncation order (default L=3).
        bsc_residual: Residual bypass coefficient \\omega in [0.0, 0.1]. Default 0.0.
    """

    def __init__(
        self,
        operator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        beta: Optional[torch.Tensor] = None,
        L: int = 3,
        bsc_residual: float = 0.0,
        omega: Optional[float] = None,
        spmm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        op = operator if operator is not None else spmm_fn
        if op is None or not callable(op):
            raise TypeError("operator (or spmm_fn) must be a callable computing SpMM (S_8 @ X).")
        if L < 0 or int(L) != L:
            raise ValueError("L must be a nonnegative integer.")

        if beta is None:
            raise ValueError("beta must be provided.")
        beta_t = torch.as_tensor(beta).detach().clone()
        if beta_t.ndim != 1 or beta_t.numel() == 0:
            raise ValueError("BSC beta must be a nonempty coordinate vector.")
        if not torch.isfinite(beta_t).all() or (beta_t < 0).any() or (beta_t >= 1).any():
            raise ValueError("BSC beta must be finite and within [0, 1).")

        res_val = omega if omega is not None else bsc_residual
        residual = float(res_val)
        if not (0.0 <= residual <= 0.1):
            raise ValueError("bsc_residual (omega) must be in [0.0, 0.1].")

        self.operator = op
        self.beta = beta_t
        self.L = int(L)
        self.omega = residual

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Smooths Adam-preconditioned direction tensor D_t [N, d].

        Args:
            features: FP32 Tensor [N, d] representing direction D_t.

        Returns:
            Smoothed direction tensor \\widetilde{D}_t of shape [N, d].
        """
        beta = self.beta.to(features.device, features.dtype)
        if features.ndim != 2 or features.shape[1] != beta.numel():
            raise ValueError("Directions must have shape [N, len(beta)].")
        norm_correction = 1.0 - beta ** (self.L + 1)

        # Baseline path when omega == 0.0 (zero extra clone or memory overhead)
        if self.omega == 0.0:
            smoothed = features
            cur = features
            for _ in range(self.L):
                cur = self.operator(cur) * beta
                smoothed = smoothed + cur
            return smoothed.mul(1.0 - beta).div(norm_correction)

        # DF-BSC residual path when omega > 0.0
        # Preserves input D_t to blend with P_8 D_t
        D_t = features
        smoothed = features
        cur = features
        for _ in range(self.L):
            cur = self.operator(cur) * beta
            smoothed = smoothed + cur
        P_8_D = smoothed.mul(1.0 - beta).div(norm_correction)

        # Equation (15): \widetilde{D}_t = \omega D_t + (1 - \omega) P_8 D_t
        return P_8_D.mul_(1.0 - self.omega).add_(D_t, alpha=self.omega)

    @torch.no_grad()
    def compute_dirichlet_energy(self, direction: torch.Tensor) -> torch.Tensor:
        """Computes the Dirichlet energy E_8(D) = Tr(D^T (I - S_8) D) for diagnostics.

        Args:
            direction: Tensor D of shape [N, d].

        Returns:
            Device scalar; callers may transfer it when logging outside the inner loop.
        """
        return compute_dirichlet_energy(self.operator, direction)


@torch.no_grad()
def compute_dirichlet_energy(
    operator_or_tensor: Any,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Computes the Dirichlet energy E_8(D) = Tr(D^T (I - S_8) D) >= 0.

    Args:
        operator_or_tensor: SpMM callable x -> S_8 @ x, or PyTorch sparse/dense tensor S_8.
        direction: Direction matrix D [N, d], FP32.

    Returns:
        Raw device scalar; nonfinites and small roundoff negatives remain observable.
    """
    if callable(operator_or_tensor):
        S_D = operator_or_tensor(direction)
    elif getattr(operator_or_tensor, "is_sparse", False) or getattr(operator_or_tensor, "is_sparse_csr", False):
        S_D = torch.sparse.mm(operator_or_tensor, direction)
    else:
        S_D = operator_or_tensor @ direction

    # Preserve the raw diagnostic: no per-step host synchronization or NaN masking.
    return torch.sum(direction * (direction - S_D))
