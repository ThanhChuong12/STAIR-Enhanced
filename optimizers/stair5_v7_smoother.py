"""optimizers/stair5_v7_smoother.py — Dynamic UCR-D Item Direction Smoother.
=============================================================================
Intercepts actual Adam update direction D_t, coordinates dynamic UCR-D operator
mutation via STAIR5V7GraphAdapter, performs 3-hop BSC convolution coordinate-wise:

    P_j(S_7^{(t)}) D_{t, :, j} = [(1 - b_j) / (1 - b_j^{L+1})] sum_{l=0}^L b_j^l (S_7^{(t)})^l D_{t, :, j}

where b_j = beta[j], L=3.
Guarantees clean snapshot lifecycle (clear_step_snapshot) in all execution paths,
including exceptions, preventing GPU memory leaks.
"""
from typing import Any, Callable, Dict, Optional
import torch


class STAIR5V7Smoother:
    """Dynamic UCR-D Smoother for item embeddings in AdamWSEvo."""

    def __init__(
        self,
        graph_adapter: Any,
        beta: torch.Tensor,
        L: int = 3,
    ) -> None:
        if L < 0 or int(L) != L:
            raise ValueError("Polynomial degree L must be a nonnegative integer.")
        beta = torch.as_tensor(beta).detach().clone()
        if not torch.isfinite(beta).all() or (beta < 0).any() or (beta >= 1).any():
            raise ValueError("BSC beta must be finite in [0, 1).")

        self.graph_adapter = graph_adapter
        self.beta = beta
        self.L = int(L)
        self.current_progress = 10.0  # Default to full target dose warmup

    def set_progress(self, progress: float) -> None:
        """Sets the current training progress (completed epochs + batch fraction)."""
        self.current_progress = float(progress)

    def clear_step_snapshot(self) -> None:
        """Explicitly releases any transient GPU snapshot buffers."""
        if hasattr(self.graph_adapter, "clear_step_snapshot"):
            self.graph_adapter.clear_step_snapshot()

    @torch.no_grad()
    def _smooth(self, operator: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Executes matrix-free coordinate-wise Neumann polynomial convolution."""
        beta = self.beta.to(features.device, features.dtype)
        smoothed = features
        cur = features
        norm_correction = 1.0 - beta ** (self.L + 1)

        for _ in range(self.L):
            cur = torch.sparse.mm(operator, cur) * beta
            smoothed = smoothed + cur

        return smoothed.mul(1.0 - beta).div(norm_correction)

    @torch.no_grad()
    def __call__(self, D: torch.Tensor) -> torch.Tensor:
        """Smooths Adam direction D_t with dynamic UCR-D operator or fast-path baseline."""
        D = D.detach()
        if not torch.isfinite(D).all():
            raise ValueError("[STAIR5-v7] Nonfinite values encountered in Adam direction tensor D.")

        # Zero-overhead delegation for v4 recovery or zero dose
        if getattr(self.graph_adapter, "is_fast_path", False):
            return self._smooth(self.graph_adapter.base_operator, D)

        try:
            active_op = self.graph_adapter.update_step_snapshot(
                D, epoch_progress=self.current_progress
            )
            return self._smooth(active_op, D)
        finally:
            self.clear_step_snapshot()
