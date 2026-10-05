"""optimizers/stair5_v51_smoother.py — STAIR5-v5.1 BSC Smoother.
==============================================================
Matrix-free Neumann smoother for static STAIR5-v5.1 operators (S4 and S_CAM).

Applies the polynomial filter to Adam-normalized gradient directions:
    P_j(S) V_{t, :, j} = [(1 - c_j) / (1 - c_j^{L+1})] sum_{l=0}^L c_j^l S^l V_{t, :, j}

Zero parameters, zero trainable state, no autograd tracking.
"""
import torch


class STAIR5V51Smoother:
    """Smooths Adam-normalized directions with the static operator S (S4 or S_CAM):

    Holds no parameters or autograd graph. The operator callable resolves the current
    graph buffer dynamically from the model.
    """

    def __init__(self, operator, beta, L: int = 3):
        if not callable(operator):
            raise TypeError("operator must be a callable returning SpMM of the graph.")
        if L < 0 or int(L) != L:
            raise ValueError("L must be a nonnegative integer.")
        beta = torch.as_tensor(beta).detach().clone()
        if not torch.isfinite(beta).all() or (beta < 0).any() or (beta >= 1).any():
            raise ValueError("BSC beta must be finite in [0, 1).")
        self.operator = operator
        self.beta = beta
        self.L = int(L)

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        beta = self.beta.to(features.device, features.dtype)
        smoothed = features
        norm_correction = 1.0 - beta ** (self.L + 1)
        for _ in range(self.L):
            features = self.operator(features) * beta
            smoothed = smoothed + features
        return smoothed.mul(1.0 - beta).div(norm_correction)
