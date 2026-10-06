"""optimizers/stair5_v6_smoother.py — Matrix-free Neumann Smoother for S_6 (BCSR).
================================================================================
Smooths Adam-normalized directions with the STAIR5-v6 blended operator S_6:

    P_j(S) V_{t, :, j} = [(1 - b_j) / (1 - b_j^{L+1})] sum_{l=0}^L b_j^l S^l V_{t, :, j}

where b_j = beta3[j], default L=3.
Holds no parameters or autograd graph. The operator callable dynamically resolves SpMM.
"""
import torch


class STAIR5V6Smoother:
    """Matrix-free Neumann polynomial smoother for AdamWSEvo on item embeddings."""

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
