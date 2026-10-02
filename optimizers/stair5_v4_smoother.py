"""Matrix-free Neumann smoother for static STAIR5-v4 blended operator S4."""
import torch


class STAIR5V4Smoother:
    """Smooths Adam-normalized directions with the blended operator S4:

        P_j(S4) V_{t, :, j} = [(1 - b_j) / (1 - b_j^{L+1})] sum_{l=0}^L b_j^l S4^l V_{t, :, j}

    Holds no parameters or autograd graph. The operator callable resolves the current
    graph buffer dynamically from the model.
    """

    def __init__(self, operator, beta, L: int):
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
