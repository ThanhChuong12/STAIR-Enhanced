"""Matrix-free Neumann smoother for a static STAIR5-v3 DP-PC-BSC operator callback."""
import torch


class STAIR5V3Smoother:
    """Smooth Adam-normalized directions, preserving baseline operation order.

    This object holds no parameters or autograd graph. A model-bound callback
    resolves the current graph buffer after device moves or checkpoint loads.
    """
    def __init__(self, operator, beta, L):
        if not callable(operator):
            raise TypeError("operator must be a callable, not a stored graph tensor.")
        if L < 0 or int(L) != L:
            raise ValueError("L must be a nonnegative integer.")
        beta = torch.as_tensor(beta).detach().clone()
        if not torch.isfinite(beta).all() or (beta < 0).any() or (beta >= 1).any():
            raise ValueError("BSC beta must be finite in [0, 1).")
        self.operator, self.beta, self.L = operator, beta, int(L)

    @torch.no_grad()
    def __call__(self, features):
        beta = self.beta.to(features.device, features.dtype)
        smoothed = features
        norm_correction = 1 - beta ** (self.L + 1)
        for _ in range(self.L):
            features = self.operator(features) * beta
            smoothed = smoothed + features
        return smoothed.mul(1 - beta).div(norm_correction)
