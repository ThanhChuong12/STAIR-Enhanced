"""Parameter-free static BSC, applied to Adam directions, not raw gradients."""
from __future__ import annotations

import torch


class STAIR4V4Smoother:
    def __init__(self, operator, beta, layers, identity_mix=0.0):
        if not callable(operator) or layers < 0:
            raise ValueError("a callable operator and nonnegative layers are required")
        if not torch.isfinite(beta).all() or not ((beta >= 0) & (beta < 1)).all():
            raise ValueError("BSC beta must lie in [0, 1)")
        if not 0 <= identity_mix <= 1:
            raise ValueError("identity_mix must lie in [0, 1]")
        self.operator = operator
        self.beta = beta.detach().clone()
        self.layers = int(layers)
        self.identity_mix = float(identity_mix)
        self._armed = False

    def arm_static(self):
        self._armed = True

    def clear_step_snapshot(self):
        self._armed = False

    @torch.no_grad()
    def __call__(self, direction):
        if not self._armed:
            raise RuntimeError("arm_static() must precede each optimizer step")
        # Cache the static coefficient on its execution device after first use.
        self.beta = self.beta.to(device=direction.device, dtype=direction.dtype)
        beta = self.beta
        features = direction
        smoothed = direction
        correction = 1 - beta ** (self.layers + 1)
        for _ in range(self.layers):
            features = self.operator(features) * beta
            smoothed = smoothed + features
        smoothed = smoothed.mul(1 - beta).div(correction)
        if self.identity_mix:
            smoothed = (1 - self.identity_mix) * smoothed + self.identity_mix * direction
        return smoothed
