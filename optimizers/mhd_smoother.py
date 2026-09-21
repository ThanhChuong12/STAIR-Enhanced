"""Step-scoped, parameter-free Neumann smoother for the existing optimizer."""

from numbers import Integral

import torch


def _snapshot(value):
    # Deliberately independent of models/__init__.py and the FreeRec runtime.
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")


class MHDSmoother:
    """Apply a supplied operator to a detached snapshot explicitly set per step.

    The callback has signature ``operator(features, snapshot)``; ``None`` means
    the exact baseline operator. The trainer must clear the step in a finally
    block after optimizer.step(), and arm it again before the next update.

    Optimizer param groups keep this object by reference. Exclude their
    ``smoother`` entry from checkpoint state and restore the live instance on
    load: pickling/deepcopying a bound callback would serialize/copy its model.
    """

    def __init__(self, operator, beta: torch.Tensor, L: int):
        if not callable(operator):
            raise TypeError("operator must be callable")
        if not isinstance(beta, torch.Tensor) or not beta.is_floating_point() or beta.ndim > 1:
            raise TypeError("beta must be a floating scalar or one-dimensional tensor")
        if beta.numel() == 0 or not torch.isfinite(beta).all() or (beta < 0).any() or (beta >= 1).any():
            raise ValueError("beta must be finite and in [0, 1)")
        if isinstance(L, bool) or not isinstance(L, Integral) or L < 0:
            raise ValueError("L must be a nonnegative integer")
        self.operator = operator
        self.beta = beta.detach().clone()
        self.L = int(L)
        self._step_snapshot = None
        self._step_armed = False

    @property
    def step_snapshot(self):
        return self._step_snapshot

    def set_step_snapshot(self, snapshot):
        if snapshot is None:
            raise ValueError("use use_baseline() to arm a baseline step")
        self._step_snapshot = _snapshot(snapshot)
        self._step_armed = True

    def use_baseline(self):
        self._step_snapshot = None
        self._step_armed = True

    def clear_step_snapshot(self):
        self._step_snapshot = None
        self._step_armed = False

    @torch.no_grad()
    def __call__(self, features: torch.Tensor):
        if not self._step_armed:
            raise RuntimeError("arm MHDSmoother with the current step snapshot or use_baseline() before optimizer.step()")
        if features.ndim != 2 or not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("smoother features must be a finite floating matrix")
        if self.beta.ndim == 1 and self.beta.numel() not in (1, features.shape[1]):
            raise ValueError("beta must be scalar or have one entry per feature column")
        # Match baseline arithmetic and casting: beta moves device only.
        if self.beta.device != features.device:
            self.beta = self.beta.to(features.device)
        smoothed = features
        norm_correction = 1 - self.beta ** (self.L + 1)
        for _ in range(self.L):
            propagated = self.operator(features, self._step_snapshot)
            if propagated.shape != features.shape or propagated.device != features.device or propagated.dtype != features.dtype:
                raise ValueError("operator output must preserve feature shape, device, and dtype")
            if not torch.isfinite(propagated).all():
                raise ValueError("operator produced nonfinite features")
            features = propagated * self.beta
            smoothed = smoothed + features
        return smoothed.mul(1 - self.beta).div(norm_correction)
