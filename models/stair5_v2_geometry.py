"""Bounded FP32 geometry for the optional audited STAIR5-v2 loss.

The Euclidean ranking backbone never calls these operations. Curvature and
radius are fixed hyperparameters; no hybrid kernel or radial temperature is used.
"""
import math
import torch
from torch import nn


class AuditedGeometry(nn.Module):
    def __init__(self, kappa=1., radius_cap=2., arm="H0", eps_taylor=1e-4):
        super().__init__()
        if not math.isfinite(radius_cap) or radius_cap <= 0:
            raise ValueError("radius_cap must be finite and positive.")
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and nonnegative.")
        self.arm = arm.upper()
        if self.arm not in ("H0", "E0", "HC"):
            raise ValueError("Geometry arm must be H0, E0 or HC.")
        if self.arm != "E0" and (kappa <= 0 or math.sqrt(kappa) * max(radius_cap, 1.) > 4):
            raise ValueError("Lorentz FP32 geometry requires 0 < sqrt(kappa)*max(R,1) <= 4.")
        if not 0 < eps_taylor <= 1e-2:
            raise ValueError("eps_taylor must be in (0, 1e-2].")
        self.kappa = float(kappa) if self.arm != "E0" else 0.
        self.radius_cap, self.eps_taylor = float(radius_cap), float(eps_taylor)

    def cap_radius(self, a, force_unit_norm=False):
        a = a.float()
        norm = a.norm(dim=-1, keepdim=True)
        if force_unit_norm or self.arm == "HC":
            return a / norm.clamp_min(1e-8)
        x = norm / self.radius_cap
        ratio = torch.where(x < 1e-4, 1 - x.square()/3,
                            torch.tanh(x) / x.clamp_min(1e-8))
        return a * ratio

    def lorentz_exp0(self, v):
        if self.kappa <= 0:
            raise ValueError("The Euclidean control has no Lorentz exponential map.")
        v = v.float()
        z = math.sqrt(self.kappa) * v.norm(dim=-1, keepdim=True)
        ratio = torch.where(z < 1e-4, 1 + z.square()/6,
                            torch.sinh(z) / z.clamp_min(1e-8))
        return torch.cat((torch.cosh(z)/math.sqrt(self.kappa), ratio*v), dim=-1)

    def fused_scale_cap_exp0(self, a, q):
        return self.lorentz_exp0(self.cap_radius(a.float()/max(float(q), 1e-4)))

    def _geodesic(self, argument):
        t = (argument - 1).clamp_min(0)
        small = t.clamp_max(self.eps_taylor)
        series = 2*small - small.square()/3 + 4*small.pow(3)/45 - small.pow(4)/35
        regular = torch.acosh(argument.clamp_min(1+1e-7)).square()
        return torch.where(t < self.eps_taylor, series, regular).clamp_min(0)/self.kappa

    def pairwise_lorentz_distance_squared(self, x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, y = x.float(), y.float()
            argument = self.kappa*(x[:, :1] @ y[:, :1].t() - x[:, 1:] @ y[:, 1:].t())
            return self._geodesic(argument)

    def pairwise_euclidean_distance_squared(self, x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x, y = x.float(), y.float()
            return (x.square().sum(-1, keepdim=True) + y.square().sum(-1)[None]
                    - 2*x @ y.t()).clamp_min(0)

    def compute_distance_matrix_from_lorentz(self, x, y):
        return self.pairwise_lorentz_distance_squared(x, y)

    def compute_distance_matrix(self, x, y):
        if self.arm == "E0":
            return self.pairwise_euclidean_distance_squared(x, y)
        return self.pairwise_lorentz_distance_squared(self.lorentz_exp0(x), self.lorentz_exp0(y))
