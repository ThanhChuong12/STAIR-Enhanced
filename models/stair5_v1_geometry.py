# -*- coding: utf-8 -*-
"""
models/stair5_v1_geometry.py — Lorentz & Euclidean Geometry Module for STAIR-LHC v1
===================================================================================
Senior AI Research Engineer Implementation:
Implements numerically stable Lorentz manifold operations and Euclidean counterpart:
1. Smooth Saturating Hyperbolic Radius Capping (cap_radius via tanh with ceiling R).
2. Lorentz Exponential Map (exp_0) from tangent space origin T_o H_kappa^D into Minkowski space R^{D+1}.
3. Taylor Series Expansion (Degree 4) for arcosh^2 near t -> 0 to eliminate gradient singularity.
4. Safe Pairwise Lorentz Distance Matrix (GEMM-vectorized inner product) without NaN gradients.
5. Hybrid Distance Kernel: D_hybrid = (1 - w) * D_L^2 + w * log(1 + D_L^2).
6. Euclidean Control Distance (E0 arm): D_E = ||v - w||_2^2 for strict baseline parity.
7. Constant-Radius Control (HC arm): ||v|| = 1.0 projection to isolate geometric capacity from kernel effect.
8. Native FP32 calculation outside mixed-precision autocast for numerical stability.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzGeometryModule(nn.Module):
    """
    Numerically robust Lorentz manifold & Euclidean geometry module for STAIR-LHC v1.

    Args:
        kappa (float): Curvature parameter (kappa > 0). Default: 1.0.
        radius_cap (float): Maximum allowed L2 radius R before hyperbolic exp map. Default: 2.0.
        w_hybrid (float): Hybrid kernel weighting factor w in [0.0, 1.0]. Default: 0.0.
        eps_taylor (float): Threshold t = delta - 1 below which Taylor series of arcosh^2 is used. Default: 1e-4.
        arm (str): Algorithm arm ('H0', 'E0', 'HC', 'H0w5', 'B0'). Default: 'H0'.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        radius_cap: float = 2.0,
        w_hybrid: float = 0.0,
        eps_taylor: float = 1e-4,
        arm: str = "H0",
    ) -> None:
        super().__init__()
        self.kappa = float(kappa)
        self.radius_cap = float(radius_cap)
        self.w_hybrid = float(w_hybrid)
        self.eps_taylor = float(eps_taylor)
        self.arm = arm.upper()

        if self.arm == "H0W5":
            self.w_hybrid = 0.5
        elif self.arm == "E0":
            self.kappa = 0.0

    def cap_radius(self, a: torch.Tensor, force_unit_norm: bool = False) -> torch.Tensor:
        """
        Projects vector 'a' into the open ball B_R of radius R = radius_cap via smooth tanh saturation.

        Formula:
            norm = ||a||_2
            v = R * tanh(norm / R) * (a / (norm + 1e-8))

        For Constant-Radius Control (HC arm):
            v = 1.0 * (a / (norm + 1e-8))
        """
        norm = torch.norm(a, p=2, dim=-1, keepdim=True)
        safe_norm = norm + 1e-8

        if force_unit_norm or self.arm == "HC":
            # Constant-Radius Control (HC arm): normalize strictly to unit sphere
            return a / safe_norm

        scale = torch.tanh(norm / self.radius_cap) / safe_norm
        return self.radius_cap * scale * a

    def lorentz_exp0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Lorentz Exponential Map from origin o = [1/sqrt(kappa), 0_D]^T:
            phi_kappa(v) = exp_o^kappa(v) = [ cosh(sqrt(kappa)*r)/sqrt(kappa), sinhc(sqrt(kappa)*r) * v ]^T
        where r = ||v||_2, sinhc(z) = sinh(z)/z.

        Output lies on the forward sheet of the two-sheeted hyperboloid:
            <x, x>_L = -x_0^2 + ||x_s||_2^2 = -1 / kappa,  with x_0 > 0.
        """
        # Enforce FP32 for numerical precision
        v_fp32 = v.float()
        r = torch.norm(v_fp32, p=2, dim=-1, keepdim=True)

        kappa = max(self.kappa, 1e-6)
        sqrt_k = math.sqrt(kappa)
        kr = sqrt_k * r

        # Stable sinhc: sinh(kr)/kr, Taylor expansion 1 + (kr)^2 / 6 for kr < 1e-5
        sinhc = torch.where(
            kr < 1e-5,
            1.0 + (kr ** 2) / 6.0,
            torch.sinh(torch.clamp(kr, max=15.0)) / (kr + 1e-8),
        )

        x0 = torch.cosh(torch.clamp(kr, max=15.0)) / sqrt_k
        xs = sinhc * v_fp32

        return torch.cat([x0, xs], dim=-1)

    def pairwise_lorentz_distance_squared(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes GEMM-vectorized matrix of squared Lorentz geodesic distances between point sets x and y.

        Args:
            x: (B_u, D+1) points on Lorentz manifold.
            y: (B_i, D+1) points on Lorentz manifold.

        Returns:
            d_squared: (B_u, B_i) non-negative squared distance matrix D_kappa(x, y).
        """
        # Ensure FP32
        x = x.float()
        y = y.float()

        # Minkowski inner product: -<x, y>_L = x0 y0^T - xs ys^T
        # Matrix multiply: (B_u, 1) @ (1, B_i) - (B_u, D) @ (D, B_i)
        inner_L = x[:, 0:1] @ y[:, 0:1].t() - x[:, 1:] @ y[:, 1:].t()

        kappa = max(self.kappa, 1e-6)
        delta = kappa * inner_L

        # Geodesic argument delta must be >= 1.0
        delta = torch.clamp(delta, min=1.0)
        t = delta - 1.0

        # Taylor expansion of arcosh^2(1 + t) around t = 0 to order 4:
        # arcosh^2(1 + t) = 2t - t^2/3 + 4t^3/45 - t^4/35 + O(t^5)
        taylor_branch = (
            2.0 * t
            - (t ** 2) / 3.0
            + (4.0 * (t ** 3)) / 45.0
            - (t ** 4) / 35.0
        )

        # Standard acosh branch with clamped input to guarantee finite non-NaN gradients during backprop
        safe_delta_for_acosh = torch.clamp(delta, min=1.0 + 1e-7)
        acosh_branch = torch.acosh(safe_delta_for_acosh).pow(2)

        # Select branch using eps_taylor threshold
        d_squared = torch.where(t < self.eps_taylor, taylor_branch, acosh_branch) / kappa
        return torch.clamp(d_squared, min=0.0)

    def pairwise_euclidean_distance_squared(
        self, v: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes pairwise squared Euclidean distances: ||v - w||_2^2 = ||v||^2 + ||w||^2 - 2 <v, w>.

        Args:
            v: (B_u, D) tangent vectors or normalized vectors.
            w: (B_i, D) tangent vectors or normalized vectors.

        Returns:
            d_squared: (B_u, B_i) non-negative squared Euclidean distance matrix.
        """
        v = v.float()
        w = w.float()

        v_sq = torch.sum(v ** 2, dim=-1, keepdim=True)  # (B_u, 1)
        w_sq = torch.sum(w ** 2, dim=-1, keepdim=True)  # (B_i, 1)
        d_squared = v_sq + w_sq.t() - 2.0 * (v @ w.t())
        return torch.clamp(d_squared, min=0.0)

    def paired_lorentz_distance_squared(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes paired squared Lorentz distances between x[k] and y[k] (N pairs).
        Args:
            x: (N, D+1) points on Lorentz manifold.
            y: (N, D+1) points on Lorentz manifold.
        Returns:
            d_squared: (N,) non-negative paired squared distance vector.
        """
        x = x.float()
        y = y.float()

        # Minkowski inner product: -<x, y>_L = x0 y0 - sum(xs * ys)
        inner_L = x[:, 0] * y[:, 0] - torch.sum(x[:, 1:] * y[:, 1:], dim=-1)
        kappa = max(self.kappa, 1e-6)
        delta = torch.clamp(kappa * inner_L, min=1.0)
        t = delta - 1.0

        taylor_branch = (
            2.0 * t
            - (t ** 2) / 3.0
            + (4.0 * (t ** 3)) / 45.0
            - (t ** 4) / 35.0
        )
        safe_delta = torch.clamp(delta, min=1.0 + 1e-7)
        acosh_branch = torch.acosh(safe_delta).pow(2)

        d_squared = torch.where(t < self.eps_taylor, taylor_branch, acosh_branch) / kappa
        return torch.clamp(d_squared, min=0.0)

    def paired_euclidean_distance_squared(
        self, v: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Computes paired squared Euclidean distance ||v[k] - w[k]||_2^2 for N pairs."""
        v = v.float()
        w = w.float()
        d_squared = torch.sum((v - w) ** 2, dim=-1)
        return torch.clamp(d_squared, min=0.0)

    def compute_paired_distance(
        self, v: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Computes paired distance vector between tangent vectors v and w."""
        if self.arm == "E0":
            d2 = self.paired_euclidean_distance_squared(v, w)
        else:
            x = self.lorentz_exp0(v)
            y = self.lorentz_exp0(w)
            d2 = self.paired_lorentz_distance_squared(x, y)

        if self.w_hybrid > 0.0:
            return (1.0 - self.w_hybrid) * d2 + self.w_hybrid * torch.log1p(d2)
        return d2

    def compute_distance_matrix(
        self,
        u_vec: torch.Tensor,
        i_vec: torch.Tensor,
        is_embedded_lorentz: bool = False,
    ) -> torch.Tensor:
        """
        High-level dispatcher computing pairwise distance matrix according to the designated arm.

        Args:
            u_vec: (B_u, D) tangent vectors (or (B_u, D+1) if already Lorentz embedded).
            i_vec: (B_i, D) tangent vectors (or (B_i, D+1) if already Lorentz embedded).
            is_embedded_lorentz: Flag indicating if inputs are already in Minkowski space R^{D+1}.

        Returns:
            D_matrix: (B_u, B_i) pairwise hybrid distance matrix.
        """
        # Euclidean control arm (E0): compute Euclidean distance directly
        if self.arm == "E0":
            d2 = self.pairwise_euclidean_distance_squared(u_vec, i_vec)
        else:
            # Lorentz branches (H0, HC, H0w5)
            if not is_embedded_lorentz:
                u_lor = self.lorentz_exp0(u_vec)
                i_lor = self.lorentz_exp0(i_vec)
            else:
                u_lor = u_vec
                i_lor = i_vec
            d2 = self.pairwise_lorentz_distance_squared(u_lor, i_lor)

        # Apply Hybrid Kernel weighting if w_hybrid > 0
        if self.w_hybrid > 0.0:
            return (1.0 - self.w_hybrid) * d2 + self.w_hybrid * torch.log1p(d2)
        return d2
