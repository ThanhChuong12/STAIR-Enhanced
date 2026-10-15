"""Bounded soft-consensus residual for the V8.1 contrastive query only.

The module never stores forward tensors. Only the cap and invariant gate
context are detached; message and query embeddings retain their gradients.
"""

import math
import numbers
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def consensus_dose(epoch, rho_max=0.2, warmup_epochs=30, ramp_epochs=30):
    """One-based schedule: epochs 1--30 are zero and epoch 60 reaches 0.2.

    A zero-length ramp activates the full dose immediately after warmup.
    """
    epoch = _nonnegative_integer(epoch, "epoch")
    if epoch == 0:
        raise ValueError("epoch must be one-based (>= 1).")
    warmup_epochs = _nonnegative_integer(warmup_epochs, "warmup_epochs")
    ramp_epochs = _nonnegative_integer(ramp_epochs, "ramp_epochs")
    if isinstance(rho_max, bool) or not isinstance(rho_max, numbers.Real) or not math.isfinite(rho_max) or not 0 <= rho_max < 1:
        raise ValueError("rho_max must be finite and in [0, 1).")
    if epoch <= warmup_epochs:
        return 0.0
    fraction = 1.0 if ramp_epochs == 0 else min((epoch - warmup_epochs) / ramp_epochs, 1.0)
    return float(rho_max * fraction)


dose_schedule = consensus_dose


class ConsensusResidual(nn.Module):
    """Norm-capped residual with an invariant scalar learned or fixed gate.

    Inputs have shape [number_of_unique_sampled_items, dimension]. Coverage
    is a static [number_of_unique_sampled_items] tensor indicating nonempty
    consensus rows. The learned 3 -> 8 -> 1 gate has exactly 41 parameters.
    Forward performs no CPU transfer or scalar GPU synchronization.
    """

    def __init__(self, gate="learned", hidden=8, eps=1e-8, seed=2718):
        super().__init__()
        if gate not in ("learned", "fixed"):
            raise ValueError("gate must be 'learned' or 'fixed'.")
        if isinstance(hidden, bool) or not isinstance(hidden, numbers.Integral) or hidden <= 0:
            raise ValueError("hidden must be a positive integer.")
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be an integer in [0, 2**64).")
        self.gate_mode = gate
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.seed = int(seed)
        if gate == "learned":
            # CPU construction and initialization consume no caller RNG. Model
            # transfer to CUDA happens afterwards, without CUDA RNG initialization.
            with torch.random.fork_rng(devices=[]):
                generator = torch.Generator(device="cpu").manual_seed(self.seed)
                self.scalar_gate = nn.Sequential(
                    nn.Linear(3, self.hidden), nn.SiLU(), nn.Linear(self.hidden, 1)
                )
                nn.init.xavier_uniform_(self.scalar_gate[0].weight, generator=generator)
                nn.init.zeros_(self.scalar_gate[0].bias)
                nn.init.normal_(self.scalar_gate[2].weight, std=1e-3, generator=generator)
                nn.init.zeros_(self.scalar_gate[2].bias)
        else:
            self.scalar_gate = None

    @staticmethod
    def _validate(x, m, coverage):
        if x.ndim != 2 or m.shape != x.shape or x.shape[1] == 0:
            raise ValueError("x and m must have matching 2D nonzero-width shapes.")
        if not x.is_floating_point() or not m.is_floating_point():
            raise TypeError("x and m must be real floating-point tensors.")
        if x.device != m.device or x.dtype != m.dtype:
            raise ValueError("x and m must have the same device and dtype.")
        if coverage.ndim != 1 or coverage.shape[0] != x.shape[0] or coverage.device != x.device:
            raise ValueError("coverage must be a same-device 1D tensor with one entry per row.")

    @torch.no_grad()
    def gate_context(self, x, m, coverage):
        """Return detached [B,1] cap and [B,3] rotation-invariant context."""
        self._validate(x, m, coverage)
        # The intended engine is FP32. Keeping norms in FP32 also makes this
        # robust to half-precision inputs without amplifying tiny messages.
        xf, mf = x.float(), m.float()
        nx = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
        nm = torch.linalg.vector_norm(mf, dim=-1, keepdim=True)
        cap = (nx / (nm + self.eps)).clamp(max=1.0)
        cosine = (xf * mf).sum(-1, keepdim=True) / (nx.clamp_min(self.eps) * nm.clamp_min(self.eps))
        cosine = torch.where((nx >= self.eps) & (nm >= self.eps), cosine.clamp(-1.0, 1.0), 0.0)
        ratio = ((nm + self.eps).log() - (nx + self.eps).log()).clamp(-4.0, 4.0) / 4.0
        context = torch.cat((cosine, ratio, (coverage != 0).float().unsqueeze(-1)), dim=-1)
        return cap.to(dtype=x.dtype), context

    def gate_values(self, context):
        """Evaluate the scalar gate on an already detached context."""
        if context.ndim != 2 or context.shape[1] != 3:
            raise ValueError("context must have shape [B, 3].")
        if self.scalar_gate is None:
            return context.new_full((context.shape[0], 1), 0.5)
        return torch.sigmoid(self.scalar_gate(context.detach().to(dtype=self.scalar_gate[0].weight.dtype)))

    def forward(self, x, m, coverage, rho):
        if isinstance(rho, bool) or not isinstance(rho, numbers.Real) or not math.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError("rho must be a finite Python scalar in [0, 1).")
        self._validate(x, m, coverage)
        if rho == 0:
            return x
        cap, context = self.gate_context(x, m, coverage)
        gate = self.gate_values(context).to(dtype=x.dtype)
        return x + float(rho) * gate * cap * m

    def forward_with_diagnostics(self, x, m, coverage, rho):
        """Return live query and detached scalars, evaluating the gate only once.

        Use this on periodically sampled batches, rather than every optimizer
        step. As in forward, zero dose does not evaluate the auxiliary gate.
        """
        if isinstance(rho, bool) or not isinstance(rho, numbers.Real) or not math.isfinite(rho) or not 0 <= rho < 1:
            raise ValueError("rho must be a finite Python scalar in [0, 1).")
        self._validate(x, m, coverage)
        if rho == 0:
            zeros = x.new_zeros((x.shape[0], 1))
            return x, self.diagnostic_summary(x, m, x, g=zeros, cap=zeros)
        cap, context = self.gate_context(x, m, coverage)
        gate = self.gate_values(context).to(dtype=x.dtype)
        query = x + float(rho) * gate * cap * m
        return query, self.diagnostic_summary(x, m, query, g=gate, cap=cap)

    @torch.no_grad()
    def diagnostic_summary(self, x, m, query, g=None, cap=None) -> Dict[str, torch.Tensor]:
        """Detached device scalars; transfer or aggregate at epoch boundaries.

        Gate values can be supplied to avoid recomputing them for diagnostics.
        Missing g is reconstructed from detached context; no tensors are cached.
        Zero queries contribute zero angular shift rather than an undefined angle.
        """
        if query.shape != x.shape or query.device != x.device:
            raise ValueError("query must match x in shape and device.")
        if g is None:
            coverage = torch.linalg.vector_norm(m.float(), dim=-1) > 0
            computed_cap, context = self.gate_context(x, m, coverage)
            cap = computed_cap if cap is None else cap
            g = self.gate_values(context)
        xf, mf, qf = x.float(), m.float(), query.float()
        nx = torch.linalg.vector_norm(xf, dim=-1)
        nm = torch.linalg.vector_norm(mf, dim=-1)
        nq = torch.linalg.vector_norm(qf, dim=-1)
        ratio = torch.linalg.vector_norm(qf - xf, dim=-1) / nx.clamp_min(self.eps)
        unit_x = F.normalize(xf, dim=-1, eps=self.eps)
        perpendicular = mf - (mf * unit_x).sum(-1, keepdim=True) * unit_x
        valid = (nx >= self.eps) & (nq >= self.eps)
        cosine = (xf * qf).sum(-1) / (nx.clamp_min(self.eps) * nq.clamp_min(self.eps))
        cosine_distance = torch.where(valid, 1.0 - cosine.clamp(-1.0, 1.0), 0.0)
        # This is arccos(unit_x dot unit_q), evaluated through the equivalent
        # half-angle formula for better precision near parallel vectors.
        unit_q = F.normalize(qf, dim=-1, eps=self.eps)
        angle = 2.0 * torch.atan2(
            torch.linalg.vector_norm(unit_q - unit_x, dim=-1),
            torch.linalg.vector_norm(unit_q + unit_x, dim=-1),
        )
        shift = torch.where(valid, angle, 0.0)
        denominator = max(x.shape[0], 1)
        if cap is None:
            cap = (nx / (nm + self.eps)).clamp(max=1.0).unsqueeze(-1)
        gate_mean = g.float().sum() / denominator
        gate_second_moment = g.float().square().sum() / denominator
        perpendicular_fraction = (torch.linalg.vector_norm(perpendicular, dim=-1) / nm.clamp_min(self.eps)).sum() / denominator
        orthogonal_ratio = (torch.linalg.vector_norm(cap.float() * perpendicular, dim=-1) / (nx + self.eps)).sum() / denominator
        return {
            "count": nx.new_tensor(x.shape[0]),
            "gate_mean": gate_mean,
            "gate_second_moment": gate_second_moment,
            "gate_std": (gate_second_moment - gate_mean.square()).clamp_min(0.0).sqrt(),
            "gate_saturation_fraction": ((g < 0.05) | (g > 0.95)).float().sum() / denominator,
            "cap_active_fraction": (cap < 1.0).float().sum() / denominator,
            "residual_ratio_mean": ratio.sum() / denominator,
            "residual_ratio_max": ratio.amax() if ratio.numel() else ratio.new_zeros(()),
            "angular_shift_mean": shift.sum() / denominator,
            "cosine_distance_mean": cosine_distance.sum() / denominator,
            "perpendicular_fraction_mean": perpendicular_fraction,
            "orthogonal_ratio_mean": orthogonal_ratio,
            "zero_message_fraction": (nm < self.eps).float().sum() / denominator,
        }
