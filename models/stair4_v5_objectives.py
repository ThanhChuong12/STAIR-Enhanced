"""Direct sampled ranking objective for STAIR-RAM, report equation 11."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def masked_candidate_cross_entropy(
    scores: Tensor,
    negative_mask: Tensor | None = None,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """Classify the positive in column zero against valid sampled negatives.

    The boolean mask covers negative columns only. Rows without any valid
    negative contribute zero and are excluded from the mean denominator. An
    all-empty batch returns a differentiable zero. The caller must also skip
    its optimizer step for such a batch, because AdamW decay/moments can move
    parameters even when this loss has zero gradients.

    For microbatch accumulation, use reduction='sum' and divide each chunk by
    the total number of eligible rows in the entire effective batch.
    """
    if scores.ndim != 2 or scores.size(1) < 1 or not scores.is_floating_point():
        raise ValueError("scores must be a floating-point [batch, 1 + negatives] tensor.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be none, sum, or mean.")
    if negative_mask is None:
        negative_mask = torch.ones_like(scores[:, 1:], dtype=torch.bool)
    if negative_mask.shape != scores[:, 1:].shape or negative_mask.dtype != torch.bool:
        raise ValueError("negative_mask must be boolean with shape [batch, negatives].")
    if negative_mask.device != scores.device:
        raise ValueError("negative_mask and scores must reside on the same device.")

    eligible = negative_mask.any(dim=-1)
    logits = scores / temperature
    negative_logits = logits[:, 1:].masked_fill(~negative_mask, -torch.inf)
    # Use log(1 + sum exp(negative - positive)) in a logsumexp formulation.
    # Including a zero avoids cancellation between large positive scores.
    differences = negative_logits - logits[:, :1]
    rows = torch.logsumexp(torch.cat((torch.zeros_like(logits[:, :1]), differences), dim=1), dim=1)
    rows = torch.where(eligible, rows, torch.zeros_like(rows))
    if reduction == "none":
        return rows
    total = rows.sum()
    return total if reduction == "sum" else total / eligible.sum().clamp_min(1)

