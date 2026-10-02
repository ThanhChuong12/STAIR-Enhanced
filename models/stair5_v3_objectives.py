"""Bidirectional multi-positive CE with a shared, fixed-temperature critic for STAIR5-v3."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def remove_self_return(h1_target, source, normalized_edge_weight, beta, eps=1e-8):
    """Subtract the exact original normalized edge contribution; no fallback."""
    corrected = h1_target - source * normalized_edge_weight.reshape(-1, 1) * beta
    return corrected, corrected.norm(dim=-1) > eps


def spectral_reweight(features, beta, eps_beta=0.05, max_norm=None):
    result = features / beta.clamp_min(eps_beta)
    if max_norm is not None:
        result = result * (torch.as_tensor(max_norm, device=result.device) /
                           result.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
    return result


def multi_positive_ce(logits, positives):
    """Uniform positive-target CE; rows without positives do not contribute."""
    if logits.shape != positives.shape or logits.ndim != 2:
        raise ValueError("Logits and positive mask must be matching matrices.")
    if logits.numel() == 0:
        return logits.sum() * 0
    positive = positives.to(logits.dtype)
    counts = positive.sum(-1)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    per_query = -(positive * log_probs).sum(-1) / counts.clamp_min(1)
    valid = counts > 0
    return (per_query * valid).sum() / valid.sum().clamp_min(1)


class AuditedContrastiveLoss(nn.Module):
    """U0→I1 and I0→U1 CE, without label-dependent key correction."""
    def __init__(self, geo, radius_cap=2.0, tau=0.3, eps_beta=0.05, disable_reweight=False):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(radius_cap) or radius_cap <= 0:
            raise ValueError("Loss temperature and radius must be positive and finite.")
        if not math.isfinite(eps_beta) or eps_beta <= 0:
            raise ValueError("eps_beta must be positive and finite.")
        self.geo, self.radius_cap, self.tau = geo, radius_cap, tau
        self.eps_beta, self.disable_reweight = eps_beta, disable_reweight
        self.chunk_size = 256

    def _direction(self, queries, keys, mask, scale):
        """Checkpoint query blocks; keep all candidates in each CE denominator."""
        zero = (queries.sum() + keys.sum()) * 0
        if not len(queries) or not len(keys):
            return zero, 0.0, 0.0
        sums, valid_rows = [], 0
        alignment_sum, positive_count, negative_count = queries.new_zeros(()), 0, 0
        negative_logsum = None
        counts = mask.sum(-1).detach().cpu()
        for start in range(0, len(queries), self.chunk_size):
            target = mask[start:start + self.chunk_size]
            block_counts = counts[start:start + self.chunk_size]
            valid_count = int((block_counts > 0).sum())
            def block(q, k, positives):
                distance = self.geo.compute_distance_matrix(q, k)
                ce = multi_positive_ce(-distance / scale, positives)
                with torch.no_grad():
                    pos = positives > 0
                    alignment = (distance * pos).sum()
                    negatives = distance[~pos]
                    uniformity = torch.logsumexp(-2 * negatives, 0) if negatives.numel() else distance.new_tensor(-float("inf"))
                return ce, alignment, uniformity
            query = queries[start:start + self.chunk_size]
            if torch.is_grad_enabled() and (query.requires_grad or keys.requires_grad):
                ce, align, uniform = checkpoint(block, query, keys, target, use_reentrant=False)
            else:
                ce, align, uniform = block(query, keys, target)
            sums.append(ce * valid_count)
            valid_rows += valid_count
            alignment_sum = alignment_sum + align.detach()
            block_positives = int(block_counts.sum())
            positive_count += block_positives
            negative_count += target.numel() - block_positives
            negative_logsum = uniform if negative_logsum is None else torch.logaddexp(negative_logsum, uniform)
        loss = torch.stack(sums).sum() / max(valid_rows, 1)
        uniformity = float(negative_logsum) - math.log(negative_count) if negative_count else 0.0
        return loss, float(alignment_sum) / max(positive_count, 1), uniformity

    def forward(self, u_0, i_0, u_1, i_1, P, deg_u=None, deg_i=None, beta=None,
                q_u0=1.0, q_i0=1.0, q_u1=1.0, q_i1=1.0, M_norm=10.0):
        if P.shape != (len(u_0), len(i_0)):
            raise ValueError("Positive mask must be [unique users, unique items].")
        with torch.autocast(device_type=u_0.device.type, enabled=False):
            if not self.disable_reweight:
                u_1 = spectral_reweight(u_1.float(), beta.float(), self.eps_beta, M_norm)
                i_1 = spectral_reweight(i_1.float(), beta.float(), self.eps_beta, M_norm)
            u0 = self.geo.cap_radius(u_0.float() / max(float(q_u0), 1e-4))
            i0 = self.geo.cap_radius(i_0.float() / max(float(q_i0), 1e-4))
            u1 = self.geo.cap_radius(u_1.float() / max(float(q_u1), 1e-4))
            i1 = self.geo.cap_radius(i_1.float() / max(float(q_i1), 1e-4))
            scale = 2 * self.radius_cap**2 * self.tau
            loss1, align, unif = self._direction(u0, i1, P, scale)
            loss2, _, _ = self._direction(i0, u1, P.t(), scale)
            return (loss1 + loss2) / 2, {"alignment": align, "uniformity": unif}
