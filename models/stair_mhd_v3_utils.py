"""Static construction and differentiable incidence operations for STAIR-MHD v3.

No routine builds an item-by-item propagation matrix. Behavioral statistics
must receive the training split only; that provenance is the caller's contract.
"""

from itertools import combinations
import math
from numbers import Integral

import torch
from torch import Tensor


def _integer_matrix(value: Tensor, name: str, rows=None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must contain int32 or int64 indices")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")


def _neighbors(neighbors: Tensor) -> int:
    _integer_matrix(neighbors, "neighbors")
    n = neighbors.shape[0]
    if n == 0:
        raise ValueError("neighbors must describe at least one item")
    if neighbors.numel() and ((neighbors < 0).any() or (neighbors >= n).any()):
        raise ValueError("neighbors contain an item index outside [0, N)")
    return n


def _features(features: Tensor) -> None:
    if not isinstance(features, Tensor) or features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("features must have nonempty shape [N, d]")
    if features.layout != torch.strided or features.dtype not in (torch.float32, torch.float64):
        raise TypeError("features must be dense float32 or float64 tensors")
    if not torch.isfinite(features).all():
        raise ValueError("features contain nonfinite values")


@torch.no_grad()
def build_incidence(neighbors: Tensor) -> Tensor:
    """Return binary sparse COO H[i, c] for {c} union neighbors[c].

    Duplicate neighbors and a listed center are accepted and deduplicated.
    H uses float32 values on the neighbors' device; callers may cast its dtype.
    """
    n = _neighbors(neighbors)
    centers = torch.arange(n, device=neighbors.device, dtype=torch.long)
    members = torch.cat((centers[:, None], neighbors.long()), dim=1)
    columns = centers[:, None].expand_as(members)
    indices = torch.stack((members.reshape(-1), columns.reshape(-1)))
    h = torch.sparse_coo_tensor(
        indices, torch.ones(indices.shape[1], device=neighbors.device), (n, n)
    ).coalesce()
    # Coalescing sums duplicates, whereas incidence is a binary membership.
    return torch.sparse_coo_tensor(
        h.indices(), torch.ones_like(h.values()), h.shape, device=h.device
    ).coalesce()


@torch.no_grad()
def exact_knn(features: Tensor, k: int, block_size: int = 256) -> Tensor:
    """Exact cosine top-k, excluding self, using strictly sub-N row blocks.

    Tied similarities follow torch.topk's device-specific tie ordering. Zero
    feature rows are rejected because no missing-modality policy is assumed.
    """
    _features(features)
    n = features.shape[0]
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k < n:
        raise ValueError("k must be an integer in [0, N-1]")
    if isinstance(block_size, bool) or not isinstance(block_size, Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("modality features must have finite, nonzero row norms")
    result = torch.empty((n, k), dtype=torch.long, device=features.device)
    if k == 0:
        return result
    unit = features / norms
    width = min(int(block_size), n - 1)
    for start in range(0, n, width):
        end = min(start + width, n)
        scores = unit[start:end] @ unit.T
        rows = torch.arange(end - start, device=features.device)
        scores[rows, rows + start] = -torch.inf
        result[start:end] = scores.topk(k, dim=1).indices
    return result


@torch.no_grad()
def behavioral_statistics(
    edge_index: Tensor,
    num_users: int,
    num_items: int,
    neighbors: Tensor,
    features: Tensor,
    support_s: float = 10,
):
    """Return train-only (C, rho, standardized x), one row per hyperedge.

    User sets deduplicate interactions. Intersections are memoized only for
    unordered pairs that occur in an actual hyperedge, including leaf pairs.
    x columns are center-neighbor cosine mean/population std and member
    log1p(train degree) mean/min. Standardization uses population std over all
    catalog hyperedges and sets constant columns to zero.
    """
    _integer_matrix(edge_index, "edge_index", rows=2)
    n = _neighbors(neighbors)
    _features(features)
    if isinstance(num_users, bool) or not isinstance(num_users, Integral) or num_users < 0:
        raise ValueError("num_users must be a nonnegative integer")
    if isinstance(num_items, bool) or not isinstance(num_items, Integral) or num_items != n:
        raise ValueError("num_items must equal the number of neighbor rows")
    if features.shape[0] != n:
        raise ValueError("features and neighbors must describe the same catalog")
    if not math.isfinite(float(support_s)) or support_s <= 0:
        raise ValueError("support_s must be finite and positive")
    if edge_index.numel():
        if (edge_index < 0).any() or (edge_index[0] >= num_users).any() or (edge_index[1] >= n).any():
            raise ValueError("train edge_index contains an out-of-range user or item")

    users = [set() for _ in range(n)]
    for user, item in edge_index.detach().cpu().T.tolist():
        users[item].add(user)
    counts = [len(item_users) for item_users in users]
    log_degree = [math.log1p(count) for count in counts]
    # Static preprocessing stays on CPU, avoiding one GPU synchronization per
    # hyperedge and an incidence-sized feature gather on the accelerator.
    raw_features = features.detach().cpu().double()
    norms = torch.linalg.vector_norm(raw_features, dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("modality features must have finite, nonzero row norms")
    unit = raw_features / norms
    pair_cache = {}
    coherence, support, rows = [], [], []
    eps = 1e-12
    for center, candidates in enumerate(neighbors.detach().cpu().tolist()):
        members = sorted({center, *candidates})
        weighted_overlap = support_sum = 0.0
        pair_count = 0
        for a, b in combinations(members, 2):
            pair = (a, b)
            if pair not in pair_cache:
                minimum = min(counts[a], counts[b])
                reliability = minimum / (minimum + float(support_s))
                overlap = len(users[a].intersection(users[b])) / (math.sqrt(counts[a] * counts[b]) + eps)
                pair_cache[pair] = (reliability * overlap, reliability)
            contribution, reliability = pair_cache[pair]
            weighted_overlap += contribution
            support_sum += reliability
            pair_count += 1
        coherence.append(weighted_overlap / (support_sum + eps))
        support.append(support_sum / pair_count if pair_count else 0.0)
        noncenter = [item for item in members if item != center]
        if noncenter:
            cosine = (unit[noncenter] @ unit[center]).clamp(-1, 1)
            cosine_mean, cosine_std = cosine.mean().item(), cosine.std(unbiased=False).item()
        else:
            cosine_mean = cosine_std = 0.0
        degrees = [log_degree[item] for item in members]
        rows.append((cosine_mean, cosine_std, sum(degrees) / len(degrees), min(degrees)))
    x = torch.tensor(rows, dtype=torch.float64)
    scale = x.std(dim=0, unbiased=False)
    x = (x - x.mean(dim=0)) / torch.where(scale > eps, scale, torch.ones_like(scale))
    x[:, scale <= eps] = 0
    options = dict(dtype=features.dtype, device=features.device)
    return torch.tensor(coherence, **options), torch.tensor(support, **options), x.to(**options)


def _incidence(h: Tensor) -> None:
    if not isinstance(h, Tensor) or h.layout != torch.sparse_coo or h.ndim != 2:
        raise TypeError("incidence must be a sparse COO matrix")
    if not h.is_coalesced():
        raise ValueError("incidence must be coalesced")
    if h.shape[0] < 1 or h.shape[1] < 1:
        raise ValueError("incidence must be nonempty")
    if h.dtype not in (torch.float32, torch.float64):
        raise TypeError("incidence values must be float32 or float64")
    if h.requires_grad or not torch.all(h.values() == 1):
        raise ValueError("incidence must have fixed binary membership values")
    indices = h.indices()
    if (indices < 0).any() or (indices[0] >= h.shape[0]).any() or (indices[1] >= h.shape[1]).any():
        raise ValueError("incidence contains out-of-range indices")


def _positive_vector(value: Tensor, h: Tensor, size: int, name: str) -> None:
    if not isinstance(value, Tensor) or value.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    if value.device != h.device or value.dtype != h.dtype or value.layout != torch.strided:
        raise ValueError(f"{name} must be dense and match incidence dtype/device")
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError(f"{name} must be finite and strictly positive")


def graph_state(H: Tensor, w: Tensor) -> dict:
    """Compute a fresh differentiable weighted degree; never cache this state."""
    _incidence(H)
    _positive_vector(w, H, H.shape[1], "weights")
    degree = torch.sparse.mm(H, w[:, None]).squeeze(1)
    _positive_vector(degree, H, H.shape[0], "weighted node degree")
    return {"weights": w, "degree": degree}


def apply_P_H(X: Tensor, incidences: tuple, states: tuple, alpha: Tensor) -> Tensor:
    """Apply sum_m alpha_m Dv^-1/2 H W De^-1 H.T Dv^-1/2 X.

    Differentiation includes both weights and weighted degrees. Only sparse
    incidence times dense feature matrices are used, never sparse-sparse or N².
    """
    _features(X)
    if not incidences or len(incidences) != len(states):
        raise ValueError("incidences and states must have the same nonzero length")
    if not isinstance(alpha, Tensor) or alpha.shape != (len(incidences),):
        raise ValueError("alpha must contain one coefficient per modality")
    if alpha.device != X.device or alpha.dtype != X.dtype:
        raise ValueError("alpha and X must have the same dtype/device")
    if not torch.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError("alpha must contain finite nonnegative coefficients")
    if not torch.isclose(alpha.sum(), alpha.new_tensor(1.0), rtol=1e-5, atol=1e-7):
        raise ValueError("alpha must sum to one")
    result = torch.zeros_like(X)
    for coefficient, h, state in zip(alpha, incidences, states):
        _incidence(h)
        if h.shape[0] != X.shape[0] or h.dtype != X.dtype or h.device != X.device:
            raise ValueError("incidence and X must match node count, dtype, and device")
        weights, degree = state["weights"], state["degree"]
        _positive_vector(weights, h, h.shape[1], "weights")
        _positive_vector(degree, h, h.shape[0], "weighted node degree")
        ht = h.transpose(0, 1).coalesce()
        cardinality = torch.sparse.mm(ht, X.new_ones((X.shape[0], 1))).squeeze(1)
        _positive_vector(cardinality, h, h.shape[1], "hyperedge cardinality")
        inv_sqrt = degree.rsqrt()[:, None]
        edge_features = torch.sparse.mm(ht, X * inv_sqrt)
        node_features = torch.sparse.mm(h, edge_features * (weights / cardinality)[:, None])
        result = result + coefficient * node_features * inv_sqrt
    return result


def detach_all(value):
    """Recursively make an independent, graph-free tensor/container snapshot."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: detach_all(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_all(item) for item in value)
    if isinstance(value, list):
        return [detach_all(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")
