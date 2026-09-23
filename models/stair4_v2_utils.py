"""Static graph utilities and batch supervision for STAIR4-v2.1 BCCR.

Public API
----------
- baseline_whitening(features, d) -> Tensor
- exact_cosine_knn(features, k, block_size) -> LongTensor
- build_train_positive_index(edge_index, n_users, n_items) -> TrainPositiveIndex
- transpose_train_positive_index(index) -> TrainPositiveIndex
- CandidateBatch : NamedTuple
- make_batch_candidates(users, positives) -> CandidateBatch
- build_positive_mask(user_ids, item_ids, index) -> BoolTensor[B_u, B_i]
- validate_static_bundle(A, S, n_users, n_items, d) -> None
- hash_data_manifest(tensors) -> str

Contract
--------
- No dense U×I matrices are ever constructed.
- All graph/index construction runs under torch.no_grad(); no grad_fn survives.
- Only train split data enters the positive index; valid/test IDs must not appear.
- This module does NOT import main_stair_mhd_v3.py or any FreeRec parser/global.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _check_2d_float(t: Tensor, name: str) -> None:
    if not isinstance(t, Tensor) or t.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor, got ndim={getattr(t, 'ndim', '?')}")
    if not t.is_floating_point():
        raise TypeError(f"{name} must be floating-point, got {t.dtype}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains non-finite values")


def _check_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_nonneg_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


# ---------------------------------------------------------------------------
# MI whitening — mirrors the STAIR baseline whitening routine exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_whitening(features: Tensor, d: int) -> Tensor:
    """Center + full SVD + scale by sqrt(I/d).

    Matches the arithmetic in ``main.py`` so that baseline MI is reproduced
    subject to SVD backend determinism.

    Parameters
    ----------
    features : Tensor[I, F]  float32 or float64
        Raw modality features; must have at least *d* columns and *d* rows.
    d : int
        Embedding dimension to retain.

    Returns
    -------
    Tensor[I, d]  same dtype as *features*
    """
    _check_2d_float(features, "features")
    _check_positive_int(d, "d")
    I, F = features.shape
    if min(I, F) < d:
        raise ValueError(
            f"MI requires d <= min(item_count, feature_dim), got d={d}, I={I}, F={F}"
        )
    x = features - features.mean(0, keepdim=True)
    U, _, _ = torch.linalg.svd(x, full_matrices=False)   # U: [I, min(I,F)]
    return U[:, :d] * math.sqrt(I / d)


# ---------------------------------------------------------------------------
# Item kNN graph construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def exact_cosine_knn(features: Tensor, k: int, block_size: int = 256) -> Tensor:
    """Return exact cosine top-k neighbours without allocating an ``I x I`` matrix.

    The calculation is algebraically identical to the baseline dense cosine
    graph for non-tied similarities.  PyTorch does not guarantee a common
    ordering for tied ``topk`` values across devices, therefore each run must
    record the resulting neighbour hash in its manifest.
    """
    _check_2d_float(features, "features")
    _check_positive_int(block_size, "block_size")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < features.shape[0]:
        raise ValueError(f"k must be an integer in [0, {features.shape[0] - 1}], got {k!r}")

    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    if (norms <= 0).any():
        raise ValueError("modality features must not contain zero-norm rows")
    unit_features = features / norms
    n_items = features.shape[0]
    neighbours = torch.empty((n_items, k), dtype=torch.long, device=features.device)
    if k == 0:
        return neighbours

    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        scores = unit_features[start:stop] @ unit_features.T
        rows = torch.arange(stop - start, device=features.device)
        scores[rows, rows + start] = -torch.inf
        neighbours[start:stop] = torch.topk(scores, k, dim=1).indices
    return neighbours


# ---------------------------------------------------------------------------
# Train positive index — O(E log E) build, O(B log E) per-batch lookup
# ---------------------------------------------------------------------------

class TrainPositiveIndex(NamedTuple):
    """Immutable CSR-like index over unique train (user, item) pairs.

    *keys* are int64 = user_id * n_items + item_id, sorted ascending.
    No U×I dense matrix is ever allocated.
    """
    keys: Tensor          # int64[E] sorted unique keys on CPU
    n_users: int
    n_items: int


@torch.no_grad()
def build_train_positive_index(
    edge_index: Tensor,
    n_users: int,
    n_items: int,
) -> TrainPositiveIndex:
    """Build a compact lookup structure for train-positive (u, i) pairs.

    Parameters
    ----------
    edge_index : Tensor[2, E]  int32 or int64
        Train-split user→item edges; may contain duplicates.
    n_users, n_items : int
        Catalog sizes (used for overflow checks only).

    Returns
    -------
    TrainPositiveIndex
    """
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must be int32 or int64")
    _check_positive_int(n_users, "n_users")
    _check_positive_int(n_items, "n_items")

    if edge_index.numel():
        u_min, u_max = int(edge_index[0].min()), int(edge_index[0].max())
        i_min, i_max = int(edge_index[1].min()), int(edge_index[1].max())
        if u_min < 0 or u_max >= n_users:
            raise ValueError(f"edge_index user IDs out of [0, {n_users})")
        if i_min < 0 or i_max >= n_items:
            raise ValueError(f"edge_index item IDs out of [0, {n_items})")

    # Guard int64 overflow: n_users * n_items must fit in int64.
    max_key = (n_users - 1) * n_items + (n_items - 1)
    if max_key > torch.iinfo(torch.int64).max:
        raise OverflowError(
            f"n_users={n_users} * n_items={n_items} overflows int64; "
            "use a chunked implementation for this catalog size"
        )

    ei = edge_index.detach().cpu().to(torch.int64)
    keys = ei[0] * n_items + ei[1]
    keys = keys.unique()  # also sorts
    return TrainPositiveIndex(keys=keys, n_users=n_users, n_items=n_items)


@torch.no_grad()
def transpose_train_positive_index(index: TrainPositiveIndex) -> TrainPositiveIndex:
    """Return the item-to-user lookup index for reverse auxiliary supervision."""
    if not isinstance(index, TrainPositiveIndex):
        raise TypeError("index must be a TrainPositiveIndex")
    users = torch.div(index.keys, index.n_items, rounding_mode="floor")
    items = torch.remainder(index.keys, index.n_items)
    reverse_keys = (items * index.n_users + users).unique()
    return TrainPositiveIndex(
        keys=reverse_keys,
        n_users=index.n_items,
        n_items=index.n_users,
    )


# ---------------------------------------------------------------------------
# CandidateBatch — per-minibatch unique ID reduction
# ---------------------------------------------------------------------------

class CandidateBatch(NamedTuple):
    """Deduplicated (user, item) ID sets from a minibatch.

    Attributes
    ----------
    user_ids : Tensor  int64[B_u]  unique sorted user IDs
    item_ids : Tensor  int64[B_i]  unique sorted item IDs
    user_map : Tensor  int64[B]    maps raw batch user[k] → row in B_u
    item_map : Tensor  int64[B]    maps raw batch item[k] → row in B_i
    """
    user_ids: Tensor    # int64[B_u]
    item_ids: Tensor    # int64[B_i]
    user_map: Tensor    # int64[B]
    item_map: Tensor    # int64[B]


@torch.no_grad()
def make_batch_candidates(
    users: Tensor,
    positives: Tensor,
) -> CandidateBatch:
    """Deduplicate raw minibatch user/item IDs and build reverse maps.

    BPR negative items are intentionally excluded from the candidate set
    so that POCL treats only observed positives as negatives, not BPR noise.

    Parameters
    ----------
    users : Tensor  int64/int32  [B] or [B, 1] raw user IDs from the batch
    positives : Tensor  int64/int32  [B] or [B, 1] or [B, K] positive item IDs

    Returns
    -------
    CandidateBatch
    """
    if not isinstance(users, Tensor) or not isinstance(positives, Tensor):
        raise TypeError("users and positives must be Tensors")
    u = users.reshape(-1).detach().cpu().to(torch.int64)
    i = positives.reshape(-1).detach().cpu().to(torch.int64)
    if u.numel() == 0 or i.numel() == 0:
        raise ValueError("make_batch_candidates requires a non-empty batch")

    user_ids, user_map = torch.unique(u, sorted=True, return_inverse=True)
    item_ids, item_map = torch.unique(i, sorted=True, return_inverse=True)
    return CandidateBatch(
        user_ids=user_ids,
        item_ids=item_ids,
        user_map=user_map,
        item_map=item_map,
    )


# ---------------------------------------------------------------------------
# Positive mask — O(B_u * B_i * log E) via searchsorted; no U×I dense
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_positive_mask(
    user_ids: Tensor,
    item_ids: Tensor,
    index: TrainPositiveIndex,
) -> Tensor:
    """Return Bool[B_u, B_i] where True iff (u, i) is in the train split.

    Uses sorted key lookup (searchsorted) on the prebuilt TrainPositiveIndex.
    No U×I dense matrix is constructed.

    Parameters
    ----------
    user_ids : Tensor  int64[B_u]  unique sorted user IDs (from CandidateBatch)
    item_ids : Tensor  int64[B_i]  unique sorted item IDs (from CandidateBatch)
    index : TrainPositiveIndex

    Returns
    -------
    BoolTensor[B_u, B_i]  (on CPU; transfer to device before use in loss)
    """
    if not isinstance(index, TrainPositiveIndex):
        raise TypeError("index must be a TrainPositiveIndex")
    u = user_ids.detach().cpu().to(torch.int64)
    i = item_ids.detach().cpu().to(torch.int64)
    B_u, B_i = u.numel(), i.numel()
    if B_u == 0 or B_i == 0:
        raise ValueError("user_ids and item_ids must be non-empty")

    # Build candidate keys: B_u × B_i via broadcasting (no U×I full catalog).
    # Memory: 8 * B_u * B_i bytes for int64; for B=4096 this is ~128 MiB which
    # is acceptable (same order as the GEMM output).
    keys = u[:, None] * index.n_items + i[None, :]  # [B_u, B_i] int64

    # searchsorted on sorted unique index.keys
    sorted_keys = index.keys  # [E] sorted
    if sorted_keys.numel() == 0:
        return torch.zeros((B_u, B_i), dtype=torch.bool)
    pos = torch.searchsorted(sorted_keys, keys.reshape(-1))  # [B_u*B_i]
    # Clamp to valid range before indexing
    pos_clamped = pos.clamp(0, sorted_keys.numel() - 1)
    found = sorted_keys[pos_clamped] == keys.reshape(-1)
    return found.reshape(B_u, B_i)


# ---------------------------------------------------------------------------
# Static bundle validation
# ---------------------------------------------------------------------------

def validate_static_bundle(
    A: Tensor,
    S: Tensor,
    n_users: int,
    n_items: int,
    d: int,
    *,
    symmetry_tol: float = 1e-5,
) -> None:
    """Validate shapes, sparsity, symmetry, and finiteness of graph tensors.

    Parameters
    ----------
    A : sparse Tensor  [(U+I) × (U+I)]  normalized user-item adjacency
    S : sparse Tensor  [I × I]  normalized item-item adjacency
    n_users, n_items : int
    d : int  embedding dimension (must be ≥ 2 and even for v2 heads)
    symmetry_tol : float  absolute tolerance for symmetry check
    """
    _check_positive_int(n_users, "n_users")
    _check_positive_int(n_items, "n_items")
    _check_positive_int(d, "d")
    if d % 2 != 0:
        raise ValueError(f"d must be even for STAIR4-v2 Givens blocks, got d={d}")
    if d < 2:
        raise ValueError(f"d must be at least 2, got d={d}")

    # A validation
    if not isinstance(A, Tensor) or A.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise TypeError("A must be a sparse Tensor (COO or CSR)")
    if A.shape != (n_users + n_items, n_users + n_items):
        raise ValueError(
            f"A must have shape [{n_users + n_items}, {n_users + n_items}], got {tuple(A.shape)}"
        )
    if not A.is_floating_point():
        raise TypeError(f"A must be floating-point, got {A.dtype}")
    a_values = A.coalesce().values() if A.layout == torch.sparse_coo else A.values()
    if not torch.isfinite(a_values).all():
        raise ValueError("A contains non-finite values")

    # S validation
    if not isinstance(S, Tensor) or S.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise TypeError("S must be a sparse Tensor (COO or CSR)")
    if S.shape != (n_items, n_items):
        raise ValueError(f"S must have shape [{n_items}, {n_items}], got {tuple(S.shape)}")
    if not S.is_floating_point():
        raise TypeError(f"S must be floating-point, got {S.dtype}")
    s_values = S.coalesce().values() if S.layout == torch.sparse_coo else S.values()
    if not torch.isfinite(s_values).all():
        raise ValueError("S contains non-finite values")

    # Verify symmetry directly on sparse entries.  A dense check here would
    # allocate an I x I matrix for Baby and larger datasets, defeating the
    # memory contract of the graph builder.
    s_coo = S.to_sparse_coo().coalesce()
    rows, cols = s_coo.indices()
    values = s_coo.values()
    forward_keys = rows.to(torch.int64) * n_items + cols.to(torch.int64)
    reverse_keys = cols.to(torch.int64) * n_items + rows.to(torch.int64)
    forward_order = torch.argsort(forward_keys)
    reverse_order = torch.argsort(reverse_keys)
    if not torch.equal(forward_keys[forward_order], reverse_keys[reverse_order]):
        raise ValueError("S is structurally asymmetric")
    max_difference = (values[forward_order] - values[reverse_order]).abs().max().item()
    if max_difference > symmetry_tol:
        raise ValueError(
            f"S is not symmetric within tol={symmetry_tol}; max |S-S^T|={max_difference:.3e}"
        )

    # This is only a coarse sanity check, not a spectral-norm proof.
    row_square_sums = torch.zeros(n_items, dtype=values.dtype, device=values.device)
    row_square_sums.index_add_(0, rows, values.square())
    max_row_norm = row_square_sums.sqrt().max().item()
    if max_row_norm > 2.0:
        raise ValueError(
            "S has rows with very large norms; expected normalized adjacency "
            f"(max row norm = {max_row_norm:.3f})"
        )


# ---------------------------------------------------------------------------
# Data manifest hashing — same algorithm as _tensor_hash in stair_mhd_v3.py
# ---------------------------------------------------------------------------

def hash_data_manifest(tensors: Dict[str, Tensor]) -> str:
    """Return a single hex digest over a mapping of named tensors.

    Each tensor is hashed by (shape, dtype, raw bytes) in sorted key order.
    This mirrors `_tensor_hash` in stair_mhd_v3.py extended to multiple tensors.
    """
    if not tensors:
        raise ValueError("tensors must be non-empty")
    digest = hashlib.sha256()
    for key in sorted(tensors.keys()):
        t = tensors[key]
        if not isinstance(t, Tensor):
            raise TypeError(f"tensors[{key!r}] must be a Tensor, got {type(t).__name__}")
        t_cpu = t.detach().cpu().contiguous()
        header = f"{key}:{tuple(t_cpu.shape)}:{t_cpu.dtype}".encode()
        digest.update(header)
        digest.update(t_cpu.numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint utilities (atomic write, plain-dict conversion)
# ---------------------------------------------------------------------------

def _to_plain(obj):
    """Recursively convert defaultdicts/mappings to plain dict."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_plain(v) for v in obj]
        return type(obj)(converted)
    return obj


def save_checkpoint_atomic(path: Union[Path, str], payload: dict) -> None:
    """Write payload to a tmp file then rename; avoids partial writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(_to_plain(payload), str(tmp))
    tmp.replace(path)


def load_checkpoint_checked(path: Union[Path, str],
                             expected_manifest: Optional[dict] = None) -> dict:
    """Load checkpoint; validate manifest keys if provided."""
    path = Path(path)
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        ckpt = torch.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint must contain a dictionary, got {type(ckpt).__name__}")
    if expected_manifest:
        for key, val in expected_manifest.items():
            if ckpt.get(key) != val:
                raise ValueError(
                    f"checkpoint mismatch: {key!r} expected {val!r}, "
                    f"got {ckpt.get(key)!r}"
                )
    return ckpt


# ---------------------------------------------------------------------------
# STAIR4-v2.1 BCCR Mathematical Utilities (§6.2, §6.4 of STAIR4_v2_1_Report.md)
# ---------------------------------------------------------------------------

def check_view_validity(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Return BoolTensor[B] where True iff ||x||_2 >= eps.

    Per §6.2 and §6.5 of STAIR4_v2_1_Report.md: Near-zero views are excluded
    from auxiliary supervision while remaining in BPR.
    """
    _check_2d_float(x, "x")
    norms = torch.linalg.norm(x, ord=2, dim=-1)
    return norms >= eps


def bounded_phase_encoder(
    x: Tensor,
    kappa: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Map real embeddings to bounded complex phase features ψ(x) with unit row norm.

    r(x) = x / max(||x||_2, eps)
    ψ_k(x) = (1 / sqrt(d)) * exp(i * kappa * sqrt(d) * r_k(x))
           = (1 / sqrt(d)) * [cos(kappa * sqrt(d) * r_k(x)) + i * sin(kappa * sqrt(d) * r_k(x))]

    Guarantees:
    ||ψ(x)||_2^2 = sum_k (cos^2 + sin^2) / d = sum_k (1 / d) = 1.0.

    Parameters
    ----------
    x : Tensor[B, d]  real float32
    kappa : float  phase scaling coefficient (pilot default 0.5)
    eps : float  denominator stabilizer for radial normalization

    Returns
    -------
    (psi_real, psi_imag) : Tuple[Tensor[B, d], Tensor[B, d]]
    """
    _check_2d_float(x, "x")
    B, d = x.shape
    norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    r = x / norm  # [B, d] bounded in unit ball
    angle = (kappa * math.sqrt(d)) * r  # [B, d]
    inv_sqrt_d = 1.0 / math.sqrt(d)
    psi_real = torch.cos(angle) * inv_sqrt_d
    psi_imag = torch.sin(angle) * inv_sqrt_d
    return psi_real, psi_imag


def bounded_real_encoder(
    x: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Radial L2-projection for real-space cosine control (Ablation C0).

    r(x) = x / max(||x||_2, eps)
    """
    _check_2d_float(x, "x")
    norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def complex_hybrid_similarity(
    q_real: Tensor,
    q_imag: Tensor,
    k_real: Tensor,
    k_imag: Tensor,
    eta: float = 0.25,
    temperature: float = 0.2,
) -> Tensor:
    """Compute hybrid complex similarity matrix using 2-D GEMM operations.

    Inner product: h(q, k) = q^H k = (q_r - i q_i)^T (k_r + i k_i)
    Re(h) = q_r k_r^T + q_i k_i^T
    Im(h) = q_r k_i^T - q_i k_r^T
    |h|^2 = (Re h)^2 + (Im h)^2

    Hybrid similarity kernel (§6.4):
    s_eta(q, k) = (1 - eta) * Re(h) + eta * |h|^2
    logits = s_eta / temperature

    Never materializes 3-D [B_q, B_k, d] tensors.
    """
    _check_2d_float(q_real, "q_real")
    _check_2d_float(q_imag, "q_imag")
    _check_2d_float(k_real, "k_real")
    _check_2d_float(k_imag, "k_imag")
    if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and in [0, 1]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if q_real.shape != q_imag.shape or k_real.shape != k_imag.shape:
        raise ValueError("real and imaginary tensors must have matching shapes")
    if q_real.shape[1] != k_real.shape[1]:
        raise ValueError("query and key tensors must have the same feature dimension")
    if q_real.device != q_imag.device or k_real.device != k_imag.device:
        raise ValueError("real and imaginary tensors must be on the same device")
    if q_real.device != k_real.device:
        raise ValueError("query and key tensors must be on the same device")

    # Real and Imaginary parts of inner product via GEMM
    re_h = torch.mm(q_real, k_real.T) + torch.mm(q_imag, k_imag.T)
    im_h = torch.mm(q_real, k_imag.T) - torch.mm(q_imag, k_real.T)
    mod_sq = re_h.square() + im_h.square()

    # Hybrid blend: eta=0 -> pure signed overlap; eta=1 -> pure squared fidelity
    s = (1.0 - eta) * re_h + eta * mod_sq
    return s / temperature
