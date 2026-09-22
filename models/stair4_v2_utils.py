"""Static graph utilities and batch supervision for STAIR4-v2 BSF–POCL.

Public API
----------
- baseline_whitening(features, d) -> Tensor
- build_train_positive_index(edge_index, n_users, n_items) -> TrainPositiveIndex
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
from typing import Dict, NamedTuple, Optional, Sequence

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
# MI whitening — mirrors STAIR_MHD_v3.whitening() exactly
# ---------------------------------------------------------------------------

@torch.no_grad()
def baseline_whitening(features: Tensor, d: int) -> Tensor:
    """Center + full SVD + scale by sqrt(I/d).

    Matches the arithmetic in STAIR_MHD_v3.whitening() so that baseline MI
    is reproduced bitwise (subject to SVD backend determinism).

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

    # Symmetry check via dense probe on small graphs; skip for large catalogs
    if n_items <= 8192:
        S_dense = S.to_dense()
        diff = (S_dense - S_dense.T).abs().max().item()
        if diff > symmetry_tol:
            raise ValueError(
                f"S is not symmetric within tol={symmetry_tol}; max |S-S^T|={diff:.3e}"
            )
        if not torch.isfinite(S_dense).all():
            raise ValueError("S contains non-finite values")
        # Spectral norm proxy: Frobenius norm of each row should be ≤ 1 for
        # normalized adjacency (not a tight bound but catches obvious errors).
        row_norms = S_dense.norm(dim=1)
        if row_norms.max().item() > 2.0:
            raise ValueError(
                "S has rows with very large norms; expected normalized adjacency "
                f"(max row norm = {row_norms.max().item():.3f})"
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
