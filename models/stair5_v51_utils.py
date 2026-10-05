"""models/stair5_v51_utils.py — STAIR5-v5.1 Utilities.
=======================================================
Train-pair membership index, GPU lookup, ID validation, diagnostics,
and checkpoint helpers for NLGCL-KPE.

All membership operations use train-only data. No validation/test labels
are used during training. Membership buffer is a nonlearnable, nonpersistent
sorted int64 tensor of (u * n_items + i) pair keys.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hashing / I/O helpers (mirror stair5_v4_utils interface)
# ---------------------------------------------------------------------------

def file_sha256(path: Union[str, Path]) -> str:
    """Stream a complete file digest without loading large feature files twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Union[str, Path]) -> None:
    """Saves a torch object atomically via a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint_", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ---------------------------------------------------------------------------
# RNG state capture / restore (identical API to v4)
# ---------------------------------------------------------------------------

def rng_state() -> dict:
    """Captures full RNG state across Python, NumPy, CPU Torch, and CUDA."""
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    """Restores full RNG state across Python, NumPy, CPU Torch, and CUDA."""
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


# ---------------------------------------------------------------------------
# Optimizer checkpoint helpers
# ---------------------------------------------------------------------------

def optimizer_state_for_checkpoint(optimizer: torch.optim.Optimizer) -> dict:
    """Excludes non-serializable smoother callbacks from the saved optimizer state."""
    state = optimizer.state_dict()
    state["param_groups"] = [
        {k: v for k, v in g.items() if k != "smoother"}
        for g in state["param_groups"]
    ]
    return state


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    """Restores optimizer state while preserving the model's active smoother callbacks."""
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer group count differs from current model.")
    if any(len(saved["params"]) != len(live["params"])
           for saved, live in zip(state["param_groups"], optimizer.param_groups)):
        raise ValueError("Checkpoint optimizer parameter layout differs from current model.")
    runtime_smoothers = [g.get("smoother") for g in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, smoother in zip(optimizer.param_groups, runtime_smoothers):
        group["smoother"] = smoother


# ---------------------------------------------------------------------------
# Training checkpoint save / load
# ---------------------------------------------------------------------------

def save_training_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    atomic_torch_save(
        {
            "version": 51,
            "model": model.state_dict(),
            "optimizer": optimizer_state_for_checkpoint(optimizer),
            "epoch": int(epoch),
            "rng": rng_state(),
            "extra": extra or {},
        },
        path,
    )


def load_training_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    restore_random: bool = True,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 51:
        raise ValueError("Unsupported STAIR5-v5.1 training checkpoint version.")
    if payload["model"].get("_extra_state") != model.get_extra_state():
        raise ValueError("Checkpoint model/configuration/data provenance mismatch.")
    if payload.get("epoch", -1) < 0:
        raise ValueError("Checkpoint epoch must be nonnegative.")
    model.load_state_dict(payload["model"])
    load_optimizer_state(optimizer, payload["optimizer"])
    if restore_random and "rng" in payload:
        restore_rng(payload["rng"])
    return payload


# ---------------------------------------------------------------------------
# Train-pair membership index
# ---------------------------------------------------------------------------

def build_train_pair_index(train_edges: torch.Tensor, n_users: int, n_items: int) -> torch.Tensor:
    """Build a sorted int64 tensor of unique train pair keys: u * n_items + i.

    Args:
        train_edges: [2, E] integer tensor (rows: user IDs, cols: item IDs).
        n_users: Total user count (for range check).
        n_items: Total item count (for range check and key encoding).

    Returns:
        1D int64 sorted unique pair keys.
    """
    if n_users <= 0 or n_items <= 0:
        raise ValueError("Entity counts must be positive.")
    overflow = int(n_users) * int(n_items)
    if overflow > (2**63 - 1):
        raise ValueError("n_users * n_items overflows int64; reduce catalog size.")
    if train_edges.ndim != 2 or train_edges.shape[0] != 2:
        raise ValueError("train_edges must be [2, E].")
    edges = train_edges.detach().cpu().long()
    if edges.numel():
        if edges[0].min() < 0 or edges[0].max() >= n_users:
            raise ValueError("User IDs in train_edges are out of range.")
        if edges[1].min() < 0 or edges[1].max() >= n_items:
            raise ValueError("Item IDs in train_edges are out of range.")
    flat = torch.unique(edges[0] * n_items + edges[1], sorted=True)
    return flat


def build_train_pair_index_fingerprint(
    train_pair_index: torch.Tensor, n_users: int, n_items: int
) -> str:
    """Content-addressed fingerprint for the train membership index."""
    h = hashlib.sha256()
    h.update(f"v51,n_users={n_users},n_items={n_items}".encode())
    arr = train_pair_index.numpy().astype(np.int64)
    h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# GPU membership lookup (searchsorted, no Python loops or .item() calls)
# ---------------------------------------------------------------------------

def gpu_membership(
    train_index: torch.Tensor,
    query_keys: torch.Tensor,
) -> torch.Tensor:
    """GPU-resident binary search for train membership.

    Args:
        train_index: 1D sorted int64 tensor (unique train pair keys).
        query_keys: N-dimensional int64 tensor of pair keys to probe.

    Returns:
        Boolean tensor of the same shape as query_keys; True iff the key is
        in the training set.
    """
    if train_index.numel() == 0:
        return torch.zeros_like(query_keys, dtype=torch.bool)
    flat = query_keys.reshape(-1).contiguous()
    pos = torch.searchsorted(train_index, flat)
    # Clamp positions to valid range before indexing.
    pos_clamped = pos.clamp(max=train_index.numel() - 1)
    found = train_index[pos_clamped] == flat
    # Positions equal to numel() mean the key is past the end; mark as not found.
    found = found & (pos < train_index.numel())
    return found.reshape(query_keys.shape)


# ---------------------------------------------------------------------------
# Chunked membership for large B x m tensors (avoids allocating B x m int64)
# ---------------------------------------------------------------------------

def chunked_membership(
    train_index: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    n_items: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Compute membership for all (user_ids[b], item_ids[k]) pairs in chunks.

    Args:
        train_index: 1D sorted int64 train-pair keys on the same device.
        user_ids: (B,) int64 user indices.
        item_ids: (m,) int64 item indices.
        n_items: Catalog size, used to encode pair keys.
        chunk_size: Number of query rows per chunk.

    Returns:
        Boolean (B, m) mask; True means (user_ids[b], item_ids[k]) is a train pair.
    """
    B, m = user_ids.size(0), item_ids.size(0)
    result = torch.empty(B, m, dtype=torch.bool, device=user_ids.device)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        u_chunk = user_ids[start:end]  # (C,)
        keys = u_chunk[:, None] * n_items + item_ids[None, :]  # (C, m)
        result[start:end] = gpu_membership(train_index, keys)
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_kpe_diagnostics(
    train_index: torch.Tensor,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    n_items: int,
    chunk_size: int = 512,
) -> Dict:
    """Compute collision statistics for a sampled batch.

    Returns a dict summarising unique-key fractions, additional-positive
    counts, and expected zero-negative fraction — all computed without
    accessing validation or test labels.
    """
    B = user_ids.size(0)
    unique_users = user_ids.unique()
    unique_items = item_ids.unique()
    m_u = unique_users.size(0)
    m_i = unique_items.size(0)

    # Item-side: for each batch query (user), check which unique item keys are known positives.
    # Shape: (B, m_i)
    mask_i = chunked_membership(train_index, user_ids, unique_items, n_items, chunk_size)
    # target_idx_i[b] = index of item_ids[b] in unique_items
    _, target_idx_i = (unique_items[None, :] == item_ids[:, None]).max(dim=1)
    # additional positives per query: total positives minus the designated one
    additional_i = mask_i.sum(dim=1).float() - 1.0
    admissible_neg_i = (m_i - mask_i.sum(dim=1)).float()

    # User-side: for each batch query (item), check which unique user keys are known positives.
    mask_u = chunked_membership(train_index, unique_users, item_ids, n_items, chunk_size).T  # (B, m_u)
    additional_u = mask_u.sum(dim=1).float() - 1.0
    admissible_neg_u = (m_u - mask_u.sum(dim=1)).float()

    def _stats(x: torch.Tensor) -> dict:
        x = x.cpu()
        return {
            "mean": float(x.mean()),
            "p10": float(x.quantile(0.10)),
            "median": float(x.median()),
            "p90": float(x.quantile(0.90)),
            "zero_fraction": float((x <= 0).float().mean()),
        }

    return {
        "batch_size": B,
        "unique_users": m_u,
        "unique_items": m_i,
        "unique_user_fraction": m_u / max(B, 1),
        "unique_item_fraction": m_i / max(B, 1),
        "item_side_additional_positives": _stats(additional_i),
        "user_side_additional_positives": _stats(additional_u),
        "item_side_admissible_negatives": _stats(admissible_neg_i),
        "user_side_admissible_negatives": _stats(admissible_neg_u),
    }


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def parse_telemetry_jsonl(artifact_dir: Union[str, Path]) -> List[dict]:
    """Reads ordered training telemetry records from training_telemetry.jsonl."""
    path = Path(artifact_dir) / "training_telemetry.jsonl"
    if not path.is_file():
        return []
    records = []
    seen_epochs = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                epoch = data.get("epoch")
                if epoch not in seen_epochs:
                    seen_epochs.add(epoch)
                    records.append(data)
            except Exception:
                continue
    return records
