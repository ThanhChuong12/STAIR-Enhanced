"""models/stair5_v7_utils.py — Checkpointing, Provenance & Diagnostics for STAIR5-v7.
===================================================================================
Provides atomic checkpointing, cryptographic file hashes, sparse fingerprints,
RNG state restoration, and memory profiling gates for STAIR5-v7 (UCR-D).
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch


def file_sha256(path: Union[str, Path]) -> str:
    """Computes SHA-256 digest of a local file in 1 MiB chunks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_torch_save(obj: Any, destination: Union[str, Path]) -> None:
    """Atomic write via temporary file replacement to prevent corrupt checkpoints."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix="atomic_v7_", suffix=".tmp", delete=False
    ) as tmp:
        tmp_name = tmp.name
    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, destination)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _digest_array(hasher: Any, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array)
    hasher.update(str((array.shape, array.dtype.str)).encode())
    hasher.update(memoryview(array).cast("B"))


def sparse_fingerprint(matrix: sp.csr_matrix) -> str:
    """Hash canonical CSR shape, indices and data without dense materialization."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    h = hashlib.sha256(str(matrix.shape).encode())
    for array in (
        matrix.indptr.astype(np.int64),
        matrix.indices.astype(np.int64),
        matrix.data.astype(np.float32),
    ):
        _digest_array(h, array)
    return h.hexdigest()


def validate_csr_operator(operator: torch.Tensor, name: str = "CSR Operator") -> None:
    """Strictly validates CSR tensor topology and finite float values."""
    if operator.layout != torch.sparse_csr:
        raise ValueError(f"{name} must have torch.sparse_csr layout; got {operator.layout}")
    if operator.ndim != 2 or operator.size(0) != operator.size(1):
        raise ValueError(f"{name} must be a square 2D matrix; got shape {tuple(operator.shape)}")
    values = operator.values()
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Infinite values.")


def capture_rng() -> Dict[str, Any]:
    """Captures CPU, CUDA and NumPy RNG states."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    """Restores CPU, CUDA and NumPy RNG states exactly."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_state_for_checkpoint(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Excludes non-serializable smoother callbacks from the saved optimizer state."""
    state = optimizer.state_dict()
    state["param_groups"] = [
        {k: v for k, v in g.items() if k != "smoother"}
        for g in state["param_groups"]
    ]
    return state


def load_optimizer_state(optimizer: torch.optim.Optimizer, state: Dict[str, Any]) -> None:
    """Restores optimizer state while preserving the model's active smoother callbacks."""
    runtime_smoothers = [g.get("smoother") for g in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, smoother in zip(optimizer.param_groups, runtime_smoothers):
        group["smoother"] = smoother


def save_training_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a complete training checkpoint with model weights, Adam moments, and RNG."""
    payload = {
        "version": "7.0",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer_state_for_checkpoint(optimizer),
        "rng": capture_rng(),
        "extra": extra or {},
        "timestamp": time.time(),
    }
    atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    restore_random: bool = True,
) -> Dict[str, Any]:
    """Restores model weights, Adam moments, and optionally RNG state."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    load_optimizer_state(optimizer, payload["optimizer"])
    if restore_random and "rng" in payload:
        restore_rng(payload["rng"])
    return payload


def check_electronics_vram_limit(
    device: torch.device,
    dataset_name: str,
    threshold_mib: float = 800.0,
    strict: bool = False,
) -> float:
    """Checks peak allocated VRAM against the 800 MiB Electronics execution gate.

    Args:
        device: Active torch device.
        dataset_name: Name of dataset being evaluated.
        threshold_mib: Memory threshold in MiB (default: 800.0).
        strict: If True, raises MemoryError when breached. If False (default), logs advisory warning.

    Returns:
        allocated_mib: Peak memory allocated in MiB.
    """
    if device.type != "cuda":
        return 0.0
    allocated_bytes = torch.cuda.max_memory_allocated(device)
    allocated_mib = float(allocated_bytes) / (1024.0 * 1024.0)
    if "electronics" in dataset_name.lower() and allocated_mib > threshold_mib:
        msg = (
            f"[STAIR5-v7 Memory Gate {'Violation' if strict else 'Advisory'}] Peak allocated memory {allocated_mib:.2f} MiB "
            f"exceeds execution gate of {threshold_mib:.1f} MiB on {dataset_name}. "
            f"Fallback to reduced anchor chunks (B_anchor=128) recommended."
        )
        if strict:
            raise MemoryError(msg)
        print(f"⚠️ {msg}", flush=True)
    return allocated_mib

