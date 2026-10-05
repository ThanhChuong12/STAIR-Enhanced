"""Utilities for STAIR5-v5.2: Atomic checkpoints, RNG state management, and validation."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp
import torch


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
            try:
                os.unlink(temporary)
            except OSError:
                pass


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


def save_training_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    atomic_torch_save(
        {
            "version": "5.2",
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
    if payload.get("version") not in (5.2, "5.2"):
        raise ValueError(f"Unsupported STAIR5-v5.2 training checkpoint version: {payload.get('version')}")
    # Reject mismatched graphs/configs before copying any embedding or optimizer state.
    if payload["model"].get("_extra_state") != model.get_extra_state():
        raise ValueError("Checkpoint model/configuration/data provenance mismatch.")
    if payload.get("epoch", -1) < 0:
        raise ValueError("Checkpoint epoch must be nonnegative.")
    model.load_state_dict(payload["model"])
    load_optimizer_state(optimizer, payload["optimizer"])
    if restore_random and "rng" in payload:
        restore_rng(payload["rng"])
    return payload


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


def validate_csr_operator(matrix: sp.csr_matrix, n_items: int) -> None:
    """Asserts that a SciPy CSR operator is square, finite, non-negative, and symmetric."""
    if matrix.shape != (n_items, n_items):
        raise ValueError(f"Expected CSR operator shape ({n_items}, {n_items}), got {matrix.shape}")
    if not np.isfinite(matrix.data).all():
        raise ValueError("CSR operator contains non-finite values (NaN/Inf).")
    if (matrix.data < 0).any():
        raise ValueError("CSR operator contains negative values.")
    diff = matrix - matrix.T
    if diff.nnz and np.max(np.abs(diff.data)) > 1e-6:
        raise ValueError("CSR operator is not symmetric within tolerance 1e-6.")
