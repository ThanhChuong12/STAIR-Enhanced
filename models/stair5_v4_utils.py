"""Utilities for STAIR5-v4: Atomic checkpoints, RNG state management, and manifests."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


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
            "version": 4,
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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != 4:
        # Support fallback or version 4
        pass
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
