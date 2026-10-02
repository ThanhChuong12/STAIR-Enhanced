"""Atomic, weights-only-loadable training checkpoints for C-HET."""
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def atomic_torch_save(payload, path):
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


def optimizer_state_for_checkpoint(optimizer):
    """Exclude runtime callback objects from the serialized parameter groups."""
    state = optimizer.state_dict()
    state["param_groups"] = [{k: v for k, v in g.items() if k != "smoother"}
                             for g in state["param_groups"]]
    return state


def load_optimizer_state(optimizer, state):
    """Reattach the current model's callbacks after native moment restoration."""
    if len(state["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer group count differs from current model.")
    runtime = [g.get("smoother") for g in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, smoother in zip(optimizer.param_groups, runtime):
        group["smoother"] = smoother


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_training_checkpoint(path, model, optimizer, epoch, extra=None):
    atomic_torch_save({"version": 1, "model": model.state_dict(),
                       "optimizer": optimizer_state_for_checkpoint(optimizer),
                       "epoch": int(epoch), "rng": rng_state(), "extra": extra or {}}, path)


def load_training_checkpoint(path, model, optimizer, restore_random=True):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("Unsupported training checkpoint version.")
    # Validate source/config metadata before copying any state into the model.
    expected, received = model.get_extra_state(), payload["model"]["_extra_state"]
    for key in ("version", "data_fingerprint", "graph_fingerprint", "arm", "config"):
        if received.get(key) != expected[key]:
            raise ValueError(f"Training checkpoint incompatible with current {key}.")
    model.load_state_dict(payload["model"])
    load_optimizer_state(optimizer, payload["optimizer"])
    if restore_random:
        restore_rng(payload["rng"])
    return payload
