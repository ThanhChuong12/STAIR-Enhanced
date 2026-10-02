"""Atomic, weights-only-loadable training checkpoints and utilities for STAIR5-v3."""
import json
import os
from pathlib import Path
import random
import re
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
    expected, received = model.get_extra_state(), payload["model"]["_extra_state"]
    for key in ("version", "data_fingerprint", "graph_fingerprint", "arm", "config"):
        if received.get(key) != expected[key]:
            raise ValueError(f"Training checkpoint incompatible with current {key}.")
    model.load_state_dict(payload["model"])
    load_optimizer_state(optimizer, payload["optimizer"])
    if restore_random:
        restore_rng(payload["rng"])
    return payload


def parse_telemetry_jsonl(artifact_dir):
    """Load ordered training telemetry records from an artifact directory."""
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
    return sorted(records, key=lambda x: x.get("epoch", 0))


def parse_training_losses(log_path, telemetry=None):
    """Parse pure BPR loss and total loss curves from logs or telemetry."""
    bpr_losses = []
    total_losses = []
    if telemetry and len(telemetry) > 0:
        for r in telemetry:
            ep = r.get("epoch")
            bpr = r.get("bpr_loss")
            cl = r.get("cl_loss", 0.0)
            lam = r.get("lambda", 0.0)
            if ep is not None and bpr is not None:
                bpr_losses.append((ep, float(bpr)))
                total_losses.append((ep, float(bpr + lam * cl)))
        return bpr_losses, total_losses

    if log_path and Path(log_path).is_file():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"TRAIN @Epoch:\s*(\d+)\s*>>>\s*\|\|\s*LOSS Avg:\s*([0-9.]+)", line)
                if match:
                    ep = int(match.group(1))
                    val = float(match.group(2))
                    total_losses.append((ep, val))
                    bpr_losses.append((ep, val))
    return bpr_losses, total_losses


def parse_valid_metric(log_path, metric_name="NDCG@20"):
    """Parse validation metric progression by epoch from standard FreeRec log."""
    results = []
    if not log_path or not Path(log_path).is_file():
        return results
    pattern = rf"VALID @Epoch:\s*(\d+)\s*>>>.*{re.escape(metric_name)}\s*Avg:\s*([0-9.]+)"
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                results.append((int(match.group(1)), float(match.group(2))))
    return results
