"""Unrounded native-evaluator receipts; missing measurements never become baselines."""

import json
import math
import os
from pathlib import Path
import tempfile

PRIMARY_METRICS = ("RECALL@10", "RECALL@20", "NDCG@10", "NDCG@20")


def atomic_json(path, payload):
    """Replace one UTF-8 JSON receipt atomically; reject nonfinite JSON numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def native_metric_snapshot(monitors, mode):
    """Read FreeRec 0.9.7 AverageMeter averages before step() resets them.

    This only observes existing measurements. It neither implements metrics nor
    evaluates another checkpoint. Never substitute a historical/comparator value.
    """
    result = {}
    for meters in monitors[mode].values():
        for meter in meters:
            if not meter.active or meter.count <= 0:
                continue
            name, value = str(meter.name).upper(), float(meter.avg)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite native evaluation metric: {name}")
            if name in result:
                raise ValueError(f"Duplicate native metric name: {name}")
            result[name] = value
    return result


def validate_selected_receipt(receipt):
    """Require a measured TEST result at the validation-selected checkpoint."""
    if (receipt.get("mode") != "test" or receipt.get("selected_checkpoint") is not True
            or receipt.get("selection") != "validation NDCG@20"):
        raise ValueError("A selected-checkpoint TEST receipt is required")
    metrics = receipt.get("metrics", {})
    if any(name not in metrics for name in PRIMARY_METRICS):
        raise ValueError("Missing measured primary metric in selected-checkpoint receipt")
    for name in PRIMARY_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid measured metric: {name}")
    epoch = receipt.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Missing or invalid selected-checkpoint epoch")
    return receipt
