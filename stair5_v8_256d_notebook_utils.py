"""Dependency-free command and artifact readers for the V8-256D notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

METRICS = ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20")
DATASETS = {
    "baby": "Amazon2014Baby_550_MMRec",
    "sports": "Amazon2014Sports_550_MMRec",
    "electronics": "Amazon2014Electronics_550_MMRec",
}
CONFIGS = {
    key: f"Amazon2014{label}_STAIR5_v8_256D.yaml"
    for key, label in (("baby", "Baby"), ("sports", "Sports"), ("electronics", "Electronics"))
}
REQUIRED_DATA = ("train.txt", "valid.txt", "test.txt", "textual_modality.pkl", "visual_modality.pkl")


def link_processed_dataset(source, target):
    """Keep FreeRec's dataset directory writable, with immutable input file links.

    Linking the whole directory into /kaggle/input would make generated dataset
    caches unwritable. Only the original split/feature files are linked here.
    """
    source, target = Path(source).resolve(), Path(target)
    for filename in REQUIRED_DATA:
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    if target.is_symlink():
        raise ValueError(f"Dataset directory must be writable, not a directory symlink: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_DATA:
        destination = target / filename
        original = source / filename
        if destination.exists() or destination.is_symlink():
            if not os.path.samefile(destination, original):
                raise ValueError(f"Dataset file already points to another source: {destination}")
        else:
            try:
                destination.symlink_to(original)
            except OSError as error:
                if getattr(error, "winerror", None) != 1314:
                    raise
                # Windows without Developer Mode can still link immutable
                # input files on the same volume. Kaggle uses symlinks above.
                destination.hardlink_to(original)
    return target


def training_command(repo, key, config, root, artifact, cache, *, seed=1,
                     epochs=500, eval_freq=5, device="0", resume_from="",
                     support_files="", executable=None):
    """Construct an argv list without changing the architecture or batch size."""
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset: {key}")
    for name, value, minimum in (("seed", seed, 0), ("epochs", epochs, 1),
                                 ("eval_freq", eval_freq, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {name}: {value}")
    repo, config, root = Path(repo), Path(config), Path(root)
    runner = repo / "main_stair5_v8_256d.py"
    for required in (runner, config):
        if not required.is_file():
            raise FileNotFoundError(required)
    dataset = root / "Processed" / DATASETS[key]
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if resume_from and not Path(resume_from).is_file():
        raise FileNotFoundError(resume_from)
    if resume_from and Path(resume_from).name in ("best_model.pt", "last_model.pt"):
        raise ValueError("Resume requires training_checkpoint.pt with optimizer/RNG state")
    command = [executable or sys.executable, "-u", str(runner),
               "--config", str(config), "--root", str(root), "--device", str(device),
               "--artifact-dir", str(artifact), "--graph-cache-dir", str(cache),
               "--seed", str(seed), "--epochs", str(epochs), "--eval-freq", str(eval_freq),
               "--embedding-dim", "256", "--ranking", "full", "--knn-device", "cpu"]
    if resume_from:
        command += ["--resume-from", str(resume_from)]
    if support_files:
        paths = str(support_files).split(",")
        if len(paths) != 2 or any(not Path(p).is_file() for p in paths):
            raise ValueError("Supply existing text,visual candidate .pt files in modality order")
        command += ["--v4-support-files", str(support_files)]
    return command


def read_telemetry(artifact):
    """Read actual Coach fields; duplicate epochs resolve to the latest record."""
    path = Path(artifact) / "training_telemetry.jsonl"
    if not path.is_file():
        return []
    records = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            epoch = int(record["epoch"])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid telemetry {path}:{number}") from error
        records[epoch] = record
    return [records[epoch] for epoch in sorted(records)]


def read_evaluation(log):
    """Separate validation history from test of the validation-selected model.

    A final-epoch test before 'Load best model' is never substituted for the
    selected test. Rounded FreeRec values cannot establish statistical gains.
    """
    path = Path(log)
    if not path.is_file():
        return {"validation": [], "best_epoch": None, "test_metrics": {}}
    history, best_epoch, selected = {}, None, {}
    marker = re.compile(r"Load best model\s*@Epoch:\s*(\d+)", re.I)
    evaluation = re.compile(r"\b(VALID|TEST)\s+@Epoch:\s*(\d+)", re.I)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = marker.search(line)
        if match:
            best_epoch, selected = int(match.group(1)), {}
        match = evaluation.search(line)
        if not match:
            continue
        mode, epoch = match.group(1).upper(), int(match.group(2))
        values = {}
        for metric in METRICS:
            value = re.search(re.escape(metric) + r"\s+Avg:\s*([0-9.eE+-]+)", line, re.I)
            if value:
                values[metric] = float(value.group(1))
        if mode == "VALID":
            history[epoch] = {"epoch": epoch, **values}
        elif best_epoch is not None and epoch == best_epoch:
            selected = values
    return {"validation": [history[k] for k in sorted(history)],
            "best_epoch": best_epoch, "test_metrics": selected}
