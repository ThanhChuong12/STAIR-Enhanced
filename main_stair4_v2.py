"""STAIR4-v2.1 training engine (BCCR: Baseline-preserving Complex Cross-layer Regularization).

Usage
-----
    python main_stair4_v2.py \
        --auxiliary-kernel hybrid \
        --rotation-mode identity \
        --pocl-weight-target 0.0001 \
        --eta 0.25 \
        --kappa 0.5 \
        --smoother-mode baseline

This file follows the same CLI pattern as main.py (baseline) exactly:
  - freerec.declare() version pin
  - cfg.add_argument() for all knobs
  - CoachForSTAIR4V2 subclasses freerec.launcher.Coach
  - Overrides set_optimizer() and train_per_epoch()
  - main() builds dataset, pipes, model, coach and calls coach.fit()
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional
import yaml

import torch
import torch.nn as nn
try:
    import models.freerec_compat
except Exception:
    pass
import freerec

from optimizers.AdamW import AdamWSEvo
from models.stair4_v2 import STAIR4V2, STAIR4V2Options

# ── Version pin (mirrors main.py) ──────────────────────────────────────────
freerec.declare(version="1.0.1")

# ── Explicit YAML inheritance ──────────────────────────────────────────────
def load_config(path, _parents=()):
    """Explicit YAML inheritance: base_config is relative to its own file or configs/."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("cyclic base_config inheritance")
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    parent = values.pop("base_config", None)
    if parent:
        parent_path = path.parent / parent
        if not parent_path.exists():
            parent_path = Path("configs") / parent
        merged = load_config(parent_path, (*_parents, path))
    else:
        merged = {}
    merged.update(values)
    return merged


class InheritingParser(freerec.parser.Parser):
    def load(self):
        args, _ = self.parser.parse_known_args()
        if getattr(args, "config", None):
            self.set_defaults(**load_config(args.config))
        args, _ = self.parser.parse_known_args()
        return args


# ── CLI arguments ───────────────────────────────────────────────────────────
cfg = InheritingParser()

# Baseline shared args
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--mfiles", type=str,
                 default="textual_modality.pkl,visual_modality.pkl")
cfg.add_argument("--num-neighbors", type=str, default="5-1")
cfg.add_argument("--gamma", type=float, default=0.2)

# V2.1 BCCR-specific args
cfg.add_argument("--auxiliary-kernel", type=str, default="hybrid",
                 help="'none'|'hybrid'|'signed'|'fidelity'|'cosine'")
cfg.add_argument("--rotation-mode", type=str, default="identity",
                 help="'identity'|'learned_givens'")
cfg.add_argument("--smoother-mode", type=str, default="baseline",
                 help="'baseline'|'residual_spectral'")
cfg.add_argument("--eta", type=float, default=0.25,
                 help="Hybrid kernel blend: (1-eta) Re(h) + eta |h|^2")
cfg.add_argument("--kappa", type=float, default=0.5,
                 help="Phase scaling factor for bounded phase encoder")
cfg.add_argument("--pocl-weight-target", type=float, default=0.0001,
                 help="Target weight lambda_max for contrastive loss (default: 1e-4)")
cfg.add_argument("--contrastive-temperature", type=float, default=0.2)
cfg.add_argument("--warmup-epochs", type=int, default=10)
cfg.add_argument("--ramp-epochs", type=int, default=20)
cfg.add_argument("--xi", type=float, default=0.0,
                 help="Residual spectral correction factor (0.0 = pure baseline BSC)")
cfg.add_argument("--aux-lr-ratio", type=float, default=0.1)
cfg.add_argument("--aux-weight-decay", type=float, default=0.0)
cfg.add_argument("--knn-block-size", type=int, default=256)
cfg.add_argument("--query-chunk-size", type=int, default=256,
                 help="Maximum number of auxiliary queries evaluated per GEMM chunk")
cfg.add_argument("--ablation-id", type=str, default="A0")

cfg.set_defaults(
    description="STAIR4-v2.1",
    root="../../data",
    dataset="Amazon2014Baby_550_MMRec",
    epochs=500,
    batch_size=1024,
    optimizer="adamwsevo",
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
    monitors=["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
    which4best="NDCG@20",
)
_compiled = False


def compile_cfg():
    """Compile runtime configuration if not already compiled."""
    global _compiled
    if not _compiled:
        cfg.compile()
        cfg.mfiles = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else cfg.mfiles
        cfg.num_neighbors = list(map(int, cfg.num_neighbors.split("-"))) if isinstance(cfg.num_neighbors, str) else cfg.num_neighbors
        cfg.beta3 = (
            0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
        ).to(cfg.device)
        _compiled = True
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
from models.stair4_v2_utils import (
    _to_plain,
    save_checkpoint_atomic,
    load_checkpoint_checked,
)


def _write_run_manifest(run_dir: Path, model: STAIR4V2, ablation_id: str) -> dict:
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent),
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(Path(__file__).parent),
        ).decode().strip())
    except Exception:
        sha = "unknown"
        dirty = None

    manifest = {
        "run_id": str(uuid.uuid4()),
        "git_sha": sha,
        "git_dirty": dirty,
        "dataset": getattr(cfg, "dataset", "unknown"),
        "ablation_id": ablation_id,
        "options": asdict(model.options),
        "embedding_dim": model.embedding_dim,
        "num_layers": model.num_layers,
        "gamma": float(cfg.gamma),
        "data_hashes": model.data_hashes,
        "status": "running",
    }
    path = run_dir / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Coach
# ---------------------------------------------------------------------------

class CoachForSTAIR4V2(freerec.launcher.Coach):
    """FreeRec Coach for STAIR4-v2.1."""

    def _checkpoint_optimizer_state(self) -> dict:
        state = self.optimizer.state_dict()
        for group in state.get("param_groups", []):
            group.pop("smoother", None)
        return state

    def save_checkpoint(self, epoch: int) -> None:
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        payload = {"epoch": int(epoch)}
        for module_name in self.cfg.CHECKPOINT_MODULES:
            if module_name == "optimizer":
                payload[module_name] = self._checkpoint_optimizer_state()
            else:
                payload[module_name] = getattr(self, module_name).state_dict()
        payload["monitors"] = self.monitors.state_dict()
        save_checkpoint_atomic(path, payload)

    def load_checkpoint(self) -> int:
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        checkpoint = load_checkpoint_checked(path)
        for module_name in self.cfg.CHECKPOINT_MODULES:
            if module_name == "optimizer":
                self.optimizer.load_state_dict(checkpoint[module_name])
                for group in self.optimizer.param_groups:
                    group["smoother"] = (
                        self.model.bsf_smoother
                        if group.get("role") == "items"
                        else None
                    )
            else:
                getattr(self, module_name).load_state_dict(checkpoint[module_name])
        self.monitors.load_state_dict(checkpoint["monitors"])
        return int(checkpoint["epoch"])

    def set_optimizer(self):
        model: STAIR4V2 = self.model
        groups = model.parameter_groups()
        beta1 = getattr(
            self.cfg,
            "beta1",
            getattr(self.cfg, "optim_first_moment_decay", 0.9),
        )
        beta2 = getattr(
            self.cfg,
            "beta2",
            getattr(self.cfg, "optim_second_moment_decay", 0.999),
        )
        if not (0.0 <= float(beta1) < 1.0 and 0.0 <= float(beta2) < 1.0):
            raise ValueError(
                f"invalid AdamWSEvo betas: beta1={beta1!r}, beta2={beta2!r}"
            )
        if self.cfg.optimizer.lower() in ("adamwsevo", "adamw"):
            self.optimizer = AdamWSEvo(
                groups,
                lr=self.cfg.lr,
                betas=(float(beta1), float(beta2)),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer {self.cfg.optimizer!r}; "
                "STAIR4-v2.1 requires adamwsevo"
            )

    def train_per_epoch(self, epoch: int):
        model: STAIR4V2 = self.model
        model.set_epoch(epoch)

        for data in self.dataloader:
            data = self.dict_to_device(data)
            model.train()
            # Keep the baseline coach's clearing semantics for Gate 0 parity.
            self.optimizer.zero_grad()
            try:
                loss = model.fit(data)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss={loss.item():.4f} at epoch={epoch}"
                    )
                loss.backward()
                model.update_post_backward_diagnostics()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"Nonfinite gradient in {parameter.__class__.__name__} at epoch={epoch}"
                        )
                self.optimizer.step()
                model.update_post_step_diagnostics()
            finally:
                model.bsf_smoother.clear_step_snapshot()
                model.smoother.clear_step_snapshot()

            self.monitor(
                loss.item(),
                n=len(data[self.model.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )
            self._append_diagnostics(model.last_diagnostics)

    def _append_diagnostics(self, diagnostics: Dict) -> None:
        """Append detached per-step telemetry when the caller configured a path."""
        path = getattr(self, "diagnostics_path", None)
        interval = max(1, int(self.model.options.diagnostic_interval_steps))
        if path is None or diagnostics.get("step", 0) % interval != 0:
            return
        record = {key: value for key, value in diagnostics.items() if isinstance(value, (str, int, float, bool))}
        record["total_loss"] = float(diagnostics.get("bpr", 0.0) + diagnostics.get("cl_weighted", 0.0))
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Robust dataset loading
# ---------------------------------------------------------------------------

def _build_dataset():
    processed_dir = os.path.join(cfg.root, "Processed", cfg.dataset)
    if os.path.islink(processed_dir) and not os.path.exists(processed_dir):
        try:
            os.unlink(processed_dir)
        except Exception:
            pass

    if not os.path.exists(processed_dir) or (
        os.path.isdir(processed_dir) and not os.listdir(processed_dir)
    ):
        script_dir = (
            os.path.dirname(os.path.abspath(__file__))
            if "__file__" in dir()
            else "."
        )
        candidates = [
            os.path.join(cfg.root, cfg.dataset),
            os.path.join("/kaggle/data", cfg.dataset),
            os.path.join("/kaggle/data/Processed", cfg.dataset),
            os.path.join("/kaggle/working/STAIR/data", cfg.dataset),
            os.path.join("/kaggle/working/STAIR-Enhanced/data", cfg.dataset),
            os.path.join(script_dir, "data", cfg.dataset),
            os.path.join("data", cfg.dataset),
        ]
        for cand in candidates:
            if (
                os.path.exists(cand)
                and os.path.isdir(cand)
                and os.path.abspath(cand) != os.path.abspath(processed_dir)
                and os.listdir(cand)
            ):
                os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
                try:
                    os.symlink(cand, processed_dir)
                    print(f"[DataSet] Auto-bridged symlink: {cand} -> {processed_dir}")
                except Exception:
                    import shutil
                    shutil.copytree(cand, processed_dir, dirs_exist_ok=True)
                    print(f"[DataSet] Auto-bridged copy: {cand} -> {processed_dir}")
                break

    tasktag = getattr(cfg, "tasktag", None) or getattr(
        freerec.data.tags, "MATCHING", None
    )
    if hasattr(freerec.data.datasets, "RecDataSet"):
        freerec.data.datasets.RecDataSet.TASK = tasktag
    if hasattr(freerec.data.datasets, "base") and hasattr(
        freerec.data.datasets.base, "BaseSet"
    ):
        freerec.data.datasets.base.BaseSet.TASK = tasktag

    ds_cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(ds_cls, type):
        try:
            dataset = ds_cls(root=cfg.root)
        except Exception:
            try:
                from freerec.data.datasets.base import MatchingRecDataSet
                dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
            except Exception:
                dataset = freerec.data.datasets.RecDataSet(
                    cfg.root, cfg.dataset, tasktag=tasktag
                )
    else:
        try:
            from freerec.data.datasets.base import MatchingRecDataSet
            dataset = MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=tasktag)
        except Exception:
            dataset = freerec.data.datasets.RecDataSet(
                cfg.root, cfg.dataset, tasktag=tasktag
            )

    if not hasattr(dataset, "TASK") or dataset.TASK is None:
        dataset.TASK = tasktag

    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    compile_cfg()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = _build_dataset()
    model = STAIR4V2(dataset, cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    ablation_id = getattr(cfg, "ablation_id", "A0")
    run_name = f"stair4v2_{cfg.dataset}_{ablation_id}_{ts}"
    run_dir = Path("logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(run_dir, model, ablation_id)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSTAIR4V2(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.diagnostics_path = run_dir / "diagnostics.jsonl"

    coach.fit()

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"\n[GPU MEMORY TELEMETRY]")
        print(f"  Peak allocated : {peak_alloc:.2f} MB")
        print(f"  Peak reserved  : {peak_reserved:.2f} MB")

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[Done] Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
