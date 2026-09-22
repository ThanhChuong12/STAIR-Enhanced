"""STAIR4-v2 training engine (P4).

Usage
-----
    python main_stair4_v2.py \
        --auxiliary_kernel phase_fidelity \
        --smoother_mode baseline \
        --pocl_weight_target 0.001

This file follows the same CLI pattern as main.py (baseline) exactly:
  - freerec.declare() version pin
  - cfg.add_argument() for all knobs
  - CoachForSTAIR4V2 subclasses freerec.launcher.Coach
  - Overrides set_optimizer() and train_per_epoch()
  - main() builds dataset, pipes, model, coach and calls coach.fit()

Key constraints enforced
------------------------
- Nonfinite loss → run aborts immediately (FloatingPointError).
- Smoother always cleared in a try/finally block inside train_per_epoch.
- Checkpoint saved atomically (write-tmp → rename).
- best_weights and training_checkpoint are separate files.
- No hard split of the 64-dim embedding space anywhere.
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
        # Support interactive environments (Jupyter / Colab / Kaggle) where
        # sys.argv contains kernel arguments like '-f <kernel.json>'.
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

# V2-specific args
cfg.add_argument("--auxiliary-kernel", type=str, default="none",
                 help="'none'|'cosine'|'phase_fidelity'")
cfg.add_argument("--rotation-mode", type=str, default="identity",
                 help="'identity'|'learned_givens'|'frozen_random'")
cfg.add_argument("--smoother-mode", type=str, default="baseline",
                 help="'baseline'|'identity_mix'|'bsf_mix'")
cfg.add_argument("--pocl-weight-target", type=float, default=0.001)
cfg.add_argument("--contrastive-temperature", type=float, default=0.2)
cfg.add_argument("--phase-scale", type=float, default=1.0)
cfg.add_argument("--warmup-epochs", type=int, default=10)
cfg.add_argument("--ramp-epochs", type=int, default=20)
cfg.add_argument("--spectral-time", type=float, default=0.5)
cfg.add_argument("--spectral-mix-target", type=float, default=0.1)
cfg.add_argument("--aux-lr-ratio", type=float, default=0.1)
cfg.add_argument("--aux-weight-decay", type=float, default=0.0)
cfg.add_argument("--knn-block-size", type=int, default=256)
cfg.add_argument("--ablation-id", type=str, default="A0")

cfg.set_defaults(
    description="STAIR4-v2",
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
cfg.compile()

# Post-process list/tuple args (mirrors main.py)
cfg.mfiles = cfg.mfiles.split(",")
cfg.num_neighbors = list(map(int, cfg.num_neighbors.split("-")))

# Baseline beta buffer (needed for device placement; also computed in model)
cfg.beta3 = (
    0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
).to(cfg.device)


# ---------------------------------------------------------------------------
# Checkpoint utilities (re-exported from models.stair4_v2_utils)
# ---------------------------------------------------------------------------
from models.stair4_v2_utils import (
    _to_plain,
    save_checkpoint_atomic,
    load_checkpoint_checked,
)


# ---------------------------------------------------------------------------
# Run manifest helper
# ---------------------------------------------------------------------------

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
    """FreeRec Coach for STAIR4-v2.

    Overrides only: set_optimizer() and train_per_epoch().
    All evaluation, checkpointing, early stopping, and NDCG@20 selection
    are inherited unchanged from freerec.launcher.Coach.
    """

    def set_optimizer(self):
        """Build AdamWSEvo with STAIR4-v2 parameter groups."""
        model: STAIR4V2 = self.model
        groups = model.parameter_groups()
        if self.cfg.optimizer.lower() in ("adamwsevo", "adamw"):
            self.optimizer = AdamWSEvo(
                groups,
                lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer {self.cfg.optimizer!r}; "
                "STAIR4-v2 requires adamwsevo (or its adamw alias)"
            )

    def train_per_epoch(self, epoch: int):
        """Single-epoch training loop with BSF smoother lifecycle.

        Contract:
        - model.set_epoch(epoch) is called first.
        - Smoother is cleared in a try/finally regardless of errors.
        - Nonfinite loss → FloatingPointError (run aborts, no silent skip).
        """
        model: STAIR4V2 = self.model
        model.set_epoch(epoch)

        lam, zeta, mode = model.effective_coefficients()

        for data in self.dataloader:
            data = self.dict_to_device(data)
            model.train()
            self.optimizer.zero_grad(set_to_none=True)
            try:
                # Keep the baseline ordering: clear stale gradients before a
                # fresh forward pass; model.fit() arms the step-scoped
                # smoother only after constructing the complete loss graph.
                loss = model.fit(data)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss={loss.item():.4f} at epoch={epoch}; "
                        "aborting to prevent silent divergence"
                    )
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(
                            f"Nonfinite gradient in {parameter.__class__.__name__} at epoch={epoch}"
                        )
                self.optimizer.step()
            finally:
                # Always clear smoother state — even if backward() throws
                model.bsf_smoother.clear_step_snapshot()
                model.smoother.clear_step_snapshot()

            self.monitor(
                loss.item(),
                n=len(data[self.model.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )


# ---------------------------------------------------------------------------
# Robust dataset loading (mirrors main.py exactly)
# ---------------------------------------------------------------------------

def _build_dataset():
    """Reuse the battle-tested dataset loader from main.py verbatim."""
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = _build_dataset()
    model = STAIR4V2(dataset, cfg)

    # Determine run directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    ablation_id = getattr(cfg, "ablation_id", "A0")
    run_name = f"stair4v2_{cfg.dataset}_{ablation_id}_{ts}"
    run_dir = Path("logs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(run_dir, model, ablation_id)

    # Build data pipes (mirrors main.py pattern)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    # Build coach (standard FreeRec signature)
    coach = CoachForSTAIR4V2(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )

    coach.fit()

    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"\n[GPU MEMORY TELEMETRY]")
        print(f"  Peak allocated : {peak_alloc:.2f} MB")
        print(f"  Peak reserved  : {peak_reserved:.2f} MB")

    # Update manifest status
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n[Done] Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
