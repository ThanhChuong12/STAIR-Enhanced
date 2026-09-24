"""STAIR4-CSGC training; FreeRec owns ranking, masking, metrics and selection."""
from __future__ import annotations

import json
import importlib.metadata
import math
import os
import platform
import sys
import time
from pathlib import Path
from dataclasses import asdict
import yaml
import torch
import models.freerec_compat
import freerec
from models.stair4_v4 import STAIR4V4
from models.stair4_v4_graph import CSGCOptions
from models.stair4_v4_utils import atomic_save, capture_rng, restore_rng, source_hash, capture_sampler, restore_sampler
from optimizers.AdamW import AdamWSEvo

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


cfg = InheritingParser()
for name, kind, default in (("embedding-dim", int, 64), ("num-layers", int, 3),
    ("mfiles", str, "textual_modality.pkl,visual_modality.pkl"),
    ("num-neighbors", str, "5-1"), ("gamma", float, .1)):
    cfg.add_argument("--" + name, type=kind, default=default)
for name, field in CSGCOptions.__dataclass_fields__.items():
    cfg.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
cfg.add_argument("--graph-cache-dir", default=".cache/stair4_v4")
cfg.add_argument("--audit-only", action="store_true", help="Build graph and write diagnostics without training")
cfg.add_argument("--run-dir", default=None)
cfg.add_argument("--checkpoint-dir", default=None, help="Default: this run's FreeRec log directory/checkpoints")
cfg.add_argument("--finite-check-interval", type=int, default=0,
                 help="Optional gradient scan interval; zero avoids full-table scans")
cfg.set_defaults(description="STAIR4-CSGC", root="data", dataset="Amazon2014Baby_550_MMRec",
                 epochs=500, batch_size=1024, optimizer="adamwsevo", lr=.001,
                 weight_decay=.3, seed=1, num_workers=0,
                 monitors=["LOSS", "Recall@1", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
                 which4best="NDCG@20")


def compile_cfg():
    cfg.compile()
    cfg.CHECKPOINT_PATH = str(Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else Path(cfg.LOG_PATH) / "checkpoints")
    Path(cfg.CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    CSGCOptions.from_config(cfg)
    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("CSGC requires the baseline AdamWSEvo optimizer")
    if cfg.which4best.upper() != "NDCG@20":
        raise ValueError("The registered protocol selects by validation NDCG@20")
    if cfg.finite_check_interval < 0:
        raise ValueError("finite_check_interval must be nonnegative")
    if cfg.resume and cfg.num_workers != 0:
        raise ValueError("Verified epoch-boundary resume requires num_workers=0")
    return cfg


def build_dataset(config):
    """Use FreeRec's processed layout; do not silently copy or mutate input data."""
    path = Path(config.root) / "Processed" / config.dataset
    if not path.is_dir():
        raise FileNotFoundError(f"Expected {path}. Place train/valid/test files and modality pickles there.")
    task = getattr(freerec.data.tags, "MATCHING")
    freerec.data.datasets.RecDataSet.TASK = task
    cls = getattr(freerec.data.datasets, config.dataset, None)
    if isinstance(cls, type):
        return cls(root=config.root)
    matching = getattr(freerec.data.datasets.base, "MatchingRecDataSet", None)
    if matching is not None:
        return matching(config.root, config.dataset, tasktag=task)
    return freerec.data.datasets.RecDataSet(config.root, config.dataset, tasktag=task)


class CoachForSTAIR4V4(freerec.launcher.Coach):
    """Baseline evaluation and selection are inherited without modification."""
    def set_optimizer(self):
        self.optimizer = AdamWSEvo(self.model.parameter_groups(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(getattr(self.cfg, "beta1", getattr(self.cfg, "optim_first_moment_decay", .9)),
                   getattr(self.cfg, "beta2", getattr(self.cfg, "optim_second_moment_decay", .999))))

    def _resume_protocol(self):
        return {key: getattr(self.cfg, key) for key in ("seed", "batch_size", "lr", "weight_decay",
            "ranking", "retain_seen", "eval_freq", "eval_valid", "eval_test", "which4best",
            "optim_first_moment_decay", "optim_second_moment_decay")}

    def _append(self, name, record):
        if hasattr(self, "run_dir"):
            with (self.run_dir / name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def train_per_epoch(self, epoch):
        start = time.perf_counter()
        count, total = 0, 0.0
        for step, data in enumerate(self.dataloader):
            data = self.dict_to_device(data)
            loss = self.model(data)
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.finite_check_interval and step % self.cfg.finite_check_interval == 0:
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, batch {step}")
            self.model.smoother.arm_static()
            try:
                self.optimizer.step()
            finally:
                self.model.smoother.clear_step_snapshot()
            batch = len(data[self.User])
            self.monitor(value, n=batch, reduction="mean", mode="train", pool=["LOSS"])
            count += batch
            total += value * batch
        self._append("epochs.jsonl", {"epoch": epoch, "train_seconds": time.perf_counter() - start,
            "examples": count, "bpr": total / max(1, count),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0})

    def evaluate(self, epoch, step=-1, mode="valid"):
        start = time.perf_counter()
        super().evaluate(epoch, step, mode)
        metrics = {m.name: float(m.avg) for meters in self.monitors[mode].values() for m in meters if m.active}
        self._append("evaluation.jsonl", {"epoch": epoch, "mode": mode, "metrics": metrics,
            "selected_checkpoint": getattr(self, "_evaluating_selected", False),
            "seconds": time.perf_counter() - start})

    def eval_at_best(self):
        self._evaluating_selected = True
        try:
            return super().eval_at_best()
        finally:
            self._evaluating_selected = False

    def save_checkpoint(self, epoch):
        payload = {"epoch": int(epoch), "rng": capture_rng(), "modules": {},
                   "protocol": self._resume_protocol(),
                   "selection": {key: getattr(self, key) for key in
                       ("_best", "_best_epoch", "_best_step", "_stopping_steps")},
                   "monitors": dict(self.monitors.state_dict()),
                   "runtime": {"torch": str(torch.__version__), "freerec": str(freerec.__version__),
                               "workers": self.cfg.num_workers},
                   "loaders": {}}
        for name in self.cfg.CHECKPOINT_MODULES:
            state = getattr(self, name).state_dict()
            if name == "optimizer":
                state = {**state, "param_groups": [{k: v for k, v in g.items() if k != "smoother"}
                                                   for g in state["param_groups"]]}
            payload["modules"][name] = state
        for name in ("trainloader", "validloader", "testloader"):
            loader = getattr(self, name)
            if not hasattr(loader, "state_dict"):
                raise RuntimeError("Checkpointing requires a stateful DataLoader2 implementation")
            payload["loaders"][name] = {"loader": loader.state_dict(), "sampler": capture_sampler(loader.datapipe)}
        best = Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME
        payload["best_model"] = torch.load(best, map_location="cpu", weights_only=True) if best.exists() else None
        atomic_save(Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME, payload)

    def load_checkpoint(self):
        # DataLoader2's serialized pipeline is trusted local checkpoint data.
        payload = torch.load(Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME,
                             map_location="cpu", weights_only=True)
        expected = {"torch": str(torch.__version__), "freerec": str(freerec.__version__), "workers": self.cfg.num_workers}
        if payload["runtime"] != expected:
            raise ValueError("Resume requires identical torch/FreeRec versions and worker count")
        if payload["protocol"] != self._resume_protocol():
            raise ValueError("Resume optimization, sampling or evaluation protocol differs")
        self.model.set_extra_state(payload["modules"]["model"].get("_extra_state"))
        for name in self.cfg.CHECKPOINT_MODULES:
            getattr(self, name).load_state_dict(payload["modules"][name])
        for group in self.optimizer.param_groups:
            group["smoother"] = self.model.smoother if group["role"] == "items" else None
        for name, state in payload["loaders"].items():
            loader = getattr(self, name)
            # Rebuild field objects in this process: FreeRec caches Python hashes
            # on Field keys, which cannot be reused across interpreter processes.
            # Epoch-boundary randomness is restored onto the fresh pipeline.
            fresh = loader.state_dict()
            loader_state = {**state["loader"], "serialized_datapipe": fresh["serialized_datapipe"]}
            loader.load_state_dict(loader_state)
            restore_sampler(loader.datapipe, state["sampler"])
        self.monitors.load_state_dict(payload["monitors"])
        for key, value in payload["selection"].items():
            setattr(self, key, value)
        if payload["best_model"] is not None:
            atomic_save(Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME, payload["best_model"])
        restore_rng(payload["rng"])
        return int(payload["epoch"])


def _safe_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not-installed"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    compile_cfg()
    run_dir = Path(cfg.run_dir or ("logs/stair4v4_" + cfg.dataset + "_" + cfg.ablation_id + "_" + time.strftime("%Y%m%d_%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "preparing", "dataset": cfg.dataset, "seed": cfg.seed,
                "options": asdict(CSGCOptions.from_config(cfg)), "source_hash": source_hash(),
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "freerec": str(freerec.__version__),
                            **{name: _safe_version(name) for name in
                               ("numpy", "scipy", "numba", "torchdata", "torch-geometric")}},
                "protocol": {"ranking": cfg.ranking, "selection": cfg.which4best,
                             "lr": cfg.lr, "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
                             "epochs": cfg.epochs, "gamma": cfg.gamma, "num_workers": cfg.num_workers,
                             "freerec_log_path": str(cfg.LOG_PATH), "checkpoint_path": str(cfg.CHECKPOINT_PATH)}}
    manifest_path = run_dir / "run_manifest.json"
    start = time.perf_counter()
    try:
        dataset = build_dataset(cfg)
        model = STAIR4V4(dataset, cfg)
        manifest.update(graph_identity=model.graph_identity, graph_diagnostics=model.graph_diagnostics,
                        preparation_seconds=time.perf_counter() - start)
        (run_dir / "graph_audit.json").write_text(json.dumps(model.graph_diagnostics, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not cfg.audit_only:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            coach = CoachForSTAIR4V4(dataset=dataset, model=model, cfg=cfg,
                trainpipe=model.sure_trainpipe(cfg.batch_size), validpipe=model.sure_validpipe(cfg.ranking),
                testpipe=model.sure_testpipe(cfg.ranking))
            coach.run_dir = run_dir
            coach.fit()
            manifest["selected_epoch"] = coach._best_epoch
            manifest["selected_valid_ndcg20"] = coach._best
        manifest["status"] = "audit_completed" if cfg.audit_only else "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter() - start
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[STAIR4-CSGC] Artifacts: {run_dir}")


if __name__ == "__main__":
    main()


