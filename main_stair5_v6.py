"""main_stair5_v6.py — STAIR5-v6 Training Engine: NLGCL-BCSR.
===========================================================
Behavior-Conditioned Semantic Retention for Item Update Smoothing in RecSys.

Usage Example:
    python main_stair5_v6.py --config configs/Amazon2014Sports_STAIR5_v6.yaml --v6-arm P-BCSR
"""
import json
import argparse
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import models.freerec_compat  # Fail visibly if required compatibility cannot load.

import freerec
import torch

from models.stair5_v6 import ARMS_V6, STAIR5_v6_Model, canonicalize_arm_name
from models.stair5_v6_utils import (
    atomic_torch_save,
    load_training_checkpoint,
    save_training_checkpoint,
    restore_rng,
    file_sha256,
)
from optimizers.AdamW import AdamWSEvo


def build_config():
    cfg = freerec.parser.Parser()
    for flag, kind, default in (
        ("embedding-dim", int, 64),
        ("num-layers", int, 3),
        ("mfiles", str, "textual_modality.pkl,visual_modality.pkl"),
        ("num-neighbors", str, "5-1"),
        ("gamma", float, 0.2),
        ("beta1", float, 0.9),
        ("beta2", float, 0.999),
        ("eta", float, 0.1),
        ("k-cf", int, 5),
        ("c-min", int, 2),
        ("t-shrinkage", float, 5.0),
        ("v6-theta", float, 0.25),
        ("v6-t-rel", float, 5.0),
        ("v6-candidate-chunk-size", int, 65536),
        ("lambda-nlgcl", float, 0.01),
        ("nlgcl-tau", float, 0.2),
        ("nlgcl-G", int, 1),
        ("nlgcl-alpha", float, 0.5),
        ("cl-chunk-size", int, 1024),
        ("knn-chunk-size", int, 1024),
        ("cf-block-size", int, 64),
        ("cf-memory-budget-mib", float, 128.0),
        ("graph-cache-dir", str, ""),
        ("artifact-dir", str, ""),
        ("resume-from", str, ""),
    ):
        cfg.add_argument("--" + flag, type=kind, default=default)

    cfg.add_argument("--v6-arm", choices=ARMS_V6, default="P-BCSR")
    cfg.add_argument("--knn-device", choices=("cpu", "cuda", "auto"), default="auto")

    cfg.set_defaults(
        description="STAIR5-v6-NLGCL-BCSR",
        root="data",
        dataset="Amazon2014Sports_550_MMRec",
        epochs=500,
        batch_size=1024,
        optimizer="adamwsevo",
        lr=1e-3,
        weight_decay=0.1,
        seed=1,
        monitors=["LOSS", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
        which4best="NDCG@20",
    )
    cfg.compile()

    cfg.v6_arm = canonicalize_arm_name(cfg.v6_arm)

    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("STAIR5-v6 requires AdamWSEvo optimizer with BSC smoother.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Must select by validation NDCG@20; disable test evaluation during training.")
    if cfg.ranking != "full":
        raise ValueError("Reference experiments require full ranking; pool scorer remains available for testing.")
    if not math.isfinite(cfg.eta) or not (0.0 <= cfg.eta <= 1.0) or cfg.k_cf < 0 or cfg.c_min < 1:
        raise ValueError("Require eta in [0, 1], k_cf >= 0 and c_min >= 1.")
    if not math.isfinite(cfg.v6_theta) or not (0.0 <= cfg.v6_theta <= 0.5):
        raise ValueError("Require v6_theta in [0, 0.5].")
    if not math.isfinite(cfg.v6_t_rel) or cfg.v6_t_rel <= 0.0:
        raise ValueError("Require finite v6_t_rel > 0.")
    if not math.isfinite(cfg.t_shrinkage) or cfg.t_shrinkage <= 0:
        raise ValueError("Require finite t_shrinkage > 0.")
    if not math.isfinite(cfg.lambda_nlgcl) or cfg.lambda_nlgcl < 0 or not math.isfinite(cfg.nlgcl_tau) or cfg.nlgcl_tau <= 0:
        raise ValueError("Require finite lambda_nlgcl >= 0 and nlgcl_tau > 0.")
    if cfg.nlgcl_G < 1 or not math.isfinite(cfg.nlgcl_alpha) or not 0 <= cfg.nlgcl_alpha <= 1:
        raise ValueError("Require nlgcl_G >= 1 and nlgcl_alpha in [0, 1].")
    if cfg.knn_chunk_size < 1 or cfg.cf_block_size < 1 or cfg.cl_chunk_size < 0:
        raise ValueError("Invalid preprocessing/loss chunk sizes.")
    if not math.isfinite(cfg.cf_memory_budget_mib) or cfg.cf_memory_budget_mib <= 0:
        raise ValueError("CF memory budget must be finite and positive.")
    if cfg.eval_freq < 1 or cfg.CHECKPOINT_FREQ < 1:
        raise ValueError("Evaluation and checkpoint frequencies must be positive.")
    if cfg.seed < 0:
        raise ValueError("Seed must be non-negative.")
    if cfg.embedding_dim <= 0 or cfg.num_layers < 0 or not math.isfinite(cfg.gamma) or cfg.gamma <= 0:
        raise ValueError("Require embedding_dim > 0, num_layers >= 0 and finite gamma > 0.")

    cfg.mfiles = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else list(cfg.mfiles)
    cfg.num_neighbors = (
        list(map(int, cfg.num_neighbors.split("-")))
        if isinstance(cfg.num_neighbors, str)
        else cfg.num_neighbors
    )
    if not cfg.mfiles or len(cfg.mfiles) != len(cfg.num_neighbors) or any(k <= 0 for k in cfg.num_neighbors):
        raise ValueError("Each modality requires one positive neighbor count.")
    cfg.beta3 = (
        0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    ).to(cfg.device)

    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH) / "nlgcl_bcsr_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    return cfg


class CoachForSTAIR5_v6(freerec.launcher.Coach):
    def load_best(self):
        super().load_best()
        self._selected_checkpoint_loaded = True

    def eval_at_best(self):
        """Use native evaluation unchanged, exporting only the selected checkpoint."""
        self._selected_checkpoint_loaded = False
        try:
            return super().eval_at_best()
        finally:
            self._selected_checkpoint_loaded = False

    def test(self, epoch: int, step: int = -1):
        result = super().test(epoch, step)
        if getattr(self, "_selected_checkpoint_loaded", False):
            metrics = {
                meter.name: float(meter.history[-1])
                for meters in self.monitors["test"].values()
                for meter in meters if meter.history
            }
            best_path = Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME
            payload = {
                "split": "test",
                "epoch": int(epoch),
                "step": int(step),
                "ranking": self.cfg.ranking,
                "arm": self.model.v6_arm,
                "seed": self.cfg.seed,
                "metrics": metrics,
                "checkpoint_sha256": file_sha256(best_path),
                "graph_fingerprint": self.model.graph_fingerprint,
                "selection": "validation NDCG@20",
            }
            self._artifact("selected_test_metrics.json").write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
            )
        return result

    def set_optimizer(self):
        self.optimizer = AdamWSEvo(
            self.model.marked_params(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=self.cfg.weight_decay,
        )

    def _artifact(self, name: str) -> Path:
        return Path(self.cfg.artifact_dir) / name

    def save_best(self):
        super().save_best()
        atomic_torch_save(self.model.state_dict(), self._artifact("best_model.pt"))

    def save_last(self):
        super().save_last()
        atomic_torch_save(self.model.state_dict(), self._artifact("last_model.pt"))
        if not getattr(self, "_epoch_in_progress", False):
            self.save_checkpoint(getattr(self, "_completed_epochs", 0))

    def save_checkpoint(self, epoch: int):
        extra = {
            "monitors": self.monitors.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self._best,
            "best_epoch": self._best_epoch,
            "best_step": self._best_step,
            "stopping_steps": self._stopping_steps,
            "dataset_rng": self.dataset.rng.getstate() if hasattr(self.dataset, "rng") else None,
            "loaders": {
                name: getattr(self, name).state_dict()
                for name in ("trainloader", "validloader", "testloader")
            },
        }
        best_path = Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME
        if best_path.exists():
            extra["best_model"] = torch.load(best_path, map_location="cpu", weights_only=True)
        target = self._artifact("training_checkpoint.pt")
        save_training_checkpoint(target, self.model, self.optimizer, epoch, extra)

    def load_checkpoint(self) -> int:
        target = self.cfg.resume_from or self._artifact("training_checkpoint.pt")
        payload = load_training_checkpoint(target, self.model, self.optimizer, restore_random=False)
        extra = payload["extra"]
        self.monitors.load_state_dict(extra["monitors"])
        self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        self._best, self._best_epoch, self._best_step = extra["best"], extra["best_epoch"], extra["best_step"]
        self._stopping_steps = extra["stopping_steps"]
        if extra.get("dataset_rng") is not None:
            self.dataset.rng.setstate(extra["dataset_rng"])
        if "best_model" in extra:
            self.save(extra["best_model"], self.cfg.BEST_FILENAME)
        for name, state in extra.get("loaders", {}).items():
            getattr(self, name).load_state_dict(state)
        restore_rng(payload["rng"])
        return payload["epoch"]

    def resume(self) -> int:
        if self.cfg.resume_from or self.cfg.resume:
            epoch = self.load_checkpoint()
            print(f"[STAIR5-v6] Resuming after completed epoch {epoch}", flush=True)
            telemetry_path = self._artifact("training_telemetry.jsonl")
            if telemetry_path.is_file():
                valid_lines = []
                with telemetry_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            if d.get("epoch", 0) <= epoch:
                                valid_lines.append(line.strip())
                        except Exception:
                            pass
                with telemetry_path.open("w", encoding="utf-8") as f:
                    for line in valid_lines:
                        f.write(line + "\n")
            self._completed_epochs = epoch
            return epoch
        else:
            telemetry_path = self._artifact("training_telemetry.jsonl")
            if telemetry_path.is_file():
                raise FileExistsError("Fresh-run artifact directory already contains telemetry; use a new directory or --resume-from.")
        self._completed_epochs = 0
        return 0

    def train_per_epoch(self, epoch: int):
        self.model.current_epoch = epoch
        self._epoch_in_progress = True
        started = time.perf_counter()
        totals = torch.zeros(3, device=self.device, dtype=torch.float64)
        samples, batches = 0, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch_index, data in enumerate(self.dataloader):
            data = self.dict_to_device(data)
            self.optimizer.zero_grad(set_to_none=True)
            loss, bpr, cl = self.model.training_objective(data)
            loss.backward()
            self.optimizer.step()

            batch_samples = len(data[self.model.User])
            samples += batch_samples
            batches += 1
            totals[0] += loss.detach().to(torch.float64) * batch_samples
            totals[1] += bpr.detach().to(torch.float64) * batch_samples
            totals[2] += cl.detach().to(torch.float64) * batch_samples
            self.monitor(loss.item(), n=batch_samples, mode="mean", pool="train")

        seconds = time.perf_counter() - started
        means = (totals / max(1, samples)).tolist()
        vram_alloc = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        vram_res = torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0

        record = {
            "epoch": int(epoch),
            "seconds": float(seconds),
            "samples": int(samples),
            "batches": int(batches),
            "total_loss": float(means[0]),
            "bpr_loss": float(means[1]),
            "cl_loss": float(means[2]),
            "max_memory_allocated_bytes": int(vram_alloc),
            "max_memory_reserved_bytes": int(vram_res),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        with self._artifact("training_telemetry.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        self._epoch_in_progress = False
        self._completed_epochs = epoch
        return means[0]

    def fit(self):
        start_epoch = self.resume()
        try:
            for epoch in range(start_epoch + 1, self.cfg.epochs + 1):
                self.epoch = epoch
                self.train_per_epoch(epoch)
                if epoch % self.cfg.eval_freq == 0:
                    self.evaluate(epoch)
                if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                    self.save_checkpoint(epoch)
                self.lr_scheduler.step()
        finally:
            self.save_last()
            self.eval_at_best()


def main():
    cfg = build_config()
    dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset)
    model = STAIR5_v6_Model(dataset, cfg)
    coach = CoachForSTAIR5_v6(trainpipe=model.sure_trainpipe(cfg.batch_size),
                              device=cfg.device, dataset=dataset, model=model, cfg=cfg)

    # Export execution manifest
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        pass

    manifest = {
        "architecture": "STAIR5-v6 (NLGCL-BCSR)",
        "version": 6,
        "arm": model.v6_arm,
        "git_commit": commit,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "freerec": freerec.__version__,
        "dataset": cfg.dataset,
        "users": model.User.count,
        "items": model.Item.count,
        "data_fingerprint": model.data_fingerprint,
        "graph_fingerprint": model.graph_fingerprint,
        "graph_metadata": model.graph_metadata,
        "config": {
            k: getattr(cfg, k)
            for k in (
                "embedding_dim", "num_layers", "gamma", "lr", "weight_decay",
                "beta1", "beta2", "batch_size", "epochs", "seed", "eval_freq",
                "ranking", "which4best", "eta", "k_cf", "c_min", "t_shrinkage",
                "v6_arm", "v6_theta", "v6_t_rel", "lambda_nlgcl", "nlgcl_tau",
                "nlgcl_G", "nlgcl_alpha"
            )
        },
    }
    (Path(cfg.artifact_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )

    coach.compile()
    coach.fit()


if __name__ == "__main__":
    main()
