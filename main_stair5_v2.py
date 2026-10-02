"""STAIR5-v2 training engine with unchanged FreeRec ranking/selection rules.

Example: python main_stair5_v2.py --config configs/Amazon2014Sports_STAIR5_v2.yaml
"""
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import models.freerec_compat  # Install compatibility before importing FreeRec.
import freerec
import torch

from models.stair5_v2 import ARMS, STAIR5_v2_Model
from models.stair5_v2_utils import (
    atomic_torch_save, load_training_checkpoint, save_training_checkpoint,
)
from optimizers.AdamW import AdamWSEvo


def build_config():
    cfg = freerec.parser.Parser()
    for flag, kind, default in (
        ("embedding-dim", int, 64), ("num-layers", int, 3),
        ("mfiles", str, "textual_modality.pkl,visual_modality.pkl"),
        ("num-neighbors", str, "5-1"), ("gamma", float, .2),
        ("beta1", float, .9), ("beta2", float, .999),
        ("edge-strength", float, .25), ("edge-mix", float, .25),
        ("evidence-shrinkage", float, 5.), ("placebo-seed", int, 1),
        ("lambda-lhc", float, 3e-4), ("lhc-tau", float, .3),
        ("lhc-kappa", float, 1.), ("radius-cap", float, 2.),
        ("eps-taylor", float, 1e-4), ("eps-floor", float, .01), ("eps-beta", float, .05),
        ("warmup-start", int, 20), ("warmup-end", int, 50),
        ("grad-diag-every", int, 10), ("knn-chunk-size", int, 1024),
        ("lhc-chunk-size", int, 256), ("graph-cache-dir", str, ""),
        ("artifact-dir", str, ""), ("resume-from", str, ""),
    ):
        cfg.add_argument("--"+flag, type=kind, default=default)
    cfg.add_argument("--v2-arm", choices=ARMS, default="ET")
    cfg.add_argument("--knn-device", choices=("cpu", "cuda", "auto"), default="cpu")
    cfg.set_defaults(description="STAIR5-v2-C-HET", root="data",
                     dataset="Amazon2014Sports_550_MMRec", epochs=500, batch_size=1024,
                     optimizer="adamwsevo", lr=1e-3, weight_decay=.1, seed=1,
                     monitors=["LOSS", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
                     which4best="NDCG@20")
    cfg.compile()
    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("C-HET requires AdamWSEvo; other optimizers bypass BSC.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Select by validation NDCG@20; disable test evaluation during tuning.")
    if cfg.seed < 0:
        raise ValueError("Use an explicit nonnegative seed for reproducibility.")
    if cfg.embedding_dim <= 0 or cfg.num_layers < 0 or not math.isfinite(cfg.gamma) or cfg.gamma <= 0:
        raise ValueError("Require embedding_dim > 0, num_layers >= 0 and finite gamma > 0.")
    cfg.mfiles = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else list(cfg.mfiles)
    cfg.num_neighbors = list(map(int, cfg.num_neighbors.split("-"))) if isinstance(cfg.num_neighbors, str) else cfg.num_neighbors
    cfg.beta3 = (.1 + .9*(torch.arange(cfg.embedding_dim)/cfg.embedding_dim).pow(cfg.gamma)).to(cfg.device)
    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH)/"c_het_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    return cfg


class CoachForSTAIR5_v2(freerec.launcher.Coach):
    def set_optimizer(self):
        self.optimizer = AdamWSEvo(self.model.marked_params(), lr=self.cfg.lr,
                                  betas=(self.cfg.beta1, self.cfg.beta2), weight_decay=self.cfg.weight_decay)

    def _artifact(self, name):
        return Path(self.cfg.artifact_dir)/name

    def save_best(self):
        # Preserve native model-file format/selection and export a durable copy.
        super().save_best()
        atomic_torch_save(self.model.state_dict(), self._artifact("best_model.pt"))

    def save_last(self):
        super().save_last()
        atomic_torch_save(self.model.state_dict(), self._artifact("last_model.pt"))
        self.save_checkpoint(self.model.current_epoch)

    def save_checkpoint(self, epoch):
        extra = {"monitors": self.monitors.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(),
                 "best": self._best, "best_epoch": self._best_epoch, "best_step": self._best_step,
                 "stopping_steps": self._stopping_steps,
                 "dataset_rng": self.dataset.rng.getstate() if hasattr(self.dataset, "rng") else None}
        best_path = Path(self.cfg.LOG_PATH)/self.cfg.BEST_FILENAME
        if best_path.exists():
            extra["best_model"] = torch.load(best_path, map_location="cpu", weights_only=True)
        target = self._artifact("training_checkpoint.pt")
        save_training_checkpoint(target, self.model, self.optimizer, epoch, extra)

    def load_checkpoint(self):
        target = self.cfg.resume_from or self._artifact("training_checkpoint.pt")
        payload = load_training_checkpoint(target, self.model, self.optimizer)
        extra = payload["extra"]
        self.monitors.load_state_dict(extra["monitors"])
        self.lr_scheduler.load_state_dict(extra["lr_scheduler"])
        self._best, self._best_epoch, self._best_step = extra["best"], extra["best_epoch"], extra["best_step"]
        self._stopping_steps = extra["stopping_steps"]
        if extra.get("dataset_rng") is not None:
            self.dataset.rng.setstate(extra["dataset_rng"])
        if "best_model" in extra:
            self.save(extra["best_model"], self.cfg.BEST_FILENAME)
        return payload["epoch"]

    def resume(self):
        if self.cfg.resume_from or self.cfg.resume:
            epoch = self.load_checkpoint()
            print(f"[C-HET] Resuming after completed epoch {epoch}", flush=True)
            return epoch
        return 0

    def train_per_epoch(self, epoch):
        self.model.update_epoch(epoch)
        started = time.perf_counter()
        total_bpr, total_cl, samples, batches = 0., 0., 0, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch_index, data in enumerate(self.dataloader):
            data = self.dict_to_device(data)
            self.optimizer.zero_grad(set_to_none=True)
            loss, bpr, auxiliary = self.model.training_objective(data)
            interval = self.cfg.grad_diag_every
            if interval > 0 and epoch % interval == 0 and batch_index == 0 and self.model.current_lambda > 0:
                parameters = [self.model.User.embeddings.weight, self.model.Item.embeddings.weight]
                bpr_gradients = torch.autograd.grad(bpr, parameters, retain_graph=True)
                cl_gradients = torch.autograd.grad(auxiliary, parameters, retain_graph=True)
                values = {}
                for name, g_bpr, g_cl in zip(("user", "item"), bpr_gradients, cl_gradients):
                    norm_bpr, norm_cl = g_bpr.norm(), g_cl.norm()
                    values[name+"_raw_cosine"] = float((g_bpr*g_cl).sum()/(norm_bpr*norm_cl).clamp_min(1e-12))
                    values[name+"_weighted_norm_ratio"] = float(self.model.current_lambda*norm_cl/norm_bpr.clamp_min(1e-12))
                print(f"[Raw gradient probe epoch={epoch} batch=0] "+json.dumps(values), flush=True)
                del bpr_gradients, cl_gradients
            loss.backward()
            self.optimizer.step()
            count = len(data[self.User])
            self.monitor(loss.item(), n=count, reduction="mean", mode="train", pool=["LOSS"])
            diagnostics = self.model.last_diagnostics
            total_bpr += diagnostics["bpr_loss"]*count
            total_cl += diagnostics["cl_loss"]*count
            samples += count
            batches += 1
        if not batches:
            raise RuntimeError("Training dataloader produced no batches.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter()-started
        record = {"epoch": epoch, "bpr_loss": total_bpr/samples, "cl_loss": total_cl/samples,
                  "lambda": self.model.current_lambda, "samples": samples, "batches": batches,
                  "seconds": elapsed, "samples_per_second": samples/elapsed,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device)/2**30 if self.device.type == "cuda" else 0.}
        with self._artifact("training_telemetry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record)+"\n")
        print(f"[Epoch {epoch:03d} Telemetry] BPR: {record['bpr_loss']:.6f} | "
              f"LHC: {record['cl_loss']:.6f} (lambda={self.model.current_lambda:.6f}) | "
              f"{record['samples_per_second']:.1f} samples/s | {elapsed:.2f}s", flush=True)
        # A completed-epoch checkpoint enables resume even after a session ends.
        if epoch % self.cfg.CHECKPOINT_FREQ == 0 or epoch == self.cfg.epochs:
            self.save_checkpoint(epoch)


def load_dataset(cfg):
    from freerec.data.datasets.base import MatchingRecDataSet
    processed = Path(cfg.root)/"Processed"/cfg.dataset
    if not processed.is_dir():
        direct = Path(cfg.root)/cfg.dataset
        if not direct.is_dir():
            raise FileNotFoundError(f"Expected processed dataset at {processed}; prepare data before training.")
        processed.parent.mkdir(parents=True, exist_ok=True)
        # Never silently copy a large dataset or replace an existing directory.
        processed.symlink_to(direct.resolve(), target_is_directory=True)
    return MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=freerec.data.tags.MATCHING)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = build_config()
    # Match baseline FP32 math; record optional GPU preprocessing separately.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    dataset = load_dataset(cfg)
    model = STAIR5_v2_Model(dataset, cfg)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    manifest = {"argv": sys.argv, "commit": commit, "python": platform.python_version(),
                "torch": str(torch.__version__), "freerec": freerec.__version__,
                "device": str(cfg.device), "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "arm": cfg.v2_arm, "seed": cfg.seed, "graph": model.graph_metadata,
                "features": model.feature_manifest, "preprocessing_seconds": time.perf_counter()-started,
                "knn_preprocessing": model.knn_metadata,
                "ranking": cfg.ranking, "selection": "validation NDCG@20",
                "knn_device": cfg.knn_device, "knn_chunk_size": cfg.knn_chunk_size,
                "lhc_chunk_size": cfg.lhc_chunk_size,
                "baseline_config": {key: getattr(cfg, key) for key in (
                    "embedding_dim", "num_layers", "gamma", "lr", "weight_decay", "batch_size",
                    "epochs", "beta1", "beta2", "mfiles", "num_neighbors", "eval_freq")},
                "freerec_log_path": cfg.LOG_PATH,
                "test_records": "Native FreeRec final and selected; report selected only."}
    Path(cfg.artifact_dir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[C-HET graph] "+json.dumps(model.graph_metadata), flush=True)
    coach = CoachForSTAIR5_v2(dataset=dataset, model=model, cfg=cfg,
                            trainpipe=model.sure_trainpipe(cfg.batch_size),
                            validpipe=model.sure_validpipe(cfg.ranking), testpipe=model.sure_testpipe(cfg.ranking))
    coach.fit()


if __name__ == "__main__":
    main()
