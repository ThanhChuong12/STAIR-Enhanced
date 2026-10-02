"""main_stair5_v4.py — STAIR5-v4 Training Engine: NLGCL-CSE.
============================================================
Neighborhood-enriched Graph Contrastive Learning with Candidate Support Expansion.

Example:
    python main_stair5_v4.py --config configs/Amazon2014Sports_STAIR5_v4.yaml --v4-arm N-CSE
"""
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

try:
    import models.freerec_compat
except Exception:
    try:
        import freerec_compat
    except Exception:
        pass

import freerec
import torch

from models.stair5_v4 import ARMS_V4, STAIR5_v4_Model
from models.stair5_v4_utils import (
    atomic_torch_save,
    load_training_checkpoint,
    save_training_checkpoint,
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
        ("placebo-seed", int, 1),
        ("lambda-nlgcl", float, 0.01),
        ("nlgcl-tau", float, 0.2),
        ("nlgcl-G", int, 1),
        ("nlgcl-alpha", float, 0.5),
        ("cl-chunk-size", int, 1024),
        ("graph-cache-dir", str, ""),
        ("artifact-dir", str, ""),
        ("resume-from", str, ""),
    ):
        cfg.add_argument("--" + flag, type=kind, default=default)

    cfg.add_argument("--v4-arm", choices=ARMS_V4, default="N-CSE")

    cfg.set_defaults(
        description="STAIR5-v4-NLGCL-CSE",
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

    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("STAIR5-v4 requires AdamWSEvo optimizer with BSC smoother.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Must select by validation NDCG@20; disable test evaluation during training.")
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
    cfg.beta3 = (
        0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    ).to(cfg.device)

    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH) / "nlgcl_cse_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    return cfg


class CoachForSTAIR5_v4(freerec.launcher.Coach):
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
        self.save_checkpoint(self.model.current_epoch if hasattr(self.model, "current_epoch") else self.epochs)

    def save_checkpoint(self, epoch: int):
        extra = {
            "monitors": self.monitors.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self._best,
            "best_epoch": self._best_epoch,
            "best_step": self._best_step,
            "stopping_steps": self._stopping_steps,
            "dataset_rng": self.dataset.rng.getstate() if hasattr(self.dataset, "rng") else None,
        }
        best_path = Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME
        if best_path.exists():
            extra["best_model"] = torch.load(best_path, map_location="cpu", weights_only=True)
        target = self._artifact("training_checkpoint.pt")
        save_training_checkpoint(target, self.model, self.optimizer, epoch, extra)

    def load_checkpoint(self) -> int:
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

    def resume(self) -> int:
        if self.cfg.resume_from or self.cfg.resume:
            epoch = self.load_checkpoint()
            print(f"[STAIR5-v4] Resuming after completed epoch {epoch}", flush=True)
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
            return epoch
        else:
            telemetry_path = self._artifact("training_telemetry.jsonl")
            if telemetry_path.is_file():
                telemetry_path.unlink()
        return 0

    def train_per_epoch(self, epoch: int):
        self.model.current_epoch = epoch
        started = time.perf_counter()
        total_loss, samples, batches = 0.0, 0, 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch_index, data in enumerate(self.dataloader):
            data = self.dict_to_device(data)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model.fit(data)
            loss.backward()
            self.optimizer.step()

            count = len(data[self.User])
            self.monitor(loss.item(), n=count, reduction="mean", mode="train", pool=["LOSS"])
            total_loss += loss.item() * count
            samples += count
            batches += 1

        if not batches:
            raise RuntimeError("Training dataloader produced no batches.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "loss": total_loss / samples,
            "lambda_nlgcl": self.model.lambda_nlgcl,
            "samples": samples,
            "batches": batches,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(self.device) / 2**30
                if self.device.type == "cuda"
                else 0.0
            ),
        }
        with self._artifact("training_telemetry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"[Epoch {epoch:03d} Telemetry] Loss: {record['loss']:.6f} | "
            f"lambda_nlgcl={self.model.lambda_nlgcl:.4f} | "
            f"{record['samples_per_second']:.1f} samples/s | {elapsed:.2f}s",
            flush=True,
        )

        if epoch % self.cfg.CHECKPOINT_FREQ == 0 or epoch == self.cfg.epochs:
            self.save_checkpoint(epoch)


def load_dataset(cfg):
    from freerec.data.datasets.base import MatchingRecDataSet

    processed = Path(cfg.root) / "Processed" / cfg.dataset
    if not processed.is_dir():
        direct = Path(cfg.root) / cfg.dataset
        if not direct.is_dir():
            for cand_root in [
                "/kaggle/data",
                "/kaggle/working/STAIR-Enhanced/data",
                "data",
                "../data",
                str(Path(cfg.root).parent),
            ]:
                p_cand = Path(cand_root) / "Processed" / cfg.dataset
                d_cand = Path(cand_root) / cfg.dataset
                if p_cand.is_dir():
                    direct = p_cand
                    break
                elif d_cand.is_dir():
                    direct = d_cand
                    break
            else:
                raise FileNotFoundError(
                    f"Expected dataset at {processed} or {direct}; prepare data before training."
                )
        processed.parent.mkdir(parents=True, exist_ok=True)
        try:
            processed.symlink_to(direct.resolve(), target_is_directory=True)
        except Exception:
            import shutil

            shutil.copytree(direct.resolve(), processed, dirs_exist_ok=True)
    return MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=freerec.data.tags.MATCHING)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = build_config()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    started = time.perf_counter()
    dataset = load_dataset(cfg)
    model = STAIR5_v4_Model(dataset, cfg)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    manifest = {
        "argv": sys.argv,
        "commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "freerec": freerec.__version__,
        "device": str(cfg.device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "arm": cfg.v4_arm,
        "seed": cfg.seed,
        "graph": model.graph_metadata,
        "features": model.feature_manifest,
        "preprocessing_seconds": time.perf_counter() - started,
        "ranking": cfg.ranking,
        "selection": "validation NDCG@20",
        "baseline_config": {
            key: getattr(cfg, key)
            for key in (
                "embedding_dim",
                "num_layers",
                "gamma",
                "lr",
                "weight_decay",
                "batch_size",
                "epochs",
                "beta1",
                "beta2",
                "mfiles",
                "num_neighbors",
                "eval_freq",
                "lambda_nlgcl",
                "nlgcl_tau",
                "nlgcl_G",
                "nlgcl_alpha",
                "eta",
                "k_cf",
                "c_min",
                "t_shrinkage",
            )
        },
        "freerec_log_path": cfg.LOG_PATH,
        "test_records": "Native FreeRec final and selected; report selected only.",
    }
    Path(cfg.artifact_dir, "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[STAIR5-v4 graph] " + json.dumps(model.graph_metadata), flush=True)

    coach = CoachForSTAIR5_v4(
        dataset=dataset,
        model=model,
        cfg=cfg,
        trainpipe=model.sure_trainpipe(cfg.batch_size),
        validpipe=model.sure_validpipe(cfg.ranking),
        testpipe=model.sure_testpipe(cfg.ranking),
    )
    coach.fit()


if __name__ == "__main__":
    main()
