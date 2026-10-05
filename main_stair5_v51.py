"""main_stair5_v51.py — STAIR5-v5.1 Training Engine: NLGCL-KPE.
==============================================================
Neighborhood-enriched Graph Contrastive Learning with Known-Positive Exclusion (KPE).

Usage Example:
    python main_stair5_v51.py --config configs/Amazon2014Sports_STAIR5_v51.yaml --v51-arm V51-KPE
"""
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import models.freerec_compat  # Fail visibly if required compatibility cannot load.
import freerec
import torch

from models.stair5_v51 import ARMS_V51, STAIR5_v51_Model
from models.stair5_v51_utils import (
    atomic_torch_save,
    compute_kpe_diagnostics,
    load_training_checkpoint,
    restore_rng,
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
        ("cam-delta", float, 0.05),
        ("placebo-seed", int, 1),
        ("lambda-nlgcl", float, 0.01),
        ("nlgcl-tau", float, 0.2),
        ("nlgcl-G", int, 1),
        ("nlgcl-alpha", float, 0.5),
        ("nlgcl-rho", float, 1.0),
        ("cl-chunk-size", int, 1024),
        ("knn-chunk-size", int, 1024),
        ("cf-block-size", int, 64),
        ("cf-memory-budget-mib", float, 128.0),
        ("graph-cache-dir", str, ""),
        ("artifact-dir", str, ""),
        ("resume-from", str, ""),
    ):
        cfg.add_argument("--" + flag, type=kind, default=default)

    cfg.add_argument("--v51-arm", choices=ARMS_V51, default="V51-KPE")
    cfg.add_argument("--knn-device", choices=("cpu", "cuda", "auto"), default="auto")

    cfg.set_defaults(
        description="STAIR5-v5.1-NLGCL-KPE",
        root="data",
        dataset="Amazon2014Sports_550_MMRec",
        epochs=500,
        batch_size=1024,
        optimizer="adamwsevo",
        lr=1e-3,
        weight_decay=0.1,
        seed=1,
        monitors=["LOSS", "Recall@1", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
        which4best="NDCG@20",
    )
    cfg.compile()

    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("STAIR5-v5.1 requires AdamWSEvo optimizer with BSC smoother.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Must select by validation NDCG@20; disable test evaluation during training.")
    if cfg.ranking != "full":
        raise ValueError("Reference experiments require full ranking; pool scorer remains available for testing.")
    if not math.isfinite(cfg.eta) or not 0 <= cfg.eta <= 1 or cfg.k_cf < 0 or cfg.c_min < 1:
        raise ValueError("Require eta in [0, 1], k_cf >= 0 and c_min >= 1.")
    if not math.isfinite(cfg.t_shrinkage) or cfg.t_shrinkage <= 0:
        raise ValueError("Require finite t_shrinkage > 0.")
    if not math.isfinite(cfg.cam_delta) or cfg.cam_delta < 0:
        raise ValueError("Require finite cam_delta >= 0.")
    if not math.isfinite(cfg.lambda_nlgcl) or cfg.lambda_nlgcl < 0 or not math.isfinite(cfg.nlgcl_tau) or cfg.nlgcl_tau <= 0:
        raise ValueError("Require finite lambda_nlgcl >= 0 and nlgcl_tau > 0.")
    if cfg.nlgcl_G < 1 or not math.isfinite(cfg.nlgcl_alpha) or not 0 <= cfg.nlgcl_alpha <= 1:
        raise ValueError("Require nlgcl_G >= 1 and nlgcl_alpha in [0, 1].")
    if not math.isfinite(cfg.nlgcl_rho) or not 0 <= cfg.nlgcl_rho <= 1:
        raise ValueError("Require nlgcl_rho in [0, 1].")
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

    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH) / "nlgcl_kpe_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    return cfg


class CoachForSTAIR5_v51(freerec.launcher.Coach):
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
        self._best, self._best_epoch, self._best_step = (
            extra["best"],
            extra["best_epoch"],
            extra["best_step"],
        )
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
            print(f"[STAIR5-v5.1] Resuming after completed epoch {epoch}", flush=True)
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
                raise FileExistsError(
                    "Fresh-run artifact directory already contains telemetry; use a new directory or --resume-from."
                )
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

            # Run preflight diagnostics on the very first batch of epoch 1
            if epoch == 1 and batch_index == 0:
                try:
                    diag = compute_kpe_diagnostics(
                        train_index=self.model.train_pair_keys,
                        user_ids=data[self.User],
                        item_ids=data[self.Item],
                        n_items=self.model.Item.count,
                        chunk_size=self.cfg.cl_chunk_size,
                    )
                    with self._artifact("diagnostics_epoch1.json").open("w", encoding="utf-8") as dh:
                        json.dump(diag, dh, indent=2)
                except Exception as e:
                    print(f"[Diagnostics Warning] Could not record batch diagnostics: {e}", flush=True)

            count = len(data[self.User])
            totals += torch.stack((loss.detach(), bpr.detach(), cl.detach())).to(torch.float64) * count
            samples += count
            batches += 1

        if not batches:
            raise RuntimeError("Training dataloader produced no batches.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        averages = (totals / samples).cpu().tolist()
        self.monitor(averages[0], n=samples, reduction="mean", mode="train", pool=["LOSS"])
        self._completed_epochs = epoch
        self._epoch_in_progress = False
        elapsed = time.perf_counter() - started

        peak_allocated_gib = (
            torch.cuda.max_memory_allocated(self.device) / (1024**3)
            if self.device.type == "cuda"
            else 0.0
        )
        vram_allocated_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024**2)
            if self.device.type == "cuda"
            else 0.0
        )

        record = {
            "epoch": epoch,
            "loss": averages[0],
            "bpr_loss": averages[1],
            "nlgcl_loss": averages[2],
            "weighted_nlgcl_loss": self.model.lambda_nlgcl * averages[2],
            "lambda_nlgcl": self.model.lambda_nlgcl,
            "samples": samples,
            "batches": batches,
            "seconds": elapsed,
            "samples_per_second": samples / elapsed,
            "peak_allocated_gib": peak_allocated_gib,
            "vram_allocated_mb": vram_allocated_mb,
        }
        with self._artifact("training_telemetry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"[Epoch {epoch:03d} Telemetry] Loss: {record['loss']:.6f} | "
            f"BPR: {record['bpr_loss']:.6f} | CL: {record['nlgcl_loss']:.6f} | "
            f"VRAM: {vram_allocated_mb:.2f} MiB | {record['samples_per_second']:.1f} samples/s | {elapsed:.2f}s",
            flush=True,
        )


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

            shutil.copytree(str(direct.resolve()), str(processed), dirs_exist_ok=True)

    return MatchingRecDataSet(cfg.root, cfg.dataset, tasktag=freerec.data.tags.MATCHING)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    cfg = build_config()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    artifact = Path(cfg.artifact_dir)
    if not (cfg.resume or cfg.resume_from) and any(
        (artifact / name).exists()
        for name in ("manifest.json", "training_checkpoint.pt", "training_telemetry.jsonl")
    ):
        raise FileExistsError(
            "Artifact directory belongs to an existing attempt; use a new directory or explicit resume."
        )
    started = time.perf_counter()
    dataset = load_dataset(cfg)
    model = STAIR5_v51_Model(dataset, cfg)

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
        "arm": cfg.v51_arm,
        "seed": cfg.seed,
        "graph": model.graph_metadata,
        "features": model.feature_manifest,
        "knn": model.knn_metadata,
        "checkpoint_contract": model.get_extra_state(),
        "effective_lambda_nlgcl": model.lambda_nlgcl,
        "precision": {"tf32": False, "amp": False},
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
                "nlgcl_rho",
                "eta",
                "k_cf",
                "c_min",
                "t_shrinkage",
                "cam_delta",
                "cl_chunk_size",
                "knn_chunk_size",
                "knn_device",
                "cf_block_size",
                "cf_memory_budget_mib",
            )
        },
        "freerec_log_path": cfg.LOG_PATH,
        "test_records": "Native FreeRec final and selected; report selected only.",
    }
    manifest_name = "resume_manifest.json" if cfg.resume or cfg.resume_from else "manifest.json"
    Path(cfg.artifact_dir, manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[STAIR5-v5.1 graph] " + json.dumps(model.graph_metadata), flush=True)

    coach = CoachForSTAIR5_v51(
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

