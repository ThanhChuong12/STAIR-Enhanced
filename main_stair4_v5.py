"""Two-stage STAIR-RAM training with unchanged FreeRec ranking and metrics.

Examples:
    python main_stair4_v5.py --config configs/dataset_stair4_v5_baby.yaml --stage all
    python main_stair4_v5.py --stage b --teacher-dir RUN/stage_a --run-dir NEW_RUN ...

Test metrics are emitted only for the validation-selected checkpoint. Use
--no-final-test for development/pilot runs. No CLI is compiled during import.
"""
from __future__ import annotations

import copy
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import models.freerec_compat
import freerec

from main_stair4_v4 import InheritingParser, load_config, build_dataset, CoachForSTAIR4V4
from models.stair4_v4 import STAIR4V4
from models.stair4_v4_utils import (
    atomic_save, capture_rng, restore_rng, capture_sampler, restore_sampler,
    array_hash, identity_hash,
)
from models.stair4_v5 import STAIR4V5, STAIR4V5Continuation
from models.stair4_v5_features import prepare_modality_features, TrainHistory, build_train_csr
from models.stair4_v5_sampling import UniformTrainUnseenSampler
from models.stair4_v5_objectives import masked_candidate_cross_entropy
from models.stair4_v5_utils import (
    write_json, file_hash, implementation_hash, baseline_source_hash, evaluator_hash,
    dataset_fingerprint, calibrate_scale, id_only_features, read_teacher,
)
from optimizers.AdamW import AdamWSEvo


def make_parser():
    cfg = InheritingParser()
    for name, kind, default in (
        ("embedding-dim", int, 64), ("num-layers", int, 3), ("gamma", float, .1),
        ("mfiles", str, "textual_modality.pkl,visual_modality.pkl"),
        ("num-neighbors", str, "5-1"), ("knn-block-size", int, 256),
        ("graph-cache-dir", str, ".cache/stair4_v4"), ("feature-cache-dir", str, ".cache/stair4_v5"),
        ("head-epochs", int, 100), ("head-lr", float, .001), ("gate-lr", float, .001),
        ("amplitude-lr", float, .0001), ("head-weight-decay", float, .0001),
        ("eta-scale", float, .5), ("calibration-samples", int, 100_000),
        ("pca-dim", int, 128), ("residual-dim", int, 32),
        ("num-negatives", int, 32), ("temperature", float, 1.),
        ("microbatch-size", int, 512), ("head-patience", int, 6),
        ("clip-norm", float, 5.), ("finite-check-interval", int, 0),
    ):
        cfg.add_argument("--" + name, type=kind, default=default)
    cfg.add_argument("--stage", choices=("a", "b", "all"), default="all")
    cfg.add_argument("--arm", choices=("mm-ss", "mm-bpr", "id-ss", "baseline-ss", "baseline-bpr"), default="mm-ss")
    cfg.add_argument("--gate-mode", choices=("personalized", "global", "fixed"), default="personalized")
    cfg.add_argument("--pca-algorithm", choices=("randomized", "exact"), default="randomized")
    cfg.add_argument("--run-dir", default=None)
    cfg.add_argument("--teacher-dir", default=None)
    cfg.add_argument("--no-final-test", action="store_true")
    cfg.set_defaults(description="STAIR-RAM", root="data", dataset="Amazon2014Baby_550_MMRec",
                     epochs=500, batch_size=1024, optimizer="adamwsevo", lr=.001,
                     weight_decay=.3, seed=1, num_workers=0, eval_freq=5,
                     eval_valid=True, eval_test=False, ranking="full", retain_seen=False,
                     monitors=["LOSS", "Recall@1", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20"],
                     which4best="NDCG@20")
    return cfg


def validate_config(cfg):
    for name in ("epochs", "head_epochs", "embedding_dim", "num_layers", "batch_size",
                 "eval_freq", "pca_dim", "residual_dim", "num_negatives",
                 "microbatch_size", "calibration_samples", "head_patience", "knn_block_size"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("head_lr", "gate_lr", "amplitude_lr", "eta_scale", "temperature", "clip_norm"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.head_weight_decay) or cfg.head_weight_decay < 0:
        raise ValueError("head_weight_decay must be finite and nonnegative")
    if cfg.which4best.upper() != "NDCG@20" or not cfg.eval_valid or cfg.eval_test:
        raise ValueError("Use validation NDCG@20 selection; per-epoch test evaluation is disabled")
    if cfg.optimizer.lower() != "adamwsevo" or cfg.retain_seen:
        raise ValueError("The baseline contract requires AdamWSEvo and seen-item masking")
    if cfg.num_workers != 0:
        raise ValueError("Verified deterministic loaders/resume currently require --num-workers 0")
    if cfg.stage == "b" and not cfg.teacher_dir:
        raise ValueError("--stage b requires a verified --teacher-dir from Stage A")
    if cfg.stage != "b" and cfg.teacher_dir:
        raise ValueError("--teacher-dir is only used with --stage b")
    if len(cfg.mfiles.split(",")) != 2:
        raise ValueError("The registered v5 core requires text and image feature files")
    if cfg.arm.endswith("bpr") and cfg.temperature != 1.:
        raise ValueError("BPR control requires temperature=1")
    # Never inherit calibration from a v4 treatment configuration.
    cfg.alpha, cfg.identity_mix, cfg.ablation_id = 0., 0., "V4-B1"


def baseline_protocol(cfg):
    return {**{k: getattr(cfg, k) for k in (
        "embedding_dim", "num_layers", "gamma", "mfiles", "num_neighbors",
        "lr", "weight_decay", "batch_size", "epochs", "ranking", "retain_seen", "eval_freq")},
        "selection": "NDCG@20", "evaluator_hash": evaluator_hash(),
        "torch": str(torch.__version__), "freerec": str(freerec.__version__),
        "device": str(cfg.device), "cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "optimizer": "AdamWSEvo", "beta1": cfg.optim_first_moment_decay,
        "beta2": cfg.optim_second_moment_decay, "num_workers": cfg.num_workers}


def phase_config(cfg, directory, *, stage_b=False):
    result = copy.copy(cfg)
    result.LOG_PATH = str(directory)
    result.CHECKPOINT_PATH = str(directory / "checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    Path(result.CHECKPOINT_PATH).mkdir(exist_ok=True)
    (directory / result.SUMMARY_DIR).mkdir(exist_ok=True)
    if stage_b:
        result.epochs = cfg.head_epochs
    return result


def optimizer_state_without_smoother(optimizer):
    state = optimizer.state_dict()
    return {**state, "param_groups": [{k: v for k, v in g.items() if k != "smoother"}
                                     for g in state["param_groups"]]}


def guard_run_directory(directory, resume):
    path = directory / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not resume or manifest.get("status") == "completed":
            raise FileExistsError(f"Run already exists (completed runs cannot resume): {directory}")
        evaluations = directory / "evaluation.jsonl"
        if evaluations.exists() and any(json.loads(line).get("mode") == "test"
                for line in evaluations.read_text(encoding="utf-8").splitlines() if line.strip()):
            raise ValueError("Cannot resume optimization after this run has disclosed test metrics")
    elif resume:
        raise FileNotFoundError(f"No run manifest to resume: {directory}")


class BaselineCoach(CoachForSTAIR4V4):
    """Baseline training/selection, plus selected optimizer export and test gating."""

    def save_best(self):
        atomic_save(Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME, self.model.state_dict())
        atomic_save(Path(self.cfg.LOG_PATH) / "selected_training.pt", {
            "optimizer": optimizer_state_without_smoother(self.optimizer),
            "rng": capture_rng(), "train_sampler": capture_sampler(self.trainloader.datapipe),
            "epoch": int(self._best_epoch),
        })

    def test(self, epoch, step=-1):
        # FreeRec.fit also requests a final-epoch test; publish only selected test.
        if getattr(self, "_evaluating_selected", False) and not self.cfg.no_final_test:
            return super().test(epoch, step)

    def eval_at_best(self):
        self._evaluating_selected = True
        try:
            self.load_best()
            self.valid(self._best_epoch, self._best_step)
            self.test(self._best_epoch, self._best_step)
        finally:
            self._evaluating_selected = False

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        payload = torch.load(path, map_location="cpu", weights_only=True)
        selected = Path(self.cfg.LOG_PATH) / "selected_training.pt"
        payload["selected_training"] = torch.load(selected, map_location="cpu", weights_only=True) if selected.exists() else None
        atomic_save(path, payload)

    def load_checkpoint(self):
        epoch = super().load_checkpoint()
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("selected_training") is not None:
            atomic_save(Path(self.cfg.LOG_PATH) / "selected_training.pt", payload["selected_training"])
        return epoch

    def fit(self):
        """Preserve baseline update/validation order without plotting side effects."""
        start = self.resume()
        completed = start
        try:
            try:
                for epoch in range(start, self.cfg.epochs):
                    if epoch % self.cfg.CHECKPOINT_FREQ == 0:
                        self.save_checkpoint(epoch)
                    if epoch % self.cfg.eval_freq == 0:
                        self.valid(epoch)
                    self.train(epoch + 1)
                    completed = epoch + 1
                self.valid(self.cfg.epochs)
            except freerec.launcher.EarlyStopError:
                pass
            self.save_last()
            self.save_checkpoint(completed)
            self._stopping_steps = -1
            self.eval_at_best()
        finally:
            self.shutdown()


def run_stage_a(dataset, cfg, directory, fingerprint, hashes):
    guard_run_directory(directory, cfg.resume)
    phase = phase_config(cfg, directory)
    protocol = baseline_protocol(cfg)
    if cfg.resume:
        previous = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        for key, expected in (("source_hash", implementation_hash()), ("data_fingerprint", fingerprint),
                              ("baseline_protocol", protocol), ("seed", cfg.seed)):
            if previous.get(key) != expected:
                raise ValueError(f"Stage A resume {key} differs")
    write_json(directory / "resolved_config.json", json.loads(json.dumps(dict(phase), default=str)))
    manifest = {
        "schema": 1, "status": "preparing", "stage": "a", "arm": "B0",
        "dataset": cfg.dataset, "seed": cfg.seed, "source_hash": implementation_hash(),
        "baseline_source_hash": baseline_source_hash(), "data_fingerprint": fingerprint,
        "data_hashes": hashes, "protocol": protocol, "baseline_protocol": protocol,
        "runtime": {"python": platform.python_version(), "device": str(cfg.device)},
        "test_disclosed": not cfg.no_final_test,
    }
    manifest["comparison_id"] = identity_hash({
        k: manifest[k] for k in ("baseline_source_hash", "data_fingerprint", "baseline_protocol", "seed")
    })
    start = time.perf_counter()
    coach = None
    try:
        model = STAIR4V4(dataset, phase)
        manifest.update(graph_identity=model.graph_identity, preparation_seconds=time.perf_counter() - start)
        write_json(directory / "run_manifest.json", manifest)
        coach = BaselineCoach(dataset=dataset, model=model, cfg=phase,
            trainpipe=model.sure_trainpipe(cfg.batch_size), validpipe=model.sure_validpipe(cfg.ranking),
            testpipe=model.sure_testpipe(cfg.ranking))
        coach.run_dir = directory
        coach.fit()  # Same baseline sampling, updates and validation selection.
        model.eval().requires_grad_(False)
        with torch.no_grad():
            user_vectors, item_vectors = model.encode()
        selected = torch.load(directory / "selected_training.pt", map_location="cpu", weights_only=True)
        if selected["epoch"] != coach._best_epoch:
            raise ValueError("Selected optimizer and model epochs differ")
        manifest.update(selected_epoch=int(coach._best_epoch), selected_valid_ndcg20=float(coach._best))
        atomic_save(directory / "teacher.pt", {
            "schema": 1, "user_vectors": user_vectors.cpu(), "item_vectors": item_vectors.cpu(),
            "model_state": torch.load(directory / phase.BEST_FILENAME, map_location="cpu", weights_only=True),
            "training_state": selected, "data_fingerprint": fingerprint,
            "comparison_id": manifest["comparison_id"], "selected_epoch": manifest["selected_epoch"],
        })
        manifest.update(teacher_hash=file_hash(directory / "teacher.pt"), status="completed")
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter() - start
        write_json(directory / "run_manifest.json", manifest)
    return directory


class ResidualCoach(BaselineCoach):
    """Head training and epoch-boundary resume; evaluator is inherited unchanged."""

    def set_optimizer(self):
        self.continuation = isinstance(self.model, STAIR4V5Continuation)
        if self.continuation:
            return super().set_optimizer()
        groups = self.model.head.parameter_groups(
            projection_lr=self.cfg.head_lr, gate_lr=self.cfg.gate_lr,
            amplitude_lr=self.cfg.amplitude_lr, projection_weight_decay=self.cfg.head_weight_decay)
        self.optimizer = torch.optim.AdamW(groups)

    def set_other(self):
        edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index.cpu().numpy()
        self.train_csr = build_train_csr(edges, self.model.User.count, self.model.Item.count)
        users, items = self.train_csr.nonzero()
        self.pairs = torch.as_tensor(np.stack([users, items]), dtype=torch.long)
        self.order_rng = torch.Generator().manual_seed(self.cfg.seed + 7301)
        self.negative_sampler = UniformTrainUnseenSampler(self.train_csr, seed=self.cfg.seed + 7302)
        self.negative_count = 1 if self.cfg.arm.endswith("bpr") else self.cfg.num_negatives
        self.steps = 0
        self.completed_epoch = 0
        self.head_protocol = {
            "head_epochs": self.cfg.head_epochs, "batch_size": self.cfg.batch_size,
            "microbatch_size": self.cfg.microbatch_size, "eval_freq": self.cfg.eval_freq,
            "num_negatives": self.negative_count, "temperature": self.cfg.temperature,
            "sampler": "uniform_distinct_train_unseen", "seed": self.cfg.seed,
            "head_patience": self.cfg.head_patience,
        }

    def save_best(self):
        atomic_save(Path(self.cfg.LOG_PATH) / self.cfg.BEST_FILENAME, self.model.state_dict())

    def train_per_epoch(self, epoch):
        start = time.perf_counter()
        if not self.continuation:
            self.model.residual_enabled = True
        order = torch.randperm(self.pairs.shape[1], generator=self.order_rng)
        total, count, skipped, clipped = 0., 0, 0, 0
        gradient_sums = {}
        diagnostic_sums = {}
        diagnostic_count = 0
        for begin in range(0, len(order), self.cfg.batch_size):
            user_ids, positives = self.pairs[:, order[begin:begin + self.cfg.batch_size]]
            negatives, mask = self.negative_sampler.sample(user_ids, self.negative_count)
            keep = mask.any(dim=1)
            skipped += int((~keep).sum())
            user_ids, positives, negatives, mask = (x[keep] for x in (user_ids, positives, negatives, mask))
            if not len(user_ids):
                continue
            candidates = torch.cat([positives[:, None], negatives], dim=1)
            self.optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for offset in range(0, len(user_ids), self.cfg.microbatch_size):
                sl = slice(offset, offset + self.cfg.microbatch_size)
                ids, pos, cand, valid = (x[sl].to(self.device) for x in (user_ids, positives, candidates, mask))
                scores = self.model.score_candidates(ids, pos, cand)
                loss = masked_candidate_cross_entropy(scores, valid, temperature=self.cfg.temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite candidate loss: epoch={epoch}, step={self.steps}")
                fraction = len(ids) / len(user_ids)
                (loss * fraction).backward()
                batch_loss += float(loss.detach()) * fraction
                with torch.no_grad():
                    negative_gaps = scores[:, :1] - scores[:, 1:]
                    eligible_gaps = negative_gaps[valid]
                    diagnostic_sums["candidate_margin_sum"] = diagnostic_sums.get("candidate_margin_sum", 0.) + float(eligible_gaps.sum())
                    diagnostic_sums["positive_probability_sum"] = diagnostic_sums.get("positive_probability_sum", 0.) + float(
                        torch.softmax(torch.cat([scores[:, :1], scores[:, 1:].masked_fill(~valid, -torch.inf)], 1)
                                      / self.cfg.temperature, 1)[:, 0].sum())
                    diagnostic_count += int(valid.sum())
            for group in self.optimizer.param_groups:
                grads = [p.grad.detach().norm().square() for p in group["params"] if p.grad is not None]
                norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.
                if not math.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient in {group.get('role', 'head')}")
                key = str(group.get("role", group.get("name", "head")))
                gradient_sums[key] = gradient_sums.get(key, 0.) + norm
            # Preserve the original update in objective-only backbone controls.
            if not self.continuation:
                norm = torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), self.cfg.clip_norm,
                                                     error_if_nonfinite=True)
                clipped += int(norm > self.cfg.clip_norm)
            if self.continuation:
                self.model.smoother.arm_static()
            try:
                self.optimizer.step()
            finally:
                if self.continuation:
                    self.model.smoother.clear_step_snapshot()
            self.model.invalidate_ranking_cache()
            self.steps += 1
            total += batch_loss * len(user_ids)
            count += len(user_ids)
            self.monitor(batch_loss, n=len(user_ids), reduction="mean", mode="train", pool=["LOSS"])
        if not count:
            raise ValueError("No train row has an unobserved candidate")
        record = {"epoch": epoch, "train_seconds": time.perf_counter() - start,
                  "examples": count, "skipped_no_unseen": skipped, "loss": total / count,
                  "steps_total": self.steps, "clipped_steps": clipped, "gradient_norm_sums": gradient_sums,
                  "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
                  "current_allocated_mb": torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
                  "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else 0}
        record.update(mean_candidate_margin=diagnostic_sums["candidate_margin_sum"] / max(1, diagnostic_count),
                      mean_positive_probability=diagnostic_sums["positive_probability_sum"] / count)
        if not self.continuation:
            probe = user_ids[:min(1024, len(user_ids))].to(self.device)
            record["head"] = self.model.head.diagnostics(self.model.h_user[probe], self.model.history.degrees[probe])
        self._append("epochs.jsonl", record)

    def _checkpoint_contract(self):
        return {"model_identity": getattr(self.model, "identity", getattr(self.model, "graph_identity", None)),
                "head_protocol": self.head_protocol, "arm": self.cfg.arm,
                "optimization": {k: getattr(self.cfg, k) for k in ("head_lr", "gate_lr", "amplitude_lr",
                    "head_weight_decay", "clip_norm", "lr", "weight_decay")},
                "torch": str(torch.__version__), "freerec": str(freerec.__version__),
                "source_hash": implementation_hash(), "device": str(self.device)}

    def save_checkpoint(self, epoch):
        path = Path(self.cfg.LOG_PATH)
        atomic_save(path / "checkpoints" / "last.pt", {
            "schema": 1, "completed_epoch": epoch, "contract": self._checkpoint_contract(),
            "model": self.model.state_dict(), "optimizer": optimizer_state_without_smoother(self.optimizer),
            "rng": capture_rng(), "order_rng": self.order_rng.get_state(),
            "negative_sampler": self.negative_sampler.state_dict(), "steps": self.steps,
            "selection": {k: getattr(self, k) for k in ("_best", "_best_epoch", "_best_step", "_stopping_steps")},
            "best_model": torch.load(path / self.cfg.BEST_FILENAME, map_location="cpu", weights_only=True),
            "eval_samplers": {name: capture_sampler(getattr(self, name).datapipe)
                              for name in ("validloader", "testloader")},
        })

    def load_checkpoint(self):
        path = Path(self.cfg.LOG_PATH)
        payload = torch.load(path / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
        if payload.get("schema") != 1 or payload.get("contract") != self._checkpoint_contract():
            raise ValueError("Resume contract differs from checkpoint")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.continuation:
            for group in self.optimizer.param_groups:
                group["smoother"] = self.model.smoother if group["role"] == "items" else None
        self.order_rng.set_state(payload["order_rng"])
        self.negative_sampler.load_state_dict(payload["negative_sampler"])
        for name, state in payload["eval_samplers"].items():
            restore_sampler(getattr(self, name).datapipe, state)
        for key, value in payload["selection"].items():
            setattr(self, key, value)
        atomic_save(path / self.cfg.BEST_FILENAME, payload["best_model"])
        self.steps = payload["steps"]
        restore_rng(payload["rng"])
        return int(payload["completed_epoch"])

    def fit(self):
        start = self.load_checkpoint() if self.cfg.resume else 0
        self.completed_epoch = start
        self._early_stop_patience = 2**31 - 1  # Explicit head stopping below.
        try:
            if not self.cfg.resume:
                self.valid(0)  # Disabled residual must participate in selection.
                self.save_checkpoint(0)
            for epoch in range(start + 1, self.cfg.head_epochs + 1):
                self.train(epoch)
                self.completed_epoch = epoch
                if epoch % self.cfg.eval_freq == 0 or epoch == self.cfg.head_epochs:
                    self.valid(epoch)
                self.save_checkpoint(epoch)
                if self._stopping_steps >= self.cfg.head_patience:
                    break
            self.save_last()
            self._stopping_steps = -1
            self.eval_at_best()
        finally:
            self.shutdown()


def run_stage_b(dataset, cfg, directory, teacher_directory, fingerprint, hashes):
    guard_run_directory(directory, cfg.resume)
    payload, teacher_manifest = read_teacher(
        teacher_directory, dataset=cfg.dataset, seed=cfg.seed,
        data_fingerprint=fingerprint, protocol=baseline_protocol(cfg))
    phase = phase_config(cfg, directory, stage_b=True)
    write_json(directory / "resolved_config.json", json.loads(json.dumps(dict(phase), default=str)))
    manifest = {k: teacher_manifest[k] for k in (
        "dataset", "seed", "data_fingerprint", "baseline_source_hash",
        "comparison_id", "teacher_hash", "baseline_protocol")}
    manifest.update(schema=1, status="preparing", stage="b", arm=cfg.arm,
                    source_hash=implementation_hash(), teacher_dir=str(Path(teacher_directory).resolve()),
                    protocol=teacher_manifest["protocol"], test_disclosed=not cfg.no_final_test)
    start = time.perf_counter()
    try:
        edges = dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index.cpu().numpy()
        hu, hi = payload["user_vectors"], payload["item_vectors"]
        csr = build_train_csr(edges, len(hu), len(hi))
        if cfg.arm.startswith("baseline"):
            model = STAIR4V5Continuation(dataset, phase)
            model.load_state_dict(payload["model_state"])
        else:
            metadata = {}
            if cfg.arm == "id-ss":
                features = id_only_features(hi, cfg.pca_dim, seed=cfg.seed + 3001)
                metadata = {"control": "fixed_id_lifts", "seed": cfg.seed + 3001}
            else:
                features = {}
                for number, (name, filename) in enumerate(zip(("text", "image"), cfg.mfiles.split(","))):
                    raw = freerec.utils.import_pickle(str(Path(dataset.path) / filename))
                    cache_key = identity_hash({"raw": hashes[filename], "pca": cfg.pca_dim,
                                              "algorithm": cfg.pca_algorithm, "seed": cfg.seed + number,
                                              "preprocessing_source": file_hash(Path(prepare_modality_features.__code__.co_filename))})
                    result = prepare_modality_features(raw, output_dim=cfg.pca_dim,
                        algorithm=cfg.pca_algorithm, seed=cfg.seed + number,
                        cache_path=Path(cfg.feature_cache_dir) / f"{cache_key}.pt")
                    features[name], metadata[name] = result.values, result.metadata
                    del raw, result
            history = TrainHistory(csr, features)
            sigma, calibration = calibrate_scale(hu, hi, csr, seed=cfg.seed + 6101, limit=cfg.calibration_samples)
            identity = {"teacher_hash": manifest["teacher_hash"], "data_fingerprint": fingerprint,
                        "features": {k: array_hash(v) for k, v in features.items()},
                        "history": history.fingerprint, "eta": cfg.eta_scale * sigma,
                        "dimension": cfg.residual_dim, "gate": cfg.gate_mode, "arm": cfg.arm}
            # Head randomness is separate from sampler/PCA/Stage A.
            with torch.random.fork_rng(devices=[]):
                # Seed only the CPU generator: torch.manual_seed also touches
                # CUDA generators, which are intentionally outside this fork.
                torch.random.default_generator.manual_seed(cfg.seed + 4001)
                model = STAIR4V5(dataset, hu, hi, history, eta=cfg.eta_scale * sigma,
                                residual_dim=cfg.residual_dim, gate_mode=cfg.gate_mode, identity=identity)
            manifest.update(calibration=calibration, feature_metadata=metadata, model_identity=identity)
        manifest["preparation_seconds"] = time.perf_counter() - start
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        write_json(directory / "run_manifest.json", manifest)
        coach = ResidualCoach(dataset=dataset, model=model, cfg=phase,
            trainpipe=model.sure_trainpipe(cfg.batch_size), validpipe=model.sure_validpipe(cfg.ranking),
            testpipe=model.sure_testpipe(cfg.ranking))
        coach.run_dir = directory
        manifest["head_protocol"] = coach.head_protocol
        if coach.continuation and not cfg.resume:
            coach.optimizer.load_state_dict(payload["training_state"]["optimizer"])
            for group in coach.optimizer.param_groups:
                group["smoother"] = model.smoother if group["role"] == "items" else None
            manifest["optimizer_initialization"] = "selected_stage_a_moments"
        coach.fit()
        manifest.update(status="completed", selected_epoch=int(coach._best_epoch),
                        selected_valid_ndcg20=float(coach._best), completed_epoch=coach.completed_epoch)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["total_seconds"] = time.perf_counter() - start
        write_json(directory / "run_manifest.json", manifest)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    cfg = make_parser()
    cfg.compile()
    validate_config(cfg)
    if cfg.resume and cfg.stage == "all":
        raise ValueError("Resume a specific stage using --stage a or --stage b")
    directory = Path(cfg.run_dir or f"logs/stair4v5_{cfg.dataset}_{cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}")
    dataset = build_dataset(cfg)
    fingerprint, hashes = dataset_fingerprint(dataset, cfg.mfiles)
    teacher_directory = Path(cfg.teacher_dir) if cfg.teacher_dir else directory / "stage_a"
    print(f"[STAIR-RAM] stage={cfg.stage}, arm={cfg.arm}, A epochs={cfg.epochs}, "
          f"B epochs={cfg.head_epochs}, test_disclosure={not cfg.no_final_test}")
    if cfg.stage in ("a", "all"):
        run_stage_a(dataset, cfg, teacher_directory, fingerprint, hashes)
    if cfg.stage in ("b", "all"):
        run_stage_b(dataset, cfg, directory / "stage_b", teacher_directory, fingerprint, hashes)
    print(f"[STAIR-RAM] Artifacts: {directory.resolve()}")


if __name__ == "__main__":
    main()
