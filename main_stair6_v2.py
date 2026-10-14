"""Train STAIR6-v2 with the exact V8 coach/evaluator/checkpoint lifecycle.

Example:
    python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml
    python main_stair6_v2.py --config configs/Amazon2014Baby_STAIR6_v2.yaml --can-mode off
"""

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import models.freerec_compat
import freerec
import torch

from main_stair5_v8 import CoachForSTAIR5_v8
from main_stair6_v1 import load_dataset
from models.stair5_v8_graph import ARMS_V8, resolve_v8_config
from models.stair6_v2 import STAIR6_v2_Model
from stair6_v2_config import CAN_OPTIONS, CAN_NAMES, validate_can_config


def _compile_config():
    cfg = freerec.parser.Parser()
    cfg.add_argument("--preprocess-only", action="store_true", default=False)
    for flag, kind, default in CAN_OPTIONS:
        cfg.add_argument("--" + flag, type=kind, default=default)
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
        # V8 specific parameters
        ("semantic-mode", str, "weighted"),
        ("edge-power", float, 2.0),
        ("edge-floor", float, 0.05),
        ("semantic-mix", float, 1.0),
        ("alignment", str, "off"),
        ("anchor-seed", int, 2718),
        ("anchor-per-stratum", int, 1024),
        ("relation-gate", str, "off"),
        ("relation-strength", float, 0.0),
        ("relation-threshold", float, 0.7),
        ("relation-temperature", float, 0.1),
        ("bsc-residual", float, 0.0),
        ("lambda-dirichlet", float, 0.0),
        ("v4-support-files", str, ""),
        ("metadata-file", str, ""),
        ("placebo-seed", int, 1),
    ):
        cfg.add_argument("--" + flag, type=kind, default=default)

    cfg.add_argument("--v8-arm", choices=ARMS_V8, default="WMSG-core")
    cfg.add_argument("--knn-device", choices=("cpu",), default="cpu")

    cfg.set_defaults(
        description="STAIR6-v2-AKP-WMSG-CSE",
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
    validate_can_config(cfg)  # Reject incompatible raw arms before parent resolution.
    cfg = resolve_v8_config(cfg)

    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("STAIR5-v8 requires AdamWSEvo optimizer with BSC smoother.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Must select by validation NDCG@20; disable test evaluation during training.")
    if cfg.ranking != "full":
        raise ValueError("Reference experiments require full ranking; pool scorer remains available for testing.")
    if not math.isfinite(cfg.eta) or not 0 <= cfg.eta <= 1 or cfg.k_cf < 0 or cfg.c_min < 1:
        raise ValueError("Require eta in [0, 1], k_cf >= 0 and c_min >= 1.")
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

    # V8 specific validations
    if not (0.0 <= cfg.semantic_mix <= 1.0):
        raise ValueError("semantic_mix (epsilon) must be in [0.0, 1.0].")
    if not (0.0 <= cfg.bsc_residual <= 0.1):
        raise ValueError("bsc_residual (omega) must be in [0.0, 0.1].")
    if not math.isfinite(cfg.edge_power) or cfg.edge_power not in (1.0, 1.5, 2.0, 3.0):
        raise ValueError("Canonical edge_power grid is {1, 1.5, 2, 3}.")
    if not math.isfinite(cfg.edge_floor) or not 0 < cfg.edge_floor < 1:
        raise ValueError("edge_floor must be finite and in (0, 1).")
    if cfg.semantic_mode not in ("weighted", "reference"):
        raise ValueError("semantic_mode must be weighted or reference.")
    if cfg.lambda_dirichlet != 0:
        raise ValueError("The V8 specification excludes a Dirichlet loss.")
    if cfg.knn_device != "cpu":
        raise ValueError("V8 preprocessing must run on CPU.")
    if cfg.anchor_seed < 0 or cfg.placebo_seed < 0 or cfg.anchor_per_stratum < 2:
        raise ValueError("Anchor/placebo seeds must be nonnegative; each stratum needs at least two anchors.")
    if not math.isfinite(cfg.relation_temperature) or cfg.relation_temperature <= 0:
        raise ValueError("relation_temperature must be finite and positive.")
    if not math.isfinite(cfg.relation_threshold) or not 0 <= cfg.relation_threshold <= 1:
        raise ValueError("relation_threshold must be in [0, 1].")
    if cfg.alignment not in ("off", "stratified_procrustes"):
        raise ValueError(f"Unknown alignment mode: {cfg.alignment}")
    if cfg.relation_gate not in ("off", "metadata_proxy"):
        raise ValueError(f"Unknown relation_gate mode: {cfg.relation_gate}")
    if not (0.0 <= cfg.relation_strength <= 0.25):
        raise ValueError("relation_strength (kappa) must be in [0.0, 0.25].")

    cfg.mfiles = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else list(cfg.mfiles)
    cfg.num_neighbors = (
        list(map(int, cfg.num_neighbors.split("-")))
        if isinstance(cfg.num_neighbors, str)
        else cfg.num_neighbors
    )
    if not cfg.mfiles or len(cfg.mfiles) != len(cfg.num_neighbors) or any(k <= 0 for k in cfg.num_neighbors):
        raise ValueError("Each modality requires one positive neighbor count.")
    if len(cfg.mfiles) != 2 or list(cfg.num_neighbors) != [5, 1]:
        raise ValueError("Canonical V8 requires text/visual modalities in that order with support [5, 1].")
    cfg.v4_support_files = (
        [p.strip() for p in cfg.v4_support_files.split(",") if p.strip()]
        if isinstance(cfg.v4_support_files, str) else list(cfg.v4_support_files)
    )
    if cfg.v4_support_files and (len(cfg.v4_support_files) != 2 or not all(Path(p).is_file() for p in cfg.v4_support_files)):
        raise ValueError("Provide two existing frozen V4 support files, in text/visual order.")
    cfg.metadata_dict = None
    cfg.metadata_fingerprint = None
    if cfg.metadata_file:
        metadata_path = Path(cfg.metadata_file)
        raw_metadata = metadata_path.read_bytes()
        cfg.metadata_dict = json.loads(raw_metadata)
        if not isinstance(cfg.metadata_dict, dict):
            raise ValueError("Metadata must be an item-aligned JSON object.")
        if cfg.relation_gate != "off" and not (
            cfg.metadata_dict.get("mapping_verified") is True
            and cfg.metadata_dict.get("comparable_snapshot") is True
        ):
            raise ValueError("RPG requires verified item mapping and a comparable price snapshot.")
        cfg.metadata_fingerprint = hashlib.sha256(raw_metadata).hexdigest()
    cfg.beta3 = (
        0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    ).to(cfg.device)

    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH) / "akp_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    for name, value in validate_can_config(cfg).items():
        setattr(cfg, name, value)
    return cfg


def build_config(arguments=None):
    """Compile/validate the native CLI plus isolated CAN arguments."""
    original = sys.argv
    try:
        if arguments is not None:
            sys.argv = [original[0], *arguments]
        return _compile_config()
    finally:
        sys.argv = original


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
    if torch.device(cfg.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(cfg.device)
    model = STAIR6_v2_Model(dataset, cfg)
    preprocessing_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg.device),
    } if torch.device(cfg.device).type == "cuda" else {}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    manifest = {
        "architecture": "STAIR6-v2-AKP-WMSG-CSE",
        "can": model.can_metadata,
        "can_config": {name: getattr(model.cfg, name) for name in CAN_NAMES},
        "effective_graph_cache_dir": model.cfg.graph_cache_dir,
        "argv": sys.argv,
        "commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "freerec": freerec.__version__,
        "device": str(cfg.device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "arm": cfg.v8_arm,
        "seed": cfg.seed,
        "graph": model.graph_metadata,
        "features": model.feature_manifest,
        "knn": model.knn_metadata,
        "sap_audit": model.sap_audit,
        "checkpoint_contract": model.get_extra_state(),
        "effective_lambda_nlgcl": model.lambda_nlgcl,
        "precision": {"tf32": False, "amp": False},
        "preprocessing_seconds": time.perf_counter() - started,
        "preprocessing_gpu_memory": preprocessing_memory,
        "metadata_fingerprint": cfg.metadata_fingerprint,
        "ranking": cfg.ranking,
        "selection": "validation NDCG@20",
        "v8_config": {
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
                "v8_arm",
                "semantic_mode",
                "edge_power",
                "edge_floor",
                "semantic_mix",
                "alignment",
                "anchor_seed",
                "anchor_per_stratum",
                "relation_gate",
                "relation_strength",
                "relation_threshold",
                "relation_temperature",
                "bsc_residual",
                "lambda_dirichlet",
                "cl_chunk_size",
                "knn_chunk_size",
                "knn_device",
                "cf_block_size",
                "cf_memory_budget_mib",
                "v4_support_files",
                "metadata_file",
                "placebo_seed",
            )
        },
        "freerec_log_path": cfg.LOG_PATH,
        "test_records": "Native FreeRec final and selected; report selected only.",
    }
    manifest_name = "resume_manifest.json" if cfg.resume or cfg.resume_from else "manifest.json"
    Path(cfg.artifact_dir, manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[STAIR6-v2 graph] " + json.dumps(model.graph_metadata), flush=True)
    print("[STAIR6-v2 SAP] " + json.dumps(model.sap_audit), flush=True)

    if cfg.preprocess_only:
        print("[STAIR6-v2] Preprocessing completed; no training/evaluation executed.", flush=True)
        return

    coach = CoachForSTAIR5_v8(
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
