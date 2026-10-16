"""Train STAIR-REEC (versionICAART) with exact V8 coach/evaluator/checkpoint lifecycle.

Example CLI usage:
    python main_stair_icaart.py --config configs/Amazon2014Baby_STAIR_ICAART.yaml
    python main_stair_icaart.py --config configs/Amazon2014Baby_STAIR_ICAART.yaml --evidence-delta 0.0
    python main_stair_icaart.py --config configs/Amazon2014Baby_STAIR_ICAART.yaml --preprocess-only
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

from main_stair5_v8 import CoachForSTAIR5_v8, load_dataset
from models.stair5_v8_graph import ARMS_V8, resolve_v8_config
from models.stair_icaart import STAIR_ICAART_Model
from stair_icaart_config import ICAART_OPTIONS, ICAART_NAMES, validate_icaart_config


def _compile_config():
    cfg = freerec.parser.Parser()
    cfg.add_argument("--preprocess-only", action="store_true", default=False)
    for flag, kind, default in ICAART_OPTIONS:
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
        description="STAIR-REEC-versionICAART",
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
    validate_icaart_config(cfg)
    cfg = resolve_v8_config(cfg)

    if cfg.optimizer.lower() != "adamwsevo":
        raise ValueError("STAIR-REEC requires AdamWSEvo optimizer with BSC smoother.")
    if not cfg.eval_valid or cfg.eval_test or cfg.which4best.upper() != "NDCG@20":
        raise ValueError("Must select by validation NDCG@20; disable test evaluation during training.")
    if cfg.ranking != "full":
        raise ValueError("Reference experiments require full ranking.")
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

    # Backbone contract validations
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

    cfg.mfiles = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else list(cfg.mfiles)
    cfg.num_neighbors = (
        list(map(int, cfg.num_neighbors.split("-")))
        if isinstance(cfg.num_neighbors, str)
        else cfg.num_neighbors
    )
    if not cfg.mfiles or len(cfg.mfiles) != len(cfg.num_neighbors) or any(k <= 0 for k in cfg.num_neighbors):
        raise ValueError("Each modality requires one positive neighbor count.")
    if len(cfg.mfiles) != 2 or list(cfg.num_neighbors) != [5, 1]:
        raise ValueError("Canonical V8 requires text/visual modalities with support [5, 1].")

    cfg.beta3 = (
        0.1 + 0.9 * (torch.arange(cfg.embedding_dim) / cfg.embedding_dim).pow(cfg.gamma)
    ).to(cfg.device)

    cfg.artifact_dir = cfg.artifact_dir or str(Path(cfg.LOG_PATH) / "icaart_artifacts")
    Path(cfg.artifact_dir).mkdir(parents=True, exist_ok=True)
    cfg.graph_cache_dir = cfg.graph_cache_dir or str(Path(cfg.root) / ".stair_icaart_graph_cache")
    Path(cfg.graph_cache_dir).mkdir(parents=True, exist_ok=True)
    for name, value in validate_icaart_config(cfg).items():
        setattr(cfg, name, value)
    return cfg


def build_config(arguments=None):
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

    model = STAIR_ICAART_Model(dataset, cfg)

    preprocessing_memory = (
        {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg.device),
        }
        if torch.device(cfg.device).type == "cuda"
        else {}
    )

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    manifest = {
        "architecture": "STAIR-REEC-versionICAART",
        "icaart_manifest": model.icaart_manifest,
        "icaart_config": {name: getattr(model.cfg, name) for name in ICAART_NAMES},
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
        "checkpoint_contract": model.get_extra_state(),
        "precision": {"tf32": False, "amp": False},
        "preprocessing_seconds": time.perf_counter() - started,
        "preprocessing_gpu_memory": preprocessing_memory,
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
                "cl_chunk_size",
                "knn_chunk_size",
                "knn_device",
                "cf_block_size",
                "cf_memory_budget_mib",
            )
        },
        "freerec_log_path": cfg.LOG_PATH,
    }

    manifest_name = "resume_manifest.json" if cfg.resume or cfg.resume_from else "manifest.json"
    Path(cfg.artifact_dir, manifest_name).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Mechanism diagnostics report printout
    print("\n" + "=" * 80, flush=True)
    print("STAIR-REEC (versionICAART) — MECHANISM DIAGNOSTICS REPORT", flush=True)
    print("=" * 80, flush=True)
    icaart_meta = model.icaart_metadata
    print(f"Evidence Mode:              {cfg.evidence_mode}")
    print(f"Delta (strength):           {cfg.evidence_delta}")
    print(f"Tau (count regularizer):    {cfg.evidence_tau}")
    print(f"Exact V8 Bypass:            {icaart_meta.get('exact_v8_bypass')} (reason: {icaart_meta.get('inactive_reason')})")
    print(f"Candidate Edges E:          {icaart_meta.get('num_candidate_edges')}")
    print(f"Edges with score > 0:       {icaart_meta.get('num_edges_with_replicated_excess')} ({icaart_meta.get('frac_edges_with_replicated_excess', 0.0):.4%})")
    print(f"Weighted Coverage:          {icaart_meta.get('weighted_replicated_coverage', 0.0):.4%}")
    print(f"Single-Half Disagreement:   {icaart_meta.get('frac_single_half_disagreement', 0.0):.4%}")
    print(f"Endpoint Contrast (Eq 10):  mean={icaart_meta.get('endpoint_contrast_mean', 0.0):.6f}, std={icaart_meta.get('endpoint_contrast_std', 0.0):.6f}")
    print(f"Operator Relative Dose:     {icaart_meta.get('relative_operator_dose', 0.0):.6f}")
    print(f"Polynomial Dose (Eq 11):    {icaart_meta.get('relative_polynomial_dose', 0.0):.6f}")
    print(f"S_ICA Nonzeros (nnz):       {icaart_meta.get('s_ica_nnz')}")
    print(f"S_ICA Fingerprint:          {icaart_meta.get('s_ica_fingerprint')}")
    print("=" * 80 + "\n", flush=True)

    if cfg.preprocess_only:
        print("[STAIR-REEC] Preprocessing completed; no training/evaluation executed.", flush=True)
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
