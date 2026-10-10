"""Train the isolated STAIR5-v8 256D capacity experiment.

Example:
    python main_stair5_v8_256d.py --config configs/Amazon2014Sports_STAIR5_v8_256D.yaml

The original parser, optimizer coach, evaluator and checkpoint lifecycle are
reused without modifying or monkey-patching the original engine.
"""

import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from stair5_v8_256d_config import (
    EXPERIMENT_ID,
    tensor_payload_ledger,
    validate_capacity_config,
    with_default_config,
)


def build_config(arguments=None):
    """Compile the inherited CLI, then validate the capacity-only contract."""
    from main_stair5_v8 import build_config as build_reference_config

    original_argv = sys.argv
    supplied = original_argv[1:] if arguments is None else arguments
    effective = with_default_config(supplied)
    try:
        sys.argv = [original_argv[0], *effective]
        cfg = build_reference_config()
    finally:
        sys.argv = original_argv
    validate_capacity_config(cfg)
    cfg.capacity_experiment = EXPERIMENT_ID
    cfg.capacity_effective_argv = effective
    return cfg


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # Load the reference engine first: it installs the existing compatibility
    # shims before importing the runtime. Importing this launcher alone remains
    # dependency-free for CLI argument/contract checks.
    from main_stair5_v8 import CoachForSTAIR5_v8, load_dataset
    import freerec
    import torch
    from models.stair5_v8_256d import STAIR5_v8_256D_Model

    cfg = build_config()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact = Path(cfg.artifact_dir)
    resuming = bool(cfg.resume or cfg.resume_from)
    if not resuming and any(
        (artifact / name).exists()
        for name in ("manifest.json", "training_checkpoint.pt", "training_telemetry.jsonl")
    ):
        raise FileExistsError("Use a new 256D artifact directory, or explicitly resume its own checkpoint")

    started = time.perf_counter()
    dataset = load_dataset(cfg)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = STAIR5_v8_256D_Model(dataset, cfg)
    ledger = tensor_payload_ledger(model.User.count, model.Item.count)
    preprocessing_memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    } if device.type == "cuda" else {}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    config_names = (
        "embedding_dim", "num_layers", "gamma", "lr", "weight_decay",
        "batch_size", "epochs", "beta1", "beta2", "mfiles", "num_neighbors",
        "eval_freq", "lambda_nlgcl", "nlgcl_tau", "nlgcl_G", "nlgcl_alpha",
        "eta", "k_cf", "c_min", "t_shrinkage", "v8_arm", "semantic_mode",
        "edge_power", "edge_floor", "semantic_mix", "alignment", "anchor_seed",
        "anchor_per_stratum", "relation_gate", "relation_strength",
        "relation_threshold", "relation_temperature", "bsc_residual",
        "lambda_dirichlet", "cl_chunk_size", "knn_chunk_size", "knn_device",
        "cf_block_size", "cf_memory_budget_mib", "v4_support_files",
        "metadata_file", "placebo_seed", "seed", "num_workers", "CHECKPOINT_FREQ",
    )
    manifest = {
        "argv": list(sys.argv),
        "effective_argv": [sys.argv[0], *cfg.capacity_effective_argv],
        "commit": commit,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "freerec": freerec.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "experiment": EXPERIMENT_ID,
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
        "tensor_payload_ledger": ledger,
        "metadata_fingerprint": cfg.metadata_fingerprint,
        "ranking": cfg.ranking,
        "selection": "validation NDCG@20",
        "v8_config": {name: getattr(cfg, name) for name in config_names},
        "freerec_log_path": cfg.LOG_PATH,
        "test_records": "Native FreeRec final and selected; report selected only.",
        "memory_scope": "Inherited epoch training peaks; evaluation/process peaks require separate profiling.",
    }
    filename = "resume_manifest.json" if resuming else "manifest.json"
    (artifact / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[STAIR5-v8-256D payload] " + json.dumps(ledger), flush=True)
    print("[STAIR5-v8-256D graph] " + json.dumps(model.graph_metadata), flush=True)

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
