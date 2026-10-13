"""Generate/check latent groups from native feature rows; no ASIN recovery.

This preflight measures clustering only, not final graph/BSC intervention or
ranking. Feature PKLs must be trusted artifacts, as with the native V8 loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.stair6_v1_latent_categories import build_latent_category_data


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mfiles", default="textual_modality.pkl,visual_modality.pkl")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/stair6_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--memory-budget-mib", type=float, default=128.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    files = [args.dataset_dir / name for name in args.mfiles.split(",")]
    splits = sorted(p for p in args.dataset_dir.iterdir() if p.suffix in (".txt", ".csv") and p.is_file())
    if len(files) != 2 or not all(p.is_file() for p in files) or not splits:
        parser.error("Require two native modality PKLs and baseline split files")
    if args.output.resolve() in {p.resolve() for p in [*files, *splits]}:
        parser.error("Output cannot overwrite any baseline input")
    features = []
    for path in files:
        with path.open("rb") as stream:
            features.append(torch.as_tensor(pickle.load(stream), dtype=torch.float32))
    state = build_latent_category_data(features, dataset=args.dataset or args.dataset_dir.name,
        feature_manifest=[{"file": p.name, "sha256": digest(p)} for p in files],
        split_manifest={p.name: digest(p) for p in splits}, clusters=args.clusters,
        projection_dim=args.projection_dim, iterations=args.iterations, chunk_size=args.chunk_size,
        seed=args.seed, memory_budget_mib=args.memory_budget_mib, cache_dir=args.cache_dir)
    record = {"status": "READY_FOR_GRAPH_DIAGNOSTICS", "scope": "latent clustering only",
              "not_measured": ["normalized graph dose", "BSC dose", "ranking gain", "GPU memory"],
              "audit": state.data.audit, "provenance": state.data.provenance,
              "metadata": state.metadata, "fingerprint": state.data.metadata_fingerprint}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"status": record["status"], "output": str(args.output), **state.metadata}), flush=True)


if __name__ == "__main__":
    main()
