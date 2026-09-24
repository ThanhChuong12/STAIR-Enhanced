"""Summarize only metrics evaluated at the validation-selected checkpoint.

Use v4 run directories, not FreeRec's per-metric maxima or final-epoch tables.
Missing selected-checkpoint records remain missing rather than being inferred.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    result = {"run": str(directory), "dataset": manifest["dataset"], "seed": manifest["seed"],
              "ablation": manifest["options"]["ablation_id"], "status": manifest["status"],
              "epoch": manifest.get("selected_epoch"), "seconds": manifest.get("total_seconds")}
    path = directory / "evaluation.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["selected_checkpoint"]:
                for key, value in record["metrics"].items():
                    result[f'{record["mode"]}_{key}'] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
