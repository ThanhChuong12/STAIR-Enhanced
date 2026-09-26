"""Audit selected-checkpoint v5 results and summarize genuinely paired experiments.

This module uses only the standard library. It never selects checkpoints or tunes
hyperparameters from test metrics. Missing/failed/unmatched runs remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


METRICS = ("RECALL@10", "RECALL@20", "NDCG@10", "NDCG@20")
DATASETS = ("baby", "sports", "electronics")
EVALUATION_KEYS = ("ranking", "retain_seen", "selection", "eval_freq", "evaluator_hash")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{number}") from exc
    return rows


def canonical_dataset(value: Any) -> str:
    text = str(value).lower()
    for key in DATASETS:
        if key in text or (key == "sports" and "sport" in text):
            return key
    raise ValueError(f"Unknown dataset {value!r}")


def metric_values(record: dict[str, Any]) -> dict[str, float]:
    raw = {str(k).upper(): v for k, v in record.get("metrics", {}).items()}
    values = {}
    for name in METRICS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing/non-numeric selected metric: {name}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid selected metric {name}={value}")
        values[name] = float(value)
    return values


def read_run(manifest_path: Path) -> dict[str, Any]:
    """Require one final test at the validation-selected checkpoint, or flag it."""
    result: dict[str, Any] = {"directory": str(manifest_path.parent), "error": None}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result.update(manifest=manifest, dataset=canonical_dataset(manifest["dataset"]),
                      seed=int(manifest["seed"]), arm=manifest.get("arm", "B0" if
                      manifest_path.parent.name == "stage_a" else None))
        if not result["arm"]:
            raise ValueError("Manifest does not identify the experiment arm")
        if manifest.get("status") != "completed":
            raise ValueError(f"Run status is {manifest.get('status')!r}, not completed")
        tests = [row for row in read_jsonl(manifest_path.parent / "evaluation.jsonl")
                 if str(row.get("mode", "")).lower() == "test"]
        if not tests:
            raise ValueError("No final test: validation-only pilot or incomplete evaluation")
        if len(tests) != 1:
            raise ValueError("Expected exactly one final test; repeated test disclosure detected")
        selected = tests[0]
        if selected.get("selected_checkpoint") is not True:
            raise ValueError("Final test is not marked selected_checkpoint=True")
        if selected.get("epoch") != manifest.get("selected_epoch"):
            raise ValueError("Final test epoch differs from the manifest selected_epoch")
        result["metrics"] = metric_values(selected)
        result["selected_epoch"] = selected["epoch"]
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result["error"] = str(exc)
    return result


def collect_runs(root: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(root).rglob("run_manifest.json"))
    return [read_run(path) for path in paths if path.parent.name in {"stage_a", "stage_b"}]


def pairing_error(control: dict[str, Any], treatment: dict[str, Any]) -> str | None:
    """Fail closed on absent provenance instead of accepting two unknown identities."""
    for run in (control, treatment):
        if run.get("error"):
            return run["error"]
    a, b = control["manifest"], treatment["manifest"]
    fields = ("data_fingerprint", "baseline_source_hash", "comparison_id", "teacher_hash")
    for key in fields:
        if not a.get(key) or not b.get(key):
            return f"Missing provenance: {key}"
        if a[key] != b[key]:
            return f"Unmatched provenance: {key}"
    for key in EVALUATION_KEYS:
        x, y = a.get("protocol", {}), b.get("protocol", {})
        if key not in x or key not in y or x[key] is None or y[key] is None:
            return f"Missing evaluation protocol: {key}"
        if x[key] != y[key]:
            return f"Unmatched evaluation protocol: {key}"
    if control["arm"] != "B0":
        if not a.get("source_hash") or a.get("source_hash") != b.get("source_hash"):
            return "Unmatched Stage B source_hash"
        x, y = dict(a.get("head_protocol", {})), dict(b.get("head_protocol", {}))
        # BPR-vs-SS is a declared objective/negative-count ablation, not a
        # supposedly equal-compute comparison. Keep all other budgets matched.
        if control["arm"].endswith("bpr") != treatment["arm"].endswith("bpr"):
            bpr = x if control["arm"].endswith("bpr") else y
            if bpr.get("num_negatives") != 1 or bpr.get("temperature") != 1.:
                return "BPR arm does not use the registered K1/T1 objective"
            x.pop("num_negatives", None)
            y.pop("num_negatives", None)
        if not x or x != y:
            return "Unmatched Stage B head_protocol"
    return None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lo, hi = math.floor(coordinate), math.ceil(coordinate)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (coordinate - lo)


def summarize_metric(baseline: list[float], treatment: list[float], *,
                     bootstrap_samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("Nonempty equally sized paired observations are required")
    mean_b, mean_t = statistics.mean(baseline), statistics.mean(treatment)
    if mean_b <= 0:
        raise ValueError("Relative growth requires a positive baseline mean")
    differences = [v - b for b, v in zip(baseline, treatment)]
    gain = 100 * (mean_t / mean_b - 1)
    # Compare the unrounded means: reporting precision must not promote 6% to >6%.
    exceeds = mean_t > 1.06 * mean_b
    result = {"n_pairs": len(baseline), "baseline_mean": mean_b,
              "baseline_sd": statistics.stdev(baseline) if len(baseline) > 1 else None,
              "treatment_mean": mean_t,
              "treatment_sd": statistics.stdev(treatment) if len(treatment) > 1 else None,
              "paired_difference_mean": statistics.mean(differences),
              "paired_difference_sd": statistics.stdev(differences) if len(differences) > 1 else None,
              "paired_differences": differences, "gain_percent": gain,
              "above_six_percent": exceeds, "descriptive_gain_ci95": None,
              "descriptive_difference_ci95": None}
    if len(baseline) > 1 and bootstrap_samples > 0:
        rng, gains, diffs = random.Random(seed), [], []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(len(baseline)) for _ in baseline]
            b = statistics.mean(baseline[i] for i in indices)
            v = statistics.mean(treatment[i] for i in indices)
            if b > 0:
                gains.append(100 * (v / b - 1))
            diffs.append(v - b)
        result["descriptive_gain_ci95"] = [_quantile(gains, p) for p in (.025, .975)] if gains else None
        result["descriptive_difference_ci95"] = [_quantile(diffs, p) for p in (.025, .975)]
    return result


def analyze_runs(runs: list[dict[str, Any]], *, treatment: str = "mm-ss", control: str = "B0",
                 datasets: Iterable[str] = DATASETS, seeds: Iterable[int] | None = None,
                 bootstrap_samples: int = 2000) -> dict[str, Any]:
    """Report ratio of paired means, never mean of seed-wise percentage gains."""
    if treatment == control:
        raise ValueError("Treatment and control must differ")
    datasets = tuple(datasets)
    expected = sorted(set(seeds)) if seeds is not None else sorted({
        row["seed"] for row in runs if row.get("arm") in {control, treatment} and "seed" in row})
    report: dict[str, Any] = {"treatment": treatment, "control": control, "expected_seeds": expected,
        "datasets": {}, "missing_or_invalid_pairs": [], "point_estimate_target_met": False,
        "confirmation_coverage": False, "statistical_confirmation": False,
        "notes": ["The target is >6% in at least three of four metrics separately on every dataset.",
                  "Intervals are descriptive paired-seed percentile bootstrap intervals; no p-values are claimed.",
                  "With five pairs, an exact two-sided sign-flip test cannot attain p<0.05.",
                  "A point estimate above 6% is not statistical confirmation of a gain above 6%."]}
    for dataset in datasets:
        pairs = []
        for seed in expected:
            by_arm = {arm: [r for r in runs if r.get("dataset") == dataset and
                           r.get("seed") == seed and r.get("arm") == arm]
                      for arm in (control, treatment)}
            reason = next((f"Expected one {arm} run, found {len(rows)}"
                           for arm, rows in by_arm.items() if len(rows) != 1), None)
            if reason is None:
                reason = pairing_error(by_arm[control][0], by_arm[treatment][0])
            if reason:
                report["missing_or_invalid_pairs"].append({"dataset": dataset, "seed": seed,
                                                           "reason": reason})
            else:
                pairs.append((seed, by_arm[control][0], by_arm[treatment][0]))
        metrics = {name: summarize_metric([a["metrics"][name] for _, a, _ in pairs],
                   [b["metrics"][name] for _, _, b in pairs], bootstrap_samples=bootstrap_samples)
                   for name in METRICS} if pairs else {}
        count = sum(m["above_six_percent"] for m in metrics.values())
        nonregression = bool(metrics) and all(metrics[m]["treatment_mean"] >= metrics[m]["baseline_mean"]
                                              for m in ("RECALL@20", "NDCG@20"))
        complete = bool(expected) and len(pairs) == len(expected)
        report["datasets"][dataset] = {"paired_seeds": [s for s, _, _ in pairs], "metrics": metrics,
            "metrics_above_six_percent": count, "r20_n20_nonregression": nonregression,
            "complete": complete, "target_met_on_available_pairs": count >= 3 and nonregression,
            "target_met": complete and count >= 3 and nonregression,
            "pair_directories": [{"seed": s, "control": a["directory"], "treatment": b["directory"]}
                                 for s, a, b in pairs]}
    all_three = set(datasets) == set(DATASETS)
    report["point_estimate_target_met"] = all_three and all(r["target_met"] for r in report["datasets"].values())
    report["confirmation_coverage"] = all_three and len(expected) >= 5 and all(
        r["complete"] for r in report["datasets"].values())
    return report


def write_reports(report: dict[str, Any], runs: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "run_audit.json").write_text(json.dumps(runs, indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for dataset, entry in report["datasets"].items():
        for metric, values in entry["metrics"].items():
            row = {"dataset": dataset, "metric": metric, "complete": entry["complete"], **values}
            for key in ("paired_differences", "descriptive_gain_ci95", "descriptive_difference_ci95"):
                row[key] = json.dumps(row[key])
            rows.append(row)
    with (output / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["dataset", "metric", "complete"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "missing_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "seed", "reason"])
        writer.writeheader()
        writer.writerows(report["missing_or_invalid_pairs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--treatment", default="mm-ss")
    parser.add_argument("--control", default="B0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    runs = collect_runs(args.root)
    report = analyze_runs(runs, treatment=args.treatment, control=args.control, datasets=args.datasets,
                          seeds=args.seeds, bootstrap_samples=args.bootstrap_samples)
    write_reports(report, runs, args.output)
    print(json.dumps({"runs": len(runs), "output": str(args.output),
                      "point_estimate_target_met": report["point_estimate_target_met"],
                      "missing_or_invalid_pairs": len(report["missing_or_invalid_pairs"])}, indent=2))


if __name__ == "__main__":
    main()
