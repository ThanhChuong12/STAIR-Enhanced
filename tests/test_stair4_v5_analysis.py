"""Selected-only reporting and notebook orchestration regression contracts."""
import ast
import copy
import json
from pathlib import Path

from scripts.analyze_stair4_v5 import analyze_runs, pairing_error, read_run, summarize_metric, METRICS

ROOT = Path(__file__).resolve().parents[1]


def run(dataset, seed, arm, metric):
    manifest = {"status": "completed", "dataset": dataset, "seed": seed, "arm": arm,
                "data_fingerprint": dataset, "baseline_source_hash": "base", "source_hash": "head",
                "comparison_id": f"{dataset}-{seed}", "teacher_hash": f"teacher-{dataset}-{seed}",
                "protocol": {"ranking": "full", "retain_seen": False, "selection": "NDCG@20",
                             "eval_freq": 5, "evaluator_hash": "evaluator"},
                "head_protocol": {"head_epochs": 100, "num_negatives": 32, "temperature": 1.}}
    return {"directory": f"{dataset}/{seed}/{arm}", "manifest": manifest, "error": None,
            "dataset": dataset, "seed": seed, "arm": arm, "metrics": dict.fromkeys(METRICS, metric)}


def test_target_requires_each_dataset_and_ratio_of_means():
    rows = [run(d, s, a, .1 if a == "B0" else (.108 if d != "baby" else .102))
            for d in ("baby", "sports", "electronics") for s in (1, 2)
            for a in ("B0", "mm-ss")]
    report = analyze_runs(rows, seeds=[1, 2], bootstrap_samples=0)
    assert not report["point_estimate_target_met"]
    assert report["datasets"]["sports"]["target_met"]
    report = analyze_runs([r for r in rows if not (r["dataset"] == "sports" and r["seed"] == 2)],
                          seeds=[1, 2], bootstrap_samples=0)
    assert not report["datasets"]["sports"]["complete"]
    assert report["missing_or_invalid_pairs"]
    result = summarize_metric([.1, .3], [.2, .3], bootstrap_samples=0)
    assert abs(result["gain_percent"] - 25.) < 1e-12
    assert not summarize_metric([.1], [.106], bootstrap_samples=0)["above_six_percent"]


def test_pairing_fails_closed_and_allows_declared_bpr_ablation():
    base, treatment = run("baby", 1, "B0", .1), run("baby", 1, "mm-ss", .11)
    assert pairing_error(base, treatment) is None
    broken = copy.deepcopy(treatment)
    del broken["manifest"]["teacher_hash"]
    assert "Missing provenance" in pairing_error(base, broken)
    bpr = run("baby", 1, "mm-bpr", .1)
    bpr["manifest"]["head_protocol"]["num_negatives"] = 1
    assert pairing_error(bpr, treatment) is None
    bpr["manifest"]["head_protocol"]["head_epochs"] = 200
    assert "head_protocol" in pairing_error(bpr, treatment)


def test_repeated_or_wrong_epoch_test_is_not_accepted(tmp_path):
    manifest = run("baby", 1, "mm-ss", .1)["manifest"]
    manifest["selected_epoch"] = 5
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    record = {"mode": "test", "selected_checkpoint": True, "epoch": 5, "metrics": dict.fromkeys(METRICS, .1)}
    log = tmp_path / "evaluation.jsonl"
    log.write_text(json.dumps(record) + "\n")
    assert read_run(path)["error"] is None
    log.write_text((json.dumps(record) + "\n") * 2)
    assert "exactly one" in read_run(path)["error"]
    record["epoch"] = 10
    log.write_text(json.dumps(record) + "\n")
    assert "epoch differs" in read_run(path)["error"]


def test_notebook_compiles_and_manual_other_dataset_is_not_skipped():
    notebook = json.loads((ROOT / "notebook/P4/stair4_v5.ipynb").read_text(encoding="utf-8"))
    sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    for identifier, source in sources.items():
        ast.parse(source, filename=identifier)
    called = []
    scope = {"DATASETS_TO_RUN": ["baby"], "prepare_dataset": lambda k: called.append(k),
             "SEEDS": [], "plot_dataset": lambda k: None}
    exec(sources["training-functions"], scope)
    scope["train_dataset"]("sports")
    scope["train_dataset"]("electronics")
    assert called == ["sports", "electronics"]
    assert "'--epochs', '500'" in sources["training-functions"]
    assert "command.append('--no-final-test')" in sources["training-functions"]
    assert "if return_code:" in sources["training-functions"]
