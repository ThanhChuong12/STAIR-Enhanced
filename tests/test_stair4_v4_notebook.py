"""Regression tests for the executable Kaggle notebook, without hidden kernel state."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook/P4/stair4_v4.ipynb"


@pytest.fixture
def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"]}


@pytest.fixture
def scope(cells, tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    namespace = dict(Path=Path, os=os, sys=sys, json=json, hashlib=hashlib, time=time,
                     shutil=shutil, subprocess=subprocess, REPO=ROOT, ARTIFACT_ROOT=tmp_path,
                     REPORT_DIR=tmp_path / "reports", RUN_MODE="full", RESUME_RUN=None,
                     DEVICE="cpu", EXPERIMENT_ID="test", CACHE_ROOT=tmp_path / "cache",
                     ENV=dict(os.environ, PYTHONUTF8="1"), SOURCE_HASHES={"test": "fixture"},
                     code_fingerprints=lambda: {"test": "fixture"},
                     DATASETS_TO_RUN=["baby"], SEEDS=[1], ARMS=["V4-B1", "V4-C"],
                     PILOT_EPOCHS=100, CONFIGS={"baby": {"epochs": 500}},
                     PREPARED={"baby": {"root": str(tmp_path), "fingerprint": "split-a"}},
                     PRESETS={"V4-B1": {"alpha": 0.0}, "V4-C": {"alpha": .25}})
    # In particular: do not inject numpy/np, pandas/pd or any plotting aliases.
    for name in ("v4-runner-functions", "train-helper", "v4-report-functions",
                 "v4-plot-functions", "v4-resume-functions"):
        exec(compile(cells[name], name, "exec"), namespace)
    return namespace


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_run(scope, arm="V4-C", stage="full", fingerprint="split-a", budget=500, status="completed"):
    directory = scope["ARTIFACT_ROOT"] / "baby" / stage / (arm + "_seed1")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": "baby", "arm": arm, "stage": stage, "seed": 1,
                "data_fingerprint": fingerprint}
    command = [sys.executable, "--device", "0", "--epochs", str(budget),
               "--run-dir", str(directory), "--root", str(scope["ARTIFACT_ROOT"])]
    (directory / "command.json").write_text(json.dumps({
        "argv": command, "metadata": metadata, "source_files": scope["SOURCE_HASHES"]}))
    (directory / "run_manifest.json").write_text(json.dumps({
        "status": status, "options": {"ablation_id": arm}, "seed": 1, "selected_epoch": 5,
        "protocol": {"epochs": budget, "ranking": "full", "selection": "NDCG@20"},
        "graph_identity": {"train_col": "train", "data": "features"},
        "source_hash": "source", "runtime": {"torch": "fixture"}, "preparation_seconds": .2,
    }))
    write_jsonl(directory / "epochs.jsonl", [
        {"epoch": 1, "bpr": .6, "train_seconds": 2., "peak_allocated_mb": 100.},
        {"epoch": 5, "bpr": .4, "train_seconds": 1., "peak_allocated_mb": 110.},
        {"epoch": 10, "bpr": .3, "train_seconds": 1.1, "peak_allocated_mb": 110.},
        {"epoch": 10, "bpr": .2, "train_seconds": 1.2, "peak_allocated_mb": 120.},
    ])
    write_jsonl(directory / "evaluation.jsonl", [
        {"epoch": 5, "mode": "valid", "selected_checkpoint": False,
         "metrics": {"NDCG@20": .05, "RECALL@20": .09}},
        {"epoch": 10, "mode": "valid", "selected_checkpoint": False,
         "metrics": {"NDCG@20": .04, "RECALL@20": .10}},
        {"epoch": 10, "mode": "test", "selected_checkpoint": False,
         "metrics": {"NDCG@20": .99, "RECALL@20": .99}},
        {"epoch": 5, "mode": "valid", "selected_checkpoint": True,
         "metrics": {"NDCG@20": .05, "RECALL@20": .09}},
        {"epoch": 5, "mode": "test", "selected_checkpoint": True,
         "metrics": {"NDCG@20": .06, "RECALL@20": .11}},
    ])
    return directory


def test_cell_syntax_and_explicit_full_default(cells):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(cells) == len(notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
            assert not cell["outputs"] and cell["execution_count"] is None
    constants = {}
    for node in ast.parse(cells["v4-config"]).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            constants[node.targets[0].id] = node.value.value
    assert constants["RUN_MODE"] == "full"
    assert constants["PILOT_EPOCHS"] == 100


@pytest.mark.parametrize("mode,budget", [("full", 500), ("pilot", 100)])
def test_training_budget_cli_and_no_plot_side_effect(scope, mode, budget):
    commands = []
    scope["RUN_MODE"] = mode
    scope["launch"] = lambda *args: commands.append(scope["make_command"](*args))
    scope["plot_vram_profile"] = lambda *args: pytest.fail("Training must not invoke plotting")
    scope["plot_single_dataset_learning_curves"] = scope["plot_vram_profile"]
    scope["train_dataset"]("baby")
    assert len(commands) == 2
    for (argv, directory), arm in zip(commands, scope["ARMS"]):
        assert argv[argv.index("--epochs") + 1] == str(budget)
        assert argv[argv.index("--ablation-id") + 1] == arm
        assert argv[argv.index("--num-workers") + 1] == "0"
        assert directory.parent.name == mode
    assert commands[0][0][-2:] == ["--alpha", "0.0"]
    assert commands[1][0][-2:] == ["--alpha", "0.25"]


def test_learning_curves_fresh_namespace_uppercase_and_selected_epoch(scope):
    make_run(scope)
    captured = []
    scope["save_figure"] = lambda fig, path: captured.append(fig)
    scope["plot_single_dataset_learning_curves"]("baby")
    fig = captured[0]
    assert "np" not in scope
    assert list(fig.axes[0].lines[0].get_ydata()) == [.6, .4, .2]
    assert list(fig.axes[2].lines[0].get_ydata()) == [.09, .10]
    # Recall's maximum occurs at epoch 10, but selection is by NDCG at epoch 5.
    assert fig.axes[2].collections[0].get_offsets().tolist() == [[5., .09]]
    assert len(fig.axes[1].lines) == 2  # Observed curve + selected epoch; no historical reference.
    plt.close(fig)
    assert scope["observed_metric"]([{"epoch": 1, "metrics": {}},
                                     {"epoch": 2, "metrics": {"Recall@20": .12}},
                                     {"epoch": 3, "metrics": {"RECALL@20": None}}], "RECALL@20") == [(2, .12)]


def test_plotting_renders_without_training_and_separates_stages(scope):
    make_run(scope, stage="pilot", budget=50)
    assert scope["dataset_plot_runs"]("baby") == []
    assert len(scope["dataset_plot_runs"]("baby", stage="pilot")) == 1
    for function in ("plot_vram_profile", "plot_single_dataset_learning_curves"):
        path = scope[function]("baby", stage="pilot")
        assert path.is_file() and path.stat().st_size > 1000
    assert not plt.get_fignums()


def test_reporting_selected_metrics_and_matched_heldout_data(scope):
    make_run(scope, "V4-B1")
    core = make_run(scope)
    frame = scope["collect_results"](scope["ARTIFACT_ROOT"])
    row = frame[frame.arm == "V4-C"].iloc[0]
    assert row["test_RECALL@20"] == .11 and row["test_NDCG@20"] == .06
    assert row["delta_test_RECALL@20_pct"] == 0
    assert math.isnan(row["test_RECALL@10"])  # Missing data is not zero.
    assert row.budget_epochs == 500
    saved = json.loads((core / "command.json").read_text())
    saved["metadata"]["data_fingerprint"] = "different-valid-test"
    (core / "command.json").write_text(json.dumps(saved))
    frame = scope["collect_results"](scope["ARTIFACT_ROOT"])
    assert math.isnan(frame[frame.arm == "V4-C"].iloc[0]["delta_test_RECALL@20_pct"])


def test_failed_or_different_budget_runs_are_not_comparable(scope):
    make_run(scope, "V4-B1", budget=50)
    make_run(scope, budget=500)
    frame = scope["collect_results"](scope["ARTIFACT_ROOT"])
    assert math.isnan(frame[frame.arm == "V4-C"].iloc[0]["delta_test_RECALL@20_pct"])
    make_run(scope, status="failed")
    frame = scope["collect_results"](scope["ARTIFACT_ROOT"])
    assert math.isnan(frame[frame.arm == "V4-C"].iloc[0]["test_RECALL@20"])


def test_resume_uses_latest_extended_budget(scope):
    directory = make_run(scope, stage="pilot", budget=50)
    (directory / "checkpoints").mkdir()
    (directory / "checkpoints" / "checkpoint.tar").touch()
    argv = json.loads((directory / "command.json").read_text())["argv"]
    argv[argv.index("--epochs") + 1] = "500"
    write_jsonl(directory / "attempts.jsonl", [{"argv": argv}])
    calls = []
    scope["run_command"] = lambda *args, **kwargs: calls.append((args, kwargs))
    scope["resume_training"](directory)
    assert calls[0][0][0][calls[0][0][0].index("--epochs") + 1] == "500"
    assert calls[0][0][0][-1] == "--resume"
    with pytest.raises(ValueError, match="smaller"):
        scope["resume_training"](directory, 100)


def test_subprocess_failure_is_preserved(scope):
    directory = scope["ARTIFACT_ROOT"] / "failed"
    with pytest.raises(subprocess.CalledProcessError):
        scope["run_command"]([sys.executable, "-c", "print('failure-probe'); raise SystemExit(3)"],
                             directory, {"probe": True})
    assert "failure-probe" in (directory / "console.log").read_text()
    assert scope["read_jsonl"](directory / "attempt_results.jsonl")[-1]["status"] == "failed"


def test_completed_run_is_reused_without_retraining(scope, monkeypatch):
    directory = make_run(scope)
    saved = json.loads((directory / "command.json").read_text())
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Must reuse completed run"))
    assert scope["run_command"](saved["argv"], directory, saved["metadata"]) == directory


def test_report_cells_export_real_jsonl_without_hidden_imports(scope, cells):
    make_run(scope, "V4-B1")
    make_run(scope)
    scope["REPORT_DIR"].mkdir()
    scope["display"] = lambda value: None
    exec(compile(cells["result-tables"], "result-tables", "exec"), scope)
    assert (scope["REPORT_DIR"] / "selected_checkpoint_metrics.csv").is_file()
    assert (scope["REPORT_DIR"] / "selected_metrics.tex").is_file()
    exec(compile(cells["render-plots"], "render-plots", "exec"), scope)
    assert len(scope["FIGURES"]) == 2
    assert all(path.stat().st_size > 1000 for path in scope["FIGURES"])
    assert not plt.get_fignums()


@pytest.mark.parametrize("key", ["sports", "electronics"])
def test_explicit_dataset_request_from_baby_kernel(scope, cells, tmp_path, key):
    """Stage real fixture files and construct both commands from a Baby-only kernel."""
    scope.update(WORK=tmp_path, INPUT_ROOT=tmp_path / "input", DATA_ROOT=tmp_path / "staged",
                 DATASET_SOURCES={}, DATASET_NAMES={
                     "sports": "Amazon2014Sports_550_MMRec",
                     "electronics": "Amazon2014Electronics_550_MMRec"},
                 REQUIRED_FILES=("train.txt", "valid.txt", "test.txt",
                                 "textual_modality.pkl", "visual_modality.pkl"))
    from main_stair4_v4 import load_config
    scope["CONFIGS"][key] = load_config(ROOT / f"configs/dataset_stair4_v4_{key}.yaml")
    for cell in ("v4-source-functions", "v4-data-functions"):
        exec(compile(cells[cell], cell, "exec"), scope)
    directory = scope["INPUT_ROOT"] / key
    directory.mkdir(parents=True)
    for name in scope["REQUIRED_FILES"]:
        (directory / name).write_text("fixture")
    calls = []
    scope["launch"] = lambda *args: calls.append(scope["make_command"](*args))
    scope["train_dataset"](key)
    assert scope["DATASETS_TO_RUN"] == ["baby"]  # No hidden selection mutation.
    assert key in scope["PREPARED"] and len(calls) == 2
    assert (Path(scope["PREPARED"][key]["processed"]) / "train.txt").read_text() == "fixture"
    for argv, directory in calls:
        assert argv[argv.index("--config") + 1].endswith(f"dataset_stair4_v4_{key}.yaml")
        assert argv[argv.index("--epochs") + 1] == "500"
        assert directory.parent.parent.name == key
    # Two possible attached splits must not be resolved by arbitrary path length.
    other = scope["INPUT_ROOT"] / (key + "_other")
    shutil.copytree(scope["INPUT_ROOT"] / key, other)
    with pytest.raises(RuntimeError, match="ambiguous"):
        scope["discover_dataset"](key, scope["INPUT_ROOT"])


def test_run_all_does_not_retrain_completed_baby(scope, cells):
    scope["DATASETS_TO_RUN"] = ["sports", "electronics"]
    scope["launch"] = lambda *a: pytest.fail("Baby must remain unselected")
    exec(cells["train-baby"], scope)
    with pytest.raises(ValueError, match="Unknown"):
        scope["train_dataset"]("typo")
