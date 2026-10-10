"""Static notebook and artifact-reader contracts without the ML runtime."""

import ast
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from stair5_v8_256d_notebook_utils import (
    CONFIGS, DATASETS, REQUIRED_DATA, link_processed_dataset,
    read_evaluation, read_telemetry, training_command,
)

ROOT = Path(__file__).resolve().parents[1]


class TestNotebookContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def test_notebook_cells_parse_and_have_no_saved_execution(self):
        notebook = json.loads((ROOT / "notebook/P5/stair5_v8_256D.ipynb").read_text(encoding="utf-8"))
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                ast.parse(source, filename=f"cell_{index}")
                code.append(source)
        combined = "\n".join(code)
        self.assertIn('MODE = "full"', combined)
        self.assertIn('(500, 5)', combined)
        for key in DATASETS:
            self.assertIn(f'train_dataset("{key}", seed)', combined)
        self.assertNotIn("KGAT_", combined)
        self.assertNotIn("git reset", combined)
        self.assertNotIn("max_memory_allocated_bytes", combined)
        self.assertIn('1024 * r[field]', combined)
        self.assertIn('result.skipped', combined)

    def test_commands_select_256d_for_all_datasets(self):
        (self.path / "main_stair5_v8_256d.py").touch()
        for key in DATASETS:
            config = self.path / CONFIGS[key]
            config.touch()
            (self.path / "Processed" / DATASETS[key]).mkdir(parents=True)
            args = training_command(self.path, key, config, self.path,
                                    self.path / "attempt", self.path / "cache")
            self.assertEqual(args[args.index("--epochs") + 1], "500")
            self.assertEqual(args[args.index("--embedding-dim") + 1], "256")
            self.assertEqual(args[args.index("--knn-device") + 1], "cpu")
            self.assertNotIn(str(self.path / "main_stair5_v8.py"), args)
            self.assertNotIn("--batch-size", args)
        with self.assertRaises(ValueError):
            training_command(self.path, "unknown", config, self.path, "attempt", "cache")
        with self.assertRaises(ValueError):
            training_command(self.path, "baby", config, self.path, "attempt", "cache", epochs=0)

    def test_input_links_leave_dataset_cache_directory_writable(self):
        source, target = self.path / "input", self.path / "processed"
        source.mkdir()
        for filename in REQUIRED_DATA:
            (source / filename).write_text("fixture", encoding="utf-8")
        link_processed_dataset(source, target)
        self.assertFalse(target.is_symlink())
        for filename in REQUIRED_DATA:
            self.assertTrue(os.path.samefile(target / filename, source / filename))
        (target / "generated_cache.pkl").write_bytes(b"fixture-cache")
        self.assertFalse((source / "generated_cache.pkl").exists())
        link_processed_dataset(source, target)

    def test_resume_rejects_model_only_and_requires_existing_supports(self):
        (self.path / "main_stair5_v8_256d.py").touch()
        config = self.path / CONFIGS["baby"]
        config.touch()
        (self.path / "Processed" / DATASETS["baby"]).mkdir(parents=True)
        best = self.path / "best_model.pt"
        best.touch()
        base = (self.path, "baby", config, self.path, "attempt", "cache")
        with self.assertRaises(ValueError):
            training_command(*base, resume_from=best)
        checkpoint = self.path / "training_checkpoint.pt"
        checkpoint.touch()
        args = training_command(*base, resume_from=checkpoint)
        self.assertIn("--resume-from", args)
        with self.assertRaises(ValueError):
            training_command(*base, support_files="absent_text.pt,absent_visual.pt")

    def test_selected_test_is_not_last_epoch_or_best_test_metric(self):
        log = self.path / "training.log"
        log.write_text(
            "[Coach] >>> VALID @Epoch: 500 >>> || RECALL@20 Avg: 0.100 || NDCG@20 Avg: 0.044\n"
            "[Coach] >>> TEST @Epoch: 500 >>> || RECALL@20 Avg: 0.300 || NDCG@20 Avg: 0.200\n"
            "[Coach] >>> Load best model @Epoch: 430 (-1)\n"
            "[Coach] >>> VALID @Epoch: 430 >>> || RECALL@20 Avg: 0.101 || NDCG@20 Avg: 0.045\n"
            "[Coach] >>> TEST @Epoch: 430 >>> || RECALL@10 Avg: 0.0675 || RECALL@20 Avg: 0.1054 "
            "|| NDCG@10 Avg: 0.0365 || NDCG@20 Avg: 0.0462\n", encoding="utf-8")
        result = read_evaluation(log)
        self.assertEqual(result["best_epoch"], 430)
        self.assertEqual(result["test_metrics"]["NDCG@20"], 0.0462)
        self.assertEqual([r["epoch"] for r in result["validation"]], [430, 500])
        log.write_text("TEST @Epoch: 500 >>> NDCG@20 Avg: 0.2\n", encoding="utf-8")
        self.assertEqual(read_evaluation(log)["test_metrics"], {})

    def test_resume_marker_cannot_reuse_an_earlier_selected_test(self):
        log = self.path / "training.log"
        log.write_text("Load best model @Epoch: 20\nTEST @Epoch: 20 >>> NDCG@20 Avg: 0.1\n"
                       "Load best model @Epoch: 30\nTEST @Epoch: 40 >>> NDCG@20 Avg: 0.9\n",
                       encoding="utf-8")
        self.assertEqual(read_evaluation(log)["test_metrics"], {})

    def test_telemetry_retains_real_fields_and_rejects_corruption(self):
        path = self.path / "training_telemetry.jsonl"
        path.write_text('\n'.join(json.dumps(record) for record in (
            {"epoch": 2, "loss": 0.5, "peak_allocated_gib": 2.0},
            {"epoch": 1, "loss": 0.6, "peak_reserved_gib": 3.0},
            {"epoch": 2, "loss": 0.4, "peak_allocated_gib": 2.5})), encoding="utf-8")
        result = read_telemetry(self.path)
        self.assertEqual([r["epoch"] for r in result], [1, 2])
        self.assertEqual(result[-1]["loss"], 0.4)
        self.assertEqual(result[-1]["peak_allocated_gib"] * 1024, 2560)
        path.write_text("invalid json\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid telemetry"):
            read_telemetry(self.path)

    def runner_namespace(self, source):
        """Execute the actual notebook runner against a stdlib-only child fixture."""
        notebook = json.loads((ROOT / "notebook/P5/stair5_v8_256D.ipynb").read_text(encoding="utf-8"))
        function = next(node for cell in notebook["cells"] if cell["cell_type"] == "code"
                        for node in ast.parse("".join(cell["source"])).body
                        if isinstance(node, ast.FunctionDef) and node.name == "train_dataset")
        (self.path / "main_stair5_v8_256d.py").write_text(source, encoding="utf-8")
        config = self.path / CONFIGS["baby"]
        config.touch()
        (self.path / "Processed" / DATASETS["baby"]).mkdir(parents=True)
        session = self.path / "session"
        session.mkdir()
        namespace = dict(Path=Path, json=json, shutil=shutil, threading=threading,
                         time=time, subprocess=subprocess, sys=sys,
                         training_command=training_command, read_evaluation=read_evaluation,
                         METRICS=("Recall@10", "Recall@20", "NDCG@10", "NDCG@20"),
                         PREFLIGHT_OK=True, ACTIVE_RUN=None, REPO=self.path,
                         SESSION=session, OUTPUT_ROOT=self.path, PREPARED={"baby": self.path},
                         RESUME_FROM={}, RUNTIME_CONFIGS={"baby": config}, DATA_ROOT=self.path,
                         EPOCHS=500, EVAL_FREQ=5, MODE="full", DEVICE="0", SUPPORT_FILES={},
                         SOURCE_SHA="fixture", RESULTS={}, SUB_ENV=os.environ.copy(),
                         monitor_memory=lambda pid, target, stopped: None)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_runner", "exec"), namespace)
        return namespace

    def test_actual_runner_records_selected_metrics_and_refuses_overwrite(self):
        namespace = self.runner_namespace(
            'print("Load best model @Epoch: 430")\n'
            'print("TEST @Epoch: 430 >>> RECALL@10 Avg: 0.0675 || RECALL@20 Avg: 0.1054 '
            '|| NDCG@10 Avg: 0.0365 || NDCG@20 Avg: 0.0462")\n')
        result = namespace["train_dataset"]("baby")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["best_epoch"], 430)
        self.assertIsNone(namespace["ACTIVE_RUN"])
        status = Path(result["artifact"]) / "runner_status.json"
        self.assertTrue(status.is_file())
        before = status.read_bytes()
        with self.assertRaises(FileExistsError):
            namespace["train_dataset"]("baby")
        self.assertEqual(status.read_bytes(), before)

    def test_actual_runner_fails_and_preserves_child_error_log(self):
        namespace = self.runner_namespace('import sys\nprint("fixture failed", flush=True)\nsys.exit(7)\n')
        with self.assertRaisesRegex(RuntimeError, "Training failed"):
            namespace["train_dataset"]("baby")
        result = namespace["RESULTS"][("baby", 1)]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("fixture failed", Path(result["log"]).read_text(encoding="utf-8"))
        self.assertIsNone(namespace["ACTIVE_RUN"])


if __name__ == "__main__":
    unittest.main()
