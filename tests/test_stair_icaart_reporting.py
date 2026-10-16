"""Native-evaluation receipt contracts, without fabricated statistical samples."""
import json
from types import SimpleNamespace
import pytest
from stair_icaart_reporting import atomic_json, native_metric_snapshot, validate_selected_receipt


def test_native_snapshot_keeps_unrounded_values_and_excludes_inactive():
    meter = SimpleNamespace(active=True, count=12, name="NDCG@20", avg=.046234567891)
    inactive = SimpleNamespace(active=False, count=0, name="RECALL@20", avg=.9)
    assert native_metric_snapshot({"test": {"NDCG": [meter, inactive]}}, "test") == {"NDCG@20": .046234567891}


def test_missing_selected_metrics_are_errors_not_baseline_substitutes():
    with pytest.raises(ValueError, match="Missing"):
        validate_selected_receipt({"mode": "test", "selected_checkpoint": True,
                                   "selection": "validation NDCG@20", "metrics": {}})
    with pytest.raises(ValueError, match="selected-checkpoint"):
        validate_selected_receipt({"mode": "test", "selected_checkpoint": False})


def test_atomic_receipt_rejects_nonfinite_values_without_overwriting(tmp_path):
    path = tmp_path / "metrics.json"
    atomic_json(path, {"status": "NOT_RUN"})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"status": "NOT_RUN"}
    assert not list(tmp_path.glob("*.tmp"))


def test_selected_receipt_accepts_only_valid_measured_values():
    receipt = {"mode": "test", "selected_checkpoint": True, "selection": "validation NDCG@20",
               "epoch": 430, "metrics": {"RECALL@10": .06, "RECALL@20": .10,
                                          "NDCG@10": .03, "NDCG@20": .046234567891}}
    assert validate_selected_receipt(receipt) is receipt
    receipt["metrics"]["NDCG@20"] = 1.2
    with pytest.raises(ValueError, match="Invalid measured metric"):
        validate_selected_receipt(receipt)
