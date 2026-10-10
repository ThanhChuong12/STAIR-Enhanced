"""Dependency-free experiment guards and analytical memory checks."""

import ast
import copy
from pathlib import Path
import types
import unittest

from stair5_v8_256d_config import (
    CAPACITY_CONTRACT,
    DEFAULT_CONFIG,
    tensor_payload_ledger,
    validate_capacity_config,
    with_default_config,
)


ROOT = Path(__file__).resolve().parents[1]


def flat_config_fixture(path):
    """Read the intentionally flat fixtures without requiring PyYAML.

This is only a test helper, not a production YAML parser. The native FreeRec
parser remains covered by the runtime integration tests.
"""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value in ("true", "false"):
            values[key] = value == "true"
        else:
            try:
                values[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                values[key] = value
    return values


def valid_config():
    return {
        **CAPACITY_CONTRACT,
        "num_neighbors": [5, 1],
        "optimizer": "adamwsevo",
        "eval_valid": True,
        "eval_test": False,
        "which4best": "NDCG@20",
        "gamma": 0.2,
        "lr": 0.001,
    }


class TestCapacityContract(unittest.TestCase):
    def test_three_dataset_configs_and_effective_electronics_chunk(self):
        expected = {"Baby": (0.1, 0.3, 1024, 1024),
                    "Sports": (0.2, 0.1, 1024, 1024),
                    "Electronics": (0.4, 0.1, 4096, 128)}
        for dataset, settings in expected.items():
            with self.subTest(dataset=dataset):
                cfg = flat_config_fixture(ROOT / "configs" / f"Amazon2014{dataset}_STAIR5_v8_256D.yaml")
                validate_capacity_config(cfg)
                self.assertEqual(tuple(cfg[k] for k in ("gamma", "weight_decay", "batch_size", "cl_chunk_size")), settings)
                self.assertEqual(cfg["epochs"], 500)
                self.assertEqual(cfg["eval_freq"], 5)
                self.assertEqual(cfg["mfiles"], "textual_modality.pkl,visual_modality.pkl")

    def test_attribute_config_is_supported_without_mutation(self):
        cfg = valid_config()
        before = copy.deepcopy(cfg)
        validate_capacity_config(types.SimpleNamespace(**cfg))
        self.assertEqual(cfg, before)

    def test_each_architecture_change_is_rejected(self):
        for key in CAPACITY_CONTRACT:
            with self.subTest(key=key):
                cfg = valid_config()
                cfg[key] = "unexpected"
                with self.assertRaisesRegex(ValueError, key):
                    validate_capacity_config(cfg)

    def test_wrong_dimension_or_additional_loss_is_rejected(self):
        for changes in ({"embedding_dim": 64}, {"embedding_dim": 128},
                        {"lambda_cl": 0.01}, {"consensus_dose": 0.2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_capacity_config({**valid_config(), **changes})

    def test_evaluation_protocol_changes_are_rejected(self):
        for changes in ({"eval_test": True}, {"eval_valid": False},
                        {"which4best": "Recall@20"}, {"optimizer": "adamw"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_capacity_config({**valid_config(), **changes})

    def test_invalid_scalar_values_are_rejected(self):
        for key in ("gamma", "lr"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_capacity_config({**valid_config(), key: value})

    def test_explicit_config_arguments_are_preserved(self):
        for args in (["--config", "custom.yaml", "--seed", "2"],
                     ["--config=custom.yaml"], ["-c", "custom.yaml"]):
            self.assertEqual(with_default_config(args), args)

    def test_default_config_does_not_change_other_arguments(self):
        args = ["--seed", "3", "--epochs", "5"]
        self.assertEqual(with_default_config(args), ["--config", str(DEFAULT_CONFIG), *args])
        self.assertEqual(args, ["--seed", "3", "--epochs", "5"])
        self.assertTrue(DEFAULT_CONFIG.is_file())

    def test_electronics_tensor_payload_is_not_an_800_mib_claim(self):
        ledger = tensor_payload_ledger(192403, 63001)
        self.assertEqual(ledger["embedding_parameters"], 65383424)
        self.assertEqual(ledger["embedding_weights_grad_two_adam_moments_bytes_fp32"] / 2**20, 997.671875)
        self.assertEqual(ledger["one_item_activation_bytes_fp32"] / 2**20, 61.5244140625)
        self.assertNotIn("peak_memory", ledger)

    def test_ledger_rejects_invalid_shapes(self):
        for args in ((0, 10), (10, -1), (True, 10), (10, 10, 0), (10, 10, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tensor_payload_ledger(*args)


if __name__ == "__main__":
    unittest.main()
