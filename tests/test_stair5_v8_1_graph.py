"""Restricted-support consensus algebra and cache integrity tests.

Numerical tests require genuine NumPy/SciPy/PyTorch; they are explicitly skipped
when unavailable. The isolated package import avoids executing FreeRec-backed
``models/__init__.py`` and does not mock scientific computation.
"""

import ast
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
try:
    import numpy as np
    import scipy.sparse as sp
    import torch
except ImportError:
    np = sp = torch = None

graph = None
if torch is not None:
    package = types.ModuleType("models")
    package.__path__ = [str(ROOT / "models")]
    with mock.patch.dict(sys.modules, {"models": package}):
        # patch.dict restores the import registry when exiting; test discovery
        # cannot replace the real project's package for unrelated tests.
        for name in list(sys.modules):
            if name.startswith("models."):
                del sys.modules[name]
        graph = importlib.import_module("models.stair5_v8_1_graph")


class TestConsensusSourceContract(unittest.TestCase):
    def test_no_full_catalog_dense_pairwise_operations(self):
        source = (ROOT / "models/stair5_v8_1_graph.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
        self.assertNotIn("to_dense", attributes)
        self.assertNotIn("cdist", attributes)
        self.assertNotIn("toarray", attributes)
        self.assertIn("compute_selected_edge_cosines", source)
        self.assertIn("weights_only=True", source)
        self.assertIn("@torch.no_grad()", source)


@unittest.skipIf(torch is None, "Requires genuine PyTorch, NumPy and SciPy")
class TestConsensusGraph(unittest.TestCase):
    def setUp(self):
        self.edges = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 1, 2, 0, 1, 2, 0, 2, 3, 0, 2, 3],
        ], dtype=torch.int64)
        self.text = torch.tensor([
            [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
            [-1., 0., 0.], [0., 0., 0.],
        ])
        self.visual = torch.tensor([
            [0., 1., 0.], [0., 1., 1.], [0., -1., 0.],
            [1., 0., 0.], [0., 0., 0.],
        ])
        self.options = dict(train_edges=self.edges, n_users=4, n_items=5,
                            mfeats=[self.text, self.visual], k_cf=5,
                            c_min=2, t_shrinkage=5.0)

    def build(self, **changes):
        return graph.build_consensus_graph(**{**self.options, **changes})

    def test_formula_symmetry_support_and_zero_isolate(self):
        state = self.build()
        actual = graph.tensor_to_scipy(state.operator)
        raw = graph.build_candidate_support_graph(
            self.edges, n_items=5, n_users=4, k_cf=5, c_min=2, t_shrinkage=5.0)
        dense = raw.toarray().astype(np.float64)
        expected_weights = dense.copy()
        for i, j in zip(*np.nonzero(dense)):
            def positive_cos(feature):
                x, y = feature[i].numpy(), feature[j].numpy()
                return max(0., float(np.dot(x, y) / max(1e-12, np.linalg.norm(x) * np.linalg.norm(y))))
            semantic = (5 * positive_cos(self.text) + positive_cos(self.visual)) / 6
            expected_weights[i, j] *= .05 + .95 * semantic
        degrees = expected_weights.sum(1)
        inverse = np.zeros_like(degrees)
        inverse[degrees > 0] = 1 / np.sqrt(degrees[degrees > 0])
        expected = inverse[:, None] * expected_weights * inverse[None, :]
        np.testing.assert_allclose(actual.toarray(), expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(actual.astype(bool).toarray(), raw.astype(bool).toarray())
        self.assertEqual(actual[4, 4], 0)
        self.assertEqual(actual.getrow(4).nnz, 0)
        self.assertFalse(state.coverage[4])
        self.assertTrue(torch.equal(state.operator.values(), state.operator.values().detach()))
        self.assertIsNone(state.operator.grad_fn)
        self.assertEqual(state.operator.device.type, "cpu")
        self.assertLessEqual(np.linalg.norm(actual.toarray(), ord=2), 1 + 1e-6)

    def test_cf_only_and_constant_floor_cancel_under_normalization(self):
        raw = self.build(mode="cf-only")
        floor_one = self.build(floor=1.)
        torch.testing.assert_close(raw.operator.to_dense(), floor_one.operator.to_dense())
        self.assertTrue(torch.equal(raw.coverage, floor_one.coverage))

    def test_empty_support_and_off_do_not_invent_identity(self):
        empty = self.build(k_cf=0)
        self.assertEqual(empty.operator._nnz(), 0)
        self.assertFalse(empty.coverage.any())
        self.assertFalse(empty.metadata["active"])
        with mock.patch.object(graph, "build_candidate_support_graph", side_effect=AssertionError("CF rebuild")):
            off = self.build(mode="off", mfeats=None)
        self.assertEqual(off.operator._nnz(), 0)
        self.assertFalse(off.coverage.any())

    def test_duplicate_train_pairs_and_input_order_do_not_change_statistics(self):
        duplicate = torch.cat((self.edges.flip(1), self.edges[:, :4]), dim=1)
        first, second = self.build(), self.build(train_edges=duplicate)
        torch.testing.assert_close(first.operator.to_dense(), second.operator.to_dense())
        self.assertEqual(first.metadata["graph_fingerprint"], second.metadata["graph_fingerprint"])

    def test_cache_reuse_feature_invalidation_and_raw_cf_reuse(self):
        with tempfile.TemporaryDirectory() as cache:
            first = self.build(cache_dir=cache)
            self.assertFalse(first.metadata["cache_hit"])
            with mock.patch.object(graph, "build_candidate_support_graph", side_effect=AssertionError("Extra CF build")):
                warm = self.build(cache_dir=cache)
                self.assertTrue(warm.metadata["cache_hit"])
                self.assertTrue(warm.metadata["cf_cache_hit"])
                changed_features = self.text.clone()
                changed_features[1] = torch.tensor([-1., 0., 0.])
                changed = self.build(cache_dir=cache, mfeats=[changed_features, self.visual])
                self.assertFalse(changed.metadata["cache_hit"])
                self.assertTrue(changed.metadata["cf_cache_hit"])
            self.assertNotEqual(first.metadata["cache_key"], changed.metadata["cache_key"])
            self.assertNotEqual(first.metadata["graph_fingerprint"], changed.metadata["graph_fingerprint"])
            self.assertEqual(len(list(Path(cache).glob("cf_*.pt"))), 1)

    def test_raw_cache_signature_matches_v8_exactly(self):
        with tempfile.TemporaryDirectory() as cache:
            self.build(cache_dir=cache)
            raw_file = next(Path(cache).glob("cf_*.pt"))
            payload = torch.load(raw_file, weights_only=True)
            pairs = graph.binary_train_keys(self.edges, 4, 5)
            digest = graph.hashlib.sha256()
            graph._digest_array(digest, pairs)
            expected = {
                "cache_version": graph.CACHE_VERSION, "train": digest.hexdigest(),
                "users": 4, "items": 5, "k": 5, "cmin": 2, "t": 5.0,
                "source": graph.file_sha256(ROOT / "models/stair5_v4_graph.py"),
            }
            self.assertEqual(payload["signature"], expected)
            cache_key = graph.hashlib.sha256(graph.json.dumps(expected, sort_keys=True).encode()).hexdigest()
            self.assertEqual(raw_file.name, f"cf_{cache_key}.pt")

    def test_corrupt_consensus_cache_and_coverage_are_rebuilt(self):
        with tempfile.TemporaryDirectory() as cache:
            first = self.build(cache_dir=cache)
            file = next(Path(cache).glob("consensus_v81_*.pt"))
            file.write_bytes(b"bad checkpoint")
            with self.assertWarns(RuntimeWarning):
                rebuilt = self.build(cache_dir=cache)
            torch.testing.assert_close(first.operator.to_dense(), rebuilt.operator.to_dense())
            payload = torch.load(file, weights_only=True)
            payload["coverage"] = torch.ones(5, dtype=torch.bool)
            torch.save(payload, file)
            with self.assertWarns(RuntimeWarning):
                final = self.build(cache_dir=cache)
            self.assertFalse(final.coverage[4])
            payload = torch.load(file, weights_only=True)
            payload["operator"].values().mul_(2.)
            torch.save(payload, file)
            with self.assertWarns(RuntimeWarning):
                restored = self.build(cache_dir=cache)
            torch.testing.assert_close(first.operator.to_dense(), restored.operator.to_dense())

    def test_raw_cf_cache_corruption_rebuilds_from_binary_train_only(self):
        with tempfile.TemporaryDirectory() as cache:
            first = self.build(cache_dir=cache)
            raw_file = next(Path(cache).glob("cf_*.pt"))
            payload = torch.load(raw_file, weights_only=True)
            payload["operator"].values().mul_(3.)
            torch.save(payload, raw_file)
            with mock.patch.object(graph, "build_candidate_support_graph",
                                   wraps=graph.build_candidate_support_graph) as rebuild:
                with self.assertWarns(RuntimeWarning):
                    restored = self.build(cache_dir=cache)
            rebuild.assert_called_once()
            self.assertFalse(restored.metadata["cf_cache_hit"])
            self.assertTrue(restored.metadata["cache_hit"])
            torch.testing.assert_close(first.operator.to_dense(), restored.operator.to_dense())

    def test_raw_modality_rotations_preserve_operator(self):
        text_rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        visual_rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        first = self.build()
        rotated = self.build(mfeats=[self.text @ text_rotation, self.visual @ visual_rotation])
        torch.testing.assert_close(first.operator.to_dense(), rotated.operator.to_dense())
        self.assertNotEqual(first.metadata["feature_hashes"], rotated.metadata["feature_hashes"])

    def test_placebo_permutation_is_stratified_deterministic_and_symmetric(self):
        # Many edges in a single degree-pair stratum, with distinct multipliers.
        row = np.zeros(12, dtype=np.int64)
        column = np.array([1, 2, 3] * 4, dtype=np.int64)
        values = np.arange(12, dtype=np.float64) / 12
        pairs = np.arange(20, dtype=np.int64)
        a, audit = graph._stratified_placebo(values, row, column, pairs, 20, 17)
        b, _ = graph._stratified_placebo(values, row, column, pairs, 20, 17)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.sort(values))
        self.assertGreater(audit["placebo_changed_edges"], 0)
        state = self.build(mode="placebo", placebo_seed=17)
        matrix = graph.tensor_to_scipy(state.operator)
        np.testing.assert_array_equal(matrix.toarray(), matrix.T.toarray())
        self.assertTrue(torch.equal(state.coverage, self.build().coverage))

    def test_bounded_cosine_chunks_match_and_validate_zero_rows(self):
        large = self.build(feature_memory_budget_mib=64.)
        tiny = self.build(feature_memory_budget_mib=0.0001)
        torch.testing.assert_close(large.operator.to_dense(), tiny.operator.to_dense())
        self.assertEqual(tiny.metadata["text_zero_rows"], 1)
        self.assertEqual(tiny.metadata["visual_zero_rows"], 1)
        with self.assertRaises(MemoryError):
            self.build(feature_memory_budget_mib=1e-10)

    def test_invalid_inputs_rejected(self):
        for changes in ({"mode": "invalid"}, {"floor": 0}, {"floor": float("nan")},
                        {"text_weight": 2}, {"memory_budget_mib": 0},
                        {"feature_memory_budget_mib": 0}, {"k_cf": -1},
                        {"c_min": 0}, {"t_shrinkage": 0}, {"placebo_seed": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(**changes)
        bad = self.text.clone()
        bad[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            self.build(mfeats=[bad, self.visual])
        with self.assertRaises(ValueError):
            self.build(mfeats=[self.text.double(), self.visual])
        with self.assertRaises(ValueError):
            self.build(mfeats=[self.text[:-1], self.visual])


if __name__ == "__main__":
    unittest.main()
