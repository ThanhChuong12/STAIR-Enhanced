"""Sparse CAE graph algebra, exact bypass and native-cache integrity tests.

Tiny dense fixtures are used only to verify spectral claims independently.
Production graph construction never allocates a catalog-squared dense array.
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
        for name in list(sys.modules):
            if name.startswith("models."):
                del sys.modules[name]
        graph = importlib.import_module("models.stair6_v1_graph")


class SourceContract(unittest.TestCase):
    def test_no_catalog_dense_construction(self):
        source = (ROOT / "models/stair6_v1_graph.py").read_text(encoding="utf-8")
        attributes = {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)}
        self.assertFalse({"to_dense", "toarray", "cdist"} & attributes)
        self.assertIn("weights_only=True", source)
        self.assertIn("@torch.no_grad()", source)


@unittest.skipIf(torch is None, "Requires genuine NumPy, SciPy and PyTorch")
class CAEGraphTests(unittest.TestCase):
    def setUp(self):
        self.raw = sp.csr_matrix(np.array([
            [0, 1, 2, 0, 0], [1, 0, 3, 0, 0], [2, 3, 0, 1, 0],
            [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
        self.cf = sp.csr_matrix((5, 5), dtype=np.float32)
        original, _ = graph._blend(self.raw, self.cf, .1)
        self.baseline = types.SimpleNamespace(operator=graph.scipy_to_tensor(original), metadata={"version": 8})
        self.edges = torch.tensor([[0, 0, 1, 1, 2, 3], [0, 1, 1, 2, 3, 4]])
        self.category = types.SimpleNamespace(
            metadata_fingerprint="verified-fixture", audit={"informative_coverage": .8},
            memberships=sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]], dtype=np.float32)))
        score_matrix = self.raw.copy()
        score_matrix.data[:] = 0
        score_matrix[0, 1] = score_matrix[1, 0] = .4
        self.scores = score_matrix.data.copy()
        self.options = dict(baseline_state=self.baseline, raw_semantic=self.raw, raw_cf=self.cf,
                            train_edges=self.edges, n_users=4, n_items=5, category_data=self.category)

    def _build(self, **options):
        arguments = dict(self.options)
        arguments.update(options)
        with mock.patch.object(graph, "build_category_edge_scores", return_value=(self.scores, {"positive_category_pairs": 1})):
            return graph.build_stair6_v1_graph(**arguments)

    def test_bypass_is_tensor_identity_without_metadata_or_graph(self):
        for options in ({"strength": 0}, {"category_mode": "off"}):
            with mock.patch.object(graph, "build_category_edge_scores", side_effect=AssertionError("Bypass must not fit metadata")):
                state = graph.build_stair6_v1_graph(self.baseline, **options)
            self.assertIs(state.operator, self.baseline.operator)
            self.assertFalse(state.metadata["category_active"])

    def test_enrichment_formula_same_support_spectral_bound_and_no_grad(self):
        state = self._build(strength=.5, beta=np.array([.1, .5, .9]))
        enriched = graph.enrich_existing_weights(self.raw, self.scores, .5)
        expected = graph.normalize_graph_with_isolated_fallback(enriched)
        actual = graph.tensor_to_scipy(state.operator)
        np.testing.assert_array_equal(enriched.indices, self.raw.indices)
        np.testing.assert_array_equal(enriched.indptr, self.raw.indptr)
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-7)
        self.assertLessEqual(float(np.linalg.norm(actual.toarray(), 2)), 1 + 1e-6)
        self.assertEqual(actual[4, 4], 1)
        self.assertEqual(state.metadata["effective_eta"], 0)
        self.assertGreater(state.metadata["relative_operator_delta_vs_v8"], 0)
        self.assertGreater(state.metadata["bsc_probe_relative_delta"], 0)
        self.assertEqual(state.operator.device.type, "cpu")
        self.assertEqual(state.operator.layout, torch.sparse_csr)
        self.assertIsNone(state.operator.grad_fn)
        self.assertFalse(state.operator.requires_grad)

    def test_uniform_enrichment_cancels_after_normalization(self):
        uniform = np.full(self.raw.nnz, .6, dtype=np.float32)
        changed = graph.normalize_graph_with_isolated_fallback(graph.enrich_existing_weights(self.raw, uniform, .5))
        native = graph.normalize_graph_with_isolated_fallback(self.raw)
        np.testing.assert_allclose(changed.toarray(), native.toarray(), atol=1e-7)

    def test_cf_blend_uses_native_isolated_fallback(self):
        cf = sp.csr_matrix(np.array([[0, 0, 0, 0, 0], [0, 0, .3, 0, 0], [0, .3, 0, 0, 0],
                                    [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.float32))
        original, _ = graph._blend(self.raw, cf, .1)
        baseline = types.SimpleNamespace(operator=graph.scipy_to_tensor(original), metadata={})
        state = self._build(raw_cf=cf, baseline_state=baseline)
        enriched = graph.normalize_graph_with_isolated_fallback(graph.enrich_existing_weights(self.raw, self.scores, .5))
        expected = .9 * enriched + .1 * graph.normalize_cf_graph_with_fallback(cf)
        np.testing.assert_allclose(graph.tensor_to_scipy(state.operator).toarray(), expected.toarray(), atol=1e-7)
        self.assertEqual(state.metadata["effective_eta"], .1)

    def test_real_train_association_to_final_operator(self):
        category = types.SimpleNamespace(
            metadata_fingerprint="association-fixture", audit={"informative_coverage": 1.0},
            memberships=sp.csr_matrix(np.array([
                [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)))
        # n_A=2, n_B=3, n_AB=2, N*=4. Equation (6) gives
        # (2/7)*tanh(log(4/3)) = 0.08; same-category evidence is zero.
        category_scores = self.raw.copy()
        category_scores.data[:] = .08
        category_scores[0, 2] = category_scores[2, 0] = 0
        expected = graph.normalize_graph_with_isolated_fallback(
            graph.enrich_existing_weights(self.raw, category_scores.data, .5))
        arguments = dict(self.options, category_data=category)
        state = graph.build_stair6_v1_graph(**arguments)
        np.testing.assert_allclose(graph.tensor_to_scipy(state.operator).toarray(), expected.toarray(), atol=1e-7)
        self.assertGreater(state.metadata["relative_operator_delta_vs_v8"], 0)

    def test_invalid_and_inactive_sources_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "verified item mapping"):
            self._build(category_data=None)
        with mock.patch.object(graph, "build_category_edge_scores", return_value=(np.zeros_like(self.scores), {})):
            with self.assertRaises(graph.InactiveCategorySignal):
                graph.build_stair6_v1_graph(**self.options)
        wrong = self.raw.copy()
        wrong.data[0] *= 2
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self._build(raw_semantic=wrong)
        other_cf = sp.csr_matrix((np.array([.3, .3], dtype=np.float32), ([0, 1], [1, 0])), shape=(5, 5))
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            self._build(eta=.2, raw_cf=other_cf)

    def test_score_bounds_and_symmetry_are_enforced(self):
        for scores in (np.full(self.raw.nnz, np.nan), np.full(self.raw.nnz, -.1), np.full(self.raw.nnz, 1.1)):
            with self.assertRaises(ValueError):
                graph.enrich_existing_weights(self.raw, scores, .5)
        asymmetric = self.scores.copy()
        asymmetric[0] = .7
        with self.assertRaisesRegex(ValueError, "symmetric"):
            graph.enrich_existing_weights(self.raw, asymmetric, .5)

    def test_content_cache_and_corruption_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._build(cache_dir=directory)
            second = self._build(cache_dir=directory)
            self.assertFalse(first.metadata["category_cache_hit"])
            self.assertTrue(second.metadata["category_cache_hit"])
            torch.testing.assert_close(first.operator, second.operator, rtol=0, atol=0)
            path = next(Path(directory).glob("cae_*.pt"))
            payload = torch.load(path, weights_only=True)
            payload["scores"][0] += .1
            torch.save(payload, path)
            with self.assertWarnsRegex(RuntimeWarning, "checksum"):
                repaired = self._build(cache_dir=directory)
            self.assertFalse(repaired.metadata["category_cache_hit"])
            torch.testing.assert_close(first.operator, repaired.operator, rtol=0, atol=0)
            changed = self._build(cache_dir=directory, strength=.25)
            self.assertNotEqual(first.metadata["category_cache_key"], changed.metadata["category_cache_key"])

    def test_native_cf_cache_matches_builder_and_duplicate_pairs(self):
        edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 1, 2, 3, 4]])
        with tempfile.TemporaryDirectory() as directory:
            raw, first = graph.load_native_raw_cf(edges, 4, 5, cache_dir=directory)
            duplicated = torch.cat([edges, edges[:, :2]], dim=1)
            loaded, second = graph.load_native_raw_cf(duplicated, 4, 5, cache_dir=directory)
            np.testing.assert_array_equal(raw.toarray(), loaded.toarray())
            self.assertFalse(first["cf_cache_hit"])
            self.assertTrue(second["cf_cache_hit"])
            self.assertEqual(first["cf_cache_key"], second["cf_cache_key"])
            path = next(Path(directory).glob("cf_*.pt"))
            payload = torch.load(path, weights_only=True)
            payload["fingerprint"] = "corrupt"
            torch.save(payload, path)
            with self.assertWarnsRegex(RuntimeWarning, "checksum"):
                rebuilt, audit = graph.load_native_raw_cf(edges, 4, 5, cache_dir=directory)
            self.assertFalse(audit["cf_cache_hit"])
            np.testing.assert_array_equal(raw.toarray(), rebuilt.toarray())


if __name__ == "__main__":
    unittest.main()
