"""Torch-only V8.1 algebra, autograd and unchanged-InfoNCE contracts.

Load leaf modules without importing the FreeRec-backed models package. A
missing PyTorch runtime is an explicit skip, never reported as a pass.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

    package_name = "_stair5_v81_leaf_tests"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "models")]
    sys.modules[package_name] = package

    def load_leaf(name):
        fullname = f"{package_name}.{name}"
        specification = importlib.util.spec_from_file_location(fullname, Path(package.__path__[0]) / f"{name}.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[fullname] = module
        specification.loader.exec_module(module)
        return module

    residual_module = load_leaf("stair5_v8_1_consensus")
    objectives_module = load_leaf("stair5_v8_1_objectives")
    baseline_module = sys.modules[f"{package_name}.stair5_v4_objectives"]
    ConsensusResidual = residual_module.ConsensusResidual
    consensus_dose = residual_module.consensus_dose
    NLGCLConsensus_Module = objectives_module.NLGCLConsensus_Module
    NLGCL_Module = baseline_module.NLGCL_Module


@unittest.skipUnless(HAS_TORCH, "PyTorch runtime not installed")
class TestConsensusResidual(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(83)
        self.x = torch.randn(7, 16, generator=generator)
        self.m = 3.0 * torch.randn(7, 16, generator=generator)
        self.coverage = torch.ones(7, dtype=torch.bool)

    def test_seed_isolation_parameter_count_and_reproducibility(self):
        original = torch.random.get_rng_state().clone()
        module = ConsensusResidual()
        self.assertTrue(torch.equal(original, torch.random.get_rng_state()))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 41)
        duplicate = ConsensusResidual()
        for first, second in zip(module.parameters(), duplicate.parameters()):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(sum(parameter.numel() for parameter in ConsensusResidual(gate="fixed").parameters()), 0)
        _, context = module.gate_context(self.x, self.m, self.coverage)
        self.assertTrue(torch.allclose(module.gate_values(context), torch.full((7, 1), 0.5), atol=0.005))

    def test_zero_dose_is_object_identity_and_has_no_gate_gradient(self):
        module = ConsensusResidual()
        x = self.x.clone().requires_grad_()
        m = self.m.clone().requires_grad_()
        query = module(x, m, self.coverage, rho=0.0)
        self.assertIs(query, x)
        query.square().mean().backward()
        self.assertIsNone(m.grad)
        self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_cap_context_detachment_and_forward_norm_bound(self):
        module = ConsensusResidual()
        x, m = self.x.clone().requires_grad_(), self.m.clone().requires_grad_()
        cap, context = module.gate_context(x, m, self.coverage)
        self.assertFalse(cap.requires_grad)
        self.assertFalse(context.requires_grad)
        query = module(x, m, self.coverage, rho=0.2)
        relative = torch.linalg.vector_norm(query - x, dim=-1) / torch.linalg.vector_norm(x, dim=-1)
        self.assertTrue(torch.all(relative <= 0.2 + 2e-6))
        self.assertTrue(torch.all(torch.linalg.vector_norm(cap * m, dim=-1) <= torch.linalg.vector_norm(x, dim=-1) + 2e-6))

    def test_orthogonal_rotation_equivariance(self):
        module = ConsensusResidual()
        generator = torch.Generator().manual_seed(11)
        rotation, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
        direct = module(self.x, self.m, self.coverage, rho=0.2) @ rotation
        rotated = module(self.x @ rotation, self.m @ rotation, self.coverage, rho=0.2)
        self.assertTrue(torch.allclose(direct, rotated, atol=2e-6, rtol=2e-6))

    def test_fixed_gate_zero_rows_and_tiny_norms(self):
        module = ConsensusResidual(gate="fixed")
        x = torch.tensor([[0.0, 0.0], [1e-12, 0.0], [1.0, 0.0]])
        m = torch.tensor([[2.0, 1.0], [0.0, 1e-4], [0.0, 0.0]])
        coverage = torch.tensor([True, True, False])
        cap, context = module.gate_context(x, m, coverage)
        query = module(x, m, coverage, rho=0.2)
        self.assertTrue(torch.isfinite(query).all())
        self.assertTrue(torch.equal(query[0], x[0]))
        self.assertTrue(torch.equal(query[2], x[2]))
        self.assertEqual(float(context[0, 0]), 0.0)
        self.assertEqual(float(context[1, 0]), 0.0)
        self.assertTrue(torch.allclose(query, x + 0.1 * cap * m))
        norms = torch.linalg.vector_norm(query - x, dim=-1)
        self.assertLessEqual(float(norms[1]), 0.2e-12)

    def test_gradient_flow_and_parameter_change_over_three_active_steps(self):
        module = ConsensusResidual()
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        before = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            x, m = self.x.clone().requires_grad_(), self.m.clone().requires_grad_()
            query = module(x, m, self.coverage, rho=0.2)
            query.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(torch.isfinite(m.grad).all())
            self.assertGreater(float(m.grad.abs().sum()), 0.0)
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
            optimizer.step()
        for name, parameter in module.named_parameters():
            self.assertFalse(torch.equal(before[name], parameter))
        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in module.__dict__.values()))

    def test_diagnostics_are_detached_device_scalars(self):
        module = ConsensusResidual()
        query = module(self.x.requires_grad_(), self.m.requires_grad_(), self.coverage, 0.2)
        results = module.diagnostic_summary(self.x, self.m, query)
        for value in results.values():
            self.assertEqual(value.ndim, 0)
            self.assertEqual(value.device, self.x.device)
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value))
        # A parallel residual disappears under the inherited unit normalization.
        parallel_query = module(self.x, self.x, self.coverage, 0.2)
        parallel = module.diagnostic_summary(self.x, self.x, parallel_query)
        self.assertLess(float(parallel["angular_shift_mean"]), 2e-7)

    def test_schedule_and_validation(self):
        self.assertEqual(consensus_dose(1), 0.0)
        self.assertEqual(consensus_dose(30), 0.0)
        self.assertAlmostEqual(consensus_dose(31), 0.2 / 30)
        self.assertEqual(consensus_dose(60), 0.2)
        self.assertEqual(consensus_dose(100), 0.2)
        self.assertEqual(consensus_dose(1, warmup_epochs=0, ramp_epochs=0), 0.2)
        for epoch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                consensus_dose(epoch)
        for rho in (-0.1, 1.0, float("nan"), True):
            with self.assertRaises(ValueError):
                ConsensusResidual()(self.x, self.m, self.coverage, rho)
            with self.assertRaises(ValueError):
                consensus_dose(1, rho_max=rho)
        with self.assertRaises(ValueError):
            ConsensusResidual(gate="unknown")
        with self.assertRaises(ValueError):
            ConsensusResidual(hidden=0)


@unittest.skipUnless(HAS_TORCH, "PyTorch runtime not installed")
class TestConsensusObjective(unittest.TestCase):
    def setUp(self):
        self.n_users, self.n_items = 3, 4
        generator = torch.Generator().manual_seed(42)
        self.layers = [torch.randn(7, 5, dtype=torch.float64, generator=generator) for _ in range(2)]
        self.users = torch.tensor([0, 0, 1, 2, 1])
        self.items = torch.tensor([1, 1, 2, 3, 1])

    def test_exact_baseline_bypass_with_duplicate_ids(self):
        base = NLGCL_Module(3, 4)
        consensus = NLGCLConsensus_Module(3, 4)
        expected = base(self.layers, self.users, self.items)
        self.assertTrue(torch.equal(expected, consensus(self.layers, self.users, self.items)))
        query = self.layers[1][3:][self.items]
        self.assertTrue(torch.equal(expected, consensus(self.layers, self.users, self.items, item_query=query)))

    def test_orientation_matches_manual_row_cross_entropy_and_preserves_layers(self):
        module = NLGCLConsensus_Module(3, 4)
        query = self.layers[1][3:][self.items] + 0.25
        snapshots = [layer.clone() for layer in self.layers]
        loss = module(self.layers, self.users, self.items, query)
        keys_u = F.normalize(self.layers[0][:3][self.users], dim=-1)
        keys_i = F.normalize(self.layers[0][3:][self.items], dim=-1)
        query_u = F.normalize(self.layers[1][:3][self.users], dim=-1)
        labels = torch.arange(self.users.numel())
        manual = 0.5 * F.cross_entropy(F.normalize(query, dim=-1) @ keys_u.T / 0.2, labels)
        manual += 0.5 * F.cross_entropy(query_u @ keys_i.T / 0.2, labels)
        self.assertTrue(torch.allclose(loss, manual, atol=1e-12))
        for before, after in zip(snapshots, self.layers):
            self.assertTrue(torch.equal(before, after))

    def test_dense_vs_checkpointed_chunks_loss_and_all_gradient_paths(self):
        values, gradients = [], []
        query_seed = self.layers[1][3:][self.items] + 0.25
        for chunk in (None, 2):
            layers = [layer.clone().requires_grad_() for layer in self.layers]
            query = query_seed.clone().requires_grad_()
            module = NLGCLConsensus_Module(3, 4, chunk_size=chunk)
            value = module(layers, self.users[:, None], self.items[:, None], query)
            value.backward()
            values.append(value.detach())
            gradients.append([layers[0].grad.clone(), layers[1].grad.clone(), query.grad.clone()])
            self.assertGreater(float(layers[0].grad[:3].abs().sum()), 0.0)
            self.assertGreater(float(layers[0].grad[3:].abs().sum()), 0.0)
            self.assertGreater(float(layers[1].grad[:3].abs().sum()), 0.0)
            self.assertGreater(float(query.grad.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(values[0], values[1], atol=1e-12))
        for dense, chunked in zip(*gradients):
            self.assertTrue(torch.allclose(dense, chunked, atol=1e-12, rtol=1e-10))

    def test_reject_multiple_gaps_and_bad_query_shape(self):
        with self.assertRaises(ValueError):
            NLGCLConsensus_Module(3, 4, G=2)
        module = NLGCLConsensus_Module(3, 4)
        with self.assertRaises(ValueError):
            module(self.layers, self.users, self.items, torch.zeros(4, 5, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
