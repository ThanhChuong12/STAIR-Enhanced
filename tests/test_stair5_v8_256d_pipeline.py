"""Same-dimension parity, graph invariance, autograd and resumable state tests.

Run in the intended PyTorch/FreeRec environment. Missing runtime dependencies
are explicit skips; an installed but broken runtime must fail visibly.
"""

import copy
import importlib.util
from pathlib import Path
import pickle
import tempfile
import types
import unittest

from stair5_v8_256d_config import CAPACITY_CONTRACT


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None
                  for name in ("torch", "freerec", "numpy", "scipy"))

if HAS_RUNTIME:
    import torch
    import models.freerec_compat
    from freerec.data.fields import Field, FieldTuple
    from freerec.data.tags import USER, ITEM, ID, LABEL
    from main_stair5_v8_256d import build_config
    from models.stair5_v8 import STAIR5_v8_Model
    from models.stair5_v8_256d import STAIR5_v8_256D_Model
    from models.stair5_v4_utils import save_training_checkpoint, load_training_checkpoint
    from optimizers.AdamW import AdamWSEvo

    class Dataset:
        def __init__(self, path):
            self.path = str(path)
            user, item = Field("User", USER, ID), Field("Item", ITEM, ID)
            user.count, item.count = 20, 270
            self.fields = FieldTuple([user, item, Field("Label", LABEL)])
            generator = torch.Generator().manual_seed(17)
            self.edge_index = torch.stack((torch.randint(20, (600,), generator=generator),
                                           torch.randint(270, (600,), generator=generator)))

        def train(self):
            return self

        def to_bigraph(self, edge_type="u2i"):
            return {"u2i": types.SimpleNamespace(edge_index=self.edge_index)}

        def to_normalized_adj(self, normalization="sym"):
            users, items = self.edge_index
            items = items + 20
            rows, cols = torch.cat((users, items)), torch.cat((items, users))
            degree = torch.bincount(rows, minlength=290).float().clamp_min(1)
            weights = degree[rows].rsqrt() * degree[cols].rsqrt()
            return torch.sparse_coo_tensor(torch.stack((rows, cols)), weights, (290, 290)).coalesce().to_sparse_csr()


@unittest.skipUnless(HAS_RUNTIME, "PyTorch/FreeRec/NumPy/SciPy runtime not installed")
class TestCapacityPipeline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.dataset = Dataset(self.path)
        generator = torch.Generator().manual_seed(23)
        for filename in ("text.pkl", "image.pkl"):
            with (self.path / filename).open("wb") as handle:
                pickle.dump(torch.randn(270, 280, generator=generator), handle)
        self.cfg = types.SimpleNamespace(
            **CAPACITY_CONTRACT, mfiles=["text.pkl", "image.pkl"], num_neighbors=[5, 1],
            gamma=0.2, beta1=0.9, beta2=0.999, cl_chunk_size=2, knn_chunk_size=32,
            cf_block_size=16, cf_memory_budget_mib=128.0, graph_cache_dir=str(self.path),
            artifact_dir=str(self.path), device="cpu", root=str(self.path), dataset="mock",
            lr=0.001, weight_decay=0.1, optimizer="adamwsevo", eval_valid=True,
            eval_test=False, which4best="NDCG@20", batch_size=4, seed=1,
            eval_freq=5, num_workers=0, anchor_seed=2718, anchor_per_stratum=1024,
            relation_threshold=0.7, relation_temperature=0.1, placebo_seed=1,
        )
        self.cfg.beta3 = 0.1 + 0.9 * (torch.arange(256) / 256).pow(self.cfg.gamma)

    def model(self, specialized=True, dimension=256):
        cfg = copy.copy(self.cfg)
        cfg.embedding_dim = dimension
        cfg.beta3 = 0.1 + 0.9 * (torch.arange(dimension) / dimension).pow(cfg.gamma)
        torch.manual_seed(31)
        cls = STAIR5_v8_256D_Model if specialized else STAIR5_v8_Model
        return cls(self.dataset, cfg)

    def batch(self, model):
        return {model.User: torch.tensor([0, 0, 1, 2]),
                model.Item: torch.tensor([0, 1, 1, 3]),
                model.INeg: torch.tensor([4, 4, 5, 6])}

    def test_same_256d_forward_gradient_and_two_step_optimizer_parity(self):
        reference, capacity = self.model(False), self.model()
        optimizers = [AdamWSEvo(model.marked_params(), lr=0.001, weight_decay=0.1)
                      for model in (reference, capacity)]
        for _ in range(2):
            losses = []
            for model, optimizer in zip((reference, capacity), optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss, bpr, cl = model.training_objective(self.batch(model))
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(loss, bpr + 0.01 * cl)
                loss.backward()
                losses.append(loss.detach())
            torch.testing.assert_close(*losses)
            for left, right in zip(reference.parameters(), capacity.parameters()):
                torch.testing.assert_close(left.grad, right.grad)
                self.assertTrue(torch.isfinite(right.grad).all())
            for optimizer in optimizers:
                optimizer.step()
            for left, right in zip(reference.parameters(), capacity.parameters()):
                torch.testing.assert_close(left, right)

    def test_dimension_does_not_change_raw_semantic_or_final_graph(self):
        reference64, capacity = self.model(False, 64), self.model()
        self.assertEqual(reference64.graph_fingerprint, capacity.graph_fingerprint)
        self.assertEqual([x["candidate_sha256"] for x in reference64.knn_metadata],
                         [x["candidate_sha256"] for x in capacity.knn_metadata])
        self.assertEqual(capacity.User.embeddings.weight.shape, (20, 256))
        self.assertEqual(capacity.Item.embeddings.weight.shape, (270, 256))
        groups = capacity.marked_params()
        self.assertEqual(len(groups), 2)
        self.assertIsNone(groups[0]["smoother"])
        self.assertIsNotNone(groups[1]["smoother"])

    def test_checkpoint_roundtrip_continuation_and_64d_rejection(self):
        model = self.model()
        optimizer = AdamWSEvo(model.marked_params(), lr=0.001, weight_decay=0.1)
        model.fit(self.batch(model)).backward()
        optimizer.step()
        path = self.path / "capacity.pt"
        save_training_checkpoint(path, model, optimizer, 1)
        restored = self.model()
        restored_optimizer = AdamWSEvo(restored.marked_params(), lr=0.001, weight_decay=0.1)
        payload = load_training_checkpoint(path, restored, restored_optimizer)
        self.assertEqual(payload["epoch"], 1)
        for current, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
            current_optimizer.zero_grad(set_to_none=True)
            current.fit(self.batch(current)).backward()
            current_optimizer.step()
        for left, right in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right)
        reference64 = self.model(False, 64)
        before = reference64.Item.embeddings.weight.detach().clone()
        with self.assertRaises(ValueError):
            reference64.load_state_dict(model.state_dict())
        torch.testing.assert_close(before, reference64.Item.embeddings.weight)

    def test_inherited_full_and_pool_scoring(self):
        model = self.model()
        model.eval()
        model.reset_ranking_buffers()
        users = torch.tensor([0, 1])
        full = model.recommend_from_full({model.User: users})
        pool_ids = torch.tensor([[2, 5], [4, 7]])
        pool = model.recommend_from_pool({model.User: users, model.IUnseen: pool_ids})
        torch.testing.assert_close(pool, full.gather(1, pool_ids))

    def test_stale_64d_or_modified_coordinate_schedule_is_rejected(self):
        for coefficients in (self.cfg.beta3[:64], self.cfg.beta3 + 0.001):
            cfg = copy.copy(self.cfg)
            cfg.beta3 = coefficients
            with self.assertRaises(ValueError):
                STAIR5_v8_256D_Model(self.dataset, cfg)

    def test_cli_compiles_all_three_256d_configs(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("Baby", "Sports", "Electronics"):
            config = root / "configs" / f"Amazon2014{dataset}_STAIR5_v8_256D.yaml"
            cfg = build_config(["--config", str(config), "--device", "cpu",
                                "--artifact-dir", str(self.path / dataset)])
            self.assertEqual(cfg.embedding_dim, 256)
            self.assertEqual(cfg.beta3.shape, (256,))


if __name__ == "__main__":
    unittest.main()
