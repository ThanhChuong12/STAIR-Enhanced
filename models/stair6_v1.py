"""STAIR6-v1: static category-association enrichment of the V8 item smoother.

All representation, loss, ranking and optimizer methods are inherited unchanged.
Category tensors live only during CPU preprocessing; the final CSR replaces S8.
"""

import copy
from pathlib import Path

import torch

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v8_graph import (
    GraphStateV8, build_wmsg_semantic_graph, sparse_fingerprint, tensor_to_scipy,
)
from models.stair5_v4_utils import file_sha256
from models.stair6_v1_categories import load_category_data, assert_category_provenance
from models.stair6_v1_latent_categories import build_latent_category_data
from models.stair6_v1_graph import InactiveCategorySignal, build_stair6_v1_graph, load_native_raw_cf
from stair6_v1_config import CATEGORY_NAMES, category_enabled, validate_category_config


class STAIR6_v1_Model(STAIR5_v8_Model):
    """Change only the fixed item BSC operator, preserving the V8 backbone."""

    def __init__(self, dataset, cfg):
        cfg = copy.copy(cfg)
        for name, value in validate_category_config(cfg, check_files=True).items():
            setattr(cfg, name, value)
        schedule = getattr(cfg, "beta3", None)
        expected = 0.1 + 0.9 * (torch.arange(cfg.embedding_dim, dtype=torch.float32) / cfg.embedding_dim).pow(cfg.gamma)
        if (not isinstance(schedule, torch.Tensor) or schedule.shape != (cfg.embedding_dim,)
                or schedule.dtype != torch.float32 or not torch.equal(schedule, expected.to(schedule.device))):
            raise ValueError("beta3 must match the native FP32 coordinate schedule for this dimension")
        if category_enabled(cfg) and not getattr(cfg, "graph_cache_dir", ""):
            # Native V8 and CAE share this raw CF cache. This prevents a second
            # catalog-wide co-occurrence construction during active preparation.
            cfg.graph_cache_dir = str(Path(getattr(cfg, "artifact_dir", ".cache")) / "graph_cache")
        super().__init__(dataset, cfg)
        root = Path(__file__).resolve().parents[1]
        for source in (
            Path(__file__), Path(__file__).with_name("stair6_v1_categories.py"),
            Path(__file__).with_name("stair6_v1_graph.py"),
            Path(__file__).with_name("stair6_v1_latent_categories.py"),
            root / "stair6_v1_config.py", root / "main_stair6_v1.py",
        ):
            self.source_manifest[source.name] = file_sha256(source)

    @torch.no_grad()
    def get_knn_graph(self, features, k=5, cache_key=None):
        """Observe native preprocessing inputs without changing its result/RNG."""
        edges = super().get_knn_graph(features, k=k, cache_key=cache_key)
        if getattr(self, "_capture_category_inputs", False):
            self._category_raw_features.append(features)
            self._category_knn_edges.append(edges)
        return edges

    @torch.no_grad()
    def prepare(self, path):
        active = category_enabled(self.cfg)
        category_data = None
        if active and self.cfg.category_source == "metadata":
            # Reject structural/provenance failures before the expensive SVD.
            category_data = load_category_data(
                self.cfg.category_mapping_file, self.cfg.category_metadata_file,
                self.cfg.category_manifest_file, self.Item.count,
                expected_provenance={"dataset": self.cfg.dataset},
            )
            if category_data.audit["informative_items"] == 0:
                raise InactiveCategorySignal("Category-enabled training has no informative terminal paths")
        self._capture_category_inputs = active
        self._category_raw_features, self._category_knn_edges = [], []
        try:
            super().prepare(path)
            self.baseline_graph_fingerprint = self.graph_fingerprint
            if not active:
                self.category_metadata = {
                    "requested_mode": self.cfg.category_mode, "effective_mode": "off",
                    "active": False, "strength": float(self.cfg.category_strength),
                    "reason": "explicit_off_or_zero_strength", "graph_fingerprint": self.graph_fingerprint,
                }
                self.category_provenance = None
                return
            expected = {
                "dataset": self.cfg.dataset,
                "features": {record["file"]: record["sha256"] for record in self.feature_manifest},
                "splits": self.split_manifest,
            }
            self.latent_preprocessing = None
            if self.cfg.category_source == "latent":
                latent = build_latent_category_data(
                    self._category_raw_features, dataset=self.cfg.dataset,
                    feature_manifest=self.feature_manifest, split_manifest=self.split_manifest,
                    clusters=self.cfg.latent_clusters, projection_dim=self.cfg.latent_projection_dim,
                    iterations=self.cfg.latent_iterations, chunk_size=self.cfg.latent_chunk_size,
                    seed=self.cfg.latent_seed, memory_budget_mib=self.cfg.latent_memory_budget_mib,
                    cache_dir=self.cfg.graph_cache_dir or None,
                )
                category_data = latent.data
                self.latent_preprocessing = latent.metadata
                del latent
            assert_category_provenance(category_data, expected)
            raw_semantic, _ = build_wmsg_semantic_graph(
                mfeats=self._category_raw_features, knn_edges_list=self._category_knn_edges,
                n_items=self.Item.count, semantic_mode="weighted", edge_power=2.0,
                edge_floor=0.05, relation_gate="off", relation_strength=0.0,
            )
            edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
            raw_cf, _ = load_native_raw_cf(
                train_edges=edges, n_users=self.User.count, n_items=self.Item.count,
                k_cf=self.cfg.k_cf, c_min=self.cfg.c_min, t_shrinkage=self.cfg.t_shrinkage,
                cache_dir=self.cfg.graph_cache_dir or None, block_size=self.cfg.cf_block_size,
                memory_budget_mib=self.cfg.cf_memory_budget_mib,
            )
            state = build_stair6_v1_graph(
                baseline_state=GraphStateV8(self.mAdj, self.graph_metadata),
                raw_semantic=raw_semantic, raw_cf=raw_cf, train_edges=edges,
                n_users=self.User.count, n_items=self.Item.count, category_data=category_data,
                category_mode=self.cfg.category_mode, strength=self.cfg.category_strength,
                min_count=self.cfg.category_min_count, shrinkage=self.cfg.category_shrinkage,
                eta=self.cfg.eta, cache_dir=self.cfg.graph_cache_dir or None,
                memory_budget_mib=self.cfg.category_memory_budget_mib,
                probe_dim=self.cfg.category_probe_dim, probe_seed=self.cfg.category_probe_seed,
                beta=self.beta3.detach().cpu(), num_layers=self.num_layers,
            )
            # Replace, rather than retain both full item operators on GPU.
            self.mAdj = state.operator
            self.graph_fingerprint = sparse_fingerprint(tensor_to_scipy(self.mAdj))
            self.category_metadata = state.metadata
            self.category_metadata["category_source"] = self.cfg.category_source
            self.category_metadata["latent_preprocessing"] = self.latent_preprocessing
            self.category_provenance = {
                "fingerprint": category_data.metadata_fingerprint,
                "audit": category_data.audit, "source": category_data.provenance,
            }
            self.graph_metadata = {
                **self.graph_metadata, "version": "6.1", "architecture": (
                    "LCAE-WMSG-CSE" if self.cfg.category_source == "latent" else "CAE-WMSG-CSE"),
                "baseline_graph_fingerprint": self.baseline_graph_fingerprint,
                "graph_fingerprint": self.graph_fingerprint, "category": state.metadata,
            }
        finally:
            self._capture_category_inputs = False
            del self._category_raw_features, self._category_knn_edges

    def get_extra_state(self):
        """Bind model/checkpoint identity to verified metadata and source revision."""
        state = super().get_extra_state()
        state["stair6_category"] = {
            "architecture": ("STAIR6-v1-LCAE-WMSG-CSE" if self.cfg.category_source == "latent"
                             else "STAIR6-v1-CAE-WMSG-CSE"), "contract_version": 2,
            "config": {key: getattr(self.cfg, key) for key in CATEGORY_NAMES},
            "baseline_graph_fingerprint": self.baseline_graph_fingerprint,
            "metadata": self.category_provenance,
        }
        return state
