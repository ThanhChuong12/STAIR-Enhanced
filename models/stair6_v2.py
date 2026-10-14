"""STAIR6-v2 / AKP-WMSG-CSE: inherit V8, replace only static item BSC graph."""

import copy
from pathlib import Path

import torch

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v8_graph import GraphStateV8
from models.stair5_v4_utils import file_sha256
from models.stair6_v2_graph import build_stair6_v2_graph
from stair6_v2_config import can_enabled, validate_can_config


class STAIR6_v2_Model(STAIR5_v8_Model):
    """Keep MI, FSC, losses, parameter groups and ranking methods unchanged."""

    def __init__(self, dataset, cfg):
        cfg = copy.copy(cfg)
        for name, value in validate_can_config(cfg).items():
            setattr(cfg, name, value)
        expected = 0.1 + 0.9 * (torch.arange(cfg.embedding_dim, dtype=torch.float32) /
                               cfg.embedding_dim).pow(cfg.gamma)
        schedule = getattr(cfg, "beta3", None)
        if (not isinstance(schedule, torch.Tensor) or schedule.dtype != torch.float32
                or not torch.equal(schedule, expected.to(schedule.device))):
            raise ValueError("beta3 must match the native FP32 schedule for this embedding dimension")
        if can_enabled(cfg) and not getattr(cfg, "graph_cache_dir", ""):
            cfg.graph_cache_dir = str(Path(getattr(cfg, "artifact_dir", ".cache")) / "graph_cache")
        super().__init__(dataset, cfg)
        root = Path(__file__).resolve().parents[1]
        for source in (Path(__file__), Path(__file__).with_name("stair6_v2_can.py"),
                       Path(__file__).with_name("stair6_v2_graph.py"),
                       Path(__file__).with_name("stair6_v1_graph.py"),
                       root / "stair6_v2_config.py", root / "main_stair6_v2.py",
                       root / "main_stair6_v1.py"):
            self.source_manifest[source.name] = file_sha256(source)

    @torch.no_grad()
    def get_knn_graph(self, features, k=5, cache_key=None):
        """Observe native CPU inputs, without repeating search or changing RNG."""
        edges = super().get_knn_graph(features, k, cache_key)
        if getattr(self, "_capture_can_inputs", False):
            self._can_features.append(features)
            self._can_candidates.append(edges)
        return edges

    @torch.no_grad()
    def prepare(self, path):
        self._capture_can_inputs = can_enabled(self.cfg)
        self._can_features, self._can_candidates = [], []
        try:
            super().prepare(path)
            self.baseline_graph_fingerprint = self.graph_fingerprint
            state = build_stair6_v2_graph(
                GraphStateV8(self.mAdj, self.graph_metadata), features=self._can_features,
                candidates=self._can_candidates,
                train_edges=(self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
                             if self._capture_can_inputs else None),
                n_users=self.User.count, n_items=self.Item.count, cfg=self.cfg,
                beta=self.beta3.detach().cpu(), num_layers=self.num_layers,
            )
            self.mAdj = state.operator
            self.can_metadata, self.can_contract = state.metadata, state.contract
            self.graph_fingerprint = state.metadata["graph_fingerprint"]
            self.graph_metadata = {**self.graph_metadata, "version": "6.2", "architecture": "AKP-WMSG-CSE",
                                   "baseline_graph_fingerprint": self.baseline_graph_fingerprint,
                                   "graph_fingerprint": self.graph_fingerprint, "can": self.can_metadata}
        finally:
            self._capture_can_inputs = False
            del self._can_features, self._can_candidates

    def get_extra_state(self):
        """Bind resume to settings, exact masks/costs/operators and sources, not telemetry."""
        state = super().get_extra_state()
        state["stair6_can"] = {"architecture": "STAIR6-v2-AKP-WMSG-CSE", "contract": self.can_contract}
        return state
