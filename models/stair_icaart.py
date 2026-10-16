"""models/stair_icaart.py — PEEC / legacy REEC calibrated-graph architectures.
=============================================================================
Inherits directly from STAIR5_v8_Model (WMSG-core platform).
Overrides ONLY graph preparation to substitute S_8 with S_ICA.

Contracts:
- Preserves exact MI (SVD initialization), FSC (bipartite propagation),
  BPR loss, local NLGCL, dot-product scorer, parameter groups, and seen masking.
- Zero auxiliary consensus queries, zero learned gates, zero new loss functions,
  zero new trainable parameters.
- Optimizer BSC uses native STAIR5V8Smoother directly.
"""

import copy
from pathlib import Path
from typing import Any, Dict

import torch

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v4_graph import tensor_to_scipy
from models.stair5_v4_utils import file_sha256
from models.stair_icaart_graph import build_stair_icaart_graph
from optimizers.stair5_v8_smoother import STAIR5V8Smoother
from stair_icaart_config import evidence_identity, validate_icaart_config


class STAIR_ICAART_Model(STAIR5_v8_Model):
    """Versioned PEEC/REEC graph calibration on the unchanged V8 backbone."""

    def __init__(self, dataset: Any, cfg: Any):
        cfg = copy.copy(cfg)
        for name, value in validate_icaart_config(cfg).items():
            setattr(cfg, name, value)

        # Validate beta3 coordinate schedule against native FP32 reference
        expected = 0.1 + 0.9 * (
            torch.arange(cfg.embedding_dim, dtype=torch.float32) / cfg.embedding_dim
        ).pow(cfg.gamma)
        schedule = getattr(cfg, "beta3", None)
        if (
            not isinstance(schedule, torch.Tensor)
            or schedule.dtype != torch.float32
            or not torch.equal(schedule, expected.to(schedule.device))
        ):
            raise ValueError("beta3 must match the native FP32 schedule for this embedding dimension")

        super().__init__(dataset, cfg)

        # Register source files in provenance manifest
        root = Path(__file__).resolve().parents[1]
        for src in (
            Path(__file__),
            Path(__file__).with_name("stair_icaart_evidence.py"),
            Path(__file__).with_name("stair_icaart_graph.py"),
            root / "stair_icaart_config.py",
            root / "main_stair_icaart.py",
            root / "stair_icaart_reporting.py",
        ):
            if src.is_file():
                self.source_manifest[src.name] = file_sha256(src)

    @torch.no_grad()
    def get_knn_graph(self, features: torch.Tensor, k: int = 5, cache_key: Any = None) -> torch.Tensor:
        """Observe candidate edges and features without altering RNG or repeating retrieval."""
        edges = super().get_knn_graph(features, k, cache_key)
        if getattr(self, "_capture_icaart_inputs", False):
            self._icaart_features.append(features)
            self._icaart_candidates.append(edges)
        return edges

    @torch.no_grad()
    def prepare(self, path: str) -> None:
        """Constructs MI, baseline V8 graph, and overrides mAdj with calibrated S_ICA."""
        self._capture_icaart_inputs = True
        self._icaart_features = []
        self._icaart_candidates = []

        try:
            # The native parent writes its raw-CF cache here. The REEC builder
            # then reuses that exact artifact instead of repeating CF SpGEMM.
            if not getattr(self.cfg, "graph_cache_dir", ""):
                self.cfg.graph_cache_dir = str(
                    Path(getattr(self.cfg, "artifact_dir", "") or path) / "graph_cache"
                )
            Path(self.cfg.graph_cache_dir).mkdir(parents=True, exist_ok=True)
            # 1. Run baseline prepare (MI initialization, train interaction table, baseline S8)
            super().prepare(path)
            baseline_mAdj = self.mAdj
            baseline_fp = self.graph_fingerprint
            baseline_scipy = tensor_to_scipy(baseline_mAdj)

            # 2. Calibrate only the frozen semantic support for the selected evidence mode
            bigraph = self.dataset.train().to_bigraph(edge_type="u2i")
            train_edges = getattr(bigraph, "edge_index", None)
            if not isinstance(train_edges, torch.Tensor):
                # Resolve only the requested user-to-item relation. Never select
                # an arbitrary edge store whose IDs might belong to another graph.
                train_edges = None
                if hasattr(bigraph, "__getitem__"):
                    for key in ("u2i", ("USER", "u2i", "ITEM"),
                                ("user", "u2i", "item"), ("User", "u2i", "Item")):
                        try:
                            candidate = bigraph[key].edge_index
                        except (AttributeError, KeyError, TypeError):
                            continue
                        if isinstance(candidate, torch.Tensor):
                            train_edges = candidate
                            break
                if train_edges is None:
                    raise AttributeError("Cannot resolve the training user-to-item edge relation")
            state = build_stair_icaart_graph(
                mfeats=self._icaart_features,
                knn_edges_list=self._icaart_candidates,
                train_edges=train_edges,
                n_users=self.User.count,
                n_items=self.Item.count,
                cfg=self.cfg,
                V8_operator_cached=baseline_scipy,
            )

            # 3. Replace operator with S_ICA
            self.mAdj = baseline_mAdj if state.metadata["exact_v8_bypass"] else state.operator
            self.graph_fingerprint = state.manifest["graph_fingerprint"]
            self.icaart_metadata = state.metadata
            self.icaart_manifest = state.manifest
            self.icaart_manifest.update(
                features=self.feature_manifest,
                candidate_hashes=[m["candidate_sha256"] for m in self.knn_metadata],
                splits=self.split_manifest,
                training_pair_multiplicity_hash=self.data_fingerprint,
            )
            self.graph_metadata = {
                **self.graph_metadata,
                "version": "ICAART",
                "architecture": self.icaart_manifest["architecture"],
                "baseline_v8_fingerprint": baseline_fp,
                "graph_fingerprint": self.graph_fingerprint,
                "icaart": self.icaart_metadata,
            }
        finally:
            self._capture_icaart_inputs = False
            del self._icaart_features, self._icaart_candidates

    def get_extra_state(self) -> Dict[str, Any]:
        """Binds checkpoint resume to the exact calibrated graph fingerprint and manifest."""
        state = super().get_extra_state()
        state["stair_icaart"] = {
            **evidence_identity(validate_icaart_config(self.cfg)),
            "manifest": getattr(self, "icaart_manifest", {}),
            "graph_fingerprint": self.graph_fingerprint,
        }
        return state

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """Validates graph fingerprint integrity upon checkpoint resume."""
        rec = state.get("stair_icaart") if isinstance(state, dict) else None
        if not isinstance(rec, dict) or rec.get("graph_fingerprint") != self.graph_fingerprint:
            label = "STAIR-PEEC" if self.icaart_manifest["null_scope"] == "full_activity" else "STAIR-REEC"
            raise ValueError(f"{label} graph fingerprint mismatch on resume")
        super().set_extra_state(state)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Reject a changed sparse graph payload before any parameter is copied."""
        from models.stair5_v4_graph import sparse_fingerprint

        graph = state_dict.get("mAdj")
        if (not isinstance(graph, torch.Tensor) or graph.dtype != torch.float32
                or sparse_fingerprint(tensor_to_scipy(graph)) != self.graph_fingerprint):
            label = "STAIR-PEEC" if self.icaart_manifest["null_scope"] == "full_activity" else "STAIR-REEC"
            raise ValueError(f"{label} checkpoint graph payload checksum mismatch")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
