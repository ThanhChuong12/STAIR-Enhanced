"""STAIR5-v2 C-HET: frozen behavioral graph calibration with optional LHC.

The unchanged v1 backbone supplies MI/FSC/BPR and Euclidean recommendation.
V1 historical objectives are not reused by the audited auxiliary branch.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.stair5_v1 import STAIR5_v1_Model
from models.stair5_v2_geometry import AuditedGeometry
from models.stair5_v2_graph import build_calibrated_graph
from models.stair5_v2_objectives import AuditedContrastiveLoss
from optimizers.stair5_v2_smoother import STAIR5V2Smoother


ARMS = ("B0", "ET", "H0-A", "E0-A", "HC-A", "ET-H0", "ET-placebo")
LHC_ARMS = ("H0-A", "E0-A", "HC-A", "ET-H0")
EDGE_ARMS = ("ET", "ET-H0", "ET-placebo")


class STAIR5_v2_Model(STAIR5_v1_Model):
    """A static item-update graph; the user-item FSC graph is never changed."""
    def __init__(self, dataset, cfg):
        effective = copy.copy(cfg)
        arm = getattr(cfg, "v2_arm", "ET")
        if arm not in ARMS:
            raise ValueError(f"Unknown v2 arm: {arm}")
        if cfg.num_layers < 1 and arm in LHC_ARMS:
            raise ValueError("The auxiliary branch requires at least one FSC layer.")
        if not 0 <= getattr(cfg, "warmup_start", 20) < getattr(cfg, "warmup_end", 50):
            raise ValueError("Require 0 <= warmup_start < warmup_end.")
        effective.v2_arm = arm
        effective.lhc_arm = "E0" if arm == "E0-A" else "HC" if arm == "HC-A" else "H0"
        effective.disable_self_return = True
        effective.w_hybrid = 0.
        if arm not in LHC_ARMS:
            effective.lambda_lhc = 0.
        if not math.isfinite(getattr(effective, "lambda_lhc", 3e-4)) or getattr(effective, "lambda_lhc", 3e-4) < 0:
            raise ValueError("lambda_lhc must be nonnegative.")
        super().__init__(dataset, effective)
        self.geo = AuditedGeometry(
            kappa=getattr(cfg, "lhc_kappa", 1.), radius_cap=getattr(cfg, "radius_cap", 2.),
            arm=effective.lhc_arm, eps_taylor=getattr(cfg, "eps_taylor", 1e-4),
        )
        self.lhc_loss_fn = AuditedContrastiveLoss(
            self.geo, radius_cap=getattr(cfg, "radius_cap", 2.), tau=getattr(cfg, "lhc_tau", .3),
            eps_beta=getattr(cfg, "eps_beta", .05),
        )
        self.lhc_loss_fn.chunk_size = getattr(cfg, "lhc_chunk_size", 256)
        if self.lhc_loss_fn.chunk_size <= 0:
            raise ValueError("lhc_chunk_size must be positive.")
        if self.lambda_lhc > 0:
            # Freeze scales at initial MI embeddings, before warmup and any step.
            self.initialize_scale_buffers()

    def _build_train_interaction_table(self, edges):
        edges = edges.detach().cpu().long()
        keys = torch.unique(edges[0]*self.Item.count + edges[1], sorted=True)
        self.register_buffer("train_pair_keys", keys, persistent=False)

    def _build_positive_matrix_vectorized(self, users, items, device):
        """Train-only membership in bounded query blocks; no Python pair loop."""
        users, items = users.to(device), items.to(device)
        keys = self.train_pair_keys.to(device)
        result = torch.zeros((len(users), len(items)), dtype=torch.bool, device=device)
        if not keys.numel():
            return result
        for start in range(0, len(users), 128):
            queries = users[start:start+128, None]*self.Item.count + items[None]
            positions = torch.searchsorted(keys, queries)
            valid = positions < len(keys)
            result[start:start+128] = valid & (keys[positions.clamp_max(len(keys)-1)] == queries)
        return result

    @torch.no_grad()
    def get_knn_graph(self, features, k=5, cache_key=None):
        """Exact all-catalog top-k in bounded row blocks, optionally on GPU.

        GPU/CPU backends may differ for near ties. Use the same preprocessing
        backend/cache for paired arms; no ANN approximation is introduced.
        """
        from models.stair5_v2_utils import atomic_torch_save
        features = torch.as_tensor(features, dtype=torch.float32)
        chunk = getattr(self.cfg, "knn_chunk_size", 1024)
        requested = getattr(self.cfg, "knn_device", "cpu")
        if chunk <= 0:
            raise ValueError("knn_chunk_size must be positive.")
        cuda = requested in ("cuda", "auto") and torch.cuda.is_available()
        if requested == "cuda" and not cuda:
            raise RuntimeError("CUDA preprocessing requested but CUDA is unavailable.")
        device = torch.device(self.cfg.device) if cuda and str(self.cfg.device).startswith("cuda") else torch.device("cuda" if cuda else "cpu")
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            remaining = int(.65*free) - 2*features.numel()*features.element_size()
            if remaining <= 0:
                if requested == "cuda":
                    raise RuntimeError("Insufficient GPU memory for normalized features; use --knn-device cpu.")
                device = torch.device("cpu")
            else:
                chunk = max(1, min(chunk, remaining//(features.shape[0]*4)))
        signature = json.dumps({"feature": cache_key, "k": k,
                                "torch": str(torch.__version__), "device": str(device),
                                "chunk": chunk, "algorithm": "fp32_exact_topk_v1",
                                "tf32": torch.backends.cuda.matmul.allow_tf32}, sort_keys=True)
        cache_root = getattr(self.cfg, "graph_cache_dir", None)
        cache = Path(cache_root)/("knn_"+hashlib.sha256(signature.encode()).hexdigest()+".pt") if cache_root and cache_key else None
        if not hasattr(self, "knn_metadata"):
            self.knn_metadata = []
        self.knn_metadata.append({"device": str(device), "chunk": chunk, "k": k,
                                  "cache_hit": bool(cache and cache.is_file()),
                                  "fingerprint": hashlib.sha256(signature.encode()).hexdigest()})
        if cache and cache.is_file():
            return torch.load(cache, map_location="cpu", weights_only=True)
        normalized = F.normalize(features.to(device), dim=-1)
        rows, cols = [], []
        for start in range(0, len(features), chunk):
            stop = min(start+chunk, len(features))
            similarity = normalized[start:stop] @ normalized.t()
            local = torch.arange(stop-start, device=device)
            similarity[local, start+local] = -10.
            neighbors = similarity.topk(k, dim=1).indices
            rows.append(torch.arange(start, stop, device=device)[:, None].expand(-1, k).reshape(-1).cpu())
            cols.append(neighbors.reshape(-1).cpu())
            del similarity
        result = torch.stack((torch.cat(rows), torch.cat(cols)))
        if cache:
            atomic_torch_save(result, cache)
        return result

    @torch.no_grad()
    def prepare(self, path):
        """Capture raw baseline modality support, then calibrate only BSC."""
        import freerec
        from freerec.utils import import_pickle

        cfg = self.cfg
        if len(cfg.mfiles) != len(cfg.num_neighbors) or not cfg.mfiles:
            raise ValueError("Each modality needs one positive kNN count.")
        features, feature_manifest = [], []
        for filename in cfg.mfiles:
            candidates = (Path(path)/filename, Path(cfg.root)/cfg.dataset/filename,
                          Path(cfg.root)/"Processed"/cfg.dataset/filename)
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing modality feature: {filename} under {path}")
            feature = import_pickle(str(source))
            if len(feature) != self.Item.count:
                raise ValueError(f"Feature row count mismatch: {source}")
            features.append(feature)
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1024*1024), b""):
                    digest.update(block)
            feature_manifest.append({"file": filename, "sha256": digest.hexdigest()})
        if any(k <= 0 or k >= self.Item.count for k in cfg.num_neighbors):
            raise ValueError("kNN counts must be between 1 and item_count-1.")
        candidates = []
        for feature, k, manifest in zip(features, cfg.num_neighbors, feature_manifest):
            candidates.append(self.get_knn_graph(feature, k, cache_key=manifest["sha256"]))
        edge_index = torch.cat(candidates, 1)
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float)
        edge_index, edge_weight = freerec.graph.coalesce(edge_index, edge_weight, reduce="sum")
        edge_index, edge_weight = freerec.graph.to_undirected(edge_index, edge_weight, reduce="max")
        raw = torch.sparse_coo_tensor(edge_index, edge_weight, (self.Item.count, self.Item.count)).coalesce()
        index, weight = freerec.graph.to_normalized(edge_index, edge_weight, normalization="sym")
        baseline = torch.sparse_coo_tensor(index, weight, raw.shape).to_sparse_csr()
        train_edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        active = cfg.v2_arm in EDGE_ARMS
        state = build_calibrated_graph(
            raw, baseline, train_edges.long(), self.User.count,
            strength=getattr(cfg, "edge_strength", .25),
            mix=getattr(cfg, "edge_mix", .25) if active else 0.,
            shrinkage=getattr(cfg, "evidence_shrinkage", 5.),
            placebo=cfg.v2_arm == "ET-placebo", seed=getattr(cfg, "placebo_seed", getattr(cfg, "seed", 1)),
            cache_dir=getattr(cfg, "graph_cache_dir", None) or None,
        )
        self.register_buffer("mAdj", state.operator)
        self.graph_metadata = state.metadata
        self.feature_manifest = feature_manifest
        # Fingerprint applies to B0 as well; prevents resuming on another split.
        digest = hashlib.sha256(json.dumps(feature_manifest, sort_keys=True).encode())
        for tensor in (raw.indices(), raw.values(), self.train_pair_keys):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.data_fingerprint = digest.hexdigest()
        self.graph_metadata["data_fingerprint"] = self.data_fingerprint

        # MI: exact baseline whitening, weighted modality sum, left-normalized R.
        initial = sum(self.whitening(f)*k for f,k in zip(features, cfg.num_neighbors))/sum(cfg.num_neighbors)
        self.Item.embeddings.weight.copy_(initial)
        index_ui, weight_ui = freerec.graph.to_normalized(train_edges, normalization="left")
        interaction = torch.sparse_coo_tensor(index_ui, weight_ui, (self.User.count, self.Item.count)).to_sparse_csr()
        self.User.embeddings.weight.copy_(interaction @ initial)

    def update_epoch(self, epoch):
        self.current_epoch = int(epoch)
        if self.lambda_lhc <= 0:
            self.current_lambda = 0.
        else:
            if not self.is_scale_initialized:
                self.initialize_scale_buffers()
            fraction = min(1., max(0., (epoch-self.warmup_start)/(self.warmup_end-self.warmup_start)))
            self.current_lambda = self.lambda_lhc*fraction

    def marked_params(self):
        groups = [
            {"params": list(self.User.parameters()), "smoother": None},
            {"params": list(self.Item.parameters()),
             "smoother": STAIR5V2Smoother(lambda x: self.mAdj @ x, self.beta3, self.num_layers)},
        ]
        parameters = [p for group in groups for p in group["params"]]
        if len({id(p) for p in parameters}) != len(parameters) or {id(p) for p in parameters} != {
            id(p) for p in self.parameters() if p.requires_grad
        }:
            raise RuntimeError("Each trainable parameter must appear in exactly one optimizer group.")
        return groups

    def training_objective(self, data):
        """Return total, BPR and raw CE for optional one-batch diagnostics."""
        user, item, layers = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        bpr = self.criterion(
            torch.einsum("BKD,BKD->BK", user[users], item[positives]),
            torch.einsum("BKD,BKD->BK", user[users], item[negatives]),
        )
        cl, diagnostics = bpr*0, {"alignment": 0., "uniformity": 0.}
        if self.training and self.current_lambda > 0:
            unique_users, unique_items = torch.unique(users.flatten()), torch.unique(positives.flatten())
            mask = self._build_positive_matrix_vectorized(unique_users, unique_items, user.device)
            u0, i0 = layers[0].split((self.User.count, self.Item.count))
            u1, i1 = layers[1].split((self.User.count, self.Item.count))
            cl, diagnostics = self.lhc_loss_fn(
                u_0=u0[unique_users], i_0=i0[unique_items], u_1=u1[unique_users], i_1=i1[unique_items],
                P=mask, beta=1-self.beta3,
                q_u0=self.q_u0.item(), q_i0=self.q_i0.item(),
                q_u1=self.q_u1.item(), q_i1=self.q_i1.item(), M_norm=self.M_norm.item(),
            )
        self.last_diagnostics = {"bpr_loss": float(bpr.detach()), "cl_loss": float(cl.detach()),
                                 "current_lambda": self.current_lambda, **diagnostics}
        return bpr+self.current_lambda*cl, bpr, cl

    def fit(self, data):
        return self.training_objective(data)[0]

    def get_extra_state(self):
        return {"version": 1, "data_fingerprint": self.data_fingerprint,
                "graph_fingerprint": self.graph_metadata.get("fingerprint"),
                "arm": self.cfg.v2_arm, "epoch": self.current_epoch,
                "config": {key: getattr(self.cfg, key, None) for key in (
                    "embedding_dim", "num_layers", "gamma", "lhc_kappa", "radius_cap", "lhc_tau",
                    "eps_beta", "eps_taylor", "lambda_lhc", "warmup_start", "warmup_end",
                    "lr", "weight_decay", "beta1", "beta2", "seed", "optimizer")}}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        for key in ("version", "data_fingerprint", "graph_fingerprint", "arm", "config"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Checkpoint incompatible with current {key}.")
        self.update_epoch(state["epoch"])
