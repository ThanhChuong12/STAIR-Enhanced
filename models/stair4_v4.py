"""STAIR4-CSGC: baseline MI/FSC/BPR with static behavioral BSC calibration."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import models.freerec_compat  # Existing environment bridge; no baseline edits.
import freerec
import torch
from torch import nn

from .stair4_v4_graph import CSGCOptions, build_baseline_raw_graph, build_operator
from .stair4_v4_utils import array_hash, identity_hash
from optimizers.stair4_v4_smoother import STAIR4V4Smoother


class STAIR4V4(freerec.models.GenRecArch):
    def __init__(self, dataset, cfg):
        super().__init__(dataset)
        self.cfg = cfg
        self.options = CSGCOptions.from_config(cfg)
        self.num_layers = int(cfg.num_layers)
        self.embedding_dim = int(cfg.embedding_dim)
        if self.num_layers < 0 or self.embedding_dim < 1 or not math.isfinite(cfg.gamma) or cfg.gamma <= 0:
            raise ValueError("nonnegative layers, positive dimension and finite positive gamma required")
        self.User.add_module("embeddings", nn.Embedding(self.User.count, self.embedding_dim))
        self.Item.add_module("embeddings", nn.Embedding(self.Item.count, self.embedding_dim))
        self.register_buffer("Adj", dataset.train().to_normalized_adj(normalization="sym"), persistent=False)
        self.register_buffer("beta", .1 + .9 * (torch.arange(self.embedding_dim) / self.embedding_dim).pow(cfg.gamma))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.e-4)
        self.prepare(dataset.path)
        self.criterion = freerec.criterions.BPRLoss(reduction="mean")
        self.smoother = STAIR4V4Smoother(lambda x: self.mAdj @ x, self.beta,
                                        self.num_layers, self.options.identity_mix)

    @torch.no_grad()
    def prepare(self, path):
        cfg = self.cfg
        names = cfg.mfiles.split(",") if isinstance(cfg.mfiles, str) else list(cfg.mfiles)
        ks = list(map(int, cfg.num_neighbors.split("-"))) if isinstance(cfg.num_neighbors, str) else list(cfg.num_neighbors)
        if len(names) != len(ks) or not names or min(ks) <= 0:
            raise ValueError("one positive neighbor count per feature file required")
        modalities = []
        for name in names:
            candidates = [Path(path) / name, Path(cfg.root) / cfg.dataset / name]
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f"Missing feature file {name}: searched {candidates}")
            features = torch.as_tensor(freerec.utils.import_pickle(str(resolved)), dtype=torch.float32)
            if features.ndim != 2 or features.shape[0] != self.Item.count or min(features.shape) < self.embedding_dim:
                raise ValueError("feature shape must match item mapping and support the MI dimension")
            modalities.append(features)
        raw, hashes = build_baseline_raw_graph(modalities, ks, self.options.knn_block_size,
                                               getattr(cfg, "graph_cache_dir", None))
        indices, values = freerec.graph.to_normalized(raw.to_sparse_coo().indices(), raw.values(), normalization="sym")
        s0 = torch.sparse_coo_tensor(indices, values, raw.shape).coalesce().to_sparse_csr()
        train_edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index.cpu()
        # Hash explicit mapping files when provided; aligned feature rows and train
        # integer IDs are always included in the cache/checkpoint identity.
        for mapping in sorted(Path(path).glob("*")):
            if mapping.is_file() and any(token in mapping.name.lower() for token in ("mapping", "id2", "2id")):
                hashes[f"mapping:{mapping.name}"] = hashlib.sha256(mapping.read_bytes()).hexdigest()
        hashes["neighbors_config"] = list(ks)
        hashes["knn_backend"] = "torch.topk/cpu/blocked"
        hashes["dimension"] = self.embedding_dim
        hashes["layers"] = self.num_layers
        hashes["gamma"] = float(cfg.gamma)
        hashes["forward_adj"] = array_hash(self.Adj.to_sparse_coo().coalesce().values())
        bundle = build_operator(raw, train_edges.numpy(), self.User.count, self.options,
                                baseline=s0, data_identity=hashes, cache_dir=getattr(cfg, "graph_cache_dir", None))
        self.register_buffer("mAdj", bundle.operator, persistent=False)
        self.graph_identity = bundle.identity
        self.graph_diagnostics = bundle.diagnostics
        self.data_hashes = hashes
        mi = []
        for features, k in zip(modalities, ks):
            centered = features - features.mean(0, keepdim=True)
            u, _, _ = torch.linalg.svd(centered, full_matrices=False)
            mi.append(u[:, :self.embedding_dim] * math.sqrt(self.Item.count / self.embedding_dim) * k)
        mi = sum(mi).div(sum(ks))
        self.Item.embeddings.weight.copy_(mi)
        r_indices, r_values = freerec.graph.to_normalized(train_edges, normalization="left")
        r = torch.sparse_coo_tensor(r_indices, r_values, (self.User.count, self.Item.count)).to_sparse_csr()
        self.User.embeddings.weight.copy_(r @ mi)
        if any(buffer.grad_fn is not None or buffer.requires_grad for buffer in self.buffers()):
            raise RuntimeError("prepare must produce detached static buffers")

    def get_extra_state(self):
        return {"schema": 1, "identity": self.graph_identity}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint architecture, graph, data or source identity differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Fail before changing any parameter, even if the caller disables strict.
        self.set_extra_state(state_dict.get("_extra_state"))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def parameter_groups(self):
        groups = [dict(params=list(self.User.parameters()), smoother=None, role="users"),
                  dict(params=list(self.Item.parameters()), smoother=self.smoother, role="items")]
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters()}:
            raise RuntimeError("Every trainable parameter must appear in exactly one group")
        for group in groups:
            group.update(lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return groups

    def sure_trainpipe(self, batch_size):
        return self.dataset.train().shuffled_pairs_source().gen_train_sampling_neg_(num_negatives=1).batch_(batch_size).tensor_()

    def encode(self):
        features = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        smoothed = features
        beta = 1 - self.beta
        correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        # Preserve baseline arithmetic, including the complement round-trip.
        result = smoothed.mul(1 - beta).div(correction)
        return torch.split(result, (self.User.count, self.Item.count))

    def fit(self, data):
        users, items = self.encode()
        u = users[data[self.User]]
        return self.criterion(torch.einsum("BKD,BKD->BK", u, items[data[self.Item]]),
                              torch.einsum("BKD,BKD->BK", u, items[data[self.INeg]]))

    def reset_ranking_buffers(self):
        users, items = self.encode()
        self.ranking_buffer = {self.User: users.detach().clone(), self.Item: items.detach().clone()}

    def recommend_from_full(self, data):
        users = self.ranking_buffer[self.User][data[self.User]]
        return torch.einsum("BKD,ND->BN", users, self.ranking_buffer[self.Item])

    def recommend_from_pool(self, data):
        users = self.ranking_buffer[self.User][data[self.User]]
        items = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", users, items)
