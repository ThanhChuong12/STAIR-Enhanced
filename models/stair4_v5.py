"""STAIR-RAM: a frozen, encoded STAIR teacher with a trainable modality residual.

FreeRec still owns ranking masks and metrics. Static teacher/profile tensors are
reconstructed from verified artifacts; checkpoints contain the head and identity.
"""
from __future__ import annotations

import models.freerec_compat  # Load the existing environment bridge first.
import freerec
import torch

from .stair4_v5_heads import ResidualHead
from .stair4_v4 import STAIR4V4


class STAIR4V5(freerec.models.GenRecArch):
    def __init__(self, dataset, user_vectors, item_vectors, history, *, eta,
                 residual_dim=32, gate_mode="personalized", identity=None):
        super().__init__(dataset)
        if user_vectors.ndim != 2 or item_vectors.ndim != 2:
            raise ValueError("Teacher vectors must be matrices")
        if (len(user_vectors), len(item_vectors)) != (self.User.count, self.Item.count):
            raise ValueError("Teacher vectors do not match the dataset ID mapping")
        if user_vectors.shape[1] != item_vectors.shape[1]:
            raise ValueError("Teacher user/item dimensions differ")
        if not all(torch.isfinite(x).all() for x in (user_vectors, item_vectors)):
            raise ValueError("Nonfinite teacher vectors")
        self.register_buffer("h_user", user_vectors.detach().clone(), persistent=False)
        self.register_buffer("h_item", item_vectors.detach().clone(), persistent=False)
        self.history = history
        self.identity = dict(identity or {})
        self.head = ResidualHead(
            user_vectors.shape[1], {k: x.shape[1] for k, x in history.features.items()},
            residual_dim=residual_dim, eta=eta,
            max_log_degree=float(torch.log1p(history.degrees.float()).max()),
            gate_mode=gate_mode,
        )
        self.residual_enabled = False  # Epoch zero is the exact baseline.
        self.invalidate_ranking_cache()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "history"):
            self.history.to(self.h_user.device)
        self.invalidate_ranking_cache()
        return result

    def get_extra_state(self):
        return {"schema": 1, "identity": self.identity,
                "residual_enabled": self.residual_enabled}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != 1 or state.get("identity") != self.identity:
            raise ValueError("Teacher, feature, architecture or protocol identity differs")
        if not isinstance(state.get("residual_enabled"), bool):
            raise ValueError("Missing residual checkpoint mode")
        self.residual_enabled = state["residual_enabled"]
        self.invalidate_ranking_cache()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.set_extra_state(state_dict.get("_extra_state"))
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.invalidate_ranking_cache()
        return result

    def invalidate_ranking_cache(self):
        self.ranking_buffer = {}

    def sure_trainpipe(self, batch_size):
        # Stage B owns its train-only sampler; this pipe supplies FreeRec fields.
        return self.dataset.train().shuffled_pairs_source().batch_(batch_size).tensor_()

    def score_candidates(self, users, positives, candidates):
        users = users.flatten().long()
        positives = positives.flatten().long()
        candidates = candidates.long()
        if candidates.ndim != 2 or candidates.shape[0] != len(users):
            raise ValueError("Expected candidate matrix [rows, positive-plus-negatives]")
        baseline_users = self.h_user[users]
        baseline_items = self.h_item[candidates]
        if not self.residual_enabled:
            return torch.einsum("BD,BKD->BK", baseline_users, baseline_items)
        profiles = self.history.loo(users, positives)
        # Project each distinct item once, then recover every row/slot. Repeated
        # users still retain their own positive-specific LOO queries.
        unique, inverse = torch.unique(candidates, return_inverse=True)
        queries, weights = self.head.encode_queries(baseline_users, profiles, self.history.degrees[users])
        unique_keys = self.head.encode_items({name: values[unique] for name, values in self.history.features.items()})
        keys = {name: value[inverse] for name, value in unique_keys.items()}
        baseline = torch.einsum("BD,BKD->BK", baseline_users, baseline_items)
        return baseline + self.head.residual_scores(queries, keys, weights)

    @torch.no_grad()
    def reset_ranking_buffers(self):
        users, items = self.h_user, self.h_item
        if self.residual_enabled:
            ids = torch.arange(len(users), device=users.device)
            users, items = self.head.evaluation_vectors(
                users, items, self.history.full(ids), self.history.features,
                self.history.degrees,
            )
        self.ranking_buffer = {self.User: users.detach(), self.Item: items.detach()}

    def recommend_from_full(self, data):
        users = self.ranking_buffer[self.User][data[self.User]]
        return torch.einsum("BKD,ND->BN", users, self.ranking_buffer[self.Item])

    def recommend_from_pool(self, data):
        users = self.ranking_buffer[self.User][data[self.User]]
        items = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum("BKD,BKD->BK", users, items)


class STAIR4V5Continuation(STAIR4V4):
    """Objective-only control; retains trainable STAIR and its original BSC."""

    def score_candidates(self, users, positives, candidates):
        user_vectors, item_vectors = self.encode()
        return torch.einsum("BD,BKD->BK", user_vectors[users.flatten()],
                            item_vectors[candidates])

    def invalidate_ranking_cache(self):
        self.ranking_buffer = {}
