"""Uniform train-unseen candidate sampling with isolated, resumable RNG."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from models.stair4_v5_features import canonical_train_csr, checked_ids, train_csr_hash


def _ordered_unique_unseen(draws: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Retain first occurrences, preserving the order of an IID uniform stream."""
    _, first = np.unique(draws, return_index=True)
    draws = draws[np.sort(first)]
    if not seen.size:
        return draws
    positions = np.searchsorted(seen, draws)
    matches = (positions < seen.size) & (seen[np.minimum(positions, seen.size - 1)] == draws)
    return draws[~matches]


class UniformTrainUnseenSampler:
    """Sample without replacement per row from catalog minus binary train set.

    The constructor deliberately has no held-out-interaction argument. Duplicate
    users are separate random draws. Returned CPU tensors have shape [B, K];
    masked padding uses item 0 and MUST be ignored by the ranking objective.
    Rows with no unseen items have an all-false mask and must be excluded from
    the effective loss denominator. State restoration requires identical train
    membership. No global NumPy or torch RNG is consumed.
    """

    def __init__(self, train_csr: sp.spmatrix, seed: int = 1):
        self.csr = canonical_train_csr(train_csr)
        self.num_users, self.num_items = self.csr.shape
        self.fingerprint = train_csr_hash(self.csr)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def sample(self, users, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("The number of requested negatives must be a positive integer")
        users = checked_ids(users, self.num_users, "User")
        items = torch.zeros((users.numel(), k), dtype=torch.long)
        mask = torch.zeros_like(items, dtype=torch.bool)
        if not users.numel():
            return items, mask
        # A shared draw reduces Python->torch dispatch for sparse benchmark data.
        proposal_width = max(16, 2 * k)
        proposals = torch.randint(
            self.num_items, (users.numel(), proposal_width), generator=self.generator,
        ).numpy()
        for row, user in enumerate(users.tolist()):
            seen = self.csr.indices[self.csr.indptr[user]:self.csr.indptr[user + 1]]
            unseen_count = self.num_items - seen.size
            target = min(k, unseen_count)
            if not target:
                continue
            if unseen_count <= 4 * k or seen.size > self.num_items // 2:
                # Enumeration avoids unbounded rejection for nearly full users.
                available_mask = np.ones(self.num_items, dtype=bool)
                available_mask[seen] = False
                available = np.flatnonzero(available_mask)
                order = torch.randperm(unseen_count, generator=self.generator)[:target].numpy()
                selected = available[order]
            else:
                selected = _ordered_unique_unseen(proposals[row], seen)[:target]
                while selected.size < target:
                    extra = torch.randint(self.num_items, (proposal_width,), generator=self.generator).numpy()
                    selected = _ordered_unique_unseen(np.concatenate((selected, extra)), seen)[:target]
            items[row, :target] = torch.from_numpy(selected.astype(np.int64, copy=False))
            mask[row, :target] = True
        return items, mask

    def state_dict(self) -> dict:
        return {
            "schema_version": 1, "train_fingerprint": self.fingerprint,
            "rng_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1 or state.get("train_fingerprint") != self.fingerprint:
            raise ValueError("Negative-sampler state does not match this train graph")
        rng = state.get("rng_state")
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("Invalid negative-sampler RNG state")
        self.generator.set_state(rng.cpu())
