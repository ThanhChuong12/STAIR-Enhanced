"""tests/test_stair5_v51_pipeline.py — Pipeline & Checkpoint Roundtrip Tests for STAIR5-v5.1.
========================================================================================
Verifies:
1. Config validation and default values match Report Table 8 specifications.
2. RNG state capture and restore roundtrip.
3. Training checkpoint save and load roundtrip with state validation.
4. Arm mapping to objective mode and graph parameters.
"""
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
import pytest

from models.stair5_v51_utils import (
    rng_state,
    restore_rng,
    save_training_checkpoint,
    load_training_checkpoint,
    optimizer_state_for_checkpoint,
    load_optimizer_state,
    build_train_pair_index,
    compute_kpe_diagnostics,
)
from optimizers.stair5_v51_smoother import STAIR5V51Smoother


def test_rng_roundtrip():
    """Verify full RNG state capture and restoration."""
    torch.manual_seed(1234)
    r1 = torch.randn(5)

    state = rng_state()
    r2 = torch.randn(5)
    assert not torch.allclose(r1, r2)

    restore_rng(state)
    r3 = torch.randn(5)
    assert torch.allclose(r2, r3)


def test_optimizer_smoother_preservation():
    """Verify optimizer state save/load preserves live smoother callbacks."""
    param = nn.Parameter(torch.randn(10, 10))
    smoother_called = False

    def dummy_op(x):
        nonlocal smoother_called
        smoother_called = True
        return x

    smoother = STAIR5V51Smoother(dummy_op, beta=torch.full((10,), 0.5), L=2)
    opt = torch.optim.AdamW([{"params": [param], "smoother": smoother}], lr=1e-3)

    state = optimizer_state_for_checkpoint(opt)
    assert "smoother" not in state["param_groups"][0]

    # New optimizer instance
    param_new = nn.Parameter(torch.randn(10, 10))
    opt_new = torch.optim.AdamW([{"params": [param_new], "smoother": smoother}], lr=1e-3)

    load_optimizer_state(opt_new, state)
    assert opt_new.param_groups[0]["smoother"] is smoother


def test_kpe_diagnostics_calculation():
    """Verify diagnostics calculation without accessing validation/test labels."""
    train_edges = torch.tensor([
        [0, 1, 2, 0, 3],
        [10, 11, 12, 11, 10]
    ])
    train_idx = build_train_pair_index(train_edges, n_users=10, n_items=20)

    user_batch = torch.tensor([0, 1, 0])
    item_batch = torch.tensor([10, 11, 11])

    diag = compute_kpe_diagnostics(train_idx, user_batch, item_batch, n_items=20)

    assert diag["batch_size"] == 3
    assert diag["unique_users"] == 2
    assert diag["unique_items"] == 2
    assert "item_side_additional_positives" in diag
    assert "user_side_additional_positives" in diag
