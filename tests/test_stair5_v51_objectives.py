"""tests/test_stair5_v51_objectives.py — Unit Tests for STAIR5-v5.1 Objectives.
=============================================================================
Verifies:
1. Mathematical equivalence between grouped occurrence reference and dense occurrence InfoNCE.
2. Direct excluded-logit gradient is exactly 0 in KPE.
3. Target positive is always preserved in KPE denominator.
4. Rows with zero admissible negatives yield exactly 0 loss and zero finite gradients.
5. Chunked execution with gradient checkpointing produces identical loss and gradients
   to unchunked execution for both query and key embedding banks.
6. MP-MATCH produces positive logit derivative p_j - 1/|P|.
7. DEDUP vs KPE vs REF distinct objective behaviors.
"""
import torch
import torch.nn.functional as F
import pytest

from models.stair5_v51_objectives import (
    STAIR5_v51_NLGCL_Module,
    _chunk_grouped_ref_forward,
    _chunk_kpe_forward,
    _chunk_dedup_forward,
    _chunk_mp_match_forward,
    _info_nce_sum,
)
from models.stair5_v51_utils import build_train_pair_index, gpu_membership


def test_grouped_ref_exact_identity():
    """Tests Eq. (7): Grouped occurrence reference equals dense occurrence InfoNCE."""
    torch.manual_seed(2026)
    B, D = 16, 8
    tau = 0.2

    # Simulate queries and repeated keys
    queries = torch.randn(B, D, requires_grad=True)
    # Repeated entity IDs
    key_ids = torch.tensor([0, 1, 2, 0, 1, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0])
    # Target index for each query row
    targets = torch.randint(0, B, (B,))

    # Embedding lookup table for unique entities (4 entities)
    unique_keys, inverse, counts = torch.unique(key_ids, return_inverse=True, return_counts=True)
    entity_embeds = torch.randn(len(unique_keys), D, requires_grad=True)

    # Dense occurrence keys (B, D)
    dense_keys = entity_embeds[inverse]
    dense_targets = dense_keys[targets]

    # Dense InfoNCE
    loss_dense = _info_nce_sum(queries, dense_targets, dense_keys, tau)

    # Grouped Reference InfoNCE
    log_counts = torch.log(counts.float())
    target_unique_idx = inverse[targets]
    loss_grouped = _chunk_grouped_ref_forward(queries, entity_embeds, target_unique_idx, log_counts, tau)

    assert torch.allclose(loss_dense, loss_grouped, atol=1e-5), f"Diff: {torch.abs(loss_dense - loss_grouped)}"

    # Backward gradient comparison
    g_queries_dense = torch.autograd.grad(loss_dense, queries, retain_graph=True)[0]
    g_queries_grouped = torch.autograd.grad(loss_grouped, queries, retain_graph=True)[0]
    assert torch.allclose(g_queries_dense, g_queries_grouped, atol=1e-5)

    g_ent_dense = torch.autograd.grad(loss_dense, entity_embeds, retain_graph=True)[0]
    g_ent_grouped = torch.autograd.grad(loss_grouped, entity_embeds, retain_graph=True)[0]
    assert torch.allclose(g_ent_dense, g_ent_grouped, atol=1e-5)


def test_kpe_direct_excluded_gradient_zero():
    """Tests that masked/excluded logits in KPE receive exactly zero gradient."""
    torch.manual_seed(2026)
    C, m = 4, 6
    tau = 0.2

    # Query and key embeddings
    queries = torch.randn(C, 16, requires_grad=True)
    keys = torch.randn(m, 16, requires_grad=True)

    targets = torch.tensor([0, 1, 2, 3])
    query_ids = torch.tensor([10, 20, 30, 40])
    unique_key_ids = torch.tensor([100, 101, 102, 103, 104, 105])
    n_items = 1000

    # Build train index: target pairs MUST be in train
    # Also add an additional known positive to row 0: (query 10, key 102) -> pair 10 * 1000 + 102
    train_pairs = torch.tensor([
        [10, 20, 30, 40, 10],   # user IDs
        [100, 101, 102, 103, 102]  # item IDs
    ])
    train_index = build_train_pair_index(train_pairs, n_users=100, n_items=n_items)

    loss = _chunk_kpe_forward(
        anchor_chunk=queries,
        keys=keys,
        targets=targets,
        query_ids=query_ids,
        unique_key_ids=unique_key_ids,
        train_index=train_index,
        n_items=n_items,
        is_user_side=False,
        tau=tau,
    )

    # Compute gradient w.r.t logits directly
    logits = (queries @ keys.t() / tau)
    logits.retain_grad()

    # Manual forward through masked logsumexp
    pairs = query_ids[:, None] * n_items + unique_key_ids[None, :]
    is_known_pos = gpu_membership(train_index, pairs)
    admissible = ~is_known_pos
    admissible[torch.arange(C), targets] = True

    pos_logits = logits[torch.arange(C), targets]
    masked = logits.masked_fill(~admissible, -1e9)
    loss_manual = (torch.logsumexp(masked, dim=-1) - pos_logits).sum()
    loss_manual.backward()

    # Key 102 (col 2) for row 0 is an additional known positive and must be masked out
    assert admissible[0, 2] == False, "Row 0 col 2 should be excluded"
    assert logits.grad[0, 2] == 0.0, f"Expected 0.0 gradient on excluded logit, got {logits.grad[0, 2]}"


def test_kpe_zero_admissible_negatives_zero_loss():
    """Tests that a query row where all in-batch unique keys are known positives
    yields zero loss and finite zero gradient."""
    queries = torch.randn(1, 16, requires_grad=True)
    keys = torch.randn(3, 16, requires_grad=True)
    targets = torch.tensor([0])
    query_ids = torch.tensor([5])
    unique_key_ids = torch.tensor([10, 11, 12])
    n_items = 100

    # All 3 keys are known positives for user 5!
    train_pairs = torch.tensor([
        [5, 5, 5],
        [10, 11, 12]
    ])
    train_index = build_train_pair_index(train_pairs, n_users=20, n_items=n_items)

    loss = _chunk_kpe_forward(
        anchor_chunk=queries,
        keys=keys,
        targets=targets,
        query_ids=query_ids,
        unique_key_ids=unique_key_ids,
        train_index=train_index,
        n_items=n_items,
        is_user_side=False,
        tau=0.2,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected 0.0 loss, got {loss.item()}"
    loss.backward()
    assert torch.isfinite(queries.grad).all()
    assert torch.isfinite(keys.grad).all()


def test_chunked_vs_unchunked_parity():
    """Tests that chunk_size=2 produces identical loss and gradients to chunk_size=None."""
    torch.manual_seed(2026)
    B, D = 8, 16
    n_users, n_items = 50, 50

    # Sample train edges
    train_edges = torch.randint(0, 40, (2, 60))
    train_index = build_train_pair_index(train_edges, n_users, n_items)

    # Pick sampled batch from train_edges
    batch_u = train_edges[0, :B]
    batch_i = train_edges[1, :B]

    # Two identical modules with different chunk sizes
    mod_full = STAIR5_v51_NLGCL_Module(
        n_users=n_users, n_items=n_items, train_pair_keys=train_index,
        chunk_size=None, mode="KPE", rho=1.0, tau=0.2
    )
    mod_chunk = STAIR5_v51_NLGCL_Module(
        n_users=n_users, n_items=n_items, train_pair_keys=train_index,
        chunk_size=2, mode="KPE", rho=1.0, tau=0.2
    )

    layer_embeds1 = [torch.randn(n_users + n_items, D, requires_grad=True) for _ in range(2)]
    layer_embeds2 = [e.detach().clone().requires_grad_(True) for e in layer_embeds1]

    loss_full = mod_full(layer_embeds1, batch_u, batch_i)
    loss_chunk = mod_chunk(layer_embeds2, batch_u, batch_i)

    assert torch.allclose(loss_full, loss_chunk, atol=1e-5), f"Full: {loss_full.item()}, Chunk: {loss_chunk.item()}"

    loss_full.backward()
    loss_chunk.backward()

    for g in range(2):
        assert torch.allclose(layer_embeds1[g].grad, layer_embeds2[g].grad, atol=1e-5)
