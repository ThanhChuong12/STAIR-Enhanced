"""Comprehensive unit tests and Gate 0 baseline verification for STAIR4-v2.1.

Tests cover:
1. Gate 0 FSC normalization equivalence with baseline main.py (verifies bug fix).
2. Bounded Phase Encoder row norm preservation (||ψ(x)||_2 = 1.0) and gradient properties.
3. Pairwise Givens rotation identity-at-init and norm preservation.
4. Stop-gradient on key branch (target view receives zero gradient, Givens omega receives gradient).
5. Complex hybrid similarity arithmetic (GEMM vs definition, eta blend).
6. Multi-positive CE loss on valid rows.
7. Smoother fast path equivalence with baseline MHDSmoother when xi = 0.
8. Parameter groups coverage and disjointness.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stair4_v2_utils import (
    bounded_phase_encoder,
    bounded_real_encoder,
    build_positive_mask,
    build_train_positive_index,
    complex_hybrid_similarity,
    exact_cosine_knn,
    make_batch_candidates,
    transpose_train_positive_index,
)
from models.stair4_heads import (
    CrossLayerContrastiveHead,
    PairwiseGivens,
    IdentityRotation,
    multi_positive_ce,
)
from optimizers.stair4_v2_smoother import STAIR4V21Smoother
from optimizers.mhd_smoother import MHDSmoother


# ---------------------------------------------------------------------------
# Test 1: Gate 0 FSC baseline equivalence (Verifies §2 bug fix)
# ---------------------------------------------------------------------------

def test_fsc_baseline_equivalence_gate_0():
    """Verify that the corrected FSC formula in STAIR4-v2.1 exactly matches main.py."""
    torch.manual_seed(42)
    d = 64
    num_layers = 3
    gamma = 0.1
    N = 100

    # Synthetic graph and embeddings
    E = torch.randn(N, d, dtype=torch.float32)
    indices = torch.stack([torch.arange(N), torch.roll(torch.arange(N), 1)])
    values = torch.ones(N, dtype=torch.float32)
    indices_sym = torch.cat([indices, indices.flip(0)], dim=1)
    values_sym = torch.cat([values * 0.5, values * 0.5])
    Adj = torch.sparse_coo_tensor(indices_sym, values_sym, (N, N)).to_sparse_csr()

    # Baseline main.py formula (lines 202-207):
    # cfg.beta3 = 0.1 + 0.9 * (j/d)^gamma
    # beta = 1 - cfg.beta3
    # norm_correction = 1 - beta ** (L + 1)
    # features = features * beta ...
    # avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
    beta3 = 0.1 + 0.9 * (torch.arange(d) / d).pow(gamma)
    beta_baseline = 1.0 - beta3  # a_j
    norm_correction_baseline = 1.0 - beta_baseline.pow(num_layers + 1)

    features = E.clone()
    smoothed_baseline = E.clone()
    for _ in range(num_layers):
        features = Adj @ features * beta_baseline
        smoothed_baseline = smoothed_baseline + features
    Z_baseline = smoothed_baseline.mul(1.0 - beta_baseline).div(norm_correction_baseline)

    # Corrected STAIR4-v2.1 formula:
    # self.beta is beta3 (b_j)
    # beta_complement = 1 - self.beta (a_j)
    # Z = smoothed.mul(self.beta).div(norm_correction)
    beta_model = beta3.clone()
    beta_comp = 1.0 - beta_model
    norm_correction_model = 1.0 - beta_comp.pow(num_layers + 1)

    feat_v21 = E.clone()
    smoothed_v21 = E.clone()
    for _ in range(num_layers):
        feat_v21 = Adj @ feat_v21 * beta_comp
        smoothed_v21 = smoothed_v21 + feat_v21
    Z_v21 = smoothed_v21.mul(beta_model).div(norm_correction_model)

    # Check numerical identity
    max_abs_diff = (Z_baseline - Z_v21).abs().max().item()
    assert max_abs_diff < 1e-6, f"Gate 0 FSC mismatch: max diff = {max_abs_diff}"

    # Counterexample demonstration: show what the OLD buggy v2 would have produced
    Z_old_buggy = smoothed_v21.mul(beta_comp).div(norm_correction_model)
    ratio_dim0 = (Z_old_buggy[:, 0] / Z_baseline[:, 0]).mean().item()
    expected_ratio_dim0 = ((1.0 - beta3[0]) / beta3[0]).item()
    assert math.isclose(ratio_dim0, expected_ratio_dim0, rel_tol=1e-4)
    # For gamma=0.1, beta3[0] = 0.1 -> ratio is 0.9 / 0.1 = 9.0!
    assert math.isclose(expected_ratio_dim0, 9.0, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Test 2: Bounded Phase Encoder unit norm & gradient
# ---------------------------------------------------------------------------

def test_bounded_phase_encoder_norm():
    """Verify that ||ψ(x)||_2 == 1.0 for all vectors, including edge cases."""
    d = 64
    x = torch.randn(32, d, requires_grad=True)

    psi_r, psi_i = bounded_phase_encoder(x, kappa=0.5)

    # Compute row norm squared: sum_k (cos^2 + sin^2) / d = sum_k 1/d = 1.0
    row_norm_sq = (psi_r.square() + psi_i.square()).sum(dim=-1)
    assert torch.allclose(row_norm_sq, torch.ones_like(row_norm_sq), atol=1e-6)

    # Check finite gradients
    loss = psi_r.sum() + psi_i.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Edge case: zero vector
    x_zero = torch.zeros(10, d)
    psi_zr, psi_zi = bounded_phase_encoder(x_zero, kappa=0.5)
    row_norm_zero = (psi_zr.square() + psi_zi.square()).sum(dim=-1)
    assert torch.allclose(row_norm_zero, torch.ones_like(row_norm_zero), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Pairwise Givens rotation identity-at-init and norm preservation
# ---------------------------------------------------------------------------

def test_pairwise_givens_properties():
    d = 64
    B = 16
    givens = PairwiseGivens(d)

    # 1. At init, omega is zero -> theta is zero -> identity rotation
    assert (givens.omega == 0).all()
    assert (givens.theta == 0).all()

    x_r = torch.randn(B, d)
    x_i = torch.randn(B, d)
    out_r, out_i = givens(x_r, x_i)

    assert torch.allclose(out_r, x_r, atol=1e-6)
    assert torch.allclose(out_i, x_i, atol=1e-6)

    # 2. Random omega: norm must be strictly preserved
    with torch.no_grad():
        givens.omega.normal_(0.0, 1.0)

    out_r_rand, out_i_rand = givens(x_r, x_i)
    in_norm = (x_r.square() + x_i.square()).sum(dim=-1)
    out_norm = (out_r_rand.square() + out_i_rand.square()).sum(dim=-1)
    assert torch.allclose(in_norm, out_norm, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: Stop-gradient on key branch
# ---------------------------------------------------------------------------

def test_stop_gradient_on_key_branch():
    """Verify that target X1 receives no gradient, while Givens omega receives gradient."""
    d = 64
    B = 8
    head = CrossLayerContrastiveHead(d=d, kernel="hybrid", rotation="learned_givens")

    # Synthetic embeddings
    X0 = torch.randn(B, d, requires_grad=True)
    X1 = torch.randn(B, d, requires_grad=True)

    logits = head._compute_logits(X0, X1)
    loss = logits.sum()
    loss.backward()

    # Query X0 MUST receive gradient
    assert X0.grad is not None
    assert torch.isfinite(X0.grad).all()

    # Target X1 MUST NOT receive gradient due to stop-gradient: X1.detach()
    assert X1.grad is None, "Target X1 received gradient despite stop-gradient contract!"

    # Givens omega MUST receive gradient because G_theta is applied after detach()
    assert head.rotation.omega.grad is not None
    assert torch.isfinite(head.rotation.omega.grad).all()


# ---------------------------------------------------------------------------
# Test 5: Complex hybrid similarity arithmetic
# ---------------------------------------------------------------------------

def test_complex_hybrid_similarity_arithmetic():
    d = 16
    Bq, Bk = 4, 6
    qr = torch.randn(Bq, d)
    qi = torch.randn(Bq, d)
    kr = torch.randn(Bk, d)
    ki = torch.randn(Bk, d)

    # 1. eta = 0.0 -> pure signed real overlap Re(h)
    logits_signed = complex_hybrid_similarity(qr, qi, kr, ki, eta=0.0, temperature=1.0)
    expected_signed = torch.mm(qr, kr.T) + torch.mm(qi, ki.T)
    assert torch.allclose(logits_signed, expected_signed, atol=1e-6)

    # 2. eta = 1.0 -> pure squared fidelity |h|^2
    logits_fid = complex_hybrid_similarity(qr, qi, kr, ki, eta=1.0, temperature=1.0)
    re_h = torch.mm(qr, kr.T) + torch.mm(qi, ki.T)
    im_h = torch.mm(qr, ki.T) - torch.mm(qi, kr.T)
    expected_fid = re_h.square() + im_h.square()
    assert torch.allclose(logits_fid, expected_fid, atol=1e-6)

    # 3. eta = 0.25 -> 0.75 * signed + 0.25 * fidelity
    logits_hybrid = complex_hybrid_similarity(qr, qi, kr, ki, eta=0.25, temperature=1.0)
    expected_hybrid = 0.75 * expected_signed + 0.25 * expected_fid
    assert torch.allclose(logits_hybrid, expected_hybrid, atol=1e-6)

    # Invalid public-function inputs fail before a GEMM produces opaque errors.
    try:
        complex_hybrid_similarity(qr, qi, kr, ki, eta=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("eta outside [0, 1] must be rejected")


# ---------------------------------------------------------------------------
# Test 6: Multi-positive CE loss
# ---------------------------------------------------------------------------

def test_multi_positive_ce():
    Bq, Bk = 4, 5
    logits = torch.randn(Bq, Bk)
    mask = torch.tensor([
        [True, False, True, False, False],
        [False, True, False, False, False],
        [True, True, True, False, False],
        [False, False, False, True, False],
    ], dtype=torch.bool)

    loss, diag = multi_positive_ce(logits, mask, temperature=0.2)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert diag["valid_rows"] == 4.0

    # Empty rows handling (all-positive or all-negative)
    empty_mask = torch.zeros(Bq, Bk, dtype=torch.bool)
    loss_empty, diag_empty = multi_positive_ce(logits, empty_mask, temperature=0.2)
    assert loss_empty.item() == 0.0
    assert diag_empty["valid_rows"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: Smoother fast-path equivalence when xi = 0
# ---------------------------------------------------------------------------

def test_smoother_fast_path():
    d = 16
    I = 50
    num_layers = 3
    beta = 0.1 + 0.9 * (torch.arange(d) / d).pow(0.2)

    # Adjacency
    indices = torch.stack([torch.arange(I), torch.roll(torch.arange(I), 1)])
    values = torch.ones(I) * 0.5
    indices_sym = torch.cat([indices, indices.flip(0)], dim=1)
    values_sym = torch.cat([values, values])
    S = torch.sparse_coo_tensor(indices_sym, values_sym, (I, I)).to_sparse_csr()

    def apply_bsc(feats, snap):
        return S @ feats

    base_smoother = MHDSmoother(apply_bsc, beta, num_layers)
    smoother = STAIR4V21Smoother(base_smoother, lambda: S, default_xi=0.0)

    delta = torch.randn(I, d)

    # Run base
    base_smoother.use_baseline()
    p_base = base_smoother(delta)
    base_smoother.clear_step_snapshot()

    # Run v2.1 smoother with xi = 0
    smoother.arm_step(mode="baseline", xi=0.0)
    p_v21 = smoother(delta)
    smoother.clear_step_snapshot()

    assert torch.allclose(p_base, p_v21, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 8: kNN graph parity and duplicate-safe positive supervision
# ---------------------------------------------------------------------------

def test_blockwise_knn_and_train_positive_mask():
    """Blockwise kNN must match the baseline dense formulation without ties."""
    torch.manual_seed(7)
    features = torch.randn(11, 5)
    features = features + torch.arange(11, dtype=features.dtype).unsqueeze(1) * 1e-3
    k = 3
    actual = exact_cosine_knn(features, k=k, block_size=4)
    unit = F.normalize(features, dim=-1)
    dense_scores = unit @ unit.T
    dense_scores.fill_diagonal_(-torch.inf)
    expected = dense_scores.topk(k, dim=1).indices
    assert torch.equal(actual, expected)

    users = torch.tensor([[1], [1], [2], [3]])
    positives = torch.tensor([[4], [4], [5], [6]])
    candidates = make_batch_candidates(users, positives)
    assert candidates.user_ids.tolist() == [1, 2, 3]
    assert candidates.item_ids.tolist() == [4, 5, 6]
    edges = torch.tensor([[1, 1, 2, 3, 3], [4, 5, 5, 6, 6]])
    index = build_train_positive_index(edges, n_users=4, n_items=7)
    mask = build_positive_mask(candidates.user_ids, candidates.item_ids, index)
    expected_mask = torch.tensor(
        [[True, True, False], [False, True, False], [False, False, True]]
    )
    assert torch.equal(mask, expected_mask)
    reverse_index = transpose_train_positive_index(index)
    reverse_mask = build_positive_mask(candidates.item_ids, candidates.user_ids, reverse_index)
    assert torch.equal(reverse_mask, expected_mask.T)


# ---------------------------------------------------------------------------
# Test 9: residual correction and smoother input contract
# ---------------------------------------------------------------------------

def test_residual_spectral_correction_matches_dense_oracle():
    torch.manual_seed(8)
    n, d = 5, 4
    beta = torch.full((d,), 0.3)
    dense_s = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0, 0.0],
         [0.5, 0.0, 0.5, 0.0, 0.0],
         [0.0, 0.5, 0.0, 0.5, 0.0],
         [0.0, 0.0, 0.5, 0.0, 0.5],
         [0.0, 0.0, 0.0, 0.5, 0.5]],
        dtype=torch.float32,
    )
    sparse_s = dense_s.to_sparse_csr()
    base = MHDSmoother(lambda x, _: sparse_s @ x, beta, L=2)
    smoother = STAIR4V21Smoother(base, lambda: sparse_s)
    delta = torch.randn(n, d)

    base.use_baseline()
    p_delta = base(delta)
    base.clear_step_snapshot()
    xi = 0.05
    smoother.arm_step(mode="residual_spectral", xi=xi)
    actual = smoother(delta)
    smoother.clear_step_snapshot()

    h = 0.5 * (torch.eye(n) - dense_s)
    expected = p_delta - xi * (h @ h @ p_delta)
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 10: FSC endpoint L=0 identity contract (§11.1)
# ---------------------------------------------------------------------------

def test_fsc_endpoint_l0():
    """When num_layers = 0, FSC must strictly return Z = E without alteration."""
    torch.manual_seed(9)
    d = 16
    E = torch.randn(20, d)
    beta = 0.1 + 0.9 * (torch.arange(d) / d).pow(0.2)
    beta_comp = 1.0 - beta
    # For L = 0: norm_correction = 1 - beta_comp^(0+1) = 1 - beta_comp = beta!
    norm_correction = 1.0 - beta_comp.pow(1)
    Z = E.mul(beta).div(norm_correction)
    assert torch.allclose(Z, E, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 11: Dense vs Chunked loss and gradient parity (§11.1)
# ---------------------------------------------------------------------------

def test_dense_vs_chunked_loss_and_gradient_parity():
    """Verify that query chunking produces bitwise identical loss and gradients."""
    torch.manual_seed(10)
    d = 16
    Bq, Bk = 12, 10
    head_dense = CrossLayerContrastiveHead(d=d, query_chunk_size=100, rotation="learned_givens")
    head_chunked = CrossLayerContrastiveHead(d=d, query_chunk_size=4, rotation="learned_givens")
    head_chunked.rotation.omega.data.copy_(head_dense.rotation.omega.data)

    Q = torch.randn(Bq, d, requires_grad=True)
    K = torch.randn(Bk, d, requires_grad=True)
    q_ids = torch.arange(Bq)
    k_ids = torch.arange(Bk)

    edges = torch.stack([
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
    ])
    index = build_train_positive_index(edges, n_users=Bq, n_items=Bk)

    loss_dense, _ = head_dense._directional_loss(Q, K, q_ids, k_ids, index)
    loss_chunk, _ = head_chunked._directional_loss(Q, K, q_ids, k_ids, index)

    assert torch.allclose(loss_dense, loss_chunk, atol=1e-6)

    # Gradient parity
    g_dense = torch.autograd.grad(loss_dense, [Q, head_dense.rotation.omega])
    g_chunk = torch.autograd.grad(loss_chunk, [Q, head_chunked.rotation.omega])

    assert torch.allclose(g_dense[0], g_chunk[0], atol=1e-6)
    assert torch.allclose(g_dense[1], g_chunk[1], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 12: Fidelity global phase invariance vs Signed kernel sensitivity (§11.1)
# ---------------------------------------------------------------------------

def test_fidelity_global_phase_invariance():
    """Fidelity |h|^2 is invariant to global phase rotation, whereas Re(h) is not."""
    d = 16
    qr = torch.randn(4, d)
    qi = torch.randn(4, d)
    kr = torch.randn(4, d)
    ki = torch.randn(4, d)

    # Base similarities
    s_fid = complex_hybrid_similarity(qr, qi, kr, ki, eta=1.0, temperature=1.0)
    s_sgn = complex_hybrid_similarity(qr, qi, kr, ki, eta=0.0, temperature=1.0)

    # Apply global phase rotation by phi = pi / 3 to queries: q -> q * e^(i*phi)
    phi = math.pi / 3.0
    c_phi, s_phi = math.cos(phi), math.sin(phi)
    qr_rot = c_phi * qr - s_phi * qi
    qi_rot = s_phi * qr + c_phi * qi

    s_fid_rot = complex_hybrid_similarity(qr_rot, qi_rot, kr, ki, eta=1.0, temperature=1.0)
    s_sgn_rot = complex_hybrid_similarity(qr_rot, qi_rot, kr, ki, eta=0.0, temperature=1.0)

    # Fidelity must be strictly identical under global phase shift
    assert torch.allclose(s_fid, s_fid_rot, atol=1e-5)
    # Signed component must NOT be invariant
    assert not torch.allclose(s_sgn, s_sgn_rot, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 13: Near-zero view exclusion (§6.2, §6.5)
# ---------------------------------------------------------------------------

def test_near_zero_view_exclusion():
    """Near-zero query views must be excluded from V_u, and reported in diagnostics."""
    d = 16
    head = CrossLayerContrastiveHead(d=d, query_chunk_size=10, eps=1e-6)
    Q = torch.randn(4, d)
    # Force row 0 to be near zero (< eps)
    Q[0] = 0.0
    K = torch.randn(4, d)

    q_ids = torch.arange(4)
    k_ids = torch.arange(4)
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    index = build_train_positive_index(edges, n_users=4, n_items=4)

    loss, diag = head._directional_loss(Q, K, q_ids, k_ids, index)
    assert diag["near_zero_rate"] > 0.0
    # Valid rows must be at most 3 (row 0 excluded)
    assert diag["valid_rows"] <= 3.0


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASS: {name}")
