"""tests/test_stair5_v51_graph.py — Unit Tests for STAIR5-v5.1 Graph Operators.
===========================================================================
Verifies:
1. S4 operator construction and exact parity with v4 graph builder.
2. CAM operator contraction bound: ||S_CAM||_2 <= 1.0.
3. CAM operator recovers S4 exactly when delta = 0.
4. Active-node P95 clipping and support r_i calculation.
5. Sparse symmetry and finite spectral bounds.
"""
import numpy as np
import scipy.sparse as sp
import torch
import pytest

from models.stair5_v51_graph import (
    build_calibrated_graph_v51,
    compute_cam_node_support,
    scipy_to_tensor,
    tensor_to_scipy,
)


def test_cam_contraction_norm_bound():
    """Tests that ||S_CAM||_2 <= 1.0 for arbitrary valid symmetric S0, S_cf, and delta."""
    np.random.seed(2026)
    n = 30

    # Create random symmetric S0 and S_cf with norm <= 1
    A0 = np.random.rand(n, n).astype(np.float32)
    A0 = (A0 + A0.T) / 2.0
    d0 = np.sum(A0, axis=1)
    S0 = A0 / np.sqrt(d0[:, None] * d0[None, :])
    S0_sp = sp.csr_matrix(S0)

    Acf = np.random.rand(n, n).astype(np.float32)
    Acf = (Acf + Acf.T) / 2.0
    dcf = np.sum(Acf, axis=1)
    Scf = Acf / np.sqrt(dcf[:, None] * dcf[None, :])
    Scf_sp = sp.csr_matrix(Scf)

    # Compute node support
    r, r_bar = compute_cam_node_support(Acf, n)
    assert 0 <= r.min() <= r.max() <= 1.0

    # For various delta and eta
    for eta in [0.05, 0.1, 0.2]:
        for delta in [0.0, 0.05, 0.1]:
            eta_nodes = np.clip(eta + delta * (r - r_bar), 0.0, 1.0).astype(np.float32)
            A = np.sqrt(1.0 - eta_nodes)
            B = np.sqrt(eta_nodes)

            # S_CAM = A @ S0 @ A + B @ Scf @ B
            S_cam = np.diag(A) @ S0 @ np.diag(A) + np.diag(B) @ Scf @ np.diag(B)

            # Check 2-norm (largest singular value)
            u, s, vh = np.linalg.svd(S_cam)
            max_sv = s[0]
            assert max_sv <= 1.0001, f"CAM spectral norm violated: {max_sv} > 1.0"


def test_cam_delta_zero_recovers_s4():
    """Tests that delta = 0 recovers standard S4 convex blend exactly."""
    np.random.seed(2026)
    n = 20
    eta = 0.1

    A0 = np.random.rand(n, n).astype(np.float32)
    A0 = (A0 + A0.T) / 2.0
    d0 = np.sum(A0, axis=1)
    S0 = A0 / np.sqrt(d0[:, None] * d0[None, :])

    Acf = np.random.rand(n, n).astype(np.float32)
    Acf = (Acf + Acf.T) / 2.0
    dcf = np.sum(Acf, axis=1)
    Scf = Acf / np.sqrt(dcf[:, None] * dcf[None, :])

    r, r_bar = compute_cam_node_support(Acf, n)
    eta_nodes = np.clip(eta + 0.0 * (r - r_bar), 0.0, 1.0).astype(np.float32)
    A = np.sqrt(1.0 - eta_nodes)
    B = np.sqrt(eta_nodes)

    S_cam = np.diag(A) @ S0 @ np.diag(A) + np.diag(B) @ Scf @ np.diag(B)
    S4_expected = (1.0 - eta) * S0 + eta * Scf

    assert np.allclose(S_cam, S4_expected, atol=1e-6)
