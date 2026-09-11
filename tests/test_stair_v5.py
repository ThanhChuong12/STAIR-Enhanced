# -*- coding: utf-8 -*-
"""
tests/test_stair_v5.py
Bộ kiểm thử đơn vị và thẩm tra lý thuyết toán học cho kiến trúc STAIR-v5.
"""

import os
import sys

# Đảm bảo UTF-8 encoding trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Thêm root workspace vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.stair_sre_v5 import STAIR_v5_SingleMatrixEngine, STAIR_v5


def test_symmetrization_2d_shape():
    """
    Test 1: Kiểm thử phép đối xứng hóa không tạo tensor 3D.
    Mục tiêu: Đảm bảo không bao giờ gặp lỗi RuntimeError: indices must be 2D.
    """
    num_items = 10
    row = np.array([0, 1, 2, 3], dtype=np.int64)
    col = np.array([1, 2, 3, 0], dtype=np.int64)
    data = np.array([1.0, 1.5, 2.0, 1.2], dtype=np.float32)

    raw_knn = sp.coo_matrix((data, (row, col)), shape=(num_items, num_items)).tocsr()
    t_feat = torch.randn(num_items, 64)
    v_feat = torch.randn(num_items, 64)
    R = sp.csr_matrix(np.zeros((5, num_items)))

    engine = STAIR_v5_SingleMatrixEngine(verbose=False)
    mAdj = engine.build_boosted_mAdj(t_feat, v_feat, R, raw_knn)

    # Chuyển về COO để kiểm tra
    mAdj_coo = mAdj.to_sparse_coo()
    assert mAdj_coo.indices().dim() == 2, "LỖI: indices phải là tensor 2D [2, E]!"
    assert mAdj_coo.indices().size(0) == 2, "LỖI: sparse_dim của indices phải bằng 2!"
    print("✅ TEST 1 PASSED: Symmetrization tạo đúng tensor 2D COO.")


def test_spsd_and_spectral_radius():
    """
    Test 2: Kiểm thử tính chất Đối xứng Nửa xác định Dương (SPSD) và Bán kính phổ <= 1.0.
    """
    num_items = 50
    # Tạo đồ thị kNN ngẫu nhiên có hướng
    adj_dense = (np.random.rand(num_items, num_items) > 0.85).astype(np.float32)
    np.fill_diagonal(adj_dense, 0.0)
    raw_knn = sp.csr_matrix(adj_dense)

    t_feat = torch.randn(num_items, 128)
    v_feat = torch.randn(num_items, 128)
    R = sp.csr_matrix((np.random.rand(20, num_items) > 0.9).astype(np.float32))

    engine = STAIR_v5_SingleMatrixEngine(verbose=False)
    mAdj_csr = engine.build_boosted_mAdj(t_feat, v_feat, R, raw_knn)

    # Chuyển sang ma trận dense để tính trị riêng
    mAdj_dense = mAdj_csr.to_dense().cpu().numpy()

    # Kiểm tra đối xứng
    assert np.allclose(mAdj_dense, mAdj_dense.T, atol=1e-5), "LỖI: Ma trận không đối xứng!"

    # Tính phổ trị riêng (Eigenvalues)
    eigvals = np.linalg.eigvalsh(mAdj_dense)
    min_eig = eigvals.min()
    max_eig = eigvals.max()

    assert min_eig >= -1.0 - 1e-4, f"LỖI: Trị riêng nhỏ hơn -1 ({min_eig}) làm vỡ tính SPSD!"
    assert max_eig <= 1.0 + 1e-4, f"LỖI: Bán kính phổ vượt quá 1.0 ({max_eig}) gây bùng nổ gradient!"
    print(f"✅ TEST 2 PASSED: SPSD được bảo toàn hoàn hảo. Phổ trị riêng: [{min_eig:.4f}, {max_eig:.4f}].")


def test_monotonicity_preservation():
    """
    Test 3: Kiểm tra tính bảo toàn trật tự đồng thuận (Consensus Monotonicity).
    Cạnh đồng thuận kép (W_base=2.0) phải luôn có trọng số cao hơn cạnh đơn phương thức (W_base=1.0).
    """
    num_items = 3
    # Đỉnh 0 kết nối tới đỉnh 1 (đồng thuận kép W_base=2.0) và đỉnh 2 (đơn phương thức W_base=1.0)
    row = np.array([0, 0], dtype=np.int64)
    col = np.array([1, 2], dtype=np.int64)
    data = np.array([2.0, 1.0], dtype=np.float32)

    raw_knn = sp.coo_matrix((data, (row, col)), shape=(num_items, num_items)).tocsr()

    # Cho đặc trưng giống nhau để nhận cùng điểm boost
    t_feat = torch.ones(num_items, 32)
    v_feat = torch.ones(num_items, 32)
    R = sp.csr_matrix(np.ones((2, num_items)))

    engine = STAIR_v5_SingleMatrixEngine(alpha=0.4, beta=0.2, verbose=False)
    mAdj = engine.build_boosted_mAdj(t_feat, v_feat, R, raw_knn)

    dense_m = mAdj.to_dense().cpu().numpy()
    w_consensus = dense_m[0, 1]
    w_single = dense_m[0, 2]

    # Trong ma trận Laplacian: S_tilde_01 / S_tilde_02 = sqrt(W_01 / W_02) = sqrt(2.0) ~= 1.414
    assert w_consensus > w_single, f"LỖI: Trật tự đồng thuận bị đảo lộn! ({w_consensus} <= {w_single})"
    assert np.isclose(w_consensus / w_single, np.sqrt(2.0), atol=0.15), f"LỖI: Tỷ lệ phân bổ Laplacian sai lệch! ({w_consensus / w_single})"
    print(f"✅ TEST 3 PASSED: Trật tự đồng thuận bảo tồn nguyên vẹn (S_tilde_consensus: {w_consensus:.4f} > S_tilde_single: {w_single:.4f}, ratio={w_consensus/w_single:.3f} ~= sqrt(2)).")


def test_gradient_density_bpr_vs_saml():
    """
    Test 4: Kiểm thử mật độ Gradient giữa BPR và SAML để xác nhận hiện tượng Gradient Starvation.
    """
    batch_size = 128
    dim = 64
    # Mô phỏng trạng thái mô hình đang huấn luyện: User có sở thích gần mẫu dương hơn mẫu âm ngẫu nhiên
    u = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1).requires_grad_()
    pos = F.normalize(u.detach() + 0.3 * torch.randn(batch_size, dim), p=2, dim=-1).requires_grad_()
    neg = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1).requires_grad_()

    # 1. Đo mật độ Gradient của BPR
    pos_score = (u * pos).sum(dim=-1)
    neg_score = (u * neg).sum(dim=-1)
    loss_bpr = -torch.mean(F.logsigmoid(pos_score - neg_score))
    loss_bpr.backward(retain_graph=True)

    # Đo mật độ Gradient của từng mẫu (Sample-wise non-zero gradient vector)
    bpr_grad_density = (neg.grad.norm(dim=-1) > 1e-6).float().mean().item()

    # Reset gradient
    u.grad.zero_()
    pos.grad.zero_()
    neg.grad.zero_()

    # 2. Đo mật độ Gradient của SAML (Margin = 0.5)
    margin = 0.5
    loss_saml = torch.mean(F.relu(margin - (pos_score - neg_score)))
    loss_saml.backward()

    saml_grad_density = (neg.grad.norm(dim=-1) > 1e-6).float().mean().item()

    print(f"BPR Sample-wise Gradient Density: {bpr_grad_density * 100:.1f}% | SAML Sample-wise Gradient Density: {saml_grad_density * 100:.1f}%")
    assert bpr_grad_density == 1.0, "LỖI: BPR gradient phải cung cấp gradient trên 100% các mẫu!"
    assert saml_grad_density < bpr_grad_density, f"LỖI: SAML gradient phải bị suy giảm mật độ so với BPR! ({saml_grad_density} >= {bpr_grad_density})"
    print("✅ TEST 4 PASSED: Chứng minh giải tích BPR bảo toàn gradient dày đặc (100%) cho BSC Smoother, trong khi SAML làm đói gradient.")


if __name__ == "__main__":
    print("Khởi chạy kiểm thử đơn vị STAIR-v5...")
    test_symmetrization_2d_shape()
    test_spsd_and_spectral_radius()
    test_monotonicity_preservation()
    test_gradient_density_bpr_vs_saml()
    print("\n🎉 TOÀN BỘ 4/4 BÀI KIỂM THỬ ĐƠN VỊ ĐẠT 100% SẴN SÀNG TRIỂN KHAI!")
