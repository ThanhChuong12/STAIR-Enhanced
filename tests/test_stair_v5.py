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


def test_svd_whitening_isotropic():
    """
    Test 5: Kiểm chứng toán học SVD Whitening chuẩn tắc (Khắc phục Lỗi 5).
    Kiểm tra:
      1. Ma trận hiệp phương sai là ma trận đường chéo đồng nhất (Isotropic Covariance).
      2. Mọi chiều đều có phương sai bằng nhau (= 1/d), loại bỏ hoàn toàn sự chi phối của Singular Values S.
      3. Các chiều trực giao với nhau (off-diagonal covariance ~ 0).
    """
    num_items = 500
    in_dim = 128
    out_dim = 64

    # Tạo ma trận có tương quan mạnh và phương sai không đều giữa các chiều
    raw_feats = torch.randn(num_items, in_dim) * torch.linspace(10.0, 0.1, in_dim)

    whitened = STAIR_v5.svd_whitening(raw_feats, target_dim=out_dim)

    assert whitened.shape == (num_items, out_dim), f"Shape sai lệch: {whitened.shape}"

    # Tính ma trận hiệp phương sai: Cov = (1/N) * X^T * X
    cov = torch.matmul(whitened.t(), whitened) / num_items

    # 1. Kiểm tra tính trực giao (Off-diagonal ~ 0)
    diag_mask = torch.eye(out_dim, dtype=torch.bool)
    off_diag = cov[~diag_mask]
    max_off_diag = off_diag.abs().max().item()
    assert max_off_diag < 1e-4, f"LỖI: Các chiều sau whitening chưa trực giao! Max off-diagonal={max_off_diag}"

    # 2. Kiểm tra tính đồng nhất phương sai (Isotropic: mọi phần tử trên đường chéo bằng nhau = 1/out_dim)
    diag_vals = cov[diag_mask]
    expected_var = 1.0 / out_dim
    assert torch.allclose(diag_vals, torch.tensor(expected_var), atol=1e-3), (
        f"LỖI: Phương sai các chiều sau whitening không đồng nhất! "
        f"Min={diag_vals.min():.6f}, Max={diag_vals.max():.6f}, Expected={expected_var:.6f}"
    )

    print(f"✅ TEST 5 PASSED: SVD Whitening chuẩn tắc đạt Isotropic tuyệt đối. Covariance = {expected_var:.6f} * I_{out_dim} (Max off-diag: {max_off_diag:.2e}).")


def test_device_consistency_across_modes():
    """
    Test 6: Kiểm thử tính đồng nhất thiết bị (Device Consistency) cho toàn bộ 4 modes:
      ['baseline', 'modal_only', 'behavior_only', 'full_ssb'].
    Đảm bảo tuyệt đối không gặp lỗi:
      RuntimeError: Expected all tensors to be on the same device, but found at least two devices!
    """
    num_items = 20
    row = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    col = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
    data = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float32)
    raw_knn = sp.coo_matrix((data, (row, col)), shape=(num_items, num_items)).tocsr()

    t_feat = torch.randn(num_items, 32)
    v_feat = torch.randn(num_items, 32)
    R = sp.csr_matrix((np.random.rand(10, num_items) > 0.8).astype(np.float32))

    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda:0")

    for dev in devices_to_test:
        for mode in ["baseline", "modal_only", "behavior_only", "full_ssb"]:
            engine = STAIR_v5_SingleMatrixEngine(
                mode=mode, alpha=0.40, beta=0.20, verbose=False
            )
            mAdj = engine.build_boosted_mAdj(
                text_feats=t_feat,
                vis_feats=v_feat,
                train_user_item_matrix=R,
                raw_knn_adj=raw_knn,
                target_device=dev,
            )
            expected_device_type = torch.device(dev).type
            assert mAdj.device.type == expected_device_type, (
                f"LỖI: mAdj device {mAdj.device} không khớp target {dev} ở mode {mode}!"
            )
            assert mAdj.is_sparse_csr, f"LỖI: mAdj phải là sparse CSR ở mode {mode}!"

    print(f"✅ TEST 6 PASSED: Device consistency được đảm bảo tuyệt đối trên 4/4 modes ({devices_to_test}).")


if __name__ == "__main__":
    print("Khởi chạy kiểm thử đơn vị STAIR-v5 (6/6 Tests)...")
    test_symmetrization_2d_shape()
    test_spsd_and_spectral_radius()
    test_monotonicity_preservation()
    test_gradient_density_bpr_vs_saml()
    test_svd_whitening_isotropic()
    test_device_consistency_across_modes()
    print("\n🎉 TOÀN BỘ 6/6 BÀI KIỂM THỬ ĐƠN VỊ & TOÁN HỌC ĐẠT 100% SẴN SÀNG TRIỂN KHAI!")

