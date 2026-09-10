# -*- coding: utf-8 -*-
"""
tests/test_sbn_bsc_v4_1_ssb.py
================================
Unit tests cho STAIR-BSC-Reweight v4.1-SSB (models/stair_sbn_bsc_v4_1_ssb.py).
Kiểm định tự động:
  1. Bảo tồn 100% tô-pô kNN gốc (0% cắt tỉa, số cạnh sau đối xứng hóa bằng với baseline).
  2. Bậc đỉnh trung bình >= 5.0, không có nút nào có bậc = 0.
  3. Trọng số w_ij strictly in [1.0, 1.8] trong mọi tình huống.
  4. Tính đối xứng nửa xác định dương của Laplacian (W_sym = W_sym^T).
  5. Thao tác 100% thưa (sparse CSR), không có NaN hay Inf.
  6. Các chế độ bóc tách (baseline, modal_only, behavior_only, full_ssb).

Chạy: python -m pytest tests/test_sbn_bsc_v4_1_ssb.py -v
"""

import os
import sys
import pytest
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.stair_sbn_bsc_v4_1_ssb import STAIR_BSC_Reweight_Engine


@pytest.fixture
def synthetic_graph_data():
    """Tạo dữ liệu thử nghiệm giả lập (80 items, 150 users)."""
    torch.manual_seed(42)
    np.random.seed(42)

    num_items = 80
    num_users = 150

    text_feats = torch.randn(num_items, 64)
    vis_feats = torch.randn(num_items, 128)

    # Ma trận tương tác người dùng - sản phẩm (5% density)
    R = sp.random(num_users, num_items, density=0.05, format='csr', dtype=np.float32)
    R.data[:] = 1.0

    # kNN graph (k=5 láng giềng cho mỗi item)
    k = 5
    rows = np.repeat(np.arange(num_items), k)
    cols = []
    for i in range(num_items):
        choices = [j for j in range(num_items) if j != i]
        cols.extend(np.random.choice(choices, size=k, replace=False))
    cols = np.array(cols, dtype=np.int64)

    raw_knn = torch.stack([torch.from_numpy(rows).long(), torch.from_numpy(cols).long()], dim=0)

    return {
        'text_feats': text_feats,
        'vis_feats': vis_feats,
        'train_R': R,
        'raw_knn': raw_knn,
        'num_items': num_items,
    }


def test_initialization():
    """Kiểm tra khởi tạo Engine với các tham số hợp lệ và ngoại lệ khi sai mode."""
    engine = STAIR_BSC_Reweight_Engine(mode="full_ssb", alpha=0.5, beta=0.3)
    assert engine.mode == "full_ssb"
    assert engine.alpha == 0.5
    assert engine.beta == 0.3

    with pytest.raises(ValueError):
        STAIR_BSC_Reweight_Engine(mode="invalid_mode")


def test_topology_preservation_and_zero_pruning(synthetic_graph_data):
    """
    Kiểm định cam kết 1 & 2:
    - 0% cạnh bị cắt tỉa.
    - Số cạnh sau đối xứng hóa giống hệt baseline.
    - Bậc đỉnh trung bình >= 5.0, 0% nút bị cô lập.
    """
    data = synthetic_graph_data

    # Chạy baseline
    engine_base = STAIR_BSC_Reweight_Engine(mode="baseline")
    mAdj_base = engine_base.build_boosted_mAdj(
        data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
    )
    base_stats = engine_base.get_stats()

    # Chạy full_ssb
    engine_ssb = STAIR_BSC_Reweight_Engine(mode="full_ssb", alpha=0.5, beta=0.3)
    mAdj_ssb = engine_ssb.build_boosted_mAdj(
        data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
    )
    ssb_stats = engine_ssb.get_stats()

    # Khẳng định số cạnh hoàn toàn được bảo tồn (0% pruning)
    assert mAdj_base._nnz() == mAdj_ssb._nnz(), "Lỗi: Số cạnh của v4.1-SSB bị lệch so với baseline!"
    assert ssb_stats['deg_min'] >= 5.0, f"Lỗi: Có nút có bậc < 5 (deg_min={ssb_stats['deg_min']})!"


def test_weight_bounds(synthetic_graph_data):
    """
    Kiểm định cam kết 3:
    - Trọng số w_ij strictly in [1.0, 1.8].
    """
    data = synthetic_graph_data
    engine = STAIR_BSC_Reweight_Engine(mode="full_ssb", alpha=0.5, beta=0.3, min_weight=1.0, max_weight=1.8)
    engine.build_boosted_mAdj(
        data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
    )
    stats = engine.get_stats()

    assert stats['w_min'] >= 0.9999, f"w_min = {stats['w_min']} < 1.0"
    assert stats['w_max'] <= 1.8001, f"w_max = {stats['w_max']} > 1.8"
    assert stats['w_mean'] >= 1.0, f"w_mean = {stats['w_mean']} < 1.0"


def test_laplacian_symmetry(synthetic_graph_data):
    """
    Kiểm định cam kết 4:
    - Ma trận Laplacian sau cùng phải đối xứng: A_sym = A_sym^T.
    """
    data = synthetic_graph_data
    engine = STAIR_BSC_Reweight_Engine(mode="full_ssb", alpha=0.5, beta=0.3)
    mAdj = engine.build_boosted_mAdj(
        data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
    )

    mAdj_coo = mAdj.to_sparse_coo()
    r = mAdj_coo.indices()[0].numpy()
    c = mAdj_coo.indices()[1].numpy()
    d = mAdj_coo.values().numpy()

    csr = sp.coo_matrix((d, (r, c)), shape=(data['num_items'], data['num_items'])).tocsr()
    diff = np.abs((csr - csr.T).data)
    max_asym = diff.max() if len(diff) > 0 else 0.0

    assert max_asym < 1e-5, f"Ma trận Laplacian không đối xứng! Lệch max = {max_asym}"


def test_no_nan_inf(synthetic_graph_data):
    """
    Kiểm định cam kết 5:
    - Tensor đầu ra không chứa bất kỳ giá trị NaN hoặc Inf nào.
    """
    data = synthetic_graph_data
    engine = STAIR_BSC_Reweight_Engine(mode="full_ssb", alpha=0.5, beta=0.3)
    mAdj = engine.build_boosted_mAdj(
        data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
    )

    vals = mAdj.values()
    assert not torch.isnan(vals).any(), "Phát hiện NaN trong mAdj!"
    assert not torch.isinf(vals).any(), "Phát hiện Inf trong mAdj!"
    assert mAdj.is_sparse_csr, "mAdj không ở định dạng torch.sparse_csr!"


def test_ablation_modes(synthetic_graph_data):
    """
    Kiểm định cả 4 chế độ bóc tách:
      - baseline
      - modal_only
      - behavior_only
      - full_ssb
    """
    data = synthetic_graph_data

    for mode in ["baseline", "modal_only", "behavior_only", "full_ssb"]:
        engine = STAIR_BSC_Reweight_Engine(mode=mode, alpha=0.5, beta=0.3)
        mAdj = engine.build_boosted_mAdj(
            data['text_feats'], data['vis_feats'], data['train_R'], data['raw_knn'], data['num_items']
        )
        assert mAdj._nnz() > 0
        stats = engine.get_stats()
        if mode == "baseline":
            assert abs(stats['w_max'] - 1.0) < 1e-4
        elif mode == "modal_only":
            assert stats['w_max'] > 1.0
        elif mode == "behavior_only":
            assert stats['w_max'] >= 1.0
        elif mode == "full_ssb":
            assert stats['w_max'] > 1.0
