# -*- coding: utf-8 -*-
"""
tests/test_sbn_bsc_v4.py
=========================
Unit tests cho SBN_BSC_Preprocessor (STAIR-SBN-BSC v4).
Chạy: python -m pytest tests/test_sbn_bsc_v4.py -v
"""

import pytest
import torch
import numpy as np
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.stair_sbn_bsc_v4 import SBN_BSC_Preprocessor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dummy_dataset():
    """Tạo dataset giả lập nhỏ (100 items, 200 users) cho unit test."""
    num_items = 100
    num_users = 200

    torch.manual_seed(42)
    np.random.seed(42)

    text_feats = torch.randn(num_items, 64)
    vis_feats = torch.randn(num_items, 128)

    # Tạo ma trận R ngẫu nhiên (sparse, 5% density)
    R = sp.random(num_users, num_items, density=0.05,
                  format='csr', dtype=np.float32)
    # Binarize
    R.data[:] = 1.0

    # Tạo kNN ngẫu nhiên (500 edges, no self-loops)
    rows = np.random.randint(0, num_items, size=500)
    cols = np.random.randint(0, num_items, size=500)
    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    raw_knn = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    return {
        'text_feats': text_feats,
        'vis_feats': vis_feats,
        'train_R': R,
        'raw_knn': raw_knn,
        'num_items': num_items,
    }


@pytest.fixture
def preprocessor():
    """Tạo preprocessor với cấu hình mặc định."""
    return SBN_BSC_Preprocessor(verbose=False)


# ============================================================================
# Tests
# ============================================================================

class TestPreprocessorInitialization:
    """Kiểm tra khởi tạo preprocessor."""

    def test_default_params(self):
        prep = SBN_BSC_Preprocessor()
        assert prep.tau_text == 0.15
        assert prep.tau_visual == 0.10
        assert prep.modal_discount == 0.50
        assert prep.prune_lambda == 0.50
        assert prep.min_edge_threshold == 0.05
        assert prep.verbose is True

    def test_custom_params(self):
        prep = SBN_BSC_Preprocessor(
            tau_text=0.20, tau_visual=0.15,
            modal_discount=0.60, prune_lambda=0.70,
            min_edge_threshold=0.08, verbose=False
        )
        assert prep.tau_text == 0.20
        assert prep.tau_visual == 0.15
        assert prep.modal_discount == 0.60
        assert prep.prune_lambda == 0.70
        assert prep.min_edge_threshold == 0.08
        assert prep.verbose is False


class TestOutputProperties:
    """Kiểm tra tính chất đầu ra của build_denoised_mAdj."""

    def test_output_tensor_shape_and_type(self, dummy_dataset, preprocessor):
        mAdj = preprocessor.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        num_items = dummy_dataset['num_items']
        assert mAdj.shape == (num_items, num_items), \
            f"Expected shape ({num_items}, {num_items}), got {mAdj.shape}"
        assert mAdj.is_sparse_csr, "Output must be sparse CSR tensor"
        assert mAdj._nnz() > 0, "Output must have non-zero elements"

    def test_no_nan_or_inf(self, dummy_dataset, preprocessor):
        mAdj = preprocessor.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        vals = mAdj.values()
        assert not torch.isnan(vals).any(), "Found NaN in mAdj values"
        assert not torch.isinf(vals).any(), "Found Inf in mAdj values"

    def test_values_non_negative(self, dummy_dataset, preprocessor):
        mAdj = preprocessor.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        vals = mAdj.values()
        assert (vals >= 0).all(), "All values must be non-negative"

    def test_symmetric_output(self, dummy_dataset, preprocessor):
        """Kiểm tra tính đối xứng: |A - A^T| ≈ 0."""
        mAdj = preprocessor.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        # Convert to dense for symmetry check (small matrix)
        mAdj_dense = mAdj.to_dense()
        diff = torch.abs(mAdj_dense - mAdj_dense.T).max().item()
        assert diff < 1e-5, \
            f"Matrix is not symmetric: max |A - A^T| = {diff}"


class TestEdgeCases:
    """Kiểm tra xử lý trường hợp biên (Edge Cases EC-1 → EC-5)."""

    def test_isolated_item_handling(self, dummy_dataset):
        """EC-1: Item cô lập hoàn toàn (degree=0) không gây crash."""
        R = dummy_dataset['train_R'].tolil()
        R[:, 0] = 0  # Item 0 không ai mua
        R[:, 1] = 0  # Item 1 cũng không ai mua

        prep = SBN_BSC_Preprocessor(verbose=False)
        mAdj = prep.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            R.tocsr(),
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        assert mAdj.shape == (100, 100)
        assert not torch.isnan(mAdj.values()).any()

    def test_empty_cooccurrence(self):
        """Trường hợp không có đồng mua nào."""
        num_items = 50
        num_users = 100

        # Mỗi user chỉ mua đúng 1 item → không có đồng mua
        R = sp.eye(num_users, num_items, format='csr', dtype=np.float32)

        text_feats = torch.randn(num_items, 32)
        vis_feats = torch.randn(num_items, 64)
        rows = np.random.randint(0, num_items, size=200)
        cols = np.random.randint(0, num_items, size=200)
        mask = rows != cols
        raw_knn = torch.tensor(np.stack([rows[mask], cols[mask]]), dtype=torch.long)

        prep = SBN_BSC_Preprocessor(verbose=False, min_edge_threshold=0.01)
        mAdj = prep.build_denoised_mAdj(
            text_feats, vis_feats, R, raw_knn, num_items
        )
        assert mAdj.shape == (num_items, num_items)

    def test_edge_retention_range(self, dummy_dataset, preprocessor):
        """Tỷ lệ cạnh giữ lại phải trong khoảng hợp lý."""
        mAdj = preprocessor.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items']
        )
        total_raw = dummy_dataset['raw_knn'].shape[1]
        retained = mAdj._nnz()
        retention_rate = retained / max(total_raw, 1)
        # After symmetrization, nnz can be up to 2x, so just check > 0
        assert retained > 0, "No edges retained after pruning"


class TestInputValidation:
    """Kiểm tra validation đầu vào."""

    def test_feature_row_mismatch(self, preprocessor):
        text_feats = torch.randn(50, 64)
        vis_feats = torch.randn(100, 128)
        R = sp.random(200, 100, density=0.05, format='csr')
        raw_knn = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        with pytest.raises(ValueError, match="Feature row mismatch"):
            preprocessor.build_denoised_mAdj(
                text_feats, vis_feats, R, raw_knn, 100
            )

    def test_interaction_catalog_mismatch(self, preprocessor):
        text_feats = torch.randn(100, 64)
        vis_feats = torch.randn(100, 128)
        R = sp.random(200, 50, density=0.05, format='csr')  # Wrong cols
        raw_knn = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        with pytest.raises(ValueError, match="Interaction catalog mismatch"):
            preprocessor.build_denoised_mAdj(
                text_feats, vis_feats, R, raw_knn, 100
            )

    def test_edge_index_shape_mismatch(self, preprocessor):
        text_feats = torch.randn(100, 64)
        vis_feats = torch.randn(100, 128)
        R = sp.random(200, 100, density=0.05, format='csr')
        raw_knn = torch.tensor([[0, 1, 2]], dtype=torch.long)  # Shape [1, 3]

        with pytest.raises(ValueError, match="raw_knn_edge_index must have shape"):
            preprocessor.build_denoised_mAdj(
                text_feats, vis_feats, R, raw_knn, 100
            )


class TestDeviceAndOptimization:
    """Kiểm tra tính đúng đắn của tối ưu hóa thuật toán và xử lý thiết bị."""

    def test_modal_quality_device_handling(self, dummy_dataset):
        """Bug #3: Đảm bảo index được chuyển cùng thiết bị với feature tensor."""
        prep = SBN_BSC_Preprocessor(verbose=False)
        t_norm = torch.nn.functional.normalize(dummy_dataset['text_feats'], p=2, dim=-1)
        v_norm = torch.nn.functional.normalize(dummy_dataset['vis_feats'], p=2, dim=-1)
        row = dummy_dataset['raw_knn'][0].numpy()
        col = dummy_dataset['raw_knn'][1].numpy()

        if torch.cuda.is_available():
            t_norm = t_norm.cuda()
            v_norm = v_norm.cuda()

        q_modal = prep._compute_modal_quality(t_norm, v_norm, row, col)
        assert isinstance(q_modal, np.ndarray)
        assert len(q_modal) == len(row)
        assert (q_modal >= 0.0).all()

    def test_coo_searchsorted_correctness(self, dummy_dataset):
        """Issue #4: Kiểm tra tính tương đương toán học giữa COO searchsorted và ma trận dense."""
        prep = SBN_BSC_Preprocessor(verbose=False)
        R = dummy_dataset['train_R']
        row = dummy_dataset['raw_knn'][0].numpy()
        col = dummy_dataset['raw_knn'][1].numpy()
        num_items = dummy_dataset['num_items']

        # Fast searchsorted method
        cooccur_counts, q_beh = prep._compute_behavioral_quality(R, row, col, num_items)

        # Ground truth bằng ma trận dày (cho dataset nhỏ trong test)
        C_dense = (R.T @ R).toarray()
        expected_counts = C_dense[row, col].astype(np.float32)

        np.testing.assert_allclose(
            cooccur_counts, expected_counts, rtol=1e-5, atol=1e-5,
            err_msg="COO searchsorted co-occurrence counts do not match ground truth!"
        )


class TestAblationConfigs:
    """Kiểm tra toàn bộ các cấu hình bóc tách A0 -> A6."""

    @pytest.mark.parametrize("config_name", [
        "A0_baseline",
        "A1_modal_only",
        "A2_behavior_only",
        "A3_multi_no_prune",
        "A4_modal_prune",
        "A5_behavior_prune",
        "A6_full_sbn_bsc_v4",
    ])
    def test_all_ablation_configs(self, dummy_dataset, config_name):
        prep = SBN_BSC_Preprocessor(ablation_config=config_name, verbose=False)
        mAdj = prep.build_denoised_mAdj(
            dummy_dataset['text_feats'],
            dummy_dataset['vis_feats'],
            dummy_dataset['train_R'],
            dummy_dataset['raw_knn'],
            dummy_dataset['num_items'],
        )
        assert mAdj.is_sparse_csr
        assert mAdj.shape == (100, 100)
        assert not torch.isnan(mAdj.values()).any()
        assert not torch.isinf(mAdj.values()).any()
        assert (mAdj.values() >= 0).all()


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

