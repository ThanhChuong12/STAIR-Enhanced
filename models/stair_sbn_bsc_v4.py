# -*- coding: utf-8 -*-
"""
models/stair_sbn_bsc_v4.py
==========================
STAIR-SBN-BSC (v4): Precomputed Structural Denoising for Backward Stepwise Convolution
Synthesized from SIGE (AAAI 2026) and EVEN (AAAI 2025).

Module chính chứa class SBN_BSC_Preprocessor — bộ tiền xử lý đồ thị ngoại tuyến
đánh giá độ tin cậy cạnh kNN dựa trên tín hiệu đa phương thức và hành vi đồng mua
để lọc nhiễu cho BSC Smoother trong AdamWSEvo.

Constraints:
    - Không sử dụng .toarray() trên ma trận thưa
    - Không sử dụng dense matrix multiplication cho large graphs
    - Không thêm learnable params vào graph construction
    - Không thêm auxiliary loss vào training loop
"""

import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class SBN_BSC_Preprocessor:
    """
    Tiền xử lý đồ thị ngoại tuyến (Offline Graph Preprocessor):
    Đánh giá độ tin cậy cạnh kNN dựa trên tín hiệu đa phương thức (EVEN)
    và hành vi đồng mua người dùng (SIGE) để lọc nhiễu cho BSC Smoother.

    Pipeline 7 giai đoạn:
        1. L2 Feature Normalization
        2. Cross-Modal Agreement (Thresholded Geometric Mean)
        3. Behavioral Ochiai Co-occurrence (Fast COO searchsorted lookup)
        4. Joint Quality Combination (Max-Combination)
        5. Adaptive Edge Pruning
        6. Symmetric Laplacian Normalization
        7. Sparse CSR Tensor Conversion

    Thread Safety:
        - build_denoised_mAdj() KHÔNG thiết kế cho đa luồng (Non-thread-safe).
        - Gọi DUY NHẤT 1 LẦN trong prepare() trước epoch 1.
    """

    def __init__(
        self,
        tau_text: float = 0.15,
        tau_visual: float = 0.10,
        modal_discount: float = 0.50,
        prune_lambda: float = 0.50,
        min_edge_threshold: float = 0.05,
        use_modal: bool = True,
        use_behavior: bool = True,
        use_pruning: bool = True,
        ablation_config: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Khởi tạo preprocessor với các siêu tham số lọc nhiễu và cờ bóc tách (ablation).

        Args:
            tau_text: Ngưỡng lọc tương đồng văn bản [0.05, 0.30]. Mặc định: 0.15.
            tau_visual: Ngưỡng lọc tương đồng hình ảnh [0.05, 0.25]. Mặc định: 0.10.
            modal_discount: Hệ số chiết khấu modal rho [0.30, 0.80]. Mặc định: 0.50.
            prune_lambda: Hệ số độ lệch chuẩn cắt tỉa lambda [0.30, 0.90]. Mặc định: 0.50.
            min_edge_threshold: Ngưỡng cắt tối thiểu tau_min [0.01, 0.10]. Mặc định: 0.05.
            use_modal: Bật/tắt thành phần lọc tương đồng đa phương thức (EVEN). Mặc định: True.
            use_behavior: Bật/tắt thành phần lọc hành vi đồng mua (SIGE). Mặc định: True.
            use_pruning: Bật/tắt cắt tỉa cạnh tự thích ứng. Mặc định: True.
            ablation_config: Tên cấu hình thực nghiệm bóc tách (A0_baseline -> A6_full_sbn_bsc_v4).
            verbose: In thông tin tiến trình ra stdout. Mặc định: True.
        """
        self.tau_text = float(tau_text)
        self.tau_visual = float(tau_visual)
        self.modal_discount = float(modal_discount)
        self.prune_lambda = float(prune_lambda)
        self.min_edge_threshold = float(min_edge_threshold)
        self.use_modal = bool(use_modal)
        self.use_behavior = bool(use_behavior)
        self.use_pruning = bool(use_pruning)
        self.ablation_config = ablation_config
        self.verbose = verbose

        # Nếu chỉ định ablation_config, load cấu hình từ dictionary
        if ablation_config is not None:
            try:
                from models.stair_sbn_bsc_v4_utils import get_ablation_config
                cfg_dict = get_ablation_config(ablation_config)
                self.use_modal = cfg_dict['use_modal']
                self.use_behavior = cfg_dict['use_behavior']
                self.use_pruning = cfg_dict['use_pruning']
                self._log(
                    f"Kích hoạt cấu hình bóc tách '{ablation_config}': "
                    f"use_modal={self.use_modal}, use_behavior={self.use_behavior}, "
                    f"use_pruning={self.use_pruning}"
                )
            except Exception as e:
                self._log(f"Cảnh báo: Không thể nạp ablation_config '{ablation_config}' ({e}). Giữ cờ mặc định.")

    def _log(self, message: str) -> None:
        """In thông báo kèm tiền tố [SBN-BSC v4]."""
        if self.verbose:
            msg = f"[SBN-BSC v4] {message}"
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', errors='replace').decode('ascii'))

    def _compute_modal_quality(
        self,
        t_norm: torch.Tensor,
        v_norm: torch.Tensor,
        row: np.ndarray,
        col: np.ndarray,
    ) -> np.ndarray:
        """
        Tính điểm đồng thuận đa phương thức q_modal qua Thresholded Geometric Mean.

        q_modal = sqrt(relu(sim_t - tau_t) * relu(sim_v - tau_v))

        Args:
            t_norm: Text features đã chuẩn hóa L2 [N_i, d_t]
            v_norm: Visual features đã chuẩn hóa L2 [N_i, d_v]
            row: Chỉ số hàng cạnh kNN [E_raw]
            col: Chỉ số cột cạnh kNN [E_raw]

        Returns:
            q_modal: np.ndarray [E_raw], dtype float32, range [0, 1]
        """
        if not self.use_modal:
            return np.zeros(len(row), dtype=np.float32)

        # Đảm bảo index nằm cùng thiết bị (CPU/GPU) với feature tensors
        device = t_norm.device
        row_t = torch.from_numpy(row).long().to(device)
        col_t = torch.from_numpy(col).long().to(device)

        # Cosine similarity từng cạnh (vectorized dot product)
        sim_t = (t_norm[row_t] * t_norm[col_t]).sum(dim=-1).cpu().numpy()
        sim_v = (v_norm[row_t] * v_norm[col_t]).sum(dim=-1).cpu().numpy()

        # Cắt cụt ngưỡng ReLU
        sim_t_thresh = np.maximum(sim_t - self.tau_text, 0.0)
        sim_v_thresh = np.maximum(sim_v - self.tau_visual, 0.0)

        # Trung bình nhân có ngưỡng (Thresholded Geometric Mean)
        q_modal = np.sqrt(sim_t_thresh * sim_v_thresh)
        return q_modal.astype(np.float32)

    def _compute_behavioral_quality(
        self,
        R: sp.csr_matrix,
        row: np.ndarray,
        col: np.ndarray,
        num_items: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính số lần đồng mua C[row, col] và điểm Ochiai q_behavior qua COO searchsorted lookup.

        q_behavior = C_ij / (sqrt(D_i * D_j) + epsilon)

        Args:
            R: Ma trận tương tác huấn luyện [N_u, N_i] scipy CSR
            row: Chỉ số hàng cạnh kNN [E_raw]
            col: Chỉ số cột cạnh kNN [E_raw]
            num_items: Tổng số sản phẩm trong catalog (dùng làm hash multiplier)

        Returns:
            cooccur_counts: np.ndarray [E_raw] — số lần đồng mua
            q_behavior: np.ndarray [E_raw] — điểm Ochiai chuẩn hóa [0, 1]
        """
        if not self.use_behavior:
            zeros = np.zeros(len(row), dtype=np.float32)
            return zeros, zeros

        # Bậc người dùng tương tác với item (EC-1: cold items có degree=0)
        degrees = np.array(R.sum(axis=0)).flatten().astype(np.float32)

        # Tích ma trận đồng mua C = R^T @ R dạng CSR -> COO
        # tocsr() gộp các phần tử trùng lặp và loại bỏ explicit zeros
        C_matrix = (R.T @ R).tocsr().tocoo()

        # Vectorized COO lookup via searchsorted (tối ưu 10x so với CSR fancy indexing)
        edge_key = row.astype(np.int64) * num_items + col.astype(np.int64)
        c_key = C_matrix.row.astype(np.int64) * num_items + C_matrix.col.astype(np.int64)

        sort_idx = np.argsort(c_key)
        sorted_keys = c_key[sort_idx]

        if len(sorted_keys) == 0:
            cooccur_counts = np.zeros(len(row), dtype=np.float32)
        else:
            positions = np.searchsorted(sorted_keys, edge_key)
            positions = np.clip(positions, 0, len(sorted_keys) - 1)
            valid = (sorted_keys[positions] == edge_key)
            cooccur_counts = np.zeros(len(row), dtype=np.float32)
            cooccur_counts[valid] = C_matrix.data[sort_idx[positions[valid]]]

        # Chuẩn hóa Ochiai tránh thiên lệch sản phẩm hot (EC-1: epsilon cho degree=0)
        deg_prod = np.sqrt(degrees[row] * degrees[col]) + 1e-5
        q_behavior = cooccur_counts / deg_prod
        return cooccur_counts, q_behavior.astype(np.float32)

    def _joint_quality_combination(
        self,
        q_modal: np.ndarray,
        q_behavior: np.ndarray,
        cooccur_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Hòa trộn điểm chất lượng theo cơ chế Max-Combination có chiết khấu rho.

        q_joint = max(q_behavior, rho * q_modal)
        Giữ lại nếu có hành vi HOẶC có sự đồng thuận modal.

        Args:
            q_modal: Điểm chất lượng đa phương thức [E_raw]
            q_behavior: Điểm chất lượng hành vi Ochiai [E_raw]
            cooccur_counts: Số lần đồng mua [E_raw]

        Returns:
            q_joint: np.ndarray [E_raw], dtype float32
        """
        # Baseline A0: Không dùng cả 2 tín hiệu
        if not self.use_modal and not self.use_behavior:
            return np.ones_like(q_modal, dtype=np.float32)

        # Modal only
        if not self.use_behavior:
            return q_modal.astype(np.float32)

        # Behavior only
        if not self.use_modal:
            return q_behavior.astype(np.float32)

        has_behavior = (cooccur_counts > 0)
        q_joint = np.maximum(q_behavior, self.modal_discount * q_modal)
        valid_mask = has_behavior | (q_modal > 0.0)
        q_joint = np.where(valid_mask, q_joint, 0.0)
        return q_joint.astype(np.float32)

    def _adaptive_pruning(
        self,
        q_joint: np.ndarray,
        row: np.ndarray,
        col: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Thực hiện lọc ngưỡng tự thích ứng.

        tau_prune = max(tau_min, mu_q + lambda * sigma_q)

        Args:
            q_joint: Điểm chất lượng liên hợp [E_raw]
            row: Chỉ số hàng [E_raw]
            col: Chỉ số cột [E_raw]

        Returns:
            row_clean, col_clean, val_clean: Arrays sau khi cắt tỉa [E_clean]
        """
        if not self.use_pruning:
            self._log("Bỏ qua cắt tỉa tự thích ứng (use_pruning=False). Giữ toàn bộ cạnh.")
            return row, col, q_joint

        active_mask = (q_joint > 0.0)
        q_active = q_joint[active_mask]

        if len(q_active) > 0:
            mu_q = float(q_active.mean())
            sigma_q = float(q_active.std())
            adaptive_thresh = max(self.min_edge_threshold, mu_q + self.prune_lambda * sigma_q)
        else:
            mu_q = 0.0
            sigma_q = 0.0
            adaptive_thresh = self.min_edge_threshold

        keep_mask = (q_joint >= adaptive_thresh)
        row_clean = row[keep_mask]
        col_clean = col[keep_mask]
        val_clean = q_joint[keep_mask]

        self._log(
            f"Adaptive Pruning: mu_q={mu_q:.4f}, sigma_q={sigma_q:.4f} => "
            f"Thresh={adaptive_thresh:.4f}. Giữ lại {len(val_clean):,}/{len(q_joint):,} cạnh "
            f"({len(val_clean)/max(len(q_joint), 1)*100:.2f}%)."
        )
        return row_clean, col_clean, val_clean

    def _laplacian_normalization(
        self,
        row_clean: np.ndarray,
        col_clean: np.ndarray,
        val_clean: np.ndarray,
        num_items: int,
    ) -> sp.coo_matrix:
        """
        Thực hiện đối xứng hóa cực đại và chuẩn hóa Laplacian D^(-1/2) A D^(-1/2).

        EC-4: Over-pruned disconnect safety — deg_clean clamped to 1e-5
        EC-5: kNN asymmetry -> adj.maximum(adj.T)

        Args:
            row_clean: Chỉ số hàng sau cắt tỉa [E_clean]
            col_clean: Chỉ số cột sau cắt tỉa [E_clean]
            val_clean: Trọng số cạnh sau cắt tỉa [E_clean]
            num_items: Tổng số sản phẩm

        Returns:
            L_norm: scipy.sparse.coo_matrix [num_items, num_items] đã chuẩn hóa
        """
        adj_clean = sp.coo_matrix(
            (val_clean, (row_clean, col_clean)),
            shape=(num_items, num_items),
            dtype=np.float32,
        )

        # Đối xứng hóa cực đại A_sym = max(A, A^T) — EC-5
        adj_sym = adj_clean.maximum(adj_clean.T)

        # Tính bậc và chuẩn hóa Laplacian D^(-1/2) * A_sym * D^(-1/2)
        deg_clean = np.array(adj_sym.sum(axis=1)).flatten()
        # EC-4: Bảo vệ item bị cô lập sau cắt tỉa
        deg_clean = np.maximum(deg_clean, 1e-5)
        deg_inv_sqrt = np.power(deg_clean, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

        D_inv = sp.diags(deg_inv_sqrt)
        L_norm = (D_inv @ adj_sym @ D_inv).tocoo()

        self._log(
            f"Laplacian Normalization: degree min={deg_clean.min():.4f}, "
            f"max={deg_clean.max():.4f}, mean={deg_clean.mean():.4f}. "
            f"Final nnz={L_norm.nnz:,}"
        )
        return L_norm

    def _to_sparse_csr(
        self,
        L_norm: sp.coo_matrix,
        num_items: int,
    ) -> torch.Tensor:
        """
        Chuyển đổi ma trận COO sang PyTorch Sparse COO -> Coalesce -> Sparse CSR.

        Args:
            L_norm: scipy.sparse.coo_matrix đã chuẩn hóa Laplacian
            num_items: Tổng số sản phẩm

        Returns:
            mAdj_clean: torch.sparse_csr_tensor [num_items, num_items]
        """
        indices = torch.from_numpy(
            np.vstack((L_norm.row.astype(np.int64), L_norm.col.astype(np.int64)))
        ).long()
        values = torch.from_numpy(L_norm.data.astype(np.float32)).float()

        mAdj_coo = torch.sparse_coo_tensor(
            indices, values, size=(num_items, num_items)
        ).coalesce()

        mAdj_clean = mAdj_coo.to_sparse_csr()
        return mAdj_clean

    def build_denoised_mAdj(
        self,
        text_feats: torch.Tensor,
        vis_feats: torch.Tensor,
        train_user_item_matrix: sp.csr_matrix,
        raw_knn_edge_index: torch.Tensor,
        num_items: int,
    ) -> torch.Tensor:
        """
        Xây dựng ma trận kNN đã làm sạch và chuẩn hóa Laplacian.

        Orchestrates the 7-stage pipeline:
            Stage 1: L2 Feature Normalization
            Stage 2: Cross-Modal Similarity & Agreement
            Stage 3: Behavioral Signal (Ochiai Co-occurrence)
            Stage 4: Joint Quality Combination
            Stage 5: Adaptive Edge Pruning
            Stage 6: Symmetric Laplacian Normalization
            Stage 7: Sparse Tensor Conversion & Buffer Registration

        Args:
            text_feats: Tensor đặc trưng văn bản raw [num_items, d_t]
            vis_feats: Tensor đặc trưng hình ảnh raw [num_items, d_v]
            train_user_item_matrix: Ma trận tương tác huấn luyện [N_u, N_i] scipy CSR
            raw_knn_edge_index: Chỉ số cạnh kNN thô [2, E_raw] (COO format)
            num_items: Tổng số sản phẩm trong catalog

        Returns:
            mAdj_clean: torch.sparse_csr_tensor [num_items, num_items]

        Raises:
            ValueError: Feature/matrix shape mismatch
            TypeError: train_user_item_matrix not scipy sparse
        """
        start_time = time.perf_counter()

        # === INPUT VALIDATION ===
        if text_feats.shape[0] != num_items or vis_feats.shape[0] != num_items:
            raise ValueError(
                f"Feature row mismatch: text={text_feats.shape[0]}, "
                f"vis={vis_feats.shape[0]}, items={num_items}"
            )
        if train_user_item_matrix.shape[1] != num_items:
            raise ValueError(
                f"Interaction catalog mismatch: matrix_cols="
                f"{train_user_item_matrix.shape[1]}, items={num_items}"
            )
        if raw_knn_edge_index.shape[0] != 2:
            raise ValueError(
                f"raw_knn_edge_index must have shape [2, E], "
                f"got {raw_knn_edge_index.shape}"
            )
        if not sp.issparse(train_user_item_matrix):
            raise TypeError(
                "train_user_item_matrix must be a scipy.sparse matrix, "
                f"got {type(train_user_item_matrix)}"
            )

        self._log("=" * 60)
        self._log("BẮT ĐẦU TIỀN XỬ LÝ LỌC NHIỄU ĐỒ THỊ SBN-BSC v4")
        self._log("=" * 60)
        self._log(
            f"Config: tau_t={self.tau_text:.3f}, tau_v={self.tau_visual:.3f}, "
            f"rho={self.modal_discount:.3f}, lambda={self.prune_lambda:.3f}, "
            f"tau_min={self.min_edge_threshold:.3f}, "
            f"modal={self.use_modal}, beh={self.use_behavior}, prune={self.use_pruning}"
        )

        # Ensure CSR format for sparse operations
        R = train_user_item_matrix.tocsr()

        # === STAGE 1: L2 Feature Normalization ===
        t_norm = F.normalize(text_feats.float(), p=2, dim=-1)
        v_norm = F.normalize(vis_feats.float(), p=2, dim=-1)

        # Make explicit writable copies to prevent NumPy 2.x read-only array warnings
        row = raw_knn_edge_index[0].cpu().numpy().copy()
        col = raw_knn_edge_index[1].cpu().numpy().copy()
        self._log(f"1. Tiếp nhận {len(row):,} cạnh kNN thô ban đầu.")

        # === STAGE 2: Cross-Modal Agreement (EVEN-inspired) ===
        q_modal = self._compute_modal_quality(t_norm, v_norm, row, col)
        non_zero_modal = (q_modal > 0).sum()
        self._log(
            f"2. Modal Quality q_modal: mean={q_modal.mean():.4f}, "
            f"max={q_modal.max():.4f}, non-zero={non_zero_modal:,} "
            f"({non_zero_modal/len(q_modal)*100:.2f}%)"
        )

        # === STAGE 3: Behavioral Ochiai Co-occurrence (SIGE-inspired) ===
        cooccur_counts, q_behavior = self._compute_behavioral_quality(
            R, row, col, num_items
        )
        co_purchase_edges = (cooccur_counts > 0).sum()
        self._log(
            f"3. Behavioral Ochiai q_beh: mean={q_behavior.mean():.4f}, "
            f"co-occur count={co_purchase_edges:,} "
            f"({co_purchase_edges/len(q_behavior)*100:.2f}%)"
        )

        # === STAGE 4: Joint Quality Combination ===
        q_joint = self._joint_quality_combination(q_modal, q_behavior, cooccur_counts)
        active_edges = (q_joint > 0).sum()
        self._log(
            f"4. Joint Quality q_joint: mean={q_joint.mean():.4f}, "
            f"active={active_edges:,} ({active_edges/len(q_joint)*100:.2f}%)"
        )

        # === STAGE 5: Adaptive Edge Pruning ===
        row_clean, col_clean, val_clean = self._adaptive_pruning(q_joint, row, col)

        # === STAGE 6: Symmetric Laplacian Normalization ===
        L_norm = self._laplacian_normalization(row_clean, col_clean, val_clean, num_items)

        # === STAGE 7: Sparse CSR Tensor Conversion ===
        mAdj_clean = self._to_sparse_csr(L_norm, num_items)

        # === POST-PROCESSING VALIDATION ===
        vals = mAdj_clean.values()
        assert not torch.isnan(vals).any(), "Phát hiện NaN trong mAdj.values()!"
        assert not torch.isinf(vals).any(), "Phát hiện Inf trong mAdj.values()!"

        elapsed = (time.perf_counter() - start_time) * 1000
        self._log("=" * 60)
        self._log(
            f"TIỀN XỬ LÝ HOÀN TẤT TRONG {elapsed:.2f} ms | "
            f"Final nnz={mAdj_clean._nnz():,}"
        )
        self._log("=" * 60)

        return mAdj_clean
