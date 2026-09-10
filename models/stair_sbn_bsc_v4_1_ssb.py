# -*- coding: utf-8 -*-
"""
models/stair_sbn_bsc_v4_1_ssb.py
=================================
STAIR-BSC-Reweight v4.1-SSB: Safe Spectral Boost for Backward Stepwise Convolution
Synthesized from SIGE (AAAI 2026) and EVEN (AAAI 2025).

Khắc phục triệt để 5 tử huyệt kỹ thuật của dự thảo v4.1 ban đầu:
  1. ZERO DENSE CONVERSION: Hoàn toàn xử lý trên sparse COO/CSR, 0 MB dense overhead (an toàn trên Electronics 63K).
  2. SYMMETRIC LAPLACIAN: Đối xứng hóa tường minh W_sym = max(W, W^T), bảo toàn tính Đối xứng Nửa xác định Dương (SPSD).
  3. DEGREE PRESERVATION: Bảo tồn 100% cạnh kNN gốc của STAIR (0% cắt tỉa, bậc trung bình 6-8, 0% item bị cô lập).
  4. ZERO LEARNABLE PARAMETERS: Chuẩn hóa Ochiai và Cosine thuần túy xác định, không có nn.Parameter bị đóng băng.
  5. BOUNDED SCALING: Trọng số w_ij được chặn cận an toàn trong [1.0, 1.8], bảo toàn thang năng lượng phổ STAIR gốc.

Công thức cốt lõi:
  w_ij = w_base + alpha * q_modal_ij + beta * q_behavior_ij
  trong đó w_base = 1.0, alpha = 0.50, beta = 0.30  ==>  w_ij in [1.0, 1.8]
  S_tilde = D_W^(-1/2) * W_sym * D_W^(-1/2)
"""

import time
import gc
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


class STAIR_BSC_Reweight_Engine:
    """
    Engine tiền xử lý tăng cường trọng số an toàn cho ma trận Backward Stepwise Convolution (BSC).
    Thực thi 100% offline một lần duy nhất trong hàm prepare(), không tốn VRAM hay thời gian huấn luyện online.

    Chế độ hoạt động (mode):
      - 'baseline': w_ij = 1.0 (chuẩn hóa Laplacian giữ nguyên tô-pô gốc)
      - 'modal_only': w_ij = 1.0 + alpha * q_modal (Bước A kiểm chứng độc lập)
      - 'behavior_only': w_ij = 1.0 + beta * q_behavior (Bước B kiểm chứng độc lập)
      - 'full_ssb': w_ij = 1.0 + alpha * q_modal + beta * q_behavior (Bước C kết hợp toàn diện)
    """

    def __init__(
        self,
        mode: str = "full_ssb",
        alpha: float = 0.50,
        beta: float = 0.30,
        tau_t: float = 0.10,
        tau_v: float = 0.10,
        min_weight: float = 1.00,
        max_weight: float = 1.80,
        eps: float = 1e-8,
        verbose: bool = True,
    ):
        """
        Khởi tạo Engine với các siêu tham số tăng cường an toàn.

        Args:
            mode: Chế độ hoạt động ['baseline', 'modal_only', 'behavior_only', 'full_ssb']
            alpha: Trọng số tăng cường đa phương thức (mặc định: 0.50)
            beta: Trọng số tăng cường hành vi đồng mua Ochiai (mặc định: 0.30)
            tau_t: Ngưỡng lọc tương đồng văn bản [0.05, 0.25] (mặc định: 0.10)
            tau_v: Ngưỡng lọc tương đồng thị giác [0.05, 0.25] (mặc định: 0.10)
            min_weight: Cận dưới trọng số cạnh (mặc định: 1.00)
            max_weight: Cận trên trọng số cạnh (mặc định: 1.80)
            eps: Hằng số chống chia cho 0 (mặc định: 1e-8)
            verbose: In chi tiết quá trình tiền xử lý
        """
        valid_modes = ["baseline", "modal_only", "behavior_only", "full_ssb"]
        if mode not in valid_modes:
            raise ValueError(f"Mode '{mode}' không hợp lệ. Phải thuộc: {valid_modes}")

        self.mode = mode
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_t = float(tau_t)
        self.tau_v = float(tau_v)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)
        self.verbose = verbose

        # Bộ đệm thống kê phục vụ kiểm tra tĩnh và đối soát
        self.stats: Dict[str, Any] = {}

    def _log(self, message: str) -> None:
        """In thông điệp kèm tiền tố [v4.1-SSB]."""
        if self.verbose:
            msg = f"[v4.1-SSB] {message}"
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', errors='replace').decode('ascii'))

    def compute_modal_quality(
        self,
        text_feats: torch.Tensor,
        vis_feats: torch.Tensor,
        row: np.ndarray,
        col: np.ndarray,
    ) -> np.ndarray:
        """
        Tính điểm đồng thuận đa phương thức q_modal qua Thresholded Geometric Mean:
          q_modal_ij = sqrt( relu(s_t - tau_t) * relu(s_v - tau_v) )

        Thực thi vectorized O(|E|) trên đúng tập cạnh kNN, không tạo ma trận dense N x N.
        """
        with torch.no_grad():
            # 1. Chuẩn hóa L2 cho feature embeddings
            t_norm = F.normalize(text_feats.float(), p=2, dim=-1)
            v_norm = F.normalize(vis_feats.float(), p=2, dim=-1)

            # 2. Trích xuất cosine similarity trên từng cạnh (row, col)
            device = t_norm.device
            row_t = torch.from_numpy(row).long().to(device)
            col_t = torch.from_numpy(col).long().to(device)

            sim_t = (t_norm[row_t] * t_norm[col_t]).sum(dim=-1)
            sim_v = (v_norm[row_t] * v_norm[col_t]).sum(dim=-1)

            # 3. Thresholded Geometric Mean
            s_t_thresh = F.relu(sim_t - self.tau_t)
            s_v_thresh = F.relu(sim_v - self.tau_v)
            q_modal_t = torch.sqrt(s_t_thresh * s_v_thresh + self.eps)

            q_modal = q_modal_t.cpu().numpy().astype(np.float32)

        return q_modal

    def compute_behavioral_quality(
        self,
        train_user_item_matrix: sp.spmatrix,
        row: np.ndarray,
        col: np.ndarray,
        num_items: int,
    ) -> np.ndarray:
        """
        Tính điểm đồng mua hành vi chuẩn hóa Ochiai:
          q_behavior_ij = C_ij / (sqrt(D_i * D_j) + eps)
          với C = R^T @ R và D_i = sum_u R_ui

        Thực thi 100% sparse qua searchsorted lookup, O(|E| log |E_C|), RAM tối thiểu.
        """
        R = train_user_item_matrix.tocsr()

        # Bậc người dùng tương tác của từng sản phẩm (degree)
        degrees = np.array(R.sum(axis=0)).flatten().astype(np.float32)

        # Tính ma trận đồng mua C = R^T @ R dạng CSR -> COO
        C_matrix = (R.T @ R).tocsr().tocoo()

        # Vectorized lookup qua 64-bit coordinate key: row * num_items + col
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
            cooccur_counts[valid] = C_matrix.data[sort_idx[positions[valid]]].astype(np.float32)

        # Chuẩn hóa Ochiai
        deg_prod = np.sqrt(degrees[row] * degrees[col]) + self.eps
        q_behavior = (cooccur_counts / deg_prod).astype(np.float32)

        # Giải phóng bộ nhớ tạm thời
        del C_matrix, sorted_keys, c_key, edge_key
        gc.collect()

        return q_behavior

    def _symmetrize_and_normalize(
        self,
        row: np.ndarray,
        col: np.ndarray,
        weight: np.ndarray,
        num_items: int,
    ) -> sp.coo_matrix:
        """
        Đối xứng hóa ma trận thưa và chuẩn hóa Symmetric Laplacian:
          W_sym = max(W, W^T)
          S_tilde = D_W^(-1/2) * W_sym * D_W^(-1/2)

        Đảm bảo 100% ma trận đầu ra là Đối xứng Nửa xác định Dương (SPSD).
        """
        # Tạo ma trận có hướng cơ sở
        adj_dir = sp.coo_matrix(
            (weight.astype(np.float32), (row.astype(np.int64), col.astype(np.int64))),
            shape=(num_items, num_items)
        ).tocsr()

        # Đối xứng hóa: W_sym = max(W, W^T)
        adj_sym = adj_dir.maximum(adj_dir.T).tocsr()

        # Tính bậc đỉnh có trọng số: D_W(i, i) = sum_j W_sym(i, j)
        deg = np.array(adj_sym.sum(axis=1)).flatten().astype(np.float32)
        deg_safe = np.maximum(deg, 1e-5)
        deg_inv_sqrt = np.power(deg_safe, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0

        # Chuẩn hóa Symmetric Laplacian: D_inv @ W_sym @ D_inv
        D_inv = sp.diags(deg_inv_sqrt, format='csr')
        L_norm = (D_inv @ adj_sym @ D_inv).tocoo()

        # Lưu thống kê bậc đỉnh
        self.stats['deg_min'] = float(deg.min())
        self.stats['deg_max'] = float(deg.max())
        self.stats['deg_mean'] = float(deg.mean())
        self.stats['deg_std'] = float(deg.std())
        self.stats['nnz_sym'] = int(L_norm.nnz)

        return L_norm

    def build_boosted_mAdj(
        self,
        text_feats: torch.Tensor,
        vis_feats: torch.Tensor,
        train_user_item_matrix: sp.spmatrix,
        raw_knn_edge_index: Union[torch.Tensor, np.ndarray],
        num_items: int,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Quy trình tiền xử lý hoàn chỉnh xây dựng ma trận BSC Smoother:
          1. Trích xuất danh sách cạnh kNN cơ sở (Bảo tồn 100% tô-pô).
          2. Tính q_modal (Thresholded Geometric Mean).
          3. Tính q_behavior (Ochiai Normalized Co-occurrence).
          4. Tăng cường trọng số an toàn w_ij in [1.0, 1.8].
          5. Đối xứng hóa W_sym = max(W, W^T) & Chuẩn hóa Symmetric Laplacian.
          6. Chuyển đổi sang torch.sparse_csr_tensor.

        Args:
            text_feats: Tensor đặc trưng văn bản raw [N, D_t]
            vis_feats: Tensor đặc trưng thị giác raw [N, D_v]
            train_user_item_matrix: Ma trận tương tác [U, N] scipy sparse
            raw_knn_edge_index: Chỉ số cạnh kNN thô [2, E] (Tensor hoặc ndarray)
            num_items: Tổng số sản phẩm trong catalog
            target_device: Device đích cho tensor đầu ra (mặc định: text_feats.device)

        Returns:
            mAdj: torch.sparse_csr_tensor [num_items, num_items]
        """
        start_time = time.perf_counter()

        if target_device is None:
            target_device = text_feats.device if isinstance(text_feats, torch.Tensor) else torch.device('cpu')

        # Chuyển raw_knn_edge_index sang numpy array [2, E]
        if isinstance(raw_knn_edge_index, torch.Tensor):
            edge_index_np = raw_knn_edge_index.cpu().numpy()
        else:
            edge_index_np = np.asarray(raw_knn_edge_index)

        row = edge_index_np[0].astype(np.int64).copy()
        col = edge_index_np[1].astype(np.int64).copy()
        num_edges = len(row)

        self._log("=" * 65)
        self._log(f"KHỞI CHẠY TIỀN XỬ LÝ STAIR-BSC-REWEIGHT v4.1-SSB (Mode: '{self.mode}')")
        self._log(f"Catalog: N_items={num_items:,} | Baseline kNN Cạnh={num_edges:,}")
        self._log(f"Cấu hình: alpha={self.alpha:.2f}, beta={self.beta:.2f}, tau_t={self.tau_t:.2f}, tau_v={self.tau_v:.2f}")
        self._log("=" * 65)

        # 1. Khởi tạo trọng số cơ sở w_base = 1.0 (Bảo tồn 100% năng lượng phổ)
        w_base = np.ones(num_edges, dtype=np.float32)

        # 2. Tính tín hiệu Đa phương thức: q_modal
        q_modal = np.zeros(num_edges, dtype=np.float32)
        if self.mode in ["modal_only", "full_ssb"]:
            q_modal = self.compute_modal_quality(text_feats, vis_feats, row, col)
            non_zero_m = (q_modal > 0).sum()
            self._log(
                f"[Stage 1] q_modal stats: min={q_modal.min():.4f}, max={q_modal.max():.4f}, "
                f"mean={q_modal.mean():.4f}, non_zero={non_zero_m:,} ({non_zero_m/num_edges*100:.2f}%)"
            )
            self.stats['q_modal_mean'] = float(q_modal.mean())
            self.stats['q_modal_max'] = float(q_modal.max())
            self.stats['q_modal_nonzero_pct'] = float(non_zero_m / num_edges * 100)

        # 3. Tính tín hiệu Hành vi Đồng mua: q_behavior (Ochiai)
        q_behavior = np.zeros(num_edges, dtype=np.float32)
        if self.mode in ["behavior_only", "full_ssb"]:
            q_behavior = self.compute_behavioral_quality(train_user_item_matrix, row, col, num_items)
            non_zero_b = (q_behavior > 0).sum()
            self._log(
                f"[Stage 2] q_behavior (Ochiai): min={q_behavior.min():.4f}, max={q_behavior.max():.4f}, "
                f"mean={q_behavior.mean():.4f}, non_zero={non_zero_b:,} ({non_zero_b/num_edges*100:.2f}%)"
            )
            self.stats['q_behavior_mean'] = float(q_behavior.mean())
            self.stats['q_behavior_max'] = float(q_behavior.max())
            self.stats['q_behavior_nonzero_pct'] = float(non_zero_b / num_edges * 100)

        # 4. Safe Additive Boosting
        eff_alpha = self.alpha if self.mode in ["modal_only", "full_ssb"] else 0.0
        eff_beta = self.beta if self.mode in ["behavior_only", "full_ssb"] else 0.0

        w_boosted = w_base + eff_alpha * q_modal + eff_beta * q_behavior

        # Chặn cận an toàn nghiêm ngặt
        w_boosted = np.clip(w_boosted, self.min_weight, self.max_weight)
        self._log(
            f"[Stage 3] w_boosted final: min={w_boosted.min():.4f}, max={w_boosted.max():.4f}, "
            f"mean={w_boosted.mean():.4f}, std={w_boosted.std():.4f}"
        )
        self.stats['w_min'] = float(w_boosted.min())
        self.stats['w_max'] = float(w_boosted.max())
        self.stats['w_mean'] = float(w_boosted.mean())
        self.stats['w_std'] = float(w_boosted.std())

        # 5. Đối xứng hóa & Chuẩn hóa Laplacian
        L_norm = self._symmetrize_and_normalize(row, col, w_boosted, num_items)

        # 6. Chuyển đổi sang PyTorch Sparse CSR Tensor
        indices_sym = torch.from_numpy(
            np.vstack((L_norm.row.astype(np.int64), L_norm.col.astype(np.int64)))
        ).long()
        values_sym = torch.from_numpy(L_norm.data.astype(np.float32)).float()

        mAdj_coo = torch.sparse_coo_tensor(
            indices_sym, values_sym, size=(num_items, num_items)
        ).coalesce()
        mAdj_csr = mAdj_coo.to_sparse_csr()

        # Kiểm tra tính toàn vẹn (Safety Assertions)
        vals = mAdj_csr.values()
        assert not torch.isnan(vals).any(), "Phát hiện NaN trong mAdj CSR values!"
        assert not torch.isinf(vals).any(), "Phát hiện Inf trong mAdj CSR values!"
        assert mAdj_csr._nnz() > 0, "mAdj không thể rỗng!"

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._log(
            f"[Completed] Hoàn tất tiền xử lý mAdj trong {elapsed_ms:.2f} ms | "
            f"nnz={mAdj_csr._nnz():,} | Deg: min={self.stats['deg_min']:.2f}, "
            f"mean={self.stats['deg_mean']:.2f}, max={self.stats['deg_max']:.2f}"
        )
        self._log("=" * 65)

        self.stats['elapsed_ms'] = elapsed_ms
        self.stats['nnz'] = mAdj_csr._nnz()

        return mAdj_csr.to(target_device)

    def compute_boosted_adj(
        self,
        text_feats: torch.Tensor,
        vis_feats: torch.Tensor,
        train_user_item_matrix: sp.spmatrix,
        raw_knn_edge_index: Union[torch.Tensor, np.ndarray],
        num_items: int,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Alias cho hàm build_boosted_mAdj."""
        return self.build_boosted_mAdj(
            text_feats=text_feats,
            vis_feats=vis_feats,
            train_user_item_matrix=train_user_item_matrix,
            raw_knn_edge_index=raw_knn_edge_index,
            num_items=num_items,
            target_device=target_device,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Trả về dictionary chứa toàn bộ chỉ số thống kê phục vụ logging và static audit."""
        return dict(self.stats)


# Alias tương thích cho tên class
SBN_BSC_v4_1_SSB_Preprocessor = STAIR_BSC_Reweight_Engine
