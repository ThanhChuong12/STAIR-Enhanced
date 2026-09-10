# -*- coding: utf-8 -*-
"""
models/stair_sbn_bsc_v4_utils.py
=================================
Module phụ trợ cho STAIR-SBN-BSC v4:
  - Hàm đo đạc thống kê đồ thị (Graph Statistics)
  - Cấu hình thực nghiệm bóc tách (Ablation Study Configs A0→A6)
  - Context manager đo hiệu năng (Profiling)
"""

import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

import numpy as np
import torch


# ============================================================================
# GRAPH STATISTICS
# ============================================================================

def compute_graph_stats(mAdj: torch.Tensor) -> Dict[str, Any]:
    """
    Tính toán các thống kê cơ bản của ma trận đồ thị thưa.

    Args:
        mAdj: torch.sparse_csr_tensor hoặc torch.sparse_coo_tensor [N, N]

    Returns:
        Dict chứa: nnz, density, degree_min, degree_max, degree_mean,
                    value_min, value_max, value_mean, is_symmetric (approx)
    """
    if mAdj.layout == torch.sparse_csr:
        # Convert to COO for easier analysis
        mAdj_coo = mAdj.to_sparse_coo()
    else:
        mAdj_coo = mAdj

    N = mAdj.shape[0]
    nnz = mAdj_coo._nnz()
    density = nnz / (N * N) if N > 0 else 0.0

    values = mAdj_coo.values().cpu()
    indices = mAdj_coo.indices().cpu()

    # Degree statistics
    if nnz > 0:
        row_indices = indices[0]
        degree_counts = torch.bincount(row_indices, minlength=N).float()
        degree_min = float(degree_counts.min())
        degree_max = float(degree_counts.max())
        degree_mean = float(degree_counts.mean())

        value_min = float(values.min())
        value_max = float(values.max())
        value_mean = float(values.mean())
    else:
        degree_min = degree_max = degree_mean = 0.0
        value_min = value_max = value_mean = 0.0

    # Memory estimation (CSR: crow_indices + col_indices + values)
    mem_bytes = nnz * 4 + (N + 1) * 4 + nnz * 4  # float32 values + int32 indices
    mem_mb = mem_bytes / (1024 * 1024)

    return {
        'num_nodes': N,
        'nnz': nnz,
        'density': density,
        'degree_min': degree_min,
        'degree_max': degree_max,
        'degree_mean': degree_mean,
        'value_min': value_min,
        'value_max': value_max,
        'value_mean': value_mean,
        'estimated_memory_mb': mem_mb,
    }


def print_graph_stats(mAdj: torch.Tensor, label: str = "mAdj") -> None:
    """In thống kê đồ thị ra stdout."""
    stats = compute_graph_stats(mAdj)
    print(f"\n[Graph Stats: {label}]")
    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges (nnz): {stats['nnz']:,}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Degree: min={stats['degree_min']:.0f}, max={stats['degree_max']:.0f}, mean={stats['degree_mean']:.2f}")
    print(f"  Values: min={stats['value_min']:.4f}, max={stats['value_max']:.4f}, mean={stats['value_mean']:.4f}")
    print(f"  Est. Memory: {stats['estimated_memory_mb']:.2f} MB")


# ============================================================================
# ABLATION STUDY CONFIGURATIONS (Blueprint Section 4.10)
# ============================================================================

ABLATION_CONFIGS = {
    'A0_baseline': {
        'desc': 'STAIR Baseline (Raw unweighted kNN)',
        'use_modal': False,
        'use_behavior': False,
        'use_pruning': False,
    },
    'A1_modal_only': {
        'desc': 'Modal Filtering only (Thresholded GeoMean)',
        'use_modal': True,
        'use_behavior': False,
        'use_pruning': False,
    },
    'A2_behavior_only': {
        'desc': 'Behavior Filtering only (Ochiai Co-occurrence)',
        'use_modal': False,
        'use_behavior': True,
        'use_pruning': False,
    },
    'A3_multi_no_prune': {
        'desc': 'Dual Signal Joint Quality (No Pruning)',
        'use_modal': True,
        'use_behavior': True,
        'use_pruning': False,
    },
    'A4_modal_prune': {
        'desc': 'Modal Filtering + Adaptive Pruning',
        'use_modal': True,
        'use_behavior': False,
        'use_pruning': True,
    },
    'A5_behavior_prune': {
        'desc': 'Behavior Filtering + Adaptive Pruning',
        'use_modal': False,
        'use_behavior': True,
        'use_pruning': True,
    },
    'A6_full_sbn_bsc_v4': {
        'desc': 'Full STAIR-SBN-BSC v4 Architecture',
        'use_modal': True,
        'use_behavior': True,
        'use_pruning': True,
    },
}


def get_ablation_config(config_name: str) -> Dict[str, Any]:
    """
    Trả về cấu hình ablation theo tên.

    Args:
        config_name: Tên cấu hình (e.g., 'A0_baseline', 'A6_full_sbn_bsc_v4')

    Returns:
        Dict cấu hình

    Raises:
        KeyError: Nếu config_name không tồn tại
    """
    if config_name not in ABLATION_CONFIGS:
        available = ', '.join(ABLATION_CONFIGS.keys())
        raise KeyError(
            f"Unknown ablation config '{config_name}'. "
            f"Available: {available}"
        )
    return ABLATION_CONFIGS[config_name]


# ============================================================================
# PERFORMANCE PROFILING
# ============================================================================

@contextmanager
def profile_section(name: str, log_dict: Optional[dict] = None):
    """
    Context manager đo thời gian chuẩn xác bằng đồng hồ GPU/CPU.

    Usage:
        timings = {}
        with profile_section("Stage 1: L2 Norm", timings):
            t_norm = F.normalize(feats)
        print(timings)  # {'Stage 1: L2 Norm': 123.45}

    Args:
        name: Tên giai đoạn đo
        log_dict: Dict để lưu kết quả (optional)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    if log_dict is not None:
        log_dict[name] = elapsed
    print(f"[PROFILER] {name:<40}: {elapsed:>8.2f} ms")
