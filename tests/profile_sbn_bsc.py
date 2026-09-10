# -*- coding: utf-8 -*-
"""
tests/profile_sbn_bsc.py
=========================
Performance profiling script cho SBN_BSC_Preprocessor.

Đo thời gian chi tiết từng stage trên dữ liệu giả lập quy mô khác nhau:
  - Small: 1,000 items (Quick sanity check)
  - Medium: 10,000 items (Realistic benchmark)
  - Large: 50,000 items (Stress test)

Usage:
    python tests/profile_sbn_bsc.py
    python tests/profile_sbn_bsc.py --scale medium
"""

import argparse
import time
import sys
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.stair_sbn_bsc_v4 import SBN_BSC_Preprocessor
from models.stair_sbn_bsc_v4_utils import profile_section, compute_graph_stats


# ============================================================================
# Dataset Factory
# ============================================================================

SCALE_CONFIGS = {
    'small': {
        'num_items': 1_000,
        'num_users': 3_000,
        'density': 0.02,
        'k_total': 6,
        'text_dim': 768,
        'vis_dim': 4096,
    },
    'medium': {
        'num_items': 10_000,
        'num_users': 30_000,
        'density': 0.005,
        'k_total': 6,
        'text_dim': 768,
        'vis_dim': 4096,
    },
    'large': {
        'num_items': 50_000,
        'num_users': 150_000,
        'density': 0.001,
        'k_total': 6,
        'text_dim': 768,
        'vis_dim': 4096,
    },
}


def generate_synthetic_data(config: dict):
    """Tạo dữ liệu giả lập theo cấu hình quy mô."""
    np.random.seed(42)
    torch.manual_seed(42)

    ni = config['num_items']
    nu = config['num_users']

    print(f"\n[DATA GEN] Creating synthetic dataset: {ni:,} items, {nu:,} users")

    text_feats = torch.randn(ni, config['text_dim'])
    vis_feats = torch.randn(ni, config['vis_dim'])

    R = sp.random(nu, ni, density=config['density'],
                  format='csr', dtype=np.float32)
    R.data[:] = 1.0

    # kNN edges
    num_edges = ni * config['k_total']
    rows = np.random.randint(0, ni, size=num_edges)
    cols = np.random.randint(0, ni, size=num_edges)
    mask = rows != cols
    raw_knn = torch.tensor(np.stack([rows[mask], cols[mask]]), dtype=torch.long)

    print(f"[DATA GEN] R: {R.shape}, nnz={R.nnz:,}, density={R.nnz/(nu*ni):.6f}")
    print(f"[DATA GEN] kNN: {raw_knn.shape[1]:,} edges")

    return text_feats, vis_feats, R, raw_knn, ni


# ============================================================================
# Profiling Runner
# ============================================================================

def run_profiling(scale: str = 'small'):
    """Chạy profiling cho quy mô cho trước."""
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale '{scale}'. Use: {list(SCALE_CONFIGS.keys())}")

    config = SCALE_CONFIGS[scale]
    print(f"\n{'='*70}")
    print(f"  PROFILING SBN-BSC v4 — Scale: {scale.upper()}")
    print(f"{'='*70}")

    # Generate data
    timings = {}
    with profile_section("Data Generation", timings):
        text_feats, vis_feats, R, raw_knn, num_items = generate_synthetic_data(config)

    # Run preprocessor with detailed timing
    prep = SBN_BSC_Preprocessor(verbose=True)

    with profile_section("TOTAL build_denoised_mAdj", timings):
        mAdj = prep.build_denoised_mAdj(
            text_feats, vis_feats, R, raw_knn, num_items
        )

    # Print graph stats
    stats = compute_graph_stats(mAdj)
    print(f"\n[RESULT] Final Graph Stats:")
    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges (nnz): {stats['nnz']:,}")
    print(f"  Density: {stats['density']:.8f}")
    print(f"  Degree: min={stats['degree_min']:.0f}, max={stats['degree_max']:.0f}, "
          f"mean={stats['degree_mean']:.2f}")
    print(f"  Values: min={stats['value_min']:.4f}, max={stats['value_max']:.4f}, "
          f"mean={stats['value_mean']:.4f}")
    print(f"  Est. Memory: {stats['estimated_memory_mb']:.2f} MB")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TIMING SUMMARY ({scale.upper()})")
    print(f"{'='*70}")
    for name, elapsed in timings.items():
        print(f"  {name:<45}: {elapsed:>8.2f} ms")
    print(f"{'='*70}")

    return timings


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile SBN-BSC v4 Preprocessor')
    parser.add_argument('--scale', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Scale of synthetic data (default: small)')
    parser.add_argument('--all', action='store_true',
                        help='Run all scales')
    args = parser.parse_args()

    if args.all:
        all_timings = {}
        for scale in ['small', 'medium', 'large']:
            try:
                timings = run_profiling(scale)
                all_timings[scale] = timings
            except MemoryError:
                print(f"\n[ERROR] Out of memory for scale '{scale}'. Skipping.")
                break

        print(f"\n{'='*70}")
        print(f"  CROSS-SCALE COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Scale':<10} {'Total Time (ms)':<20} {'Status'}")
        for scale, timings in all_timings.items():
            total = timings.get('TOTAL build_denoised_mAdj', 0)
            status = '✅' if total < 5000 else '⚠️'
            print(f"  {scale:<10} {total:<20.2f} {status}")
    else:
        run_profiling(args.scale)
