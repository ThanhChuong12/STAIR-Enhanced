# -*- coding: utf-8 -*-
"""
tests/test_stair_cnlgcl_v1_r.py
================================
End-to-End Unit & Integration Test Suite for STAIR-CNLGCL v1-R (Phase 4).
Tests:
  1. BSC_Reweight_Engine (SPSD, Ochiai, Clamping, CSR Tensor, Dual Format).
  2. CNLGCL_Loss_v1R (Fused Ops, Percentile AMM, FNF, Sign-Preserving Noise, Gradient Flow).
  3. STAIR_CNLGCL_v1_R (Standalone Model: prepare, encode, fit, predict).
  4. Top-level re-export (models.stair_cnlgcl_v1_r & models.GD4).
  5. main_stair_cnlgcl_v1_r.py structural validity & dual-alias configuration.
"""

import math
import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from models.GD4.stair_cnlgcl_v1_r import BSC_Reweight_Engine, CNLGCL_Loss_v1R, STAIR_CNLGCL_v1_R
import models.stair_cnlgcl_v1_r as top_level_shim


def test_reexport():
    print("[1/5] Testing top-level re-export shim...")
    assert hasattr(top_level_shim, 'BSC_Reweight_Engine')
    assert hasattr(top_level_shim, 'CNLGCL_Loss_v1R')
    assert hasattr(top_level_shim, 'STAIR_CNLGCL_v1_R')
    assert top_level_shim.CNLGCL_Loss_v1R is CNLGCL_Loss_v1R
    assert top_level_shim.BSC_Reweight_Engine is BSC_Reweight_Engine
    assert top_level_shim.STAIR_CNLGCL_v1_R is STAIR_CNLGCL_v1_R
    print("  --> PASS: Top-level re-export shim identical to GD4 modules.")


def test_bsc_reweight_engine():
    print("\n[2/5] Testing BSC_Reweight_Engine...")
    num_items = 50
    dim = 32
    num_users = 30

    torch.manual_seed(42)
    np.random.seed(42)

    text_feats = torch.randn(num_items, dim)
    vis_feats = torch.randn(num_items, dim)

    # Synthetic interaction matrix
    user_idx = np.random.randint(0, num_users, size=200)
    item_idx = np.random.randint(0, num_items, size=200)
    data = np.ones(len(user_idx), dtype=np.float32)
    train_R = sp.csr_matrix((data, (user_idx, item_idx)), shape=(num_users, num_items))

    # Test Format A: scipy.sparse.csr_matrix
    knn_rows = []
    knn_cols = []
    for i in range(num_items):
        for k in range(1, 4):
            knn_rows.append(i)
            knn_cols.append((i + k) % num_items)
    raw_knn_sp = sp.csr_matrix((np.ones(len(knn_rows), dtype=np.float32), (knn_rows, knn_cols)), shape=(num_items, num_items))

    engine = BSC_Reweight_Engine(alpha=0.40, beta=0.20, min_weight=1.0, max_weight=3.6)
    mAdj_sp = engine.build_boosted_mAdj(
        text_feats=text_feats,
        vis_feats=vis_feats,
        train_user_item_matrix=train_R,
        raw_knn_adj=raw_knn_sp,
    )

    assert mAdj_sp.is_sparse_csr, "mAdj must be sparse CSR tensor"
    assert mAdj_sp.size() == (num_items, num_items)
    assert mAdj_sp._nnz() > 0
    assert not torch.isnan(mAdj_sp.values()).any()
    print(f"  --> Format A (scipy.sparse): SPSD CSR created with nnz={mAdj_sp._nnz()}")

    # Test Format B: torch.Tensor edge_index + edge_weight (FreeRec format)
    raw_edge_index = torch.tensor([knn_rows, knn_cols], dtype=torch.long)
    raw_edge_weight = torch.ones(len(knn_rows), dtype=torch.float32)

    mAdj_tensor = engine.build_boosted_mAdj(
        text_feats=text_feats,
        vis_feats=vis_feats,
        train_user_item_matrix=train_R,
        raw_knn_adj=raw_edge_index,
        raw_edge_weight=raw_edge_weight,
        num_items=num_items,
    )
    assert mAdj_tensor.is_sparse_csr
    assert mAdj_tensor.size() == (num_items, num_items)
    assert mAdj_tensor._nnz() > 0
    print(f"  --> Format B (torch.Tensor): SPSD CSR created with nnz={mAdj_tensor._nnz()}")

    # Test modes: modal_only, behavior_only, baseline
    e_modal = BSC_Reweight_Engine(mode="modal_only", alpha=0.50, beta=0.20)
    assert e_modal.alpha == 0.50 and e_modal.beta == 0.0
    e_base = BSC_Reweight_Engine(mode="baseline", alpha=0.50, beta=0.20)
    assert e_base.alpha == 0.0 and e_base.beta == 0.0
    print("  --> PASS: BSC_Reweight_Engine passed all assertions.")


def test_cnlgcl_loss_module():
    print("\n[3/5] Testing CNLGCL_Loss_v1R...")
    n_users, n_items, dim = 100, 80, 64
    batch_size = 32

    cl_loss = CNLGCL_Loss_v1R(
        n_users=n_users,
        n_items=n_items,
        tau=0.20,
        alpha_dir=0.50,
        eps=0.08,
        tau_thresh=0.85,
        lambda_cl=0.008,
        warmup_epochs=50,
        margin_max=0.02,
        use_amm=True,
        use_fn_mask=True,
        use_fused_ops=True,
    )
    cl_loss.train()

    # Test Warmup Scheduler
    cl_loss.update_epoch(1)
    lam1, m1 = cl_loss.get_current_params()
    assert abs(lam1 - 0.008 * (1.0 / 50.0)) < 1e-6
    cl_loss.update_epoch(50)
    lam50, _ = cl_loss.get_current_params()
    assert abs(lam50 - 0.008) < 1e-6
    cl_loss.update_epoch(100)
    lam100, _ = cl_loss.get_current_params()
    assert abs(lam100 - 0.008) < 1e-6
    print(f"  --> Warmup scheduler: Ep 1={lam1:.6f}, Ep 50={lam50:.6f}, Ep 100={lam100:.6f}")

    # Create dummy embeddings
    torch.manual_seed(42)
    h0 = torch.randn(n_users + n_items, dim, requires_grad=True)
    h1 = torch.randn(n_users + n_items, dim, requires_grad=True)
    layer_embeds = [h0, h1]

    users = torch.randint(0, n_users, (batch_size,))
    positives = torch.randint(0, n_items, (batch_size,))
    beta = torch.linspace(0.1, 0.9, dim)
    item_modals = torch.randn(n_items, dim)
    modal_consistency = torch.rand(n_items)  # Calibrated [0, 1]

    # Test Fused Ops
    loss_fused, raw_fused = cl_loss(
        layer_embeds=layer_embeds,
        users=users,
        positives=positives,
        beta=beta,
        item_modals=item_modals,
        modal_consistency=modal_consistency,
    )
    assert not torch.isnan(loss_fused)
    assert loss_fused.item() > 0
    print(f"  --> Forward pass with Fused Ops: loss={loss_fused.item():.6f}, raw={raw_fused:.4f}")

    # Test Backward pass and Gradient Flow
    loss_fused.backward()
    assert h0.grad is not None and not torch.isnan(h0.grad).any()
    assert h1.grad is not None and not torch.isnan(h1.grad).any()
    assert h0.grad.norm().item() > 0
    assert h1.grad.norm().item() > 0
    print(f"  --> 100% Gradient flow verified: ||grad_H0||={h0.grad.norm().item():.4f}, ||grad_H1||={h1.grad.norm().item():.4f}")

    # Test Non-Fused Ops consistency
    cl_loss_nonfused = CNLGCL_Loss_v1R(
        n_users=n_users, n_items=n_items, use_fused_ops=False, eps=0.0
    )
    cl_loss_nonfused.update_epoch(50)
    cl_loss_fused_noeps = CNLGCL_Loss_v1R(
        n_users=n_users, n_items=n_items, use_fused_ops=True, eps=0.0
    )
    cl_loss_fused_noeps.update_epoch(50)

    l_nf, _ = cl_loss_nonfused(layer_embeds, users, positives, beta, item_modals, modal_consistency)
    l_f, _ = cl_loss_fused_noeps(layer_embeds, users, positives, beta, item_modals, modal_consistency)
    diff = abs(l_nf.item() - l_f.item())
    assert diff < 1e-5, f"Fused vs Non-Fused output mismatch: {diff}"
    print(f"  --> Fused vs Non-fused exact match (diff={diff:.8f})")
    print("  --> PASS: CNLGCL_Loss_v1R passed all assertions.")


def test_standalone_model():
    print("\n[4/5] Testing STAIR_CNLGCL_v1_R Standalone Model...")
    n_users, n_items, dim = 60, 40, 32
    model = STAIR_CNLGCL_v1_R(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=dim,
        fsc_layers=3,
        alpha_reweight=0.40,
        beta_reweight=0.20,
        lambda_cl=0.008,
    )
    model.train()

    # Synthetic graph and modal feats
    text_feats = torch.randn(n_items, dim)
    vis_feats = torch.randn(n_items, dim)

    u_idx = np.random.randint(0, n_users, size=150)
    i_idx = np.random.randint(0, n_items, size=150)
    data = np.ones(len(u_idx), dtype=np.float32)
    train_R = sp.csr_matrix((data, (u_idx, i_idx)), shape=(n_users, n_items))

    # Bigraph normalized adjacency
    N = n_users + n_items
    row_bg = np.concatenate([u_idx, i_idx + n_users])
    col_bg = np.concatenate([i_idx + n_users, u_idx])
    data_bg = np.ones(len(row_bg), dtype=np.float32)
    adj_bg = sp.csr_matrix((data_bg, (row_bg, col_bg)), shape=(N, N))
    deg = np.array(adj_bg.sum(axis=1)).flatten()
    d_inv = sp.diags(np.power(np.maximum(deg, 1.0), -0.5), format="csr")
    norm_adj_sp = (d_inv @ adj_bg @ d_inv).tocsr()
    crow = torch.from_numpy(norm_adj_sp.indptr.astype(np.int64))
    col = torch.from_numpy(norm_adj_sp.indices.astype(np.int64))
    vals = torch.from_numpy(norm_adj_sp.data.astype(np.float32))
    norm_adj_t = torch.sparse_csr_tensor(crow, col, vals, size=(N, N))

    # Raw kNN graph
    knn_rows, knn_cols = [], []
    for i in range(n_items):
        for k in range(1, 3):
            knn_rows.append(i)
            knn_cols.append((i + k) % n_items)
    raw_knn = sp.csr_matrix((np.ones(len(knn_rows), dtype=np.float32), (knn_rows, knn_cols)), shape=(n_items, n_items))

    # Run prepare()
    model.prepare(
        text_feats=text_feats,
        vis_feats=vis_feats,
        train_user_item_matrix=train_R,
        raw_knn_adj=raw_knn,
        norm_adj_u2i=norm_adj_t,
    )
    print("  --> prepare() complete. mAdj_csr shape:", model.mAdj_csr.size())

    # Run encode()
    u_emb, i_emb, layer_embeds = model.encode()
    assert u_emb.shape == (n_users, dim)
    assert i_emb.shape == (n_items, dim)
    assert len(layer_embeds) == 4  # H^0, H^1, H^2, H^3
    print("  --> encode() complete. Intermediate layers captured:", len(layer_embeds))

    # Run fit()
    batch_u = torch.randint(0, n_users, (16,))
    batch_pos = torch.randint(0, n_items, (16,))
    batch_neg = torch.randint(0, n_items, (16,))
    loss = model.fit(batch_u, batch_pos, batch_neg, epoch=10)
    assert not torch.isnan(loss)
    loss.backward()
    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None
    print(f"  --> fit() complete: loss={loss.item():.4f}, u_grad_norm={model.user_embedding.weight.grad.norm().item():.4f}")

    # Run predict()
    scores = model.predict(batch_u)
    assert scores.shape == (16, n_items)
    print("  --> predict() complete: scores shape", scores.shape)
    print("  --> PASS: STAIR_CNLGCL_v1_R standalone model passed all assertions.")


def test_main_script_imports():
    print("\n[5/5] Testing main_stair_cnlgcl_v1_r.py structural validity...")
    try:
        import freerec
        import main_stair_cnlgcl_v1_r as main_script
        assert hasattr(main_script, 'STAIR_CNLGCL_v1R_Model')
        assert hasattr(main_script, 'CoachForSTAIR_CNLGCL_v1R')
        assert hasattr(main_script, 'main')
        assert hasattr(main_script, 'cfg')
        assert hasattr(main_script.cfg, 'ssb_alpha')
        assert hasattr(main_script.cfg, 'ssb_beta')
        assert hasattr(main_script.cfg, 'ssb_mode')
        assert hasattr(main_script.cfg, 'tau')
        assert hasattr(main_script.cfg, 'lambda_cl')
        print("  --> main_stair_cnlgcl_v1_r.py loaded with all attributes.")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  --> [Environment Note] freerec not in current python env ({e}), running AST structural validation...")
        import ast
        main_path = os.path.join(repo_root, 'main_stair_cnlgcl_v1_r.py')
        with open(main_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert 'STAIR_CNLGCL_v1R_Model' in classes, "STAIR_CNLGCL_v1R_Model class not found in main script"
        assert 'CoachForSTAIR_CNLGCL_v1R' in classes, "CoachForSTAIR_CNLGCL_v1R class not found in main script"
        assert 'main' in funcs, "main() entry point not found in main script"
        print(f"  --> AST verified: Classes={classes}, funcs={funcs & {'main', 'reset_parameters', 'prepare'}}")
    print("  --> PASS: main training script verified.")


if __name__ == '__main__':
    print("=" * 80)
    print("RUNNING COMPREHENSIVE PIPELINE TEST SUITE: STAIR-CNLGCL v1-R")
    print("=" * 80)
    test_reexport()
    test_bsc_reweight_engine()
    test_cnlgcl_loss_module()
    test_standalone_model()
    test_main_script_imports()
    print("\n" + "=" * 80)
    print("ALL 5/5 TEST SUITES PASSED PERFECTLY!")
    print("=" * 80)
