# -*- coding: utf-8 -*-
"""
models/stair_cnlgcl_v1_r.py
============================
STAIR-CNLGCL v1-R Top-Level Shim (Giai đoạn 4).
Re-exports from models.GD4.stair_cnlgcl_v1_r for flat namespace compatibility.
"""

from .GD4.stair_cnlgcl_v1_r import (
    BSC_Reweight_Engine,
    CNLGCL_Loss_v1R,
    STAIR_CNLGCL_v1_R,
)

__all__ = ['BSC_Reweight_Engine', 'CNLGCL_Loss_v1R', 'STAIR_CNLGCL_v1_R']
