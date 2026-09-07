# -*- coding: utf-8 -*-
"""
models/stair_sre_v7.py -- STAIR-SRE v1.1 Bridge Module
Provides direct export of v1.1 classes for external compatibility.
"""

from models.stair_sre_v6 import (
    DiagonalSpectralProjector,
    RegularizedDiagonalSpectralProjector,
    StepwiseSRELoss,
    StepwiseSREv2Loss,
    StepwiseSREv1_1Loss,
)

__all__ = [
    'DiagonalSpectralProjector',
    'RegularizedDiagonalSpectralProjector',
    'StepwiseSRELoss',
    'StepwiseSREv2Loss',
    'StepwiseSREv1_1Loss',
]
