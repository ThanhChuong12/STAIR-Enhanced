"""Capacity-only 256D specialization of STAIR5-v8 / WMSG-CSE.

No new layers, loss terms, graphs or inference operations are introduced.
Numerical model methods are inherited to keep same-dimension parity testable.
"""

from pathlib import Path
from typing import Any

import torch

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v4_utils import file_sha256
from stair5_v8_256d_config import (
    CONTRACT_VERSION,
    EMBEDDING_DIM,
    EXPERIMENT_ID,
    validate_capacity_config,
)


class STAIR5_v8_256D_Model(STAIR5_v8_Model):
    """The unchanged WMSG-core model, guarded to exactly 256 coordinates."""

    def __init__(self, dataset: Any, cfg: Any) -> None:
        validate_capacity_config(cfg)
        coefficients = getattr(cfg, "beta3", None)
        if not isinstance(coefficients, torch.Tensor) or coefficients.shape != (EMBEDDING_DIM,):
            raise ValueError("beta3 must contain the 256-coordinate reference BSC schedule")
        expected = 0.1 + 0.9 * (
            torch.arange(EMBEDDING_DIM, dtype=torch.float32)
            / EMBEDDING_DIM
        ).pow(cfg.gamma)
        # The reference parser computes on CPU before transfer. Computing pow
        # on CUDA here can differ by rounding and spuriously reject its schedule.
        expected = expected.to(coefficients.device)
        if coefficients.dtype != torch.float32 or not torch.equal(coefficients, expected):
            raise ValueError("beta3 must match the reference FP32 coordinate schedule exactly")
        super().__init__(dataset, cfg)

        # Keep inherited source hashes, including the reference engine. Include
        # the new specialization/launcher as well, so resume cannot silently
        # mix capacity experiments or revisions of this wrapper.
        root = Path(__file__).resolve().parents[1]
        for source in (
            Path(__file__),
            root / "stair5_v8_256d_config.py",
            root / "main_stair5_v8_256d.py",
        ):
            self.source_manifest[source.name] = file_sha256(source)

    def get_extra_state(self) -> dict:
        state = super().get_extra_state()
        state["capacity_experiment"] = {
            "id": EXPERIMENT_ID,
            "contract_version": CONTRACT_VERSION,
            "embedding_dim": EMBEDDING_DIM,
            "consensus": False,
        }
        return state
