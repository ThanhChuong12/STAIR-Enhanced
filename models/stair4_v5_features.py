"""Fixed modality preprocessing and train-only histories for STAIR-RAM.

This module intentionally has no FreeRec dependency. All fitted preprocessing
uses catalog features, never validation/test interactions. PCA belongs only to
the residual branch and does not replace the baseline MI transform.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import torch


def array_hash(value) -> str:
    """Use the repository convention: hash shape, dtype and contiguous bytes."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _identity_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ModalityFeatures:
    """CPU float32 features and fitted PCA state, padded to ``output_dim``.

    ``valid_mask`` describes raw input validity, not numerical PCA rank. Missing
    rows are exactly zero. Components after ``metadata['actual_rank']`` are zero.
    The metadata includes all preprocessing options and hashes of fitted tensors.
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_ratio: torch.Tensor
    metadata: dict

    def state_dict(self) -> dict:
        return {
            name: getattr(self, name) for name in (
                "values", "valid_mask", "mean", "components", "explained_variance",
                "explained_variance_ratio", "metadata",
            )
        }


def prepare_modality_features(
    raw: torch.Tensor | np.ndarray,
    *,
    output_dim: int = 128,
    algorithm: str = "randomized",
    seed: int = 1,
    oversampling: int = 16,
    n_iter: int = 4,
    item_ids=None,
    cache_path: str | Path | None = None,
    eps: float = 1e-8,
) -> ModalityFeatures:
    """Fit separate row-normalized, centered, non-whitened PCA features.

    Randomized SVD uses a private CPU generator and reorthogonalized power
    iteration, so it does not consume model/sampler RNG. ``exact`` uses thin SVD.
    Loading a cache fails closed on an identity or tensor-hash mismatch; callers
    must use a new cache path for changed data or preprocessing. Item IDs, if
    supplied, must confirm that rows are in internal catalog order 0..I-1.
    """
    if output_dim < 1 or oversampling < 0 or n_iter < 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid PCA dimension, iterations, oversampling or epsilon")
    if algorithm not in {"exact", "randomized"}:
        raise ValueError("PCA algorithm must be 'exact' or 'randomized'")
    raw_tensor = torch.as_tensor(raw)
    if raw_tensor.ndim != 2 or not all(raw_tensor.shape):
        raise ValueError("Raw modality features must have nonempty shape [items, features]")
    if raw_tensor.is_complex() or raw_tensor.dtype == torch.bool:
        raise ValueError("Raw modality features must be real-valued")
    raw_tensor = raw_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    num_items, input_dim = raw_tensor.shape
    expected_ids = np.arange(num_items, dtype=np.int64)
    mapping = expected_ids if item_ids is None else np.asarray(item_ids)
    if mapping.shape != expected_ids.shape or not np.array_equal(mapping, expected_ids):
        raise ValueError("Feature rows must follow internal item IDs 0..num_items-1")
    identity = {
        "schema_version": 1, "input_hash": array_hash(raw_tensor),
        "item_mapping_hash": array_hash(expected_ids), "input_shape": list(raw_tensor.shape),
        "output_dim": int(output_dim), "algorithm": algorithm, "seed": int(seed),
        "oversampling": int(oversampling), "power_iterations": int(n_iter), "eps": float(eps),
        "torch_version": str(torch.__version__), "dtype": "float32",
        "preprocessing_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cache_key = _identity_hash(identity)
    if cache_path is not None and Path(cache_path).is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("metadata", {}).get("cache_key") != cache_key:
            raise ValueError("PCA cache identity differs from the requested data or configuration")
        result = ModalityFeatures(**payload)
        for name, expected_hash in result.metadata["tensor_hashes"].items():
            if array_hash(getattr(result, name)) != expected_hash:
                raise ValueError(f"PCA cache integrity check failed for {name}")
        required_hashes = set(result.state_dict()) - {"metadata"}
        if set(result.metadata["tensor_hashes"]) != required_hashes:
            raise ValueError("PCA cache does not authenticate every fitted tensor")
        if result.values.shape != (num_items, output_dim) or result.components.shape != (input_dim, output_dim):
            raise ValueError("PCA cache tensor shapes differ from the requested configuration")
        return result

    finite = torch.isfinite(raw_tensor).all(dim=1)
    # Float64 norm prevents overflow for otherwise finite float32 input rows.
    norms = torch.linalg.vector_norm(raw_tensor.to(torch.float64), dim=1)
    valid = finite & torch.isfinite(norms) & (norms > eps)
    num_valid = int(valid.sum())
    if num_valid < 2:
        raise ValueError("PCA requires at least two finite, nonzero modality rows")
    normalized = (raw_tensor[valid].to(torch.float64) / norms[valid, None]).to(torch.float32)
    # Accumulate in float64 so identical rows center to exact zero instead of
    # amplifying float32 summation round-off into a spurious unit PCA direction.
    mean = normalized.to(torch.float64).mean(dim=0).to(torch.float32)
    centered = normalized - mean
    requested_rank = min(output_dim, input_dim, num_valid - 1)
    if algorithm == "exact":
        _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        sketch_dim = min(num_valid, input_dim, requested_rank + oversampling)
        omega = torch.randn(input_dim, sketch_dim, generator=generator)
        basis = torch.linalg.qr(centered @ omega, mode="reduced").Q
        for _ in range(n_iter):
            right_basis = torch.linalg.qr(centered.T @ basis, mode="reduced").Q
            basis = torch.linalg.qr(centered @ right_basis, mode="reduced").Q
        _, singular, right = torch.linalg.svd(basis.T @ centered, full_matrices=False)
    cutoff = float(singular[0]) * max(centered.shape) * torch.finfo(torch.float32).eps
    actual_rank = min(requested_rank, int((singular > cutoff).sum()))
    components = torch.zeros(input_dim, output_dim, dtype=torch.float32)
    if actual_rank:
        retained = right[:actual_rank].T.contiguous()
        pivots = retained.abs().argmax(dim=0)
        signs = retained[pivots, torch.arange(actual_rank)].sign()
        components[:, :actual_rank] = retained * signs
    projected = centered @ components
    projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(eps)
    values = torch.zeros(num_items, output_dim, dtype=torch.float32)
    values[valid] = projected
    variance = torch.zeros(output_dim, dtype=torch.float32)
    variance[:actual_rank] = singular[:actual_rank].square() / (num_valid - 1)
    total_variance = centered.square().sum() / (num_valid - 1)
    ratio = variance / total_variance.clamp_min(torch.finfo(torch.float32).tiny)
    metadata = {
        **identity, "cache_key": cache_key, "requested_rank": requested_rank,
        "actual_rank": actual_rank, "num_valid": num_valid, "num_missing": num_items - num_valid,
        "explained_variance_fraction": float(ratio.sum()), "canonical_sign": "largest_loading_positive",
    }
    result = ModalityFeatures(values, valid, mean, components, variance, ratio, metadata)
    metadata["tensor_hashes"] = {
        name: array_hash(value) for name, value in result.state_dict().items() if name != "metadata"
    }
    if cache_path is not None:
        _atomic_save(result.state_dict(), Path(cache_path))
    return result


def canonical_train_csr(train_csr: sp.spmatrix) -> sp.csr_matrix:
    """Validate and copy nonnegative train interactions into canonical binary CSR."""
    if not sp.issparse(train_csr) or len(train_csr.shape) != 2 or train_csr.shape[1] < 1:
        raise ValueError("Train interactions must be sparse with at least one item")
    csr = train_csr.tocsr(copy=True).astype(np.float32)
    if not np.isfinite(csr.data).all() or (csr.data < 0).any():
        raise ValueError("Train interaction weights must be finite and nonnegative")
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.data.fill(1.0)
    csr.sort_indices()
    return csr


def build_train_csr(edges, num_users: int, num_items: int) -> sp.csr_matrix:
    """Construct binary train CSR from internal IDs with shape [2, interactions]."""
    if num_users < 0 or num_items < 1:
        raise ValueError("Invalid user or item count")
    if isinstance(edges, torch.Tensor):
        edges = edges.detach().cpu().numpy()
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("Train edges must be integer IDs with shape [2, interactions]")
    edges = edges.astype(np.int64, copy=False)
    if edges.size and (edges.min() < 0 or edges[0].max() >= num_users or edges[1].max() >= num_items):
        raise ValueError("Train edge ID outside the dataset mapping")
    return canonical_train_csr(sp.csr_matrix(
        (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
        shape=(num_users, num_items),
    ))


def train_csr_hash(csr: sp.csr_matrix) -> str:
    """Fingerprint binary membership independently of SciPy's index dtype."""
    return _identity_hash({
        "shape": list(csr.shape), "indptr": array_hash(csr.indptr.astype(np.int64)),
        "indices": array_hash(csr.indices.astype(np.int64)),
    })


def checked_ids(ids, count: int, name: str) -> torch.Tensor:
    ids = torch.as_tensor(ids).detach().cpu()
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer IDs")
    ids = ids.reshape(-1).long()
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= count):
        raise ValueError(f"{name} ID outside the dataset mapping")
    return ids


class TrainHistory:
    """Fixed binary histories and feature sums; all query methods preserve rows.

    Only training interactions are accepted. ``loo`` rejects a positive outside
    the history and removes exactly one distinct item even if input train edges
    contained duplicates. Feature-less items still contribute to the degree.
    """

    def __init__(self, train_csr: sp.spmatrix, features: Mapping[str, torch.Tensor]):
        self.csr = canonical_train_csr(train_csr)
        self.num_users, self.num_items = self.csr.shape
        self.fingerprint = train_csr_hash(self.csr)
        if not features:
            raise ValueError("At least one modality is required")
        self.features: dict[str, torch.Tensor] = {}
        self.sums: dict[str, torch.Tensor] = {}
        self.degrees = torch.from_numpy(np.diff(self.csr.indptr).astype(np.int64))
        for name, value in features.items():
            value = torch.as_tensor(value)
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError("History features must be static tensors without autograd")
            if value.is_complex() or value.ndim != 2 or value.shape[0] != self.num_items or value.shape[1] < 1 or not torch.isfinite(value).all():
                raise ValueError(f"Invalid fixed modality features for {name}")
            value = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            self.features[name] = value
            self.sums[name] = torch.from_numpy(np.asarray(self.csr @ value.numpy(), dtype=np.float32))

    def to(self, device: torch.device | str) -> "TrainHistory":
        self.degrees = self.degrees.to(device)
        self.features = {name: value.to(device) for name, value in self.features.items()}
        self.sums = {name: value.to(device) for name, value in self.sums.items()}
        return self

    def full(self, users) -> dict[str, torch.Tensor]:
        users = checked_ids(users, self.num_users, "User").to(self.degrees.device)
        denominator = self.degrees[users].clamp_min(1).unsqueeze(1)
        return {name: value[users] / denominator for name, value in self.sums.items()}

    def loo(self, users, positives) -> dict[str, torch.Tensor]:
        users = checked_ids(users, self.num_users, "User")
        positives = checked_ids(positives, self.num_items, "Positive")
        if users.numel() != positives.numel():
            raise ValueError("Each user row must have exactly one positive item")
        if users.numel():
            # SciPy can set NumPy array WRITEABLE flags internally; torch-backed
            # array views do not support that operation on some NumPy versions.
            membership = np.asarray(self.csr[users.numpy().copy(), positives.numpy().copy()]).reshape(-1)
            if not (membership > 0).all():
                raise ValueError("LOO positive is not a member of the user's train history")
        users = users.to(self.degrees.device)
        positives = positives.to(self.degrees.device)
        remaining = self.degrees[users] - 1
        denominator = remaining.clamp_min(1).unsqueeze(1)
        return {
            name: torch.where(
                (remaining > 0).unsqueeze(1),
                (value[users] - self.features[name][positives]) / denominator,
                torch.zeros_like(value[users]),
            ) for name, value in self.sums.items()
        }
