"""Bounded, separable multimodal ranking head for STAIR-RAM (v5).

All supplied embeddings and features are fixed inputs. The head detaches them
at its boundary while recomputing trainable projections on every invocation.
No projected-item or user-query autograd graph is stored in this module.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualHead(nn.Module):
    """Implement report equations 4--10 without changing the baseline scorer.

    ``score`` accepts either one item per user or a row-specific candidate
    matrix. ``evaluation_vectors`` uses full train profiles and returns vectors
    whose matrix product produces the same final scores. The caller is
    responsible for constructing correct train-only, leave-positive-out profiles.
    """

    def __init__(
        self,
        embedding_dim: int,
        feature_dims: Mapping[str, int],
        residual_dim: int = 32,
        *,
        eta: float,
        max_log_degree: float,
        gate_mode: str = "personalized",
        fixed_weights: Sequence[float] | None = None,
        initial_amplitude_fraction: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or residual_dim <= 0:
            raise ValueError("Embedding and residual dimensions must be positive.")
        if not feature_dims or any(dim <= 0 for dim in feature_dims.values()):
            raise ValueError("At least one positive modality dimension is required.")
        if any(not name or "." in name for name in feature_dims):
            raise ValueError("Modality names must be nonempty and cannot contain '.'.")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive.")
        if not math.isfinite(max_log_degree) or max_log_degree < 0:
            raise ValueError("max_log_degree must be finite and nonnegative.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        if not 0 < abs(initial_amplitude_fraction) < 1:
            raise ValueError("Use a nonzero initial amplitude fraction inside (-1, 1).")
        if gate_mode not in {"personalized", "global", "fixed"}:
            raise ValueError("gate_mode must be personalized, global, or fixed.")
        if fixed_weights is not None and gate_mode != "fixed":
            raise ValueError("fixed_weights is only meaningful for a fixed gate.")

        self.embedding_dim = embedding_dim
        self.feature_dims = dict(feature_dims)
        self.modalities = tuple(feature_dims)
        self.residual_dim = residual_dim
        self.gate_mode = gate_mode
        self.eps = eps
        self.register_buffer("eta", torch.tensor(float(eta)))
        self.register_buffer("max_log_degree", torch.tensor(float(max_log_degree)))
        self.theta = nn.Parameter(torch.tensor(math.atanh(initial_amplitude_fraction)))
        self.user_projectors = nn.ModuleDict({
            name: nn.Linear(embedding_dim, residual_dim, bias=False)
            for name in self.modalities
        })
        self.profile_projectors = nn.ModuleDict({
            name: nn.Linear(dim, residual_dim, bias=False)
            for name, dim in self.feature_dims.items()
        })
        self.item_projectors = nn.ModuleDict({
            name: nn.Linear(dim, residual_dim, bias=False)
            for name, dim in self.feature_dims.items()
        })
        count = len(self.modalities)
        self.gate = nn.Linear(embedding_dim + 1, count) if gate_mode == "personalized" else None
        self.register_parameter(
            "global_logits", nn.Parameter(torch.zeros(count)) if gate_mode == "global" else None
        )
        weights = torch.ones(count) if fixed_weights is None else torch.as_tensor(fixed_weights, dtype=torch.float32)
        if weights.shape != (count,) or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Fixed gate weights must be finite, nonnegative, and have positive sum.")
        self.register_buffer("fixed_weights", weights / weights.sum())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Copy initial projector values without sharing Parameter instances."""
        for name in self.modalities:
            nn.init.orthogonal_(self.item_projectors[name].weight)
            with torch.no_grad():
                self.profile_projectors[name].weight.copy_(self.item_projectors[name].weight)
            nn.init.xavier_uniform_(self.user_projectors[name].weight, gain=0.1)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        if self.global_logits is not None:
            nn.init.zeros_(self.global_logits)

    @property
    def amplitude(self) -> Tensor:
        return self.eta * self.theta.tanh()

    def gate_weights(self, user_embeddings: Tensor, degrees: Tensor) -> Tensor:
        """Return user-only simplex weights; degree normalization is fixed."""
        users = self._users(user_embeddings)
        degrees = degrees.detach().reshape(-1)
        if degrees.shape != users.shape[:1]:
            raise ValueError("degrees must contain exactly one value per user row.")
        if self.gate_mode == "fixed":
            return self.fixed_weights.expand(users.size(0), -1)
        if self.gate_mode == "global":
            return self.global_logits.softmax(dim=-1).expand(users.size(0), -1)
        degree_feature = degrees.to(dtype=users.dtype).log1p() / (self.max_log_degree + self.eps)
        gate_input = torch.cat((F.normalize(users, dim=-1, eps=self.eps), degree_feature[:, None]), dim=-1)
        return self.gate(gate_input).softmax(dim=-1)

    def _users(self, user_embeddings: Tensor) -> Tensor:
        if user_embeddings.ndim != 2 or user_embeddings.size(-1) != self.embedding_dim:
            raise ValueError("user_embeddings must have shape [batch, embedding_dim].")
        return user_embeddings.detach()

    def _features(self, features: Mapping[str, Tensor]) -> None:
        if set(features) != set(self.modalities):
            raise ValueError("Feature mappings must contain exactly the configured modalities.")
        for name, dim in self.feature_dims.items():
            if features[name].size(-1) != dim:
                raise ValueError(f"Incorrect feature width for modality {name!r}.")

    def encode_queries(
        self, user_embeddings: Tensor, profiles: Mapping[str, Tensor], degrees: Tensor
    ) -> tuple[dict[str, Tensor], Tensor]:
        users = self._users(user_embeddings)
        self._features(profiles)
        users_normalized = F.normalize(users, dim=-1, eps=self.eps)
        queries = {}
        for name in self.modalities:
            if profiles[name].shape != (users.size(0), self.feature_dims[name]):
                raise ValueError("Each profile must have shape [batch, modality_dim].")
            projected = self.user_projectors[name](users_normalized) + self.profile_projectors[name](profiles[name].detach())
            queries[name] = F.normalize(projected, dim=-1, eps=self.eps)
        return queries, self.gate_weights(users, degrees)

    def encode_items(self, item_features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Project fresh keys; zero/missing features remain exactly zero."""
        self._features(item_features)
        return {
            name: F.normalize(self.item_projectors[name](item_features[name].detach()), dim=-1, eps=self.eps)
            for name in self.modalities
        }

    def residual_scores(
        self, queries: Mapping[str, Tensor], keys: Mapping[str, Tensor], weights: Tensor
    ) -> Tensor:
        """Score paired [B,r] keys or row-specific [B,K,r] candidates."""
        result = None
        for index, name in enumerate(self.modalities):
            query, key = queries[name], keys[name]
            if key.ndim == 2:
                if key.shape != query.shape:
                    raise ValueError("Paired keys must have the same shape as queries.")
                term = weights[:, index] * (query * key).sum(dim=-1)
            elif key.ndim == 3:
                if key.size(0) != query.size(0) or key.size(-1) != query.size(-1):
                    raise ValueError("Candidate keys must have shape [batch, candidates, residual_dim].")
                term = weights[:, index, None] * (query[:, None, :] * key).sum(dim=-1)
            else:
                raise ValueError("Keys must be paired or row-specific candidate tensors.")
            result = term if result is None else result + term
        return self.amplitude * result

    def score(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
        profiles: Mapping[str, Tensor],
        item_features: Mapping[str, Tensor],
        degrees: Tensor,
        *,
        enabled: bool = True,
    ) -> Tensor:
        users = self._users(user_embeddings)
        items = item_embeddings.detach()
        if items.ndim == 2 and items.shape == users.shape:
            baseline = (users * items).sum(dim=-1)
        elif items.ndim == 3 and items.size(0) == users.size(0) and items.size(-1) == self.embedding_dim:
            baseline = (users[:, None, :] * items).sum(dim=-1)
        else:
            raise ValueError("Item embeddings must have shape [batch, dim] or [batch, candidates, dim].")
        if not enabled:
            return baseline
        queries, weights = self.encode_queries(users, profiles, degrees)
        keys = self.encode_items(item_features)
        residual = self.residual_scores(queries, keys, weights)
        if residual.shape != baseline.shape:
            raise ValueError("Item features and embeddings must identify the same candidate slots.")
        return baseline + residual

    def forward(self, *args, **kwargs) -> Tensor:
        return self.score(*args, **kwargs)

    def evaluation_vectors(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
        profiles: Mapping[str, Tensor],
        item_features: Mapping[str, Tensor],
        degrees: Tensor,
        *,
        enabled: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Build separable vectors; call under no_grad for evaluation caching.

        Gate-0 returns the original-width tensors so callers can use exactly
        the original baseline GEMM, without zero-column rounding differences.
        """
        users = self._users(user_embeddings)
        items = item_embeddings.detach()
        if items.ndim != 2 or items.size(-1) != self.embedding_dim:
            raise ValueError("Evaluation items must have shape [num_items, embedding_dim].")
        if not enabled:
            return users, items
        queries, weights = self.encode_queries(users, profiles, degrees)
        keys = self.encode_items(item_features)
        if any(key.shape != (items.size(0), self.residual_dim) for key in keys.values()):
            raise ValueError("Evaluation features must contain one row per catalog item.")
        user_parts = [users] + [
            self.amplitude * weights[:, index, None] * queries[name]
            for index, name in enumerate(self.modalities)
        ]
        return torch.cat(user_parts, dim=-1), torch.cat([items, *keys.values()], dim=-1)

    def parameter_groups(
        self,
        projection_lr: float = 1e-3,
        gate_lr: float = 1e-3,
        amplitude_lr: float = 1e-4,
        projection_weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Return disjoint AdamW groups; no BSC smoother belongs in Stage B."""
        if any(not math.isfinite(lr) or lr <= 0 for lr in (projection_lr, gate_lr, amplitude_lr)):
            raise ValueError("All learning rates must be finite and positive.")
        if not math.isfinite(projection_weight_decay) or projection_weight_decay < 0:
            raise ValueError("Projection weight decay must be finite and nonnegative.")
        projectors = [*self.user_projectors.parameters(), *self.profile_projectors.parameters(), *self.item_projectors.parameters()]
        gate_parameters = list(self.gate.parameters()) if self.gate is not None else []
        if self.global_logits is not None:
            gate_parameters.append(self.global_logits)
        groups = [dict(name="projection", params=projectors, lr=projection_lr, weight_decay=projection_weight_decay)]
        if gate_parameters:
            groups.append(dict(name="gate", params=gate_parameters, lr=gate_lr, weight_decay=0.0))
        groups.append(dict(name="amplitude", params=[self.theta], lr=amplitude_lr, weight_decay=0.0))
        identifiers = [id(parameter) for group in groups for parameter in group["params"]]
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != {id(p) for p in self.parameters()}:
            raise RuntimeError("Every head parameter must appear in exactly one optimizer group.")
        return groups

    @torch.no_grad()
    def diagnostics(self, user_embeddings: Tensor | None = None, degrees: Tensor | None = None) -> dict[str, float]:
        """Return detached scalars, including gradient norms before zero_grad."""
        result = {"amplitude": self.amplitude.item(), "eta": self.eta.item(), "amplitude_fraction": self.theta.tanh().item()}
        for group in self.parameter_groups():
            gradients = [p.grad.detach().square().sum() for p in group["params"] if p.grad is not None]
            result[f"gradient_norm/{group['name']}"] = torch.stack(gradients).sum().sqrt().item() if gradients else 0.0
        if (user_embeddings is None) != (degrees is None):
            raise ValueError("Supply both user_embeddings and degrees for gate diagnostics.")
        if user_embeddings is not None and user_embeddings.size(0):
            weights = self.gate_weights(user_embeddings, degrees)
            result["gate_entropy"] = -(weights * weights.clamp_min(self.eps).log()).sum(dim=-1).mean().item()
            for index, name in enumerate(self.modalities):
                result[f"gate_mean/{name}"] = weights[:, index].mean().item()
        return result

