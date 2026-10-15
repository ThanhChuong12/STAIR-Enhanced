"""STAIR5-v8.1: bounded soft-consensus queries on the unchanged V8 backbone."""

import copy
from pathlib import Path
import warnings

import torch

from models.stair5_v8 import STAIR5_v8_Model
from models.stair5_v8_1_consensus import ConsensusResidual, consensus_dose
from models.stair5_v8_1_graph import build_consensus_graph
from models.stair5_v8_1_objectives import NLGCLConsensus_Module
from models.stair5_v4_utils import file_sha256
from optimizers.stair5_v8_smoother import STAIR5V8Smoother
from stair5_v8_1_config import CONSENSUS_NAMES, validate_consensus_config


class STAIR5_v8_1_Model(STAIR5_v8_Model):
    """Add one auxiliary item query without changing encoding or ranking.

    Static consensus buffers are rebuilt and fingerprinted at construction.
    Fresh messages retain autograd only for the current optimization step.
    """

    def __init__(self, dataset, cfg):
        cfg = copy.copy(cfg)
        for name, value in validate_consensus_config(cfg).items():
            setattr(cfg, name, value)
        expected = .1 + .9 * (torch.arange(cfg.embedding_dim, dtype=torch.float32) / cfg.embedding_dim).pow(cfg.gamma)
        schedule = getattr(cfg, "beta3", None)
        if (not isinstance(schedule, torch.Tensor) or schedule.dtype != torch.float32
                or not torch.equal(schedule, expected.to(schedule.device))):
            raise ValueError("beta3 must match the native FP32 schedule for this embedding dimension")
        if cfg.consensus_mode != "off" and cfg.consensus_rho > 0 and not cfg.graph_cache_dir:
            cfg.graph_cache_dir = str(Path(getattr(cfg, "artifact_dir", ".cache")) / "graph_cache")
        super().__init__(dataset, cfg)
        self.nlgcl = NLGCLConsensus_Module(
            n_users=self.User.count, n_items=self.Item.count,
            G=self.cfg.nlgcl_G, tau=self.cfg.nlgcl_tau,
            alpha=self.cfg.nlgcl_alpha, chunk_size=self.cfg.cl_chunk_size,
        )
        self.consensus_residual = ConsensusResidual(
            gate=self.cfg.consensus_gate, hidden=self.cfg.consensus_hidden,
            eps=self.cfg.consensus_eps, seed=self.cfg.consensus_seed,
        )
        self.register_buffer("consensus_epoch_state", torch.zeros((), dtype=torch.int64))
        self.current_epoch = 0
        self._diagnostic_step = 0
        self.reset_consensus_diagnostics()
        extra_sources = (
            Path(__file__), Path(__file__).with_name("stair5_v8_1_graph.py"),
            Path(__file__).with_name("stair5_v8_1_consensus.py"),
            Path(__file__).with_name("stair5_v8_1_objectives.py"),
            Path(__file__).resolve().parents[1] / "main_stair5_v8_1.py",
            Path(__file__).resolve().parents[1] / "stair5_v8_1_config.py",
            Path(__file__).resolve().parents[1] / "main_stair6_v1.py",
        )
        self.source_manifest.update({p.name: file_sha256(p) for p in extra_sources if p.is_file()})

    @torch.no_grad()
    def get_knn_graph(self, features, k=5, cache_key=None):
        edges = super().get_knn_graph(features, k, cache_key)
        if getattr(self, "_capture_consensus_inputs", False):
            self._consensus_features.append(features)
        return edges

    @torch.no_grad()
    def prepare(self, path):
        """Observe parent's CPU features without repeating feature IO or kNN."""
        cfg = self.cfg
        self._capture_consensus_inputs = cfg.consensus_mode != "off" and cfg.consensus_rho > 0
        self._consensus_features = []
        try:
            super().prepare(path)
            self._prepare_consensus_graph()
        finally:
            self._capture_consensus_inputs = False
            del self._consensus_features

    def _prepare_consensus_graph(self):
        cfg = self.cfg
        edges = self.dataset.train().to_bigraph(edge_type="u2i")["u2i"].edge_index
        state = build_consensus_graph(
            train_edges=edges, n_users=self.User.count, n_items=self.Item.count,
            mfeats=self._consensus_features, cache_dir=cfg.graph_cache_dir or None,
            k_cf=cfg.k_cf, c_min=cfg.c_min, t_shrinkage=cfg.t_shrinkage,
            mode=cfg.consensus_mode if self._capture_consensus_inputs else "off", floor=cfg.consensus_floor,
            text_weight=cfg.consensus_text_weight, placebo_seed=cfg.consensus_placebo_seed,
            block_size=cfg.cf_block_size, memory_budget_mib=cfg.cf_memory_budget_mib,
            feature_memory_budget_mib=cfg.consensus_memory_budget_mib,
        )
        layout = cfg.consensus_sparse_layout
        operator = state.operator.to_sparse_coo().coalesce() if layout == "coo" else state.operator.to_sparse_csr()
        self.register_buffer("consensus_csr", operator, persistent=False)
        self.register_buffer("consensus_coverage", state.coverage, persistent=False)
        self.consensus_metadata = state.metadata
        self.consensus_is_empty = operator._nnz() == 0
        self.consensus_fingerprint = state.metadata["graph_fingerprint"]
        self.consensus_backend = {"requested": layout, "resolved": "coo" if layout == "coo" else "csr",
                                  "verified": False, "fallback_reason": None}

    @staticmethod
    def _probe_sparse_backward(layout, device):
        """Probe a tiny same-dtype kernel once, without consuming sampler RNG."""
        with torch.enable_grad():
            operator = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [1, 0]], device=device),
                torch.ones(2, device=device), (2, 2)).coalesce()
            if layout == "csr":
                operator = operator.to_sparse_csr()
            x = torch.ones((2, 3), device=device, requires_grad=True)
            torch.sparse.mm(operator, x).sum().backward()
            if x.grad is None or not torch.equal(x.grad, torch.ones_like(x)):
                raise RuntimeError("Sparse SpMM dense-input backward returned incorrect gradients")

    def configure_consensus_backend(self):
        """Resolve auto CSR/COO on the final device before optimizer/resume.

        Only an unsupported sparse kernel can trigger fallback. OOM and other
        failures propagate; switching layout must never conceal those errors.
        """
        if self.consensus_is_empty:
            return
        requested = self.cfg.consensus_sparse_layout
        resolved = "coo" if requested == "coo" else "csr"
        reason = None
        try:
            self._probe_sparse_backward(resolved, self.consensus_csr.device)
        except (RuntimeError, NotImplementedError) as error:
            message = str(error).lower()
            unsupported = isinstance(error, NotImplementedError) or any(
                phrase in message for phrase in ("not implemented", "not supported", "unsupported", "could not run"))
            if requested != "auto" or not unsupported or "out of memory" in message:
                raise
            resolved, reason = "coo", str(error)
            self._probe_sparse_backward(resolved, self.consensus_csr.device)
            warnings.warn(f"Consensus CSR backward unavailable; using COO: {reason}", RuntimeWarning)
        self.consensus_csr = (self.consensus_csr.to_sparse_coo().coalesce() if resolved == "coo"
                              else self.consensus_csr.to_sparse_csr())
        self.consensus_backend.update(resolved=resolved, verified=True, fallback_reason=reason)

    @property
    def current_epoch(self):
        return self._current_consensus_epoch

    @current_epoch.setter
    def current_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Completed/current epoch must be a nonnegative integer")
        self._current_consensus_epoch = epoch
        if hasattr(self, "consensus_epoch_state"):
            self.consensus_epoch_state.fill_(epoch)

    @property
    def consensus_rho(self):
        if self.current_epoch == 0 or self.cfg.consensus_mode == "off":
            return 0.0
        return consensus_dose(
            self.current_epoch, rho_max=self.cfg.consensus_rho,
            warmup_epochs=self.cfg.consensus_warmup, ramp_epochs=self.cfg.consensus_ramp,
        )

    def marked_params(self):
        """Three isolated groups; gate parameters never enter the item smoother."""
        groups = [
            {"params": list(self.User.parameters()), "smoother": None},
            {"params": list(self.Item.parameters()), "smoother": STAIR5V8Smoother(
                lambda x: torch.sparse.mm(self.mAdj, x), beta=self.beta3, L=self.num_layers, omega=0.0,
            )},
        ]
        auxiliary = list(self.consensus_residual.parameters())
        if auxiliary:
            groups.append({"params": auxiliary, "smoother": None,
                           "lr": self.cfg.lr * self.cfg.lr_aux_scale, "weight_decay": 0.0})
        ids = [id(p) for group in groups for p in group["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise RuntimeError("Every trainable parameter must occur in exactly one optimizer group")
        return groups

    def reset_consensus_diagnostics(self):
        self._diagnostic_sums = {}
        self._diagnostic_count = 0
        self._diagnostic_step = 0

    def consensus_diagnostics(self):
        """Pool sampled unique-row moments; maxima are not batch averages."""
        if not self._diagnostic_sums:
            return {}
        count = self._diagnostic_sums["count"]
        result = {name: value if name in ("count", "residual_ratio_max") else value / count.clamp_min(1)
                  for name, value in self._diagnostic_sums.items()}
        result["gate_std"] = (result["gate_second_moment"] - result["gate_mean"].square()).clamp_min(0).sqrt()
        return result

    def _accumulate_consensus_diagnostics(self, diagnostics):
        count = diagnostics["count"].detach()
        for name, raw in diagnostics.items():
            if name == "gate_std":
                continue
            value = raw.detach()
            if name not in ("count", "residual_ratio_max"):
                value = value * count
            if name == "residual_ratio_max" and name in self._diagnostic_sums:
                value = torch.maximum(self._diagnostic_sums[name], value)
            else:
                value = self._diagnostic_sums.get(name, 0) + value
            self._diagnostic_sums[name] = value
        self._diagnostic_count += 1

    def training_objective(self, data):
        rho = self.consensus_rho
        if rho == 0.0 or self.consensus_is_empty or not self.training or self.lambda_nlgcl <= 0:
            return super().training_objective(data)
        user_emb, item_emb, layers = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        u_idx = users.unsqueeze(1) if users.ndim == 1 else users
        p_idx = positives.unsqueeze(1) if positives.ndim == 1 else positives
        n_idx = negatives.unsqueeze(1) if negatives.ndim == 1 else negatives
        rec = self.criterion(
            torch.einsum("BKD,BKD->BK", user_emb[u_idx], item_emb[p_idx]),
            torch.einsum("BKD,BKD->BK", user_emb[u_idx], item_emb[n_idx]),
        )
        _, v0 = torch.split(layers[0], (self.User.count, self.Item.count))
        _, i1 = torch.split(layers[1], (self.User.count, self.Item.count))
        messages = torch.sparse.mm(self.consensus_csr, v0)
        unique_items, inverse = torch.unique(positives.reshape(-1), return_inverse=True)
        x, message = i1[unique_items], messages[unique_items]
        coverage = self.consensus_coverage[unique_items]
        interval = self.cfg.consensus_diagnostic_interval
        sampled = interval > 0 and self._diagnostic_step % interval == 0
        if sampled:
            query, diagnostics = self.consensus_residual.forward_with_diagnostics(x, message, coverage, rho)
            self._accumulate_consensus_diagnostics(diagnostics)
        else:
            query = self.consensus_residual(x, message, coverage, rho)
        self._diagnostic_step += 1
        cl = self.nlgcl(layers, users, positives, item_query=query[inverse])
        return rec + self.lambda_nlgcl * cl, rec, cl

    def get_extra_state(self):
        contract = super().get_extra_state()
        return {
            "architecture": "STAIR5-v8.1-BSCV", "version": 1, "backbone": contract,
            "consensus_config": {name: getattr(self.cfg, name) for name in CONSENSUS_NAMES},
            "consensus_fingerprint": self.consensus_fingerprint,
            "consensus_layout": self.consensus_backend["resolved"],
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        epoch = state_dict.get("consensus_epoch_state")
        if not isinstance(epoch, torch.Tensor) or epoch.ndim != 0 or epoch.dtype != torch.int64 or epoch.item() < 0:
            raise ValueError("Checkpoint must contain a valid consensus epoch before model state can be loaded")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.current_epoch = int(epoch.item())
        self.reset_consensus_diagnostics()
        return result
