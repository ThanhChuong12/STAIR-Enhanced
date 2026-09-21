"""FreeRec STAIR-MHD v3; baseline MI/FSC/ranking with step-local auxiliaries.

The gate receives gradients from InfoNCE, never through AdamWSEvo's step.
See docs/giai_doan_4/v3.md for the method and its empirical limitations.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import freerec
import torch
from torch import nn
from torch.nn import functional as F

from models.stair_mhd_v3_utils import (
    apply_P_H, behavioral_statistics, build_incidence, exact_knn, graph_state,
)
from optimizers.mhd_smoother import MHDSmoother


@dataclass(frozen=True)
class MHDOptions:
    knn_block_size: int = 256
    gate_hidden_dim: int = 16
    gate_floor: float = 0.05
    gate_prior: float = 0.8
    behavior_support_s: float = 10.0
    behavior_eta: float = 1.0
    modality_weights: tuple = (5 / 6, 1 / 6)
    learn_modality_weights: bool = False
    bsc_mix_target: float = 0.25
    cl_weight_target: float = 0.001
    cl_temperature: float = 0.2
    budget_weight_target: float = 0.0001
    warmup_epochs: int = 10
    ramp_epochs: int = 20
    aux_lr_ratio: float = 0.1
    aux_weight_decay: float = 0.0
    gate_mode: str = 'soft'
    gumbel_temperature: float = 0.5
    graph_weight_refresh: str = 'every_step'
    projector: str = 'shared_linear_identity_init'

    def __post_init__(self):
        if not 0 < self.gate_floor < 1 or not 0 < self.gate_prior < 1:
            raise ValueError('gate_floor and gate_prior must be in (0, 1)')
        if not 0 <= self.bsc_mix_target <= 1:
            raise ValueError('bsc_mix_target must be in [0, 1]')
        if self.cl_temperature <= 0 or self.gumbel_temperature <= 0:
            raise ValueError('temperatures must be positive')
        if self.warmup_epochs < 0 or self.ramp_epochs < 0 or self.gate_hidden_dim < 1:
            raise ValueError('invalid warmup/ramp/hidden size')
        if self.cl_weight_target < 0 or self.budget_weight_target < 0 or self.behavior_eta < 0:
            raise ValueError('loss weights and behavior_eta must be nonnegative')
        if self.aux_lr_ratio != 0.1 or self.aux_weight_decay != 0:
            raise ValueError('v3 requires auxiliary lr = 0.1 * baseline lr and zero decay')
        if self.graph_weight_refresh != 'every_step':
            raise ValueError('primary v3 requires fresh graph weights every step')
        if self.gate_mode not in ('soft', 'concrete', 'straight_through'):
            raise ValueError('unknown gate_mode')
        if self.projector not in ('shared_linear_identity_init', 'identity'):
            raise ValueError('unknown projector')
        if len(self.modality_weights) != 2 or min(self.modality_weights) < 0:
            raise ValueError('two nonnegative modality_weights required')
        if not math.isclose(sum(self.modality_weights), 1.0, abs_tol=1e-7):
            raise ValueError('modality_weights must sum to one')
        if self.learn_modality_weights and min(self.modality_weights) <= 0:
            raise ValueError('learned alpha requires strictly positive initialization')
        if any(isinstance(v, float) and not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('options must be finite')

    @classmethod
    def from_config(cls, cfg):
        args = {key: getattr(cfg, key) for key in cls.__dataclass_fields__ if hasattr(cfg, key)}
        if 'modality_weights' in args:
            value = args['modality_weights']
            args['modality_weights'] = tuple(map(float, value.split(','))) if isinstance(value, str) else tuple(value)
        return cls(**args)


def symmetric_infonce(first, second, item_ids, projector, temperature):
    """Deduplicate IDs before selecting either view; singleton loss is zero."""
    ids = torch.unique(item_ids.reshape(-1))
    if ids.numel() < 2:
        return (first.sum() + second.sum()) * 0.0
    u = F.normalize(projector(first[ids]), dim=-1)
    v = F.normalize(projector(second[ids]), dim=-1)
    logits = u @ v.T / temperature
    labels = torch.arange(ids.numel(), device=ids.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def _tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(tensor.shape), tensor.dtype)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class STAIR_MHD_v3(freerec.models.GenRecArch):
    def __init__(self, dataset, cfg):
        super().__init__(dataset)
        self.cfg = cfg
        self.options = MHDOptions.from_config(cfg)
        self.num_layers = int(cfg.num_layers)
        self.embedding_dim = int(cfg.embedding_dim)
        if self.num_layers < 0 or self.embedding_dim < 1 or cfg.gamma <= 0:
            raise ValueError('invalid baseline layer count, dimension, or gamma')
        self.current_epoch = 0
        self.User.add_module('embeddings', nn.Embedding(self.User.count, self.embedding_dim))
        self.Item.add_module('embeddings', nn.Embedding(self.Item.count, self.embedding_dim))
        self.register_buffer('Adj', dataset.train().to_normalized_adj(normalization='sym'))
        self.register_buffer('beta3', 0.1 + 0.9 * (
            torch.arange(self.embedding_dim) / self.embedding_dim).pow(cfg.gamma))
        self.reset_parameters()
        self.prepare(dataset.path)
        # Auxiliary initialization must not consume the baseline sampling RNG.
        with torch.random.fork_rng(devices=[]):
            self.gates = nn.ModuleList([
                nn.Sequential(nn.Linear(4, self.options.gate_hidden_dim), nn.GELU(),
                              nn.Linear(self.options.gate_hidden_dim, 1)) for _ in range(2)
            ])
            for gate in self.gates:
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, math.log(self.options.gate_prior / (1 - self.options.gate_prior)))
            if self.options.projector == 'identity':
                self.projector = nn.Identity()
            else:
                self.projector = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
                nn.init.eye_(self.projector.weight)
        alpha = torch.tensor(self.options.modality_weights, dtype=torch.float32)
        self.register_buffer('fixed_alpha', alpha)
        if self.options.learn_modality_weights:
            self.alpha_logits = nn.Parameter(alpha.log())
        else:
            self.register_parameter('alpha_logits', None)
        self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        self.smoother = MHDSmoother(self.apply_bsc_operator, self.beta3, self.num_layers)
        self.last_diagnostics = {}

    def reset_parameters(self):
        # Same initialization order and arithmetic as main.py, before auxiliaries.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=1.e-4)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1.)
                nn.init.constant_(module.bias, 0.)

    def whitening(self, feats):
        feats = torch.as_tensor(feats, dtype=torch.float32)
        if min(feats.shape) < self.embedding_dim:
            raise ValueError('MI requires embedding_dim <= min(item count, modality width)')
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :self.embedding_dim] * math.sqrt(self.Item.count / self.embedding_dim)

    @torch.no_grad()
    def prepare(self, path):
        """Build only static, train-only buffers and baseline MI."""
        mfiles = self.cfg.mfiles.split(',') if isinstance(self.cfg.mfiles, str) else self.cfg.mfiles
        ks = list(map(int, self.cfg.num_neighbors.split('-'))) if isinstance(self.cfg.num_neighbors, str) else list(self.cfg.num_neighbors)
        if len(mfiles) != 2 or len(ks) != 2 or min(ks) < 1:
            raise ValueError('v3 requires exactly two modalities and positive neighbor counts')
        mfeats = []
        for mfile in mfiles:
            candidates = [Path(path) / mfile, Path(self.cfg.root) / self.cfg.dataset / mfile,
                          Path('/kaggle/data') / self.cfg.dataset / mfile,
                          Path('/kaggle/working/STAIR/data') / self.cfg.dataset / mfile,
                          Path('/kaggle/working/STAIR-Enhanced/data') / self.cfg.dataset / mfile,
                          Path('data') / self.cfg.dataset / mfile]
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(f'missing modality {mfile} for dataset {self.cfg.dataset}')
            features = torch.as_tensor(freerec.utils.import_pickle(str(resolved)), dtype=torch.float32)
            if features.shape[0] != self.Item.count:
                raise ValueError('feature rows must match item ID mapping')
            mfeats.append(features)
        edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        canonical = torch.unique(edge_index.detach().cpu().T, dim=0).T.contiguous()
        self.data_hashes = {'train': _tensor_hash(canonical)}
        directed = []
        for m, (features, k) in enumerate(zip(mfeats, ks)):
            neighbors = exact_knn(features, k, self.options.knn_block_size)
            self.register_buffer(f'neighbors_{m}', neighbors)
            self.register_buffer(f'incidence_{m}', build_incidence(neighbors))
            C, rho, x = behavioral_statistics(canonical, self.User.count, self.Item.count,
                                             neighbors, features, self.options.behavior_support_s)
            for key, value in [('coherence', C), ('support', rho), ('edge_features', x)]:
                self.register_buffer(f'{key}_{m}', value)
            centers = torch.arange(self.Item.count)[:, None].expand_as(neighbors)
            directed.append(torch.stack((centers.reshape(-1), neighbors.reshape(-1))))
            self.data_hashes[f'features_{m}'] = _tensor_hash(features)
            self.data_hashes[f'neighbors_{m}'] = _tensor_hash(neighbors)
        # The exact coalesce/symmetrization/normalization chain in main.py.
        indices = torch.cat(directed, dim=1)
        weights = torch.ones_like(indices[0], dtype=torch.float)
        indices, weights = freerec.graph.coalesce(indices, weights, reduce='sum')
        indices, weights = freerec.graph.to_undirected(indices, weights, reduce='max')
        indices, weights = freerec.graph.to_normalized(indices, weights, normalization='sym')
        self.register_buffer('mAdj', torch.sparse_coo_tensor(
            indices, weights, size=(self.Item.count, self.Item.count)).to_sparse_csr())
        mi = sum(self.whitening(f) * k for f, k in zip(mfeats, ks)).div(sum(ks))
        self.Item.embeddings.weight.copy_(mi)
        r_indices, r_weights = freerec.graph.to_normalized(edge_index, normalization='left')
        R = torch.sparse_coo_tensor(r_indices, r_weights,
                                   size=(self.User.count, self.Item.count)).to_sparse_csr()
        self.User.embeddings.weight.copy_(R @ mi)
        if any(value.grad_fn is not None or value.requires_grad for value in self.buffers()):
            raise RuntimeError('prepare created a non-static buffer')

    @property
    def incidences(self):
        return self.incidence_0, self.incidence_1

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def coefficients(self):
        elapsed = self.current_epoch - self.options.warmup_epochs
        ramp = min(1.0, max(0.0, elapsed / max(1, self.options.ramp_epochs)))
        return (ramp * self.options.cl_weight_target,
                ramp * self.options.budget_weight_target,
                ramp * self.options.bsc_mix_target)

    def fresh_graph_state(self, stochastic=None):
        """Never store this live autograd object on the model."""
        if stochastic is None:
            stochastic = self.training
        states, probabilities = [], []
        for m, gate in enumerate(self.gates):
            logits = gate(getattr(self, f'edge_features_{m}')).squeeze(-1)
            logits = logits + self.options.behavior_eta * getattr(self, f'support_{m}') * getattr(self, f'coherence_{m}')
            p = logits.sigmoid()
            probabilities.append(p)
            mask = p
            if self.options.gate_mode != 'soft' and stochastic:
                uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
                noise = uniform.log() - torch.log1p(-uniform)
                soft = ((logits + noise) / self.options.gumbel_temperature).sigmoid()
                mask = soft if self.options.gate_mode == 'concrete' else (
                    (soft > 0.5).to(soft.dtype) - soft).detach() + soft
            w = self.options.gate_floor + (1 - self.options.gate_floor) * mask
            states.append(graph_state(self.incidences[m], w))
        alpha = self.fixed_alpha if self.alpha_logits is None else self.alpha_logits.softmax(0)
        return {'states': tuple(states), 'alpha': alpha, 'probabilities': tuple(probabilities)}

    def apply_bsc_operator(self, features, snapshot):
        base = self.mAdj @ features
        if snapshot is None or snapshot['zeta'] == 0:
            return base
        hyper = apply_P_H(features, self.incidences, snapshot['states'], snapshot['alpha'])
        return (1 - snapshot['zeta']) * base + snapshot['zeta'] * hyper

    def marked_params(self):
        users, items = list(self.User.parameters()), list(self.Item.parameters())
        backbone_ids = {id(p) for p in users + items}
        auxiliary = [p for p in self.parameters() if id(p) not in backbone_ids]
        groups = [
            {'params': users, 'smoother': None, 'lr': self.cfg.lr,
             'weight_decay': self.cfg.weight_decay, 'role': 'users'},
            {'params': items, 'smoother': self.smoother, 'lr': self.cfg.lr,
             'weight_decay': self.cfg.weight_decay, 'role': 'items'},
            {'params': auxiliary, 'smoother': None, 'lr': 0.1 * self.cfg.lr,
             'weight_decay': 0.0, 'role': 'auxiliary'},
        ]
        seen = [id(p) for group in groups for p in group['params']]
        if len(seen) != len(set(seen)) or set(seen) != {id(p) for p in self.parameters()}:
            raise RuntimeError('each model parameter must appear in exactly one optimizer group')
        return groups

    def sure_trainpipe(self, batch_size):
        return self.dataset.train().shuffled_pairs_source().gen_train_sampling_neg_(
            num_negatives=1).batch_(batch_size).tensor_()

    def encode(self):
        allEmbds = torch.cat((self.User.embeddings.weight, self.Item.embeddings.weight), dim=0)
        features = allEmbds
        smoothed = allEmbds
        beta = 1 - self.beta3
        norm_correction = 1 - beta ** (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features * beta
            smoothed = smoothed + features
        avgEmbds = smoothed.mul(1 - beta).div(norm_correction)
        return torch.split(avgEmbds, (self.User.count, self.Item.count))

    def fit(self, data):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        rec_loss = self.criterion(
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[positives]),
            torch.einsum('BKD,BKD->BK', userEmbds[users], itemEmbds[negatives]))
        cl_weight, budget_weight, zeta = self.coefficients()
        self.last_diagnostics = {'bpr': rec_loss.detach().item(), 'lambda_cl': cl_weight,
                                 'lambda_budget': budget_weight, 'zeta': zeta}
        if cl_weight == 0 and budget_weight == 0 and zeta == 0:
            self.smoother.use_baseline()
            return rec_loss
        state = self.fresh_graph_state()
        hyper = apply_P_H(self.Item.embeddings.weight, self.incidences, state['states'], state['alpha'])
        cl = symmetric_infonce(itemEmbds, hyper, positives, self.projector, self.options.cl_temperature)
        budget = sum((p.mean() - self.options.gate_prior).square() for p in state['probabilities']) / 2
        # This copies/detaches the entire state BEFORE any parameter group steps.
        self.smoother.set_step_snapshot({**state, 'zeta': zeta})
        self.last_diagnostics.update(cl=cl.detach().item(), budget=budget.detach().item(),
                                     unique_cl_items=int(torch.unique(positives).numel()))
        for m, p in enumerate(state['probabilities']):
            self.last_diagnostics[f'gate_{m}_mean'] = p.detach().mean().item()
            self.last_diagnostics[f'gate_{m}_std'] = p.detach().std(unbiased=False).item()
        return rec_loss + cl_weight * cl + budget_weight * budget

    def gradient_diagnostics(self):
        result = {}
        for m, gate in enumerate(self.gates):
            for label, module in [('hidden', gate[0]), ('output', gate[-1])]:
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                result[f'gate_{m}_{label}_grad'] = math.sqrt(sum(g.detach().square().sum().item() for g in grads))
                if any(not torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError('nonfinite gate gradient')
        return result

    # Scoring and caches deliberately match main.py. Seen masking/metrics and
    # checkpoint selection remain in the unchanged FreeRec Coach.
    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = {self.User: userEmbds.detach().clone(),
                               self.Item: itemEmbds.detach().clone()}

    def recommend_from_full(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum('BKD,ND->BN', userEmbds, itemEmbds)

    def recommend_from_pool(self, data):
        userEmbds = self.ranking_buffer[self.User][data[self.User]]
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]]
        return torch.einsum('BKD,BKD->BK', userEmbds, itemEmbds)

    def get_extra_state(self):
        return {'schema': 1, 'epoch': self.current_epoch, 'data_hashes': self.data_hashes,
                'options': asdict(self.options), 'gamma': self.cfg.gamma,
                'num_layers': self.num_layers, 'embedding_dim': self.embedding_dim}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        for key in ('schema', 'data_hashes', 'options', 'gamma', 'num_layers', 'embedding_dim'):
            if state.get(key) != expected[key]:
                raise ValueError(f'checkpoint mismatch in {key}; rebuild with the same data/config')
        self.current_epoch = int(state['epoch'])
        if hasattr(self, 'smoother'):
            self.smoother.clear_step_snapshot()


STAIRMHD = STAIR_MHD_v3
