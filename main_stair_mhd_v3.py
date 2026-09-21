"""STAIR-MHD v3 training, using FreeRec's unchanged evaluation protocol.

Run: python main_stair_mhd_v3.py --config configs/dataset_mhd_v3.yaml
Unit tests: python -m pytest tests/test_stair_mhd_v3.py -q
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

try:
    import models.freerec_compat
except Exception:
    pass
import freerec
import numpy as np
import torch
import yaml

from models.stair_mhd_v3 import MHDOptions, STAIR_MHD_v3
from optimizers.AdamW import AdamWSEvo


class CheckpointAdamWSEvo(AdamWSEvo):
    """Same update implementation; serialize no smoother/model callbacks."""
    def state_dict(self):
        state = super().state_dict()
        state['param_groups'] = [{k: v for k, v in group.items() if k != 'smoother'}
                                 for group in state['param_groups']]
        return state

    def load_state_dict(self, state_dict):
        live = [group.get('smoother') for group in self.param_groups]
        if len(live) != len(state_dict['param_groups']):
            raise ValueError('checkpoint optimizer group count mismatch')
        roles = [group.get('role') for group in self.param_groups]
        if roles != [group.get('role') for group in state_dict['param_groups']]:
            raise ValueError('checkpoint optimizer group ordering mismatch')
        # Never pass a callback into Optimizer.load_state_dict's deepcopy.
        super().load_state_dict(state_dict)
        for group, smoother in zip(self.param_groups, live):
            group['smoother'] = smoother
            if smoother is not None:
                smoother.clear_step_snapshot()


def build_optimizer(model, cfg):
    if getattr(cfg, 'optimizer', 'adamwsevo').lower() != 'adamwsevo':
        raise ValueError('v3 requires AdamWSEvo; another optimizer would bypass BSC')
    return CheckpointAdamWSEvo(model.marked_params(), lr=cfg.lr,
                              betas=(getattr(cfg, 'beta1', getattr(cfg, 'optim_first_moment_decay', 0.9)),
                                     getattr(cfg, 'beta2', getattr(cfg, 'optim_second_moment_decay', 0.999))),
                              weight_decay=cfg.weight_decay)


def training_step(model, optimizer, data):
    """One fresh forward/backward/update; clear even on a failed step."""
    optimizer.zero_grad(set_to_none=True)
    try:
        loss = model(data)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite v3 loss')
        loss.backward()
        diagnostics = {**model.last_diagnostics, **model.gradient_diagnostics()}
        optimizer.step()
        return loss.detach(), diagnostics
    finally:
        model.smoother.clear_step_snapshot()


def seed_everything(seed):
    if seed < 0:
        raise ValueError('v3 requires an explicit nonnegative seed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some GPU sparse kernels are nondeterministic. Warn rather than silently
    # promise exact CUDA reproducibility or replace the baseline kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('cannot exactly restore checkpoint CUDA RNG on this device topology')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_training_checkpoint(path, model, optimizer, epoch):
    """Standalone epoch-boundary checkpoint, also used by unit tests."""
    if model.smoother.step_snapshot is not None:
        raise RuntimeError('checkpoint must be saved after clearing the step snapshot')
    payload = {'schema': 1, 'epoch': int(epoch), 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'rng': rng_state()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_training_checkpoint(path, model, optimizer, restore_rng=True):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 1:
        raise ValueError('unknown MHD checkpoint schema')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    model.smoother.clear_step_snapshot()
    if restore_rng:
        restore_rng_state(payload['rng'])
    return payload['epoch']


def load_config(path, _parents=()):
    """Explicit YAML inheritance: base_config is relative to its own file."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError('cyclic base_config inheritance')
    with path.open(encoding='utf-8') as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f'configuration must be a mapping: {path}')
    parent = values.pop('base_config', None)
    merged = load_config(path.parent / parent, (*_parents, path)) if parent else {}
    merged.update(values)
    return merged


class InheritingParser(freerec.parser.Parser):
    def load(self):
        args = self.parser.parse_args()
        if args.config:
            self.set_defaults(**load_config(args.config))
        # Same CLI-over-YAML precedence as FreeRec's Parser.load.
        return self.parser.parse_args()


def make_config(argv=None):
    cfg = InheritingParser()
    for name, typ, default in [('embedding_dim', int, 64), ('num_layers', int, 3),
                               ('mfiles', str, 'textual_modality.pkl,visual_modality.pkl'),
                               ('num_neighbors', str, '5-1'), ('gamma', float, 0.2)]:
        cfg.add_argument('--' + name.replace('_', '-'), type=typ, default=default)
    for name, field in MHDOptions.__dataclass_fields__.items():
        default = field.default
        if isinstance(default, bool):
            cfg.add_argument('--' + name.replace('_', '-'), action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            cfg.add_argument('--' + name.replace('_', '-'), default=default)
        else:
            cfg.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    cfg.set_defaults(description='STAIR-MHD-v3', root='../../data',
                     dataset='Amazon2014Baby_550_MMRec', epochs=500, batch_size=1024,
                     optimizer='adamwsevo', lr=1e-3, weight_decay=0.1, seed=1,
                     monitors=['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20'],
                     which4best='NDCG@20')
    original = sys.argv
    try:
        if argv is not None:
            sys.argv = [original[0], *argv]
        cfg.compile()
    finally:
        sys.argv = original
    if cfg.which4best != 'NDCG@20':
        raise ValueError('v3 comparison requires baseline validation NDCG@20 selection')
    if int(getattr(cfg, 'num_workers', 0)) != 0:
        freerec.utils.warnLogger('Exact RNG resume is only verified with num_workers=0.')
    MHDOptions.from_config(cfg)
    return cfg


class CoachForMHD(freerec.launcher.Coach):
    def set_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.cfg)

    def set_dataloader(self):
        super().set_dataloader()
        for offset, loader in enumerate((self.trainloader, self.validloader, self.testloader)):
            loader.seed(self.cfg.seed + offset)

    def train_per_epoch(self, epoch):
        # Coach.fit calls train(epoch + 1); this hook already receives 1-based epochs.
        self.model.set_epoch(epoch)
        # Epoch-boundary resume regenerates the same training sampling stream.
        self.trainloader.seed(self.cfg.seed + epoch)
        # FreeRec Launcher shuffles its index list in place. Canonicalize that
        # list before reseeding so an epoch does not depend on prior shuffles.
        from freerec.data.postprocessing.base import Launcher
        from torch.utils.data.graph import traverse_dps
        from torch.utils.data.graph_settings import get_all_graph_pipes
        for pipe in get_all_graph_pipes(traverse_dps(self.trainloader.datapipe)):
            if isinstance(pipe, Launcher):
                pipe.source.sort()
        iterator = iter(self.dataloader)
        # DataLoader2 startup and evaluation can consume global RNG state.
        # FreeRec's negative sampler uses Python random, so seed after startup.
        seed_everything(self.cfg.seed + epoch)
        started = time.perf_counter()
        sums, count = {}, 0
        for data in iterator:
            data = self.dict_to_device(data)
            loss, diagnostics = training_step(self.model, self.optimizer, data)
            self.monitor(loss.item(), n=len(data[self.User]), reduction='mean',
                         mode='train', pool=['LOSS'])
            for key, value in diagnostics.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
        record = {'epoch': epoch, 'batches': count, 'seconds': time.perf_counter() - started,
                  **{key: value / max(count, 1) for key, value in sums.items()}}
        if torch.cuda.is_available():
            record['cuda_peak_allocated'] = torch.cuda.max_memory_allocated()
            record['cuda_peak_reserved'] = torch.cuda.max_memory_reserved()
        with (Path(self.cfg.LOG_PATH) / 'mhd_diagnostics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        freerec.utils.infoLogger('[MHD] ' + json.dumps(record))

    def save_checkpoint(self, epoch):
        # Preserve FreeRec's modules, monitors and epoch semantics; append RNG.
        super().save_checkpoint(epoch)
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        payload = torch.load(path, map_location='cpu', weights_only=True)
        payload['mhd_rng'] = rng_state()
        torch.save(payload, path)

    def load_checkpoint(self):
        epoch = super().load_checkpoint()
        path = Path(self.cfg.CHECKPOINT_PATH) / self.cfg.CHECKPOINT_FILENAME
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if 'mhd_rng' not in payload:
            raise ValueError('MHD training checkpoint is missing RNG state')
        restore_rng_state(payload['mhd_rng'])
        self.model.smoother.clear_step_snapshot()
        return epoch


def make_dataset(cfg):
    """Use the baseline dataset class selection, without copying raw data."""
    processed = Path(cfg.root) / 'Processed' / cfg.dataset
    if not processed.is_dir():
        raise FileNotFoundError(f'FreeRec processed dataset required at {processed}; '
                                'use the same prepared dataset as baseline main.py')
    tasktag = getattr(cfg, 'tasktag', None) or freerec.data.tags.MATCHING
    cls = getattr(freerec.data.datasets, cfg.dataset, None)
    if isinstance(cls, type):
        dataset = cls(root=cfg.root)
    else:
        matching = getattr(freerec.data.datasets.base, 'MatchingRecDataSet', None)
        dataset = matching(cfg.root, cfg.dataset, tasktag=tasktag) if matching else (
            freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=tasktag))
    if not getattr(dataset, 'TASK', None):
        dataset.TASK = tasktag
    return dataset


def main(argv=None):
    freerec.declare(version='1.0.1')  # Same declaration as baseline; runtime logged below.
    cfg = make_config(argv)
    seed_everything(cfg.seed)
    dataset = make_dataset(cfg)
    model = STAIR_MHD_v3(dataset, cfg)
    freerec.utils.infoLogger(f'[MHD runtime] torch={torch.__version__}, freerec={freerec.__version__}; '
                             f'data_hashes={model.data_hashes}')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    coach = CoachForMHD(dataset=dataset, trainpipe=model.sure_trainpipe(cfg.batch_size),
                        validpipe=model.sure_validpipe(cfg.ranking),
                        testpipe=model.sure_testpipe(cfg.ranking), model=model, cfg=cfg)
    coach.fit()


if __name__ == '__main__':
    main()
