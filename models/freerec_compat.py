"""FreeRec compatibility shims for PyTorch 2.x and TorchData.
Ensures seamless import and execution across Python 3.10-3.14 and Torch 2.0-2.6.
"""
import sys
import types
import os
import torch

# 1. PyTorch Dynamo device index patch
try:
    import torch._utils
except Exception:
    pass

if not hasattr(torch, '_utils'):
    try:
        import torch._utils_internal as _utils
        torch._utils = _utils
    except Exception:
        pass

if hasattr(torch, '_utils') and not hasattr(torch._utils, '_get_device_index'):
    def _get_device_index(device=None, optional=False, allow_cpu=False):
        if device is None:
            return torch.cuda.current_device() if torch.cuda.is_available() else 0
        if isinstance(device, int):
            return device
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except Exception:
                return 0
        return device.index if hasattr(device, 'index') and device.index is not None else 0
    torch._utils._get_device_index = _get_device_index

# 2. TorchData datapipes compatibility shims
try:
    import torchdata
    import torchdata.datapipes as dp
except Exception:
    dp = None

if dp is None or 'torchdata.datapipes' not in sys.modules or not hasattr(sys.modules.get('torchdata.datapipes'), 'iter'):
    if 'torchdata' not in sys.modules or not isinstance(sys.modules.get('torchdata'), types.ModuleType):
        td = types.ModuleType('torchdata')
        sys.modules['torchdata'] = td
    else:
        td = sys.modules['torchdata']
    if not hasattr(td, '__path__'):
        td.__path__ = []
    dp = types.ModuleType('torchdata.datapipes')
    dp.__path__ = []
    td.datapipes = dp
    sys.modules['torchdata.datapipes'] = dp

if not hasattr(dp, 'iter'):
    iter_mod = types.ModuleType('torchdata.datapipes.iter')
    dp.iter = iter_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod
else:
    iter_mod = dp.iter

if 'torchdata.dataloader2' not in sys.modules:
    dl2 = types.ModuleType('torchdata.dataloader2')
    sys.modules['torchdata.dataloader2'] = dl2
    if 'torchdata' in sys.modules:
        sys.modules['torchdata'].dataloader2 = dl2

    class DataLoader2:
        def __init__(self, datapipe, reading_service=None):
            self.datapipe = datapipe
            self.reading_service = reading_service

        def __iter__(self):
            return iter(self.datapipe)

        def __len__(self):
            try:
                return len(self.datapipe)
            except Exception:
                return 0

        def state_dict(self):
            return {"serialized_datapipe": None}

        def load_state_dict(self, state):
            pass

        def shutdown(self):
            pass

    class MultiProcessingReadingService:
        def __init__(self, num_workers=0):
            self.num_workers = num_workers

    class SequentialReadingService:
        def __init__(self, *services):
            self.services = services

    class DistributedReadingService:
        def __init__(self):
            pass

    dl2.DataLoader2 = DataLoader2
    dl2.MultiProcessingReadingService = MultiProcessingReadingService
    dl2.SequentialReadingService = SequentialReadingService
    dl2.DistributedReadingService = DistributedReadingService

import torch.utils.data

if not hasattr(iter_mod, 'IterDataPipe'):
    try:
        from torch.utils.data import IterDataPipe as _IDP
    except Exception:
        class _IDP(torch.utils.data.IterableDataset):
            def __iter__(self):
                return iter([])
    iter_mod.IterDataPipe = _IDP
    if 'torchdata.datapipes.iter' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.iter'], 'IterDataPipe', _IDP)

if not hasattr(iter_mod, 'IterableWrapper'):
    try:
        from torch.utils.data.datapipes.iter import IterableWrapper as _IW
    except Exception:
        _IW = None
    if _IW is None:
        class _IW(iter_mod.IterDataPipe):
            def __init__(self, iterable=None):
                super().__init__()
                self.iterable = iterable if iterable is not None else []
            def __iter__(self):
                return iter(self.iterable)
            def __len__(self):
                try:
                    return len(self.iterable)
                except Exception:
                    return 0
            def __getitem__(self, idx):
                if hasattr(self.iterable, '__getitem__'):
                    return self.iterable[idx]
                raise NotImplementedError
    iter_mod.IterableWrapper = _IW
    setattr(dp, 'IterableWrapper', _IW)
    if 'torchdata.datapipes.iter' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.iter'], 'IterableWrapper', _IW)
    if 'torchdata.datapipes' in sys.modules:
        setattr(sys.modules['torchdata.datapipes'], 'IterableWrapper', _IW)

if not hasattr(dp, 'map'):
    map_mod = types.ModuleType('torchdata.datapipes.map')
    dp.map = map_mod
    sys.modules['torchdata.datapipes.map'] = map_mod
else:
    map_mod = dp.map

if not hasattr(map_mod, 'MapDataPipe'):
    try:
        from torch.utils.data import MapDataPipe as _MDP
    except Exception:
        class _MDP(torch.utils.data.Dataset):
            def __getitem__(self, idx):
                raise NotImplementedError
            def __len__(self):
                return 0
    map_mod.MapDataPipe = _MDP
    if 'torchdata.datapipes.map' in sys.modules:
        setattr(sys.modules['torchdata.datapipes.map'], 'MapDataPipe', _MDP)

if not hasattr(dp, 'functional_datapipe'):
    def functional_datapipe(name, enable_df_datapipes_support=False):
        def decorator(cls):
            def method(self, *args, **kwargs):
                return cls(self, *args, **kwargs)
            if hasattr(dp, 'iter') and hasattr(dp.iter, 'IterDataPipe'):
                setattr(dp.iter.IterDataPipe, name, method)
            if hasattr(dp, 'map') and hasattr(dp.map, 'MapDataPipe'):
                setattr(dp.map.MapDataPipe, name, method)
            try:
                if hasattr(torch.utils.data, 'IterDataPipe'):
                    setattr(torch.utils.data.IterDataPipe, name, method)
                if hasattr(torch.utils.data, 'MapDataPipe'):
                    setattr(torch.utils.data.MapDataPipe, name, method)
            except Exception:
                pass
            return cls
        return decorator
    dp.functional_datapipe = functional_datapipe

# 3. PyTorch 2.6+ Serialization weights_only safe globals and Monitor sanitizer
import collections
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([collections.defaultdict])
except Exception:
    pass

try:
    from freerec.utils import Monitor
    _orig_monitor_state_dict = Monitor.state_dict
    def _safe_monitor_state_dict(self):
        raw = _orig_monitor_state_dict(self)
        safe = {}
        for k, v in raw.items():
            if isinstance(v, (dict, collections.defaultdict)):
                safe[k] = {ik: iv for ik, iv in v.items()}
            else:
                safe[k] = v
        return safe
    Monitor.state_dict = _safe_monitor_state_dict
except Exception:
    pass

# 4. Torch Geometric compatibility shims (required by freerec.graph and dataset.to_normalized_adj)
try:
    import torch_geometric
    import torch_geometric.utils
    import torch_geometric.utils.num_nodes
except Exception:
    pass

_need_tg_shim = (
    'torch_geometric' not in sys.modules
    or 'torch_geometric.utils' not in sys.modules
    or not hasattr(sys.modules.get('torch_geometric.utils'), 'maybe_num_nodes')
    or not hasattr(sys.modules.get('torch_geometric.utils'), 'to_undirected')
)

if _need_tg_shim:
    if 'torch_geometric' not in sys.modules or not isinstance(sys.modules.get('torch_geometric'), types.ModuleType):
        tg = types.ModuleType('torch_geometric')
        sys.modules['torch_geometric'] = tg
    else:
        tg = sys.modules['torch_geometric']

    if 'torch_geometric.utils' not in sys.modules or not isinstance(sys.modules.get('torch_geometric.utils'), types.ModuleType):
        tg_utils = types.ModuleType('torch_geometric.utils')
        sys.modules['torch_geometric.utils'] = tg_utils
    else:
        tg_utils = sys.modules['torch_geometric.utils']
    tg.utils = tg_utils

    if 'torch_geometric.utils.num_nodes' not in sys.modules or not isinstance(sys.modules.get('torch_geometric.utils.num_nodes'), types.ModuleType):
        tg_num_nodes = types.ModuleType('torch_geometric.utils.num_nodes')
        sys.modules['torch_geometric.utils.num_nodes'] = tg_num_nodes
    else:
        tg_num_nodes = sys.modules['torch_geometric.utils.num_nodes']
    tg_utils.num_nodes = tg_num_nodes

    def maybe_num_nodes(edge_index: torch.Tensor, num_nodes=None) -> int:
        if num_nodes is not None:
            return int(num_nodes)
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0

    def coalesce(
        edge_index: torch.Tensor,
        edge_attr=None,
        num_nodes=None,
        reduce: str = "sum",
        is_sorted: bool = False,
        sort_by_row: bool = True,
    ):
        N = maybe_num_nodes(edge_index, num_nodes)
        has_attr = edge_attr is not None
        if not has_attr:
            edge_attr = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)

        if reduce == "max":
            row, col = edge_index[0].long(), edge_index[1].long()
            keys = row * N + col
            perm = torch.argsort(keys)
            sorted_keys = keys[perm]
            sorted_attr = edge_attr[perm]
            sorted_edges = edge_index[:, perm]
            mask = torch.cat([torch.tensor([True], device=keys.device), sorted_keys[1:] != sorted_keys[:-1]])
            if hasattr(torch, "scatter_reduce"):
                unique_keys, inverse = torch.unique(keys, return_inverse=True)
                out_attr = torch.zeros(unique_keys.size(0), dtype=edge_attr.dtype, device=edge_attr.device)
                out_attr = torch.scatter_reduce(out_attr, 0, inverse, edge_attr, reduce="amax", include_self=False)
                unique_edge_index = sorted_edges[:, mask]
                return unique_edge_index, out_attr if has_attr else None
            else:
                return sorted_edges[:, mask], sorted_attr[mask] if has_attr else None
        else:
            coo = torch.sparse_coo_tensor(edge_index, edge_attr, (N, N), device=edge_index.device).coalesce()
            return coo.indices(), coo.values() if has_attr else None

    def to_undirected(
        edge_index: torch.Tensor,
        edge_attr=None,
        num_nodes=None,
        reduce: str = "max",
    ):
        row, col = edge_index[0], edge_index[1]
        new_row = torch.cat([row, col], dim=0)
        new_col = torch.cat([col, row], dim=0)
        new_edge_index = torch.stack([new_row, new_col], dim=0)
        new_edge_attr = torch.cat([edge_attr, edge_attr], dim=0) if edge_attr is not None else None
        return coalesce(new_edge_index, new_edge_attr, num_nodes=num_nodes, reduce=reduce)

    def _stub(*args, **kwargs):
        pass

    for fn_name, fn_impl in [
        ("maybe_num_nodes", maybe_num_nodes),
        ("coalesce", coalesce),
        ("to_undirected", to_undirected),
    ]:
        if not hasattr(tg_utils, fn_name):
            setattr(tg_utils, fn_name, fn_impl)

    if not hasattr(tg_num_nodes, "maybe_num_nodes"):
        setattr(tg_num_nodes, "maybe_num_nodes", maybe_num_nodes)

    for _stub_fn in [
        "add_remaining_self_loops",
        "remove_self_loops",
        "scatter",
        "spmm",
        "to_edge_index",
        "k_hop_subgraph",
        "dropout_node",
        "dropout_edge",
        "dropout_path",
    ]:
        if not hasattr(tg_utils, _stub_fn):
            setattr(tg_utils, _stub_fn, _stub)


# 4. scikit-learn compatibility shim for freerec.metrics
try:
    import sklearn
    import sklearn.metrics
except ImportError:
    sklearn = types.ModuleType("sklearn")
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    sklearn.metrics = sklearn_metrics
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = sklearn_metrics

    def roc_auc_score(y_true, y_score, *args, **kwargs):
        # Basic CPU ROC-AUC fallback if scikit-learn is absent
        import numpy as np
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()
        pos = y_true == 1
        n_pos = np.sum(pos)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(y_score)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        return float((np.sum(rank[pos]) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))

    sklearn_metrics.roc_auc_score = roc_auc_score


# 5. prettytable compatibility shim for freerec.data.datasets.base
try:
    import prettytable
except ImportError:
    prettytable = types.ModuleType("prettytable")
    sys.modules["prettytable"] = prettytable

    class PrettyTable:
        def __init__(self, *args, **kwargs):
            self.field_names = []
            self.rows = []

        def add_row(self, row):
            self.rows.append(row)

        def __str__(self):
            return ""

    prettytable.PrettyTable = PrettyTable
