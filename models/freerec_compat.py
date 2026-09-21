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

if dp is None or 'torchdata.datapipes' not in sys.modules:
    if 'torchdata' not in sys.modules:
        td = types.ModuleType('torchdata')
        sys.modules['torchdata'] = td
    else:
        td = sys.modules['torchdata']
    dp = types.ModuleType('torchdata.datapipes')
    td.datapipes = dp
    sys.modules['torchdata.datapipes'] = dp

if not hasattr(dp, 'iter'):
    iter_mod = types.ModuleType('torchdata.datapipes.iter')
    dp.iter = iter_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod
else:
    iter_mod = dp.iter

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
