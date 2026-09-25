import pytest
from pathlib import Path

@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    for i, arg in enumerate(args):
        if arg == "--basetemp" and i + 1 < len(args):
            Path(args[i + 1]).mkdir(parents=True, exist_ok=True)
        elif arg.startswith("--basetemp="):
            Path(arg.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        Path(basetemp).mkdir(parents=True, exist_ok=True)
