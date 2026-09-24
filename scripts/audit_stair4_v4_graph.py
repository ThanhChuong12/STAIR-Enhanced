"""Run the production model preparation path without starting optimization."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, str(root / "main_stair4_v4.py"),
                                     "--audit-only", *sys.argv[1:]], cwd=root))
