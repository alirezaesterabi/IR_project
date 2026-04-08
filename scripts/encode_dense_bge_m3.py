"""Thin CLI: same as `build_dense_embeddings.py` with `--model bge-m3` fixed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    path = _ROOT / "scripts" / "build_dense_embeddings.py"
    spec = importlib.util.spec_from_file_location("_build_dense_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    code = mod.main(["--model", "bge-m3", *sys.argv[1:]])
    return 0 if code is None else code


if __name__ == "__main__":
    raise SystemExit(main())
