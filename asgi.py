"""Alternative Vercel/root ASGI entrypoint: exposes the FastAPI ``app``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tjtb.api.app import app

__all__ = ["app"]
