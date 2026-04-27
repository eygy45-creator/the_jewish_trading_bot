"""
Canonical filesystem layout for the TJTB paper stack.

PROJECT_ROOT is the directory that contains ``src/tjtb`` (the inner repo root),
resolved from this file's location so it does not depend on the process cwd.
"""

from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    # This file: <PROJECT_ROOT>/src/tjtb/runtime_paths.py
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT: Path = _project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
LIVE_DATA_DIR: Path = DATA_DIR / "live"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
PAPER_TRADES_PATH: Path = LIVE_DATA_DIR / "paper_trades.csv"
OPPORTUNITIES_PATH: Path = LIVE_DATA_DIR / "opportunities.csv"


def ensure_runtime_dirs() -> None:
    """Create all directories used by the live paper bot and dashboard."""
    for d in (RAW_DATA_DIR, LIVE_DATA_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
