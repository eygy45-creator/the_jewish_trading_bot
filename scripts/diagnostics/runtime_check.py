#!/usr/bin/env python3
"""Print runtime path resolution and paper-trade file diagnostics (read-only)."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        from tjtb.runtime_paths import PAPER_TRADES_PATH, PROJECT_ROOT
    except ImportError:
        print("ERROR: cannot import tjtb.runtime_paths (set PYTHONPATH=src from repo root).", file=sys.stderr)
        return 1

    cwd = Path.cwd()
    print("cwd:", cwd)
    print("PROJECT_ROOT:", PROJECT_ROOT.resolve())
    print("PAPER_TRADES_PATH:", PAPER_TRADES_PATH.resolve())

    p = PAPER_TRADES_PATH
    exists = p.is_file()
    print("paper_trades exists:", exists)
    if exists:
        try:
            sz = p.stat().st_size
            print("paper_trades size_bytes:", sz)
            with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                rows = list(csv.reader(fh))
            if len(rows) <= 1:
                print("last_data_rows: (no data rows beyond header)")
            else:
                print("last_data_rows (up to 5):")
                for row in rows[-5:]:
                    print(" ", row)
        except OSError as e:
            print("paper_trades read error:", e)
    else:
        print("paper_trades size_bytes: n/a")

    def _pgrep_f(pat: str) -> bool:
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        return r.returncode == 0 and bool((r.stdout or "").strip())

    live = _pgrep_f("tjtb.live.live_paper_crypto")
    dash = _pgrep_f("streamlit run dashboard/app.py")
    print("live_paper_crypto running:", live)
    print("streamlit dashboard running:", dash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
