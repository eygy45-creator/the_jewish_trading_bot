#!/usr/bin/env python3
"""Full read-only diagnostics for PROJECT_ROOT, CSVs, heartbeat, processes, port 8501, log errors."""

from __future__ import annotations

import csv
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _venv_active() -> bool:
    return bool(sys.prefix != sys.base_prefix or getattr(sys, "real_prefix", None))


def _pgrep_f(pat: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return r.returncode == 0 and bool((r.stdout or "").strip())


def _port_listening(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _csv_body_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            n = sum(1 for _ in f)
    except OSError:
        return 0
    return max(0, n - 1)


def _last_csv_rows(path: Path, n: int) -> list[list[str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return []
    if len(rows) <= 1:
        return []
    return rows[-n:]


def _heartbeat_age_sec(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not txt:
            return None
        ts = datetime.fromisoformat(txt[0].strip().replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return max(0.0, (now - ts).total_seconds())
    except (ValueError, OSError):
        return None


def _tail_errors(log_path: Path, max_lines: int = 120) -> list[str]:
    if not log_path.is_file():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    err_pat = re.compile(r"(error|exception|traceback|critical|failed)", re.I)
    hits = [ln for ln in lines if err_pat.search(ln)]
    return hits[-max_lines:]


def main() -> int:
    try:
        from tjtb.runtime_paths import (
            DASHBOARD_LOG_PATH,
            DATA_DIR,
            HEARTBEAT_PATH,
            LIVE_BOT_LOG_PATH,
            LIVE_DATA_DIR,
            LOGS_DIR,
            OPPORTUNITIES_PATH,
            PAPER_TRADES_PATH,
            PROJECT_ROOT,
            RAW_DATA_DIR,
            REPORTS_DIR,
        )
    except ImportError:
        print("ERROR: set PYTHONPATH=src from repo root.", file=sys.stderr)
        return 1

    print("cwd:", Path.cwd().resolve())
    print("PROJECT_ROOT:", PROJECT_ROOT.resolve())
    print("python_executable:", sys.executable)
    print("venv_active:", _venv_active())

    print("\n=== required directories ===")
    for name, p in (
        ("DATA_DIR", DATA_DIR),
        ("LIVE_DATA_DIR", LIVE_DATA_DIR),
        ("RAW_DATA_DIR", RAW_DATA_DIR),
        ("REPORTS_DIR", REPORTS_DIR),
        ("LOGS_DIR", LOGS_DIR),
    ):
        ok = p.is_dir()
        print(f"  {name}: {p.resolve()} exists={ok}")

    print("\n=== paper_trades.csv ===")
    pt = PAPER_TRADES_PATH
    print("path:", pt.resolve())
    print("exists:", pt.is_file())
    if pt.is_file():
        print("size_bytes:", pt.stat().st_size)
        print("data_rows_excl_header:", _csv_body_rows(pt))
        print("last_5_data_rows:")
        for row in _last_csv_rows(pt, 5):
            print(" ", row)

    print("\n=== opportunities.csv ===")
    op = OPPORTUNITIES_PATH
    print("path:", op.resolve())
    print("exists:", op.is_file())
    if op.is_file():
        print("size_bytes:", op.stat().st_size)
        print("data_rows_excl_header:", _csv_body_rows(op))
        print("last_5_data_rows:")
        for row in _last_csv_rows(op, 5):
            print(" ", row)

    print("\n=== heartbeat ===")
    hb = HEARTBEAT_PATH
    print("path:", hb.resolve())
    print("exists:", hb.is_file())
    age = _heartbeat_age_sec(hb)
    print("age_sec_approx:", age if age is not None else "n/a")

    print("\n=== processes ===")
    print("live_paper_crypto_running:", _pgrep_f("tjtb.live.live_paper_crypto"))
    print("streamlit_dashboard_running:", _pgrep_f("streamlit run dashboard/app.py"))

    print("\n=== port 8501 ===")
    print("listening_localhost:", _port_listening(8501))

    print("\n=== recent log errors (live_bot / dashboard) ===")
    for label, lp in (("live_bot", LIVE_BOT_LOG_PATH), ("dashboard", DASHBOARD_LOG_PATH)):
        errs = _tail_errors(lp, 40)
        print(f"--- {label} ({lp.name}) last {len(errs)} matching lines ---")
        if not errs:
            print("  (none or file missing)")
        for ln in errs[-15:]:
            print(" ", ln[:500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
