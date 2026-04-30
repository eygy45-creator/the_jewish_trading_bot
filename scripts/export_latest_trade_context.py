"""
Temporary isolation helper: does NOT call export_trade_context.

Reads paper trades and prints metadata + first 5 l2_data lines from latest raw NDJSON.
Stops as soon as 5 matching lines are found (no full-file scan).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_NDJSON_GLOB = "coinbase_*.ndjson"
# Safety: never read the entire NDJSON; stop after N lines or 5 l2_data hits (whichever first).
MAX_LINES_TO_SCAN = 500_000
_CHANNEL_RE = re.compile(r'"channel"\s*:\s*"([^"]+)"')


def _latest_raw_ndjson_path() -> Path | None:
    raw_dir = ROOT / "data" / "raw"
    files = sorted(
        [p for p in raw_dir.glob(RAW_NDJSON_GLOB) if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def main() -> None:
    paper_trades_path = ROOT / "data" / "live" / "paper_trades.csv"

    if not paper_trades_path.is_file():
        raise FileNotFoundError(f"Missing file: {paper_trades_path}")

    trades = pd.read_csv(paper_trades_path, encoding="utf-8")
    if trades.empty:
        raise ValueError("paper_trades.csv is empty")
    if "entry_ts" not in trades.columns:
        raise ValueError("paper_trades.csv missing required column: entry_ts")

    if "exit_ts" in trades.columns:
        trades = trades.copy()
        trades["_exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True, errors="coerce")
        latest = trades.sort_values("_exit_ts", ascending=False, na_position="last").iloc[0]
    else:
        latest = trades.iloc[-1]

    print("=== latest trade ===")
    print(latest.to_string())

    raw_path = _latest_raw_ndjson_path()
    if raw_path is None:
        print("=== raw NDJSON ===")
        print(f"No files matching {RAW_NDJSON_GLOB} under {ROOT / 'data' / 'raw'}")
        sys.exit(1)

    size = raw_path.stat().st_size
    print("=== latest raw NDJSON ===")
    print(f"path: {raw_path}")
    print(f"size_bytes: {size}")

    print(
        "=== first 5 lines with channel == \"l2_data\" "
        f"(line-by-line, stop after 5 hits or {MAX_LINES_TO_SCAN} lines) ==="
    )
    found = 0
    line_no = 0
    with raw_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_no += 1
            if line_no > MAX_LINES_TO_SCAN:
                print(
                    f"(stopped: scanned {MAX_LINES_TO_SCAN} lines, "
                    f"found {found} l2_data line(s); not a full-file scan)"
                )
                break
            m = _CHANNEL_RE.search(line)
            if not m or m.group(1) != "l2_data":
                continue
            found += 1
            print(f"--- match {found} (file line {line_no}) ---")
            print(line.rstrip("\n"))
            if found >= 5:
                break

    if found == 0 and line_no <= MAX_LINES_TO_SCAN:
        print("(no l2_data lines in scanned prefix)")


if __name__ == "__main__":
    main()
