#!/usr/bin/env python3
"""Bybit vs Coinbase feature-family parity diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from tjtb.features.build_features import parse_ts_to_unix

LOOKBACK_SEC = 15.0
EPS = 1e-9


class RollingStat:
    def __init__(self, window_sec: float) -> None:
        self.window_sec = window_sec
        self.q: deque[tuple[float, float]] = deque()
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def _expire(self, ts: float) -> None:
        lo = ts - self.window_sec
        while self.q and self.q[0][0] < lo:
            _, v = self.q.popleft()
            self.n -= 1
            self.s -= v
            self.s2 -= v * v

    def zscore_before(self, ts: float, y: float) -> float | None:
        self._expire(ts)
        if self.n < 2:
            return None
        mu = self.s / self.n
        var = (self.s2 - (self.s * self.s / self.n)) / (self.n - 1)
        if var <= 1e-12:
            return None
        return (y - mu) / math.sqrt(var)

    def add(self, ts: float, y: float) -> None:
        self.q.append((ts, y))
        self.n += 1
        self.s += y
        self.s2 += y * y


def _safe_float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_present(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def _read_trade_ts(path: Path) -> list[float]:
    out: list[float] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_raw = str(row.get("ts", "")).strip()
            if not ts_raw:
                continue
            try:
                out.append(parse_ts_to_unix(ts_raw))
            except ValueError:
                continue
    out.sort()
    return out


def _read_pressure_by_ts(path: Path) -> dict[float, float]:
    out: dict[float, float] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_raw = str(row.get("ts", "")).strip()
            side = str(row.get("side", "")).strip().lower()
            px = _safe_float(row.get("price"))
            nq = _safe_float(row.get("size"))
            if not ts_raw or side not in {"bid", "ask"} or px is None or nq is None:
                continue
            try:
                ts = parse_ts_to_unix(ts_raw)
            except ValueError:
                continue
            # Approximate per-ts pressure from final quantity only; parse stage should
            # ideally preserve deltas, but this is enough for cross-source diagnostics.
            signed = nq if side == "bid" else -nq
            out[ts] = out.get(ts, 0.0) + signed
    return out


def _read_book_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_raw = str(_first_present(row, ("ts",)) or "").strip()
            if not ts_raw:
                continue
            try:
                ts = parse_ts_to_unix(ts_raw)
            except ValueError:
                continue
            bb = _safe_float(_first_present(row, ("best_bid",)))
            ba = _safe_float(_first_present(row, ("best_ask",)))
            bsz = _safe_float(_first_present(row, ("best_bid_size", "bid_size")))
            asz = _safe_float(_first_present(row, ("best_ask_size", "ask_size")))
            spread = _safe_float(_first_present(row, ("spread",)))
            mid = _safe_float(_first_present(row, ("mid", "mid_price")))
            micro = _safe_float(_first_present(row, ("microprice",)))
            tob = _safe_float(_first_present(row, ("tob_imbalance", "top_of_book_imbalance")))
            if None in {bb, ba, bsz, asz, spread, mid, micro, tob}:
                continue
            rows.append(
                {
                    "ts": ts,
                    "bb": float(bb),
                    "ba": float(ba),
                    "bsz": float(bsz),
                    "asz": float(asz),
                    "spread": float(spread),
                    "mid": float(mid),
                    "micro_dev": float(micro - mid),
                    "tob": float(tob),
                    "signed_pressure": _safe_float(row.get("signed_pressure")) or 0.0,
                }
            )
    rows.sort(key=lambda x: x["ts"])
    return rows


def _summary(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p95": None, "p99": None}
    n = len(xs)
    mean = sum(xs) / n
    if n >= 2:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    ys = sorted(xs)
    def q(p: float) -> float:
        i = min(max(int(round((n - 1) * p)), 0), n - 1)
        return ys[i]
    return {"n": n, "mean": mean, "std": std, "p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


def build_feature_diagnostics(book_rows: list[dict[str, float]], trade_ts: list[float], pressure_by_ts: dict[float, float]) -> dict[str, object]:
    l2_times: deque[float] = deque()
    trade_q: deque[float] = deque()
    mid_window: deque[tuple[float, float]] = deque()

    z_stats = {
        "tob": RollingStat(LOOKBACK_SEC),
        "micro": RollingStat(LOOKBACK_SEC),
        "pressure": RollingStat(LOOKBACK_SEC),
        "event_rate": RollingStat(LOOKBACK_SEC),
        "trade_count": RollingStat(LOOKBACK_SEC),
        "spread": RollingStat(LOOKBACK_SEC),
        "mid_vol": RollingStat(LOOKBACK_SEC),
    }

    idx_trade = 0
    updates_hz: list[float] = []
    trade_hz: list[float] = []
    spreads: list[float] = []
    tobs: list[float] = []
    micro_devs: list[float] = []
    pressures: list[float] = []
    anomalies: list[float] = []
    z_abs: list[float] = []

    for row in book_rows:
        ts = row["ts"]
        lo = ts - LOOKBACK_SEC
        while l2_times and l2_times[0] < lo:
            l2_times.popleft()
        while trade_q and trade_q[0] < lo:
            trade_q.popleft()
        while mid_window and mid_window[0][0] < lo:
            mid_window.popleft()

        l2_times.append(ts)
        mid_window.append((ts, row["mid"]))
        while idx_trade < len(trade_ts) and trade_ts[idx_trade] <= ts:
            t = trade_ts[idx_trade]
            if t >= lo:
                trade_q.append(t)
            idx_trade += 1

        event_rate = len(l2_times) / max(LOOKBACK_SEC, EPS)
        trade_count = float(len(trade_q))
        mid_vals = [m for _, m in mid_window]
        mid_vol = 0.0
        if len(mid_vals) >= 2:
            mu = sum(mid_vals) / len(mid_vals)
            var = sum((x - mu) ** 2 for x in mid_vals) / (len(mid_vals) - 1)
            mid_vol = math.sqrt(max(var, 0.0))

        pressure = row["signed_pressure"] if row["signed_pressure"] != 0.0 else pressure_by_ts.get(ts, 0.0)
        z_tob = z_stats["tob"].zscore_before(ts, row["tob"])
        z_micro = z_stats["micro"].zscore_before(ts, row["micro_dev"])
        z_pressure = z_stats["pressure"].zscore_before(ts, pressure)
        z_event = z_stats["event_rate"].zscore_before(ts, event_rate)
        z_trade = z_stats["trade_count"].zscore_before(ts, trade_count)
        z_spread = z_stats["spread"].zscore_before(ts, row["spread"])
        z_mid_vol = z_stats["mid_vol"].zscore_before(ts, mid_vol)

        for k, v in (
            ("tob", row["tob"]),
            ("micro", row["micro_dev"]),
            ("pressure", pressure),
            ("event_rate", event_rate),
            ("trade_count", trade_count),
            ("spread", row["spread"]),
            ("mid_vol", mid_vol),
        ):
            z_stats[k].add(ts, v)

        abs_parts = [abs(z) for z in (z_tob, z_micro, z_pressure, z_event) if z is not None]
        if abs_parts:
            anomaly = max(abs_parts)
            anomalies.append(anomaly)
            z_abs.extend(abs_parts)

        updates_hz.append(event_rate)
        trade_hz.append(trade_count / max(LOOKBACK_SEC, EPS))
        spreads.append(row["spread"])
        tobs.append(row["tob"])
        micro_devs.append(row["micro_dev"])
        pressures.append(pressure)

    return {
        "rows": len(book_rows),
        "duration_sec": (book_rows[-1]["ts"] - book_rows[0]["ts"]) if len(book_rows) >= 2 else 0.0,
        "update_rate_hz": _summary(updates_hz),
        "trade_rate_hz": _summary(trade_hz),
        "spread": _summary(spreads),
        "tob_imbalance": _summary(tobs),
        "microprice_dev": _summary(micro_devs),
        "signed_pressure": _summary(pressures),
        "anomaly_score": _summary(anomalies),
        "z_abs": _summary(z_abs),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Bybit vs Coinbase feature-family behavior")
    p.add_argument("--bybit-book", type=Path, default=Path("data/parsed/bybit_book_state.csv"))
    p.add_argument("--bybit-trades", type=Path, default=Path("data/parsed/bybit_trades.csv"))
    p.add_argument("--bybit-updates", type=Path, default=Path("data/parsed/bybit_book_updates.csv"))
    p.add_argument("--coinbase-book", type=Path, default=Path("data/parsed/book_state.csv"))
    p.add_argument("--coinbase-trades", type=Path, default=Path("data/parsed/trades.csv"))
    p.add_argument("--coinbase-updates", type=Path, default=Path("data/parsed/book_updates.csv"))
    p.add_argument("--output", type=Path, default=Path("reports/bybit_feature_parity.json"))
    args = p.parse_args()

    bybit_rows = _read_book_rows(args.bybit_book)
    bybit_tr = _read_trade_ts(args.bybit_trades)
    bybit_pr = _read_pressure_by_ts(args.bybit_updates)
    bybit_diag = build_feature_diagnostics(bybit_rows, bybit_tr, bybit_pr)

    coin_rows = _read_book_rows(args.coinbase_book)
    coin_tr = _read_trade_ts(args.coinbase_trades)
    coin_pr = _read_pressure_by_ts(args.coinbase_updates)
    coin_diag = build_feature_diagnostics(coin_rows, coin_tr, coin_pr)

    report = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "inputs": {
            "bybit": {
                "book": str(args.bybit_book),
                "trades": str(args.bybit_trades),
                "updates": str(args.bybit_updates),
            },
            "coinbase": {
                "book": str(args.coinbase_book),
                "trades": str(args.coinbase_trades),
                "updates": str(args.coinbase_updates),
            },
        },
        "bybit": bybit_diag,
        "coinbase": coin_diag,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps({"bybit_rows": bybit_diag["rows"], "coinbase_rows": coin_diag["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

