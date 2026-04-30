"""Research runner: evaluate stop-loss geometry grid on recorded raw data."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, RAW_GLOB, LivePaperEngine
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.stop_grid")
DEFAULT_STOPS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
RISK_PER_TRADE = 0.0025
ACCOUNT_SIZES = [5_000.0, 10_000.0, 50_000.0]


@dataclass
class StopGridResult:
    stop_size: float
    total_trades: int
    win_rate: float
    timeout_rate: float
    average_r: float
    total_realized_r: float
    max_drawdown_r: float
    max_losing_streak: int
    profit_factor: float
    average_trade_duration_sec: float
    average_notional_required: float
    leverage_required_5k: float
    leverage_required_10k: float
    leverage_required_50k: float


class StopGridEngine(LivePaperEngine):
    """Reuse live paper signal/exits, with configurable absolute stop size."""

    def __init__(self, logger: logging.Logger, data_source: str, stop_size: float) -> None:
        super().__init__(logger, data_source=data_source)
        self.stop_size = float(stop_size)
        # Force research mode: no dry-run execution side effects in this runner.
        self.execution_mode = "paper"
        self._bybit_execution = None

    def _close_trade(self, exit_ts: str, exit_price: float, outcome: str, r_value: float) -> None:
        # Same as parent minus CSV append side effect.
        t = self.open_trade
        if t is None:
            return
        rec = {
            "entry_ts": t["entry_ts"],
            "exit_ts": exit_ts,
            "side": t["side"],
            "entry_price": t["entry_price"],
            "exit_price": exit_price,
            "outcome": outcome,
            "r_value": r_value,
            "regime": t["regime"],
        }
        self.closed_trades.append(rec)
        self.realized_pnl_r += r_value
        self.equity_curve.append(self.realized_pnl_r)
        if r_value < 0:
            self._curr_losing_streak += 1
            self.max_losing_streak = max(self.max_losing_streak, self._curr_losing_streak)
        else:
            self._curr_losing_streak = 0
        self.open_trade = None

    def _take_trade(self, top, regime: str, tp_r: float, be_trigger: float | None) -> None:
        entry = top.mid
        self.open_trade = {
            "entry_ts": top.ts_text,
            "entry_ts_unix": top.ts,
            "side": "short",
            "entry_price": entry,
            "tp_price": entry - (tp_r * self.stop_size),
            "sl_price": entry + self.stop_size,
            "regime": regime,
            "be_trigger_r": (be_trigger * self.stop_size) if be_trigger is not None else None,
        }
        self.trades_taken += 1
        self.last_entry_ts = top.ts
        d = datetime.fromtimestamp(top.ts, tz=timezone.utc).strftime("%Y-%m-%d")
        s = f"{d}|{self._session_label(top.ts)}"
        self.daily_counts[d] = self.daily_counts.get(d, 0) + 1
        self.session_counts[s] = self.session_counts.get(s, 0) + 1

    @staticmethod
    def _session_label(ts: float) -> str:
        h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        if 0 <= h < 6:
            return "asia"
        if 6 <= h < 12:
            return "london"
        if 12 <= h < 20:
            return "ny"
        return "off_hours"

    def process_object(self, obj: dict[str, Any]) -> None:
        self.raw_events_seen += 1
        self._process_trade_msg(obj)
        top, pressure = self._process_l2_msg(obj)
        if top is None:
            return

        self._expire_windows(top.ts)
        self.mid_window.append((top.ts, top.mid))
        self.last_mid = top.mid
        self._maybe_manage_open_trade(top)

        event_rate = len(self.l2_times) / max(15.0, 1e-9)
        trade_count = float(len(self.trade_times))
        mid_vals = [m for _, m in self.mid_window]
        mid_vol = 0.0
        if len(mid_vals) >= 2:
            mu = sum(mid_vals) / len(mid_vals)
            var = sum((x - mu) ** 2 for x in mid_vals) / (len(mid_vals) - 1)
            mid_vol = (var if var > 0 else 0.0) ** 0.5

        z_tob = self.z_stats["tob"].zscore_before(top.ts, top.tob_imb)
        z_micro = self.z_stats["micro"].zscore_before(top.ts, top.micro_dev)
        z_pressure = self.z_stats["pressure"].zscore_before(top.ts, pressure)
        z_event = self.z_stats["event_rate"].zscore_before(top.ts, event_rate)
        z_trade = self.z_stats["trade_count"].zscore_before(top.ts, trade_count)
        z_spread = self.z_stats["spread"].zscore_before(top.ts, top.spread)
        z_mid_vol = self.z_stats["mid_vol"].zscore_before(top.ts, mid_vol)

        for k, v in (
            ("tob", top.tob_imb),
            ("micro", top.micro_dev),
            ("pressure", pressure),
            ("event_rate", event_rate),
            ("trade_count", trade_count),
            ("spread", top.spread),
            ("mid_vol", mid_vol),
        ):
            self.z_stats[k].add(top.ts, v)

        abs_parts = [abs(z) for z in (z_tob, z_micro, z_pressure, z_event) if z is not None]
        if not abs_parts:
            return
        anomaly_score = max(abs_parts)
        bullish = max([z for z in (z_tob, z_micro, z_pressure) if z is not None] + [0.0])
        bearish = max([(-z) for z in (z_tob, z_micro, z_pressure) if z is not None] + [0.0])
        direction = "bearish" if bearish > bullish else ("bullish" if bullish > bearish else "neutral")

        pct = self.rank.rank_before(top.ts, anomaly_score)
        self.rank.add(top.ts, anomaly_score)
        current_regime = self._regime(z_event, z_spread, z_mid_vol, z_trade)

        if pct is None:
            return
        if direction == "bearish" and pct >= 0.99:
            self.signals_seen += 1
            can, reason = self._can_take_trade(top.ts, current_regime)
            if can and self.open_trade is None:
                tp_r, be_trigger = self._regime_params(current_regime)
                self._take_trade(top, current_regime, tp_r, be_trigger)
                self.last_signal_ts = top.ts
            else:
                self.trades_blocked += 1
                self.trades_blocked_by_reason[reason] = self.trades_blocked_by_reason.get(reason, 0) + 1


def _iter_objects(raw_dir: Path, glob_pat: str) -> Any:
    files = sorted([p for p in raw_dir.glob(glob_pat) if p.is_file()], key=lambda p: p.stat().st_mtime)
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def _profit_factor(rs: list[float]) -> float:
    wins = sum(x for x in rs if x > 0)
    losses = -sum(x for x in rs if x < 0)
    if losses <= 1e-12:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _max_dd(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        d = peak - x
        if d > mdd:
            mdd = d
    return mdd


def _duration_sec(entry_ts: str, exit_ts: str) -> float:
    try:
        a = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
        b = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
        if a.tzinfo is None:
            a = a.replace(tzinfo=timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=timezone.utc)
        return max(0.0, (b - a).total_seconds())
    except Exception:
        return 0.0


def _avg_notional_and_lev(entry_prices: list[float], stop_size: float, account: float) -> tuple[float, float]:
    if not entry_prices:
        return 0.0, 0.0
    risk_usd = account * RISK_PER_TRADE
    vals = []
    levs = []
    for px in entry_prices:
        qty = risk_usd / stop_size
        notional = qty * px
        vals.append(notional)
        levs.append(notional / account if account > 0 else 0.0)
    return mean(vals), mean(levs)


def run_stop_grid(
    *,
    data_source: str = "coinbase",
    raw_dir: Path = RAW_DATA_DIR,
    stops: list[float] | None = None,
) -> list[StopGridResult]:
    stop_vals = stops or DEFAULT_STOPS
    if any(s <= 0 for s in stop_vals):
        raise ValueError("all stop sizes must be > 0")
    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else RAW_GLOB
    objs = list(_iter_objects(raw_dir, glob_pat))
    if not objs:
        return []

    out: list[StopGridResult] = []
    for stop in stop_vals:
        eng = StopGridEngine(LOGGER, data_source=data_source, stop_size=stop)
        for obj in objs:
            eng.process_object(obj)
        closed = eng.closed_trades
        rs = [float(t["r_value"]) for t in closed]
        n = len(closed)
        wins = sum(1 for x in rs if x > 0)
        timeouts = sum(1 for t in closed if t.get("outcome") == "timeout")
        avg_r = mean(rs) if rs else 0.0
        total_r = sum(rs)
        mdd = _max_dd(eng.equity_curve)
        pf = _profit_factor(rs)
        durs = [_duration_sec(str(t.get("entry_ts", "")), str(t.get("exit_ts", ""))) for t in closed]
        avg_dur = mean(durs) if durs else 0.0
        entry_prices = [float(t["entry_price"]) for t in closed]
        avg_not_10k, lev_10k = _avg_notional_and_lev(entry_prices, stop, 10_000.0)
        _, lev_5k = _avg_notional_and_lev(entry_prices, stop, 5_000.0)
        _, lev_50k = _avg_notional_and_lev(entry_prices, stop, 50_000.0)
        out.append(
            StopGridResult(
                stop_size=stop,
                total_trades=n,
                win_rate=(wins / n if n else 0.0),
                timeout_rate=(timeouts / n if n else 0.0),
                average_r=avg_r,
                total_realized_r=total_r,
                max_drawdown_r=mdd,
                max_losing_streak=eng.max_losing_streak,
                profit_factor=pf,
                average_trade_duration_sec=avg_dur,
                average_notional_required=avg_not_10k,
                leverage_required_5k=lev_5k,
                leverage_required_10k=lev_10k,
                leverage_required_50k=lev_50k,
            )
        )
    return out


def _write_reports(results: list[StopGridResult], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(
            [
                "stop_size",
                "total_trades",
                "win_rate",
                "timeout_rate",
                "average_r",
                "total_realized_r",
                "max_drawdown_r",
                "max_losing_streak",
                "profit_factor",
                "average_trade_duration_sec",
                "average_notional_required",
                "leverage_required_5k",
                "leverage_required_10k",
                "leverage_required_50k",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.stop_size,
                    r.total_trades,
                    r.win_rate,
                    r.timeout_rate,
                    r.average_r,
                    r.total_realized_r,
                    r.max_drawdown_r,
                    r.max_losing_streak,
                    r.profit_factor,
                    r.average_trade_duration_sec,
                    r.average_notional_required,
                    r.leverage_required_5k,
                    r.leverage_required_10k,
                    r.leverage_required_50k,
                ]
            )
    ranked = sorted(
        results,
        key=lambda x: (
            -x.total_trades,
            x.max_drawdown_r,
            x.max_losing_streak,
            x.leverage_required_10k,
            -x.average_r,
            -x.total_realized_r,
        ),
    )
    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stops_tested": [r.stop_size for r in results],
        "best_by_expectancy": (max(results, key=lambda x: x.average_r).stop_size if results else None),
        "best_by_survivability": (ranked[0].stop_size if ranked else None),
        "results": [r.__dict__ for r in results],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stop-loss geometry grid research")
    parser.add_argument("--data-source", choices=("coinbase", "bybit"), default="bybit")
    parser.add_argument("--stops", default="1,2,3,5,8,10,15,20")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--csv-output", type=Path, default=REPORTS_DIR / "stop_grid_results.csv")
    parser.add_argument("--json-output", type=Path, default=REPORTS_DIR / "stop_grid_summary.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stops = [float(x.strip()) for x in str(args.stops).split(",") if x.strip()]
    results = run_stop_grid(data_source=args.data_source, raw_dir=args.raw_dir, stops=stops)
    _write_reports(results, args.csv_output, args.json_output)
    LOGGER.info("wrote %s and %s (rows=%s)", args.csv_output, args.json_output, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

