"""Research runner: ETHUSDT stop-loss × timeout grid with leverage and taker-fee drag."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, RAW_GLOB, TopState
from tjtb.research.stop_grid_runner import (
    StopGridEngine,
    _avg_notional_and_lev,
    _duration_sec,
    _iter_objects,
    _max_dd,
    _profit_factor,
)
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.eth_geometry")

DEFAULT_STOPS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
DEFAULT_TIMEOUTS_SEC = [120.0, 180.0, 300.0, 480.0, 600.0, 900.0]
RISK_PER_TRADE = 0.0025
ROUND_TRIP_TAKER_FEE = 0.0011  # 0.11% of notional, round-trip
ACCOUNT_SIZES = [5_000.0, 10_000.0, 50_000.0]


@dataclass
class EthGeometryResult:
    stop_size: float
    timeout_sec: float
    total_trades: int
    win_rate: float
    tp_rate: float
    timeout_rate: float
    average_r: float
    total_realized_r: float
    max_drawdown_r: float
    max_losing_streak: int
    profit_factor_net: float
    average_trade_duration_sec: float
    average_notional_required: float
    leverage_required_5k: float
    leverage_required_10k: float
    leverage_required_50k: float
    round_trip_fee_rate: float
    average_round_trip_fee_usd: float
    average_fee_cost_r: float
    net_average_r_after_fees: float
    net_total_r_after_fees: float


class EthGeometryEngine(StopGridEngine):
    """Same signal/exits as StopGridEngine; configurable timeout seconds."""

    def __init__(
        self,
        logger: logging.Logger,
        data_source: str,
        stop_size: float,
        timeout_sec: float,
    ) -> None:
        super().__init__(logger, data_source=data_source, stop_size=stop_size)
        self.timeout_sec = float(timeout_sec)

    def _maybe_manage_open_trade(self, top: TopState) -> None:
        t = self.open_trade
        if t is None:
            return
        entry = float(t["entry_price"])
        sl_price = float(t["sl_price"])
        tp_price = float(t["tp_price"])
        be_trigger = t.get("be_trigger_r")
        if be_trigger is not None and top.mid <= entry - float(be_trigger):
            t["sl_price"] = min(t["sl_price"], entry)
            sl_price = float(t["sl_price"])

        if top.mid >= sl_price:
            r = entry - sl_price
            self._close_trade(top.ts_text, top.mid, "sl_or_be", r)
            return
        if top.mid <= tp_price:
            r = entry - tp_price
            self._close_trade(top.ts_text, top.mid, "tp", r)
            return
        if (top.ts - float(t["entry_ts_unix"])) >= self.timeout_sec:
            r = entry - top.mid
            self._close_trade(top.ts_text, top.mid, "timeout", r)


def _per_trade_net_r(entry_price: float, gross_r: float, fee_rate: float = ROUND_TRIP_TAKER_FEE) -> float:
    """Subtract round-trip taker fee expressed in price-per-unit terms."""
    return gross_r - float(entry_price) * fee_rate


def _average_fee_cost_r(entry_prices: list[float], stop_size: float, fee_rate: float = ROUND_TRIP_TAKER_FEE) -> float:
    """Fee drag in R units where 1R == stop_size price distance."""
    if not entry_prices or stop_size <= 0:
        return 0.0
    return mean((fee_rate * ep) / stop_size for ep in entry_prices)


def _average_fee_usd(
    entry_prices: list[float],
    stop_size: float,
    account: float,
    risk_frac: float = RISK_PER_TRADE,
    fee_rate: float = ROUND_TRIP_TAKER_FEE,
) -> float:
    if not entry_prices or stop_size <= 0:
        return 0.0
    risk_usd = account * risk_frac
    fees = []
    for ep in entry_prices:
        qty = risk_usd / stop_size
        notional = qty * ep
        fees.append(notional * fee_rate)
    return mean(fees)


def run_eth_geometry_grid(
    *,
    data_source: str = "bybit",
    raw_dir: Path = RAW_DATA_DIR,
    stops: list[float] | None = None,
    timeouts_sec: list[float] | None = None,
    symbol: str = "ETHUSDT",
    fee_rate: float = ROUND_TRIP_TAKER_FEE,
) -> list[EthGeometryResult]:
    stop_vals = stops or DEFAULT_STOPS
    timeout_vals = timeouts_sec or DEFAULT_TIMEOUTS_SEC
    if any(s <= 0 for s in stop_vals):
        raise ValueError("all stop sizes must be > 0")
    if any(t <= 0 for t in timeout_vals):
        raise ValueError("all timeouts must be > 0")

    import os

    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else RAW_GLOB
    objs = list(_iter_objects(raw_dir, glob_pat))
    if not objs:
        return []

    out: list[EthGeometryResult] = []
    prev_symbol = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    try:
        for stop in stop_vals:
            for tout in timeout_vals:
                eng = EthGeometryEngine(LOGGER, data_source=data_source, stop_size=stop, timeout_sec=tout)
                for obj in objs:
                    eng.process_object(obj)
                closed = eng.closed_trades
                rs_gross = [float(t["r_value"]) for t in closed]
                n = len(closed)
                wins = sum(1 for x in rs_gross if x > 0)
                tp_n = sum(1 for t in closed if str(t.get("outcome", "")) == "tp")
                timeouts_n = sum(1 for t in closed if str(t.get("outcome", "")) == "timeout")
                avg_r_gross = mean(rs_gross) if rs_gross else 0.0
                total_gross = sum(rs_gross)
                net_rs = [
                    _per_trade_net_r(float(t["entry_price"]), float(t["r_value"]), fee_rate) for t in closed
                ]
                avg_net = mean(net_rs) if net_rs else 0.0
                total_net = sum(net_rs)
                mdd = _max_dd(eng.equity_curve)
                pf_net = _profit_factor(net_rs)
                durs = [_duration_sec(str(t.get("entry_ts", "")), str(t.get("exit_ts", ""))) for t in closed]
                avg_dur = mean(durs) if durs else 0.0
                entry_prices = [float(t["entry_price"]) for t in closed]
                avg_not_10k, lev_10k = _avg_notional_and_lev(entry_prices, stop, 10_000.0)
                _, lev_5k = _avg_notional_and_lev(entry_prices, stop, 5_000.0)
                _, lev_50k = _avg_notional_and_lev(entry_prices, stop, 50_000.0)
                avg_fee_usd = _average_fee_usd(entry_prices, stop, 10_000.0, RISK_PER_TRADE, fee_rate)
                avg_fee_r = _average_fee_cost_r(entry_prices, stop, fee_rate)

                out.append(
                    EthGeometryResult(
                        stop_size=stop,
                        timeout_sec=tout,
                        total_trades=n,
                        win_rate=(wins / n if n else 0.0),
                        tp_rate=(tp_n / n if n else 0.0),
                        timeout_rate=(timeouts_n / n if n else 0.0),
                        average_r=avg_r_gross,
                        total_realized_r=total_gross,
                        max_drawdown_r=mdd,
                        max_losing_streak=eng.max_losing_streak,
                        profit_factor_net=pf_net,
                        average_trade_duration_sec=avg_dur,
                        average_notional_required=avg_not_10k,
                        leverage_required_5k=lev_5k,
                        leverage_required_10k=lev_10k,
                        leverage_required_50k=lev_50k,
                        round_trip_fee_rate=fee_rate,
                        average_round_trip_fee_usd=avg_fee_usd,
                        average_fee_cost_r=avg_fee_r,
                        net_average_r_after_fees=avg_net,
                        net_total_r_after_fees=total_net,
                    )
                )
    finally:
        if prev_symbol is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev_symbol

    return out


def _results_with_trades(results: list[EthGeometryResult]) -> list[EthGeometryResult]:
    return [r for r in results if r.total_trades > 0]


def _rank_survivability(results: list[EthGeometryResult]) -> list[EthGeometryResult]:
    """Lower drawdown/streak first; more trades as tie-break."""
    return sorted(
        results,
        key=lambda x: (
            x.max_drawdown_r,
            x.max_losing_streak,
            -x.total_trades,
            x.leverage_required_10k,
            x.average_fee_cost_r,
        ),
    )


def _rank_realistic_execution(results: list[EthGeometryResult]) -> list[EthGeometryResult]:
    """Survivability, leverage sanity, fee drag, drawdown, net expectancy, net total."""
    return sorted(
        results,
        key=lambda x: (
            x.max_drawdown_r,
            x.max_losing_streak,
            x.leverage_required_10k,
            x.average_fee_cost_r,
            -x.net_average_r_after_fees,
            -x.net_total_r_after_fees,
            -x.total_trades,
        ),
    )


def _pick_best_raw_expectancy(results: list[EthGeometryResult]) -> EthGeometryResult | None:
    pool = _results_with_trades(results)
    if not pool:
        return None
    return max(pool, key=lambda x: x.average_r)


def _pick_best_survivability(results: list[EthGeometryResult]) -> EthGeometryResult | None:
    pool = _results_with_trades(results)
    if not pool:
        return None
    return _rank_survivability(pool)[0]


def _pick_best_realistic_execution(results: list[EthGeometryResult]) -> EthGeometryResult | None:
    pool = _results_with_trades(results)
    if not pool:
        return None
    return _rank_realistic_execution(pool)[0]


def _sanitize_json_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    pf = out.get("profit_factor_net")
    if pf == float("inf") or pf == "inf":
        out["profit_factor_net"] = None
    return out


def _result_summary_dict(r: EthGeometryResult | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return _sanitize_json_row(asdict(r))


def write_eth_geometry_reports(
    results: list[EthGeometryResult],
    csv_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else [f.name for f in fields(EthGeometryResult)]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = asdict(r)
            if row.get("profit_factor_net") == float("inf"):
                row["profit_factor_net"] = "inf"
            w.writerow(row)

    best_exp = _pick_best_raw_expectancy(results)
    best_surv = _pick_best_survivability(results)
    best_real = _pick_best_realistic_execution(results)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "stops_tested": sorted({r.stop_size for r in results}),
        "timeouts_sec_tested": sorted({r.timeout_sec for r in results}),
        "round_trip_taker_fee_assumption": ROUND_TRIP_TAKER_FEE,
        "risk_per_trade": RISK_PER_TRADE,
        "reference_account_sizes_usd": ACCOUNT_SIZES,
        "best_by_raw_expectancy": _result_summary_dict(best_exp),
        "best_by_survivability": _result_summary_dict(best_surv),
        "best_realistic_execution_candidate": _result_summary_dict(best_real),
        "results": [_sanitize_json_row(asdict(r)) for r in results],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETH geometry grid: stops × timeouts, fees, leverage")
    parser.add_argument("--data-source", choices=("coinbase", "bybit"), default="bybit")
    parser.add_argument("--stops", default="1,2,3,5,8,10")
    parser.add_argument("--timeouts", default="120,180,300,480,600,900", help="Comma-separated timeout seconds")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--fee-rate", type=float, default=ROUND_TRIP_TAKER_FEE, help="Round-trip fee as decimal")
    parser.add_argument("--csv-output", type=Path, default=REPORTS_DIR / "eth_geometry_results.csv")
    parser.add_argument("--json-output", type=Path, default=REPORTS_DIR / "eth_geometry_summary.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stops = [float(x.strip()) for x in str(args.stops).split(",") if x.strip()]
    timeouts = [float(x.strip()) for x in str(args.timeouts).split(",") if x.strip()]
    results = run_eth_geometry_grid(
        data_source=args.data_source,
        raw_dir=args.raw_dir,
        stops=stops,
        timeouts_sec=timeouts,
        symbol=str(args.symbol).strip().upper(),
        fee_rate=float(args.fee_rate),
    )
    write_eth_geometry_reports(results, args.csv_output, args.json_output)
    LOGGER.info("wrote %s and %s (rows=%s)", args.csv_output, args.json_output, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
