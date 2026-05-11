"""
PR-ULTIMATE: executable edge validation (research only).

Phase 1 — entry filter matrix on signal-level rows (pre-entry features only; thresholds
computed once on the full deduped signal set — exploratory, same-sample).

Phase 2 — geometry grid with optional entry_filter + fixed TP/BE + partial exit mode.

Phase 3 — fee/slippage stress cases on closed trades (no fill simulator).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import math
from typing import Any

from tjtb.research.eth_geometry_runner import (
    EthGeometryEngine,
    EthGeometryResult,
    PartialExitEthGeometryEngine,
    ROUND_TRIP_TAKER_FEE,
    RISK_PER_TRADE,
    _average_fee_usd,
    _average_present,
    _cohort_for_trade,
    _duration_sec,
    _max_dd,
    _per_trade_net_r,
    _percentile,
    _profit_factor,
)
from tjtb.research.stop_grid_runner import _avg_notional_and_lev, _iter_objects
from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, RAW_GLOB
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.ultimate_edge")

PHASE2_STOPS = [2.0, 3.0, 4.0, 5.0]
PHASE2_TIMEOUTS = [120.0, 300.0, 600.0, 900.0]
PHASE2_TP_RS = [0.75, 1.0, 1.25, 1.5, 2.0]
# Explicit BE only (None would re-enable regime-varying BE and confound the grid).
PHASE2_BE_MODES: list[str] = ["none", "0.75", "1.0"]

REFERENCE_LABEL_STOP = 5.0
REFERENCE_LABEL_TIMEOUT = 900.0

MIN_TRADES_GATE = 25
TARGET_REACH_1R = 0.35


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass
class DistributionThresholds:
    event_rate_median: float
    event_rate_p75: float
    event_rate_p90: float
    trade_count_median: float
    trade_count_p75: float
    trade_count_p90: float
    bid_ask_ratio_median: float


def _signal_rows_for_variant(results: list[EthGeometryResult], stop: float, timeout: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        if r.stop_size != stop or r.timeout_sec != timeout:
            continue
        for t in r.trades:
            out.append(dict(t))
    return out


def _dedupe_signals_keep_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_k: dict[str, dict[str, Any]] = {}
    for row in rows:
        k = str(row.get("entry_signal_key") or "")
        if k not in by_k:
            by_k[k] = dict(row)
    return list(by_k.values())


def _compute_distribution_thresholds(signals: list[dict[str, Any]]) -> DistributionThresholds:
    er = [_safe_float(s.get("entry_event_rate")) for s in signals]
    er = [x for x in er if math.isfinite(x)]
    tc = [_safe_float(s.get("entry_trade_count")) for s in signals]
    tc = [x for x in tc if math.isfinite(x)]
    ratios = [_safe_float(s.get("entry_bid_ask_size_ratio")) for s in signals]
    ratios = [x for x in ratios if math.isfinite(x)]
    return DistributionThresholds(
        event_rate_median=_percentile(er, 50) or 0.0,
        event_rate_p75=_percentile(er, 75) or 0.0,
        event_rate_p90=_percentile(er, 90) or 0.0,
        trade_count_median=_percentile(tc, 50) or 0.0,
        trade_count_p75=_percentile(tc, 75) or 0.0,
        trade_count_p90=_percentile(tc, 90) or 0.0,
        bid_ask_ratio_median=_percentile(ratios, 50) or 0.0,
    )


def _summarize_signal_subset(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    tiers = [_cohort_for_trade(t) for t in trades]
    tier_a = sum(1 for x in tiers if x == "tier_a_fast_clean_winner")
    tier_b = sum(1 for x in tiers if x == "tier_b_slow_noisy_winner")
    tier_c = sum(1 for x in tiers if x == "tier_c_dead_signal")
    mfe_vals = [_safe_float(t.get("mfe_r")) for t in trades]
    mfe_vals = [x for x in mfe_vals if math.isfinite(x)]
    mae_vals = [_safe_float(t.get("mae_r")) for t in trades if t.get("mae_r") is not None]

    def reach_pct(field: str) -> float:
        return sum(1 for t in trades if t.get(field) is not None) / n if n else 0.0

    return {
        "signal_count": n,
        "tier_a_rate": tier_a / n if n else 0.0,
        "tier_b_rate": tier_b / n if n else 0.0,
        "tier_c_rate": tier_c / n if n else 0.0,
        "percent_reaching_0_5R": reach_pct("time_to_first_0_5R"),
        "percent_reaching_1R": reach_pct("time_to_first_1R"),
        "percent_reaching_1_5R": reach_pct("time_to_first_1_5R"),
        "median_mfe": _percentile(mfe_vals, 50),
        "p75_mfe": _percentile(mfe_vals, 75),
        "average_mae": mean(mae_vals) if mae_vals else None,
        "average_time_to_0_5R": _average_present([t.get("time_to_first_0_5R") for t in trades]),
        "average_time_to_1R": _average_present([t.get("time_to_first_1R") for t in trades]),
    }


def _session_accept_factory(sessions: frozenset[str]) -> Callable[[dict[str, Any]], bool]:
    return lambda f: str(f.get("entry_session") or "") in sessions


def _exclude_overlap(f: dict[str, Any]) -> bool:
    return str(f.get("entry_session") or "") != "london_ny_overlap"


def build_phase1_scenarios(th: DistributionThresholds) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    scenarios: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []

    scenarios.append(("p1_baseline_all_signals", lambda f: True))
    scenarios.append(("p1_exclude_london_ny_overlap", _exclude_overlap))
    scenarios.append(("p1_asia_only", _session_accept_factory(frozenset({"asia"}))))
    scenarios.append(("p1_london_only", _session_accept_factory(frozenset({"london"}))))
    scenarios.append(("p1_asia_plus_london", _session_accept_factory(frozenset({"asia", "london"}))))
    scenarios.append(
        (
            "p1_asia_london_offhours_no_overlap",
            lambda f: str(f.get("entry_session") or "") in {"asia", "london", "off_hours"},
        )
    )

    for thr in (-0.60, -0.75, -0.85, -0.90):
        scenarios.append((f"p1_tob_imbalance_lte_{abs(thr):g}", lambda f, t=thr: _safe_float(f.get("entry_tob_imbalance")) <= t))

    for thr in (-0.60, -0.75, -0.85):
        scenarios.append(
            (f"p1_liquidity_imbalance_lte_{abs(thr):g}", lambda f, t=thr: _safe_float(f.get("entry_liquidity_imbalance")) <= t)
        )

    scenarios.extend(
        [
            ("p1_event_rate_above_median", lambda f: _safe_float(f.get("entry_event_rate")) > th.event_rate_median),
            ("p1_event_rate_above_p75", lambda f: _safe_float(f.get("entry_event_rate")) > th.event_rate_p75),
            ("p1_event_rate_above_p90", lambda f: _safe_float(f.get("entry_event_rate")) > th.event_rate_p90),
            ("p1_trade_count_above_median", lambda f: _safe_float(f.get("entry_trade_count")) > th.trade_count_median),
            ("p1_trade_count_above_p75", lambda f: _safe_float(f.get("entry_trade_count")) > th.trade_count_p75),
            ("p1_trade_count_above_p90", lambda f: _safe_float(f.get("entry_trade_count")) > th.trade_count_p90),
        ]
    )

    scenarios.extend(
        [
            ("p1_repeated_signal_30s", lambda f: bool(f.get("is_repeated_signal_30s"))),
            ("p1_repeated_signal_60s", lambda f: bool(f.get("is_repeated_signal_60s"))),
            ("p1_repeated_signal_120s", lambda f: bool(f.get("is_repeated_signal_120s"))),
        ]
    )

    scenarios.extend(
        [
            ("p1_bucket_99_to_99_5", lambda f: 0.99 <= _safe_float(f.get("entry_anomaly_percentile")) < 0.995),
            ("p1_bucket_99_5_to_99_9", lambda f: 0.995 <= _safe_float(f.get("entry_anomaly_percentile")) < 0.999),
            ("p1_bucket_99_9_plus", lambda f: _safe_float(f.get("entry_anomaly_percentile")) >= 0.999),
        ]
    )

    # Priority bundles (hypothesis-led); still pre-entry only.
    scenarios.append(
        (
            "p1_bundle_exclude_overlap_repeated_60s",
            lambda f: _exclude_overlap(f) and bool(f.get("is_repeated_signal_60s")),
        )
    )
    scenarios.append(
        (
            "p1_bundle_exclude_overlap_tob85_rep60",
            lambda f: _exclude_overlap(f)
            and bool(f.get("is_repeated_signal_60s"))
            and _safe_float(f.get("entry_tob_imbalance")) <= -0.85,
        )
    )
    scenarios.append(
        (
            "p1_failed_absorption_proxy_A",
            lambda f: _safe_float(f.get("entry_tob_imbalance")) <= -0.85
            and bool(f.get("is_repeated_signal_60s"))
            and _safe_float(f.get("entry_bid_ask_size_ratio")) <= th.bid_ask_ratio_median,
        )
    )
    scenarios.append(
        (
            "p1_failed_absorption_proxy_B",
            lambda f: _exclude_overlap(f)
            and _safe_float(f.get("entry_tob_imbalance")) <= -0.75
            and bool(f.get("is_repeated_signal_120s"))
            and _safe_float(f.get("entry_liquidity_imbalance")) <= -0.60,
        )
    )

    # Session × strength grids (bounded).
    for sess_key, sess_pred in [
        ("asia", _session_accept_factory(frozenset({"asia"}))),
        ("london", _session_accept_factory(frozenset({"london"}))),
    ]:
        for thr in (-0.75, -0.85):
            scenarios.append(
                (
                    f"p1_{sess_key}_tob_lte_{abs(thr):g}",
                    lambda f, sp=sess_pred, t=thr: sp(f) and _safe_float(f.get("entry_tob_imbalance")) <= t,
                )
            )
        scenarios.append(
            (
                f"p1_{sess_key}_repeat60",
                lambda f, sp=sess_pred: sp(f) and bool(f.get("is_repeated_signal_60s")),
            )
        )

    return scenarios


def _scenario_score(row: dict[str, Any]) -> float:
    """Higher is better for Phase 1 ranking (exploratory composite)."""
    n = int(row.get("signal_count") or 0)
    if n < MIN_TRADES_GATE:
        return -1e18
    reach1 = float(row.get("percent_reaching_1R") or 0.0)
    tier_c = float(row.get("tier_c_rate") or 1.0)
    return reach1 * 100.0 - tier_c * 50.0 + min(n, 500) * 0.01


def _pick_best_phase1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [( _scenario_score(r), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def scenario_predicate_from_row(best: dict[str, Any], all_scenarios: list[tuple[str, Callable[..., bool]]]) -> Callable[[dict[str, Any]], bool] | None:
    sid = best.get("scenario_id")
    for name, pred in all_scenarios:
        if name == sid:
            return pred
    return None


def run_labeling_grid(
    objs: list[dict[str, Any]],
    *,
    data_source: str,
    symbol: str,
    stop: float,
    timeout: float,
    fee_rate: float = ROUND_TRIP_TAKER_FEE,
) -> list[EthGeometryResult]:
    prev = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    try:
        eng = EthGeometryEngine(LOGGER, data_source=data_source, stop_size=stop, timeout_sec=timeout)
        for obj in objs:
            eng.process_object(obj)
        eng.finalize_excursions()
        return [_closed_trades_to_geometry_result(eng, stop, timeout, fee_rate)]
    finally:
        if prev is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev


def _closed_trades_to_geometry_result(
    eng: EthGeometryEngine | PartialExitEthGeometryEngine,
    stop: float,
    timeout: float,
    fee_rate: float,
) -> EthGeometryResult:
    closed = [dict(t, stop_size=stop, timeout_sec=timeout) for t in eng.closed_trades]
    rs_gross = [float(t["r_value"]) for t in closed]
    n = len(closed)
    net_rs = [_per_trade_net_r(float(t["entry_price"]), float(t["r_value"]), fee_rate) for t in closed]
    entry_prices = [float(t["entry_price"]) for t in closed]
    avg_fee_r = mean((fee_rate * ep) / stop for ep in entry_prices) if entry_prices and stop > 0 else 0.0
    avg_fee_usd = _average_fee_usd(entry_prices, stop, 10_000.0, RISK_PER_TRADE, fee_rate)
    _, lev_10k = _avg_notional_and_lev(entry_prices, stop, 10_000.0)
    return EthGeometryResult(
        stop_size=stop,
        timeout_sec=timeout,
        total_trades=n,
        win_rate=(sum(1 for x in rs_gross if x > 0) / n if n else 0.0),
        tp_rate=(sum(1 for t in closed if str(t.get("outcome")) == "tp") / n if n else 0.0),
        timeout_rate=(sum(1 for t in closed if str(t.get("outcome")) == "timeout") / n if n else 0.0),
        average_r=mean(rs_gross) if rs_gross else 0.0,
        total_realized_r=sum(rs_gross),
        max_drawdown_r=_max_dd(eng.equity_curve),
        max_losing_streak=eng.max_losing_streak,
        profit_factor_net=_profit_factor(net_rs),
        average_trade_duration_sec=mean(
            [_duration_sec(str(t.get("entry_ts", "")), str(t.get("exit_ts", ""))) for t in closed]
        )
        if closed
        else 0.0,
        average_notional_required=_avg_notional_and_lev(entry_prices, stop, 10_000.0)[0],
        leverage_required_5k=_avg_notional_and_lev(entry_prices, stop, 5_000.0)[1],
        leverage_required_10k=lev_10k,
        leverage_required_50k=_avg_notional_and_lev(entry_prices, stop, 50_000.0)[1],
        round_trip_fee_rate=fee_rate,
        average_round_trip_fee_usd=avg_fee_usd,
        average_fee_cost_r=avg_fee_r,
        net_average_r_after_fees=mean(net_rs) if net_rs else 0.0,
        net_total_r_after_fees=sum(net_rs),
        trades=closed,
    )


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def run_phase2_grid(
    objs: list[dict[str, Any]],
    *,
    data_source: str,
    symbol: str,
    entry_filter: Callable[[dict[str, Any]], bool] | None,
    fee_rate: float,
    use_partial: bool,
) -> list[dict[str, Any]]:
    prev = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    out: list[dict[str, Any]] = []
    try:
        for stop in PHASE2_STOPS:
            for tout in PHASE2_TIMEOUTS:
                for tp_r in PHASE2_TP_RS:
                    for be_mode in PHASE2_BE_MODES:
                        if use_partial:
                            eng = PartialExitEthGeometryEngine(
                                LOGGER,
                                data_source=data_source,
                                stop_size=stop,
                                timeout_sec=tout,
                                entry_filter=entry_filter,
                                research_fixed_tp_r=tp_r,
                                research_be_mode=be_mode,
                            )
                        else:
                            eng = EthGeometryEngine(
                                LOGGER,
                                data_source=data_source,
                                stop_size=stop,
                                timeout_sec=tout,
                                entry_filter=entry_filter,
                                research_fixed_tp_r=tp_r,
                                research_be_mode=be_mode,
                            )
                        for obj in objs:
                            eng.process_object(obj)
                        eng.finalize_excursions()
                        geom = _closed_trades_to_geometry_result(eng, stop, tout, fee_rate)
                        rs_gross = [float(t["r_value"]) for t in geom.trades]
                        net_rs = [_per_trade_net_r(float(t["entry_price"]), float(t["r_value"]), fee_rate) for t in geom.trades]
                        out.append(
                            {
                                "phase": 2,
                                "partial_exit": use_partial,
                                "stop_size": stop,
                                "timeout_sec": tout,
                                "tp_r": tp_r,
                                "be_mode": be_mode,
                                "gross_expectancy": geom.average_r,
                                "net_expectancy_after_fees": geom.net_average_r_after_fees,
                                "profit_factor_gross": _profit_factor(rs_gross),
                                "profit_factor_net": float(geom.profit_factor_net),
                                "max_drawdown_r": geom.max_drawdown_r,
                                "max_losing_streak": geom.max_losing_streak,
                                "average_holding_sec": geom.average_trade_duration_sec,
                                "leverage_required_10k": geom.leverage_required_10k,
                                "total_trades": geom.total_trades,
                                "average_fee_cost_r": geom.average_fee_cost_r,
                                "tier_a_rate_posthoc_note": "tier labels reference geometry only in phase1",
                            }
                        )
    finally:
        if prev is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev
    return out


def _rank_phase2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (
            x["max_drawdown_r"],
            x["max_losing_streak"],
            -float(x["net_expectancy_after_fees"]),
            -int(x["total_trades"]),
        ),
    )


def phase3_stress_closed_trade(
    entry_price: float,
    gross_r: float,
    *,
    round_trip_fee: float,
    entry_fee_fraction: float,
    exit_fee_fraction: float,
    slip_entry_bps: float,
    slip_exit_bps: float,
) -> float:
    """Synthetic net R after asymmetric fees and bps slippage (short-oriented heuristic)."""
    slip_entry = entry_price * slip_entry_bps / 10_000.0
    exit_px_implied = entry_price - gross_r
    slip_exit = exit_px_implied * slip_exit_bps / 10_000.0
    adjusted_gross = gross_r - slip_entry - slip_exit
    fees_price = entry_price * entry_fee_fraction + exit_px_implied * exit_fee_fraction
    return adjusted_gross - fees_price


def run_phase3_on_best_trades(
    best_phase2_row: dict[str, Any],
    trades_sample: list[dict[str, Any]],
) -> dict[str, Any]:
    """Illustrative execution stress — no fill model."""
    scenarios = {
        "A_market_taker_roundtrip": dict(entry_fee=ROUND_TRIP_TAKER_FEE / 2, exit_fee=ROUND_TRIP_TAKER_FEE / 2, se=1.0, sx=1.0),
        "B_maker_entry_taker_exit": dict(entry_fee=-0.0001, exit_fee=ROUND_TRIP_TAKER_FEE / 2, se=2.0, sx=1.0),
        "C_limit_pullback_entry": dict(entry_fee=ROUND_TRIP_TAKER_FEE / 2, exit_fee=ROUND_TRIP_TAKER_FEE / 2, se=4.0, sx=1.5),
        "D_maker_partial_exit_stub": dict(entry_fee=-0.00005, exit_fee=ROUND_TRIP_TAKER_FEE / 2, se=3.0, sx=2.0),
    }
    out: dict[str, Any] = {}
    gross_rs = [float(t["r_value"]) for t in trades_sample]
    ep = [float(t["entry_price"]) for t in trades_sample]
    for name, sc in scenarios.items():
        nets = [
            phase3_stress_closed_trade(
                e,
                g,
                round_trip_fee=ROUND_TRIP_TAKER_FEE,
                entry_fee_fraction=sc["entry_fee"],
                exit_fee_fraction=sc["exit_fee"],
                slip_entry_bps=sc["se"],
                slip_exit_bps=sc["sx"],
            )
            for e, g in zip(ep, gross_rs)
        ]
        out[name] = {
            "mean_net_r_synthetic": mean(nets) if nets else None,
            "profit_factor_net": _profit_factor(nets),
            "assumption": "bps_slippage_vs_mid_short_stub_no_partial_fill_probability",
        }
    out["reference_phase2_cell"] = {k: best_phase2_row.get(k) for k in ("stop_size", "timeout_sec", "tp_r", "be_mode", "partial_exit")}
    return out


def run_ultimate_study(
    *,
    data_source: str = "bybit",
    raw_dir: Path = RAW_DATA_DIR,
    symbol: str = "ETHUSDT",
    fee_rate: float = ROUND_TRIP_TAKER_FEE,
    json_output: Path = REPORTS_DIR / "ultimate_edge_study.json",
    label_stop: float = REFERENCE_LABEL_STOP,
    label_timeout: float = REFERENCE_LABEL_TIMEOUT,
    run_phase2: bool = True,
    run_phase3: bool = True,
) -> dict[str, Any]:
    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else RAW_GLOB
    objs = list(_iter_objects(raw_dir, glob_pat))
    if not objs:
        return {"error": "no_raw_objects", "raw_dir": str(raw_dir)}

    LOGGER.info("ultimate study: labeling grid stop=%s timeout=%s", label_stop, label_timeout)
    labeling_results = run_labeling_grid(
        objs,
        data_source=data_source,
        symbol=symbol,
        stop=label_stop,
        timeout=label_timeout,
        fee_rate=fee_rate,
    )
    signal_rows = _dedupe_signals_keep_first(_signal_rows_for_variant(labeling_results, label_stop, label_timeout))
    thresholds = _compute_distribution_thresholds(signal_rows)
    scenarios = build_phase1_scenarios(thresholds)

    phase1_rows: list[dict[str, Any]] = []
    for sid, pred in scenarios:
        filt = [dict(s) for s in signal_rows if pred(s)]
        row = {"scenario_id": sid, **_summarize_signal_subset(filt)}
        phase1_rows.append(row)

    best_phase1 = _pick_best_phase1(phase1_rows)
    best_pred = scenario_predicate_from_row(best_phase1 or {}, scenarios) if best_phase1 else None

    gate_detail = {
        "target_reach_1R_fraction": TARGET_REACH_1R,
        "min_trades": MIN_TRADES_GATE,
        "best_scenario_reach_1R": best_phase1.get("percent_reaching_1R") if best_phase1 else None,
        "best_scenario_n": best_phase1.get("signal_count") if best_phase1 else None,
        "phase1_gate_met": bool(
            best_phase1
            and int(best_phase1.get("signal_count") or 0) >= MIN_TRADES_GATE
            and float(best_phase1.get("percent_reaching_1R") or 0.0) >= TARGET_REACH_1R
        ),
    }

    phase2_full: list[dict[str, Any]] = []
    phase2_filtered: list[dict[str, Any]] = []
    if run_phase2:
        LOGGER.info("ultimate study: phase2 grid (unfiltered + filtered), cells=%s", len(PHASE2_STOPS) * len(PHASE2_TIMEOUTS) * len(PHASE2_TP_RS) * len(PHASE2_BE_MODES) * 2)
        phase2_full.extend(run_phase2_grid(objs, data_source=data_source, symbol=symbol, entry_filter=None, fee_rate=fee_rate, use_partial=False))
        phase2_full.extend(run_phase2_grid(objs, data_source=data_source, symbol=symbol, entry_filter=None, fee_rate=fee_rate, use_partial=True))
        if best_pred is not None:
            phase2_filtered.extend(
                run_phase2_grid(objs, data_source=data_source, symbol=symbol, entry_filter=best_pred, fee_rate=fee_rate, use_partial=False)
            )
            phase2_filtered.extend(
                run_phase2_grid(objs, data_source=data_source, symbol=symbol, entry_filter=best_pred, fee_rate=fee_rate, use_partial=True)
            )

    phase2_pool = phase2_filtered if phase2_filtered else phase2_full
    ranked = _rank_phase2([r for r in phase2_pool if int(r.get("total_trades") or 0) >= MIN_TRADES_GATE])
    best_phase2 = ranked[0] if ranked else None

    phase3_report: dict[str, Any] | None = None
    if run_phase3 and best_phase2 is not None:
        prev = os.environ.get("BYBIT_SYMBOL")
        os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
        try:
            use_partial = bool(best_phase2.get("partial_exit"))
            if use_partial:
                eng = PartialExitEthGeometryEngine(
                    LOGGER,
                    data_source=data_source,
                    stop_size=float(best_phase2["stop_size"]),
                    timeout_sec=float(best_phase2["timeout_sec"]),
                    entry_filter=best_pred,
                    research_fixed_tp_r=float(best_phase2["tp_r"]),
                    research_be_mode=best_phase2["be_mode"],
                )
            else:
                eng = EthGeometryEngine(
                    LOGGER,
                    data_source=data_source,
                    stop_size=float(best_phase2["stop_size"]),
                    timeout_sec=float(best_phase2["timeout_sec"]),
                    entry_filter=best_pred,
                    research_fixed_tp_r=float(best_phase2["tp_r"]),
                    research_be_mode=best_phase2["be_mode"],
                )
            for obj in objs:
                eng.process_object(obj)
            eng.finalize_excursions()
            trades = [dict(t) for t in eng.closed_trades]
            phase3_report = run_phase3_on_best_trades(best_phase2, trades)
        finally:
            if prev is None:
                os.environ.pop("BYBIT_SYMBOL", None)
            else:
                os.environ["BYBIT_SYMBOL"] = prev

    pass_pf = bool(best_phase2 and float(best_phase2.get("profit_factor_net") or 0.0) > 1.2)
    pass_ev = bool(best_phase2 and float(best_phase2.get("net_expectancy_after_fees") or 0.0) > 0.0)
    pass_n = bool(best_phase2 and int(best_phase2.get("total_trades") or 0) >= MIN_TRADES_GATE)
    pass_dd = bool(best_phase2 and float(best_phase2.get("max_drawdown_r") or 1e9) < 80.0)

    conclusion = {
        "continue_toward_production": bool(pass_pf and pass_ev and pass_n and pass_dd and gate_detail["phase1_gate_met"]),
        "checks": {
            "positive_net_expectancy": pass_ev,
            "profit_factor_net_gt_1_2": pass_pf,
            "enough_trades": pass_n,
            "drawdown_below_threshold_80R": pass_dd,
            "phase1_reach_and_sample_gate": gate_detail["phase1_gate_met"],
        },
        "best_executable_candidate": best_phase2,
        "production_candidate_rules": {
            "hypothesis": "imbalance -> absorption test -> trade failed absorption -> continuation",
            "phase1_selected_scenario_id": best_phase1.get("scenario_id") if best_phase1 else None,
            "phase1_selected_metrics": best_phase1,
            "executable_geometry": best_phase2,
            "entry_filter_note": "Apply scenario predicate built only from pre-entry fields at trigger time.",
            "research_geometry_reference_for_tiers": {
                "stop": label_stop,
                "timeout_sec": label_timeout,
            },
        },
        "if_fail_primary_reason": (
            None
            if (pass_pf and pass_ev)
            else (
                "negative_net_expectancy_or_weak_pf_after_fees"
                if not (pass_ev and pass_pf)
                else "insufficient_sample_or_drawdown"
            )
        ),
    }

    doc: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "hypothesis_statement": "Detect aggressive imbalance -> wait for absorption -> trade failed absorption -> ride continuation",
        "reference_labeling_geometry": {"stop_size": label_stop, "timeout_sec": label_timeout},
        "distribution_thresholds_note": "median_p75_p90 for activity counts computed on deduped signals at reference geometry (same-sample exploratory)",
        "distribution_thresholds": asdict(thresholds),
        "phase1_scenarios": phase1_rows,
        "phase1_best_by_exploratory_score": best_phase1,
        "phase1_gate": gate_detail,
        "phase2_unfiltered": phase2_full,
        "phase2_filtered_by_best_phase1": phase2_filtered,
        "phase2_best_ranked_survivability": best_phase2,
        "phase3_execution_stress_stub": phase3_report,
        "final_decision": conclusion,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(_sanitize_for_json(doc), indent=2, allow_nan=False), encoding="utf-8")
    LOGGER.info("wrote ultimate study %s", json_output)
    return doc


def main_inner(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="PR-ULTIMATE executable edge validation")
    p.add_argument("--data-source", choices=("coinbase", "bybit"), default="bybit")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--fee-rate", type=float, default=ROUND_TRIP_TAKER_FEE)
    p.add_argument("--json-output", type=Path, default=REPORTS_DIR / "ultimate_edge_study.json")
    p.add_argument("--label-stop", type=float, default=REFERENCE_LABEL_STOP)
    p.add_argument("--label-timeout", type=float, default=REFERENCE_LABEL_TIMEOUT)
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ultimate_study(
        data_source=args.data_source,
        raw_dir=args.raw_dir,
        symbol=args.symbol,
        fee_rate=float(args.fee_rate),
        json_output=args.json_output,
        label_stop=float(args.label_stop),
        label_timeout=float(args.label_timeout),
        run_phase2=not args.skip_phase2,
        run_phase3=not args.skip_phase3,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_inner())
