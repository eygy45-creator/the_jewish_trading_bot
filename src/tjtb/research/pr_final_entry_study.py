"""
PR-FINAL: $2 stop, rule-based entry replay, failed-absorption focus, 2R continuation (research only).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, RAW_GLOB
from tjtb.research.eth_geometry_runner import (
    EthGeometryEngine,
    PartialExitEthGeometryEngine,
    ROUND_TRIP_TAKER_FEE,
    _per_trade_net_r,
    _percentile,
    _profit_factor,
)
from tjtb.research.stop_grid_runner import _iter_objects
from tjtb.research.ultimate_edge_study import _closed_trades_to_geometry_result, _dedupe_signals_keep_first
from tjtb.runtime_paths import RAW_DATA_DIR, REPORTS_DIR

LOGGER = logging.getLogger("tjtb.research.pr_final")

STOP_FINAL = 2.0
TIMEOUT_METRICS = 900.0
MIN_TRADES_GATE = 20
PART_D_TP_R = 2.0
PART_D_BE = "none"
LIMIT_FILL_WINDOW_SEC = 180.0


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def is_trade_2r_winner(t: dict[str, Any]) -> bool:
    return float(t.get("stop_size") or 0) == STOP_FINAL and _safe_float(t.get("mfe_r")) >= 2.0


def is_clean_2r_winner(t: dict[str, Any]) -> bool:
    if not is_trade_2r_winner(t):
        return False
    if _safe_float(t.get("mae_r")) < -0.5:
        return False
    t1 = t.get("time_to_first_1R")
    t2 = t.get("time_to_first_2R")
    if t1 is None or float(t1) > 300.0:
        return False
    if t2 is None or float(t2) > 900.0:
        return False
    return True


def _sanitize_doc(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_doc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_doc(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _distribution_thresholds(signals: list[dict[str, Any]]) -> dict[str, float]:
    er = [_safe_float(s.get("entry_event_rate")) for s in signals]
    er = [x for x in er if math.isfinite(x)]
    tc = [_safe_float(s.get("entry_trade_count")) for s in signals]
    tc = [x for x in tc if math.isfinite(x)]
    return {
        "event_rate_median": float(_percentile(er, 50) or 0.0),
        "event_rate_p75": float(_percentile(er, 75) or 0.0),
        "event_rate_p90": float(_percentile(er, 90) or 0.0),
        "trade_count_median": float(_percentile(tc, 50) or 0.0),
        "trade_count_p75": float(_percentile(tc, 75) or 0.0),
        "trade_count_p90": float(_percentile(tc, 90) or 0.0),
    }


def _excursion_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if not n:
        return {
            "trade_count": 0,
            "signal_note": "trade_count_equals_executed_entries_under_filters",
            "percent_reaching_1R": 0.0,
            "percent_reaching_1_5R": 0.0,
            "percent_reaching_2R": 0.0,
            "clean_2r_winner_rate": 0.0,
            "winner_2r_rate": 0.0,
            "median_mfe": None,
            "p75_mfe": None,
            "average_mae": None,
            "average_time_to_1R": None,
            "average_time_to_2R": None,
        }
    mfes = [_safe_float(t.get("mfe_r")) for t in trades]
    mfes = [x for x in mfes if math.isfinite(x)]
    maes = [_safe_float(t.get("mae_r")) for t in trades if t.get("mae_r") is not None]

    def pct_hit(field: str) -> float:
        return sum(1 for t in trades if t.get(field) is not None) / n

    t1 = [float(t["time_to_first_1R"]) for t in trades if t.get("time_to_first_1R") is not None]
    t2 = [float(t["time_to_first_2R"]) for t in trades if t.get("time_to_first_2R") is not None]

    return {
        "trade_count": n,
        "percent_reaching_1R": pct_hit("time_to_first_1R"),
        "percent_reaching_1_5R": pct_hit("time_to_first_1_5R"),
        "percent_reaching_2R": pct_hit("time_to_first_2R"),
        "clean_2r_winner_rate": sum(1 for t in trades if is_clean_2r_winner(t)) / n,
        "winner_2r_rate": sum(1 for t in trades if is_trade_2r_winner(t)) / n,
        "median_mfe": _percentile(mfes, 50),
        "p75_mfe": _percentile(mfes, 75),
        "average_mae": mean(maes) if maes else None,
        "average_time_to_1R": mean(t1) if t1 else None,
        "average_time_to_2R": mean(t2) if t2 else None,
    }


def _metric_default_dict() -> dict[str, Any]:
    return {
        "signal_count": 0,
        "trade_count": 0,
        "percent_reaching_1R": 0.0,
        "percent_reaching_1_5R": 0.0,
        "percent_reaching_2R": 0.0,
        "clean_2r_winner_rate": 0.0,
        "median_mfe": 0.0,
        "p75_mfe": 0.0,
        "average_mae": 0.0,
        "average_time_to_1R": 0.0,
        "average_time_to_2R": 0.0,
        "gross_expectancy": 0.0,
        "net_expectancy_after_fees": 0.0,
        "net_profit_factor": 0.0,
        "max_drawdown_r": 0.0,
        "max_losing_streak": 0,
    }


def _replay_engine(
    objs: list[dict[str, Any]],
    *,
    data_source: str,
    symbol: str,
    stop: float,
    timeout_sec: float,
    entry_filter: Callable[[dict[str, Any]], bool] | None,
    tp_r: float | None,
    be_mode: str | None,
    partial_spec: tuple[float, float, float, float] | None,
) -> EthGeometryEngine | PartialExitEthGeometryEngine:
    kwargs = {
        "entry_filter": entry_filter,
        "research_fixed_tp_r": tp_r,
        "research_be_mode": be_mode,
    }
    if partial_spec is not None:
        p_r, p_scale, r_scale, run_r = partial_spec
        eng = PartialExitEthGeometryEngine(
            LOGGER,
            data_source=data_source,
            stop_size=stop,
            timeout_sec=timeout_sec,
            partial_first_r=p_r,
            partial_first_scale=p_scale,
            runner_scale=r_scale,
            runner_tp_r=run_r,
            **kwargs,
        )
    else:
        eng = EthGeometryEngine(
            LOGGER,
            data_source=data_source,
            stop_size=stop,
            timeout_sec=timeout_sec,
            **kwargs,
        )
    for obj in objs:
        eng.process_object(obj)
    eng.finalize_excursions()
    return eng


def _row_from_eng(
    eng: EthGeometryEngine | PartialExitEthGeometryEngine,
    *,
    stop: float,
    timeout_sec: float,
    fee_rate: float,
    rule_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    geom = _closed_trades_to_geometry_result(eng, stop, timeout_sec, fee_rate)
    trades = [dict(t) for t in geom.trades]
    rs_gross = [float(t["r_value"]) for t in trades]
    net_rs = [_per_trade_net_r(float(t["entry_price"]), float(t["r_value"]), fee_rate) for t in trades]
    ex = _excursion_stats(trades)
    row = {
        "rule_set_id": rule_name,
        "stop_size": stop,
        "timeout_sec": timeout_sec,
        "signals_seen_total_engine": eng.signals_seen,
        "trades_blocked_entry_filter": eng.trades_blocked_by_reason.get("entry_filter", 0),
        "gross_expectancy": geom.average_r,
        "net_expectancy_after_fees": geom.net_average_r_after_fees,
        "profit_factor_gross": _profit_factor(rs_gross),
        "profit_factor_net": float(geom.profit_factor_net),
        "max_drawdown_r": geom.max_drawdown_r,
        "max_losing_streak": geom.max_losing_streak,
        "average_holding_sec": geom.average_trade_duration_sec,
        "leverage_required_10k": geom.leverage_required_10k,
        "average_fee_cost_r": geom.average_fee_cost_r,
        "survivability_rank_key": (
            geom.max_drawdown_r,
            geom.max_losing_streak,
            -geom.net_average_r_after_fees,
            -geom.total_trades,
        ),
        **ex,
    }
    if extra:
        row.update(extra)
    return row


def _rank_2r_winners(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    win = [dict(t) for t in trades if is_trade_2r_winner(t)]

    def sort_key(t: dict[str, Any]) -> tuple:
        mae = _safe_float(t.get("mae_r"))
        t1 = _safe_float(t.get("time_to_first_1R"), 99999.0)
        t2 = _safe_float(t.get("time_to_first_2R"), 99999.0)
        tob = _safe_float(t.get("entry_tob_imbalance"))
        liq = _safe_float(t.get("entry_liquidity_imbalance"))
        rep = int(t.get("prior_qualifying_anomalies_120s") or 0)
        fa = _safe_float(t.get("entry_failed_absorption_score"))
        er = _safe_float(t.get("entry_event_rate"))
        tc = _safe_float(t.get("entry_trade_count"))
        return (-mae, t1, t2, tob, liq, rep, fa, str(t.get("entry_session")), str(t.get("regime")), er, tc)

    win.sort(key=sort_key)
    return win


@dataclass(frozen=True)
class RuleConfig:
    session_mode: str
    failed_absorption_mode: str
    repeated_window_sec: int
    tob_lte: float
    liquidity_lte: float
    event_rate_filter: str
    trade_count_filter: str

    def rule_set_id(self) -> str:
        return (
            f"session={self.session_mode}|fa={self.failed_absorption_mode}|repeat={self.repeated_window_sec}s|"
            f"tob<={self.tob_lte}|liq<={self.liquidity_lte}|er={self.event_rate_filter}|tc={self.trade_count_filter}"
        )


def _all_rule_configs() -> list[RuleConfig]:
    out: list[RuleConfig] = []
    for session_mode in ("asia_only", "london_only", "asia_plus_london", "exclude_london_ny_overlap"):
        for fa_mode in ("medium", "strict"):
            for rep in (30, 60, 120):
                for tob in (-0.75, -0.85, -0.95):
                    for liq in (-0.75, -0.85, -0.95):
                        for er in ("median", "p75", "p90"):
                            for tc in ("median", "p75", "p90"):
                                out.append(
                                    RuleConfig(
                                        session_mode=session_mode,
                                        failed_absorption_mode=fa_mode,
                                        repeated_window_sec=rep,
                                        tob_lte=tob,
                                        liquidity_lte=liq,
                                        event_rate_filter=er,
                                        trade_count_filter=tc,
                                    )
                                )
    return out


def _entry_rule_predicate(config: RuleConfig, th: dict[str, float]) -> Callable[[dict[str, Any]], bool]:
    rep_field = f"is_repeated_signal_{config.repeated_window_sec}s"

    def pred(f: dict[str, Any]) -> bool:
        session = str(f.get("entry_session") or "")
        if config.session_mode == "asia_only" and session != "asia":
            return False
        if config.session_mode == "london_only" and session != "london":
            return False
        if config.session_mode == "asia_plus_london" and session not in {"asia", "london"}:
            return False
        if config.session_mode == "exclude_london_ny_overlap" and session == "london_ny_overlap":
            return False
        if config.failed_absorption_mode == "medium" and not bool(f.get("failed_absorption_medium")):
            return False
        if config.failed_absorption_mode == "strict" and not bool(f.get("failed_absorption_strict")):
            return False
        return (
            bool(f.get(rep_field))
            and _safe_float(f.get("entry_tob_imbalance")) <= config.tob_lte
            and _safe_float(f.get("entry_liquidity_imbalance")) <= config.liquidity_lte
            and _safe_float(f.get("entry_event_rate")) > float(th[f"event_rate_{config.event_rate_filter}"])
            and _safe_float(f.get("entry_trade_count")) > float(th[f"trade_count_{config.trade_count_filter}"])
        )

    return pred


def _false_positive_rate(trades: list[dict[str, Any]]) -> float:
    n = len(trades)
    if n == 0:
        return 0.0
    return sum(1 for t in trades if t.get("time_to_first_1R") is None) / n


def _missed_winner_rate(filtered_signals: list[dict[str, Any]], baseline_signals: list[dict[str, Any]]) -> float:
    base_2r = {str(s.get("entry_signal_key")) for s in baseline_signals if _safe_float(s.get("mfe_r")) >= 2.0}
    if not base_2r:
        return 0.0
    kept = {str(s.get("entry_signal_key")) for s in filtered_signals}
    return sum(1 for k in base_2r if k not in kept) / len(base_2r)


def _extract_debug_trades(eng: EthGeometryEngine | PartialExitEthGeometryEngine) -> list[dict[str, Any]]:
    return [dict(t, stop_size=STOP_FINAL, timeout_sec=TIMEOUT_METRICS) for t in eng.closed_trades]


def _build_filter_result_row(
    *,
    rule_id: str,
    config: RuleConfig,
    signal_set: list[dict[str, Any]],
    baseline_signals: list[dict[str, Any]],
    replay_row: dict[str, Any],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    out = _metric_default_dict()
    out.update(
        {
            "rule_set_id": rule_id,
            "thresholds": {
                "session": config.session_mode,
                "failed_absorption": config.failed_absorption_mode,
                "repeated_window_sec": config.repeated_window_sec,
                "tob_imbalance_lte": config.tob_lte,
                "liquidity_imbalance_lte": config.liquidity_lte,
                "event_rate_above": config.event_rate_filter,
                "trade_count_above": config.trade_count_filter,
            },
            "signal_count": len(signal_set),
            "trade_count": int(replay_row.get("trade_count") or 0),
            "percent_reaching_1R": float(replay_row.get("percent_reaching_1R") or 0.0),
            "percent_reaching_1_5R": float(replay_row.get("percent_reaching_1_5R") or 0.0),
            "percent_reaching_2R": float(replay_row.get("percent_reaching_2R") or 0.0),
            "clean_2r_winner_rate": float(replay_row.get("clean_2r_winner_rate") or 0.0),
            "median_mfe": float(replay_row.get("median_mfe") or 0.0),
            "p75_mfe": float(replay_row.get("p75_mfe") or 0.0),
            "average_mae": float(replay_row.get("average_mae") or 0.0),
            "average_time_to_1R": float(replay_row.get("average_time_to_1R") or 0.0),
            "average_time_to_2R": float(replay_row.get("average_time_to_2R") or 0.0),
            "gross_expectancy": float(replay_row.get("gross_expectancy") or 0.0),
            "net_expectancy_after_fees": float(replay_row.get("net_expectancy_after_fees") or 0.0),
            "net_profit_factor": float(replay_row.get("profit_factor_net") or 0.0),
            "max_drawdown_r": float(replay_row.get("max_drawdown_r") or 0.0),
            "max_losing_streak": int(replay_row.get("max_losing_streak") or 0),
            "false_positive_rate": _false_positive_rate(trades),
            "missed_winner_rate": _missed_winner_rate(signal_set, baseline_signals),
        }
    )
    return out


def _eval_fill_and_improvement(trade: dict[str, Any], model: str) -> tuple[bool, float, float]:
    spread = max(0.0, _safe_float(trade.get("entry_spread"), 0.0))
    seconds_to_mae = _safe_float(trade.get("seconds_to_mae"), 9e9)
    max_px = _safe_float(trade.get("max_price_reached"), _safe_float(trade.get("entry_price"), 0.0))
    entry = _safe_float(trade.get("entry_price"), 0.0)
    if model == "best_bid_ask_maker":
        return True, spread / 2.0, 0.0
    if seconds_to_mae > LIMIT_FILL_WINDOW_SEC:
        return False, 0.0, 0.0
    if model == "pullback_0_25R":
        level = entry + 0.25 * STOP_FINAL
        return (max_px >= level), 0.25 * STOP_FINAL, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    if model == "pullback_0_5R":
        level = entry + 0.5 * STOP_FINAL
        return (max_px >= level), 0.5 * STOP_FINAL, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    if model == "breakdown_retest":
        level = entry + max(spread / 2.0, 0.25 * STOP_FINAL)
        return (max_px >= level), (level - entry), min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    return False, 0.0, 0.0


def _limit_model_summary(*, trades: list[dict[str, Any]], execution_mode: str) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "execution_mode": execution_mode,
            "fill_rate": 0.0,
            "missed_winner_rate": 0.0,
            "adverse_selection_rate": 0.0,
            "average_entry_improvement": 0.0,
            "average_time_to_fill_sec": 0.0,
            "net_expectancy_after_realistic_fees": 0.0,
            "two_r_remains_reachable_rate": 0.0,
        }
    fills = 0
    missed_winners = 0
    adverse = 0
    improvements: list[float] = []
    fill_times: list[float] = []
    nets: list[float] = []
    two_r = 0
    base_winners = sum(1 for t in trades if _safe_float(t.get("mfe_r")) >= 2.0)
    for t in trades:
        if execution_mode == "A_market_entry_market_exit":
            filled, imp_usd, fill_time = True, 0.0, 0.0
        elif execution_mode == "B_maker_entry_taker_exit":
            filled, imp_usd, fill_time = _eval_fill_and_improvement(t, "best_bid_ask_maker")
        else:
            candidates = [
                _eval_fill_and_improvement(t, "pullback_0_25R"),
                _eval_fill_and_improvement(t, "pullback_0_5R"),
                _eval_fill_and_improvement(t, "breakdown_retest"),
            ]
            filled_candidates = [c for c in candidates if c[0]]
            if filled_candidates:
                filled, imp_usd, fill_time = max(filled_candidates, key=lambda x: x[1])
            else:
                filled, imp_usd, fill_time = False, 0.0, 0.0

        if not filled:
            if _safe_float(t.get("mfe_r")) >= 2.0:
                missed_winners += 1
            continue

        fills += 1
        imp_r = imp_usd / STOP_FINAL
        improvements.append(imp_r)
        fill_times.append(fill_time)
        mfe_new = _safe_float(t.get("mfe_r")) + imp_r
        mae_new = _safe_float(t.get("mae_r")) + imp_r
        if mfe_new >= 2.0:
            two_r += 1
        if mae_new <= -0.5:
            adverse += 1
        gross_new = min(2.0, max(-1.0, _safe_float(t.get("r_value")) + imp_r))
        if execution_mode == "A_market_entry_market_exit":
            fee_rate = ROUND_TRIP_TAKER_FEE
        elif execution_mode == "B_maker_entry_taker_exit":
            fee_rate = max(0.0, -0.0001 + (ROUND_TRIP_TAKER_FEE / 2.0))
        elif execution_mode == "C_limit_pullback_entry_taker_exit":
            fee_rate = max(0.0, -0.0001 + (ROUND_TRIP_TAKER_FEE / 2.0))
        else:
            exit_fee = -0.0001 if mfe_new >= 2.0 else (ROUND_TRIP_TAKER_FEE / 2.0)
            fee_rate = max(0.0, -0.0001 + exit_fee)
        nets.append(_per_trade_net_r(_safe_float(t.get("entry_price")), gross_new, fee_rate))
    return {
        "execution_mode": execution_mode,
        "fill_rate": fills / n,
        "missed_winner_rate": missed_winners / max(1, base_winners),
        "adverse_selection_rate": adverse / max(1, fills),
        "average_entry_improvement": mean(improvements) if improvements else 0.0,
        "average_time_to_fill_sec": mean(fill_times) if fill_times else 0.0,
        "net_expectancy_after_realistic_fees": mean(nets) if nets else 0.0,
        "two_r_remains_reachable_rate": two_r / max(1, fills),
    }


def _failed_absorption_comparison(
    *,
    objs: list[dict[str, Any]],
    data_source: str,
    symbol: str,
    fee_rate: float,
    baseline_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, pred in [
        ("double_signal_only", lambda f: bool(f.get("is_repeated_signal_120s"))),
        ("failed_absorption_loose", lambda f: bool(f.get("failed_absorption_loose"))),
        ("failed_absorption_medium", lambda f: bool(f.get("failed_absorption_medium"))),
        ("failed_absorption_strict", lambda f: bool(f.get("failed_absorption_strict"))),
    ]:
        eng = _replay_engine(
            objs,
            data_source=data_source,
            symbol=symbol,
            stop=STOP_FINAL,
            timeout_sec=TIMEOUT_METRICS,
            entry_filter=pred,
            tp_r=PART_D_TP_R,
            be_mode=PART_D_BE,
            partial_spec=None,
        )
        replay = _row_from_eng(
            eng,
            stop=STOP_FINAL,
            timeout_sec=TIMEOUT_METRICS,
            fee_rate=fee_rate,
            rule_name=name,
        )
        trades = _extract_debug_trades(eng)
        signals = [dict(s) for s in baseline_signals if pred(s)]
        rows.append(
            {
                "filter": name,
                "number_of_trades": int(replay.get("trade_count") or 0),
                "two_r_reach_rate": float(replay.get("percent_reaching_2R") or 0.0),
                "clean_two_r_rate": float(replay.get("clean_2r_winner_rate") or 0.0),
                "net_expectancy_after_fees": float(replay.get("net_expectancy_after_fees") or 0.0),
                "false_positive_rate": _false_positive_rate(trades),
                "missed_winner_rate": _missed_winner_rate(signals, baseline_signals),
            }
        )
    return {"rows": rows}


def run_pr_final_study(
    *,
    data_source: str = "bybit",
    raw_dir: Path = RAW_DATA_DIR,
    symbol: str = "ETHUSDT",
    fee_rate: float = ROUND_TRIP_TAKER_FEE,
    json_output: Path = REPORTS_DIR / "pr_final_entry_study.json",
    run_part_e: bool = True,
) -> dict[str, Any]:
    glob_pat = BYBIT_RAW_GLOB if data_source == "bybit" else RAW_GLOB
    objs = list(_iter_objects(raw_dir, glob_pat))
    if not objs:
        return {"error": "no_raw_objects", "raw_dir": str(raw_dir)}

    prev = os.environ.get("BYBIT_SYMBOL")
    os.environ["BYBIT_SYMBOL"] = str(symbol).strip().upper()
    try:
        LOGGER.info("PR-FINAL baseline replay stop=%s timeout=%s", STOP_FINAL, TIMEOUT_METRICS)
        base_eng = _replay_engine(
            objs,
            data_source=data_source,
            symbol=symbol,
            stop=STOP_FINAL,
            timeout_sec=TIMEOUT_METRICS,
            entry_filter=None,
            tp_r=PART_D_TP_R,
            be_mode=PART_D_BE,
            partial_spec=None,
        )
        baseline_trades = [
            dict(t, stop_size=STOP_FINAL, timeout_sec=TIMEOUT_METRICS) for t in base_eng.closed_trades
        ]
        baseline_signals = _dedupe_signals_keep_first(baseline_trades)
        th = _distribution_thresholds(baseline_signals)

        part_b_winners = _rank_2r_winners(baseline_trades)
        part_b_clean_preview = [dict(t) for t in part_b_winners if is_clean_2r_winner(t)][:50]

        baseline_row = _row_from_eng(
            base_eng,
            stop=STOP_FINAL,
            timeout_sec=TIMEOUT_METRICS,
            fee_rate=fee_rate,
            rule_name="pf_baseline_unfiltered",
        )

        filter_rows: list[dict[str, Any]] = []
        best_candidate: dict[str, Any] | None = None
        best_candidate_trades: list[dict[str, Any]] = []
        for cfg in _all_rule_configs():
            sid = cfg.rule_set_id()
            pred = _entry_rule_predicate(cfg, th)
            eng = _replay_engine(
                objs,
                data_source=data_source,
                symbol=symbol,
                stop=STOP_FINAL,
                timeout_sec=TIMEOUT_METRICS,
                entry_filter=pred,
                tp_r=PART_D_TP_R,
                be_mode=PART_D_BE,
                partial_spec=None,
            )
            replay_row = _row_from_eng(
                eng,
                stop=STOP_FINAL,
                timeout_sec=TIMEOUT_METRICS,
                fee_rate=fee_rate,
                rule_name=sid,
            )
            trades = _extract_debug_trades(eng)
            signals = [dict(s) for s in baseline_signals if pred(s)]
            row = _build_filter_result_row(
                rule_id=sid,
                config=cfg,
                signal_set=signals,
                baseline_signals=baseline_signals,
                replay_row=replay_row,
                trades=trades,
            )
            filter_rows.append(row)
            if (
                row["trade_count"] >= MIN_TRADES_GATE
                and row["net_expectancy_after_fees"] > 0.0
                and row["net_profit_factor"] > 1.2
                and (best_candidate is None or row["net_expectancy_after_fees"] > best_candidate["net_expectancy_after_fees"])
            ):
                best_candidate = row
                best_candidate_trades = trades

        if best_candidate is None:
            best_candidate = {
                "status": "fail",
                "reason": "no_rule_met_min_quality_threshold",
                "min_trades_gate": MIN_TRADES_GATE,
                "selection_logic": "net_expectancy_after_fees>0 and net_profit_factor>1.2",
            }
            best_candidate_trades = []

        limit_rows = [
            _limit_model_summary(trades=best_candidate_trades, execution_mode="A_market_entry_market_exit"),
            _limit_model_summary(trades=best_candidate_trades, execution_mode="B_maker_entry_taker_exit"),
            _limit_model_summary(trades=best_candidate_trades, execution_mode="C_limit_pullback_entry_taker_exit"),
            _limit_model_summary(trades=best_candidate_trades, execution_mode="D_limit_pullback_entry_limit_tp_if_feasible"),
        ]
        best_exec = max(limit_rows, key=lambda x: float(x.get("net_expectancy_after_realistic_fees") or -1e9))
        fa_compare = _failed_absorption_comparison(
            objs=objs,
            data_source=data_source,
            symbol=symbol,
            fee_rate=fee_rate,
            baseline_signals=baseline_signals,
        )

        pass_ev = float(best_exec.get("net_expectancy_after_realistic_fees") or 0.0) > 0.0
        pass_pf = float(best_candidate.get("net_profit_factor") or 0.0) > 1.2 if "net_profit_factor" in best_candidate else False
        pass_n = int(best_candidate.get("trade_count") or 0) >= MIN_TRADES_GATE if "trade_count" in best_candidate else False
        pass_dd = (
            float(best_candidate.get("max_drawdown_r") or 1e9) < 120.0
            and int(best_candidate.get("max_losing_streak") or 10**6) <= 12
            if "max_drawdown_r" in best_candidate
            else False
        )
        pass_fill = 0.2 <= float(best_exec.get("fill_rate") or 0.0) < 0.98
        pass_result = bool(pass_ev and pass_pf and pass_n and pass_dd and pass_fill)
        fail_assumption = None
        if not pass_result:
            if not pass_n:
                fail_assumption = "edge_only_exists_on_tiny_sample"
            elif not pass_fill:
                fail_assumption = "limit_fills_are_unrealistic"
            elif not pass_ev:
                fail_assumption = "net_expectancy_remains_negative"
            elif not pass_pf:
                fail_assumption = "failed_absorption_filter_does_not_materially_improve_outcomes"
            else:
                fail_assumption = "risk_profile_not_survivable"

        verdict = {
            "status": "PASS" if pass_result else "FAIL",
            "checks": {
                "net_expectancy_after_realistic_fees_gt_zero": pass_ev,
                "net_profit_factor_gt_1_2": pass_pf,
                "meaningful_sample_size": pass_n,
                "max_drawdown_and_losing_streak_survivable": pass_dd,
                "fill_model_realistic_not_fantasy": pass_fill,
            },
            "failed_assumption": fail_assumption,
        }

        doc: dict[str, Any] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "hypothesis": "imbalance -> absorption test -> failed absorption -> ride continuation ($2 -> 2R)",
            "constraints": {
                "pre_entry_filters_only": True,
                "no_future_leakage": True,
                "research_only": True,
            },
            "stop_final_usd": STOP_FINAL,
            "timeout_primary_metrics_sec": TIMEOUT_METRICS,
            "target_r_multiple": PART_D_TP_R,
            "distribution_thresholds_from_baseline_signals": th,
            "final_edge_study": {
                "baseline_summary": {
                    **_metric_default_dict(),
                    "signal_count": len(baseline_signals),
                    "trade_count": int(baseline_row.get("trade_count") or 0),
                    "percent_reaching_1R": float(baseline_row.get("percent_reaching_1R") or 0.0),
                    "percent_reaching_1_5R": float(baseline_row.get("percent_reaching_1_5R") or 0.0),
                    "percent_reaching_2R": float(baseline_row.get("percent_reaching_2R") or 0.0),
                    "clean_2r_winner_rate": float(baseline_row.get("clean_2r_winner_rate") or 0.0),
                    "median_mfe": float(baseline_row.get("median_mfe") or 0.0),
                    "p75_mfe": float(baseline_row.get("p75_mfe") or 0.0),
                    "average_mae": float(baseline_row.get("average_mae") or 0.0),
                    "average_time_to_1R": float(baseline_row.get("average_time_to_1R") or 0.0),
                    "average_time_to_2R": float(baseline_row.get("average_time_to_2R") or 0.0),
                    "gross_expectancy": float(baseline_row.get("gross_expectancy") or 0.0),
                    "net_expectancy_after_fees": float(baseline_row.get("net_expectancy_after_fees") or 0.0),
                    "net_profit_factor": float(baseline_row.get("profit_factor_net") or 0.0),
                    "max_drawdown_r": float(baseline_row.get("max_drawdown_r") or 0.0),
                    "max_losing_streak": int(baseline_row.get("max_losing_streak") or 0),
                },
                "filtered_entry_results": filter_rows,
                "failed_absorption_comparison": fa_compare,
                "limit_entry_feasibility": {
                    "fill_window_sec": LIMIT_FILL_WINDOW_SEC,
                    "results": limit_rows,
                    "best_execution_model": best_exec,
                },
                "best_candidate": best_candidate,
                "pass_fail_verdict": verdict,
            },
            "legacy_sections": {
                "ranked_2r_winners_count": len(part_b_winners),
                "clean_2r_preview_features_pre_entry": part_b_clean_preview,
                "run_part_e_requested": bool(run_part_e),
            },
        }

        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(_sanitize_doc(doc), indent=2, allow_nan=False), encoding="utf-8")
        LOGGER.info("PR-FINAL wrote %s", json_output)
        return doc
    finally:
        if prev is None:
            os.environ.pop("BYBIT_SYMBOL", None)
        else:
            os.environ["BYBIT_SYMBOL"] = prev


def main_inner(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="PR-FINAL entry filter replay")
    p.add_argument("--data-source", choices=("coinbase", "bybit"), default="bybit")
    p.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--fee-rate", type=float, default=ROUND_TRIP_TAKER_FEE)
    p.add_argument("--json-output", type=Path, default=REPORTS_DIR / "pr_final_entry_study.json")
    p.add_argument("--skip-part-e", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pr_final_study(
        data_source=args.data_source,
        raw_dir=args.raw_dir,
        symbol=args.symbol,
        fee_rate=float(args.fee_rate),
        json_output=args.json_output,
        run_part_e=not args.skip_part_e,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_inner())
