"""
Maker fill quality study: passive vs taker execution on $2 stop / 2R thesis (research only).

Consumed by `write_eth_geometry_reports` → `eth_geometry_summary.json`.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from tjtb.research.eth_geometry_runner import ROUND_TRIP_TAKER_FEE, _per_trade_net_r, EthGeometryResult
from tjtb.research.stop_grid_runner import _profit_factor

STOP_FOCUS = 2.0
LIMIT_FILL_WINDOW_SEC = 180.0
MAKER_ENTRY_REBATE = -0.0001
FEE_MARKET_ROUND_TRIP = ROUND_TRIP_TAKER_FEE
FEE_MAKER_ENTRY_TAKER_EXIT = max(0.0, MAKER_ENTRY_REBATE + ROUND_TRIP_TAKER_FEE / 2.0)


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _dedupe_stop2_trades(results: list[EthGeometryResult]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for r in results:
        if float(r.stop_size) != STOP_FOCUS:
            continue
        for t in r.trades:
            row = dict(t)
            row["stop_size"] = STOP_FOCUS
            row["timeout_sec"] = float(r.timeout_sec)
            k = str(row.get("entry_signal_key") or f"{row.get('entry_ts')}|{row.get('entry_price')}")
            cur = by_key.get(k)
            if cur is None or float(row.get("timeout_sec") or 0.0) > float(cur.get("timeout_sec") or 0.0):
                by_key[k] = row
    return list(by_key.values())


def _is_core_cohort(t: dict[str, Any]) -> bool:
    if str(t.get("side", "")).lower() != "short":
        return False
    rep = bool(t.get("is_repeated_signal_30s")) or bool(t.get("is_repeated_signal_60s"))
    fa = bool(t.get("failed_absorption_medium")) or bool(t.get("failed_absorption_strict"))
    return rep and fa


def _is_strict(t: dict[str, Any]) -> bool:
    return bool(t.get("failed_absorption_strict"))


def _is_medium_only(t: dict[str, Any]) -> bool:
    return bool(t.get("failed_absorption_medium")) and not bool(t.get("failed_absorption_strict"))


def _fill_pullback(t: dict[str, Any], *, frac: float) -> tuple[bool, float, float]:
    """Short: limit above entry by frac*stop. Fill if max_price reaches level within adverse window."""
    entry = _sf(t.get("entry_price"), 0.0)
    stop = STOP_FOCUS
    spread = max(0.0, _sf(t.get("entry_spread"), 0.0))
    seconds_to_mae = _sf(t.get("seconds_to_mae"), 9e9)
    max_px = _sf(t.get("max_price_reached"), entry)
    if seconds_to_mae > LIMIT_FILL_WINDOW_SEC:
        return False, 0.0, 0.0
    level = entry + frac * stop
    if max_px >= level:
        return True, frac * stop, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    return False, 0.0, 0.0


def _fill_absorption_retest(t: dict[str, Any]) -> tuple[bool, float, float]:
    """
    Model D: retest of breakdown / bid zone — limit above entry using spread + stall + bid fade proxy.
    """
    entry = _sf(t.get("entry_price"), 0.0)
    stop = STOP_FOCUS
    spread = max(0.0, _sf(t.get("entry_spread"), 0.0))
    seconds_to_mae = _sf(t.get("seconds_to_mae"), 9e9)
    max_px = _sf(t.get("max_price_reached"), entry)
    if seconds_to_mae > LIMIT_FILL_WINDOW_SEC:
        return False, 0.0, 0.0
    bid_vs_peak = _sf(t.get("entry_bid_size_vs_peak_ratio"), 1.0)
    fade = max(0.0, min(1.0, 1.0 - bid_vs_peak))
    stall = _sf(t.get("entry_mid_range_ratio"), 0.0)
    stall_boost = 0.15 * stop if stall < 0.0022 else 0.08 * stop
    level = entry + max(spread * 0.5, 0.25 * stop, fade * 0.35 * stop) + stall_boost
    imp = level - entry
    if max_px >= level:
        return True, imp, min(seconds_to_mae, LIMIT_FILL_WINDOW_SEC)
    return False, 0.0, 0.0


def _net_r_market(t: dict[str, Any]) -> float:
    return _per_trade_net_r(_sf(t.get("entry_price")), _sf(t.get("r_value")), FEE_MARKET_ROUND_TRIP)


def _net_r_maker_pullback(t: dict[str, Any], imp_r: float) -> float:
    gross = min(2.0, max(-1.0, _sf(t.get("r_value")) + imp_r))
    return _per_trade_net_r(_sf(t.get("entry_price")), gross, FEE_MAKER_ENTRY_TAKER_EXIT)


def _model_fill(
    t: dict[str, Any], model: str
) -> tuple[bool, float, float]:
    if model == "A_market_baseline":
        return True, 0.0, 0.0
    if model == "B_limit_0_25R_pullback":
        ok, imp_usd, ft = _fill_pullback(t, frac=0.25)
        return ok, imp_usd / STOP_FOCUS if STOP_FOCUS else 0.0, ft
    if model == "C_limit_0_50R_pullback":
        ok, imp_usd, ft = _fill_pullback(t, frac=0.5)
        return ok, imp_usd / STOP_FOCUS if STOP_FOCUS else 0.0, ft
    if model == "D_limit_absorption_retest":
        ok, imp_usd, ft = _fill_absorption_retest(t)
        return ok, imp_usd / STOP_FOCUS if STOP_FOCUS else 0.0, ft
    return False, 0.0, 0.0


def _pf_finite(rs: list[float]) -> float:
    if not rs:
        return 0.0
    v = float(_profit_factor(rs))
    return 0.0 if not math.isfinite(v) else v


def _evaluate_model(trades: list[dict[str, Any]], model: str) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "model": model,
            "signal_count": 0,
            "fill_rate": 0.0,
            "missed_winners": 0,
            "missed_winner_rate": 0.0,
            "adverse_selection": {
                "baseline_loser_rate_market_net": 0.0,
                "filled_loser_rate_net_after_fees": 0.0,
                "delta_filled_minus_baseline": 0.0,
            },
            "net_expectancy_after_fees_per_signal": 0.0,
            "net_expectancy_after_fees_filled_only": 0.0,
            "net_expectancy_market_baseline_per_signal": 0.0,
            "clean_2r_plus_rate_among_fills": 0.0,
            "two_r_reachable_rate_among_fills": 0.0,
            "profit_factor_net_filled": 0.0,
            "average_time_to_fill_sec": 0.0,
            "average_entry_improvement_r": 0.0,
        }

    baseline_nets = [_net_r_market(t) for t in trades]
    baseline_loser_rate = sum(1 for x in baseline_nets if x < 0) / n

    fills = 0
    missed_winners = 0
    base_winners_2r = sum(1 for t in trades if _sf(t.get("mfe_r")) >= 2.0)
    nets_all: list[float] = []
    nets_filled: list[float] = []
    fill_times: list[float] = []
    improvements: list[float] = []
    filled_losers = 0
    clean_2r_fills = 0
    two_r_fill = 0

    for t in trades:
        filled, imp_r, ft = _model_fill(t, model)
        base_net = _net_r_market(t)
        if model == "A_market_baseline":
            nets_all.append(base_net)
            nets_filled.append(base_net)
            fills += 1
            if _sf(t.get("mfe_r")) >= 2.0 and _sf(t.get("mae_r")) >= -0.5:
                t1, t2 = t.get("time_to_first_1R"), t.get("time_to_first_2R")
                if t1 is not None and t2 is not None and float(t1) <= 300 and float(t2) <= 900:
                    clean_2r_fills += 1
            if _sf(t.get("mfe_r")) >= 2.0:
                two_r_fill += 1
            continue

        if not filled:
            nets_all.append(0.0)
            if _sf(t.get("mfe_r")) >= 2.0:
                missed_winners += 1
            continue

        fills += 1
        net_f = _net_r_maker_pullback(t, imp_r)
        nets_all.append(net_f)
        nets_filled.append(net_f)
        fill_times.append(ft)
        improvements.append(imp_r)
        if net_f < 0:
            filled_losers += 1
        mfe_new = _sf(t.get("mfe_r")) + imp_r
        mae_new = _sf(t.get("mae_r")) + imp_r
        if mfe_new >= 2.0:
            two_r_fill += 1
        if (
            mfe_new >= 2.0
            and mae_new >= -0.5
            and t.get("time_to_first_1R") is not None
            and t.get("time_to_first_2R") is not None
            and float(t["time_to_first_1R"]) <= 300
            and float(t["time_to_first_2R"]) <= 900
        ):
            clean_2r_fills += 1

    fill_rate = fills / n
    filled_loser_rate = filled_losers / fills if fills else 0.0
    return {
        "model": model,
        "signal_count": n,
        "fill_rate": fill_rate,
        "missed_winners": missed_winners,
        "missed_winner_rate": missed_winners / max(1, base_winners_2r),
        "adverse_selection": {
            "baseline_loser_rate_market_net": baseline_loser_rate,
            "filled_loser_rate_net_after_fees": filled_loser_rate,
            "delta_filled_minus_baseline": filled_loser_rate - baseline_loser_rate,
        },
        "net_expectancy_after_fees_per_signal": mean(nets_all) if nets_all else 0.0,
        "net_expectancy_after_fees_filled_only": mean(nets_filled) if nets_filled else 0.0,
        "net_expectancy_market_baseline_per_signal": mean(baseline_nets) if baseline_nets else 0.0,
        "clean_2r_plus_rate_among_fills": clean_2r_fills / fills if fills else 0.0,
        "two_r_reachable_rate_among_fills": two_r_fill / fills if fills else 0.0,
        "profit_factor_net_filled": _pf_finite(nets_filled),
        "average_time_to_fill_sec": mean(fill_times) if fill_times else 0.0,
        "average_entry_improvement_r": mean(improvements) if improvements else 0.0,
    }


def _by_session(trades: list[dict[str, Any]], model: str) -> dict[str, Any]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        s = str(t.get("entry_session") or "unknown")
        sessions.setdefault(s, []).append(t)
    return {sess: _evaluate_model(sub, model) for sess, sub in sorted(sessions.items())}


def _by_failed_absorption_tier(trades: list[dict[str, Any]], model: str) -> dict[str, Any]:
    strict_list = [t for t in trades if _is_strict(t)]
    medium_list = [t for t in trades if _is_medium_only(t)]
    return {
        "strict_failed_absorption": _evaluate_model(strict_list, model),
        "medium_not_strict_failed_absorption": _evaluate_model(medium_list, model),
    }


def _fast_adverse_unfilled_pullback(trades: list[dict[str, Any]], model: str) -> float:
    """Share of signals where adverse move begins quickly but pullback fill never occurs (passive pain)."""
    if model == "A_market_baseline" or not trades:
        return 0.0
    bad = 0
    for t in trades:
        filled, _, _ = _model_fill(t, model)
        if filled:
            continue
        stm = _sf(t.get("seconds_to_mae"), 9e9)
        if stm <= 60.0:
            bad += 1
    return bad / len(trades)


def build_maker_fill_quality_study(results: list[EthGeometryResult]) -> dict[str, Any]:
    all_stop2 = _dedupe_stop2_trades(results)
    core = [t for t in all_stop2 if _is_core_cohort(t)]

    models = (
        "A_market_baseline",
        "B_limit_0_25R_pullback",
        "C_limit_0_50R_pullback",
        "D_limit_absorption_retest",
    )
    per_model = {m: _evaluate_model(core, m) for m in models}
    per_model_session = {m: _by_session(core, m) for m in models}
    per_model_fa_tier = {m: _by_failed_absorption_tier(core, m) for m in models}

    baseline_ev = per_model["A_market_baseline"]["net_expectancy_after_fees_per_signal"]
    best_maker_key = None
    best_maker_ev = -1e18
    for m in models:
        if m == "A_market_baseline":
            continue
        ev = per_model[m]["net_expectancy_after_fees_per_signal"]
        if ev > best_maker_ev:
            best_maker_ev = ev
            best_maker_key = m

    maker_improves = bool(best_maker_key is not None and best_maker_ev > baseline_ev + 1e-12)
    d_vs_b = per_model["D_limit_absorption_retest"]["net_expectancy_after_fees_per_signal"] - per_model[
        "B_limit_0_25R_pullback"
    ]["net_expectancy_after_fees_per_signal"]
    d_vs_c = per_model["D_limit_absorption_retest"]["net_expectancy_after_fees_per_signal"] - per_model[
        "C_limit_0_50R_pullback"
    ]["net_expectancy_after_fees_per_signal"]

    d_sess = per_model_session.get("D_limit_absorption_retest", {})
    asia_ev_d = float(d_sess.get("asia", {}).get("net_expectancy_after_fees_per_signal") or 0.0)
    sess_evs = [float(v.get("net_expectancy_after_fees_per_signal") or 0.0) for v in d_sess.values()]
    best_sess_ev = max(sess_evs) if sess_evs else 0.0
    asia_dominant_after_filter = bool(d_sess and math.isfinite(asia_ev_d) and asia_ev_d >= best_sess_ev - 1e-12)

    strict_core = [t for t in core if _is_strict(t)]
    strict_d_ev = _evaluate_model(strict_core, "D_limit_absorption_retest")["net_expectancy_after_fees_per_signal"]

    min_n = 15
    sufficient = len(core) >= min_n
    maker_positive = any(
        m != "A_market_baseline" and per_model[m]["net_expectancy_after_fees_per_signal"] > 0 for m in models
    )
    maker_beats_taker = bool(
        best_maker_key
        and per_model[best_maker_key]["net_expectancy_after_fees_per_signal"] > baseline_ev + 1e-12
    )
    fill_ok = bool(
        best_maker_key
        and per_model[best_maker_key]["fill_rate"] >= 0.12
        and per_model[best_maker_key]["fill_rate"] <= 0.95
    )
    not_worse_adverse = bool(
        best_maker_key
        and per_model[best_maker_key]["adverse_selection"]["delta_filled_minus_baseline"] <= 0.25
    )

    if sufficient and maker_positive and maker_beats_taker and fill_ok and not_worse_adverse:
        verdict = "PRODUCTION_CANDIDATE_EXISTS"
        failed_assumption = ""
    else:
        verdict = "STRATEGY_MUST_BE_REDESIGNED"
        if not sufficient:
            failed_assumption = "insufficient_sample_size_for_core_cohort"
        elif not maker_positive:
            failed_assumption = "no_execution_model_achieves_positive_net_expectancy_after_fees"
        elif not maker_beats_taker:
            failed_assumption = "maker_execution_does_not_beat_taker_baseline_on_per_signal_expectancy"
        elif not fill_ok:
            failed_assumption = "limit_fill_rate_unrealistic_or_too_sparse_for_execution"
        else:
            failed_assumption = "adverse_selection_on_fills_too_severe_vs_market_baseline"

    candidates: list[dict[str, Any]] = []
    for subset_name, pred in (
        ("core_double_signal_failed_absorption", _is_core_cohort),
        ("strict_failed_absorption_only", lambda t: _is_core_cohort(t) and _is_strict(t)),
        ("medium_not_strict_failed_absorption", lambda t: _is_core_cohort(t) and _is_medium_only(t)),
    ):
        sub = [t for t in all_stop2 if pred(t)]
        for m in models:
            row = _evaluate_model(sub, m)
            row["subset"] = subset_name
            candidates.append(row)
    candidates.sort(key=lambda r: float(r.get("net_expectancy_after_fees_per_signal") or 0.0), reverse=True)
    top_candidates = candidates[:12]

    fast_escape_pullback = {
        m: _fast_adverse_unfilled_pullback(core, m) for m in models if m != "A_market_baseline"
    }

    final_questions = {
        "does_maker_materially_improve_expectancy_vs_taker": maker_improves,
        "is_0_25R_pullback_too_aggressive": bool(
            per_model["B_limit_0_25R_pullback"]["fill_rate"] >= 0.55
            and per_model["B_limit_0_25R_pullback"]["missed_winner_rate"]
            > per_model["C_limit_0_50R_pullback"]["missed_winner_rate"]
        ),
        "is_0_50R_better_tradeoff_than_0_25R_despite_misses": bool(
            per_model["C_limit_0_50R_pullback"]["net_expectancy_after_fees_per_signal"]
            > per_model["B_limit_0_25R_pullback"]["net_expectancy_after_fees_per_signal"]
        ),
        "does_absorption_retest_outperform_fixed_pullbacks": bool(d_vs_b > 1e-12 and d_vs_c > 1e-12),
        "are_fast_escapes_killing_passive_execution": bool(
            max(fast_escape_pullback.values()) > 0.35 if fast_escape_pullback else False
        ),
        "is_asia_still_dominant_after_execution_filter": asia_dominant_after_filter,
        "can_strict_failed_absorption_plus_repeated_signal_plus_maker_D_produce_positive_net_ev": bool(strict_d_ev > 0),
        "is_there_a_real_production_candidate": verdict == "PRODUCTION_CANDIDATE_EXISTS",
    }

    study = {
        "focus": {
            "stop_size_usd": STOP_FOCUS,
            "target_r": 2.0,
            "cohort_definition": "short + (is_repeated_signal_30s OR is_repeated_signal_60s) + (failed_absorption_medium OR failed_absorption_strict)",
            "fill_window_sec": LIMIT_FILL_WINDOW_SEC,
            "fee_assumptions": {
                "market_round_trip_taker": FEE_MARKET_ROUND_TRIP,
                "maker_entry_rebate_per_unit_notional": MAKER_ENTRY_REBATE,
                "maker_entry_plus_taker_exit_effective": FEE_MAKER_ENTRY_TAKER_EXIT,
                "note": "Same fee convention as geometry net R: subtract entry_price * fee_rate from gross R in price units.",
            },
        },
        "cohort_counts": {
            "deduped_stop2_all_signals": len(all_stop2),
            "core_strategy_signals": len(core),
        },
        "models_compared": list(models),
        "per_model": per_model,
        "by_session_model_D_absorption_retest": per_model_session.get("D_limit_absorption_retest", {}),
        "by_failed_absorption_tier": per_model_fa_tier,
        "pullback_vs_retest_summary": {
            "D_minus_B_net_ev_per_signal": d_vs_b,
            "D_minus_C_net_ev_per_signal": d_vs_c,
            "B_fill_rate": per_model["B_limit_0_25R_pullback"]["fill_rate"],
            "C_fill_rate": per_model["C_limit_0_50R_pullback"]["fill_rate"],
            "D_fill_rate": per_model["D_limit_absorption_retest"]["fill_rate"],
        },
        "fast_escape_unfilled_rate_when_pullback_models": fast_escape_pullback,
        "final_questions_answered": final_questions,
    }

    adverse_selection_analysis = {
        "definition": {
            "baseline_loser_rate": "Among core cohort signals, fraction where market taker net R after fees < 0.",
            "filled_loser_rate": "Among filled limit entries, fraction where maker+taker net R < 0.",
            "adverse_selection_delta": "filled_loser_rate minus baseline_loser_rate (positive means fills are worse).",
        },
        "per_model": {m: per_model[m]["adverse_selection"] for m in models},
    }

    execution_verdict = {
        "verdict": verdict,
        "failed_assumption_if_redesigned": failed_assumption if verdict == "STRATEGY_MUST_BE_REDESIGNED" else "",
        "best_execution_model_key": best_maker_key or "none",
        "best_execution_model_net_ev_per_signal": float(best_maker_ev) if best_maker_key else 0.0,
        "market_baseline_net_ev_per_signal": float(baseline_ev),
        "gates_used": {
            "min_core_signals": min_n,
            "require_positive_maker_ev": True,
            "require_maker_ev_gt_market_ev": True,
            "require_fill_rate_between_0_12_and_0_95": True,
            "require_adverse_delta_lte_0_25": True,
        },
    }

    return {
        "maker_fill_quality_study": study,
        "maker_execution_candidates": {"top_by_net_expectancy_per_signal": top_candidates},
        "adverse_selection_analysis": adverse_selection_analysis,
        "execution_verdict": execution_verdict,
    }
