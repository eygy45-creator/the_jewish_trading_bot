from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import tjtb.live.live_paper_crypto as live
from tjtb.live.live_paper_crypto import LivePaperEngine
from tjtb.research.eth_geometry_runner import (
    DEFAULT_TIMEOUTS_SEC,
    EthGeometryResult,
    EthGeometryEngine,
    PartialExitEthGeometryEngine,
    ExcursionState,
    ROUND_TRIP_TAKER_FEE,
    compute_failed_absorption_entry_features,
    _anomaly_percentile_bucket,
    _average_fee_cost_r,
    _build_regime_attribution,
    _cohort_for_trade,
    _dedupe_signal_level,
    _excursion_metrics,
    _entry_session_label,
    _per_trade_net_r,
    _percentile,
    _summarize_attribution_group,
    run_eth_geometry_grid,
    write_eth_geometry_reports,
)
from tjtb.research.stop_grid_runner import _avg_notional_and_lev
from tjtb.research.pr_final_entry_study import is_clean_2r_winner, run_pr_final_study
from tjtb.research.ultimate_edge_study import (
    _sanitize_for_json,
    _summarize_signal_subset,
    run_ultimate_study,
)


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sample_eth_rows() -> list[dict]:
    return [
        {
            "payload": {
                "topic": "orderbook.50.ETHUSDT",
                "type": "snapshot",
                "ts": 1700000000000,
                "data": {"s": "ETHUSDT", "b": [["3000", "2.0"]], "a": [["3000.5", "1.0"]]},
            }
        },
        {
            "payload": {
                "topic": "publicTrade.ETHUSDT",
                "ts": 1700000000100,
                "data": [{"T": 1700000000100, "S": "Sell", "s": "ETHUSDT", "v": "1.0", "p": "3000.2"}],
            }
        },
    ]


def _top(ts: float, mid: float) -> live.TopState:
    return live.TopState(
        ts=ts,
        ts_text=f"2023-11-14T22:{int(ts):02d}:00+00:00",
        best_bid=mid - 0.25,
        best_ask=mid + 0.25,
        best_bid_sz=1.0,
        best_ask_sz=1.0,
        spread=0.5,
        mid=mid,
        micro_dev=0.0,
        tob_imb=0.0,
    )


def test_eth_geometry_matrix_executes(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    stops = [1.0, 2.0]
    timeouts = [2.0, 5.0]
    res = run_eth_geometry_grid(
        data_source="bybit",
        raw_dir=tmp_path,
        stops=stops,
        timeouts_sec=timeouts,
        symbol="ETHUSDT",
    )
    assert len(res) == len(stops) * len(timeouts)
    for r in res:
        assert isinstance(r, EthGeometryResult)
        assert r.round_trip_fee_rate == ROUND_TRIP_TAKER_FEE


def test_eth_geometry_output_files(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], timeouts_sec=[2.0])
    csv_out = tmp_path / "eth_geometry_results.csv"
    json_out = tmp_path / "eth_geometry_summary.json"
    summary = write_eth_geometry_reports(res, csv_out, json_out)
    assert csv_out.is_file()
    assert json_out.is_file()
    assert "best_by_raw_expectancy" in summary
    assert "best_by_survivability" in summary
    assert "best_realistic_execution_candidate" in summary
    assert "excursion_analysis" in summary


def test_default_timeout_grid_includes_long_windows():
    assert DEFAULT_TIMEOUTS_SEC == [120.0, 180.0, 300.0, 480.0, 600.0, 900.0]


def test_default_timeout_grid_is_used_by_runner(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], symbol="ETHUSDT")
    assert len(res) == len(DEFAULT_TIMEOUTS_SEC)
    assert sorted({r.timeout_sec for r in res}) == DEFAULT_TIMEOUTS_SEC


def test_default_matrix_size_matches_geometry_grid(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    default_stops = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    res = run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, symbol="ETHUSDT")
    assert len(res) == len(default_stops) * len(DEFAULT_TIMEOUTS_SEC)


def test_leverage_math_sanity_eth():
    avg_notional, lev = _avg_notional_and_lev([3000.0, 3010.0], stop_size=2.0, account=10_000.0)
    assert avg_notional > 0
    assert lev > 0


def test_fee_math_sanity():
    entry = 3000.0
    gross = 10.0
    net = _per_trade_net_r(entry, gross, ROUND_TRIP_TAKER_FEE)
    assert net == pytest.approx(gross - entry * ROUND_TRIP_TAKER_FEE)
    avg_fr = _average_fee_cost_r([entry], stop_size=10.0)
    assert avg_fr == pytest.approx((ROUND_TRIP_TAKER_FEE * entry) / 10.0)


def test_short_excursion_mfe_and_mae_are_direction_aware():
    state = ExcursionState(trade_id=1, side="short", entry_ts=0.0, entry_price=2000.0, stop_distance=10.0, timeout_sec=900.0)
    for ts, price in [(1.0, 1996.0), (2.0, 1994.0), (3.0, 1998.0), (4.0, 2007.0)]:
        state.update(ts, price)
    assert state.mfe_r == pytest.approx(0.6)
    assert state.mae_r == pytest.approx(-0.7)
    assert state.seconds_to_mfe == pytest.approx(2.0)
    assert state.seconds_to_mae == pytest.approx(4.0)
    assert state.min_price_reached == pytest.approx(1994.0)
    assert state.max_price_reached == pytest.approx(2007.0)


def test_long_excursion_mfe_and_mae_are_direction_aware():
    state = ExcursionState(trade_id=1, side="long", entry_ts=0.0, entry_price=2000.0, stop_distance=10.0, timeout_sec=900.0)
    for ts, price in [(1.0, 2004.0), (2.0, 2006.0), (3.0, 2001.0), (4.0, 1993.0)]:
        state.update(ts, price)
    assert state.mfe_r == pytest.approx(0.6)
    assert state.mae_r == pytest.approx(-0.7)
    assert state.seconds_to_mfe == pytest.approx(2.0)
    assert state.seconds_to_mae == pytest.approx(4.0)
    assert state.min_price_reached == pytest.approx(1993.0)
    assert state.max_price_reached == pytest.approx(2006.0)


def test_excursion_tracking_continues_after_breakeven_exit():
    eng = EthGeometryEngine(logging.getLogger("tjtb.test.excursion"), data_source="bybit", stop_size=1.0, timeout_sec=900.0)
    eng._take_trade(_top(0.0, 100.0), "normal", tp_r=2.0, be_trigger=1.0)
    eng._maybe_manage_open_trade(_top(60.0, 99.0))
    assert eng.open_trade is not None
    assert eng.open_trade["sl_price"] == pytest.approx(100.0)
    eng._maybe_manage_open_trade(_top(120.0, 100.0))
    assert eng.open_trade is None
    assert eng.closed_trades[0]["outcome"] == "sl_or_be"
    assert eng.closed_trades[0]["mfe_r"] == pytest.approx(1.0)
    eng._maybe_manage_open_trade(_top(300.0, 98.8))
    eng._maybe_manage_open_trade(_top(900.0, 100.0))
    assert eng.closed_trades[0]["mfe_r"] == pytest.approx(1.2)
    assert eng.closed_trades[0]["time_to_first_1R"] == pytest.approx(60.0)


def test_excursion_percentiles_are_interpolated():
    assert _percentile([0.0, 1.0, 2.0, 3.0], 50) == pytest.approx(1.5)
    assert _percentile([0.0, 1.0, 2.0, 3.0], 75) == pytest.approx(2.25)
    assert _percentile([-3.0, -2.0, -1.0, 0.0], 90) == pytest.approx(-0.3)


def test_excursion_reachability_and_time_metrics():
    trades = [
        {"mfe_r": 0.2, "mae_r": 0.0, "seconds_to_mfe": 10.0, "time_to_first_0_25R": None, "time_to_first_0_5R": None, "time_to_first_1R": None},
        {"mfe_r": 0.6, "mae_r": -0.1, "seconds_to_mfe": 20.0, "time_to_first_0_25R": 5.0, "time_to_first_0_5R": 12.0, "time_to_first_1R": None},
        {"mfe_r": 1.2, "mae_r": -0.3, "seconds_to_mfe": 40.0, "time_to_first_0_25R": 4.0, "time_to_first_0_5R": 9.0, "time_to_first_1R": 35.0},
    ]
    metrics = _excursion_metrics(trades)
    assert metrics["reachability"]["percent_reaching_0_25R"] == pytest.approx(2 / 3)
    assert metrics["reachability"]["percent_reaching_0_5R"] == pytest.approx(2 / 3)
    assert metrics["reachability"]["percent_reaching_1R"] == pytest.approx(1 / 3)
    assert metrics["average_time_to_0_25R"] == pytest.approx(4.5)
    assert metrics["average_time_to_0_5R"] == pytest.approx(10.5)
    assert metrics["average_time_to_1R"] == pytest.approx(35.0)
    assert metrics["average_time_to_peak_mfe"] == pytest.approx(70.0 / 3)


def test_excursion_null_values_when_targets_never_reached():
    state = ExcursionState(trade_id=1, side="short", entry_ts=0.0, entry_price=100.0, stop_distance=10.0, timeout_sec=900.0)
    state.update(5.0, 98.0)
    out = state.to_output()
    assert out["mfe_r"] == pytest.approx(0.2)
    assert out["time_to_first_0_25R"] is None
    assert out["time_to_first_0_5R"] is None
    assert out["time_to_first_2R"] is None


def test_regime_attribution_cohort_assignment():
    assert _cohort_for_trade({"mfe_r": 0.8, "mae_r": -0.2, "time_to_first_0_5R": 120.0}) == "tier_a_fast_clean_winner"
    assert _cohort_for_trade({"mfe_r": 1.2, "mae_r": -0.8, "time_to_first_0_5R": 300.0}) == "tier_b_slow_noisy_winner"
    assert _cohort_for_trade({"mfe_r": 0.4, "mae_r": -0.1, "time_to_first_0_5R": None}) == "tier_c_dead_signal"
    assert _cohort_for_trade({"mfe_r": 0.6, "mae_r": -0.1, "time_to_first_0_5R": 200.0}) == "uncategorized_middle"


def test_regime_attribution_by_variant_aggregation():
    trades = [
        {"mfe_r": 0.8, "mae_r": -0.2, "time_to_first_0_5R": 120.0},
        {"mfe_r": 1.2, "mae_r": -0.8, "time_to_first_0_5R": 300.0},
        {"mfe_r": 0.4, "mae_r": -0.1, "time_to_first_0_5R": None},
    ]
    summary = _summarize_attribution_group(trades)
    assert summary["trade_count"] == 3
    assert summary["tier_a_count"] == 1
    assert summary["tier_b_count"] == 1
    assert summary["tier_c_count"] == 1
    assert summary["tier_a_rate"] == pytest.approx(1 / 3)
    assert summary["median_mfe"] == pytest.approx(0.8)
    assert summary["p75_mfe"] == pytest.approx(1.0)


def test_signal_level_deduplication_prefers_longest_timeout_then_mfe():
    trades = [
        {"entry_signal_key": "sig1", "stop_size": 1.0, "timeout_sec": 120.0, "mfe_price_move": 1.0, "mfe_r": 1.0, "mae_r": -0.2, "time_to_first_0_5R": 30.0},
        {"entry_signal_key": "sig1", "stop_size": 2.0, "timeout_sec": 900.0, "mfe_price_move": 0.8, "mfe_r": 0.4, "mae_r": -0.1, "time_to_first_0_5R": None},
        {"entry_signal_key": "sig1", "stop_size": 1.0, "timeout_sec": 900.0, "mfe_price_move": 2.0, "mfe_r": 2.0, "mae_r": -0.2, "time_to_first_0_5R": 20.0},
        {"entry_signal_key": "sig2", "stop_size": 1.0, "timeout_sec": 120.0, "mfe_price_move": 0.1, "mfe_r": 0.1, "mae_r": -0.2, "time_to_first_0_5R": None},
    ]
    deduped = sorted(_dedupe_signal_level(trades), key=lambda x: x["entry_signal_key"])
    assert len(deduped) == 2
    assert deduped[0]["entry_signal_key"] == "sig1"
    assert deduped[0]["timeout_sec"] == 900.0
    assert deduped[0]["mfe_price_move"] == pytest.approx(2.0)
    assert deduped[0]["variants_seen"] == 4


def test_anomaly_percentile_bucket_grouping():
    assert _anomaly_percentile_bucket(0.9901) == "99.0-99.5"
    assert _anomaly_percentile_bucket(0.996) == "99.5-99.9"
    assert _anomaly_percentile_bucket(0.999) == "99.9+"
    assert _anomaly_percentile_bucket(None) == "unknown"


def test_entry_session_classification():
    assert _entry_session_label(0) == "asia"
    assert _entry_session_label(7 * 3600) == "london"
    assert _entry_session_label(13 * 3600) == "london_ny_overlap"
    assert _entry_session_label(17 * 3600) == "new_york"
    assert _entry_session_label(22 * 3600) == "off_hours"


def test_clustering_detection_counts_prior_only():
    eng = EthGeometryEngine(logging.getLogger("tjtb.test.cluster"), data_source="bybit", stop_size=1.0, timeout_sec=900.0)
    eng._qualifying_signal_times.extend([100.0, 170.0, 210.0])
    counts = eng._prior_cluster_counts(220.0)
    assert counts["prior_qualifying_anomalies_30s"] == 1
    assert counts["prior_qualifying_anomalies_60s"] == 2
    assert counts["prior_qualifying_anomalies_120s"] == 3


def test_pre_entry_feature_capture_contains_only_current_and_prior_data():
    eng = EthGeometryEngine(logging.getLogger("tjtb.test.features"), data_source="bybit", stop_size=1.0, timeout_sec=900.0)
    top = _top(220.0, 100.0)
    counts = {"prior_qualifying_anomalies_30s": 1, "prior_qualifying_anomalies_60s": 2, "prior_qualifying_anomalies_120s": 3}
    features = eng._entry_features(
        top=top,
        pressure=-5.0,
        event_rate=3.0,
        trade_count=4.0,
        mid_vol=0.2,
        z_tob=-2.0,
        z_micro=-1.5,
        z_pressure=-2.5,
        z_event=1.2,
        z_trade=0.3,
        z_spread=0.1,
        z_mid_vol=0.4,
        anomaly_score=2.5,
        anomaly_percentile=0.999,
        direction="bearish",
        regime="volatile",
        cluster_counts=counts,
    )
    assert features["entry_anomaly_percentile"] == pytest.approx(0.999)
    assert features["entry_anomaly_score"] == pytest.approx(2.5)
    assert features["entry_signed_book_pressure"] == pytest.approx(-5.0)
    assert features["entry_z_pressure"] == pytest.approx(-2.5)
    assert features["prior_qualifying_anomalies_120s"] == 3
    assert features["is_repeated_signal_30s"] is True
    assert "mfe_r" not in features
    assert "exit_price" not in features


def test_regime_attribution_summary_sections_present():
    result = EthGeometryResult(
        stop_size=1.0,
        timeout_sec=120.0,
        total_trades=2,
        win_rate=0.5,
        tp_rate=0.0,
        timeout_rate=1.0,
        average_r=0.0,
        total_realized_r=0.0,
        max_drawdown_r=0.0,
        max_losing_streak=0,
        profit_factor_net=0.0,
        average_trade_duration_sec=120.0,
        average_notional_required=0.0,
        leverage_required_5k=0.0,
        leverage_required_10k=0.0,
        leverage_required_50k=0.0,
        round_trip_fee_rate=ROUND_TRIP_TAKER_FEE,
        average_round_trip_fee_usd=0.0,
        average_fee_cost_r=0.0,
        net_average_r_after_fees=0.0,
        net_total_r_after_fees=0.0,
        trades=[
            {
                "entry_signal_key": "a",
                "regime": "volatile",
                "entry_session": "new_york",
                "entry_anomaly_percentile": 0.999,
                "mfe_r": 0.8,
                "mae_r": -0.2,
                "time_to_first_0_5R": 100.0,
                "mfe_price_move": 0.8,
                "stop_size": 1.0,
                "timeout_sec": 120.0,
                "prior_qualifying_anomalies_120s": 1,
                "prior_qualifying_anomalies_60s": 1,
                "prior_qualifying_anomalies_30s": 0,
            },
            {
                "entry_signal_key": "b",
                "regime": "calm",
                "entry_session": "asia",
                "entry_anomaly_percentile": 0.991,
                "mfe_r": 0.2,
                "mae_r": -0.4,
                "time_to_first_0_5R": None,
                "mfe_price_move": 0.2,
                "stop_size": 1.0,
                "timeout_sec": 120.0,
                "prior_qualifying_anomalies_120s": 0,
                "prior_qualifying_anomalies_60s": 0,
                "prior_qualifying_anomalies_30s": 0,
            },
        ],
    )
    summary = _build_regime_attribution([result])
    assert "cohort_definition" in summary
    assert "by_variant" in summary
    assert "signal_level" in summary
    assert "by_regime" in summary
    assert "by_anomaly_percentile_bucket" in summary
    assert "feature_comparison" in summary
    assert "session_attribution" in summary
    assert "signal_clustering" in summary
    assert "liquidity_wall_attribution" in summary
    assert summary["by_variant"][0]["tier_a_count"] == 1


def test_invalid_stop_rejected(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    with pytest.raises(ValueError):
        run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0, -1.0], timeouts_sec=[2.0])


def test_invalid_timeout_rejected(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    with pytest.raises(ValueError):
        run_eth_geometry_grid(data_source="bybit", raw_dir=tmp_path, stops=[1.0], timeouts_sec=[2.0, 0.0])


def test_summarize_signal_subset_tier_a():
    trades = [
        {"mfe_r": 0.8, "mae_r": -0.2, "time_to_first_0_5R": 100.0, "time_to_first_1R": None},
    ]
    s = _summarize_signal_subset(trades)
    assert s["signal_count"] == 1
    assert s["tier_a_rate"] == pytest.approx(1.0)


def test_sanitize_for_json_non_finite():
    assert _sanitize_for_json({"pf": float("inf")})["pf"] is None


def test_run_ultimate_study_no_raw_files(tmp_path):
    doc = run_ultimate_study(raw_dir=tmp_path, data_source="bybit", run_phase2=False, run_phase3=False)
    assert doc.get("error") == "no_raw_objects"


def test_partial_exit_engine_registers_flags():
    eng = PartialExitEthGeometryEngine(
        logging.getLogger("tjtb.test.partial"),
        "bybit",
        stop_size=2.0,
        timeout_sec=900.0,
        research_fixed_tp_r=2.0,
        research_be_mode="none",
    )
    top = _top(0.0, 100.0)
    eng._take_trade(top, "normal", 2.0, None)
    assert eng.open_trade is not None
    assert eng.open_trade.get("partial_done") is False


def test_compute_failed_absorption_score_bounded():
    top = live.TopState(
        ts=100.0,
        ts_text="2023-11-14T22:00:00+00:00",
        best_bid=99.0,
        best_ask=101.0,
        best_bid_sz=0.5,
        best_ask_sz=50.0,
        spread=2.0,
        mid=100.0,
        micro_dev=-0.1,
        tob_imb=-0.98,
    )
    out = compute_failed_absorption_entry_features(
        top=top,
        pressure=-120.0,
        event_rate=200.0,
        trade_count=100.0,
        mid_vals=[100.0, 99.98, 100.01],
        bid_peak_recent=40.0,
        liquidity_imbalance=-0.95,
        z_pressure=-4.0,
        z_tob=-3.0,
        z_event=2.0,
        z_trade=2.0,
        is_repeated_signal_30s=True,
        is_repeated_signal_60s=True,
        is_repeated_signal_120s=True,
    )
    assert 0.0 <= out["entry_failed_absorption_score"] <= 100.0
    assert out["failed_absorption_strict"] in (True, False)


def test_clean_2r_winner_classification():
    ok = {
        "stop_size": 2.0,
        "mfe_r": 2.1,
        "mae_r": -0.4,
        "time_to_first_1R": 100.0,
        "time_to_first_2R": 400.0,
    }
    bad_mae = {**ok, "mae_r": -0.6}
    assert is_clean_2r_winner(ok)
    assert not is_clean_2r_winner(bad_mae)


def test_run_pr_final_empty_raw(tmp_path):
    doc = run_pr_final_study(raw_dir=tmp_path, run_part_e=False)
    assert doc.get("error") == "no_raw_objects"


def test_pr_final_final_edge_study_contract(tmp_path):
    raw = tmp_path / "bybit_20260501.ndjson"
    _write_ndjson(raw, _sample_eth_rows())
    doc = run_pr_final_study(raw_dir=tmp_path, data_source="bybit", run_part_e=False)
    fes = doc.get("final_edge_study")
    assert isinstance(fes, dict)
    assert fes.get("baseline_summary") is not None
    assert fes.get("filtered_entry_results") is not None
    assert fes.get("failed_absorption_comparison") is not None
    assert fes.get("limit_entry_feasibility") is not None
    assert fes.get("pass_fail_verdict") is not None
    best = fes.get("best_candidate")
    assert best is not None
    assert ("trade_count" in best) or (best.get("status") == "fail")
    lim = fes["limit_entry_feasibility"]
    assert isinstance(lim.get("results"), list)
    assert len(lim["results"]) >= 1
    assert doc["constraints"]["no_future_leakage"] is True


def test_eth_geometry_research_overrides_tp_be():
    eng = EthGeometryEngine(
        logging.getLogger("tjtb.test.override"),
        "bybit",
        stop_size=1.0,
        timeout_sec=2.0,
        research_fixed_tp_r=1.25,
        research_be_mode="none",
    )
    assert eng.research_fixed_tp_r == pytest.approx(1.25)
    assert eng.research_be_mode == "none"


def test_production_defaults_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "PAPER_TRADES_PATH", tmp_path / "paper_trades.csv")
    monkeypatch.setattr(live, "OPPORTUNITIES_PATH", tmp_path / "opportunities.csv")
    assert live.TIMEOUT_SEC == 2.0
    eng = LivePaperEngine(logging.getLogger("tjtb.test.eth_geom"))
    top = live.TopState(
        ts=1700000000.0,
        ts_text="2023-11-14T22:13:20+00:00",
        best_bid=100.0,
        best_ask=100.5,
        best_bid_sz=1.0,
        best_ask_sz=1.0,
        spread=0.5,
        mid=100.25,
        micro_dev=0.0,
        tob_imb=0.0,
    )
    eng._take_trade(top, "normal", 2.0, 1.0)
    assert eng.open_trade is not None
    assert eng.open_trade["sl_price"] == top.mid + 1.0
