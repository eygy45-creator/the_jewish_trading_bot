"""Research runner: ETHUSDT stop-loss × timeout grid with leverage and taker-fee drag.

Use `--study ultimate` or `--study pr-final` for research batches (`ultimate_edge_study`,
`pr_final_entry_study`). Does not modify live execution."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from tjtb.live.live_paper_crypto import BYBIT_RAW_GLOB, LOOKBACK_SEC, RAW_GLOB, TopState
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
EXCURSION_LEVELS_R = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
EXCURSION_TIME_FIELDS = {
    0.25: "time_to_first_0_25R",
    0.5: "time_to_first_0_5R",
    0.75: "time_to_first_0_75R",
    1.0: "time_to_first_1R",
    1.5: "time_to_first_1_5R",
    2.0: "time_to_first_2R",
}
CLUSTER_WINDOWS_SEC = [30.0, 60.0, 120.0]


def compute_failed_absorption_entry_features(
    *,
    top: TopState,
    pressure: float,
    event_rate: float,
    trade_count: float,
    mid_vals: list[float],
    bid_peak_recent: float,
    liquidity_imbalance: float | None,
    z_pressure: float | None,
    z_tob: float | None,
    z_event: float | None,
    z_trade: float | None,
    is_repeated_signal_30s: bool,
    is_repeated_signal_60s: bool,
    is_repeated_signal_120s: bool,
) -> dict[str, Any]:
    """
    Pre-entry-only proxy for bearish failed absorption (research).

    Uses current TOB/L2/trade-intensity z-scores, recent mid range (stall proxy),
    bid-size fade vs recent peak, and anomaly clustering — no post-entry paths.
    """
    score = 0.0
    mid = float(top.mid)
    # Stage 1 — aggressive flow into bids
    if z_pressure is not None and z_pressure < -1.5:
        score += min(28.0, (-z_pressure - 1.5) * 7.0)
    if float(top.tob_imb) <= -0.88:
        score += 18.0
    elif float(top.tob_imb) <= -0.78:
        score += 12.0
    elif float(top.tob_imb) <= -0.65:
        score += 6.0
    if pressure < -80.0:
        score += 10.0
    elif pressure < -40.0:
        score += 5.0
    if z_event is not None and z_event > 0.8:
        score += min(14.0, (z_event - 0.8) * 6.0)
    if z_trade is not None and z_trade > 0.8:
        score += min(14.0, (z_trade - 0.8) * 6.0)
    raw_act = event_rate + trade_count / max(15.0, 1e-9)
    if raw_act > 120.0:
        score += 8.0
    elif raw_act > 70.0:
        score += 4.0

    # Stage 2 — stall / absorption test (tight mid range in lookback)
    mid_range_ratio = 0.0
    if len(mid_vals) >= 3 and mid > 0:
        mid_range_ratio = (max(mid_vals) - min(mid_vals)) / mid
    stall_tight = mid_range_ratio < 0.0011
    stall_moderate = mid_range_ratio < 0.0022
    if stall_tight:
        score += 18.0
    elif stall_moderate:
        score += 10.0

    # Stage 3 — bid liquidity fade + bearish imbalance deepening
    bb = float(top.best_bid_sz)
    peak = max(float(bid_peak_recent), 1e-12)
    bid_vs_peak = bb / peak
    if bid_vs_peak < 0.68:
        score += 22.0
    elif bid_vs_peak < 0.82:
        score += 13.0
    elif bid_vs_peak < 0.92:
        score += 6.0

    if liquidity_imbalance is not None:
        if liquidity_imbalance <= -0.82:
            score += 10.0
        elif liquidity_imbalance <= -0.68:
            score += 5.0

    low_mid = min(mid_vals) if mid_vals else mid
    at_local_low = mid <= low_mid + mid * 1e-7
    if at_local_low and stall_moderate:
        score += 8.0

    score = max(0.0, min(100.0, score))

    loose = score >= 34.0 and (is_repeated_signal_120s or score >= 52.0)
    medium = score >= 46.0 and is_repeated_signal_60s and bid_vs_peak < 0.92 and stall_moderate
    strict = (
        score >= 62.0
        and is_repeated_signal_30s
        and stall_tight
        and bid_vs_peak < 0.78
        and float(top.tob_imb) <= -0.78
    )

    return {
        "entry_failed_absorption_score": score,
        "entry_mid_range_ratio": mid_range_ratio,
        "entry_bid_size_vs_peak_ratio": bid_vs_peak,
        "failed_absorption_loose": loose,
        "failed_absorption_medium": medium,
        "failed_absorption_strict": strict,
    }


ATTRIBUTION_FEATURE_FIELDS = [
    "entry_anomaly_percentile",
    "entry_anomaly_score",
    "entry_direction",
    "entry_regime",
    "entry_z_tob",
    "entry_z_micro",
    "entry_z_pressure",
    "entry_z_event_rate",
    "entry_z_trade_count",
    "entry_z_spread",
    "entry_z_mid_vol",
    "entry_tob_imbalance",
    "entry_microprice_deviation",
    "entry_signed_book_pressure",
    "entry_event_rate",
    "entry_trade_count",
    "entry_spread",
    "entry_mid_price_volatility",
    "entry_mid_price",
    "entry_best_bid_size",
    "entry_best_ask_size",
    "entry_bid_ask_size_ratio",
    "entry_liquidity_imbalance",
    "entry_session",
    "entry_signal_key",
    "prior_qualifying_anomalies_30s",
    "prior_qualifying_anomalies_60s",
    "prior_qualifying_anomalies_120s",
    "is_repeated_signal_30s",
    "is_repeated_signal_60s",
    "is_repeated_signal_120s",
    "entry_failed_absorption_score",
    "entry_mid_range_ratio",
    "entry_bid_size_vs_peak_ratio",
    "failed_absorption_loose",
    "failed_absorption_medium",
    "failed_absorption_strict",
]


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
    trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExcursionState:
    trade_id: int
    side: str
    entry_ts: float
    entry_price: float
    stop_distance: float
    timeout_sec: float
    mfe_r: float = 0.0
    mae_r: float = 0.0
    mfe_price_move: float = 0.0
    mae_price_move: float = 0.0
    seconds_to_mfe: float = 0.0
    seconds_to_mae: float = 0.0
    max_price_reached: float = 0.0
    min_price_reached: float = 0.0
    finalized: bool = False
    time_to_first_0_25R: float | None = None
    time_to_first_0_5R: float | None = None
    time_to_first_0_75R: float | None = None
    time_to_first_1R: float | None = None
    time_to_first_1_5R: float | None = None
    time_to_first_2R: float | None = None

    def __post_init__(self) -> None:
        self.side = str(self.side).lower()
        if self.side not in {"short", "long"}:
            raise ValueError("side must be 'short' or 'long'")
        if self.stop_distance <= 0:
            raise ValueError("stop_distance must be > 0")
        self.max_price_reached = self.entry_price
        self.min_price_reached = self.entry_price

    def update(self, ts: float, price: float) -> None:
        if self.finalized:
            return
        elapsed = max(0.0, float(ts) - self.entry_ts)
        self.max_price_reached = max(self.max_price_reached, float(price))
        self.min_price_reached = min(self.min_price_reached, float(price))

        if self.side == "short":
            favorable_r = (self.entry_price - self.min_price_reached) / self.stop_distance
            adverse_r = (self.max_price_reached - self.entry_price) / self.stop_distance
        else:
            favorable_r = (self.max_price_reached - self.entry_price) / self.stop_distance
            adverse_r = (self.entry_price - self.min_price_reached) / self.stop_distance

        if favorable_r > self.mfe_r:
            self.mfe_r = favorable_r
            self.mfe_price_move = max(0.0, favorable_r * self.stop_distance)
            self.seconds_to_mfe = elapsed
        mae_r = -max(0.0, adverse_r)
        if mae_r < self.mae_r:
            self.mae_r = mae_r
            self.mae_price_move = abs(mae_r) * self.stop_distance
            self.seconds_to_mae = elapsed

        for level, field_name in EXCURSION_TIME_FIELDS.items():
            if favorable_r >= level and getattr(self, field_name) is None:
                setattr(self, field_name, elapsed)

        if elapsed >= self.timeout_sec:
            self.finalized = True

    def to_output(self) -> dict[str, Any]:
        return {
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "mfe_price_move": self.mfe_price_move,
            "mae_price_move": self.mae_price_move,
            "seconds_to_mfe": self.seconds_to_mfe,
            "seconds_to_mae": self.seconds_to_mae,
            "max_price_reached": self.max_price_reached,
            "min_price_reached": self.min_price_reached,
            "time_to_first_0_25R": self.time_to_first_0_25R,
            "time_to_first_0_5R": self.time_to_first_0_5R,
            "time_to_first_0_75R": self.time_to_first_0_75R,
            "time_to_first_1R": self.time_to_first_1R,
            "time_to_first_1_5R": self.time_to_first_1_5R,
            "time_to_first_2R": self.time_to_first_2R,
        }


class EthGeometryEngine(StopGridEngine):
    """Same signal/exits as StopGridEngine; configurable timeout seconds."""

    def __init__(
        self,
        logger: logging.Logger,
        data_source: str,
        stop_size: float,
        timeout_sec: float,
        *,
        entry_filter: Callable[[dict[str, Any]], bool] | None = None,
        research_fixed_tp_r: float | None = None,
        research_be_mode: str | None = None,
    ) -> None:
        super().__init__(logger, data_source=data_source, stop_size=stop_size)
        self.timeout_sec = float(timeout_sec)
        self.entry_filter = entry_filter
        self.research_fixed_tp_r = research_fixed_tp_r
        self.research_be_mode = research_be_mode
        self._next_excursion_id = 1
        self._active_excursions: dict[int, ExcursionState] = {}
        self._closed_trade_by_excursion_id: dict[int, dict[str, Any]] = {}
        self._pending_entry_features: dict[str, Any] | None = None
        self._qualifying_signal_times: deque[float] = deque()
        self._bb_snapshots: deque[tuple[float, float]] = deque(maxlen=512)

    def _take_trade(self, top: TopState, regime: str, tp_r: float, be_trigger: float | None) -> None:
        super()._take_trade(top, regime, tp_r, be_trigger)
        if self.open_trade is None:
            return
        if self._pending_entry_features is not None:
            self.open_trade.update(self._pending_entry_features)
        trade_id = self._next_excursion_id
        self._next_excursion_id += 1
        self.open_trade["_excursion_id"] = trade_id
        self._active_excursions[trade_id] = ExcursionState(
            trade_id=trade_id,
            side=str(self.open_trade["side"]),
            entry_ts=float(self.open_trade["entry_ts_unix"]),
            entry_price=float(self.open_trade["entry_price"]),
            stop_distance=self.stop_size,
            timeout_sec=self.timeout_sec,
        )

    def _close_trade(self, exit_ts: str, exit_price: float, outcome: str, r_value: float) -> None:
        t = self.open_trade
        if t is None:
            return
        rec: dict[str, Any] = {
            "entry_ts": t["entry_ts"],
            "exit_ts": exit_ts,
            "side": t["side"],
            "entry_price": t["entry_price"],
            "exit_price": exit_price,
            "outcome": outcome,
            "r_value": r_value,
            "regime": t["regime"],
        }
        for field_name in ATTRIBUTION_FEATURE_FIELDS:
            if field_name in t:
                rec[field_name] = t[field_name]
        trade_id = t.get("_excursion_id")
        if isinstance(trade_id, int):
            state = self._active_excursions.get(trade_id)
            if state is not None:
                rec.update(state.to_output())
            self._closed_trade_by_excursion_id[trade_id] = rec
        self.closed_trades.append(rec)
        self.realized_pnl_r += r_value
        self.equity_curve.append(self.realized_pnl_r)
        if r_value < 0:
            self._curr_losing_streak += 1
            self.max_losing_streak = max(self.max_losing_streak, self._curr_losing_streak)
        else:
            self._curr_losing_streak = 0
        self.open_trade = None

    def _update_active_excursions(self, top: TopState) -> None:
        for trade_id, state in list(self._active_excursions.items()):
            state.update(top.ts, top.mid)
            rec = self._closed_trade_by_excursion_id.get(trade_id)
            if rec is not None:
                rec.update(state.to_output())
            if state.finalized and rec is not None:
                del self._active_excursions[trade_id]

    def finalize_excursions(self) -> None:
        for trade_id, state in list(self._active_excursions.items()):
            state.finalized = True
            rec = self._closed_trade_by_excursion_id.get(trade_id)
            if rec is not None:
                rec.update(state.to_output())
            del self._active_excursions[trade_id]

    def _prior_cluster_counts(self, ts: float) -> dict[str, int]:
        max_window = max(CLUSTER_WINDOWS_SEC)
        while self._qualifying_signal_times and self._qualifying_signal_times[0] < ts - max_window:
            self._qualifying_signal_times.popleft()
        return {
            f"prior_qualifying_anomalies_{int(window)}s": sum(
                1 for prior_ts in self._qualifying_signal_times if prior_ts >= ts - window
            )
            for window in CLUSTER_WINDOWS_SEC
        }

    def _entry_features(
        self,
        *,
        top: TopState,
        pressure: float,
        event_rate: float,
        trade_count: float,
        mid_vol: float,
        z_tob: float | None,
        z_micro: float | None,
        z_pressure: float | None,
        z_event: float | None,
        z_trade: float | None,
        z_spread: float | None,
        z_mid_vol: float | None,
        anomaly_score: float,
        anomaly_percentile: float,
        direction: str,
        regime: str,
        cluster_counts: dict[str, int],
    ) -> dict[str, Any]:
        mid_vals = [m for _, m in self.mid_window]
        size_denom = top.best_bid_sz + top.best_ask_sz
        size_ratio = top.best_bid_sz / top.best_ask_sz if top.best_ask_sz > 0 else None
        liquidity_imb = ((top.best_bid_sz - top.best_ask_sz) / size_denom) if size_denom > 0 else None
        out: dict[str, Any] = {
            "entry_anomaly_percentile": anomaly_percentile,
            "entry_anomaly_score": anomaly_score,
            "entry_direction": direction,
            "entry_regime": regime,
            "entry_z_tob": z_tob,
            "entry_z_micro": z_micro,
            "entry_z_pressure": z_pressure,
            "entry_z_event_rate": z_event,
            "entry_z_trade_count": z_trade,
            "entry_z_spread": z_spread,
            "entry_z_mid_vol": z_mid_vol,
            "entry_tob_imbalance": top.tob_imb,
            "entry_microprice_deviation": top.micro_dev,
            "entry_signed_book_pressure": pressure,
            "entry_event_rate": event_rate,
            "entry_trade_count": trade_count,
            "entry_spread": top.spread,
            "entry_mid_price_volatility": mid_vol,
            "entry_mid_price": top.mid,
            "entry_best_bid_size": top.best_bid_sz,
            "entry_best_ask_size": top.best_ask_sz,
            "entry_bid_ask_size_ratio": size_ratio,
            "entry_liquidity_imbalance": liquidity_imb,
            "entry_session": _entry_session_label(top.ts),
            "entry_signal_key": f"{top.ts_text}|{direction}|{top.mid:.8f}",
        }
        out.update(cluster_counts)
        for window in CLUSTER_WINDOWS_SEC:
            count = int(cluster_counts.get(f"prior_qualifying_anomalies_{int(window)}s", 0))
            out[f"is_repeated_signal_{int(window)}s"] = count > 0

        bid_peak = bb = float(top.best_bid_sz)
        if self._bb_snapshots:
            bid_peak = max(bb for _, bb in self._bb_snapshots)
        fa = compute_failed_absorption_entry_features(
            top=top,
            pressure=pressure,
            event_rate=event_rate,
            trade_count=trade_count,
            mid_vals=mid_vals,
            bid_peak_recent=bid_peak,
            liquidity_imbalance=liquidity_imb,
            z_pressure=z_pressure,
            z_tob=z_tob,
            z_event=z_event,
            z_trade=z_trade,
            is_repeated_signal_30s=bool(out["is_repeated_signal_30s"]),
            is_repeated_signal_60s=bool(out["is_repeated_signal_60s"]),
            is_repeated_signal_120s=bool(out["is_repeated_signal_120s"]),
        )
        out.update(fa)
        return out

    def process_object(self, obj: dict[str, Any]) -> None:
        self.raw_events_seen += 1
        self._process_trade_msg(obj)
        top, pressure = self._process_l2_msg(obj)
        if top is None:
            return

        self._expire_windows(top.ts)
        while self._bb_snapshots and self._bb_snapshots[0][0] < top.ts - LOOKBACK_SEC:
            self._bb_snapshots.popleft()
        self._bb_snapshots.append((top.ts, float(top.best_bid_sz)))
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
            cluster_counts = self._prior_cluster_counts(top.ts)
            self.signals_seen += 1
            can, reason = self._can_take_trade(top.ts, current_regime)
            if can and self.open_trade is None:
                feats = self._entry_features(
                    top=top,
                    pressure=pressure,
                    event_rate=event_rate,
                    trade_count=trade_count,
                    mid_vol=mid_vol,
                    z_tob=z_tob,
                    z_micro=z_micro,
                    z_pressure=z_pressure,
                    z_event=z_event,
                    z_trade=z_trade,
                    z_spread=z_spread,
                    z_mid_vol=z_mid_vol,
                    anomaly_score=anomaly_score,
                    anomaly_percentile=pct,
                    direction=direction,
                    regime=current_regime,
                    cluster_counts=cluster_counts,
                )
                if self.entry_filter is not None and not self.entry_filter(feats):
                    self.trades_blocked += 1
                    self.trades_blocked_by_reason["entry_filter"] = (
                        self.trades_blocked_by_reason.get("entry_filter", 0) + 1
                    )
                else:
                    tp_r, be_trigger = self._regime_params(current_regime)
                    if self.research_fixed_tp_r is not None:
                        tp_r = float(self.research_fixed_tp_r)
                    if self.research_be_mode is not None:
                        mode = str(self.research_be_mode).lower()
                        if mode == "none":
                            be_trigger = None
                        elif mode == "0.75":
                            be_trigger = 0.75
                        elif mode == "1.0" or mode == "1":
                            be_trigger = 1.0
                        else:
                            raise ValueError(f"unknown research_be_mode: {self.research_be_mode}")
                    self._pending_entry_features = feats
                    try:
                        self._take_trade(top, current_regime, tp_r, be_trigger)
                    finally:
                        self._pending_entry_features = None
                    self.last_signal_ts = top.ts
            else:
                self.trades_blocked += 1
                self.trades_blocked_by_reason[reason] = self.trades_blocked_by_reason.get(reason, 0) + 1
            self._qualifying_signal_times.append(top.ts)

    def _maybe_manage_open_trade(self, top: TopState) -> None:
        self._update_active_excursions(top)
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


class PartialExitEthGeometryEngine(EthGeometryEngine):
    """
    Research-only partial: scale `partial_first_scale` at `partial_first_r` (favorable R),
    remainder `runner_scale` seeks `runner_tp_r`. Before partial fills, behaves like full TP at research_fixed_tp_r.
    """

    def __init__(
        self,
        logger: logging.Logger,
        data_source: str,
        stop_size: float,
        timeout_sec: float,
        *,
        partial_first_r: float = 0.5,
        partial_first_scale: float = 0.7,
        runner_scale: float = 0.3,
        runner_tp_r: float = 1.5,
        entry_filter: Callable[[dict[str, Any]], bool] | None = None,
        research_fixed_tp_r: float | None = None,
        research_be_mode: str | None = None,
    ) -> None:
        super().__init__(
            logger,
            data_source,
            stop_size,
            timeout_sec,
            entry_filter=entry_filter,
            research_fixed_tp_r=research_fixed_tp_r,
            research_be_mode=research_be_mode,
        )
        self._partial_first_r = float(partial_first_r)
        self._partial_first_scale = float(partial_first_scale)
        self._runner_scale = float(runner_scale)
        self._runner_tp_r = float(runner_tp_r)

    def _take_trade(self, top: TopState, regime: str, tp_r: float, be_trigger: float | None) -> None:
        super()._take_trade(top, regime, tp_r, be_trigger)
        if self.open_trade is not None:
            self.open_trade["partial_done"] = False
            self.open_trade["acc_scaled_r_dollar"] = 0.0

    def _maybe_manage_open_trade(self, top: TopState) -> None:
        self._update_active_excursions(top)
        t = self.open_trade
        if t is None:
            return
        entry = float(t["entry_price"])
        stop_sz = float(self.stop_size)
        sl_price = float(t["sl_price"])
        be_trigger = t.get("be_trigger_r")
        if be_trigger is not None and top.mid <= entry - float(be_trigger):
            t["sl_price"] = min(t["sl_price"], entry)
            sl_price = float(t["sl_price"])

        partial_px = entry - self._partial_first_r * stop_sz
        runner_tp_px = entry - self._runner_tp_r * stop_sz
        tp_price = float(t["tp_price"])
        partial_done = bool(t.get("partial_done", False))

        def close_scaled(px: float, remainder_frac: float, outcome: str) -> None:
            acc_now = float(t.get("acc_scaled_r_dollar", 0.0))
            total_r = acc_now + remainder_frac * (entry - px)
            self._close_trade(top.ts_text, px, outcome, total_r)

        if top.mid >= sl_price:
            rem = self._runner_scale if partial_done else 1.0
            close_scaled(top.mid, rem, "sl_or_be")
            return

        if not partial_done:
            if top.mid <= partial_px:
                acc_prev = float(t.get("acc_scaled_r_dollar", 0.0))
                new_acc = acc_prev + self._partial_first_scale * self._partial_first_r * stop_sz
                t["acc_scaled_r_dollar"] = new_acc
                t["partial_done"] = True
                partial_done = True

        if partial_done:
            if top.mid <= runner_tp_px:
                close_scaled(top.mid, self._runner_scale, "tp")
                return
        else:
            if top.mid <= tp_price:
                self._close_trade(top.ts_text, top.mid, "tp", entry - tp_price)
                return

        if (top.ts - float(t["entry_ts_unix"])) >= self.timeout_sec:
            rem = self._runner_scale if partial_done else 1.0
            close_scaled(top.mid, rem, "timeout")


def _per_trade_net_r(entry_price: float, gross_r: float, fee_rate: float = ROUND_TRIP_TAKER_FEE) -> float:
    """Subtract round-trip taker fee expressed in price-per-unit terms."""
    return gross_r - float(entry_price) * fee_rate


def _entry_session_label(ts: float) -> str:
    h = datetime.fromtimestamp(float(ts), tz=timezone.utc).hour
    if 0 <= h < 6:
        return "asia"
    if 6 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "london_ny_overlap"
    if 16 <= h < 20:
        return "new_york"
    return "off_hours"


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
                eng.finalize_excursions()
                closed = [dict(t, stop_size=stop, timeout_sec=tout) for t in eng.closed_trades]
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
                        trades=[dict(t) for t in closed],
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


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if pct < 0 or pct > 100:
        raise ValueError("pct must be between 0 and 100")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (pct / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] + ((xs[hi] - xs[lo]) * frac)


def _average_present(values: list[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return mean(present) if present else None


def _excursion_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    mfe_vals = [float(t["mfe_r"]) for t in trades if t.get("mfe_r") is not None]
    mae_vals = [float(t["mae_r"]) for t in trades if t.get("mae_r") is not None]
    n = len(trades)
    reaching: dict[str, float] = {}
    for level, field_name in EXCURSION_TIME_FIELDS.items():
        reaching[f"percent_reaching_{field_name.removeprefix('time_to_first_')}"] = (
            sum(1 for t in trades if t.get(field_name) is not None) / n if n else 0.0
        )

    return {
        "total_trades": n,
        "mfe_percentiles": {
            "p50": _percentile(mfe_vals, 50),
            "p75": _percentile(mfe_vals, 75),
            "p90": _percentile(mfe_vals, 90),
            "p95": _percentile(mfe_vals, 95),
        },
        "mae_percentiles": {
            "p50": _percentile(mae_vals, 50),
            "p75": _percentile(mae_vals, 75),
            "p90": _percentile(mae_vals, 90),
            "p95": _percentile(mae_vals, 95),
        },
        "reachability": reaching,
        "average_time_to_0_25R": _average_present([t.get("time_to_first_0_25R") for t in trades]),
        "average_time_to_0_5R": _average_present([t.get("time_to_first_0_5R") for t in trades]),
        "average_time_to_1R": _average_present([t.get("time_to_first_1R") for t in trades]),
        "average_time_to_peak_mfe": _average_present([t.get("seconds_to_mfe") for t in trades]),
    }


def _build_excursion_analysis(results: list[EthGeometryResult]) -> dict[str, Any]:
    all_trades: list[dict[str, Any]] = []
    by_variant: list[dict[str, Any]] = []
    for r in results:
        trades = [dict(t) for t in r.trades]
        all_trades.extend(trades)
        by_variant.append(
            {
                "stop_size": r.stop_size,
                "timeout_sec": r.timeout_sec,
                **_excursion_metrics(trades),
            }
        )
    return {
        "overall": _excursion_metrics(all_trades),
        "by_variant": by_variant,
    }


def _cohort_for_trade(trade: dict[str, Any]) -> str:
    mfe = float(trade.get("mfe_r") or 0.0)
    mae_abs = abs(float(trade.get("mae_r") or 0.0))
    t05 = trade.get("time_to_first_0_5R")
    if mfe >= 0.75 and mae_abs <= 0.30 and t05 is not None and float(t05) <= 180.0:
        return "tier_a_fast_clean_winner"
    if mfe >= 1.0:
        return "tier_b_slow_noisy_winner"
    if mfe < 0.5:
        return "tier_c_dead_signal"
    return "uncategorized_middle"


def _anomaly_percentile_bucket(pct: Any) -> str:
    if pct is None:
        return "unknown"
    val = float(pct)
    if val >= 0.999:
        return "99.9+"
    if val >= 0.995:
        return "99.5-99.9"
    if val >= 0.99:
        return "99.0-99.5"
    return "below_99"


def _median(values: list[float]) -> float | None:
    return _percentile(values, 50)


def _median_field(trades: list[dict[str, Any]], field_name: str) -> float | None:
    vals = [float(t[field_name]) for t in trades if t.get(field_name) is not None]
    return _median(vals)


def _summarize_attribution_group(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    tiers = [_cohort_for_trade(t) for t in trades]
    mfe_vals = [float(t["mfe_r"]) for t in trades if t.get("mfe_r") is not None]
    tier_a = sum(1 for x in tiers if x == "tier_a_fast_clean_winner")
    tier_b = sum(1 for x in tiers if x == "tier_b_slow_noisy_winner")
    tier_c = sum(1 for x in tiers if x == "tier_c_dead_signal")
    return {
        "trade_count": n,
        "tier_a_count": tier_a,
        "tier_a_rate": tier_a / n if n else 0.0,
        "tier_b_count": tier_b,
        "tier_b_rate": tier_b / n if n else 0.0,
        "tier_c_count": tier_c,
        "tier_c_rate": tier_c / n if n else 0.0,
        "median_mfe": _percentile(mfe_vals, 50),
        "p75_mfe": _percentile(mfe_vals, 75),
    }


def _group_by(trades: list[dict[str, Any]], field_name: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        key = str(trade.get(field_name) or "unknown")
        groups.setdefault(key, []).append(trade)
    return groups


def _feature_medians(trades: list[dict[str, Any]], features: list[str]) -> dict[str, float | None]:
    return {feature: _median_field(trades, feature) for feature in features}


def _feature_comparison(trades: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    tier_a = [t for t in trades if _cohort_for_trade(t) == "tier_a_fast_clean_winner"]
    tier_c = [t for t in trades if _cohort_for_trade(t) == "tier_c_dead_signal"]
    return {
        "tier_a_trade_count": len(tier_a),
        "tier_c_trade_count": len(tier_c),
        "tier_a_medians": _feature_medians(tier_a, features),
        "tier_c_medians": _feature_medians(tier_c, features),
        "median_differences_tier_a_minus_tier_c": {
            feature: (
                (_median_field(tier_a, feature) - _median_field(tier_c, feature))
                if _median_field(tier_a, feature) is not None and _median_field(tier_c, feature) is not None
                else None
            )
            for feature in features
        },
    }


def _rows_from_groups(groups: dict[str, list[dict[str, Any]]], label: str) -> list[dict[str, Any]]:
    return [
        {label: key, **_summarize_attribution_group(group)}
        for key, group in sorted(groups.items(), key=lambda kv: kv[0])
    ]


def _dedupe_signal_level(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    variants: dict[str, dict[str, set[float]]] = {}
    for trade in trades:
        key = str(trade.get("entry_signal_key") or f"{trade.get('entry_ts')}|{trade.get('side')}|{trade.get('entry_price')}")
        state = variants.setdefault(key, {"stops": set(), "timeouts": set()})
        if trade.get("stop_size") is not None:
            state["stops"].add(float(trade["stop_size"]))
        if trade.get("timeout_sec") is not None:
            state["timeouts"].add(float(trade["timeout_sec"]))
        current = by_key.get(key)
        if current is None:
            by_key[key] = dict(trade)
            continue
        current_timeout = float(current.get("timeout_sec") or 0.0)
        trade_timeout = float(trade.get("timeout_sec") or 0.0)
        current_mfe_move = float(current.get("mfe_price_move") or 0.0)
        trade_mfe_move = float(trade.get("mfe_price_move") or 0.0)
        if (trade_timeout, trade_mfe_move) > (current_timeout, current_mfe_move):
            by_key[key] = dict(trade)
    out = []
    for key, trade in by_key.items():
        state = variants.get(key, {"stops": set(), "timeouts": set()})
        row = dict(trade)
        row["variants_seen"] = len(state["stops"]) * len(state["timeouts"])
        row["stops_seen"] = sorted(state["stops"])
        row["timeouts_seen"] = sorted(state["timeouts"])
        row["signal_level_cohort"] = _cohort_for_trade(row)
        out.append(row)
    return out


def _build_signal_clustering(trades: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "isolated_120s": [t for t in trades if int(t.get("prior_qualifying_anomalies_120s") or 0) == 0],
        "repeated_120s": [t for t in trades if int(t.get("prior_qualifying_anomalies_120s") or 0) > 0],
        "repeated_60s": [t for t in trades if int(t.get("prior_qualifying_anomalies_60s") or 0) > 0],
        "repeated_30s": [t for t in trades if int(t.get("prior_qualifying_anomalies_30s") or 0) > 0],
    }
    return {
        "definition": "prior qualifying bearish anomaly means direction == bearish and anomaly_percentile >= 0.99 before this entry",
        "groups": _rows_from_groups(groups, "cluster_type"),
    }


def _build_regime_attribution(results: list[EthGeometryResult]) -> dict[str, Any]:
    all_trades = [dict(t) for result in results for t in result.trades]
    micro_features = [
        "entry_tob_imbalance",
        "entry_microprice_deviation",
        "entry_signed_book_pressure",
        "entry_event_rate",
        "entry_trade_count",
        "entry_spread",
        "entry_mid_price_volatility",
        "entry_z_tob",
        "entry_z_micro",
        "entry_z_pressure",
        "entry_z_event_rate",
        "entry_z_trade_count",
        "entry_z_spread",
        "entry_z_mid_vol",
    ]
    liquidity_features = [
        "entry_best_bid_size",
        "entry_best_ask_size",
        "entry_bid_ask_size_ratio",
        "entry_liquidity_imbalance",
        "entry_signed_book_pressure",
    ]
    by_variant = [
        {
            "stop_size": result.stop_size,
            "timeout_sec": result.timeout_sec,
            **_summarize_attribution_group([dict(t) for t in result.trades]),
        }
        for result in results
    ]
    signal_level = _dedupe_signal_level(all_trades)
    anomaly_groups: dict[str, list[dict[str, Any]]] = {}
    for trade in all_trades:
        anomaly_groups.setdefault(_anomaly_percentile_bucket(trade.get("entry_anomaly_percentile")), []).append(trade)

    return {
        "cohort_definition": {
            "tier_a_fast_clean_winner": "mfe_r >= 0.75 and abs(mae_r) <= 0.30 and time_to_first_0_5R <= 180s",
            "tier_b_slow_noisy_winner": "mfe_r >= 1.0 and not Tier A",
            "tier_c_dead_signal": "mfe_r < 0.5",
            "uncategorized_middle": "all other trades",
        },
        "by_variant": by_variant,
        "signal_level": {
            "deduplication_key": "entry_signal_key; one row per entry signal, selecting the longest timeout observation and strongest absolute MFE tie-break",
            "signal_count": len(signal_level),
            "summary": _summarize_attribution_group(signal_level),
            "by_regime": _rows_from_groups(_group_by(signal_level, "regime"), "regime"),
            "by_session": _rows_from_groups(_group_by(signal_level, "entry_session"), "session"),
            "signal_clustering": _build_signal_clustering(signal_level),
        },
        "by_regime": _rows_from_groups(_group_by(all_trades, "regime"), "regime"),
        "by_anomaly_percentile_bucket": _rows_from_groups(anomaly_groups, "anomaly_percentile_bucket"),
        "feature_comparison": _feature_comparison(all_trades, micro_features),
        "session_attribution": _rows_from_groups(_group_by(all_trades, "entry_session"), "session"),
        "signal_clustering": _build_signal_clustering(all_trades),
        "liquidity_wall_attribution": {
            "available_fields": liquidity_features,
            "unavailable_fields": [
                "bid_depletion",
                "ask_depletion",
                "passive_absorption_behavior",
                "liquidity_disappears_vs_absorbs",
            ],
            "comparison": _feature_comparison(all_trades, liquidity_features),
        },
    }


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
        "excursion_analysis": _build_excursion_analysis(results),
        "regime_attribution": _build_regime_attribution(results),
        "results": [_sanitize_json_row(asdict(r)) for r in results],
    }

    from tjtb.research.maker_fill_quality_study import build_maker_fill_quality_study

    summary.update(build_maker_fill_quality_study(results))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETH geometry grid: stops × timeouts, fees, leverage")
    parser.add_argument(
        "--study",
        choices=("geometry", "ultimate", "pr-final"),
        default="geometry",
        help="geometry=classic grid; ultimate / pr-final = research JSON studies",
    )
    parser.add_argument("--data-source", choices=("coinbase", "bybit"), default="bybit")
    parser.add_argument("--stops", default="1,2,3,5,8,10")
    parser.add_argument("--timeouts", default="120,180,300,480,600,900", help="Comma-separated timeout seconds")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--fee-rate", type=float, default=ROUND_TRIP_TAKER_FEE, help="Round-trip fee as decimal")
    parser.add_argument("--csv-output", type=Path, default=REPORTS_DIR / "eth_geometry_results.csv")
    parser.add_argument("--json-output", type=Path, default=REPORTS_DIR / "eth_geometry_summary.json")
    parser.add_argument(
        "--ultimate-json-output",
        type=Path,
        default=REPORTS_DIR / "ultimate_edge_study.json",
        help="Output path when --study ultimate",
    )
    parser.add_argument("--label-stop", type=float, default=5.0, help="Reference stop for Phase 1 tier labeling (ultimate)")
    parser.add_argument("--label-timeout", type=float, default=900.0, help="Reference timeout for Phase 1 (ultimate)")
    parser.add_argument("--skip-phase2", action="store_true", help="Ultimate study: Phase 1 (+3) only")
    parser.add_argument("--skip-phase3", action="store_true", help="Ultimate study: skip execution stress stub")
    parser.add_argument(
        "--pr-final-json-output",
        type=Path,
        default=REPORTS_DIR / "pr_final_entry_study.json",
        help="Output path when --study pr-final",
    )
    parser.add_argument("--skip-part-e", action="store_true", help="PR-FINAL: Part D only (no executable geometry grid)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.study == "pr-final":
        from tjtb.research.pr_final_entry_study import run_pr_final_study

        run_pr_final_study(
            data_source=args.data_source,
            raw_dir=args.raw_dir,
            symbol=str(args.symbol).strip().upper(),
            fee_rate=float(args.fee_rate),
            json_output=args.pr_final_json_output,
            run_part_e=not args.skip_part_e,
        )
        LOGGER.info("PR-FINAL study written to %s", args.pr_final_json_output)
        return 0

    if args.study == "ultimate":
        from tjtb.research.ultimate_edge_study import run_ultimate_study

        run_ultimate_study(
            data_source=args.data_source,
            raw_dir=args.raw_dir,
            symbol=str(args.symbol).strip().upper(),
            fee_rate=float(args.fee_rate),
            json_output=args.ultimate_json_output,
            label_stop=float(args.label_stop),
            label_timeout=float(args.label_timeout),
            run_phase2=not args.skip_phase2,
            run_phase3=not args.skip_phase3,
        )
        LOGGER.info("ultimate edge study written to %s", args.ultimate_json_output)
        return 0

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
