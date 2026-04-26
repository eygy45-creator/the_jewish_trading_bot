"""
Session / hour research scaffolding.

Produces candidate tradable window recommendations based on OOS stability heuristics.
Final permission logic must consume persisted research artifacts, not hard-coded opinions.

TODO: integrate walk-forward PnL paths, drawdown constraints, regime conditioning, news proximity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tjtb.config.session_research_settings import SessionResearchSettings


@dataclass
class SessionResearchReport:
    hourly_table: pd.DataFrame
    session_table: pd.DataFrame
    stability: dict
    recommended_windows: list[str]


def hourly_bucket_table(trades: pd.DataFrame, pnl_col: str, ts_col: str, freq: str) -> pd.DataFrame:
    """Aggregate net PnL and trade counts by time bucket."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t[ts_col] = pd.to_datetime(t[ts_col])
    t = t.set_index(ts_col).sort_index()
    t.index.name = ts_col
    g = t[pnl_col].resample(freq)
    out = pd.DataFrame({"net_pnl": g.sum(), "n_trades": g.count()})
    out["expectancy"] = out["net_pnl"] / out["n_trades"].replace(0, pd.NA)
    return out.reset_index()


def _sign_consistency(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    pos = (s > 0).mean()
    neg = (s < 0).mean()
    return float(max(pos, neg))


def build_session_research_report(
    trades: pd.DataFrame,
    settings: SessionResearchSettings,
    *,
    pnl_col: str = "net_pnl",
    ts_col: str = "ts",
    session_col: str = "session_bucket",
    walk_forward_fold_stats: list[dict] | None = None,
) -> SessionResearchReport:
    hourly = hourly_bucket_table(trades, pnl_col, ts_col, settings.hour_bucket)
    if trades.empty or session_col not in trades.columns:
        session = pd.DataFrame()
    else:
        session = trades.groupby(session_col)[pnl_col].agg(["sum", "count", "mean"]).reset_index()
        session.columns = ["session_bucket", "net_pnl", "n_trades", "expectancy"]

    stability: dict = {
        "pnl_sign_consistency_hourly": _sign_consistency(hourly["net_pnl"])
        if not hourly.empty and "net_pnl" in hourly.columns
        else 0.0,
        "walk_forward_fold_stats": walk_forward_fold_stats or [],
    }

    recommended: list[str] = []
    if not session.empty:
        for _, row in session.iterrows():
            if row["n_trades"] < settings.min_trades_per_bucket:
                continue
            if row["expectancy"] > 0 and row["n_trades"] >= settings.min_positive_expectancy_samples:
                recommended.append(str(row["session_bucket"]))

    return SessionResearchReport(
        hourly_table=hourly,
        session_table=session,
        stability=stability,
        recommended_windows=recommended,
    )


def write_session_research_json(report: SessionResearchReport, path: str | Path) -> None:
    payload = {
        "recommended_windows": report.recommended_windows,
        "stability": report.stability,
        "hourly": report.hourly_table.to_dict(orient="records"),
        "session": report.session_table.to_dict(orient="records"),
    }
    Path(path).write_text(json.dumps(payload, default=str, indent=2))
