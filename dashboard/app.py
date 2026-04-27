from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# dashboard/ -> project root -> src on path (works without PYTHONPATH when layout is standard)
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tjtb.runtime_paths import (  # noqa: E402
    OPPORTUNITIES_PATH,
    PAPER_TRADES_PATH,
    PROJECT_ROOT,
)

REFRESH_MS = 5000

EXPECTED_TRADE_COLS = [
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "outcome",
    "r_value",
    "regime",
]

EXPECTED_OPP_COLS = [
    "ts",
    "anomaly_percentile",
    "anomaly_score",
    "direction",
    "regime",
    "action",
    "reason",
]


def _enable_autorefresh() -> None:
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=REFRESH_MS, key="tjtb-paper-refresh")
        return
    st.components.v1.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {REFRESH_MS});
        </script>
        """,
        height=0,
    )


def _file_mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except OSError:
        return None


def _file_size_bytes(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _csv_data_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
    except OSError:
        return None
    if n <= 0:
        return 0
    return max(0, n - 1)


def _render_path_card(label: str, path: Path) -> None:
    st.markdown(f"**{label}**")
    st.code(str(path.resolve()), language="text")
    exists = path.is_file()
    if not exists:
        st.warning(f"Missing: `{path.name}`")
        return
    rows = _csv_data_row_count(path)
    mtime = _file_mtime_iso(path)
    size_b = _file_size_bytes(path)
    st.caption(
        f"Rows (excl. header): **{rows}** · Last modified: **{mtime or '—'}** · Size: **{size_b}** bytes"
    )


def load_paper_trades(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=EXPECTED_TRADE_COLS)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=EXPECTED_TRADE_COLS)
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_TRADE_COLS)
    for c in EXPECTED_TRADE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[EXPECTED_TRADE_COLS].copy()
    for c in ("entry_price", "exit_price", "r_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["r_value"])
    df["_exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    return df


def load_opportunities(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=EXPECTED_OPP_COLS)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=EXPECTED_OPP_COLS)
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_OPP_COLS)
    for c in EXPECTED_OPP_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[EXPECTED_OPP_COLS].copy()


def max_losing_streak_r(r_series: pd.Series) -> int:
    streak = max_streak = 0
    for r in r_series:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return int(max_streak)


def max_drawdown_r(cumulative_r: pd.Series) -> float:
    if cumulative_r.empty:
        return 0.0
    peak = cumulative_r.cummax()
    dd = peak - cumulative_r
    return float(dd.max())


def main() -> None:
    st.set_page_config(page_title="Paper Trades (Live)", layout="wide")
    _enable_autorefresh()

    st.title("Live bot — paper trades (read-only)")
    st.caption(
        f"PROJECT_ROOT: `{PROJECT_ROOT}` — no orders or broker calls; CSV read only."
    )

    st.subheader("Canonical paths")
    c1, c2 = st.columns(2)
    with c1:
        _render_path_card("Paper trades", PAPER_TRADES_PATH)
    with c2:
        _render_path_card("Opportunities", OPPORTUNITIES_PATH)

    df = load_paper_trades(PAPER_TRADES_PATH)
    opps = load_opportunities(OPPORTUNITIES_PATH)

    st.subheader("Opportunities (read-only preview)")
    if opps.empty:
        st.info("No opportunity rows loaded (missing file, empty, or parse error).")
    else:
        st.dataframe(opps.tail(50), use_container_width=True, hide_index=True)

    if df.empty:
        st.info("No trades yet")
        return

    chron = df.sort_values("_exit_ts", na_position="last", ascending=True)
    r = chron["r_value"]
    cum_r = r.cumsum()
    wins = (r > 0).sum()
    n = int(len(df))
    total_r = float(r.sum())
    win_rate = float(wins / n) if n else 0.0
    avg_r = float(r.mean()) if n else 0.0
    best_r = float(r.max()) if n else 0.0
    worst_r = float(r.min()) if n else 0.0
    mdd_r = max_drawdown_r(cum_r.reset_index(drop=True))
    lose_streak = max_losing_streak_r(chron["r_value"].reset_index(drop=True))

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)
    c1.metric("Total trades", f"{n:,}")
    c2.metric("Realized PnL (R)", f"{total_r:,.3f}")
    c3.metric("Win rate", f"{win_rate:.1%}")
    c4.metric("Average R", f"{avg_r:,.3f}")
    c5.metric("Best trade (R)", f"{best_r:,.3f}")
    c6.metric("Worst trade (R)", f"{worst_r:,.3f}")
    c7.metric("Max drawdown (R)", f"{mdd_r:,.3f}")
    c8.metric("Max losing streak", f"{lose_streak:,}")

    st.subheader("Cumulative equity (R)")
    equity_df = pd.DataFrame({"trade #": range(1, len(chron) + 1), "equity_R": cum_r.values})
    st.line_chart(equity_df, x="trade #", y="equity_R")

    st.subheader("Trades (newest first)")
    display_cols = [
        "exit_ts",
        "entry_ts",
        "side",
        "entry_price",
        "exit_price",
        "outcome",
        "r_value",
        "regime",
    ]
    table = df.sort_values("_exit_ts", ascending=False, na_position="last")[display_cols].copy()
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Breakdown")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**By outcome**")
        oc = (
            df.groupby("outcome", dropna=False)["r_value"]
            .agg(trades="count", total_R="sum", avg_R="mean")
            .sort_values("trades", ascending=False)
        )
        st.dataframe(oc, use_container_width=True)
    with b2:
        st.markdown("**By regime**")
        rg = (
            df.groupby("regime", dropna=False)["r_value"]
            .agg(trades="count", total_R="sum", avg_R="mean")
            .sort_values("trades", ascending=False)
        )
        st.dataframe(rg, use_container_width=True)
    with b3:
        st.markdown("**By side**")
        sd = (
            df.groupby("side", dropna=False)["r_value"]
            .agg(trades="count", total_R="sum", avg_R="mean")
            .sort_values("trades", ascending=False)
        )
        st.dataframe(sd, use_container_width=True)


if __name__ == "__main__":
    main()
