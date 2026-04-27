from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

TRADES_PATH = Path("data/live/paper_trades.csv")
REFRESH_MS = 5000

EXPECTED_COLS = [
    "entry_ts",
    "exit_ts",
    "side",
    "entry_price",
    "exit_price",
    "outcome",
    "r_value",
    "regime",
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


def load_paper_trades(path: Path) -> pd.DataFrame:
    """Read-only load of paper trades CSV; returns empty DataFrame if missing/unreadable."""
    if not path.is_file():
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=EXPECTED_COLS)
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[EXPECTED_COLS].copy()
    for c in ("entry_price", "exit_price", "r_value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["r_value"])
    df["_exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    return df


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
    st.caption("Data source: `data/live/paper_trades.csv`. No orders or API calls.")

    df = load_paper_trades(TRADES_PATH)

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
