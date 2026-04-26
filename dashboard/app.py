from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

STATUS_PATH = Path("reports/live_status.json")
TRADES_PATH = Path("data/live/paper_trades.csv")
OPPS_PATH = Path("data/live/opportunities.csv")
LOG_PATH = Path("logs/live_paper.log")
REFRESH_MS = 5000


def _enable_autorefresh() -> None:
    # Use built-in API when available; fallback to JS timer.
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=REFRESH_MS, key="live-refresh")
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


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _safe_tail(path: Path, n: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [ln.rstrip("\n") for ln in lines[-n:]]


def _to_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    txt = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _open_trade_panel(open_trade: dict[str, Any], current_mid: float | None) -> None:
    if not open_trade:
        st.info("No open trade")
        return
    side = str(open_trade.get("side", "short"))
    entry = _to_float(open_trade.get("entry_price"))
    regime = open_trade.get("regime", "-")
    entry_ts = _parse_ts(open_trade.get("entry_ts"))
    now = datetime.now(timezone.utc)

    current_r: float | None = None
    if side == "short" and entry is not None and current_mid is not None:
        current_r = entry - current_mid
    if side == "long" and entry is not None and current_mid is not None:
        current_r = current_mid - entry

    time_open = "-"
    if entry_ts is not None:
        sec = max(0, int((now - entry_ts).total_seconds()))
        time_open = f"{sec}s"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Direction", side)
    c2.metric("Entry Price", f"{entry:.2f}" if entry is not None else "-")
    c3.metric("Current R", f"{current_r:.3f}" if current_r is not None else "-")
    c4.metric("Regime", str(regime))
    c5.metric("Time Open", time_open)


def _equity_curve(trades: list[dict[str, str]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    eq = 0.0
    for row in trades:
        r = _to_float(row.get("r_value"))
        if r is None:
            continue
        eq += r
        out.append({"equity_r": eq})
    return out


def _opportunity_stats(opps: list[dict[str, str]]) -> dict[str, Any]:
    blocked = 0
    entered = 0
    reasons: dict[str, int] = {}
    for row in opps:
        action = (row.get("action") or "").strip().lower()
        reason = (row.get("reason") or "").strip()
        if action == "entered_short":
            entered += 1
        if action == "blocked":
            blocked += 1
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    return {"entered": entered, "blocked": blocked, "reasons": reasons}


def main() -> None:
    st.set_page_config(page_title="Live Paper Dashboard", layout="wide")
    _enable_autorefresh()
    st.title("Trading Bot - Live Paper Dashboard")

    status = _safe_json(STATUS_PATH)
    trades = _safe_csv(TRADES_PATH)
    opps = _safe_csv(OPPS_PATH)
    logs = _safe_tail(LOG_PATH, 50)

    current_mid = _to_float(status.get("current_mid_price"))

    st.subheader("Header")
    h1, h2, h3, h4, h5 = st.columns(5)
    bot_status = "RUNNING" if status else "WAITING_FOR_STATUS"
    h1.metric("Bot Status", bot_status)
    h2.metric("Last Update", str(status.get("last_update", "-")))
    h3.metric("Symbol", str(status.get("symbol", "BTC-USD")))
    h4.metric("Current Mid Price", f"{current_mid:.2f}" if current_mid is not None else "-")
    h5.metric("Current Regime", str(status.get("current_regime", "-")))

    st.subheader("Performance")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("realized_pnl_r", str(status.get("realized_pnl_r", "-")))
    p2.metric("trades_taken", str(status.get("trades_taken", "-")))
    p3.metric("win_rate", str(status.get("win_rate", "-")))
    p4.metric("avg_r", str(status.get("avg_r", "-")))
    p5.metric("max_drawdown_r", str(status.get("max_drawdown_r", "-")))
    p6.metric("max_losing_streak", str(status.get("max_losing_streak", "-")))

    st.subheader("Open Trade")
    open_trade = status.get("open_trade")
    _open_trade_panel(open_trade if isinstance(open_trade, dict) else {}, current_mid)

    st.subheader("Last 10 Trades")
    last10 = status.get("last_10_trades")
    if isinstance(last10, list) and last10:
        table_rows = []
        for t in last10:
            if not isinstance(t, dict):
                continue
            table_rows.append(
                {
                    "entry_ts": t.get("entry_ts", ""),
                    "exit_ts": t.get("exit_ts", ""),
                    "direction": t.get("side", ""),
                    "entry_price": t.get("entry_price", ""),
                    "exit_price": t.get("exit_price", ""),
                    "reason": t.get("outcome", ""),
                    "r": t.get("r_value", ""),
                    "regime": t.get("regime", ""),
                }
            )
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No trade history yet")

    st.subheader("Equity Curve")
    eq = _equity_curve(trades)
    if eq:
        st.line_chart(eq, y="equity_r")
    else:
        st.info("No equity data yet")

    st.subheader("Opportunity Stats")
    s1, s2, s3 = st.columns(3)
    s1.metric("signals_seen", str(status.get("signals_seen", len(opps))))
    s2.metric("trades_blocked", str(status.get("trades_blocked", "-")))
    stats = _opportunity_stats(opps)
    s3.metric("entered_short_signals", str(stats["entered"]))

    st.markdown("**Block reasons**")
    blocked_by_status = status.get("trades_blocked")
    if isinstance(blocked_by_status, dict) and blocked_by_status:
        st.json(blocked_by_status)
    elif stats["reasons"]:
        st.json(stats["reasons"])
    else:
        st.write("No block reasons available yet")

    st.subheader("Logs (last 50 lines)")
    if logs:
        st.code("\n".join(logs))
    else:
        st.info("No logs yet")


if __name__ == "__main__":
    main()
