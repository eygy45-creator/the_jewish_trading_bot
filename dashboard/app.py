from __future__ import annotations

import os
import csv
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tjtb.runtime_paths import (  # noqa: E402
    DASHBOARD_LOG_PATH,
    HEARTBEAT_PATH,
    LIVE_BOT_LOG_PATH,
    OPPORTUNITIES_PATH,
    PAPER_TRADES_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
def _load_trade_context_exporter() -> tuple[object | None, dict[str, str | bool | None]]:
    diag: dict[str, str | bool | None] = {
        "ok": False,
        "module": "tjtb.reports.export_trade_context",
        "module_file": None,
        "exception": None,
        "py_path": os.environ.get("PYTHONPATH"),
    }
    try:
        mod = import_module("tjtb.reports.export_trade_context")
        fn = getattr(mod, "export_trade_context", None)
        sample_keys_fn = getattr(mod, "sample_raw_event_keys", None)
        if fn is None:
            diag["exception"] = "export_trade_context symbol not found in module"
            return None, diag
        diag["ok"] = True
        diag["module_file"] = str(getattr(mod, "__file__", "") or "")
        diag["sample_keys_fn"] = sample_keys_fn
        return fn, diag
    except Exception as exc:
        diag["exception"] = f"{exc.__class__.__name__}: {exc}"
        diag["traceback"] = traceback.format_exc(limit=3)
        return None, diag


export_trade_context, _TRADE_CONTEXT_IMPORT_DIAG = _load_trade_context_exporter()
_HAS_TRADE_CONTEXT_EXPORTER = bool(_TRADE_CONTEXT_IMPORT_DIAG.get("ok"))

RAW_NDJSON_GLOB = "coinbase_*.ndjson"
HEARTBEAT_STALE_SEC = float(os.environ.get("TJTB_HEARTBEAT_STALE_SEC", "90"))

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


def _quarantine_invalid_opportunities_csv(path: Path) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, None
    malformed = False
    reason = None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if header != EXPECTED_OPP_COLS:
                # Fallback case: file was written without header and first data row became "header".
                # Keep file as-is so loader can read it with explicit names.
                if len(header) == len(EXPECTED_OPP_COLS):
                    return False, "headerless_7col_fallback"
                malformed = True
                reason = f"header_mismatch expected={EXPECTED_OPP_COLS} got={header}"
            else:
                for row in reader:
                    if not row:
                        continue
                    if len(row) != len(EXPECTED_OPP_COLS):
                        malformed = True
                        reason = f"row_length_mismatch expected={len(EXPECTED_OPP_COLS)} got={len(row)}"
                        break
    except OSError as exc:
        malformed = True
        reason = f"read_error: {exc}"

    if not malformed:
        return False, None

    bad_path = path.parent / "opportunities.bad.csv"
    if bad_path.is_file():
        bad_path.unlink()
    shutil.move(str(path), str(bad_path))
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(EXPECTED_OPP_COLS)
    return True, reason


def _read_csv_safe(path: Path) -> tuple[pd.DataFrame, str | None]:
    if not path.is_file():
        return pd.DataFrame(), "missing_file"
    try:
        return pd.read_csv(path, encoding="utf-8"), None
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty_file"
    except (OSError, pd.errors.ParserError):
        return pd.DataFrame(), "parse_error"


def _missing_required_columns(df: pd.DataFrame, expected_cols: list[str]) -> list[str]:
    return [c for c in expected_cols if c not in df.columns]


def _opportunities_invalid(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    ts = pd.to_datetime(df.get("ts"), utc=True, errors="coerce")
    score = pd.to_numeric(df.get("anomaly_score"), errors="coerce")
    pct = pd.to_numeric(df.get("anomaly_percentile"), errors="coerce")
    signal_like = ts.notna() | score.notna() | pct.notna()
    return not bool(signal_like.any())


def _pgrep_f(pat: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return r.returncode == 0 and bool((r.stdout or "").strip())


def _port_8501_open() -> bool:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 8501))
        return True
    except OSError:
        return False
    finally:
        s.close()


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


def _heartbeat_age_sec() -> float | None:
    if not HEARTBEAT_PATH.is_file():
        return None
    try:
        txt = HEARTBEAT_PATH.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not txt:
            return None
        ts = datetime.fromisoformat(txt[0].strip().replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(tz=timezone.utc) - ts).total_seconds())
    except (ValueError, OSError):
        return None


def _raw_feed_status() -> tuple[str, int, float | None, str | None]:
    files = [p for p in RAW_DATA_DIR.glob(RAW_NDJSON_GLOB) if p.is_file()]
    if not files:
        return "NO_FEED", 0, None, None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    age = time.time() - latest.stat().st_mtime
    state = "STALE" if age > 120.0 else "OK"
    return state, len(files), age, latest.name


def _render_path_card(label: str, path: Path) -> tuple[int | None, bool]:
    """Returns (row_count_or_none, exists)."""
    st.markdown(f"**{label}**")
    st.code(str(path.resolve()), language="text")
    exists = path.is_file()
    if not exists:
        st.warning(f"Missing file: `{path.name}`")
        return None, False
    rows = _csv_data_row_count(path)
    mtime = _file_mtime_iso(path)
    size_b = _file_size_bytes(path)
    st.caption(
        f"Rows (excl. header): **{rows}** · Last modified: **{mtime or '—'}** · Size: **{size_b}** bytes"
    )
    return rows, True


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
    df["_exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    df["_entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["_valid_trade"] = df["r_value"].notna()
    return df


def load_opportunities(path: Path) -> tuple[pd.DataFrame, bool]:
    if not path.is_file():
        return pd.DataFrame(columns=EXPECTED_OPP_COLS), False
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=EXPECTED_OPP_COLS), False
    if df.empty:
        return pd.DataFrame(columns=EXPECTED_OPP_COLS), False
    fallback_applied = False
    if any(c not in df.columns for c in EXPECTED_OPP_COLS):
        # Header fallback: if file has 7 columns but no valid header,
        # re-read treating all rows as data.
        try:
            df_fallback = pd.read_csv(
                path,
                encoding="utf-8",
                header=None,
                names=EXPECTED_OPP_COLS,
            )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame(columns=EXPECTED_OPP_COLS), False
        if df_fallback.empty:
            return pd.DataFrame(columns=EXPECTED_OPP_COLS), False
        df = df_fallback.copy()
        fallback_applied = True
    normalized = df[EXPECTED_OPP_COLS].copy()
    return normalized, fallback_applied


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
    st.set_page_config(page_title="TJTB Live Paper Stack", layout="wide")

    with st.sidebar:
        st.header("Refresh")
        st.checkbox("Auto refresh", value=False, key="tjtb_auto_refresh")
        st.number_input(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=1,
            key="tjtb_refresh_interval_sec",
            help="When auto refresh is on, the app reruns on this interval (Streamlit autorefresh → full rerun).",
        )
        if st.button("Refresh now", key="tjtb_manual_refresh"):
            st.rerun()

    auto_on = bool(st.session_state.get("tjtb_auto_refresh", False))
    interval_sec = int(st.session_state.get("tjtb_refresh_interval_sec", 30))
    interval_sec = max(5, min(300, interval_sec))

    if auto_on and hasattr(st, "autorefresh"):
        st.autorefresh(interval=interval_sec * 1000, key=f"tjtb_autorefresh_{interval_sec}")

    st.title("Live paper stack (read-only)")
    st.caption(
        f"PROJECT_ROOT = `{PROJECT_ROOT}` — paper CSVs + status only; **no orders**, no broker API."
    )

    live_on = _pgrep_f("tjtb.live.live_paper_crypto")
    dash_on = _pgrep_f("streamlit run dashboard/app.py")
    port_on = _port_8501_open()
    hb_age = _heartbeat_age_sec()

    st.subheader("Process & port status")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Live bot process", "RUNNING" if live_on else "STOPPED")
    s2.metric("Dashboard process", "RUNNING" if dash_on else "STOPPED")
    s3.metric("Port 8501 (localhost)", "LISTENING" if port_on else "closed")
    s4.metric("Heartbeat age (sec)", f"{hb_age:.0f}" if hb_age is not None else "n/a")

    if hb_age is None:
        st.warning("Heartbeat missing or unreadable — live bot may not be running or logs dir wrong.")
    elif hb_age > HEARTBEAT_STALE_SEC:
        st.warning(
            f"Heartbeat is stale (~{hb_age:.0f}s > {HEARTBEAT_STALE_SEC:.0f}s) — bot hung, wrong path, or heartbeat thread stopped."
        )

    feed_state, n_ndjson, feed_age, latest_name = _raw_feed_status()
    st.subheader("Recorder / feed (NDJSON tail)")
    st.code(str(RAW_DATA_DIR.resolve()), language="text")
    st.write(
        f"Pattern `{RAW_NDJSON_GLOB}` · files={n_ndjson} · state=**{feed_state}** · latest_file={latest_name or '—'} · latest_age_sec={feed_age if feed_age is not None else '—'}"
    )
    if feed_state == "NO_FEED":
        st.error("NO LIVE DATA FEED — no NDJSON files under data/raw. Start the recorder that writes `coinbase_*.ndjson`.")
    elif feed_state == "STALE":
        st.warning("NDJSON file(s) exist but look stale — recorder may have stopped.")

    st.subheader("Canonical file paths")
    c1, c2, c3 = st.columns(3)
    with c1:
        trade_rows, _ = _render_path_card("Paper trades CSV", PAPER_TRADES_PATH)
    with c2:
        opp_rows, _ = _render_path_card("Opportunities CSV", OPPORTUNITIES_PATH)
    with c3:
        st.markdown("**Heartbeat file**")
        st.code(str(HEARTBEAT_PATH.resolve()), language="text")
        if not HEARTBEAT_PATH.is_file():
            st.warning("Missing heartbeat.txt")
        else:
            st.caption(
                f"Last modified: **{_file_mtime_iso(HEARTBEAT_PATH) or '—'}** · Size: **{_file_size_bytes(HEARTBEAT_PATH)}** bytes · age_sec≈**{hb_age if hb_age is not None else '—'}**"
            )

    st.subheader("Log files (paths only)")
    st.text(f"live_bot: {LIVE_BOT_LOG_PATH.resolve()}")
    st.text(f"dashboard: {DASHBOARD_LOG_PATH.resolve()}")

    st.subheader("Trade Context Import Diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Importer status", "OK" if _HAS_TRADE_CONTEXT_EXPORTER else "FAILED")
    d2.metric("PYTHONPATH set", "yes" if _TRADE_CONTEXT_IMPORT_DIAG.get("py_path") else "no")
    d3.metric("src in sys.path", "yes" if str(_SRC) in sys.path else "no")
    st.caption(f"Module: `{_TRADE_CONTEXT_IMPORT_DIAG.get('module')}`")
    st.caption(f"Loaded from: `{_TRADE_CONTEXT_IMPORT_DIAG.get('module_file') or 'n/a'}`")
    if not _HAS_TRADE_CONTEXT_EXPORTER:
        st.error(f"Import failed: {_TRADE_CONTEXT_IMPORT_DIAG.get('exception') or 'unknown error'}")
        tb = _TRADE_CONTEXT_IMPORT_DIAG.get("traceback")
        if tb:
            st.code(str(tb), language="text")

    quarantined_opp, quarantined_reason = _quarantine_invalid_opportunities_csv(OPPORTUNITIES_PATH)

    raw_trades_csv, trades_csv_err = _read_csv_safe(PAPER_TRADES_PATH)
    raw_opps_csv, opps_csv_err = _read_csv_safe(OPPORTUNITIES_PATH)
    trades_missing_cols = _missing_required_columns(raw_trades_csv, EXPECTED_TRADE_COLS) if not raw_trades_csv.empty else []

    df = load_paper_trades(PAPER_TRADES_PATH)
    opps, opps_fallback_applied = load_opportunities(OPPORTUNITIES_PATH)
    opps_missing_cols = _missing_required_columns(opps, EXPECTED_OPP_COLS) if not opps.empty else []
    valid_trades = df[df.get("_valid_trade", pd.Series(False, index=df.index)).fillna(False)].copy() if not df.empty else df
    chron = valid_trades.sort_values("_exit_ts", na_position="last", ascending=True) if not valid_trades.empty else valid_trades
    r = chron["r_value"] if not chron.empty else pd.Series(dtype=float)
    cum_r = r.cumsum() if not r.empty else pd.Series(dtype=float)
    wins = (r > 0).sum() if not r.empty else 0
    n = int(len(chron))

    st.subheader("Summary")
    if trades_csv_err is not None or trades_missing_cols:
        st.warning(
            f"Analytics input issue: row_count={len(df)} detected_columns={list(df.columns)} missing_columns={trades_missing_cols}"
        )
    total_r = float(r.sum()) if not r.empty else 0.0
    win_rate = float(wins / n) if n else 0.0
    avg_r = float(r.mean()) if n else 0.0
    best_r = float(r.max()) if n else 0.0
    worst_r = float(r.min()) if n else 0.0
    mdd_r = max_drawdown_r(cum_r.reset_index(drop=True)) if not cum_r.empty else 0.0
    lose_streak = max_losing_streak_r(chron["r_value"].reset_index(drop=True)) if n else 0
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

    st.subheader("Equity / EV Curve")
    if n < 1:
        st.warning(
            f"Could not render curves: row_count={n} detected_columns={list(chron.columns)} missing_columns={_missing_required_columns(chron, ['r_value'])}"
        )
    else:
        curve_l, curve_r = st.columns(2)
        with curve_l:
            if n < 2:
                st.info("not enough trades for EV chart")
            else:
                ev_df = pd.DataFrame({"trade #": range(1, len(chron) + 1), "ev_R": r.expanding(min_periods=1).mean().values})
                st.line_chart(ev_df, x="trade #", y="ev_R")
        with curve_r:
            equity_df = pd.DataFrame({"trade #": range(1, len(chron) + 1), "equity_R": cum_r.values})
            st.line_chart(equity_df, x="trade #", y="equity_R")

    st.subheader("Breakdown by outcome/regime")
    if n < 1:
        st.warning(
            f"Could not render breakdown: row_count={n} detected_columns={list(df.columns)} missing_columns={_missing_required_columns(df, ['outcome','regime','r_value'])}"
        )
    else:
        b1, b2 = st.columns(2)
        with b1:
            oc = df.groupby("outcome", dropna=False)["r_value"].agg(trades="count", total_R="sum", avg_R="mean").sort_values("trades", ascending=False)
            st.dataframe(oc, use_container_width=True)
        with b2:
            rg = df.groupby("regime", dropna=False)["r_value"].agg(trades="count", total_R="sum", avg_R="mean").sort_values("trades", ascending=False)
            st.dataframe(rg, use_container_width=True)

    st.subheader("Detailed Trade Analysis")
    context_df = pd.DataFrame()
    context_path: Path | None = None
    selected_entry_ts = ""
    selected_exit_ts = ""
    try:
        if df.empty:
            st.info("No trades available for detailed analysis.")
        else:
            trade_selector_df = df.sort_values("_entry_ts", ascending=False, na_position="last").reset_index(drop=True)
            trade_selector_df["_trade_label"] = trade_selector_df.apply(
                lambda r: f"entry={r.get('entry_ts', 'n/a')} | exit={r.get('exit_ts', 'n/a')} | side={r.get('side', 'n/a')} | outcome={r.get('outcome', 'n/a')} | R={r.get('r_value', 'n/a')}",
                axis=1,
            )
            selected_label = st.selectbox("Select a paper trade", options=trade_selector_df["_trade_label"].tolist(), index=0, key="tjtb_trade_context_selector")
            selected_trade = trade_selector_df.loc[trade_selector_df["_trade_label"] == selected_label].iloc[0]
            selected_entry_ts = str(selected_trade.get("entry_ts", "") or "")
            selected_exit_ts = str(selected_trade.get("exit_ts", "") or "")
            st.dataframe(selected_trade.to_frame().T, use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Detailed Trade Analysis error: {exc}")

    st.subheader("Trade Context Export")
    st.caption("Full microstructure context: bid/ask, sizes, aggressive flow, imbalance, queue imbalance, microprice, spread, volatility, anomaly/regime/action.")
    try:
        if not _HAS_TRADE_CONTEXT_EXPORTER or export_trade_context is None:
            st.info("Trade context exporter module is unavailable on this environment.")
        elif not selected_entry_ts:
            st.warning("Selected trade has no valid entry timestamp.")
        else:
            export_result = export_trade_context(entry_ts=selected_entry_ts, exit_ts=selected_exit_ts if selected_exit_ts else None)
            if isinstance(export_result, tuple) and len(export_result) >= 3:
                context_df, context_path, context_msg = export_result[0], export_result[1], export_result[2]
            else:
                context_df, context_path = export_result
                context_msg = None
            if context_msg:
                st.info(str(context_msg))
            required_preview_cols = [
                "ts", "bid", "ask", "bid_size", "ask_size", "price", "size", "side", "buy_volume", "sell_volume",
                "aggressive_buyers", "aggressive_sellers", "imbalance", "queue_imbalance", "microprice",
                "spread", "volatility", "anomaly_score", "anomaly_percentile", "direction", "regime", "action", "row_phase",
            ]
            missing_preview = [c for c in required_preview_cols if c not in context_df.columns]
            if missing_preview:
                st.warning(f"Context preview missing columns: {missing_preview}")
                st.info(f"Available columns: {list(context_df.columns)}")
            preview_cols = [c for c in required_preview_cols if c in context_df.columns]
            preview_df = context_df[preview_cols].copy() if preview_cols else context_df.copy()
            st.caption(f"Selected trade context rows: **{len(context_df)}**")
            st.dataframe(preview_df.head(500), use_container_width=True, hide_index=True)
            st.download_button(
                "Download selected trade full context CSV",
                data=context_df.to_csv(index=False).encode("utf-8"),
                file_name=context_path.name if context_path is not None else "trade_context.csv",
                mime="text/csv",
                key=f"tjtb_download_trade_context_{selected_entry_ts}",
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Trade Context Export error: {exc}")

    st.subheader("Opportunities table")
    if opps_fallback_applied:
        st.info("fallback reader applied successfully")
    if quarantined_opp:
        st.warning("opportunities.csv schema/data invalid")
        st.info(f"Malformed opportunities.csv quarantined to `data/live/opportunities.bad.csv` ({quarantined_reason})")
    if opps_csv_err is not None:
        st.warning("opportunities.csv schema/data invalid")
        st.info(f"opportunities.csv read status: {opps_csv_err}")
    elif (not opps_fallback_applied) and opps_missing_cols:
        st.warning("opportunities.csv schema/data invalid")
        st.info(f"Missing required columns: {opps_missing_cols}")
    elif (not opps_fallback_applied) and (opps.empty or _opportunities_invalid(opps)):
        st.warning("opportunities.csv schema/data invalid")
    st.download_button(
        "Download opportunities.csv",
        data=(opps.to_csv(index=False).encode("utf-8") if not opps.empty else b""),
        file_name="opportunities.csv",
        mime="text/csv",
        key="tjtb_download_opportunities_csv",
    )
    st.caption(f"opportunities.csv row count: **{len(opps)}**")
    st.dataframe(opps.tail(50), use_container_width=True, hide_index=True)

    st.subheader("Paper trades table")
    if trades_csv_err is not None:
        st.warning("paper_trades.csv schema/data invalid")
        st.info(f"paper_trades.csv read status: {trades_csv_err}")
    elif trades_missing_cols:
        st.warning("paper_trades.csv schema/data invalid")
        st.info(f"Missing required columns: {trades_missing_cols}")
    display_cols = ["exit_ts", "entry_ts", "side", "entry_price", "exit_price", "outcome", "r_value", "regime"]
    table = df.sort_values("_exit_ts", ascending=False, na_position="last")[display_cols].copy() if not df.empty else pd.DataFrame(columns=display_cols)
    st.download_button(
        "Download paper_trades.csv",
        data=(table.to_csv(index=False).encode("utf-8") if not table.empty else b""),
        file_name="paper_trades.csv",
        mime="text/csv",
        key="tjtb_download_paper_trades_csv",
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Debug diagnostics", expanded=False):
        st.markdown("**Trade context import diagnostics**")
        st.write(f"module: `{_TRADE_CONTEXT_IMPORT_DIAG.get('module')}`")
        st.write(f"loaded_from: `{_TRADE_CONTEXT_IMPORT_DIAG.get('module_file') or 'n/a'}`")
        st.write(f"importer_status: `{_TRADE_CONTEXT_IMPORT_DIAG.get('ok')}`")
        if not _HAS_TRADE_CONTEXT_EXPORTER:
            st.code(str(_TRADE_CONTEXT_IMPORT_DIAG.get("traceback") or _TRADE_CONTEXT_IMPORT_DIAG.get("exception")), language="text")
        st.markdown("**paper_trades.csv**")
        st.write(f"path: `{PAPER_TRADES_PATH}`")
        st.write(f"exists: `{PAPER_TRADES_PATH.is_file()}`")
        st.write(f"row_count(excl header): `{_csv_data_row_count(PAPER_TRADES_PATH)}`")
        st.write(f"last_modified: `{_file_mtime_iso(PAPER_TRADES_PATH)}`")
        st.write(f"columns: `{list(raw_trades_csv.columns) if not raw_trades_csv.empty else []}`")
        if not raw_trades_csv.empty:
            st.dataframe(raw_trades_csv.head(3), use_container_width=True, hide_index=True)
            st.dataframe(raw_trades_csv.tail(3), use_container_width=True, hide_index=True)
        st.markdown("**raw NDJSON key sample**")
        sample_keys_fn = _TRADE_CONTEXT_IMPORT_DIAG.get("sample_keys_fn")
        if callable(sample_keys_fn):
            try:
                keys = sample_keys_fn()
                st.write(f"sample raw keys: `{keys}`")
            except Exception as exc:  # noqa: BLE001
                st.write(f"sample raw keys error: `{exc}`")
        else:
            st.write("sample raw keys unavailable")
        st.markdown("**opportunities.csv**")
        st.write(f"path: `{OPPORTUNITIES_PATH}`")
        st.write(f"exists: `{OPPORTUNITIES_PATH.is_file()}`")
        st.write(f"row_count(excl header): `{_csv_data_row_count(OPPORTUNITIES_PATH)}`")
        st.write(f"last_modified: `{_file_mtime_iso(OPPORTUNITIES_PATH)}`")
        st.write(f"columns: `{list(raw_opps_csv.columns) if not raw_opps_csv.empty else []}`")
        if not raw_opps_csv.empty:
            st.dataframe(raw_opps_csv.head(3), use_container_width=True, hide_index=True)
            st.dataframe(raw_opps_csv.tail(3), use_container_width=True, hide_index=True)

    refreshed_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.session_state["tjtb_last_refreshed_at"] = refreshed_at

    with st.sidebar:
        st.divider()
        st.caption("Last refreshed at")
        st.write(st.session_state["tjtb_last_refreshed_at"])
        if auto_on and not hasattr(st, "autorefresh"):
            st.warning("Auto refresh needs `st.autorefresh` (newer Streamlit). Use **Refresh now** or upgrade Streamlit.")


if __name__ == "__main__":
    main()
