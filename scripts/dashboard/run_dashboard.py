"""
Minimal Streamlit research dashboard for parsed microstructure pipeline outputs.

Run from repository root:
  streamlit run scripts/dashboard/run_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run scripts/dashboard/run_dashboard.py` without PYTHONPATH=src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from tjtb.reports.dashboard_data import (
    DEFAULT_FILES,
    apply_plot_sample,
    coerce_datetime_columns,
    file_loading_report,
    first_existing_column,
    numeric_columns,
    try_read_csv,
)

st.set_page_config(page_title="TJTB Research Dashboard", layout="wide")
st.title("The Jewish Trading Bot — Research Dashboard")
st.caption("Non-production inspection of parsed book state, labels, features, and datasets.")

# --- Sidebar ---
st.sidebar.header("Data & sampling")
max_rows = st.sidebar.number_input("Max rows to plot", min_value=500, max_value=500_000, value=20_000, step=500)
sample_mode = st.sidebar.selectbox("Sample mode", ("tail", "head", "random"), index=0)

paths: dict[str, Path] = {k: (_ROOT / v).resolve() if not v.is_absolute() else v for k, v in DEFAULT_FILES.items()}
report = file_loading_report(paths)
st.sidebar.subheader("File status")
for key, info in report.items():
    icon = "ok" if info["found"] else "missing"
    st.sidebar.write(f"**{key}** ({icon}): {info['rows']} rows")

# --- Load dataframes (lazy but cached) ---
@st.cache_data(show_spinner=False)
def _load_all(paths_serial: tuple[tuple[str, str], ...]) -> dict[str, pd.DataFrame | None]:
    out: dict[str, pd.DataFrame | None] = {}
    for key, pstr in paths_serial:
        p = Path(pstr)
        df = try_read_csv(p)
        if df is not None and "ts" in df.columns:
            df = coerce_datetime_columns(df, ("ts",))
        out[key] = df
    return out


paths_serial = tuple((k, str(v)) for k, v in paths.items())
dfs = _load_all(paths_serial)

# --- 1. Overview ---
st.header("1. Overview")
cols = st.columns(4)
for i, (key, info) in enumerate(report.items()):
    with cols[i % 4]:
        st.metric(label=key, value=info["rows"] if info["found"] else "—")
        if info["found"] and info["ts_range"]:
            st.caption(f"ts: {info['ts_range'][0]} → {info['ts_range'][1]}")
        elif not info["found"]:
            st.warning(f"Missing: `{info['path']}`")

missing = [k for k, v in report.items() if not v["found"]]
if missing:
    st.info("Some files are missing — sections that depend on them will show warnings below.")

# --- helpers for plotting ---
def _plot_df(name: str, df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    return apply_plot_sample(df, int(max_rows), sample_mode)


def _warn_missing(cols_needed: list[str], df: pd.DataFrame | None, section: str) -> bool:
    if df is None:
        st.warning(f"{section}: no dataframe loaded.")
        return True
    miss = [c for c in cols_needed if c not in df.columns]
    if miss:
        st.warning(f"{section}: missing columns {miss}. Available: {list(df.columns)}")
        return True
    return False


# --- 2. Book state ---
st.header("2. Book state diagnostics")
bs = dfs.get("book_state")
plot_bs = _plot_df("book_state", bs)
if plot_bs is None:
    st.warning("book_state.csv not available.")
else:
    if not _warn_missing(["best_bid", "best_ask", "ts"], plot_bs, "Book"):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(plot_bs["ts"], plot_bs["best_bid"], label="best_bid", lw=0.8)
        ax.plot(plot_bs["ts"], plot_bs["best_ask"], label="best_ask", lw=0.8)
        ax.legend()
        ax.set_title("Best bid / ask (sampled)")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

    if "spread" in plot_bs.columns and "ts" in plot_bs.columns:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(plot_bs["ts"], plot_bs["spread"], color="tab:purple", lw=0.8)
        ax.set_title("Spread (sampled)")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)

    mid_c = first_existing_column(plot_bs, ("mid_price",))
    mic_c = first_existing_column(plot_bs, ("microprice",))
    if mid_c and mic_c and "ts" in plot_bs.columns:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(plot_bs["ts"], plot_bs[mid_c], label=mid_c, lw=0.8)
        ax.plot(plot_bs["ts"], plot_bs[mic_c], label=mic_c, lw=0.8)
        ax.legend()
        ax.set_title("Mid vs microprice (sampled)")
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Book: mid_price / microprice columns not found.")

    tob_c = first_existing_column(plot_bs, ("top_of_book_imbalance", "tob_imbalance"))
    if tob_c:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(plot_bs[tob_c].dropna(), bins=50, color="tab:blue", alpha=0.85)
        ax.set_title(f"Histogram: {tob_c}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Book: no top-of-book imbalance column for histogram.")

# --- 3. Labels ---
st.header("3. Labels diagnostics")
lb = dfs.get("labeled_book")
if lb is None or lb.empty:
    st.warning("labeled_book.csv not available.")
else:
    if "label" in lb.columns:
        vc = lb["label"].value_counts().reindex([-1, 0, 1], fill_value=0)
        st.write("Label counts (+1 / -1 / 0):")
        st.write(vc.to_dict())
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar([str(i) for i in vc.index], vc.values.astype(float), color=["tab:red", "tab:gray", "tab:green"])
        ax.set_title("Label distribution")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Labels: `label` column missing.")

    for col, title in (("up_move", "up_move"), ("down_move", "down_move")):
        if col in lb.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(lb[col].dropna(), bins=50, color="tab:orange", alpha=0.85)
            ax.set_title(f"Histogram: {title}")
            st.pyplot(fig)
            plt.close(fig)

# --- 4. Features ---
st.header("4. Feature diagnostics")
fm = dfs.get("feature_matrix")
ds = dfs.get("dataset")
feat_df = ds if ds is not None and not ds.empty else fm
feat_plot = _plot_df("features", feat_df)
if feat_plot is None or feat_plot.empty:
    st.warning("feature_matrix.csv / dataset.csv not available for feature diagnostics.")
else:
    st.write("Numeric columns detected:")
    nums = numeric_columns(feat_plot, exclude={"ts"})
    st.code(", ".join(nums) if nums else "(none)")

    if len(nums) >= 2:
        corr = feat_plot[nums].corr()
        fig, ax = plt.subplots(figsize=(0.4 * len(nums) + 2, 0.4 * len(nums) + 2))
        im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(nums)), labels=nums, rotation=45, ha="right")
        ax.set_yticks(range(len(nums)), labels=nums)
        ax.set_title("Feature correlation matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)

    def _hist_feature(candidates: tuple[str, ...], title: str) -> None:
        c = first_existing_column(feat_plot, candidates)
        if not c:
            st.warning(f"Feature histogram skipped ({title}): no matching column in {list(feat_plot.columns)}.")
            return
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(pd.to_numeric(feat_plot[c], errors="coerce").dropna(), bins=50, color="tab:cyan", alpha=0.85)
        ax.set_title(f"{title} (`{c}`)")
        st.pyplot(fig)
        plt.close(fig)

    _hist_feature(("spread",), "Spread")
    _hist_feature(("top_of_book_imbalance", "tob_imbalance"), "Top-of-book imbalance")
    _hist_feature(("microprice_dev", "microprice_deviation"), "Microprice deviation")
    _hist_feature(("ofi_k", "ofi"), "Rolling order flow imbalance")
    _hist_feature(("mid_vol_w", "volatility", "vol_proxy"), "Volatility proxy")

    imb_c = first_existing_column(feat_plot, ("top_of_book_imbalance", "tob_imbalance"))
    lab_c = "label" if "label" in feat_plot.columns else None
    if imb_c and lab_c:
        fig, ax = plt.subplots(figsize=(5, 4))
        tmp = feat_plot[[imb_c, lab_c]].dropna()
        tmp[lab_c] = pd.to_numeric(tmp[lab_c], errors="coerce")
        tmp[imb_c] = pd.to_numeric(tmp[imb_c], errors="coerce")
        tmp = tmp.dropna()
        tmp = apply_plot_sample(tmp, int(max_rows), sample_mode)
        ax.scatter(tmp[imb_c], tmp[lab_c], s=4, alpha=0.35)
        ax.set_xlabel(imb_c)
        ax.set_ylabel("label")
        ax.set_title("Imbalance vs label")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Scatter imbalance vs label: need imbalance + label on the same frame (use dataset.csv).")

# --- 5. Dataset / modeling ---
st.header("5. Dataset / modeling diagnostics")
if ds is None or ds.empty:
    st.warning("dataset.csv not available.")
else:
    st.metric("dataset rows", len(ds))
    if "label" in ds.columns:
        vc = ds["label"].astype(str).value_counts()
        st.write("Class balance:")
        st.bar_chart(vc.to_frame(name="count"))
    st.subheader("Preview")
    st.dataframe(ds.head(50), use_container_width=True)

    ts_c = "ts" if "ts" in ds.columns else None
    if ts_c and "label" in ds.columns:
        d2 = ds[[ts_c, "label"]].copy()
        d2[ts_c] = pd.to_datetime(d2[ts_c], utc=True, errors="coerce")
        d2["label"] = pd.to_numeric(d2["label"], errors="coerce")
        d2 = d2.dropna(subset=[ts_c, "label"]).sort_values(ts_c)
        d2 = apply_plot_sample(d2, int(max_rows), sample_mode)
        if len(d2) > 10:
            idx = pd.to_datetime(d2[ts_c], utc=True, errors="coerce")
            s = pd.Series(d2["label"].values, index=idx).sort_index()
            roll = s.rolling("5min", min_periods=5).mean()
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(roll.index, roll.values, lw=0.9)
            ax.set_title("Rolling mean of label (5min window, time-based index)")
            fig.autofmt_xdate()
            st.pyplot(fig)
            plt.close(fig)

# --- 6. EV / probabilities ---
st.header("6. Signal / EV diagnostics")
ev_cols = [c for c in (ds.columns if ds is not None else []) if "ev" in c.lower() or c.lower().endswith("_ev")]
prob_cols = [c for c in (ds.columns if ds is not None else []) if c.lower().startswith("p_") or "prob" in c.lower()]

frame = ds if ds is not None else fm
if frame is None or frame.empty:
    st.warning("No dataset/feature frame for EV / probability diagnostics.")
else:
    ev_hit = False
    for c in ev_cols:
        if c in frame.columns:
            ev_hit = True
            s = pd.to_numeric(frame[c], errors="coerce").dropna()
            s = s.tail(int(max_rows)) if len(s) > max_rows else s
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(s, bins=50, color="tab:green", alpha=0.85)
            ax.set_title(f"Histogram: {c}")
            st.pyplot(fig)
            plt.close(fig)
            pos = int((s > 0).sum())
            neg = int((s <= 0).sum())
            st.write(f"`{c}` — EV>0: {pos}, EV<=0: {neg}")

    if not ev_hit and not prob_cols:
        st.info("No EV or probability columns detected on dataset.csv (add columns after modeling to populate this section).")

    for c in prob_cols[:6]:
        if c in frame.columns:
            s = pd.to_numeric(frame[c], errors="coerce").dropna()
            s = s.tail(int(max_rows)) if len(s) > max_rows else s
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(s, bins=40, color="tab:brown", alpha=0.85)
            ax.set_title(f"Histogram: {c}")
            st.pyplot(fig)
            plt.close(fig)
