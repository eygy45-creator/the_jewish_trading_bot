"""
Research report for anomaly-tail setup outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY = Path("data/parsed/anomaly_matrix.csv")
DEFAULT_SETUPS = Path("data/parsed/anomaly_setups.csv")
DEFAULT_OUTPUT = Path("reports/anomaly_summary.json")
READ_BUFFER = 1024 * 1024
SMALL_SAMPLE_N = 30


def _safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    s = value.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _safe_int(value: str | None) -> int | None:
    v = _safe_float(value)
    if math.isnan(v):
        return None
    return int(v)


def _truthy_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", buffering=READ_BUFFER) as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _anomaly_flags_by_ts(matrix_rows: list[dict[str, str]]) -> dict[str, dict[str, bool]]:
    """Last row wins if duplicate timestamps."""
    out: dict[str, dict[str, bool]] = {}
    for r in matrix_rows:
        ts = (r.get("ts") or "").strip()
        if not ts:
            continue
        out[ts] = {
            "is_extreme_10pct": _truthy_flag(r.get("is_extreme_10pct")),
            "is_extreme_5pct": _truthy_flag(r.get("is_extreme_5pct")),
        }
    return out


def _setups_by_ts(setup_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in setup_rows:
        ts = (r.get("ts") or "").strip()
        if ts:
            out[ts] = r
    return out


def _label_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {"-1": 0, "0": 0, "1": 0, "n": 0}
    for r in rows:
        label = _safe_int(r.get("label"))
        if label is None:
            continue
        if label < -1:
            label = -1
        elif label > 1:
            label = 1
        out[str(label)] += 1
        out["n"] += 1
    return out


def _mean_label(rows: list[dict[str, str]]) -> float:
    vals: list[float] = []
    for r in rows:
        label = _safe_int(r.get("label"))
        if label is None:
            continue
        vals.append(float(label))
    return float(mean(vals)) if vals else float("nan")


def _hit_rate(rows: list[dict[str, str]]) -> float:
    vals: list[float] = []
    for r in rows:
        label = _safe_int(r.get("label"))
        if label is None:
            continue
        vals.append(1.0 if label == 1 else 0.0)
    return float(mean(vals)) if vals else float("nan")


def _bucket_summary(rows: list[dict[str, str]], key: str, n_buckets: int) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, str]]] = [[] for _ in range(n_buckets)]
    for r in rows:
        x = _safe_float(r.get(key))
        if math.isnan(x):
            continue
        if x < 0:
            idx = 0
        elif x >= 1:
            idx = n_buckets - 1
        else:
            idx = int(x * n_buckets)
            if idx >= n_buckets:
                idx = n_buckets - 1
        buckets[idx].append(r)

    out: list[dict[str, Any]] = []
    for i in range(n_buckets):
        lo = i / float(n_buckets)
        hi = (i + 1) / float(n_buckets)
        grp = buckets[i]
        avg_outcome = _mean_label(grp)
        rec: dict[str, Any] = {
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n": len(grp),
            "avg_outcome": None if math.isnan(avg_outcome) else avg_outcome,
        }
        if len(grp) < SMALL_SAMPLE_N:
            rec["small_sample_warning"] = True
        out.append(rec)
    return out


def _ev_by_percentile_bucket(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    bounds = [(0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    out: list[dict[str, Any]] = []
    for lo, hi in bounds:
        grp: list[dict[str, str]] = []
        for r in rows:
            p = _safe_float(r.get("anomaly_percentile_rank"))
            if math.isnan(p):
                continue
            if p >= lo and p < hi:
                grp.append(r)
        ev = _mean_label(grp)
        rec: dict[str, Any] = {
            "bucket": f"[{lo:.2f},{min(hi, 1.0):.2f})",
            "n": len(grp),
            "ev": None if math.isnan(ev) else ev,
        }
        if len(grp) < SMALL_SAMPLE_N:
            rec["small_sample_warning"] = True
        out.append(rec)
    return out


def _trade_rule_metrics(
    rows: list[dict[str, str]],
    *,
    hit_label: int,
    wrong_label: int,
    costs_per_trade: float,
) -> dict[str, Any]:
    """
    Forward-label semantics from label_anomaly_setups:
    +1 first hits profit-up before adverse-down; -1 first hits profit-down; 0 neither.

    Trade hypothesis:
    - hit_label: label value counted as favorable for the tested trade direction.
    - wrong_label: label value counted as adverse.
    - label == 0: neutral (no barrier hit in horizon).

    EV per row: +1 hit, -1 wrong, 0 neutral; then subtract costs_per_trade.
    max_losing_streak: max consecutive rows with label == wrong_label.
    """
    empty = {
        "n": 0,
        "hit_rate": None,
        "neutral_rate": None,
        "wrong_direction_rate": None,
        "avg_outcome": None,
        "ev_after_costs": None,
        "max_losing_streak": 0,
        "avg_max_favorable_move": None,
        "avg_max_adverse_move": None,
        "costs_per_trade": costs_per_trade,
        "hit_label": hit_label,
        "wrong_label": wrong_label,
        "small_sample_warning": True,
    }

    if not rows:
        return dict(empty)

    hits = 0
    neutrals = 0
    wrongs = 0
    pnl_scores: list[float] = []
    fav_moves: list[float] = []
    adv_moves: list[float] = []

    for r in rows:
        lab = _safe_int(r.get("label"))
        if lab is None:
            continue
        if lab < -1:
            lab = -1
        elif lab > 1:
            lab = 1

        if lab == hit_label:
            hits += 1
            pnl_scores.append(1.0)
        elif lab == wrong_label:
            wrongs += 1
            pnl_scores.append(-1.0)
        elif lab == 0:
            neutrals += 1
            pnl_scores.append(0.0)
        else:
            # e.g. hit_label=1 wrong_label=-1 but lab is unexpected; skip counting
            continue

        mf = _safe_float(r.get("max_favorable_move"))
        ma = _safe_float(r.get("max_adverse_move"))
        if not math.isnan(mf):
            fav_moves.append(mf)
        if not math.isnan(ma):
            adv_moves.append(ma)

    counted = hits + neutrals + wrongs
    if counted == 0:
        out = dict(empty)
        out["avg_max_favorable_move"] = float(mean(fav_moves)) if fav_moves else None
        out["avg_max_adverse_move"] = float(mean(adv_moves)) if adv_moves else None
        return out

    hit_rate = hits / float(counted)
    neutral_rate = neutrals / float(counted)
    wrong_rate = wrongs / float(counted)
    raw_labels = [x for x in (_safe_int(r.get("label")) for r in rows) if x is not None]
    avg_outcome = float(mean(float(x) for x in raw_labels)) if raw_labels else float("nan")

    mean_pnl = float(mean(pnl_scores)) if pnl_scores else float("nan")
    ev_after = mean_pnl - float(costs_per_trade) if not math.isnan(mean_pnl) else float("nan")

    max_streak = 0
    cur = 0
    for r in rows:
        lab = _safe_int(r.get("label"))
        if lab is None:
            continue
        if lab == wrong_label:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    return {
        "n": counted,
        "hit_rate": hit_rate,
        "neutral_rate": neutral_rate,
        "wrong_direction_rate": wrong_rate,
        "avg_outcome": None if math.isnan(avg_outcome) else avg_outcome,
        "ev_after_costs": None if math.isnan(ev_after) else ev_after,
        "max_losing_streak": max_streak,
        "avg_max_favorable_move": float(mean(fav_moves)) if fav_moves else None,
        "avg_max_adverse_move": float(mean(adv_moves)) if adv_moves else None,
        "costs_per_trade": costs_per_trade,
        "hit_label": hit_label,
        "wrong_label": wrong_label,
        "small_sample_warning": counted < SMALL_SAMPLE_N,
    }


def _directional_rule_block(rows: list[dict[str, str]], *, direction: str, costs_per_trade: float) -> dict[str, Any]:
    """Bearish continuation: short favors label -1. Bullish continuation: long favors label +1."""
    if direction == "bearish":
        block = _trade_rule_metrics(rows, hit_label=-1, wrong_label=1, costs_per_trade=costs_per_trade)
    else:
        block = _trade_rule_metrics(rows, hit_label=1, wrong_label=-1, costs_per_trade=costs_per_trade)
    block.pop("hit_label", None)
    block.pop("wrong_label", None)
    return block


def _filter_setups_extreme5_direction(
    matrix_rows: list[dict[str, str]],
    setup_rows: list[dict[str, str]],
    *,
    anomaly_direction: str,
) -> list[dict[str, str]]:
    """Setup rows whose ts has is_extreme_5pct on matrix and anomaly_direction matches."""
    flags = _anomaly_flags_by_ts(matrix_rows)
    want = anomaly_direction.strip().lower()
    out: list[dict[str, str]] = []
    for r in setup_rows:
        ts = (r.get("ts") or "").strip()
        if not ts:
            continue
        if (r.get("anomaly_direction") or "").strip().lower() != want:
            continue
        if flags.get(ts, {}).get("is_extreme_5pct"):
            out.append(r)
    return out


def _join_extreme_setups(
    matrix_rows: list[dict[str, str]],
    setup_rows: list[dict[str, str]],
    *,
    extreme_key: str,
) -> list[dict[str, str]]:
    """Rows from setups whose ts has extreme_key True on anomaly_matrix (fixes 10 vs 5 pct mix-up)."""
    flags = _anomaly_flags_by_ts(matrix_rows)
    setups_by_ts = _setups_by_ts(setup_rows)
    out: list[dict[str, str]] = []
    for ts, fl in flags.items():
        if not fl.get(extreme_key):
            continue
        s = setups_by_ts.get(ts)
        if s is not None:
            out.append(s)
    return out


def build_anomaly_report(
    anomaly_matrix_path: Path,
    setups_path: Path,
    output_path: Path,
    *,
    costs_per_trade: float,
) -> dict[str, Any]:
    anomaly_rows = _load_csv(anomaly_matrix_path)
    setup_rows = _load_csv(setups_path)

    total_rows = len(anomaly_rows)
    extreme10_total = sum(1 for r in anomaly_rows if _truthy_flag(r.get("is_extreme_10pct")))
    extreme5_total = sum(1 for r in anomaly_rows if _truthy_flag(r.get("is_extreme_5pct")))

    setup_extreme10 = _join_extreme_setups(anomaly_rows, setup_rows, extreme_key="is_extreme_10pct")
    setup_extreme5 = _join_extreme_setups(anomaly_rows, setup_rows, extreme_key="is_extreme_5pct")

    label_dist_10 = _label_distribution(setup_extreme10)
    label_dist_5 = _label_distribution(setup_extreme5)
    label_dist_10["n_extreme_flags_in_matrix"] = extreme10_total
    label_dist_10["n_setups_joined"] = len(setup_extreme10)
    label_dist_5["n_extreme_flags_in_matrix"] = extreme5_total
    label_dist_5["n_setups_joined"] = len(setup_extreme5)

    by_direction: dict[str, list[dict[str, str]]] = {"bullish": [], "bearish": [], "neutral": []}
    for r in setup_rows:
        d = (r.get("anomaly_direction") or "neutral").strip().lower()
        if d not in by_direction:
            d = "neutral"
        by_direction[d].append(r)

    bearish_rows = [r for r in setup_rows if (r.get("anomaly_direction") or "").strip().lower() == "bearish"]
    bullish_rows = [r for r in setup_rows if (r.get("anomaly_direction") or "").strip().lower() == "bullish"]

    rule_a_rows = _filter_setups_extreme5_direction(anomaly_rows, setup_rows, anomaly_direction="bearish")
    rule_bc_rows = _filter_setups_extreme5_direction(anomaly_rows, setup_rows, anomaly_direction="bullish")

    rule_tests = {
        "rule_a_bearish_extreme5_short_continuation": _trade_rule_metrics(
            rule_a_rows,
            hit_label=-1,
            wrong_label=1,
            costs_per_trade=costs_per_trade,
        ),
        "rule_b_bullish_extreme5_long_continuation": _trade_rule_metrics(
            rule_bc_rows,
            hit_label=1,
            wrong_label=-1,
            costs_per_trade=costs_per_trade,
        ),
        "rule_c_bullish_extreme5_short_reversal": _trade_rule_metrics(
            rule_bc_rows,
            hit_label=-1,
            wrong_label=1,
            costs_per_trade=costs_per_trade,
        ),
        "rule_descriptions": {
            "rule_a": "Bearish anomaly + is_extreme_5pct + trade = short continuation (hit label -1, wrong label +1).",
            "rule_b": "Bullish anomaly + is_extreme_5pct + trade = long continuation (hit label +1, wrong label -1).",
            "rule_c": "Bullish anomaly + is_extreme_5pct + trade = short reversal (hit label -1, wrong label +1); same rows as Rule B, different hypothesis.",
            "join": "Rows filtered from anomaly_setups joined to anomaly_matrix on ts for is_extreme_5pct.",
            "labels": "From label_anomaly_setups: +1 up-first, -1 down-first, 0 neither within horizon.",
        },
    }
    for k in ("rule_a_bearish_extreme5_short_continuation", "rule_b_bullish_extreme5_long_continuation", "rule_c_bullish_extreme5_short_reversal"):
        rule_tests[k].pop("hit_label", None)
        rule_tests[k].pop("wrong_label", None)

    report: dict[str, Any] = {
        "total_rows": total_rows,
        "extreme_10pct_events": extreme10_total,
        "extreme_5pct_events": extreme5_total,
        "label_distribution_extreme_10pct": label_dist_10,
        "label_distribution_extreme_5pct": label_dist_5,
        "average_outcome_by_anomaly_decile": _bucket_summary(setup_rows, "anomaly_percentile_rank", 10),
        "average_outcome_by_anomaly_direction": {
            k: {
                "n": len(v),
                "avg_outcome": None if math.isnan(_mean_label(v)) else _mean_label(v),
                "small_sample_warning": len(v) < SMALL_SAMPLE_N,
            }
            for k, v in by_direction.items()
        },
        "ev_by_anomaly_percentile_bucket": _ev_by_percentile_bucket(setup_rows),
        "hit_rate_bullish_anomalies": None if math.isnan(_hit_rate(by_direction["bullish"])) else _hit_rate(by_direction["bullish"]),
        "hit_rate_bearish_anomalies": None if math.isnan(_hit_rate(by_direction["bearish"])) else _hit_rate(by_direction["bearish"]),
        "directional_rule_evaluation": {
            "bearish_anomalies_only": _directional_rule_block(bearish_rows, direction="bearish", costs_per_trade=costs_per_trade),
            "bullish_anomalies_only": _directional_rule_block(bullish_rows, direction="bullish", costs_per_trade=costs_per_trade),
            "rule_notes": {
                "bearish_hit": "label == -1",
                "bearish_neutral": "label == 0",
                "bearish_wrong": "label == 1",
                "bullish_hit": "label == 1",
                "bullish_neutral": "label == 0",
                "bullish_wrong": "label == -1",
                "ev_after_costs": "mean(+1 on hit, -1 on wrong, 0 on neutral) minus costs_per_trade; research-only, not dollar PnL.",
                "max_losing_streak": "max consecutive wrong-direction labels only.",
            },
        },
        "rule_tests_extreme_5pct": rule_tests,
        "notes": [
            "Research output only; no profitability claims.",
            "Bucket-level reliability depends on sample size.",
            f"Buckets with n < {SMALL_SAMPLE_N} are explicitly marked as small-sample.",
            "label_distribution_extreme_* uses anomaly_matrix extreme flags joined to anomaly_setups by ts.",
            "Rule B vs Rule C uses the same bullish extreme-5pct rows to compare long continuation vs short reversal hypotheses.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    p = argparse.ArgumentParser(description="Build anomaly-tail research summary report.")
    p.add_argument("--anomaly-matrix", type=Path, default=DEFAULT_ANOMALY)
    p.add_argument("--anomaly-setups", type=Path, default=DEFAULT_SETUPS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--costs-per-trade",
        type=float,
        default=0.0,
        help="Subtracted from mean directional score (+1/-1/0) for ev_after_costs in rule evaluation blocks.",
    )
    args = p.parse_args(argv)

    if args.costs_per_trade < 0:
        logger.error("--costs-per-trade must be >= 0")
        return 1

    if not args.anomaly_matrix.is_file():
        logger.error("missing anomaly matrix: %s", args.anomaly_matrix)
        return 1
    if not args.anomaly_setups.is_file():
        logger.error("missing anomaly setups: %s", args.anomaly_setups)
        return 1

    report = build_anomaly_report(
        args.anomaly_matrix,
        args.anomaly_setups,
        args.output,
        costs_per_trade=args.costs_per_trade,
    )
    logger.info(
        "wrote anomaly report rows=%s extreme10=%s extreme5=%s -> %s",
        report["total_rows"],
        report["extreme_10pct_events"],
        report["extreme_5pct_events"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
