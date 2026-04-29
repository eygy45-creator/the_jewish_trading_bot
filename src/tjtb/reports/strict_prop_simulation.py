"""Strict prop simulation from historical paper trades only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


STARTING_BALANCE = 5000.0
RISK_PCT_PER_TRADE = 0.0025
MAX_RISK_PER_TRADE = 12.5

# Cost assumptions by profile (extreme remains current harsh model).
PROFILE_COSTS = {
    "realistic": {
        "commission_bps_per_side": 4.0,
        "slippage_bps_per_side": 3.0,
        "spread_bps_round_trip": 3.0,
    },
    "conservative": {
        "commission_bps_per_side": 7.0,
        "slippage_bps_per_side": 5.0,
        "spread_bps_round_trip": 5.0,
    },
    "extreme": {
        "commission_bps_per_side": 10.0,
        "slippage_bps_per_side": 8.0,
        "spread_bps_round_trip": 8.0,
    },
}
ASSUMED_STOP_DISTANCE_PCT = 0.006


def _safe_float(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _profit_factor(values: list[float]) -> float:
    gross_profit = sum(v for v in values if v > 0.0)
    gross_loss = abs(sum(v for v in values if v < 0.0))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def _max_losing_streak(values: list[float]) -> int:
    streak = 0
    max_streak = 0
    for v in values:
        if v < 0.0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _run_for_profile(trades: list[dict[str, str]], profile: str, risk_amount: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = PROFILE_COSTS[profile]
    commission_bps = float(cfg["commission_bps_per_side"])
    slippage_bps = float(cfg["slippage_bps_per_side"])
    spread_bps = float(cfg["spread_bps_round_trip"])
    risk_amount = min(STARTING_BALANCE * RISK_PCT_PER_TRADE, MAX_RISK_PER_TRADE)
    inferred_notional = risk_amount / ASSUMED_STOP_DISTANCE_PCT
    commission_cost = inferred_notional * ((2.0 * commission_bps) / 10000.0)
    slippage_cost = inferred_notional * ((2.0 * slippage_bps) / 10000.0)
    spread_cost = inferred_notional * (spread_bps / 10000.0)
    total_cost_per_trade = commission_cost + slippage_cost + spread_cost
    cost_in_r = total_cost_per_trade / risk_amount

    balance = STARTING_BALANCE
    peak = STARTING_BALANCE
    max_drawdown = 0.0

    rows: list[dict[str, object]] = []
    net_pnls: list[float] = []
    net_rs: list[float] = []

    trade_number = 0
    for trade in trades:
        r_value = _safe_float(trade.get("r_value"))
        if r_value is None:
            continue
        trade_number += 1
        gross_pnl = r_value * risk_amount
        execution_costs = total_cost_per_trade
        net_pnl = gross_pnl - execution_costs
        net_r = net_pnl / risk_amount

        balance += net_pnl
        peak = max(peak, balance)
        drawdown = peak - balance
        max_drawdown = max(max_drawdown, drawdown)

        rows.append(
            {
                "trade_number": trade_number,
                "entry_ts": str(trade.get("entry_ts", "")),
                "exit_ts": str(trade.get("exit_ts", "")),
                "side": str(trade.get("side", "")),
                "outcome": str(trade.get("outcome", "")),
                "gross_pnl": gross_pnl,
                "execution_costs": execution_costs,
                "net_pnl": net_pnl,
                "net_r": net_r,
                "balance_after_trade": balance,
                "drawdown": drawdown,
            }
        )
        net_pnls.append(net_pnl)
        net_rs.append(net_r)

    if not rows:
        raise ValueError("No valid trades with numeric r_value in paper_trades.csv.")

    total_trades = len(rows)
    wins = sum(1 for v in net_pnls if v > 0.0)
    win_rate = wins / total_trades
    net_pnl = sum(net_pnls)
    total_fees = total_cost_per_trade * total_trades
    average_r = sum(net_rs) / total_trades
    profit_factor = _profit_factor(net_pnls)
    final_balance = float(rows[-1]["balance_after_trade"])
    verdict = "PASS" if final_balance > STARTING_BALANCE and max_drawdown <= STARTING_BALANCE * 0.10 else "FAIL"

    summary = {
        "profile": profile,
        "metrics": {
            "final_balance": final_balance,
            "net_pnl": net_pnl,
            "total_fees": total_fees,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "average_r": average_r,
            "profit_factor": None if math.isinf(profit_factor) else profit_factor,
            "profit_factor_is_infinite": math.isinf(profit_factor),
            "verdict": verdict,
        },
        "cost_model": {
            "commission_bps_per_side": commission_bps,
            "slippage_bps_per_side": slippage_bps,
            "spread_bps_round_trip": spread_bps,
            "commission_per_trade": commission_cost,
            "slippage_per_trade": slippage_cost,
            "spread_per_trade": spread_cost,
            "cost_per_trade": total_cost_per_trade,
            "cost_in_r": cost_in_r,
        },
        "total_trades": total_trades,
    }
    return rows, summary


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trade_number",
                "entry_ts",
                "exit_ts",
                "side",
                "outcome",
                "gross_pnl",
                "execution_costs",
                "net_pnl",
                "net_r",
                "balance_after_trade",
                "drawdown",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_side_by_side(summaries: dict[str, dict[str, object]]) -> None:
    headers = ["profile", "final_balance", "net_pnl", "total_fees", "max_drawdown", "win_rate", "avg_R", "profit_factor", "verdict", "cost_per_trade", "cost_in_R"]
    print("=== Strict Prop Simulation: Profile Comparison ===")
    print(" | ".join(h.ljust(14) for h in headers))
    print("-" * (17 * len(headers)))
    for profile in ("realistic", "conservative", "extreme"):
        s = summaries[profile]
        m = s["metrics"]
        c = s["cost_model"]
        pf = m["profit_factor"]
        pf_text = "inf" if m["profit_factor_is_infinite"] else f"{float(pf):.4f}"
        row = [
            profile,
            f"${float(m['final_balance']):,.2f}",
            f"${float(m['net_pnl']):,.2f}",
            f"${float(m['total_fees']):,.2f}",
            f"${float(m['max_drawdown']):,.2f}",
            f"{float(m['win_rate']) * 100:.2f}%",
            f"{float(m['average_r']):.4f}",
            pf_text,
            str(m["verdict"]),
            f"${float(c['cost_per_trade']):,.2f}",
            f"{float(c['cost_in_r']):.4f}",
        ]
        print(" | ".join(v.ljust(14) for v in row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict prop simulation with multiple cost profiles.")
    parser.add_argument(
        "--profile",
        choices=["all", "realistic", "conservative", "extreme"],
        default="all",
        help="Run a single profile or all profiles (default: all).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    input_path = project_root / "data" / "live" / "paper_trades.csv"
    output_csv = project_root / "data" / "live" / "strict_prop_simulation.csv"
    output_json = project_root / "data" / "live" / "strict_prop_summary.json"

    if not input_path.is_file():
        raise FileNotFoundError(f"Missing required input: {input_path}")
    with input_path.open("r", encoding="utf-8", newline="") as f:
        trades = list(csv.DictReader(f))
    if not trades:
        raise ValueError("paper_trades.csv is empty.")

    risk_amount = min(STARTING_BALANCE * RISK_PCT_PER_TRADE, MAX_RISK_PER_TRADE)
    profiles = ["realistic", "conservative", "extreme"] if args.profile == "all" else [args.profile]
    all_summaries: dict[str, dict[str, object]] = {}

    for profile in profiles:
        rows, summary = _run_for_profile(trades, profile, risk_amount)
        all_summaries[profile] = summary
        # Keep default CSV export compatible with dashboard expectation.
        if profile == "extreme":
            _write_rows_csv(output_csv, rows)
        elif args.profile != "all":
            _write_rows_csv(output_csv, rows)

    payload = {
        "starting_balance": STARTING_BALANCE,
        "risk_per_trade_pct": RISK_PCT_PER_TRADE,
        "max_risk_per_trade": MAX_RISK_PER_TRADE,
        "risk_per_trade_fixed": risk_amount,
        "profiles": all_summaries,
    }
    if "extreme" in all_summaries:
        payload["metrics"] = all_summaries["extreme"]["metrics"]
        payload["cost_model"] = all_summaries["extreme"]["cost_model"]
        payload["verdict"] = all_summaries["extreme"]["metrics"]["verdict"]
    elif profiles:
        only = all_summaries[profiles[0]]
        payload["metrics"] = only["metrics"]
        payload["cost_model"] = only["cost_model"]
        payload["verdict"] = only["metrics"]["verdict"]

    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.profile == "all":
        printable = all_summaries
    else:
        fallback = all_summaries[profiles[0]]
        printable = {
            "realistic": all_summaries.get("realistic", fallback),
            "conservative": all_summaries.get("conservative", fallback),
            "extreme": all_summaries.get("extreme", fallback),
        }
    _print_side_by_side(printable)
    print(f"csv: {output_csv}")
    print(f"json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
