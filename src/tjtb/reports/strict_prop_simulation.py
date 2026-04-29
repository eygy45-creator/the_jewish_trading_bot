"""Strict prop simulation from historical paper trades only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


STARTING_BALANCE = 5000.0
RISK_PCT_PER_TRADE = 0.0025
MAX_RISK_PER_TRADE = 12.5

# Conservative cost assumptions (intentionally harsh).
COMMISSION_BPS_PER_SIDE = 10.0
SLIPPAGE_BPS_PER_SIDE = 8.0
SPREAD_BPS_ROUND_TRIP = 8.0
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


def main() -> int:
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
    inferred_notional = risk_amount / ASSUMED_STOP_DISTANCE_PCT
    commission_cost = inferred_notional * ((2.0 * COMMISSION_BPS_PER_SIDE) / 10000.0)
    slippage_cost = inferred_notional * ((2.0 * SLIPPAGE_BPS_PER_SIDE) / 10000.0)
    spread_cost = inferred_notional * (SPREAD_BPS_ROUND_TRIP / 10000.0)
    total_cost_per_trade = commission_cost + slippage_cost + spread_cost

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
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

    total_trades = len(rows)
    wins = sum(1 for v in net_pnls if v > 0.0)
    win_rate = wins / total_trades
    total_fees = total_cost_per_trade * total_trades
    average_r = sum(net_rs) / total_trades
    profit_factor = _profit_factor(net_pnls)
    final_balance = float(rows[-1]["balance_after_trade"])
    verdict = "PASS" if final_balance > STARTING_BALANCE and max_drawdown <= STARTING_BALANCE * 0.10 else "FAIL"

    summary = {
        "starting_balance": STARTING_BALANCE,
        "risk_per_trade_pct": RISK_PCT_PER_TRADE,
        "max_risk_per_trade": MAX_RISK_PER_TRADE,
        "risk_per_trade_fixed": risk_amount,
        "cost_model": {
            "commission_bps_per_side": COMMISSION_BPS_PER_SIDE,
            "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            "spread_bps_round_trip": SPREAD_BPS_ROUND_TRIP,
            "commission_per_trade": commission_cost,
            "slippage_per_trade": slippage_cost,
            "spread_per_trade": spread_cost,
            "total_execution_cost_per_trade": total_cost_per_trade,
        },
        "metrics": {
            "final_balance": final_balance,
            "total_fees": total_fees,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "average_r": average_r,
            "profit_factor": None if math.isinf(profit_factor) else profit_factor,
            "profit_factor_is_infinite": math.isinf(profit_factor),
            "total_trades": total_trades,
        },
        "verdict": verdict,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=== Strict Prop Simulation ===")
    print(f"final balance: ${final_balance:,.2f}")
    print(f"total fees: ${total_fees:,.2f}")
    print(f"max drawdown: ${max_drawdown:,.2f}")
    print(f"win rate: {win_rate * 100:.2f}%")
    print(f"average R: {average_r:.4f}")
    print(f"profit factor: {'inf' if math.isinf(profit_factor) else f'{profit_factor:.4f}'}")
    print(f"pass/fail verdict: {verdict}")
    print(f"csv: {output_csv}")
    print(f"json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
