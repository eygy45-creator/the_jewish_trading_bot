from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tjtb.runtime_paths import HEARTBEAT_PATH, OPPORTUNITIES_PATH, PAPER_TRADES_PATH, PROJECT_ROOT

app = FastAPI(title="TJTB Read-Only API", version="1.0.0")
bearer = HTTPBearer(auto_error=False)

STRICT_SUMMARY_PATH = PROJECT_ROOT / "data" / "live" / "strict_prop_summary.json"
STRICT_TRADES_PATH = PROJECT_ROOT / "data" / "live" / "strict_prop_simulation.csv"


def _require_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    expected = os.environ.get("API_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API token is not configured",
        )
    if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )


def _read_csv(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit > 0:
        rows = rows[-limit:]
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _heartbeat_status() -> tuple[str, str | None]:
    if not HEARTBEAT_PATH.is_file():
        return "stopped", None
    try:
        lines = HEARTBEAT_PATH.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not lines:
            return "stopped", None
        ts = datetime.fromisoformat(lines[0].replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_sec = (datetime.now(tz=timezone.utc) - ts).total_seconds()
        return ("running" if age_sec <= 120 else "stale"), ts.isoformat()
    except (OSError, ValueError):
        return "stopped", None


@app.get("/api/status", dependencies=[Depends(_require_token)])
def get_status() -> dict[str, Any]:
    bot_status, latest_update = _heartbeat_status()
    trades = _read_csv(PAPER_TRADES_PATH, limit=1_000_000)
    total_trades = len(trades)
    summary = _read_json(STRICT_SUMMARY_PATH)
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    latest_balance = metrics.get("final_balance") if isinstance(metrics, dict) else None
    if latest_balance is None and trades:
        latest_balance = trades[-1].get("balance_after_trade")
    return {
        "bot_status": bot_status,
        "latest_update_timestamp": latest_update,
        "total_trades": total_trades,
        "latest_balance": latest_balance,
    }


@app.get("/api/paper-trades", dependencies=[Depends(_require_token)])
def get_paper_trades(limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    rows = _read_csv(PAPER_TRADES_PATH, limit=limit)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/opportunities", dependencies=[Depends(_require_token)])
def get_opportunities(limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    rows = _read_csv(OPPORTUNITIES_PATH, limit=limit)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/strict-prop-summary", dependencies=[Depends(_require_token)])
def get_strict_prop_summary() -> dict[str, Any]:
    return _read_json(STRICT_SUMMARY_PATH)


@app.get("/api/strict-prop-trades", dependencies=[Depends(_require_token)])
def get_strict_prop_trades(limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    rows = _read_csv(STRICT_TRADES_PATH, limit=limit)
    return {"rows": rows, "count": len(rows)}
