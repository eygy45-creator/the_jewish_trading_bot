"""Bybit public market-data recorder (linear symbol, no auth)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from tjtb.exchanges.bybit.market_data import build_subscribe_args, get_bybit_symbol, normalize_public_message
from tjtb.runtime_paths import RAW_DATA_DIR, ensure_runtime_dirs

DEFAULT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
MAX_BACKOFF_SEC = 30.0
READ_TIMEOUT_SEC = 30.0

log = logging.getLogger("tjtb.data.bybit_recorder")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def output_path_for_day(base_dir: Path, now: datetime | None = None) -> Path:
    dt = now or datetime.now(tz=timezone.utc)
    return base_dir / f"bybit_{dt.strftime('%Y%m%d')}.ndjson"


def write_ndjson_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def subscribe_public_topics(ws: websockets.WebSocketClientProtocol, symbol: str) -> None:
    req = {"op": "subscribe", "args": build_subscribe_args(symbol)}
    await ws.send(json.dumps(req))


async def run_recorder(ws_url: str, symbol: str) -> None:
    backoff = 1.0
    seen = 0
    while True:
        try:
            log.info("connecting ws_url=%s", ws_url)
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
                max_size=20 * 1024 * 1024,
            ) as ws:
                await subscribe_public_topics(ws, symbol)
                log.info("subscribed symbol=%s args=%s", symbol, build_subscribe_args(symbol))
                backoff = 1.0
                while True:
                    raw_text = await asyncio.wait_for(ws.recv(), timeout=READ_TIMEOUT_SEC)
                    local_ts = utc_now_iso()
                    out_path = output_path_for_day(RAW_DATA_DIR)
                    try:
                        payload = json.loads(raw_text)
                    except json.JSONDecodeError:
                        write_ndjson_line(
                            out_path,
                            {
                                "source": "bybit",
                                "local_ts": local_ts,
                                "parse_error": "json_decode",
                                "raw_text": raw_text,
                            },
                        )
                        continue
                    if not isinstance(payload, dict):
                        continue
                    rec = normalize_public_message(payload, local_ts, symbol=symbol)
                    if rec is None:
                        continue
                    write_ndjson_line(out_path, rec)
                    seen += 1
                    if seen % 250 == 0:
                        log.info("recorded_messages=%s file=%s", seen, out_path)
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            log.info("stopped_by_user")
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("ws_error=%s reconnect_in_sec=%.1f", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(MAX_BACKOFF_SEC, backoff * 2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Bybit public orderbook+trades to raw NDJSON")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="Bybit public WS URL (linear category)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--symbol", default=get_bybit_symbol(), help="Bybit symbol, e.g. BTCUSDT or ETHUSDT")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    ensure_runtime_dirs()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbol = str(args.symbol).strip().upper()
    log.info("output_dir=%s symbol=%s", RAW_DATA_DIR, symbol)
    asyncio.run(run_recorder(args.ws_url, symbol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

