import asyncio
import json
import os
from datetime import datetime, timezone

import websockets

WS_URL = "wss://advanced-trade-ws.coinbase.com"
PRODUCT = "BTC-USD"
OUTPUT_DIR = "data/raw"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_output_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"coinbase_{PRODUCT.lower().replace('-', '')}_{ts}.ndjson"
    return os.path.join(OUTPUT_DIR, filename)


async def subscribe_channel(ws: websockets.WebSocketClientProtocol, channel: str) -> None:
    msg = {
        "type": "subscribe",
        "product_ids": [PRODUCT],
        "channel": channel,
    }
    await ws.send(json.dumps(msg))


async def connect() -> None:
    message_count = 0

    while True:
        try:
            print("Connecting to Coinbase Advanced Trade...")

            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=10 * 1024 * 1024,  # 10MB
            ) as ws:
                await subscribe_channel(ws, "heartbeats")
                await subscribe_channel(ws, "level2")
                await subscribe_channel(ws, "market_trades")

                print("Subscribed to heartbeats, level2, market_trades.")

                output_file = get_output_file()
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                with open(output_file, "a", encoding="utf-8") as f:
                    while True:
                        raw_msg = await ws.recv()
                        data = json.loads(raw_msg)
                        data["local_ts"] = utc_now_iso()

                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        f.flush()

                        message_count += 1
                        if message_count % 100 == 0:
                            print(f"Recorded {message_count} messages to {output_file}")

        except KeyboardInterrupt:
            print("Recorder stopped by user.")
            break
        except Exception as e:
            print(f"Connection error: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(connect())
