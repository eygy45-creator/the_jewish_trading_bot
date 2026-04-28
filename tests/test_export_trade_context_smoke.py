from __future__ import annotations

import json
import importlib
from pathlib import Path

import pandas as pd
etc = importlib.import_module("tjtb.reports.export_trade_context")


def test_export_trade_context_includes_bid_ask_columns(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    raw_dir = tmp_path / "raw"
    live_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    opp_path = live_dir / "opportunities.csv"
    pd.DataFrame(
        [
            {
                "ts": "2026-04-28T10:00:02Z",
                "anomaly_percentile": 0.995,
                "anomaly_score": 13.4,
                "direction": "bearish",
                "regime": "chaotic",
                "action": "blocked",
                "reason": "chaotic_regime",
            }
        ]
    ).to_csv(opp_path, index=False)

    ndjson_path = raw_dir / "coinbase_btcusd_2026-04-28_10-00-00.ndjson"
    msgs = [
        {
            "channel": "l2_data",
            "events": [
                {
                    "type": "snapshot",
                    "updates": [
                        {"event_time": "2026-04-28T10:00:00Z", "side": "bid", "price_level": "64000", "new_quantity": "1.2"},
                        {"event_time": "2026-04-28T10:00:00Z", "side": "offer", "price_level": "64001", "new_quantity": "0.8"},
                    ],
                }
            ],
        },
        {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {"time": "2026-04-28T10:00:02Z", "side": "buy", "size": "0.15", "price": "64000.5"},
                    ]
                }
            ],
        },
    ]
    with ndjson_path.open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(etc, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(etc, "OPPORTUNITIES_PATH", opp_path)
    monkeypatch.setattr(etc, "LIVE_DATA_DIR", live_dir)

    out_df, out_path, _msg = etc.export_trade_context(
        entry_ts="2026-04-28T10:00:01Z",
        exit_ts="2026-04-28T10:00:03Z",
    )

    assert out_path.is_file()
    assert "bid" in out_df.columns
    assert "ask" in out_df.columns
    assert "bid_size" in out_df.columns
    assert "ask_size" in out_df.columns
    assert "raw_json" in out_df.columns
    assert out_df["bid"].notna().any() or out_df["ask"].notna().any()


def test_export_trade_context_warns_when_bidask_keys_missing(tmp_path, monkeypatch):
    live_dir = tmp_path / "live"
    raw_dir = tmp_path / "raw"
    live_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    opp_path = live_dir / "opportunities.csv"
    pd.DataFrame(
        [
            {
                "ts": "2026-04-28T10:00:02Z",
                "anomaly_percentile": 0.995,
                "anomaly_score": 13.4,
                "direction": "bearish",
                "regime": "chaotic",
                "action": "blocked",
                "reason": "chaotic_regime",
            }
        ]
    ).to_csv(opp_path, index=False)

    # Intentionally omit l2_data updates so raw keys do not include bid/ask.
    ndjson_path = raw_dir / "coinbase_btcusd_2026-04-28_10-00-00.ndjson"
    msgs = [
        {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {"time": "2026-04-28T10:00:02Z", "side": "buy", "size": "0.15", "price": "64000.5"},
                    ]
                }
            ],
        }
    ]
    with ndjson_path.open("w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(etc, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(etc, "OPPORTUNITIES_PATH", opp_path)
    monkeypatch.setattr(etc, "LIVE_DATA_DIR", live_dir)

    out_df, out_path, msg = etc.export_trade_context(
        entry_ts="2026-04-28T10:00:01Z",
        exit_ts="2026-04-28T10:00:03Z",
    )

    assert out_path.is_file()
    assert "bid" in out_df.columns
    assert "ask" in out_df.columns
    assert out_df["bid"].isna().all()
    assert out_df["ask"].isna().all()
    assert "bid_size" in out_df.columns
    assert "ask_size" in out_df.columns
    assert out_df["bid_size"].isna().all()
    assert out_df["ask_size"].isna().all()
    assert "raw_keys_warning=" in msg
