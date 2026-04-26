from pathlib import Path

from tjtb.backtest.engine import run_replay_count
from tjtb.data.replay import load_market_events_csv


def test_sample_csv_replays() -> None:
    root = Path(__file__).resolve().parents[1]
    csv = root / "data" / "sample_events.csv"
    stats = run_replay_count(load_market_events_csv(csv))
    assert stats["books"] >= 1
    assert stats["trades"] >= 1
