from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "diagnostics" / "bybit_feature_parity.py"
    spec = importlib.util.spec_from_file_location("bybit_feature_parity", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_book_rows_handles_utf8_bom_and_bybit_columns(tmp_path):
    mod = _load_module()
    path = tmp_path / "bybit_book_state.csv"
    # Include UTF-8 BOM in header to ensure ts column still parses.
    content = (
        "\ufeffts,best_bid,best_ask,best_bid_size,best_ask_size,spread,mid,microprice,tob_imbalance,signed_pressure,l2_event_type,sequence_num\n"
        "2026-04-30T19:17:22.156000Z,76399.3,76399.4,3.692,0.674,0.1,76399.35,76399.3845625,0.69125057,5.907,snapshot,1\n"
    )
    path.write_text(content, encoding="utf-8")
    rows = mod._read_book_rows(path)
    assert len(rows) == 1
    assert rows[0]["bb"] == 76399.3
    assert rows[0]["signed_pressure"] == 5.907

