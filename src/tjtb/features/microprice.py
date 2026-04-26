"""Microprice and weighted mid features."""

from __future__ import annotations

from tjtb.config.feature_settings import FeatureSettings


def microprice(best_bid_px: float, best_bid_sz: float, best_ask_px: float, best_ask_sz: float) -> float:
    """Classic microprice using L1 sizes."""
    num = best_bid_px * best_ask_sz + best_ask_px * best_bid_sz
    den = best_bid_sz + best_ask_sz
    if den <= 0:
        return (best_bid_px + best_ask_px) / 2.0
    return num / den


def weighted_mid(
    best_bid_px: float,
    best_bid_sz: float,
    best_ask_px: float,
    best_ask_sz: float,
    settings: FeatureSettings,
) -> float:
    """Alias with config hook for future multi-level mids."""
    _ = settings
    return microprice(best_bid_px, best_bid_sz, best_ask_px, best_ask_sz)
