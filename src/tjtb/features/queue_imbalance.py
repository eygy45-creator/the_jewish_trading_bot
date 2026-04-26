"""Queue imbalance: L1 and multi-level weighted imbalance."""

from __future__ import annotations

import numpy as np

from tjtb.config.feature_settings import FeatureSettings


def queue_imbalance_l1(bid_vol: float, ask_vol: float, eps: float = 1e-9) -> float:
    """QI = (bid - ask) / (bid + ask)."""
    denom = bid_vol + ask_vol
    if denom <= eps:
        return 0.0
    return (bid_vol - ask_vol) / denom


def queue_imbalance_multilevel(
    bid_vols: list[float],
    ask_vols: list[float],
    settings: FeatureSettings,
) -> float:
    """Multi-level imbalance with configurable depth and weights."""
    k = min(settings.queue_top_k, len(bid_vols), len(ask_vols))
    if k == 0:
        return 0.0
    bv = np.array(bid_vols[:k], dtype=float)
    av = np.array(ask_vols[:k], dtype=float)
    if settings.queue_level_weights is None:
        w = np.ones(k, dtype=float)
    else:
        w = np.array(settings.queue_level_weights[:k], dtype=float)
    wb = float(np.dot(w, bv))
    wa = float(np.dot(w, av))
    return queue_imbalance_l1(wb, wa, eps=settings.microprice_eps)
