from tjtb.labels.barriers import BarrierLabelConfig, BarrierOutcome, barrier_hit_label_long_short


def test_long_hits_favorable_first() -> None:
    tick = 0.25
    entry = 100.0
    path = [100.0, 100.25, 100.75]  # +3 ticks from 100 -> 100.75
    cfg = BarrierLabelConfig(favorable_ticks=3, adverse_ticks=2, max_horizon=10)
    yl, ys, ol, os = barrier_hit_label_long_short(entry, path, tick, cfg)
    assert yl == 1 and ol == BarrierOutcome.HIT_FAVORABLE
    assert ys == 0


def test_short_hits_favorable_first() -> None:
    tick = 0.25
    entry = 100.0
    path = [100.0, 99.75, 99.25]  # -3 ticks
    cfg = BarrierLabelConfig(favorable_ticks=3, adverse_ticks=2, max_horizon=10)
    yl, ys, ol, os = barrier_hit_label_long_short(entry, path, tick, cfg)
    assert ys == 1 and os == BarrierOutcome.HIT_FAVORABLE
    assert yl == 0
