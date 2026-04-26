from tjtb.config.feature_settings import FeatureSettings
from tjtb.features.queue_imbalance import queue_imbalance_l1, queue_imbalance_multilevel


def test_queue_imbalance_l1_basic() -> None:
    assert abs(queue_imbalance_l1(3, 1) - 0.5) < 1e-9
    assert queue_imbalance_l1(0, 0) == 0.0


def test_queue_imbalance_multilevel_uniform_weights() -> None:
    s = FeatureSettings(queue_top_k=2, queue_level_weights=None)
    q = queue_imbalance_multilevel([3, 1], [1, 3], s)
    assert abs(q - queue_imbalance_l1(4, 4)) < 1e-9
