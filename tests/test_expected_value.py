from tjtb.config.instrument_specs import MNQ_DEFAULT_SPEC
from tjtb.signals.expected_value import ExpectedValueInputs, expected_value_net


def test_expected_value_positive_when_edge_large() -> None:
    p = ExpectedValueInputs(
        probability_success=0.6,
        expected_favorable_ticks=10,
        expected_adverse_ticks=5,
        expected_slippage_ticks=1,
        commission_currency=0.62,
    )
    ev = expected_value_net(p, MNQ_DEFAULT_SPEC)
    assert ev > 0
