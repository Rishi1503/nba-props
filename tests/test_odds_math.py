import pytest

from nbaprops.pricing import odds_math as om


@pytest.mark.parametrize(
    "american, decimal",
    [(-110, 1.0 + 100 / 110), (-250, 1.4), (150, 2.5), (100, 2.0), (-100, 2.0), (284, 3.84)],
)
def test_to_decimal(american, decimal):
    assert om.to_decimal(american) == pytest.approx(decimal)


@pytest.mark.parametrize("american", [-250, -110, 100, 150, 450])
def test_round_trip(american):
    assert om.to_american(om.to_decimal(american)) == pytest.approx(american)


@pytest.mark.parametrize("bad", [0, 50, -99.9, 99])
def test_invalid_american(bad):
    with pytest.raises(ValueError):
        om.to_decimal(bad)


def test_breakeven_table_from_plan():
    assert om.breakeven_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert om.breakeven_prob(-200) == pytest.approx(2 / 3)
    assert om.breakeven_prob(-250) == pytest.approx(0.7143, abs=1e-4)
    assert om.breakeven_prob(-300) == pytest.approx(0.75)
    assert om.breakeven_prob(-400) == pytest.approx(0.80)


@pytest.mark.parametrize("method", ["multiplicative", "power"])
def test_devig_symmetric_market_is_50_50(method):
    p_o, p_u = om.devig_two_way(-110, -110, method)
    assert p_o == pytest.approx(0.5)
    assert p_u == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["multiplicative", "power"])
def test_devig_sums_to_one_and_favors_favorite(method):
    p_o, p_u = om.devig_two_way(-150, 130, method)
    assert p_o + p_u == pytest.approx(1.0)
    assert p_o > p_u


def test_devig_multiplicative_known_value():
    p_o, _ = om.devig_two_way(-150, 130)
    assert p_o == pytest.approx(0.6 / (0.6 + 100 / 230))


def test_power_puts_more_vig_on_longshot():
    mult, _ = om.devig_two_way(-300, 240, "multiplicative")
    power, _ = om.devig_two_way(-300, 240, "power")
    assert power > mult  # favorite's fair probability is higher under the power method


def test_ev_and_kelly_from_plan_example():
    assert om.ev(0.75, -250) == pytest.approx(0.05)
    assert om.kelly(0.75, -250) == pytest.approx(0.125)
    assert om.kelly(0.75, -250, fraction=0.25) == pytest.approx(0.03125)


def test_zero_ev_and_kelly_at_fair_price():
    assert om.ev(0.5, 100) == pytest.approx(0.0)
    assert om.kelly(0.5, 100) == 0.0


def test_kelly_never_negative():
    assert om.kelly(0.40, -110) == 0.0


def test_parlay_of_four_minus_250_legs():
    d = om.parlay_decimal([om.to_decimal(-250)] * 4)
    assert d == pytest.approx(3.8416)
    assert om.to_american(d) == pytest.approx(284.16)
    p = om.independent_joint_prob([0.75] * 4)
    assert om.ev_decimal(p, d) == pytest.approx(0.2155, abs=1e-4)


def test_boost_from_plan():
    # +300 parlay with a 50% profit boost pays +450
    assert om.to_american(om.boosted_decimal(om.to_decimal(300), 0.5)) == pytest.approx(450)
    # Plan example: 4-leg at +284 (75% legs) with 30% boost -> about +49% EV
    d = om.boosted_decimal(3.8416, 0.30)
    assert om.ev_decimal(0.75**4, d) == pytest.approx(0.486, abs=1e-3)


def test_blend_and_shrink():
    assert om.blend(0.70, 0.60, 0.3) == pytest.approx(0.63)
    assert om.blend(0.70, None, 0.3) == 0.70
    assert om.shrink_toward(0.70, 0.60, 0.5) == pytest.approx(0.65)


def test_clv_sign():
    # Took +110 (implied 0.476); fair close 0.52 -> positive CLV
    assert om.clv(0.52, om.to_decimal(110)) == pytest.approx(0.52 - 1 / 2.1)
    assert om.clv(0.45, om.to_decimal(110)) < 0
