"""The quote tool must match knowledge/pricing.md: flat monthly price up to a seat limit."""

from tools import calculate_custom_quote


def test_growth_is_a_flat_price_not_per_seat():
    q = calculate_custom_quote("growth", 10)
    assert q["ok"] and q["tier"] == "growth"
    assert q["monthly_total_usd"] == 149.0
    assert q["annual_total_usd"] == round(149 * 12 * 0.85, 2)


def test_too_many_seats_moves_to_the_plan_that_fits():
    q = calculate_custom_quote("starter", 8)
    assert q["tier"] == "growth" and q["requested_tier"] == "starter"
    assert q["monthly_total_usd"] == 149.0


def test_enterprise_has_no_invented_price():
    for tier, seats in [("enterprise", 5), ("growth", 40)]:
        q = calculate_custom_quote(tier, seats)
        assert q["tier"] == "enterprise" and q["monthly_total_usd"] is None


def test_bad_input_is_rejected():
    assert not calculate_custom_quote("platinum", 3)["ok"]
    assert not calculate_custom_quote("starter", 0)["ok"]
