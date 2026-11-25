"""Unit tests for the one-period binomial model (Sec. 2, Proposition 2.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.binomial_one_period import (
    average_execution_price_option,
    average_execution_price_underlying,
    call_delta_min,
    replication_cost,
    risk_neutral_probability,
)
from src.execution_costs import execution_cost_sqrt_linear


def test_risk_neutral_probability_between_zero_and_one():
    q = risk_neutral_probability(100.0, 105.0, 95.0, 0.01)
    assert 0.0 < q < 1.0


def test_call_delta_min_out_of_the_money_is_zero():
    delta = call_delta_min(1.0, 200.0, 105.0, 95.0, 0.1)
    assert delta == 0.0


def test_call_delta_min_deep_in_the_money_equals_n():
    delta = call_delta_min(1.0, 50.0, 105.0, 95.0, 0.1)
    assert delta == pytest.approx(1.0)


def test_call_delta_min_matches_frictionless_delta_when_lambda_zero():
    n, k, su, sd = 2.0, 100.0, 105.0, 95.0
    delta = call_delta_min(n, k, su, sd, lam=1e-9)
    frictionless_delta = n * (su - k) / (su - sd)
    assert delta == pytest.approx(frictionless_delta, rel=1e-5)


def test_replication_cost_reduces_to_frictionless_price_with_no_friction():
    n, k, s0, su, sd, r = 1.0, 100.0, 100.0, 105.0, 95.0, 0.01
    delta = call_delta_min(n, k, su, sd, lam=0.0)
    cost = replication_cost(delta, n, k, s0, su, sd, lam=0.0, r=r,
                             phi_func=lambda x, **kw: 0.0 * np.asarray(x))
    q = risk_neutral_probability(s0, su, sd, r)
    frictionless = (q * max(su - k, 0.0) + (1 - q) * max(sd - k, 0.0)) / (1 + r)
    assert cost == pytest.approx(frictionless, rel=1e-6)


def test_execution_costs_increase_spread_with_lambda_and_phi():
    spreads = []
    for lam, phi in [(0.1, 0.05), (0.3, 0.15)]:
        buy = average_execution_price_underlying(0.2, 100.0, execution_cost_sqrt_linear, lam=lam, phi=phi)
        sell = average_execution_price_underlying(-0.2, 100.0, execution_cost_sqrt_linear, lam=lam, phi=phi)
        spreads.append(buy - sell)
    assert spreads[1] > spreads[0], "larger (lambda, phi) must widen the underlying spread"


def test_option_market_inherits_concavity_sign():
    """Average option execution price must move toward the frictionless price
    as |N| shrinks (execution cost per unit shrinks), i.e. concave cost."""
    from functools import partial
    lam, phi, k, s0, su, sd, r = 0.2, 0.1, 100.0, 100.0, 105.0, 95.0, 0.01
    phi_func = partial(execution_cost_sqrt_linear, lam=lam)
    small = average_execution_price_option(0.05, k, s0, su, sd, lam, r, phi_func, phi=phi)
    large = average_execution_price_option(1.0, k, s0, su, sd, lam, r, phi_func, phi=phi)
    assert small < large, "buying more contracts must raise the average execution price"


def test_buy_sell_spread_is_nonnegative():
    from functools import partial
    lam, phi, k, s0, su, sd, r = 0.2, 0.1, 100.0, 100.0, 105.0, 95.0, 0.01
    phi_func = partial(execution_cost_sqrt_linear, lam=lam)
    buy = average_execution_price_option(0.2, k, s0, su, sd, lam, r, phi_func, phi=phi)
    sell = average_execution_price_option(-0.2, k, s0, su, sd, lam, r, phi_func, phi=phi)
    assert buy >= sell
