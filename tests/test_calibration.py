"""Unit tests for the SPY calibration procedure (Appendix A, page 29)."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    calibrate_phi_midpoint,
    daily_log_return_std,
    explained_spread_pct,
    implied_contracts,
    implied_n,
    implied_phi,
    model_implied_option_spread,
    risk_neutral_probability_one_day,
    up_down_prices,
)


def test_daily_log_return_std_positive_for_nonconstant_series():
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 60)))
    sigma = daily_log_return_std(closes)
    assert sigma > 0.0


def test_up_down_prices_bracket_s0():
    su, sd = up_down_prices(100.0, 0.01)
    assert sd < 100.0 < su


def test_risk_neutral_probability_one_day_between_zero_and_one():
    su, sd = up_down_prices(100.0, 0.01)
    q = risk_neutral_probability_one_day(100.0, su, sd)
    assert 0.0 < q < 1.0


def test_calibrate_phi_positive_and_increasing_in_spread():
    phi_small = calibrate_phi_midpoint(0.01, 1000.0, 1000.0)
    phi_large = calibrate_phi_midpoint(0.05, 1000.0, 1000.0)
    assert phi_small > 0.0
    assert phi_large > phi_small


def test_implied_contracts_scales_with_multiplier():
    n1 = implied_contracts(10.0, 20.0, multiplier=100.0)
    n2 = implied_contracts(10.0, 20.0, multiplier=1.0)
    assert n1 == pytest.approx(100.0 * n2)


def test_model_implied_spread_nonnegative_and_monotone_in_n():
    s0, sigma_d = 500.0, 0.01
    su, sd = up_down_prices(s0, sigma_d)
    phi = 0.02
    small = model_implied_option_spread(10.0, s0, s0, su, sd, 2 * phi, phi)
    large = model_implied_option_spread(1000.0, s0, s0, su, sd, 2 * phi, phi)
    assert small >= -1e-9
    assert large >= small


def test_implied_n_recovers_known_n():
    s0, sigma_d = 500.0, 0.01
    su, sd = up_down_prices(s0, sigma_d)
    phi = 0.02
    n_true = 250.0
    observed = model_implied_option_spread(n_true, s0, s0, su, sd, 2 * phi, phi)
    n_hat = implied_n(observed, s0, s0, su, sd, 2 * phi, phi)
    assert n_hat == pytest.approx(n_true, rel=1e-3)


def test_implied_phi_recovers_known_phi():
    s0, sigma_d = 500.0, 0.01
    su, sd = up_down_prices(s0, sigma_d)
    phi_true = 0.03
    n_shares = 300.0
    observed = model_implied_option_spread(n_shares, s0, s0, su, sd, 2 * phi_true, phi_true)
    phi_hat = implied_phi(observed, n_shares, s0, s0, su, sd)
    assert phi_hat == pytest.approx(phi_true, rel=1e-3)


def test_explained_spread_pct_formula():
    assert explained_spread_pct(0.85, 1.0) == pytest.approx(85.0)
