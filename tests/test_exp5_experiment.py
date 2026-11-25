"""Unit tests for experiment 5 (continuous-time call replication, Sec. 4-5.2)."""
from __future__ import annotations

import numpy as np

from exp.exp5 import (
    bs_call_price_delta,
    figure6_terminal_condition,
    figure7_replication_check,
    figure8_price_position_vs_s0,
    figure9_moving_target,
)


def test_terminal_condition_matches_call_payoff_far_in_the_money():
    fig6 = figure6_terminal_condition()
    assert np.allclose(fig6["v_tilde"][-5:], fig6["raw_payoff"][-5:] + (
        0.5 * (2 * 0.06 + 0.1) * 1.0**2
    ), atol=1e-6)


def test_terminal_delta_between_zero_and_n():
    fig6 = figure6_terminal_condition()
    assert np.all(fig6["delta_tilde"] >= -1e-9)
    assert np.all(fig6["delta_tilde"] <= 1.0 + 1e-9)


def test_replication_error_shrinks_with_finer_grid():
    rows = figure7_replication_check([(20, 50), (120, 400)], n_samples=15, seed=0)
    assert rows[-1]["mean_abs_error"] < rows[0]["mean_abs_error"]


def test_price_and_position_per_contract_exceed_black_scholes():
    fig8 = figure8_price_position_vs_s0(n_space=80, n_time=120)
    bs_price = np.array(fig8["frictionless"]["price_per_contract"])
    for key, series in fig8.items():
        if key == "frictionless":
            continue
        price = np.array(series["price_per_contract"])
        assert np.mean(price >= bs_price - 1e-3) > 0.9


def test_black_scholes_delta_between_zero_and_one():
    s = np.linspace(50.0, 150.0, 20)
    _, delta = bs_call_price_delta(s, k=100.0, r=0.05, sigma=0.1, tau=1.0)
    assert np.all(delta >= 0.0) and np.all(delta <= 1.0)


def test_moving_target_search_runs_and_returns_nonnegative_count():
    out = figure9_moving_target(n_space=80, n_time=120, n_paths=40, n_steps=30, seed=2)
    assert out["n_paths"] == 40
    assert out["moving_target_count"] >= 0
