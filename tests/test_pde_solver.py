"""Unit tests for the finite-difference PDE solver (Sec. 4-5.2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.pde_solver import (
    call_terminal_delta_tilde,
    call_terminal_payoff_tilde,
    option_price_and_hedge_at_t0,
    solve_call_pde,
)


def test_terminal_payoff_zero_far_otm_and_linear_far_itm():
    k, n, lam, phi = 100.0, 1.0, 0.1, 0.06
    x = np.array([k - 2.1 * phi * n, k - 0.5 * lam * n * 0 + k + 50.0])
    payoff = call_terminal_payoff_tilde(np.array([k - 50.0]), k, n, lam, phi)
    assert payoff[0] == pytest.approx(0.0)
    deep_itm = call_terminal_payoff_tilde(np.array([k + 50.0]), k, n, lam, phi)
    expected = n * (k + 50.0 - k) + 0.5 * (2 * phi + lam) * n ** 2
    assert deep_itm[0] == pytest.approx(expected)


def test_terminal_delta_bounded_between_zero_and_n():
    k, n, lam, phi = 100.0, 1.0, 0.1, 0.06
    x = np.linspace(k - 3 * phi * n, k + 2.0, 50)
    delta = call_terminal_delta_tilde(x, k, n, lam, phi)
    assert np.all(delta >= -1e-9)
    assert np.all(delta <= n + 1e-9)


def test_pde_solution_monotone_increasing_in_price_near_t0():
    """A call's midpoint liquidation value should be increasing in the
    underlying price at a fixed time (monotone payoff)."""
    sol = solve_call_pde(k=100.0, n=1.0, lam=0.1, phi=0.06, r=0.05, sigma=0.10,
                          t_mat=0.25, x_min=40.0, x_max=200.0, n_space=90, n_time=60)
    u_t0 = sol["u"][-1]
    assert np.all(np.diff(u_t0) >= -1e-6)


def test_price_and_hedge_at_t0_positive_for_call():
    sol = solve_call_pde(k=100.0, n=1.0, lam=0.1, phi=0.06, r=0.05, sigma=0.10,
                          t_mat=0.25, x_min=40.0, x_max=200.0, n_space=90, n_time=60)
    price, hedge = option_price_and_hedge_at_t0(sol, s0=100.0, two_phi_minus_lam=2 * 0.06 - 0.1)
    assert price > 0.0
    assert 0.0 <= hedge <= 1.0 + 1e-6
