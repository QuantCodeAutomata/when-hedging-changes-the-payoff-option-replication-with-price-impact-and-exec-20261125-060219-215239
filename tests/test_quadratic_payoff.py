"""Unit tests for the quadratic-payoff closed-form solution (Sec. 5.1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.quadratic_payoff import frictionless_quadratic_price, solve_quadratic_ansatz


def test_price_exceeds_frictionless_benchmark():
    s0_grid = np.linspace(10.0, 15.0, 11)
    n, alpha, beta, gamma, lam, r, sigma, t_mat = 1.0, 0.02, -2.0, 30.0, 0.1, 0.05, 0.2, 1.0
    out = solve_quadratic_ansatz(s0_grid, n, alpha, beta, gamma, lam, r, sigma, t_mat)
    frictionless = frictionless_quadratic_price(s0_grid, n, alpha, beta, gamma, r, sigma, t_mat)
    assert np.all(out["price_per_contract"] >= frictionless - 1e-6)


def test_scaling_property_midpoint_regime():
    """N=1, lambda=0.1 must give the same per-contract price/position curve
    as N=10, lambda=0.01 in the midpoint regime (paper Sec. 5.1 scaling
    property, Figure 5)."""
    s0_grid = np.linspace(10.0, 15.0, 6)
    alpha, beta, gamma, r, sigma, t_mat = 0.02, -2.0, 30.0, 0.05, 0.2, 1.0
    out_a = solve_quadratic_ansatz(s0_grid, 1.0, alpha, beta, gamma, 0.1, r, sigma, t_mat)
    out_b = solve_quadratic_ansatz(s0_grid, 10.0, alpha, beta, gamma, 0.01, r, sigma, t_mat)
    np.testing.assert_allclose(out_a["price_per_contract"], out_b["price_per_contract"], rtol=1e-6)
    np.testing.assert_allclose(out_a["position_per_contract"], out_b["position_per_contract"], rtol=1e-6)


def test_per_contract_values_increase_with_n():
    """Even after normalizing by N, both price and hedge per contract should
    increase with N (paper Sec. 5.1 discussion of Figure 5)."""
    s0_grid = np.linspace(10.0, 15.0, 6)
    alpha, beta, gamma, r, sigma, t_mat = 0.02, -2.0, 30.0, 0.05, 0.2, 1.0
    out_n1 = solve_quadratic_ansatz(s0_grid, 1.0, alpha, beta, gamma, 0.1, r, sigma, t_mat)
    out_n10 = solve_quadratic_ansatz(s0_grid, 10.0, alpha, beta, gamma, 0.1, r, sigma, t_mat)
    assert np.all(out_n10["price_per_contract"] >= out_n1["price_per_contract"] - 1e-9)
