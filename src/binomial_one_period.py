"""One-period binomial model with linear price impact and execution costs.

Implements the call-option replication of Proposition 2.4 and the
replication-cost representation of Proposition 2.3 / Equation (2.7).
Custom implementation (paper Sec. 2, no standard library covers a
replication problem with price impact and execution costs).
"""
from __future__ import annotations

import numpy as np

from src.execution_costs import linear_impact, round_trip_cost


def risk_neutral_probability(s0: float, st_up: float, st_down: float, r: float) -> float:
    """q = (S_0(1+r) - S_T(d)) / (S_T(u) - S_T(d)), Sec. 2.1."""
    return (s0 * (1.0 + r) - st_down) / (st_up - st_down)


def call_delta_min(n: float, k: float, st_up: float, st_down: float, lam: float):
    """Minimal-in-absolute-value solution Delta*_min to the fixed point equation
    (2.5) for the call payoff V(x) = N(x-K)^+ under linear impact I(x)=lambda*x.

    Proposition 2.4:
        Delta*_min = 0                                   if K >= S_T(u)
        Delta*_min = N(S_T(u)-K) / (S_T(u)-S_T(d)-lam N)  if S_T(d)+lam N <= K < S_T(u)
        Delta*_min = N                                   if K < min(S_T(d)+lam N, S_T(u))
    """
    n_in = np.asarray(n, dtype=float)
    scalar_input = n_in.ndim == 0 and np.asarray(k).ndim == 0 and np.asarray(st_up).ndim == 0 and np.asarray(st_down).ndim == 0
    n, k_b, st_up_b, st_down_b = np.broadcast_arrays(
        n_in, np.asarray(k, dtype=float), np.asarray(st_up, dtype=float), np.asarray(st_down, dtype=float)
    )
    n = np.array(n, dtype=float)
    k_b = np.array(k_b, dtype=float)
    st_up_b = np.array(st_up_b, dtype=float)
    st_down_b = np.array(st_down_b, dtype=float)
    delta = np.empty_like(n)
    out_of_money = k_b >= st_up_b
    delta[out_of_money] = 0.0
    both_itm = (~out_of_money) & (k_b < np.minimum(st_down_b + lam * n, st_up_b))
    delta[both_itm] = n[both_itm]
    mixed = (~out_of_money) & (~both_itm)
    denom = st_up_b[mixed] - st_down_b[mixed] - lam * n[mixed]
    assert np.all(denom != 0.0), "degenerate denominator in Proposition 2.4"
    delta[mixed] = n[mixed] * (st_up_b[mixed] - k_b[mixed]) / denom
    return float(delta.reshape(-1)[0]) if scalar_input else delta


def call_payoff(x: np.ndarray | float, n: float, k: float) -> np.ndarray | float:
    """V(x) = N(x-K)^+."""
    return n * np.maximum(np.asarray(x, dtype=float) - k, 0.0)


def replication_cost(
    delta: np.ndarray | float,
    n: float,
    k: float,
    s0: float,
    st_up: float,
    st_down: float,
    lam: float,
    r: float,
    phi_func,
    **phi_kwargs,
) -> np.ndarray | float:
    """Replication cost X_0* from Equation (2.7):

        X_0* = 1/(1+r) * ( E^Q[V(P_T^Delta*)] + Delta* C(Delta*) )

    where P_T^Delta*(omega) = S_T(omega) + lambda*Delta*, q is the
    risk-neutral probability of Sec. 2.1, and C is the round-trip cost
    function (Sec. 2.3, preceding Eq. 2.7):
        C(x) = phi(x)(1+r) - phi(-x) - I(x).
    """
    delta = np.asarray(delta, dtype=float)
    q = risk_neutral_probability(s0, st_up, st_down, r)
    pt_up = st_up + linear_impact(delta, lam)
    pt_down = st_down + linear_impact(delta, lam)
    expected_payoff = q * call_payoff(pt_up, n, k) + (1.0 - q) * call_payoff(pt_down, n, k)
    c_delta = round_trip_cost(delta, lam, r, phi_func, **phi_kwargs)
    return (expected_payoff + delta * c_delta) / (1.0 + r)


def average_execution_price_underlying(n: np.ndarray | float, s0: float, phi_func, **phi_kwargs):
    """Average per-share execution price of the underlying, S_0 + phi(N) (Fig. 2
    caption, page 7: "the execution price is given by 100 + phi*sign(N)*sqrt(|N|)").

    phi(N) is itself the *average* per-share deviation from S_0 (not a total
    cost to be divided by N): it is the amount per share paid above S_0 for a
    buy order (N>0) or received below S_0 for a sell order (N<0), since phi is
    an odd function of N. The finite-N buy/sell spread 2*phi(N) is therefore
    nonzero and increasing in lambda and phi, matching Figure 2.
    """
    n = np.asarray(n, dtype=float)
    return s0 + phi_func(n, **phi_kwargs)


def average_execution_price_option(
    n: np.ndarray | float,
    k: float,
    s0: float,
    st_up: float,
    st_down: float,
    lam: float,
    r: float,
    phi_func,
    **phi_kwargs,
):
    """Average per-contract option execution price X_0*(N)/N for N != 0, using
    Proposition 2.4 for Delta*_min and Equation (2.7) for X_0*."""
    n = np.asarray(n, dtype=float)
    delta = call_delta_min(n, k, st_up, st_down, lam)
    x0 = replication_cost(delta, n, k, s0, st_up, st_down, lam, r, phi_func, **phi_kwargs)
    out = np.where(n != 0.0, x0 / np.where(n != 0, n, 1.0), np.nan)
    return out

