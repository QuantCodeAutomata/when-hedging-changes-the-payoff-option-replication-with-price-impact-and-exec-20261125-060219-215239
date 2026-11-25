"""Calibration of the one-period binomial model to SPY option market data
(Sec. 2.4.2, Appendix A, page 29).

Custom implementation (paper Appendix A): converts median quoted spreads and
median quoted sizes of the underlying and the option into the price-impact
parameter phi, the implied number of contracts N, and the model-implied
option bid-ask spread, following the calibration procedure described for
Table 1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.binomial_one_period import average_execution_price_option
from src.execution_costs import execution_cost_linear


def daily_log_return_std(closes: np.ndarray) -> float:
    """Standard deviation sigma_d of daily log returns over the lookback
    window preceding the trading session (Appendix A, page 29)."""
    closes = np.asarray(closes, dtype=float)
    assert closes.size >= 2, "need at least two closes to form one log return"
    log_returns = np.diff(np.log(closes))
    return float(np.std(log_returns, ddof=1))


def up_down_prices(s0: float, sigma_d: float) -> tuple[float, float]:
    """Up/down terminal prices built from the preceding-60-day log-return
    standard deviation: S_T(u) = S_0 e^{sigma_d}, S_T(d) = S_0 e^{-sigma_d}
    (Appendix A, page 29)."""
    return s0 * np.exp(sigma_d), s0 * np.exp(-sigma_d)


def risk_neutral_probability_one_day(s0: float, st_up: float, st_down: float) -> float:
    """q = (S_0 - S_T(d)) / (S_T(u) - S_T(d)) with r = 0 for one day to
    expiry (Appendix A, page 29)."""
    return (s0 - st_down) / (st_up - st_down)


def calibrate_phi_midpoint(half_spread: float, vol_ask: float, vol_bid: float) -> float:
    """Calibrate the execution-cost parameter phi from the underlying's
    S^S, defined as half the median quoted spread (S^S = 0.5 * median(ask -
    bid)), and the median quoted volumes at the best ask/bid:
    phi = (S^S / 2) * (1/V^{S,a} + 1/V^{S,b}), using the midpoint regime
    lambda = 2*phi (Appendix A, page 29). ``half_spread`` is S^S itself
    (already halved by the caller); this function applies the formula's own
    additional factor of 1/2 on top of it."""
    assert vol_ask > 0.0 and vol_bid > 0.0, "quoted volumes must be positive"
    return 0.5 * half_spread * (1.0 / vol_ask + 1.0 / vol_bid)


def implied_contracts(vol_ask_opt: float, vol_bid_opt: float, multiplier: float = 100.0) -> float:
    """Number of underlying shares N implied by the median option quoted
    volumes at the best ask/bid: N = multiplier * (1/2)(V^{O,a}+V^{O,b})
    (Appendix A, page 29; multiplier=100 shares per standard equity option)."""
    return multiplier * 0.5 * (vol_ask_opt + vol_bid_opt)


def model_implied_option_spread(
    n_shares: float, k: float, s0: float, st_up: float, st_down: float, lam: float, phi: float
) -> float:
    """Model-implied option bid-ask spread for a position of n_shares
    underlying shares: the difference between the average per-share
    replication cost when buying N shares worth of calls (Delta*_min solves
    the fixed point for +N) and when selling the same amount (-N), using
    Proposition 2.4 / Equation (2.7) with r=0 and linear execution costs in
    the midpoint regime lambda=2*phi."""
    ask = average_execution_price_option(
        n_shares, k, s0, st_up, st_down, lam, 0.0, execution_cost_linear, phi=phi
    )
    bid = average_execution_price_option(
        -n_shares, k, s0, st_up, st_down, lam, 0.0, execution_cost_linear, phi=phi
    )
    return float(ask - bid)


def implied_n(
    observed_spread: float, k: float, s0: float, st_up: float, st_down: float, lam: float, phi: float,
    n_bracket: tuple[float, float] = (1e-6, 1.0e7),
) -> float:
    """Solve for the number of shares N that reproduces the observed option
    spread, holding phi (and lambda=2*phi) fixed at their calibrated values.
    The model spread is monotone increasing in N, so brentq is used for the
    root of spread(N) - observed_spread (Appendix A, page 29)."""
    def objective(n_shares: float) -> float:
        return model_implied_option_spread(n_shares, k, s0, st_up, st_down, lam, phi) - observed_spread

    lo, hi = n_bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0.0:
        return float("nan")
    return float(brentq(objective, lo, hi, xtol=1e-8, maxiter=200))


def implied_phi(
    observed_spread: float, n_shares: float, k: float, s0: float, st_up: float, st_down: float,
    phi_bracket: tuple[float, float] = (1e-8, 10.0),
) -> float:
    """Solve for the execution-cost parameter phi (with lambda=2*phi) that
    reproduces the observed option spread, holding N fixed at the value
    implied by observed option volumes (Appendix A, page 29)."""
    def objective(phi: float) -> float:
        return model_implied_option_spread(n_shares, k, s0, st_up, st_down, 2.0 * phi, phi) - observed_spread

    lo, hi = phi_bracket
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0.0:
        return float("nan")
    return float(brentq(objective, lo, hi, xtol=1e-10, maxiter=200))


def explained_spread_pct(model_spread: float, observed_spread: float) -> float:
    """Percentage of the observed spread explained by the model,
    100 * model_spread / observed_spread (Table 1, page 8)."""
    assert observed_spread > 0.0, "observed spread must be positive"
    return 100.0 * model_spread / observed_spread
