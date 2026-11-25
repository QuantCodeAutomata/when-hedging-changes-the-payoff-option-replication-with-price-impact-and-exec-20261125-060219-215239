"""Closed-form replication of a quadratic payoff with price impact and
execution costs in the midpoint regime 2*phi = lambda (Sec. 5.1).

Custom implementation (paper Sec. 5.1, Eq. preceding and following (5.1));
no option-pricing library covers a quadratic payoff under price impact.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def terminal_coefficients(n: float, alpha: float, beta: float, gamma: float, phi: float, lam: float):
    """Terminal coefficients a(T), b(T), c(T) of the modified payoff
    V~(x) = a(T) x^2 + b(T) x + c(T), for V(x) = N(alpha x^2 + beta x + gamma)
    in the midpoint regime, taking the root of the quadratic terminal
    condition that minimizes terminal inventory (Sec. 5.1)."""
    disc = 1.0 - 16.0 * n * alpha * phi
    assert disc >= 0.0, "closed-form quadratic ansatz requires alpha < 1/(16*phi*N)"
    a_t = (1.0 - 4.0 * n * alpha * lam - np.sqrt(disc)) / (4.0 * (2.0 * n * alpha * lam**2 + 2.0 * phi - lam))
    b_t = (-n * beta - 2.0 * n * beta * lam * a_t) / (
        -1.0 + 2.0 * n * alpha * lam - 2.0 * lam * a_t + 4.0 * n * alpha * lam**2 * a_t + 4.0 * phi * a_t
    )
    c_t = n * gamma + n * beta * lam * b_t + n * alpha * lam**2 * b_t**2 + 0.5 * (2.0 * phi - lam) * b_t**2
    return a_t, b_t, c_t


def solve_quadratic_ansatz(
    s0_grid: np.ndarray,
    n: float,
    alpha: float,
    beta: float,
    gamma: float,
    lam: float,
    r: float,
    sigma: float,
    t_mat: float,
) -> dict:
    """Solve the PDE backward in time with the quadratic ansatz u(t,x) = a(t)x^2+b(t)x+c(t)
    in the midpoint regime 2*phi = lambda, and evaluate at t=0 for a grid of S_0 values.

    Returns per-contract derivative price u(0,S_0)/N and per-contract initial
    position (2*a(0)*S_0 + b(0))/N (Sec. 5.1).
    """
    phi = lam / 2.0  # midpoint regime
    a_t_term, b_t_term, c_t_term = terminal_coefficients(n, alpha, beta, gamma, phi, lam)
    rate = r + sigma**2

    def a_path(tau: float) -> float:
        """a(t) = a(T) rate e^{rate tau} / (rate - 2 a(T)(r lam + sigma^2 (2phi-lam))(e^{rate tau} - 1))."""
        e = np.exp(rate * tau)
        num = a_t_term * rate * e
        den = rate - 2.0 * a_t_term * (r * lam + sigma**2 * (2.0 * phi - lam)) * (e - 1.0)
        assert den != 0.0, "singular Riccati denominator in a(t)"
        return num / den

    # b(tau) solved via integrating factor: b(tau) = b(T) * exp(2 r lam * int_0^tau a(T-s) ds)
    # (from b'(t) + 2 r lam a(t) b(t) = 0, i.e. db/dtau = +2 r lam a(T-tau) b).
    def b_of_tau(tau: float) -> float:
        if tau <= 0:
            return b_t_term
        integral, _ = quad(a_path, 0.0, tau, limit=100)
        return b_t_term * np.exp(2.0 * r * lam * integral)

    def c_of_tau(tau: float) -> float:
        """Solves c'(t) - r c(t) = -(1/2) r lam b(t)^2 (Sec. 5.1 constant-term ODE).

        With tau = T - t and cc(tau) = c(T - tau), this becomes the linear ODE
        dcc/dtau + r cc(tau) = (1/2) r lam bb(tau)^2, whose integrating-factor
        solution is cc(tau) = e^{-r tau} [c(T) + (1/2) r lam int_0^tau e^{rs} bb(s)^2 ds].
        """
        if tau <= 0:
            return c_t_term
        integrand = lambda s: np.exp(r * s) * b_of_tau(s) ** 2
        integral, _ = quad(integrand, 0.0, tau, limit=100)
        return np.exp(-r * tau) * (c_t_term + 0.5 * r * lam * integral)

    tau0 = t_mat
    a0 = a_path(tau0)
    b0 = b_of_tau(tau0)
    c0 = c_of_tau(tau0)

    s0_grid = np.asarray(s0_grid, dtype=float)
    price_total = a0 * s0_grid**2 + b0 * s0_grid + c0
    position_total = 2.0 * a0 * s0_grid + b0

    return {
        "price_per_contract": price_total / n,
        "position_per_contract": position_total / n,
        "a0": a0,
        "b0": b0,
        "c0": c0,
        "a_T": a_t_term,
        "b_T": b_t_term,
        "c_T": c_t_term,
    }


def frictionless_quadratic_price(s0_grid: np.ndarray, n: float, alpha: float, beta: float, gamma: float,
                                  r: float, sigma: float, t_mat: float) -> np.ndarray:
    """No-impact Black-Scholes-type benchmark for V(x)=N(alpha x^2+beta x+gamma):
    obtained as the lambda -> 0 limit of the same quadratic ansatz (phi=lambda=0),
    i.e. discounted risk-neutral expectation of the quadratic payoff under
    geometric Brownian motion with drift r, computed in closed form via the
    moments E[S_T]=S_0 e^{rT}, E[S_T^2]=S_0^2 e^{(2r+sigma^2)T}."""
    s0_grid = np.asarray(s0_grid, dtype=float)
    e_st = s0_grid * np.exp(r * t_mat)
    e_st2 = s0_grid**2 * np.exp((2.0 * r + sigma**2) * t_mat)
    price = n * (alpha * e_st2 + beta * e_st + gamma)
    return np.exp(-r * t_mat) * price / n
