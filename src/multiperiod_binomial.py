"""Multiperiod binomial replication with linear price impact and execution costs.

Implements the backward induction scheme of Theorem 3.2 (Sec. 3.2) on the
recombining fundamental-price tree, and the frictionless benchmark used in
Corollary 3.3. Custom implementation (paper Eq. 3.7-3.9); no standard
option-pricing library implements replication under price impact.
"""
from __future__ import annotations

import math

import numpy as np

from src.binomial_one_period import call_delta_min


def build_tree(s0: float, u: float, d: float, m_steps: int) -> list[np.ndarray]:
    """Recombining binomial tree levels S_{t_0}, ..., S_{t_M}.

    Level m is an array of length m+1 with S_{t_m}(j) = s0 * u^j * d^(m-j)
    for j = 0, ..., m (number of up-moves j).
    """
    levels = []
    for m in range(m_steps + 1):
        j = np.arange(m + 1)
        levels.append(s0 * (u ** j) * (d ** (m - j)))
    return levels


def replicate_call(
    s0: float,
    u: float,
    d: float,
    m_steps: int,
    r_per_step: float,
    lam: float,
    phi: float,
    k: float,
    n: float,
) -> dict:
    """Backward induction of Theorem 3.2 for the call payoff V(x) = N(x-K)^+.

    Returns a dict with: levels (fundamental-price tree), f (replicating
    strategy Delta*_{t_m}(s) at each node, arrays indexed like levels[m-1]
    for m=1..M), g0 (replication cost X_0* = g_0(S_0)), frictionless (the
    discounted risk-neutral expected payoff), and nondegenerate (bool, True
    if condition (3.9) held at every node).
    """
    levels = build_tree(s0, u, d, m_steps)
    q = ((1.0 + r_per_step) - d) / (u - d)
    assert 0.0 < q < 1.0, "risk-neutral probability out of (0,1): check u, d, r_per_step"
    disc = 1.0 / (1.0 + r_per_step)

    # Terminal step m = M: solve the one-period fixed point at each penultimate node.
    s_prev = levels[m_steps - 1]
    f = {m_steps: call_delta_min(n, k, s_prev * u, s_prev * d, lam)}
    # g_M(s, s') = V(s' + lambda f_M(s)) evaluated at the two children s'=su, s'=sd
    g = {m_steps: {
        "up": n * np.maximum(s_prev * u + lam * f[m_steps] - k, 0.0),
        "down": n * np.maximum(s_prev * d + lam * f[m_steps] - k, 0.0),
    }}

    nondegenerate = True
    two_phi_minus_lam = 2.0 * phi - lam
    phi_rh = phi * r_per_step

    for m in range(m_steps - 1, 0, -1):
        s_prev = levels[m - 1]
        s_cur = levels[m]
        f_up_child = f[m + 1][1:]    # f_{m+1}(s*u) at nodes s = levels[m-1]
        f_down_child = f[m + 1][:-1]  # f_{m+1}(s*d)
        g_up_up = g[m + 1]["up"][1:]
        g_up_down = g[m + 1]["up"][:-1]
        g_down_up = g[m + 1]["down"][1:]
        g_down_down = g[m + 1]["down"][:-1]

        # One-step conditional expectation E^Q[g_{m+1}(s,.)|S_{t_m}=s] for s = s_prev*u (up) and s_prev*d (down)
        e_g_up = q * g_up_up + (1.0 - q) * g_up_down
        e_g_down = q * g_down_up + (1.0 - q) * g_down_down

        denom = (s_cur[1:] - s_cur[:-1]) + two_phi_minus_lam * (f_up_child - f_down_child)
        if np.any(np.abs(denom) < 1e-12):
            nondegenerate = False
        num = (e_g_up - e_g_down) + (two_phi_minus_lam + phi_rh) * (f_up_child**2 - f_down_child**2)
        f_m = num / ((1.0 + r_per_step) * denom)
        f[m] = f_m

        # g_m(s, s') for s' = s*u and s' = s*d (children of each node s = levels[m-1]).
        # E^Q[g_{m+1}(s', .)|S_tm=s'] with s'=s*u equals e_g_up (conditioning on S_tm=su
        # uses the same children g_up_up, g_up_down); similarly for s'=s*d.
        g_up = e_g_up / (1.0 + r_per_step) + f_up_child * (
            (two_phi_minus_lam + phi_rh) / (1.0 + r_per_step) * f_up_child - two_phi_minus_lam * f_m
        )
        g_down = e_g_down / (1.0 + r_per_step) + f_down_child * (
            (two_phi_minus_lam + phi_rh) / (1.0 + r_per_step) * f_down_child - two_phi_minus_lam * f_m
        )
        g[m] = {"up": g_up, "down": g_down}

    # m = 0: f_0(S_0) = 0 and g_0(S_0) = 1/(1+rh) * (E^Q[g_1(S_0,S_t1)] + (2phi-lam+phi*rh) f_1(S_0)^2)
    f1 = f[1]
    g1 = g[1]
    e_g1 = q * g1["up"][0] + (1.0 - q) * g1["down"][0]
    g0 = (e_g1 + (two_phi_minus_lam + phi_rh) * f1[0] ** 2) / (1.0 + r_per_step)

    frictionless = disc_expected_terminal_payoff(levels[m_steps], k, n, q, m_steps, r_per_step)
    return {
        "levels": levels,
        "f": f,
        "g": g,
        "x0_star": float(g0),
        "frictionless_price": float(frictionless),
        "nondegenerate": nondegenerate,
    }


def disc_expected_terminal_payoff(terminal_prices: np.ndarray, k: float, n: float, q: float,
                                   m_steps: int, r_per_step: float) -> float:
    """E^Q[V(S_T)]/(1+rh)^M, the frictionless no-arbitrage price (Corollary 3.3 benchmark)."""
    j = np.arange(m_steps + 1)
    binom = np.array([math.comb(m_steps, int(jj)) for jj in j], dtype=float)
    probs = binom * (q ** j) * ((1.0 - q) ** (m_steps - j))
    payoff = n * np.maximum(terminal_prices - k, 0.0)
    return float(np.sum(probs * payoff) / (1.0 + r_per_step) ** m_steps)
