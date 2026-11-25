"""Unit tests for the multiperiod binomial replication (Sec. 3, Theorem 3.2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.multiperiod_binomial import build_tree, replicate_call


def test_build_tree_recombines_and_has_correct_shape():
    levels = build_tree(100.0, u=1.05, d=0.95, m_steps=5)
    assert len(levels) == 6
    for m, level in enumerate(levels):
        assert level.shape[0] == m + 1
    assert levels[5][0] == pytest.approx(100.0 * 0.95 ** 5)
    assert levels[5][-1] == pytest.approx(100.0 * 1.05 ** 5)


def test_midpoint_regime_nondegenerate_and_cost_exceeds_frictionless():
    """In the midpoint regime 2*phi=lambda the nondegeneracy condition (3.9)
    always holds, and for the monotone call payoff the replication cost must
    exceed the frictionless price (Corollary 3.3)."""
    lam = 0.1
    phi = lam / 2.0
    out = replicate_call(s0=100.0, u=1.05, d=0.96, m_steps=10, r_per_step=0.001,
                          lam=lam, phi=phi, k=100.0, n=1.0)
    assert out["nondegenerate"]
    assert out["x0_star"] >= out["frictionless_price"] - 1e-8


def test_replicating_strategy_depends_only_on_fundamental_price():
    """f_m is stored as a function of the node value of the recombining tree
    only (one array per step), so two paths reaching the same fundamental
    price automatically share the same replicating position."""
    lam = 0.1
    phi = lam / 2.0
    out = replicate_call(s0=100.0, u=1.05, d=0.96, m_steps=6, r_per_step=0.001,
                          lam=lam, phi=phi, k=100.0, n=1.0)
    for m in range(1, 7):
        assert out["f"][m].shape[0] == m


def test_cost_increases_with_lambda_in_midpoint_regime():
    costs = []
    for lam in (0.02, 0.2):
        out = replicate_call(s0=100.0, u=1.05, d=0.96, m_steps=8, r_per_step=0.001,
                              lam=lam, phi=lam / 2.0, k=100.0, n=1.0)
        costs.append(out["x0_star"])
    assert costs[1] > costs[0]
