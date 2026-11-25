"""Fast tests for experiment 4 (quadratic payoff replication, Sec. 5.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from exp import exp4
from src.quadratic_payoff import solve_quadratic_ansatz


def test_run_executes_and_checks_pass():
    s0_grid = np.linspace(exp4.EXP4_S0_MIN, exp4.EXP4_S0_MAX, 6)
    computed = exp4.compute_curves(s0_grid)
    curves, frictionless = computed["curves"], computed["frictionless"]
    checks = exp4.run_checks(s0_grid, curves, frictionless)
    assert checks["price_above_frictionless"]
    assert checks["scaling_property"]
    assert checks["increasing_in_n"]


def test_price_above_frictionless_benchmark():
    s0_grid = np.linspace(exp4.EXP4_S0_MIN, exp4.EXP4_S0_MAX, 6)
    computed = exp4.compute_curves(s0_grid)
    n, lam = exp4.EXP4_N_VALUES[0], exp4.EXP4_LAMBDA_VALUES[0]
    sol = computed["curves"][(n, lam)]
    benchmark = computed["frictionless"][n]
    assert np.all(sol["price_per_contract"] >= benchmark - 1e-6)


def test_scaling_property_directly():
    s0_grid = np.linspace(exp4.EXP4_S0_MIN, exp4.EXP4_S0_MAX, 6)
    kwargs = dict(
        alpha=exp4.EXP4_ALPHA, beta=exp4.EXP4_BETA, gamma=exp4.EXP4_GAMMA,
        r=exp4.EXP4_R, sigma=exp4.EXP4_SIGMA, t_mat=exp4.EXP4_T,
    )
    sol_a = solve_quadratic_ansatz(s0_grid, 1, lam=0.1, **kwargs)
    sol_b = solve_quadratic_ansatz(s0_grid, 10, lam=0.01, **kwargs)
    assert np.allclose(sol_a["price_per_contract"], sol_b["price_per_contract"], rtol=1e-6)
    assert np.allclose(sol_a["position_per_contract"], sol_b["position_per_contract"], rtol=1e-6)


def test_results_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(exp4, "RESULTS_DIR", Path(tmp_path))
    exp4.run()
    assert (Path(tmp_path) / "exp4_quadratic_payoff.json").exists()
    assert (Path(tmp_path) / "exp4_quadratic_payoff.png").exists()
