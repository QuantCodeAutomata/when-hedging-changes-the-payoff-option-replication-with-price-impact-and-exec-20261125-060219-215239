"""Sanity checks on the experiment configuration constants."""
from __future__ import annotations

import src.config as config


def test_exp1_up_down_prices_bracket_s0():
    assert config.EXP1_ST_DOWN < config.EXP1_S0 < config.EXP1_ST_UP


def test_exp1_lambda_phi_sets_satisfy_no_manipulation_condition():
    from src.execution_costs import no_manipulation_condition_linear
    for lam, phi in config.EXP1_LAMBDA_PHI_SETS:
        assert no_manipulation_condition_linear(lam, phi)


def test_exp3_midpoint_regime_matches_definition():
    assert config.EXP3_PHI_MIDPOINT == config.EXP3_LAMBDA / 2.0


def test_exp4_alpha_below_closed_form_bound_for_every_n():
    for n in config.EXP4_N_VALUES:
        for lam in config.EXP4_LAMBDA_VALUES:
            phi = lam / 2.0
            assert config.EXP4_ALPHA < 1.0 / (16.0 * phi * n)


def test_exp5_general_regime_has_2phi_above_lambda():
    assert 2.0 * config.EXP5_PHI > config.EXP5_LAMBDA
