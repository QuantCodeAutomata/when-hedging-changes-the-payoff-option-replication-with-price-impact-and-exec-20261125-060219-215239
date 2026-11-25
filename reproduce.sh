#!/usr/bin/env bash
# Runs the test suite and every experiment of this repository, in order.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Unit tests (excluding market-data tests) ==="
python -m pytest -m "not market_data" -q

echo "=== Experiment 1: one-period binomial model, limit order book shape ==="
python -m exp.exp1

echo "=== Experiment 2: SPY call option calibration ==="
python -m exp.exp2

echo "=== Experiment 3: multiperiod binomial replication ==="
python -m exp.exp3

echo "=== Experiment 4: quadratic payoff closed form ==="
python -m exp.exp4

echo "=== Experiment 5: continuous-time call replication ==="
python -m exp.exp5

echo "=== Done. See results/RESULTS.md for the comparison against the paper. ==="
