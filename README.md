# When Hedging Changes the Payoff: Option Replication with Price Impact and Execution Costs

Replication of selected numerical experiments from David Itkin, "When Hedging
Changes the Payoff: Option Replication with Price Impact and Execution Costs".

This repository implements:

1. **One-period binomial model** (Sec. 2.4.1): the limit-order-book shape of
   execution costs in the underlying market and its inheritance by the call
   option market.
2. **Calibration to SPY option market data** (Sec. 2.4.2, Appendix A): how
   much of the observed SPY call bid-ask spread the model explains.
3. **Multiperiod binomial replication** (Sec. 3): backward-induction
   replication with linear price impact and execution costs.
4. **Quadratic payoff closed-form replication** (Sec. 5.1).
5. **Continuous-time call replication** (Sec. 4-5.2): the nonlinear pricing
   PDE, replication verification, and the "moving target" illustration.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill DATALOADER_API_KEY / MASSIVE_API_KEY for experiment 2
```

## Tests

```bash
python -m pytest -m "not market_data"
```

Experiment 2 has a market-data test that is skipped unless `DATALOADER_API_KEY`
and `MASSIVE_API_KEY` are set in the environment:

```bash
python -m pytest -m market_data
```

## Running the experiments

```bash
python -m exp.exp1   # one-period binomial model, limit order book shape
python -m exp.exp2   # SPY calibration (requires API keys)
python -m exp.exp3   # multiperiod binomial replication
python -m exp.exp4   # quadratic payoff closed-form solution
python -m exp.exp5   # continuous-time call replication (PDE + Monte Carlo)
```

`reproduce.sh` runs the test suite and every experiment in this order.

Every experiment script writes its aggregate metrics and static figures
directly into `results/`, overwriting the previous run's outputs. See
`results/RESULTS.md` for the findings, the comparison with the paper, and
every deviation.
