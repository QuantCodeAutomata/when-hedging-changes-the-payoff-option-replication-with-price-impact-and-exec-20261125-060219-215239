# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A replication of selected numerical experiments from David Itkin, *When
Hedging Changes the Payoff: Option Replication with Price Impact and
Execution Costs* (arXiv:2609.36257v2):

1. One-period binomial model, limit-order-book shape of execution costs
   (Sec. 2.4.1).
2. Calibration of the one-period binomial model to SPY call option market
   data (Sec. 2.4.2, Appendix A).
3. Multiperiod binomial replication with linear price impact and execution
   costs (Sec. 3).
4. Quadratic payoff closed-form replication (Sec. 5.1).
5. Continuous-time call replication: the nonlinear pricing PDE, replication
   verification, and the "moving target" illustration (Sec. 4-5.2).

## Layout

```
src/    flat library modules: binomial_one_period.py, execution_costs.py,
        multiperiod_binomial.py, quadratic_payoff.py, pde_solver.py,
        calibration.py, config.py (all parameters, as module-level constants)
exp/    one script per experiment: exp1.py .. exp5.py
data/   data_loader.py: SPY bars via the QuantX Dataloader, SPY/option NBBO
        quotes and option contract reference via Massive REST
tests/  pytest tests for src/ and exp/
results/  RESULTS.md plus the JSON summaries and PNG figures each experiment
        script writes
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill DATALOADER_API_KEY / MASSIVE_API_KEY for experiment 2
```

Tests (market-data tests are skipped automatically without API keys):

```bash
python -m pytest -m "not market_data"
python -m pytest -m market_data   # requires DATALOADER_API_KEY / MASSIVE_API_KEY
```

Experiments:

```bash
python -m exp.exp1
python -m exp.exp2            # reduced-scale run (see results/RESULTS.md)
python -m exp.exp2 --full     # paper-scale candidate expiration count
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp5 --full     # paper-scale PDE grid / discretization list
```

`reproduce.sh` runs the test suite and every experiment above in order.

## Conventions

* All experiment parameters live only in `src/config.py`, as module-level
  UPPER_CASE constants with a comment naming the paper section/equation they
  come from, or stating that the paper does not specify them. No classes or
  parameter dictionaries.
* Market data is loaded only through `data/data_loader.py` (QuantX Dataloader
  for SPY bars, Massive REST for option reference data and NBBO quotes).
  `src/` and `exp/` never call a data API directly.
* `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment via
  `python-dotenv`; they are never written to code, config, logs or results.
  `.env` is gitignored; `.env.example` holds only empty placeholders.
* Raw or market-derived time series (bars, quotes, returns, equity curves,
  signals, etc.) are never committed. `results/` holds only aggregate
  summary tables (JSON/CSV, not indexed by date or bar) and static `.png`
  figures.
* Tests of `src/` components use small fixtures or synthetic inputs. Tests
  that touch real market data are marked `@pytest.mark.market_data`, use the
  cached fixture in `tests/conftest.py`, and are skipped when the API keys
  are absent.
* After rerunning an experiment, its command overwrites the corresponding
  files in `results/`; the numbers and verdicts in `results/RESULTS.md` must
  then be checked against the new files and updated by hand.
