# Results — When Hedging Changes the Payoff: Option Replication with Price Impact and Execution Costs

Paper: David Itkin and Leandro Sánchez-Betancourt, *When Hedging Changes the
Payoff: Option Replication with Price Impact and Execution Costs*
(arXiv:2609.36257v2).

This file reports the outputs of the five implemented experiments, each compared
against the values and qualitative claims given in the paper, together with the
deviations made from the paper's exact methodology and the paper results that
are not reproduced here.

## Experiment 1 — One-period binomial model: limit order book shape (Sec. 2.4.1)

`python -m exp.exp1` writes `results/exp1_execution_prices.json` and
`results/exp1_execution_prices.png`.

For the three (λ, φ) pairs (0.1, 0.05), (0.2, 0.1), (0.3, 0.15), the average
execution price in the underlying, `S_0 + φ(N)/N`, and in the option market,
`X_0*/N` from Proposition 2.4 and Equation (2.7), were computed for multiples of
the illustrative trade size `n = m = 0.2`.

Representative rows (λ=0.1, φ=0.05): buying 1 unit gives an average underlying
price of 100.0224 and an average option price of 2.9871; selling 1 unit gives
99.9776 and 2.9536 respectively, so a strictly positive bid-ask spread is
produced in **both** markets purely from the nonlinear execution-cost function
(2.9), and the spread widens as the traded multiple grows, reproducing the
concave/convex (square-root-then-linear) shape of Figure 2.

**Verdict:** reproduced (qualitative shape: concavity of the execution-cost
function, inheritance of the nonlinear shape by the option price, and growth of
the spread with λ and φ, all hold as in Figure 2; the paper gives no table of
numerical values to match beyond the qualitative shape).

## Experiment 2 — Calibration to SPY call option data (Sec. 2.4.2, Appendix A)

`python -m exp.exp2` writes `results/exp2_calibration.json` and
`results/exp2_calibration.png`.

Executed configuration: 8 expiries sampled evenly across 2025-01-01 to
2025-06-30 (vs. the paper's full near-expiry SPY universe of 15,570 options
across the same six months), 68 usable option/expiry/strike observations after
excluding 12 strike/expiry combinations with no usable NBBO quotes on the
preceding session. SPY daily bars via the QuantX Dataloader; SPY and option
NBBO quotes and the listed option chain via Massive REST.

| Moneyness band | n options | Median explained spread (%) | Paper (%) |
|---|---|---|---|
| δ ≤ 0.5% | 16 | 642.9 | 85 |
| δ ≤ 1% | 38 | 600.8 | 58.53 |
| δ ≤ 2% | 55 | 553.6 | 35.72 |
| δ ≤ 5% | 65 | 359.6 | 9.04 |
| all strikes | 68 | 312.8 | 4.49 |

The qualitative ranking (explained percentage falling as the moneyness band
widens) holds in the executed sample, but every band's explained percentage is
far above 100%, i.e. the calibrated model over-predicts the observed spread by
roughly 3.5x-7.5x, instead of under-predicting it as in the paper's 85%→4.49%
sequence.

**Verdict:** not reproduced. The direction of the band ranking matches the
paper, but the magnitude is materially off (differs by far more than 50% and
has model spread exceeding observed spread rather than falling short of it).

## Experiment 3 — Multiperiod binomial replication (Sec. 3)

`python -m exp.exp3` writes `results/exp3_replication_cost.json` and
`results/exp3_replication.png`.

The paper gives no numerical example for the multiperiod case, so parameters
S_0=100, K=100, T=1, σ=0.2, r=0.05, λ=0.1 were chosen consistently with the
one-period example of Sec. 2.4.1 and the continuous-time example of Sec. 5.2,
at M∈{4,8} steps, in the midpoint regime (2φ=λ=0.1) and a general regime
(φ=0.08, 2φ>λ).

| M | regime | X_0* | frictionless price | X_0* − frictionless | nondegenerate |
|---|---|---|---|---|---|
| 4 | midpoint | 10.0081 | 9.9541 | 0.0540 | True |
| 4 | general | 10.0432 | 9.9541 | 0.0890 | True |
| 8 | midpoint | 10.2510 | 10.1968 | 0.0542 | True |
| 8 | general | 10.2862 | 10.1968 | 0.0894 | True |

**Verdict:** reproduced. In every configuration the replication cost X_0*
strictly exceeds the frictionless binomial price (Corollary 3.3 for the
monotone call payoff), the nondegeneracy condition (3.9) holds at every node in
both the midpoint and general regime, and the replicating strategy depends only
on the fundamental price at each node by construction of the backward
induction (Theorem 3.2).

## Experiment 4 — Quadratic payoff closed form (Sec. 5.1, Figure 5)

`python -m exp.exp4` writes `results/exp4_quadratic_payoff.json` and
`results/exp4_quadratic_payoff.png`.

For V(x) = N(αx²+βx+γ) with α=0.02, β=-2, γ=30, σ=0.2, r=0.05, T=1 (not given
in the paper for Figure 5; chosen so α < (16φN)⁻¹ holds for every (N,λ) pair
used), and N∈{1,10}, λ∈{0.01,0.1} (midpoint regime 2φ=λ):

* `price_above_frictionless`: **True** — every impacted price lies above the
  frictionless Black-Scholes-type benchmark.
* `scaling_property`: **True** — the per-contract price and per-contract hedge
  for (N=1, λ=0.1) coincide with (N=10, λ=0.01) to numerical precision (both
  have λN=0.1), confirming the stated scaling invariance of the midpoint
  regime.
* `increasing_in_n`: **True** — both the per-contract replication cost and the
  per-contract hedge size increase with N even after normalizing by N.

**Verdict:** reproduced. All three qualitative claims of Sec. 5.1 hold exactly
in the computed grid (S_0 ∈ [10,15], 41 points).

## Experiment 5 — Continuous-time call replication (Sec. 5.2, Figures 6-9)

`python -m exp.exp5` writes `results/exp5_results.json` and four figures
(`exp5_figure6_terminal_condition.png`, `exp5_figure7_replication_check.png`,
`exp5_figure8_price_position.png`, `exp5_figure9_moving_target.png`).

Parameters K=100, λ=0.1, φ=0.06, r=0.05, σ=0.10 are taken from the Figure 6
caption; T=1 and S_0=100 are not stated in the paper and were assumed. The PDE
(4.6) is solved by finite differences with Newton iteration on the nonlinear
`(∂_xx u)²` term (2φ>λ here), on a grid reduced from the paper scale
(800 × 2000 points) to 240 × 500 points, and the Figure 7 replication-identity
check uses {100, 1000} discretisation steps instead of {100, 1000, 10000},
because the paper-scale grid and the 10,000-step Monte Carlo path set could not
both finish in the time available; `--full` selects
`EXP5_N_SPACE_PAPER`/`EXP5_N_TIME_PAPER`/`EXP5_MC_STEPS_LIST` instead.

* Figure 7 replication check: mean |V(P_T*) − X_T*| falls from 0.359 at
  25×100 to 0.00198 at 250×1000 grid points, i.e. the replication identity
  holds with increasing accuracy as the discretization is refined, as claimed.
* Figure 9 moving-target search: of 1000 simulated paths, 1 exhibits
  S_T < K < P_T* (the hedger's own trading brings an out-of-the-money option
  into the money), versus the paper's reported count of approximately 9 out of
  1000.
* Figure 8: price and position per contract increase with N (N=10 dominates a
  10x increase in λ for fixed φ, as claimed).

**Verdict:** partially reproduced. The replication-identity convergence
(Figure 7) and the price/position ordering (Figure 8) reproduce the paper's
claims. The moving-target count (Figure 9) is an order of magnitude below the
paper's reported ≈9/1000 paths; this count is a rare-event statistic that is
sensitive to the unstated S_0, T and to the reduced PDE grid, and the specific
cause was not isolated within the time available.

## Deviations from the paper

* Experiment 2 uses the QuantX Dataloader (SPY daily bars) and Massive REST
  (SPY and SPY-option NBBO quotes, option contract reference) instead of the
  paper's WRDS/TAQ (underlying) and Cboe/Databento (options) sources, which are
  not accessible here; both deliver NBBO quotes and sizes, which is what the
  calibration formulas of Appendix A require.
* Experiment 2's "preceding trading session" median spread and volume are
  computed from three 5-minute NBBO sampling windows per session (morning,
  midday, close) rather than the full continuous session, because pulling the
  complete NBBO quote stream for every strike of every near-expiry SPY option
  over six months is not feasible in the time available; this sampling is a
  plausible, but not confirmed, contributor to the gap between the executed
  and the paper's explained-spread percentages.
* Experiment 2 uses 8 sampled expiries and 68 strike/expiry observations
  instead of the paper's full near-expiry universe of 15,570 options over the
  same six months, because downloading full-session NBBO quotes for every
  SPY expiry in Jan-Jun 2025 was estimated to exceed the time available; the
  full configuration is run with `python -m exp.exp2 --full`
  (`EXP2_N_EXPIRIES_PAPER` instead of `EXP2_N_EXPIRIES_RUN` candidate
  expiries, drawn by the same even-spacing rule over the sample period).
  12 strike/expiry combinations were excluded for having no usable NBBO quotes
  in the sampled windows of the preceding session; they are at most 15% of the
  sampled pool and are spread across bands, so they are not identified as the
  cause of the spread-magnitude gap.
* Experiment 2's observed/model spread discrepancy (bands over-explained by
  roughly 3.5x-7.5x instead of under-explained as in the paper) was checked by
  component tests of `calibrate_phi_midpoint` and `model_implied_option_spread`
  against the one-period formulas of Proposition 2.4 and Appendix A (see
  `tests/test_calibration.py`), which hold on synthetic inputs. Three further
  checks against live data and the formulas were made: (i) a direct Massive
  NBBO quote sample confirms `bid_size`/`ask_size` are already in raw shares,
  not round lots, ruling out a x100 unit error in the calibrated phi; (ii) the
  `calibrate_phi_midpoint` call site halves the quoted spread before passing it
  in and the function applies the formula's own further factor of one half,
  reproducing phi = (S^S/2)(1/V^a+1/V^b) with S^S the half-spread exactly as
  Appendix A states, so phi is not doubled or halved by a call-site mismatch;
  (iii) a numerical sweep of `model_implied_option_spread` over strike shows
  the model spread is exactly linear in the implied share count N at fixed phi
  and is zero or a single constant value for every strike outside the
  one-day up/down node pair S_T(d), S_T(u) (because Proposition 2.4's
  fixed point is degenerate once a strike is bought or sold with certainty
  at both nodes), and only varies smoothly for strikes inside that pair,
  which for SPY's roughly 0.6% one-day move spans a much narrower moneyness
  range than the 0.5%-5% bands tested; with only 8 sampled expiries most
  strikes in a given band fall in the degenerate regime, so the median
  explained percentage is dominated by a few deep-in/out-of-the-money
  observations rather than the smooth interpolation the paper's full
  15,570-option sample averages over. No formula, sign, or unit defect was
  found; the remaining gap is attributed to this sample-size/degenerate-regime
  interaction and to the sampling and universe-size reductions above, not to
  an identified coding error.
* Experiments 3, 4 and 5 use parameter values (S_0, K, T, σ, r, N and, for
  Experiment 5, the PDE spatial/time grid) that the paper does not give for
  its multiperiod, quadratic-payoff and continuous-time numerical examples
  beyond the Figure 6 caption; these are stated explicitly in `src/config.py`
  with a comment noting they are not specified in the paper, and chosen to be
  consistent with the one-period example of Sec. 2.4.1 and the Figure 6
  parameters where applicable.
* Experiment 5's PDE grid is reduced from the paper scale (800 spatial x 2000
  time points, `EXP5_N_SPACE_PAPER`/`EXP5_N_TIME_PAPER`) to 240 x 500 points
  (`EXP5_N_SPACE_RUN`/`EXP5_N_TIME_RUN`), and the Figure 7 discretization-step
  list is reduced from {100, 1000, 10000} to {100, 1000}
  (`EXP5_MC_STEPS_LIST_RUN`), because the paper-scale nonlinear PDE solve at
  10,000 Monte Carlo time steps was estimated to exceed the time available;
  `python -m exp.exp5 --full` runs the paper-scale configuration.
* Experiment 1's illustrative option trade size `m` is taken equal to the
  illustrative underlying trade size `n = 0.2` (in contract units), because the
  paper states only `n = 0.2` for the underlying and does not give the
  corresponding `m` for the options market.

## Not reproduced

* Experiment 2's full near-expiry SPY universe (15,570 options, Table 1) is not
  run; the executed 8-expiry, 68-option sample is reported in its place (see
  "Deviations from the paper").
* Experiment 5's paper-scale PDE grid (800 x 2000) and the 10,000-step
  discretization point of the Figure 7 replication check are not run; the
  reduced-grid results are reported in their place (see "Deviations from the
  paper").
