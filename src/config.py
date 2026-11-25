"""Model parameters for all experiments, as module-level constants.

Every parameter cites the paper section, equation or table it comes from,
or states that the paper does not specify it. No classes or dictionaries:
each experiment script imports the constants it needs directly.
"""

# ---------------------------------------------------------------------------
# Experiment 1: one-period binomial model, limit order book shape (Sec. 2.4.1)
# ---------------------------------------------------------------------------
EXP1_S0 = 100.0        # initial fundamental price, Sec. 2.4.1
EXP1_ST_UP = 105.0     # up-state terminal price S_T(u), Sec. 2.4.1
EXP1_ST_DOWN = 95.0    # down-state terminal price S_T(d), Sec. 2.4.1
EXP1_R = 0.01          # one-period risk-free rate, Sec. 2.4.1
EXP1_K = 100.0         # call strike, Sec. 2.4.1
# Three (lambda, phi) combinations from Figure 2 / Table (page 7)
EXP1_LAMBDA_PHI_SETS = ((0.1, 0.05), (0.2, 0.1), (0.3, 0.15))
EXP1_N_UNIT = 0.2      # illustrative underlying transaction size n, Fig. 2 caption
# Illustrative option transaction size m: the paper fixes n = 0.2 shares of the
# underlying but does not give the corresponding m in contracts (uncertainty
# noted in the task); we take m = n, the same illustrative size in contract
# units, consistent with one option contract corresponding to N shares.
EXP1_M_UNIT = EXP1_N_UNIT
EXP1_N_MULTIPLES = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)  # multiples of n/m, Fig. 2 table

# ---------------------------------------------------------------------------
# Experiment 2: SPY call spread calibration (Sec. 2.4.2, Appendix A, page 29)
# ---------------------------------------------------------------------------
EXP2_UNDERLYING = "SPY"              # NYSE Arca ETF, Sec. 2.4.2
EXP2_START_DATE = "2025-01-01"       # sample start, Sec. 2.4.2
EXP2_END_DATE = "2025-06-30"         # sample end, Sec. 2.4.2 ("January to June 2025")
EXP2_LOOKBACK_DAYS = 60              # trading days for sigma_d, Appendix A (page 29)
EXP2_MONEYNESS_BANDS = (0.005, 0.01, 0.02, 0.05, None)  # delta bands of Table 1 (None = all strikes)
# Paper's full universe of near-expiry SPY options, Table 1 (page 8)
EXP2_N_OPTIONS_PAPER = 15570
# Downloading full NBBO quote and trade history for every SPY option expiring
# between Jan and Jun 2025 (daily expirations) is not feasible in the time
# available; see "Deviations from the paper" in results/RESULTS.md for the
# executed, reduced expiration sample and the corresponding command.
EXP2_N_EXPIRIES_RUN = 8              # expirations sampled across the period for the executed run

# ---------------------------------------------------------------------------
# Experiment 3: multiperiod binomial replication (Sec. 3)
# ---------------------------------------------------------------------------
EXP3_S0 = 100.0        # initial fundamental price, not specified in the paper for Sec. 3 numerics
EXP3_T = 1.0           # maturity in years, not specified in the paper for Sec. 3 numerics
EXP3_M_STEPS = 50      # number of time steps M, not specified in the paper for Sec. 3 numerics
EXP3_SIGMA = 0.2       # annualized volatility sigma used in u = e^{sigma sqrt(h)}, Sec. 3.3
EXP3_R = 0.05          # interest rate per period (annualized), not specified in the paper
EXP3_K = 100.0         # call strike, consistent with Sec. 2.4.1 example
EXP3_N = 1.0           # number of contracts
EXP3_LAMBDA = 0.1      # price impact parameter, consistent with Sec. 2.4.1/5.2 examples
EXP3_PHI_MIDPOINT = EXP3_LAMBDA / 2.0  # midpoint regime 2*phi = lambda, Sec. 3.2/Thm 3.2
EXP3_PHI_GENERAL = 0.08  # a 2*phi > lambda case to also check nondegeneracy (3.9) numerically

# ---------------------------------------------------------------------------
# Experiment 4: quadratic payoff closed form (Sec. 5.1, Figure 5)
# ---------------------------------------------------------------------------
EXP4_ALPHA = 0.02      # quadratic coefficient alpha; not specified in the paper, chosen so
                       # alpha < (16*phi*N)^{-1} holds for every (N, lambda) combination below
EXP4_BETA = -2.0       # linear coefficient beta; not specified in the paper
EXP4_GAMMA = 30.0      # constant coefficient gamma; not specified in the paper
EXP4_SIGMA = 0.2       # volatility, not specified in the paper for Figure 5
EXP4_R = 0.05          # interest rate, not specified in the paper for Figure 5
EXP4_T = 1.0           # maturity, not specified in the paper for Figure 5
EXP4_S0_MIN = 10.0     # grid lower bound, Figure 5 table (page 21)
EXP4_S0_MAX = 15.0     # grid upper bound, Figure 5 table (page 21)
EXP4_S0_POINTS = 41    # grid resolution; the paper's table only lists 5 points (page 21)
EXP4_N_VALUES = (1, 10)           # contract counts of Figure 5
EXP4_LAMBDA_VALUES = (0.01, 0.1)  # price-impact parameters of Figure 5 (midpoint regime 2*phi=lambda)

# ---------------------------------------------------------------------------
# Experiment 5: continuous-time call replication (Sec. 5.2, Figures 6-9)
# ---------------------------------------------------------------------------
EXP5_K = 100.0         # strike, Figure 6 caption (page 24)
EXP5_LAMBDA = 0.1      # price impact, Figure 6 caption
EXP5_PHI = 0.06        # execution cost parameter, Figure 6 caption (2*phi=0.12 > lambda=0.1)
EXP5_R = 0.05          # interest rate, Figure 6 caption
EXP5_SIGMA = 0.10      # volatility, Figure 6 caption
EXP5_N = 1             # number of contracts for Figures 6-7 and 9
EXP5_T = 1.0           # maturity; not explicitly stated in the paper, taken as one year
EXP5_S0 = 100.0        # initial underlying price for Monte Carlo; not stated in the paper
EXP5_N_VALUES = (1, 10)           # contract counts of Figure 8
EXP5_LAMBDA_VALUES = (0.01, 0.1)  # price-impact parameters of Figure 8 (phi fixed at EXP5_PHI)
EXP5_SPATIAL_MIN = 20.0    # truncated spatial domain lower bound for the PDE grid
EXP5_SPATIAL_MAX = 300.0   # truncated spatial domain upper bound for the PDE grid
EXP5_N_SPACE_PAPER = 800   # paper-scale spatial grid points
EXP5_N_TIME_PAPER = 2000   # paper-scale time grid points
EXP5_N_SPACE_RUN = 240     # executed spatial grid points (time-limited; see RESULTS.md)
EXP5_N_TIME_RUN = 500      # executed time grid points (time-limited; see RESULTS.md)
EXP5_MC_PATHS_PAPER = 1000  # Monte Carlo paths for Figure 9 search, Sec. 5.2 text
EXP5_MC_PATHS_RUN = 1000    # executed Monte Carlo paths
EXP5_MC_STEPS_LIST = (100, 1000, 10000)  # discretisation steps for the Figure 7 replication check
EXP5_MC_STEPS_LIST_RUN = (100, 1000)     # time-limited subset actually executed (see RESULTS.md)
