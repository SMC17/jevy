# Numerical checks

Self-convergence and finite-difference checks. They are not a claim
that a published calibration table was reproduced. Citations:

- Baldacci, Bergault, Guéant, https://arxiv.org/abs/1907.12433
  (constant-vega HJB). The Euro Stoxx grid in their §4 is not this toy.
- Guéant, Lehalle, Fernandez-Tapia, https://arxiv.org/abs/1105.3115
  (RK4 on the ODE versus the principal eigenmode).
- Black–Scholes–Merton Greeks versus central differences.
- Gatheral, Jacquier, https://arxiv.org/abs/1204.0646
  (butterfly g(k) outside the knots used to calibrate).

## HJB value at zero vega

| N_V | N_t | w(0) | abs err vs finest |
| --- | --- | --- | --- |
| 11 | 20 | 2.133373 | 0.130090 |
| 21 | 40 | 2.220119 | 0.043344 |
| 31 | 80 | 2.248386 | 0.015076 |
| 41 | 120 | 2.263463 | 0.000000 |

The last row is the reference, so its error is zero by construction.
A smaller error on a finer grid is the check; the level of w(0) is
not a published digit.

## Guéant RK4 vs spectral (inventory = 1, cap = 4, horizon = 3y, 800 steps)

| γ | k | A | σ | RK4 δb | spectral δb | abs | RK4 δa | spectral δa | abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | 1.50 | 80.0 | 0.40 | 0.7546 | 0.7546 | 0.0000 | 0.6114 | 0.6114 | 0.0000 |
| 0.20 | 1.20 | 60.0 | 0.35 | 0.9084 | 0.9084 | 0.0000 | 0.7279 | 0.7279 | 0.0000 |
| 0.05 | 2.00 | 100.0 | 0.30 | 0.5750 | 0.5750 | 0.0000 | 0.4687 | 0.4687 | 0.0000 |

Agreement is a numerical statement about this implementation.
A large gap would mean the horizon is too short for the eigenmode
or the RK4 step is coarse. It is not a latency number.

## Greeks vs central differences

| spot | strike | call | |Δ| | |Γ| | |ν| |
| --- | --- | --- | --- | --- | --- |
| 100.0 | 100.0 | 1 | 1.801e-08 | 2.739e-09 | 4.034e-08 |
| 100.0 | 110.0 | 0 | 4.029e-08 | 3.929e-10 | 2.972e-07 |
| 80.0 | 75.0 | 1 | 2.008e-08 | 3.586e-10 | 8.682e-07 |

Steps: ΔS = max(1e-2, 1e-4 S), Δσ = 1e-4. Analytic minus central difference.

## SVI wings past the calibration knots

Truth `SviParams(0.04, 0.1, -0.4, 0, 0.2)` is sampled on k ∈ [−0.25, 0.25]
(7 knots) and refit. g(k) is then evaluated on k ∈ [−1.2, 1.2].

- fit (a, b, ρ, m, σ) = (0.0160, 0.1381, -0.2907, -0.0005, 0.3201)
- max |w error| on the knots: 2.188102e-04
- min g(k) on the full sweep: 0.281298
- min g(k) outside the knots (38 points): 0.281298

Negative g outside the knots is a real arbitrage the in-sample check can miss.

