# `NegativeBinomial` — over-dispersed counts (added 1.2.0)

**Why.** With a Poisson head, counts whose variance exceeds the mean cannot
be modelled no matter how the rate is fitted: on a Gamma–Poisson DGP with
Var = μ + μ²/4 the Poisson head's held-out dispersion index
√mean((y−λ̂)²/λ̂) stayed ≈ 2.1 for every β and every recalibration, 90 %
intervals covered 63 %, and the NLL gap to the data-generating model was
0.38 (`research/ngboost/reports/R12–R14`).

**Parameterization.** Internal columns `log μ` (mean) and `log r`
(dispersion, "size"); Var(Y) = μ + μ²/r; r → ∞ is the Poisson limit.
LogScore and CRPScore (discrete CRPS, see below). Fisher information is diagonal (mean and dispersion
are orthogonal in NB2): F_log μ = rμ/(r+μ); F_log r = r²·E[s(Y)²] summed
exactly over the pmf in the log domain (no digamma calls, no cancellation,
no underflow). Guards: r is clamped to [1e-3, 1e6] and `log r` is frozen at
the cap (as StudentT does for df); the log-r Fisher entry has a floor of
1e-2 because for r ≫ μ it decays like μ²/(2r²) while the gradient decays
only like μ/r — without the floor a near-Poisson fit would spend the whole
line search on the dispersion column. Init: method of moments
(r = μ²/(Var−μ), clamped to [0.1, 1000]).

**Default β = 1** (`Distribution::default_fisher_weight_power`), for the
same reason as Poisson: the mean column's Fisher varies with the rate, so
unweighted leaves are means of ratios. Override with
`fisher_weight_power = Some(0.0)`.

**Evidence** (R15: 10 seeds, lr 0.05; R16: lr 0.01, 5 seeds; same splits as
the Poisson runs). Held-out NLL; "disp" = √mean((y−μ̂)²/v̂) against the
*predicted* variance (1 = calibrated).

| DGP | Poisson head (β=1) | NegBin β=0 | NegBin β=1 | oracle | notes |
|---|---|---|---|---|---|
| nb_over (r=4, μ 1–40) | 3.835, cov90 0.63 | 3.036, disp 1.18 | 3.024 | 2.879 | r̂ 3.7–4.0; residual gap = mean bias 1.05 |
| nb_low (r=10, μ 3–10) | 2.568, cov90 0.81 | 2.498, cov90 0.89 | 2.500 | 2.477 | r̂ 10–13 |
| pois_nl (true Poisson, μ 1–40) | 2.596 | 2.689 | **2.571** | 2.418 | β=1: −0.117 (10/10), RMSE −15 % |
| pois_lin (true Poisson) | 1.794 | 1.804 | 1.803 | 1.767 | β neutral |
| pois_low (true Poisson, μ 3–10) | 2.289 | 2.292 | 2.295 | 2.275 | β=1 +0.003 (lr 0.01: +0.005); r̂ stalls at 50–80 |

Use NegativeBinomial when a holdout dispersion index under the Poisson head
is clearly above 1; on genuinely Poisson data the two heads are within
0.005 NLL of each other.

**Known limitation / next question.** On true-Poisson data r̂ grows slowly
(20–200 after early stopping instead of → ∞) because the dispersion
column's natural-gradient step is floored; the cost is ≤ 0.005 NLL. A
larger init cap or a dispersion-specific step size would close it.

**CRPScore (added Cycle 10).** Discrete CRPS Σ_k (F(k) − 1{k ≥ y})² with
its gradient and the full 2×2 metric 2Σ_k ∂F(k)∂F(k)ᵀ from one pass over the
pmf (unit-tested: matches a brute-force sum, FD gradients, and the metric
equals the expected Hessian of the CRPS). Training with it (R35, 10 seeds)
helps CRPS and RMSE only on strongly over-dispersed data (nb_over: CRPS
−0.056, RMSE −0.24) and costs NLL everywhere (+0.002…+0.029; pois_low 0/10).
Use LogScore unless your evaluation metric is CRPS on heavily dispersed
counts.
