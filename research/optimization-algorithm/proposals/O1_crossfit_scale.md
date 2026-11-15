# O1 — Cross-fitted residuals for the scale stage

Status: prototype in `examples/c18_bench.rs --mode oof` (harness only, no
library change). Results: `../research_log.md`, Cycle 18.

## Question

Known limitations 12, 33 and 55 say the same thing three ways: the scale is
fitted to the *in-sample* residuals of a location that has itself been
fitted to those rows, so σ̂ is too small out of sample. The only working
remedy so far is the end-of-fit affine map log σ ← a + b·log σ on validation
rows. That needs ≥ 40 validation rows, cannot be used without validation
data, and has two parameters, so it cannot correct an optimism that varies
with x. Can the optimism be removed where it arises?

## Model, parameters, objective

Normal head, θ(x) = (μ(x), log σ(x)), LogScore (proper):
S(θ; y) = log σ + (y − μ)² / (2σ²). Domains: μ ∈ ℝ, σ > 0 via the log link.
Nothing about the family or the score changes.

## Baseline (the staged fit, `NGBoost::fit_staged`)

1. μ̂₁ = argmin Σ (yᵢ − μ(xᵢ))² by boosting, early-stopped on validation.
2. log σ̂₁ boosted on in-sample residuals rᵢ = yᵢ − μ̂₁(xᵢ):
   natural gradient ½(1 − rᵢ²/σ²).
3. μ̂₂: the location refitted with weights wᵢ ∝ 1/σ̂₁(xᵢ)² (feasible GLS).
4. log σ̂₂ boosted on rᵢ = yᵢ − μ̂₂(xᵢ); then the affine recalibration.

E[rᵢ² | xᵢ] = σ²(xᵢ)·(1 − hᵢ) + bias², where hᵢ is the effective leverage
of row i in μ̂ (adaptive split search included). Steps 2 and 4 therefore
target σ²(1 − h), not σ².

## Candidate

Replace the residuals in steps 2 and 4 by K-fold cross-fitted ones:

  r̃ᵢ = yᵢ − μ̂^(−k(i))(xᵢ),

where μ̂^(−k) is the same location fit (same hyperparameters, same weights,
the round count the full-data fit kept, no early stopping) on the rows
outside fold k(i). Predictions still use the full-data μ̂.

E[r̃ᵢ² | xᵢ] = σ²(xᵢ) + E[(μ̂^(−k)(xᵢ) − μ(xᵢ))²]: the predictive variance
of a location fitted on (K−1)/K of the rows — slightly *conservative* for
the full-data location, by the usual learning-curve argument.

Required quantities: none new (the same gradients and metric). Positivity
of σ is kept by the log link. No linear system is solved.

Cost: 2K extra location fits of fixed length (K = 5: ≈ 10 fits at
(K−1)/K of the rows), no extra memory beyond the n residuals. No
convergence argument beyond the baseline's; this changes the regression
target of the scale stage, not the optimiser.

## Predictions (falsifiable)

P1. Without any recalibration, the cross-fitted variant has out-of-sample
    z-rms ≈ 1 and 90 % coverage ≈ 0.90, where the in-sample variant is
    overconfident (z-rms > 1, coverage < 0.90).
P2. Uncalibrated cross-fitted ≤ uncalibrated in-sample on test NLL, most
    at small n (R65: +0.8…+14 NLL at n = 700 for the in-sample variant).
P3. With recalibration on top, cross-fitted ≥ in-sample only if the
    optimism varies with x; otherwise the two tie and the cheaper one wins.

## Failure modes and simpler explanations

- The scale stage still early-stops on validation residuals of the
  full-data μ̂, so "no validation data needed" is *not* tested here.
- If μ̂^(−k) is much worse than μ̂ (small n, K small), r̃ is too large and
  σ̂ too wide: coverage > 0.90, NLL worse than calibrated in-sample.
- Simpler competing explanation for any gain: the scale trees are just
  being regularised (larger residual variance → different early stop).
  Check against O2 (leaf size), which regularises without cross-fitting.
- Step 3's weights come from a σ̂ fitted on all rows, so the fold fits in
  step 4 are not fully independent of the held-out fold. Second order.

## Library change (approved by the user and implemented 2026-11-13)

`NGBoost::staged_crossfit_folds: usize` (also on `NGBoostParams`), default
0 = off; ≥ 2 enables it. Training-time only; the model format and the
prediction path are unchanged.

### What the library does (differences from the harness recipe)

- `fit_staged` calls `crossfit_location` after each location stage (1 and
  3). For K folds it fits K location models — other columns frozen at the
  fold's marginal constants, the stage's weights (stage 3: the Fisher
  weights), no validation data — each stopped after the number of rounds
  the full location fit kept, with the same `n_estimators` so that a
  non-constant learning-rate schedule is replayed, not rescaled.
- The out-of-fold location column replaces the in-sample one for the
  training rows in the scale stage: in the constants' refit and as the
  frozen column of the boosting loop. Validation rows use the full model,
  as before.
- Folds: a random permutation from the model's RNG (reproducible with a
  seed), or contiguous blocks with `ValidationSplit::Chronological`. The
  harness used contiguous folds on pre-shuffled rows.
- With the option on, the default staged fit no longer requires the
  recalibration to run (`uses_staged_fit`): it stages with fewer than 40
  validation rows too. The recalibration itself is unchanged and still
  runs when it can.
- Fewer than two training rows per fold is an error.

### The protein small-sample loss (the open point before approval)

Harness recipe, 700 rows (490 training), Δ NLL of uncalibrated cross-fit
against the library default, 10 seeds:

| leaf | protein | seeds won | the other five datasets |
|---|---|---|---|
| 1 (`O1_oof_n700*.csv`) | +0.230 | 4/10 | −0.03…−0.13 on four, kin8nm +0.028 (n.s.) |
| 20 (`O1_oof_n700_leaf20.csv`) | +0.089 (0.053) | 4/10 | −0.014…−0.133 on all five |

So about two thirds of the loss was the leaf size: with single-row leaves
the scale trees turn a few small honest residuals into very small σ. The
remainder is concentrated in four seeds (1, 2, 4, 8). In all four the
validation NLL *improved* against the default (e.g. 2.72 vs 2.74, 2.81 vs
2.85, 2.91 vs 2.99) while test NLL worsened by 0.13–0.50 with coverage
unchanged or better. That pattern — better on the 70 validation rows,
worse on 140 test rows, coverage fine — is what a sharper σ(x) selected
on a small validation block does under heavy-tailed residuals: a few test
rows in confidently narrow regions carry the loss. It was not verified
row by row (the harness does not write per-row scores), so this is the
likely explanation, not an established one. The loss is not significant
at leaf 20 (1.7 se) but it is the same sign in every small-data setting
below, so it is documented as a real limitation.

### Library results (Normal, K = 5, 10 seeds, Δ NLL (se), seeds won)

| dataset | 2100 rows, vs staged default (leaf 20) | no recalibration (leaf 20) | 2100 rows, leaf 1 | 490 rows vs default (leaf 20) | 30 validation rows vs default (leaf 20) |
|---|---|---|---|---|---|
| het_a | −0.055 (0.010) 10/10 | −0.053 10/10 | −0.083 9/10 | −0.065 (0.034) 9/10 | −0.290 (0.033) 10/10 |
| het_b | +0.002 (0.005) 5/10 | +0.001 6/10 | +0.014 3/10 | −0.094 (0.027) 9/10 | −0.083 (0.018) 10/10 |
| homo | −0.024 (0.013) 7/10 | −0.023 7/10 | −0.073 10/10 | −0.033 (0.028) 7/10 | −0.184 (0.014) 10/10 |
| tnoise | −0.087 (0.018) 9/10 | −0.071 10/10 | −0.097 10/10 | −0.068 (0.027) 8/10 | −0.296 (0.038) 10/10 |
| kin8nm | −0.051 (0.006) 10/10 | −0.045 10/10 | −0.053 10/10 | −0.026 (0.013) 8/10 | −0.135 (0.021) 10/10 |
| protein | −0.012 (0.019) 8/10 | +0.016 7/10 | +0.023 7/10 | **+0.069 (0.031) 4/10** | **+0.094 (0.064) 5/10** |
| mean | −0.038 | −0.029 | −0.045 | −0.036 | −0.149 |

Files: `O1_lib_xfit_leaf20.csv`, `O1_lib_xfit_leaf1.csv`,
`O1_lib_xfit_n700_leaf20.csv`, `O1_lib_xfit_val30_leaf20.csv`. With 30
validation rows the default is a joint, uncalibrated fit, so that column
also contains the staged fit's gain; on protein the staged location,
early-stopped on 30 rows, is the worse one (RMSE +0.13).

RMSE moves little at 2100 rows (−0.02…+0.02); 90 % coverage with the
recalibration off is 0.875–0.904 (in-sample staged, uncalibrated:
0.71–0.85 in the harness runs).

### Other staged heads (K = 5, leaf 20, 10 seeds, vs the head's default)

| head | dataset | Δ NLL (se) | seeds | mean / CRPS |
|---|---|---|---|---|
| LogNormal | lognorm_het | −0.018 (0.009) | 9/10 | CRPS −0.019 |
| LogNormal | kin8nm | −0.122 (0.013) | 10/10 | **mean explodes on 8/10 seeds** (RMSE 147 … 5.6e25; default 0.15–0.57) |
| LogNormal | support_cost | −0.007 (0.002) | 10/10 | CRPS −45 (0.3 %) |
| GammaMean | gamma_het | −0.005 (0.006) | 6/10 | RMSE −0.006 |
| GammaMean | lognorm_het | −0.010 (0.007) | 7/10 | RMSE −0.059 |
| GammaMean | kin8nm | −0.084 (0.007) | 10/10 | RMSE +0.001 |
| StudentTFiniteVar | het_a | −0.048 (0.011) | 10/10 | RMSE −0.002; cov90 0.928 → 0.901 |
| StudentTFiniteVar | tnoise | −0.037 (0.015) | 9/10 | RMSE +0.001; cov90 0.926 → 0.902 |
| StudentTFiniteVar | kin8nm | −0.040 (0.009) | 9/10 | RMSE +0.002 (0.001); cov90 0.910 → 0.905 |
| StudentTFiniteVar | protein | −0.008 (0.012) | 7/10 | RMSE −0.051 (0.029); CRPS −0.028 (0.015); cov90 unchanged |

The LogNormal/kin8nm case is a failure mode of the head that
cross-fitting amplifies: honest log-scale residuals of a bounded target
have a heavy lower tail, the scale trees assign a large σ to a few rows,
and the mean exp(μ + σ²/2) is exponential in σ². NLL, coverage and
medians are fine. No guard was added (the option is opt-in and the hazard
is documented on the field, in the README and as limitation 58).

### Tests

`tests/test_cycle18.rs`: off ⇒ bit-identical to the previous staged fit;
uncalibrated cross-fit has better coverage and NLL than uncalibrated
in-sample; reproducible with a seed; serialization round trip; the
`uses_staged_fit` rule; 30 validation rows; sample weights with the
automatic split; a decaying learning-rate schedule; `partial_fit`
afterwards; too few rows per fold is an error; `NGBoostParams` carries the
option.

### Still open

- The protein mechanism row by row; a guard for LogNormal means.
- Time series: folds ignore autocorrelation.
- Folds run one after another (each fit is internally parallel).
  Measured wall time at 2100 rows: 2.6–4.8× the staged default (ten fits
  running side by side, so indicative only), with 0–40 % more rounds in
  the final model.

