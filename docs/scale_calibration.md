# Automatic scale recalibration (`auto_calibrate_scale`, default on since 1.2.0)

## The problem it fixes

Early-stopped NGBoost (Normal/LogScore) is **systematically overconfident
out of sample**: on every benchmark dataset measured (two UCI sets, four
synthetic DGPs; lr 0.05 and 0.01; n = 3k and 12k) the ratio
√(mean err² / mean σ̂²) on test data was 1.09–1.30, 90 % intervals covered
84–89 % and 50 % intervals 39–47 %. The log σ trees are fitted to in-sample
residuals of a μ that was fitted to the same rows, so σ̂ is biased low, and
early stopping does not remove the bias because both heads share one
stopping point.

## What it does

At the end of `fit` / `partial_fit` (after early-stopping truncation), when
validation data is present and the distribution has a log-scale column
(`Distribution::scale_param_index()`: Normal, Laplace, LogNormal, Cauchy,
StudentT, TFixedDf → column 1; NormalFixedMean → column 0), the model fits

    log σ' = a + b · log σ̂

by minimising the model's own score on the validation data (damped Newton;
convex for the Normal log score; a step is never accepted unless it lowers
the validation score; b stays > 0), and folds the map into the model:
`init_params[col] ← a + b·init` and every column-`col` learner's leaf
values × b. Nothing new is stored, so **saved models keep their format**,
and `pred_dist`, staged predictions and `partial_fit` are all consistent.
`model.scale_calibration` reports `(col, a, b)`; `None` means it did not run.

Guards (from the validation-size sweep, `research/ngboost/reports/R9_*`):
fewer than 300 validation rows → shift only (b = 1, the score-optimal common
inflation factor; the slope is unstable on heavy-tailed residuals until a
few hundred rows); fewer than 40 → skipped (with 20 rows even the shift is
a coin flip); a failure never fails the fit. μ is never touched (RMSE identical by construction). The validation
loss curve in `evals_result` describes the model *before* calibration.

## Evidence

Paired Δ test NLL vs no calibration, 10 seeds, lr 0.05 (R4/R7); seed wins
in parentheses. cov90 after calibration: 0.90–0.92 (from 0.84–0.89).

| het_a | het_b | homo | tnoise | kin8nm | protein |
|---|---|---|---|---|---|
| −0.037 (10/10) | −0.018 (9/10) | −0.023 (10/10) | −0.059 (10/10) | −0.031 (10/10) | −0.033 (9/10) |

Robust at lr 0.01 (5/5 wins on five datasets, 4/5 on protein) and at
n = 12k (never worse than baseline). The older moment-matching
`calibrate_uncertainty` **over-corrects on het_b** (+0.017); the affine
head is the only variant that improved every dataset. Laplace / StudentT
heads (R10): Laplace −0.007 … −0.016 (10/10 on all four datasets), StudentT
−0.011 … −0.039 (9–10/10) — the StudentT baseline is *under*confident and
the head corrects that direction as well.

## Time series

The calibration is only as good as the validation block. Use a
**chronological** block (the most recent rows before the forecast period):
pass it explicitly to `fit_with_validation`, or set
`validation_split = ValidationSplit::Chronological` so `validation_fraction`
takes the tail of the training rows instead of a random sample (the random
sample is nearly in-sample on autocorrelated data and makes both early
stopping and the calibration too optimistic). `hyper_opt` CV folds calibrate
on the fold holdout they are scored on — mildly optimistic but consistent
across trials, like the fold-level early stopping already was.

## Turning it off / doing it by hand

```rust
m.auto_calibrate_scale = false;                       // pre-1.2 behaviour
m.calibrate_scale_affine(&x_val, &y_val)?;            // explicit, any time after fit
m.calibrate_param_affine(&x_val, &y_val, col, None)?; // any column, weighted
```
Not applied to `NGBSurvival` (separate fit loop) or to distributions whose
second parameter is not a log-scale (Gamma, Weibull, Poisson, …).

## NegativeBinomial dispersion (Cycle 16)

Since Cycle 16 the same end-of-fit map runs on NegativeBinomial's log r
(its `scale_param_index`). The mean μ is untouched; only Var = μ + μ²/r
changes. Its minimum is **300 validation rows**, where the slope is fitted
as well (`Distribution::auto_calibration_min_rows`).

| data | Δ NLL (wins) | 90 % coverage |
|---|---|---|
| nb_over (r = 4) | −0.026 (10/10) | 0.867 → 0.897 |
| nb_low (r = 10) | −0.005 (10/10) | 0.895 → 0.895 |
| pois_nl / pois_lin / pois_low (true Poisson) | −0.022 (9/10) / −0.002 (7/10) / −0.005 (10/10) | |
| count time series, chronological val (tsnb_stat, tsnb_drift, tsp_stat) | −0.016 / −0.014 / −0.008 (10/10 each) | |

With 100 validation rows (shift only) it lost on nb_low and pois_nl
(+0.004, 3/10): r is weakly identified there, hence the 300-row floor.
Runs: `research/ngboost/reports/c16/R59*`, `R60_negbin_*`, `R63*`.
