# Changelog

## 1.2.0

This release changes several **defaults**. Code written for 1.1.2 still
compiles, with one exception (`NGBoostParams` struct literals, below). But
fitted models give different predictions. Every changed default has an
opt-out. With all of them set, fits are bit-identical to 1.1.2
(`tests/test_upstream_parity.rs`). The evidence for each change is in
`docs/` and `research/ngboost/`.

### Upgrading from 1.1.2

#### Changes that can break compilation

- **`NGBoostParams` has four new public fields:** `auto_calibrate_scale`,
  `validation_split`, `staged_fit` and `fisher_weight_power`. A struct
  literal that lists every field no longer compiles. Add the fields, or end
  the literal with `..Default::default()`. Code that uses `get_params()` /
  `set_params()` or `NGBoostParams::default()` is unaffected.
- **No other public item was removed or renamed.** The new trait methods on
  `Distribution` and `TrainedBaseLearner` all have default bodies, so
  custom heads and learners still compile. Two notes for custom code:
  - **Custom base learners:** implement
    `TrainedBaseLearner::scale_output(&mut self, factor) -> bool` (multiply
    the learner's output by `factor`; return `true`). Without it the
    default scale recalibration is skipped silently, and the model is left
    uncalibrated. The staged fit then runs uncalibrated as well, which is
    much worse than the joint fit (see the staged-fit row of the table
    below). With such a learner, set `staged_fit = Some(false)`. All
    built-in learners implement it.
  - **Custom distributions** keep the 1.1.2 behaviour unless they override
    one of the new hooks: `scale_param_index`, `scale_param_columns`,
    `auto_calibration_min_rows`, `default_fisher_weight_power`,
    `location_column`, `default_staged_fit` and `default_refit_intercept`.

#### Changed defaults (same code, different fitted model)

| Change | Applies to | Effect | Opt-out (per model unless noted) |
|---|---|---|---|
| Scale recalibration on the validation set at the end of `fit`: log σ ← a + b·log σ, folded into the stored model | any head with a log-scale column (Normal, LogNormal, Laplace, Cauchy, StudentT, TFixedDf, NormalFixedMean, the mixtures, the hurdle heads) when validation data is used (≥ 40 rows; the slope from 300); **NegativeBinomial** on log r, with ≥ 300 rows | intervals widen to their held-out coverage; μ unchanged; `model.scale_calibration` reports `(column, a, b)` | `auto_calibrate_scale = false` |
| Fisher-weighted (Newton / second-order) trees, β = 1 | Poisson, NegativeBinomial, Categorical (incl. Bernoulli) | count totals and base rates are no longer under-predicted | `fisher_weight_power = Some(0.0)` |
| Per-round intercept refit | Categorical with K > 2 | | `refit_intercept = Some(false)` |
| Staged fit (location, then scale, then a Fisher-weighted location refit, then scale) | Normal, LogNormal, StudentT under LogScore, with early stopping, validation data and the scale recalibration active (≥ 40 validation rows) | −0.02…−0.46 NLL on most data; fit time 1.3–3.7×, also inside `hyper_opt` folds; the model format is unchanged | `staged_fit = Some(false)` |
| Log-σ step guard (caps outlier steps) | Normal, NormalFixedMean under LogScore | a no-op except for extreme single-row outliers | env `NGBOOST_NORMAL_GUARD=0` (process-wide) |
| Correct CRPS metrics (2∫∇F∇Fᵀ) | `CRPScore` training of Normal, LogNormal, NormalFixedMean, HalfNormal, Gamma, Weibull, StudentT, TFixedDf, TFixedDfFixedVar and Poisson | CRPS-trained models change substantially; upstream's metrics had the wrong units and could diverge | env `NGBOOST_CRPS_METRIC_LEGACY=1` (process-wide) |

The defaults that need validation data apply both to `fit_with_validation`
and to `fit` with `validation_fraction` and early stopping.

#### Behaviour fixes

- **`partial_fit` no longer re-fits `init_params`** when the model already
  has trees. Re-fitting changed what the existing trees predict, and it
  discarded the calibration folded into `init_params`. Incremental fits now
  continue from the previous model.
- **StudentT:** the t cdf no longer panics on NaN parameters.

#### Saved models

`SerializedNGBoost` has the same fields as in 1.1.2:
- models saved by 1.1.2 load and predict identically;
- the calibration is folded into `init_params` and the trees, so saved
  1.2.0 models also predict the same in 1.1.2;
- whether a model was fitted in stages is not saved (relevant only to
  `refit_full`).

### Added

- **Heads:**
  - `NegativeBinomial` (log μ, log r);
  - `Hurdle<D>` with `HurdleNormal` and `HurdleLogNormal` (an atom at 0
    plus a continuous head);
  - `NormalMixture<K>` (`NormalMixture2`, `NormalMixture3`).

  NegativeBinomial, the hurdle heads and the mixtures also support
  `CRPScore`.
- **`NGBAutoMixtureRegressor`:** chooses the number of mixture components by
  validation score.
- **`CountModel::fit_auto`:** fits Poisson, then switches to
  NegativeBinomial when the validation predictions are over-dispersed
  (Dean–Lawless z > 1.645). Also `dispersion_z`.
- **New `NGBoost` methods:**
  - `refit_full(x, y, w, rounds)`: refit on all rows (e.g. train +
    validation) for the kept rounds, carrying the calibration over. Joint
    fits only.
  - `unfitted_like::<D2>()`: the same settings for another head.
  - `fit_staged`, `uses_staged_fit(n_rows, n_val)`.
  - `NGBRegressor::fit_two_stage`.
  - `calibrate_param_affine`, `calibrate_params_affine` and
    `calibrate_scale_affine`: explicit validation recalibration.
- **New options:**
  - `ValidationSplit::{Random, Chronological}`: time series should use
    `Chronological` or pass the most recent block as validation;
  - `fisher_weight_cap`, `per_param_line_search` (research options).
- **Examples:** `research_bench`, `ts_bench`, `protein_probe` and
  `mixture_sigma_probe` (research harnesses, not API).

### Docs

- `README.md`: "Reproducing upstream (1.1.2) behaviour", time-series
  guidance, the count-model switch.
- `docs/known_limitations.md`: measured negative results and caveats.
- `docs/scale_calibration.md`, `docs/normal_head.md`,
  `docs/crps_metrics.md`, `docs/fisher_weight_power.md`,
  `docs/negative_binomial.md`, `docs/hurdle.md`, `docs/normal_mixture.md`.
