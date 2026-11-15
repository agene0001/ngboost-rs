# Changelog

## Unreleased (1.3.0)

From research Cycles 18–21 (`research/optimization-algorithm/`,
`research/automl-hyperparameter/`). Five **defaults** change; each has an
opt-out, and with the leaf-1 opt-out the 1.1.2 parity test still passes
bit for bit (`tests/test_upstream_parity.rs`).

### Upgrading from 1.2.0

#### Changes that can break compilation

- **`NGBoostParams` has one new public field,** `staged_crossfit_folds`.
  A struct literal that lists every field no longer compiles: add the
  field or end the literal with `..Default::default()`.
- **`HistogramLearner`, `DecisionTreeLearner` and
  `ArenaDecisionTreeLearner` have one new public field,** `adaptive_leaf`.
  A struct literal needs `adaptive_leaf: false` to keep its meaning; the
  constructors are unchanged in signature.
- **`SerializedNGBoost` has one new public field,** `target_max` (last,
  with a serde default). A struct literal needs it; see "Saved models"
  for files.

#### Changed defaults (same code, different fitted model)

| Change | Applies to | Effect | Opt-out |
|---|---|---|---|
| `HistogramLearner::new` / `default_histogram` / `histogram_learner()` use `min_samples_leaf = 20` (was 1), shrinking to a tenth of the fitted rows below 200 rows (`adaptive_leaf`) | every model built on the default histogram learner: `NGBRegressor`, `NGBClassifier`, `NGBMultiClassifier`, the survival wrappers, the auto mixture, and any `HistogramLearner::new(depth)` | held-out NLL −0.014…−0.062 on six datasets at 2100 training rows (8–10 of 10 seeds each), −0.012…−0.12 at 210 rows; never significantly worse in 50 dataset × head cases (ten head families; survival and sample-weighted fits were not measured). Below 200 rows the shrinking leaf lowered RMSE by 1.2–5.0 % against a fixed 20 (six datasets × 20 seeds; one dataset gains 4–25 %, none significantly worse) | `HistogramLearner::with_params(depth, 255, 1, 2)` |
| `DecisionTreeLearner::new` and `ArenaDecisionTreeLearner::new` use the same leaf default (was 1) | models built with those two constructors. **Not** `default_tree_learner()`, `arena_tree_learner()` or `default_sklearn()`, which keep scikit-learn's leaf 1 | exact learner: held-out NLL −0.01…−0.17 on six datasets at 210, 490 and 2100 training rows (7–10 of 10 seeds in all 18 cases); arena learner: −0.02…−0.08 on five datasets at 2100 rows, unchanged on the sixth | `DecisionTreeLearner::with_params(depth, 2, 1)` / `default_sklearn()` |
| `LogNormal` predictions clamp σ at 4 × the model's baseline σ (`Distribution::scale_ceiling`) | every prediction of an `NGBoost<LogNormal, …>` model, including models saved by 1.2.0. Not the survival wrapper, not the hurdle heads | on a target that is not log-normal in its tails the predictive mean no longer explodes (kin8nm: RMSE max 0.57 → 0.15 with the default fit, 10²⁵ → 0.15 with cross-fitting; NLL 0.001–0.003 better); on three well-specified datasets no row reaches the ceiling and results are identical. The fit and the stored model are unchanged | `model.pred_param_unbounded(&x)`, or `Normal` on ln y |
| `predict` of a `HurdleLogNormal` model returns (1 − π)·E[min(Y, largest training target)] instead of the plain mean (`Distribution::predict_capped`, `NGBoost::target_max`) | `NGBoost<HurdleLogNormal, …>::predict` and `staged_predict` on models fitted with 1.3.0. Not `pred_dist(x).mean()`, not `HurdleNormal`, not models saved before 1.3.0 (they have no stored maximum and keep the plain mean) | the plain mean is exponential in σ²; on a target with a point mass at its floor one fit in four predicted 31 against a largest target of 1.45. Bounded by the largest target by construction. On three benchmark datasets the fits are identical and RMSE is −0.8 %, −0.6 % and +0.1 % | `model.pred_dist(&x).mean()` |
| `hyper_opt` fold patience is `clamp(round(2.5 / learning_rate), 50, 200)` (was a fixed 50) when the builder sets no `early_stopping_rounds` | searches whose models have a learning rate below 0.05 | folds are no longer stopped on noise at low learning rates (0.01–0.04 NLL at 0.01); those folds run longer | set `early_stopping_rounds = Some(50)` in the builder |

`default_tree_learner()` (the upstream-parity learner) is unchanged.

### Added

- **`GammaMean`** — the Gamma distribution in (log mean, log shape)
  coordinates, where the Fisher matrix is diagonal. Staged fit by default
  under `LogScore` (from 300 validation rows, where its shape
  recalibration runs); `CRPScore` supported. Against `Gamma` on six
  datasets × 10 seeds: held-out NLL −0.03…−0.32 (10/10 seeds on each).
  `Gamma` is unchanged.
- **`StudentTFiniteVar`** — Student's t with ν > 2.1 through
  ν = 2.1 + exp(θ). Every prediction has a finite mean and variance;
  `StudentT` routinely predicts rows with ν ≤ 2 (60 of 80 benchmark fits)
  or ν ≤ 1 (22 of 80). Held-out NLL unchanged on average (−0.001;
  protein +0.02, and up to +0.07 under last-bit rounding changes — see
  `docs/known_limitations.md` 59). `LogScore` and `CRPScore`. `StudentT`
  is unchanged.
- **`NGBoost::staged_crossfit_folds`** (opt-in, default 0) — the staged
  fit's scale stages see cross-fitted (out-of-fold) location residuals
  instead of in-sample ones. Normal, K = 5: −0.02…−0.09 NLL on four of six
  datasets against the 1.3.0 default, −0.08…−0.30 on five of six when only
  30 validation rows exist. Loses on a heavy-tailed target on small data
  (protein: +0.07 at 490 training rows). `LogNormal`: −0.005…−0.124 on
  four datasets, with the scale ceiling above keeping the means sane (see
  `docs/known_limitations.md` 58). Fit time 2.4–6.6× the default staged
  fit at K = 5 (median 5.6×), 0.9–5.0× at K = 3 (median 3.2×). Training-time only: the model format is unchanged.
- **`hyper_opt::default_search_space`, `default_histogram_model`,
  `default_round_cap`, `cv_early_stopping_rounds`** — a measured default
  search space (learning rate, depth, `min_samples_leaf`, `col_sample`;
  `n_estimators` is not searched) and the builder that goes with it. Run
  end to end (30 trials, 5 folds, five datasets × 8 seeds) it beat the
  1.2.0 module example by 0.093 held-out NLL (37 of 40 searches) and the
  untuned default by 0.077 (36 of 40); on LogNormal, Poisson and
  NegativeBinomial by 0.045 (31 of 40) and 0.037 (34 of 40). Those figures
  are for four keys; the shipped space also searches `max_bins` (8–255,
  log), which was a further 0.009 better at the same trial budget (41 of
  64 searches; 0.03–0.05 on one smooth target, noise elsewhere).
- `NGBoost::base_learner()` (read access to the learner's settings).
- `hyper_opt::default_search_space_with_crossfit()` — the default space
  plus `staged_crossfit_folds` ∈ {0, 5}, for heads with a staged fit:
  −0.043 held-out NLL against the default space at the same trial count
  (42 of 56 searches), at several times the search time.
  `default_histogram_model` reads the key.
- `HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF`,
  `learners::adaptive_min_samples_leaf`, `effective_min_samples_leaf` on
  the three tree learners.
- `Distribution::scale_ceiling` (a hook; `None` except for `LogNormal`)
  and `NGBoost::pred_param_unbounded`.
- `NGBoost::target_max` (the largest target seen in training, stored with
  the model), `Distribution::winsorized_mean` (E[min(Y, cap)]; `LogNormal`)
  and `Distribution::predict_capped`; `SerializedNGBoost::from_bincode`.

### Fixed

- Two rustdoc warnings (`dist::hurdle`, `evaluation`) and two unused
  imports in `examples/`.

### Measured, not changed

- The `Gamma` head's occasional mean blow-ups (test RMSE 114 against a
  median of 2.9) were single-row leaves driving the rate to zero; they do
  not occur with the new leaf default (60 fits). The survival wrapper's
  `LogNormal` and `HurdleLogNormal` were sane on the dataset that broke
  `LogNormal`. One hurdle failure on a constructed floor-atom target led
  to the bounded `HurdleLogNormal` prediction above.
- A change in the 12th digit of the training targets moves a single fit's
  held-out NLL by about 0.02 and a 10-seed comparison on one dataset by
  about 0.007: treat single-dataset differences below 0.02 between two
  versions of this crate as noise (`docs/known_limitations.md` 67).
- Cross-fitted residuals (`staged_crossfit_folds`) were tested as the
  default of the staged fit and stay opt-in. Always on fails on small
  data (single datasets +0.1 … +0.3 NLL below 40 validation rows). From
  40 validation rows up it is a consistent gain on all five staged heads
  — −0.045 NLL, 1543 of 1900 fits, RMSE −1 %, confirmed on a second set
  of seeds; 3 folds give −0.043 — for 3× (3 folds) to 5–6× (5 folds) the
  fit time (`docs/known_limitations.md` 71).
- The count heads were measured on real low-count data for the first time
  (14 019 football matches with basic features — odds, form, Elo, no
  feature engineering — and three public count sets). On goals Poisson
  beats NegativeBinomial by 0.001–0.007 NLL and `CountModel::fit_auto`
  chooses it; on over-dispersed counts NegativeBinomial wins by 0.02–1.5.
  The 1.3.0 leaf default is −0.002 on these data (never worse); a tuner
  search adds nothing on goals; the staged NegativeBinomial fit with
  cross-fitted residuals failed its adoption rule and stays opt-in
  (`docs/known_limitations.md` 72).

### Saved models

`SerializedNGBoost` gained one trailing field, `target_max`.

- **Files written by 1.2.0 and earlier load** through `load_model` and
  through the new `SerializedNGBoost::from_bincode`, with `target_max =
  None`. A direct `bincode::deserialize::<SerializedNGBoost>` on such bytes
  now fails (bincode is positional and the bytes end one field early):
  switch that call to `from_bincode`. JSON and other self-describing
  formats need no change.
- **Files written by 1.3.0 load in 1.2.0**, which ignores the trailing
  field.
- Old models predict identically with one exception: a `LogNormal`
  model's predictions now pass through the scale ceiling, which changes
  rows whose σ was above four times the model's baseline
  (`pred_param_unbounded` returns the old values). A `HurdleLogNormal`
  model saved before 1.3.0 has no stored maximum and keeps its plain mean
  until it is refitted.

The leaf size, the cross-fit option and the search helpers only affect
training.

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
