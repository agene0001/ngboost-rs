# Search infrastructure and search space (as of 1.2.0, read 2026-11-13)

## What `src/hyper_opt.rs` does

| Aspect | Behaviour | Where |
|---|---|---|
| Sampler | Rust `tpe` 0.3.1, **one independent optimiser per parameter** (no joint/multivariate model, no conditional parameters) | `make_optimizer` |
| Density estimators | Parzen (Gaussian kernels + one prior kernel at the range midpoint) for float/int; histogram for categorical | `tpe` crate |
| TPE settings | γ = 0.1 (best ⌈0.1·n⌉ trials are "good"), 24 candidates per ask — crate defaults, never overridden | `TpeOptimizer::new` |
| Startup | **No random startup phase.** Trial 1 is drawn from the prior kernel; from trial 2 the "good" density is built from the single best trial plus the prior. The crate's own docs suggest telling it random observations first | `hyper_opt_with_transform` |
| Transforms | float: linear or log; int: continuous range then `round()` (edge values get half the mass), optional log; categorical: index | `raw_to_value` |
| Order / RNG | specs sorted by name; all optimisers share one `StdRng(hp_seed)` | `parse_hp_specs` |
| Objective | mean over folds of `NGBoost::score` (NLL for LogScore), lower is better | `cv_with_pruning` |
| Folds | contiguous unshuffled K-fold, forward-chaining `TimeSeries`, or one 80/20 holdout (`n_folds ≤ 1`) | `CvScheme` |
| Per-fold fit | `fit_with_validation` with **the scoring fold as the validation set**: early stopping, the staged fit's four stage stops and the scale recalibration all use the rows that are then scored | `run_fold` |
| Fold patience | builder's `early_stopping_rounds`, else a fixed 50, whatever the learning rate | `CV_EARLY_STOPPING_ROUNDS` |
| Failed fits / non-finite scores | scored +∞, told to TPE as +∞, never the incumbent | `run_fold` |
| Pruning | off by default; median/percentile/threshold over running fold means. With parallel folds (the default) a trial is "pruned" only after all its folds ran: no compute is saved | `PrunerState` |
| Study stop | `max_minutes`, `plateau_patience` | config |
| Warm start | one prior (params, score) told to every optimiser before trial 1 | `warm_start` |
| Result | best params, best CV score, `opt_rounds` = mean early-stopped rounds over the best trial's folds | `HyperOptResult` |
| Model seed | `seed + trial`, spread per fold: with `minibatch_frac < 1` or `col_sample < 1` each trial is one noisy draw | `run_fold` |

The search space itself is supplied by the caller; the crate ships no
default space. The module-level example searches `learning_rate`
0.005–0.1 (log), `max_depth` 2–8 and `n_estimators` 200–1500.

Not searchable in one study: the head `D`, the score `S` and the learner
type `B` are type parameters of the builder. `staged_fit`,
`fisher_weight_power`, learner fields and everything on `NGBoostParams`
can be searched through the builder closure.

## Space used in Cycle 18 (`examples/c18_bench.rs`)

Normal head, LogScore, `HistogramLearner`, 1.2.0 defaults (staged fit +
recalibration).

| Parameter | Kind | Range | Transform | Rationale |
|---|---|---|---|---|
| `learning_rate` | numeric | 0.02–0.2 | log | below 0.02 fits get long; `lrcap` covers 0.005–0.2 separately |
| `max_depth` | numeric (int) | 2–8 | linear, rounded | library default 3 |
| `min_samples_leaf` | numeric (int) | 1–200 | log, rounded | library default 1; never varied before Cycle 18 |
| `minibatch_frac` | numeric | 0.5–1.0 | linear | row subsampling per round |
| `col_sample` | numeric | 0.4–1.0 | linear | feature subsampling per round |
| `n_estimators` | **not searched** | cap = min(6000, 100/lr) | — | with early stopping it is only a cap; searching it adds a dimension whose low values truncate low learning rates |
| early-stopping patience | **not searched** | clamp(2.5/lr, 50, 200) | — | constant *shrinkage budget* of patience rather than constant rounds |

Model *choices* (categorical, not ordinary hyperparameters), measured
separately in `--mode select`: joint fit, joint with Fisher weights
β ∈ {0.5, 1}, staged fit.

Constraints: none between parameters. Failed or non-finite fits are
recorded as NaN rows in the pool (none occurred) and as +∞ in searches.

## Shipped default space (1.3.0, `hyper_opt::default_search_space`)

Approved and implemented 2026-11-13, after the study above.

| Parameter | Kind | Range | Transform | Change from the studied space |
|---|---|---|---|---|
| `learning_rate` | numeric | 0.02–0.2 | log | none |
| `max_depth` | numeric (int) | 2–8 | linear, rounded | none |
| `min_samples_leaf` | numeric (int) | 5–200 | log, rounded | lower bound 1 → 5 |
| `col_sample` | numeric | 0.6–1.0 | linear | lower bound 0.4 → 0.6 |
| `minibatch_frac` | — | not searched (1.0) | — | dropped |
| `n_estimators` | — | `default_round_cap(lr)` = min(6000, ⌈100/lr⌉) | — | none |
| fold patience | — | `cv_early_stopping_rounds(lr)` = clamp(round(2.5/lr), 50, 200) | — | now the library rule, not only the harness's |

`default_histogram_model` is the builder; it also accepts `minibatch_frac`
and `n_estimators` keys if a caller adds them to the space.

Searched as such in Cycle 19 (`research_log.md`, A6): through
`hyper_opt_with_config` with 5 folds and 30 trials it beat the 1.2.0
example's space by 0.093 held-out NLL and the untuned default by 0.077
(five datasets × 8 seeds, Normal head).

On three more heads (A7, Cycle 20: LogNormal, Poisson, NegativeBinomial;
five datasets × 8 seeds): −0.045 against the 1.2.0 space and −0.037
against the untuned default, never worse than not tuning, no gain on a
weakly varying Poisson rate.

`max_bins` (8–255, log int) joined the shipped space in Cycle 23 (A8):
−0.009 pooled against the four-key space at equal trials, lower on 41 of
64 searches; the gain is on kin8nm (18–19 bins selected).

Cycle 24: `staged_crossfit_folds` ∈ {0, 5} as an extra key for staged
heads (`default_search_space_with_crossfit()`): −0.043 pooled, 42 of 56
searches (A9). The head is not a key and choosing it by CV score is not
recommended (A10).

