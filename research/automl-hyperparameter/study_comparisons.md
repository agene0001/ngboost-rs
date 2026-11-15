# Study comparisons — Cycle 18 (2026-11-13)

Normal head, LogScore, `HistogramLearner`, 1.2.0 defaults (staged fit +
recalibration). Space and budget rule: `search_space.md`. Files and
commands: `reports/README.md`. All numbers are NLL; Δ is relative to the
research default (lr 0.05, depth 3, leaf 1, no subsampling); negative is
better. "Validation score" is what a `hyper_opt` fold reports. Test
numbers come from a 20 % block never used for fitting, stopping,
calibration or selection.

## Tier 2 — does tuning generalise? (`reports/A1_pool.csv`)

120 configurations per (dataset, data seed), 3 data seeds; held-out test
NLL of the configuration with the best validation score among all 120:

| dataset | default | selected | Δ (se over seeds) | wins | oracle (best test) |
|---|---|---|---|---|---|
| het_a   | 1.333 | 1.176 | −0.157 (0.039) | 3/3 | 1.147 |
| het_b   | 0.627 | 0.621 | −0.006 (0.006) | 1/3 | 0.572 |
| homo    | 1.555 | 1.498 | −0.056 (0.017) | 3/3 | 1.477 |
| tnoise  | 1.421 | 1.220 | −0.201 (0.050) | 3/3 | 1.207 |
| kin8nm  | −0.528 | −0.802 | −0.274 (0.015) | 3/3 | −0.802 |
| protein | 2.866 | 2.750 | −0.116 (0.012) | 3/3 | 2.747 |

The median random configuration is *worse* than the default on four
datasets, so this is selection, not a better prior. For scale: the
staged fit, 1.2.0's largest change, was worth −0.02…−0.15 on these
datasets (R54a).

Expected Δ on test after k random trials (resampled from the pool):

| dataset | k = 5 | k = 10 | k = 20 | k = 40 | k = 80 |
|---|---|---|---|---|---|
| het_a   | −0.092 | −0.122 | −0.136 | −0.145 | −0.154 |
| het_b   | +0.025 | −0.001 | −0.011 | −0.014 | −0.011 |
| homo    | −0.037 | −0.046 | −0.050 | −0.050 | −0.050 |
| tnoise  | −0.120 | −0.155 | −0.172 | −0.184 | −0.194 |
| kin8nm  | −0.159 | −0.195 | −0.220 | −0.239 | −0.257 |
| protein | −0.083 | −0.091 | −0.097 | −0.103 | −0.112 |

Most of the gain is there by 10–20 trials.

## Sensitivity (`reports/A1_pool_summary.txt`, section 4)

Spearman correlation of each knob with test NLL over the random
configurations (negative = larger value is better):

| dataset | learning rate | depth | min leaf | minibatch | col sample |
|---|---|---|---|---|---|
| het_a   | +0.22 | +0.41 | **−0.67** | −0.08 | −0.41 |
| het_b   | +0.38 | **+0.58** | −0.24 | −0.20 | −0.47 |
| homo    | +0.16 | +0.30 | **−0.65** | −0.05 | −0.40 |
| tnoise  | +0.19 | +0.39 | **−0.67** | −0.07 | −0.41 |
| kin8nm  | +0.21 | −0.15 | −0.51 | −0.00 | **−0.55** |
| protein | **+0.58** | −0.52 | −0.26 | −0.11 | −0.25 |

- `min_samples_leaf` is the strongest knob on three datasets, a close
  second on kin8nm, and helps on all six; the library default (1) sits in the worst tercile. Selected
  values: median 51, range 1–106. Isolated in
  `research/optimization-algorithm/` (O2).
- `col_sample`: low values hurt everywhere (few informative features);
  selected median 0.97. Not worth searching below ≈ 0.6 on such data.
- `max_depth` is the one knob whose *direction* depends on the data:
  shallower on the synthetic sets, deeper on kin8nm and protein (selected
  7–8 there; RMSE correlation −0.55 and −0.67).
- `minibatch_frac` barely matters (|ρ| ≤ 0.2).
- `learning_rate`: lower is mildly better once the round budget scales
  with it (next section).

## Round cap and patience vs learning rate (`reports/A3_lrcap_summary.txt`)

Default shape, staged fit, 3 seeds, ≤ 6000 rows. Test NLL by learning
rate under four budget policies (`capS` = min(12 000, 60/lr); `pS` =
clamp(2.5/lr, 50, 500)):

| dataset | lr | cap 500, patience 50 | cap 1500, patience 50 | capS, patience 50 | capS, patience pS |
|---|---|---|---|---|---|
| het_a  | 0.05  | 1.329 | 1.333 | 1.333 | 1.333 |
| het_a  | 0.02  | 1.444 | 1.321 | 1.321 | **1.307** |
| het_a  | 0.01  | 1.540 | 1.363 | 1.352 | 1.318 |
| het_a  | 0.005 | 1.660 | 1.503 | 1.429 | 1.341 |
| kin8nm | 0.05  | −0.498 | −0.498 | −0.498 | −0.498 |
| kin8nm | 0.02  | −0.460 | −0.491 | −0.493 | **−0.504** |
| kin8nm | 0.01  | −0.412 | −0.465 | −0.464 | −0.503 |
| kin8nm | 0.005 | −0.315 | −0.443 | −0.473 | −0.495 |
| protein | 0.05  | 2.903 | 2.906 | 2.906 | 2.906 |
| protein | 0.01  | 3.042 | 2.926 | 2.916 | 2.906 |
| protein | 0.005 | 3.053 | 3.109 | 2.917 | 2.906 |

Read across a row: a fixed patience of 50 and a low round cap both make a
low learning rate look worse than it is — by 0.01–0.04 at lr 0.01 from
patience alone (cap not binding), and by 0.05–0.3 at lr ≤ 0.01 with a cap
of 500–1500.
Read down the last column: with the budget scaled, test NLL is flat to
within ≈ 0.03 from lr 0.005 to 0.05, best around 0.02. het_b is flat
(0.615–0.627) under every policy except cap 500 at lr ≤ 0.01. The joint
fit shows the same pattern at smaller size.

## Tier 1 — sampler efficiency

**Real objective** (`reports/A4_search_real_summary.txt`; data seed 0, 8
sampler seeds, 40 trials). Best validation score, mean ± se:

| dataset | sampler | t = 10 | t = 20 | t = 40 | test NLL of incumbent at 40 |
|---|---|---|---|---|---|
| het_a  | random | 1.245 ± 0.011 | 1.223 ± 0.008 | 1.197 ± 0.007 | 1.263 ± 0.015 |
| het_a  | TPE (as in `hyper_opt`) | 1.217 ± 0.014 | 1.184 ± 0.014 | **1.152 ± 0.008** | **1.214 ± 0.012** |
| het_a  | TPE after 10 random | 1.245 ± 0.011 | 1.183 ± 0.007 | 1.161 ± 0.007 | 1.227 ± 0.016 |
| kin8nm | random | −0.743 ± 0.015 | −0.762 ± 0.013 | −0.804 ± 0.011 | −0.764 ± 0.011 |
| kin8nm | TPE | −0.730 ± 0.013 | −0.800 ± 0.005 | −0.821 ± 0.006 | −0.777 ± 0.008 |
| kin8nm | TPE after 10 random | −0.743 ± 0.015 | −0.817 ± 0.009 | **−0.835 ± 0.006** | **−0.792 ± 0.006** |
| tnoise | random | 1.375 ± 0.021 | 1.344 ± 0.018 | 1.325 ± 0.010 | 1.310 ± 0.015 |
| tnoise | TPE | 1.335 ± 0.018 | 1.307 ± 0.013 | **1.281 ± 0.009** | **1.265 ± 0.008** |
| tnoise | TPE after 10 random | 1.375 ± 0.021 | 1.307 ± 0.008 | 1.282 ± 0.008 | 1.282 ± 0.014 |

TPE beats random search at 40 trials on all three, by 0.02–0.05: 3–4
standard errors on het_a and tnoise, 1.3 on kin8nm (2.4 with the startup
phase). The advantage carries to test (0.01–0.05). A random startup phase
is not reliably better or worse (ahead on kin8nm, behind on het_a, level
on tnoise; all within ≈ 1.5 se). No failed trials; ≤ 0.3 % duplicates.

**Surrogate objective** (`reports/A4_search_surrogate_summary.txt`; k-NN
over each dataset's pool, 200 sampler seeds, 80 trials). Gap to random
search in best validation score at t = 40 (se ≈ 0.002):

| dataset | TPE | TPE after 10 | TPE after 20 | TPE after 10, γ = 0.25 |
|---|---|---|---|---|
| het_a   | −0.016 | −0.017 | −0.017 | −0.014 |
| het_b   | −0.019 | −0.021 | −0.019 | −0.013 |
| homo    | −0.005 | −0.005 | −0.004 | −0.003 |
| tnoise  | −0.020 | −0.020 | −0.022 | −0.015 |
| kin8nm  | −0.022 | −0.023 | −0.021 | −0.017 |
| protein | −0.005 | −0.006 | −0.005 | −0.004 |

Same ordering with 25× the seeds: TPE > random everywhere; startup length
is immaterial by t = 30; a wider "good" quantile (γ = 0.25) is worse and
proposes 1–6 % duplicate trials. The surrogate is smooth by construction,
which favours model-based search — it is corroboration, not the evidence.

## Does scoring on the early-stopping fold distort selection? (`reports/A1_pool_summary.txt`, sections 2–3)

| dataset | optimism of the reused score (test − score) | of the untouched score | rank corr. with test: reused | untouched | test Δ of argmin at k = 40: reused − untouched |
|---|---|---|---|---|---|
| het_a   | +0.025 | −0.041 | 0.96 | 0.91 | −0.001 |
| het_b   | −0.027 | −0.089 | 0.97 | 0.94 | +0.013 |
| homo    | +0.040 | +0.004 | 0.92 | 0.72 | +0.001 |
| tnoise  | +0.060 | −0.001 | 0.96 | 0.89 | −0.009 |
| kin8nm  | +0.047 | +0.008 | 0.98 | 0.90 | −0.005 |
| protein | +0.023 | −0.003 | 0.90 | 0.86 | −0.003 |

The reused score is optimistic by 0.02–0.06, and the optimism is weakly
related to the configuration (largest: small leaves are flattered,
ρ −0.14…−0.38). But it ranks configurations *better* than the untouched
score, because it scores the very model that is then deployed, whereas the
nested score describes a model fitted on fewer rows. Selection by the
reused score is as good or better on five of six datasets; het_b is the
exception (the leaf-1 default is flattered enough to win 2 of 3 studies).

## Choosing the fit variant (`reports/A5_select_summary.txt`)

Candidates joint (β = 0, 0.5, 1) and staged, default shape, 10 seeds.

- With the staged fit in the set, 5-fold CV picks it in 56 of 60 studies
  and a single holdout in 46 of 60; both end within 0.02 of always
  choosing staged. The staged fit is the best single choice on five
  datasets (het_b: joint β = 1 is 0.013 better).
- Among joint fits only — Cycle 3's question — 5-fold CV picks β = 0 on
  protein in 10 of 10 studies (test 2.885) where a single holdout mixes
  (2.931): −0.045 (se 0.030), which removes the regression that kept β
  from being tunable. On het_a, homo and tnoise the single holdout is the
  better selector by 0.02–0.04 (0 CV wins): it is choosing a fitted model,
  not a configuration, and the holdout score of that model is informative.

## The shipped search end to end (`reports/A6_hpo_summary.txt`, Cycle 19)

`hyper_opt_with_config`, 30 trials, 5 folds, 2400 training rows, five
datasets × 8 seeds; the winner refitted on the training pool and scored on
the untouched test block. Arms: `old` = the 1.2.0 module example (learning
rate, depth, `n_estimators` 200–1500; leaf 1; patience 50), `oldspace` =
that space on the 1.3.0 defaults, `new` = `default_search_space` with
`default_histogram_model`, `untuned` = that builder with no parameters.

| held-out NLL | new − old | new − untuned | old − untuned |
|---|---|---|---|
| het_a | −0.184 | −0.117 | +0.067 |
| het_b | −0.011 | −0.046 | −0.035 |
| kin8nm | −0.069 | −0.144 | −0.075 |
| protein | −0.015 | −0.015 | +0.000 |
| tnoise | −0.187 | −0.064 | +0.123 |
| pooled (se), searches lower of 40 | −0.093 (0.015), 37 | −0.077 (0.011), 36 | +0.016 (0.014), 22 |

`oldspace − old` is −0.040 (0.009), 32 of 40: two fifths of the gain are
the changed defaults, three fifths the space. Same sampler, trials, folds
and seeds in every arm.

## The shipped search on other heads (`reports/A7_hpo_summary.txt`, Cycle 20)

Same design as the section above; 8 seeds per dataset.

| held-out NLL | new − old | new − untuned |
|---|---|---|
| LogNormal, kin8nm | −0.082 | −0.072 |
| LogNormal, lognorm_het | −0.075 | −0.047 |
| Poisson, pois_nl | −0.056 | −0.034 |
| Poisson, pois_low | +0.001 | −0.002 |
| NegativeBinomial, nb_over | −0.015 | −0.030 |
| pooled (se), searches lower of 40 | −0.045 (0.008), 31 | −0.037 (0.006), 34 |

Never worse than not tuning; about half the Normal head's margin; no gain
where the rate barely varies.

## `max_bins` in the search (`reports/A8_bins_summary.txt`, Cycle 23)

The four-key space against the same space plus `max_bins` (8–255, log),
30 trials each, eight dataset-head pairs × 8 seeds: −0.009 (0.005) pooled,
lower on 41 of 64. kin8nm −0.026 (Normal) and −0.051 (LogNormal) at a
median of 18–19 bins; six pairs within ±0.010; het_b +0.024 (1.5 se).
Adopted under a rule written before the runs.

## Cross-fitting and the head as search choices (Cycle 24)

`reports/A9_xfit_summary.txt`: the shipped space plus
`staged_crossfit_folds` ∈ {0, 5}, staged heads, seven dataset-head pairs
× 8 seeds, 30 trials: −0.043 (0.009) pooled, lower on 42 of 56; from
−0.159 (LogNormal, kin8nm) to +0.004; the winner cross-fits in 52 of 56.
No RMSE gain. Shipped as `default_search_space_with_crossfit()`.

`reports/A10_headsel_summary.txt`: picking between two heads by their
searches' CV scores. Equal total budget (2 × 15 trials): +0.007 (0.010)
against one head at 30. Double budget: −0.004 (0.007); −0.075 on
t-distributed noise, +0.037 on kin8nm where it picked GammaMean over
LogNormal 8 of 8 times. Not recommended.

## Compute

Every fit is 0.1–5 s single-threaded at these sizes (n ≤ 12 000, ≤ 10
features); the whole cycle was ≈ 15 000 fits. A 40-trial single-holdout
search is ≈ 1–3 minutes on one core. All runs shared a saturated laptop,
so no wall-clock comparison is reported; samplers were compared at equal
trial counts, and the sampler's own cost is negligible next to a fit.
