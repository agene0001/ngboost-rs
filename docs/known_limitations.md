# Known limitations and negative results (measured, 2026-11-09)

These are things that were tried or measured and did **not** work, or that
remain open. They are recorded so they are not re-tried without new
evidence. Numbers are held-out test values; runs are in
`research/ngboost/reports/` (commands in its README), the reasoning in
`research/ngboost/research_log.md`.

## Count heads (Poisson, NegativeBinomial)

1. **Post-hoc rate correction is not worth a default.** Shifting or
   affinely re-mapping `log λ` on a validation block (closed-form
   λ̂·Σy_val/Σλ̂_val, or `calibrate_param_affine(col = 0)`) helped only on
   the most biased configuration (−0.015 NLL) and was neutral-to-slightly
   negative elsewhere (+0.001…+0.003); with the Fisher-weighted default it
   adds nothing (R12–R14). Unlike the Normal σ head, a rate shift moves the
   mean, and the mean is already what boosting fits.
2. **Over-dispersion cannot be fixed by any β or recalibration.** On
   Gamma–Poisson data (Var = μ + μ²/4) the Poisson head's dispersion index
   √mean((y−λ̂)²/λ̂) stayed ≈ 2.1 for β ∈ {0, 0.25, 0.5, 1} and for every
   post-hoc correction; 90 % intervals covered 63 %. Only the
   `NegativeBinomial` head closes the gap (NLL 3.84 → 3.04, coverage 0.87;
   oracle 2.88). If your holdout dispersion index under Poisson is clearly
   above 1, switch heads — do not tune β.
3. **A residual total-count bias remains on test data — it is variance,
   not a training bias (resolved, Cycle 8).** With β = 1 the *training*
   total is exact (Σy/Σλ̂ = 1.000). The test ratio exceeds the oracle's by
   +0.03 at lr 0.05 / n = 3k, +0.017 at lr 0.01 and +0.007 at n = 12k: it
   shrinks with smaller learning rate and more data. Neither a per-round
   intercept step (`refit_intercept`) nor a validation ratio removes it.
   If calibrated totals matter, use lr ≤ 0.01.
4. **NegBin's dispersion converges slowly on truly Poisson data.** r̂ stalls
   at 20–200 after early stopping instead of → ∞ (the log-r Fisher entry is
   floored at 1e-2 because it vanishes like μ²/(2r²)); the cost is
   ≤ 0.005 NLL vs the Poisson head. Where r is identifiable it is recovered
   (r̂ 3.7–4.0 for r = 4; 10–13 for r = 10).

## Fisher-weighted base learners (β)

5. **β = 1 is not a universal default.** For Normal it gains on Gaussian-like
   heteroscedastic targets (het_b −0.10, kin8nm −0.04) but regresses on a
   bounded, skewed target (protein: +0.19 NLL, +4 % RMSE, 5× seed variance),
   and **selecting β by validation NLL does not protect against that**
   (+0.049, 3/10). It is default only for Poisson and NegativeBinomial.
6. **β has no effect, by construction, on Exponential and HalfNormal**
   (constant Fisher) and **is not applied to Gamma, Weibull, Categorical,
   MultivariateNormal** (non-diagonal Fisher; the weighted path is
   diagonal-only).

## Scale recalibration (P2)

7. **Needs ≥ 40 validation rows; the slope needs ≥ 300.** With 20 rows even
   the one-parameter shift is a coin flip; the slope is unstable on
   heavy-tailed residuals below a few hundred rows (R9). The defaults
   enforce these thresholds.
8. **Not applied without validation data** — training residuals would move
   σ the wrong way — and **not meaningful for non-log-scale columns**
   (Gamma's log-rate, Weibull) or for `NGBSurvival`.
9. **Random validation splits on time series** make both early stopping and
   the calibration too optimistic; use `ValidationSplit::Chronological` or
   an explicit chronological block. *Measured in Cycle 16 (R60): across 7
   synthetic (head, series) pairs a random split was worse in 5 (+0.01…+0.14
   NLL, 1–4/10 seeds) and about even in 2, and worse on co2 (0/5 origins).
   It was better, noisily, on the two small real series (30–70 validation
   rows).*

## Tried and rejected (Cycle 7)

12. **In-training leverage correction for log σ does not work.** σ̂ is
    nearly right on the training rows (in-sample z² 0.84–0.97) and the
    tractable hat-diagonal of the μ trees is only 0.015–0.035, versus an
    out-of-sample z² of 1.23–1.70: the optimism is adaptive-search degrees
    of freedom, invisible to leaf-size leverage. Use validation-fitted
    recalibration (the default).
13. **Capping the Fisher weights (`fisher_weight_cap` κ) does not make
    β = 1 safe.** κ = 3 still costs +3 % RMSE on protein and gives back a
    third of the gains elsewhere (R18).
14. **β = 1's protein regression is caused by a bimodal target**, not by
    Normal misspecification in general (the misspecified Normal head gains
    from β on truly log-normal data, R19). Protein's RMSD has modes at 1–3
    and 11–16: the 1/σ̂² weights give the tight low mode 58 % of the μ-tree
    weight from 40 % of rows; the high mode (36 % of rows, 19 % of weight)
    is under-fit (RMSE 5.83 → 6.29) and carries the whole loss, and the
    lowest-σ̂ decile loses +1.26 NLL per row from over-confidence (R25,
    `examples/protein_probe.rs`). Expect the same on any conditionally
    multimodal target. **Remedy (Cycle 9): the `NormalMixture2` head** —
    protein NLL 2.885 → 2.424 and CRPS −0.096 vs the calibrated Normal head
    (`docs/normal_mixture.md`).
17. **The per-round intercept step (`refit_intercept`, opt-in) does not help
    Normal, Poisson (β = 1) or NegativeBinomial** (Normal: +0.02…+0.09 NLL,
    because in-sample σ̂ is slightly wide and the step narrows it; count
    heads: neutral since Newton leaves already keep the training total). It
    does fix Bernoulli's base-rate under-prediction (log loss never worse,
    R23/R26), but the β = 1 default fixes the same bias with larger gains.
15. **Bernoulli's unweighted head under-predicts the positive rate**
    (train mean p̂/mean y 0.94 / 0.90 / 0.85 at 12 % / 6 % / 2–3 %
    positives, also at lr 0.01). Fixed by the β = 1 default since 1.2.0
    (R20/R23/R26); `fisher_weight_power = Some(0.0)` restores the old head,
    and `refit_intercept = Some(true)` fixes the base rate alone.
16. **LogNormal head point predictions:** `predict()` returns the mean
    exp(μ+σ²/2), which exploded (RMSE 1 872) on one protein seed with an
    over-estimated σ; prefer the median for point forecasts with that head.

## Cycle 9

18. **Bernoulli β = 1 default confirmed on real data** (sprint CVD, flchain
    death, support2 in-hospital death): the old unweighted head under-predicts
    the positive rate (train 0.87 on sprint, 0.96 on support2) and never beats
    the default on flchain/support2 (0/10, 0/5), costing +0.002…+0.009 log
    loss; sprint is a tie (R27).
19. **`NormalMixture2` costs up to ≈ 0.04 NLL on unimodal data** and its mean
    is a weaker point forecast on bimodal data (protein RMSE +5 %), so it is
    a modelling choice, not a default.
20. **Categorical K > 2:** diagonal-Fisher weights + a per-round intercept
    step are the default since 1.2.0 (user decision): never worse on 3
    synthetic + 4 informative real tasks (see 25), −0.032 (10/10) on support2 disease class, a
    tie on flchain cause of death (R28, R31). Evidence is two real tasks;
    `fisher_weight_power = Some(0.0)` and `refit_intercept = Some(false)`
    restore the upstream head.

## Cycle 10

21. **Unguarded CRPS training of the mixture head was worse on every metric,
    including CRPS** (0/10 seeds on all six datasets, K = 2 and 3), 6× slower,
    and collapsed a component (one test NLL of 6.8·10⁶). The full closed-form
    CRPS metric (Cycle 11) did not fix it (R39); the step guard (Cycle 12)
    did — see 31.
22. **CRPS training of NegativeBinomial** helps CRPS/RMSE only on strongly
    over-dispersed counts and costs NLL everywhere — not a default.
23. **Mixture point forecasts:** the mean is the RMSE-optimal summary, the
    median the MAE-optimal one; the mode is worst. On protein the
    single-Normal head still has the best RMSE.
24. **Categorical upstream natural gradient overflowed on separable data**
    (1/p with p → 0; "training diverged" on wine). Fixed: probabilities are
    floored at 1e-6 inside the solve (inactive unless a row saturates).
    Wine itself (178 rows, separable in 1–2 rounds) is too small to rank
    heads.
25. **Multiclass default confirmed on two more real tasks:** support2 disease
    group (8 classes) −0.082 log loss (10/10), digits (10 classes) −0.310
    (10/10), accuracy 87.0 → 96.4 %.

## Cycle 11

26. **Mixture σ blow-up (fixed):** without a guard a single tail row in a
    single-row leaf could drive a component's σ to 2·10⁸ in one round (full
    protein, K = 3). The log σ natural-gradient step is now capped at 5 per
    row; no blow-ups since, and better NLL on full protein (R37b, R38).
27. **A broad curvature guard (damping every row with r_k > π_k) was
    rejected:** it removed the blow-up but cost +0.03…+0.12 NLL by slowing
    every component's fit.
28. **Choosing K by validation NLL** reuses the early-stopping/calibration
    validation set; it matched or beat every fixed K on 7 datasets, but the
    selected model's validation NLL is optimistic by construction.

## Cycle 12

29. **Warm-starting larger mixtures from smaller ones is not a cheaper route
    to the same model.** Grown mixtures keep the Normal head's conditional
    mean: 46 % of the cost, but +0.17…+0.56 NLL on multimodal targets. As
    extra *candidates* (`KSearch::Both`, default) they are free of downside
    and help unimodal targets (−0.07…−0.10). Misfit-placed growth was worse.
30. **Continuous mixtures collapse onto point masses** (exactly repeated target
    values; 17 zero costs in support2 → NaN NLL). Remove atoms or model them
    separately.
31. **CRPS training (now guarded) is target-shape dependent:** better than
    LogScore on unimodal data, worse on multimodal data.
32. **"Real multimodal" targets here are censored follow-up times** treated as
    plain regression targets — they demonstrate multimodality, not survival
    modelling. No other real multimodal tabular data is available locally
    without a download (the files in ~/Downloads were not used).

## Cycle 13

33. **The joint Normal fit under-fits the mean.** Its early stopping fires
    when σ, fitted to shrinking in-sample residuals, turns overconfident.
    The validation-RMSE optimum comes 3–7× later. Warm-grown mixtures won on
    unimodal data for this reason, not because of shape. The fix,
    `fit_two_stage`, is **opt-in**: it costs ≈ 1.3–3.7× the tree fits, needs
    validation data, and gains nothing when the joint mean has already
    converged (docs/normal_head.md).
34. **The staged fit needs its FGLS step.** Mean-then-scale without
    re-weighting the mean by 1/σ̂² loses where σ shares the mean's features
    (het_b +0.049, 2/10). A joint "polish" after the stages does not fix
    that.
35. **Per-parameter line search does not work** (`per_param_line_search`,
    research option, +0.01…+0.10 NLL). The binary search's
    "largest non-increasing step" overshoots a single coordinate. Early
    stopping on the calibrated validation NLL helps, but only ≈ 40 % as much
    as the staged fit.
36. **Normal/LogNormal/NormalFixedMean CRPS metric was σ× too large**
    (inherited from upstream). CRPS-trained Normal fits blew up (NLL up to
    10³³). Fixed; this is a deliberate divergence from upstream
    (`NGBOOST_CRPS_METRIC_LEGACY=1` restores it). Other heads' CRPS metrics
    were not audited (Laplace's is dimensionally correct).
37. **The CRPS-trained Normal has poor NLL on multimodal targets** (flchain
    13.8 vs 8.4). CRPS barely penalises thin tails.
38. **Log-σ step guard for Normal (default, cap 25):** a no-op unless
    |z| > 7.1. Cap 5 was too tight (+0.02 on het_a, tnoise).
39. **Hurdle heads:** the atom is at exactly 0; the NLL is w.r.t. a mixed
    measure (compare by CRPS); LogScore only; `HurdleLogNormal` needs y ≥ 0.
    A small atom (1.6 % zeros) gains little by itself — the continuous part
    matters more.
40. **CRPS-trained mixture search:** better CRPS on unimodal targets, mixed
    NLL, worse on multimodal targets, up to 3.4× the fit time. Warm
    candidates add nothing under CRPS.
41. **Test-helper RNG quirk.** The `Lcg::next` used by several existing test
    files returns values in [0, 0.5), not [0, 1). Their synthetic features
    cover half the intended range and the "Gaussian" noise is bimodal. This
    is harmless for what those tests check, but not representative data
    (`tests/test_cycle13.rs` uses a corrected generator).

## Cycle 14

42. **The staged fit is now the default** for Normal, LogNormal and StudentT
    (LogScore, early stopping and validation data present; opt out with
    `staged_fit = Some(false)`). It is **not** the default for Laplace:
    large gains on synthetic data but +0.03…+0.25 NLL on three real
    skewed or multimodal targets. TFixedDf and Cauchy were not measured.
    The fit takes ≈ 1.3–3.7× the tree fits and changes results relative
    to upstream NGBoost (deliberate divergence).
43. **The joint StudentT fit is fragile.** It used to abort the process
    with a statrs panic on a NaN cdf argument; the t cdf is now NaN-safe.
    *(Corrected in Cycle 15: the "divergences" counted here are not NaN
    fits. Their NLL is finite; see item 47.)*
44. **CRPS metrics of HalfNormal, Gamma, Weibull, StudentT, TFixedDf,
    TFixedDfFixedVar and Poisson were wrong** (upstream's E_Y[g gᵀ],
    with the wrong units). Fixed to 2∫∇F∇Fᵀ (docs/crps_metrics.md); this
    also removes the "diverging Weibull CRPS" case that Tikhonov
    regularisation was documented to rescue.
45. **Hurdle CRPS training** is implemented and verified but loses to
    LogScore training (R56). LogScore is the recommendation.
46. **Correction:** the Cycle 13 statement that σ caps the Normal's joint
    line-search scale at 1 (vs 2 for μ-only boosting) was a
    floating-point rounding artifact of the binary line search; the
    under-fitted mean stands on the traces and decomposition.

## Cycle 15

47. **StudentT can predict ν ≤ 1 on some rows.** The t mean is then
    undefined (RMSE NaN) and the CRPS undefined or, for ν just above 1,
    astronomically large (> 10³), while NLL stays finite. This happened in
    15 of 80 joint fits and 1 of 80 staged fits (R54c+e). There is no ν
    floor. A floor at ν > 1 (or > 2 for a finite variance) would be cheap;
    it was not measured. *(Cycle 18: measured; `StudentTFiniteVar` (ν > 2.1)
    is the bounded head, see 59. `StudentT` itself is unchanged.)*
48. **CRPS-trained heads are overconfident relative to LogScore** where
    the shape is misspecified: with the fixed metric a CRPS Weibull wins
    CRPS on every seed (30/30) but loses NLL (median +0.07…+0.10, 5/30
    wins) with 90 % coverage 0.83–0.89 (R58). Gamma ties. Train on LogScore unless CRPS
    is the target metric.
49. **The legacy switches `NGBOOST_NORMAL_GUARD=0` and
    `NGBOOST_CRPS_METRIC_LEGACY=1` are process-wide** (read once), not
    per model. They exist for A/B comparison; the struct fields cover
    the other changed defaults per model. Bit-for-bit 1.1.2 parity with
    all six set is checked by `tests/test_upstream_parity.rs`, which only
    runs on macOS aarch64 with Accelerate (where its hashes were
    recorded).

## Cycle 16 (count heads, Normal, time series)

50. **NegativeBinomial dispersion recalibration needs ≥ 300 validation
    rows.** With 300+ rows the default `log r ← a + b·log r` map never lost
    (R63: −0.002…−0.026 NLL, 7–10/10). A shift-only map on 100 rows lost on
    nb_low and pois_nl (+0.004, 3/10): r is weakly identified there.
    Below 300 rows NB is left uncalibrated
    (`Distribution::auto_calibration_min_rows`).
51. **The staged NegativeBinomial fit is opt-in only.** It improves the
    mean (CRPS −0.06…−0.14 on over-dispersed data) but fits r to the
    shrunken in-sample residuals, so NB alone is overconfident (nb_over
    NLL +0.06, 1/10; cov90 0.79). With the dispersion recalibration it
    matches the recalibration alone on NLL, at 2–3× the cost. On the
    300-row sunspots series it lost +1.05 NLL (0/5). With Fisher-weighted
    learners (NB's default β = 1), stage 3 now divides out the learner's
    own weight so the leaf is the FGLS Newton step; uncorrected F² weights
    gave the same picture (R59r).
52. **On time series the staged Normal fit is NLL-neutral against joint.**
    Synthetic lag-feature series at n = 600–4000: −0.04…+0.07 NLL, mostly
    within noise, with CRPS usually better. On the small elnino series
    (≈ 500 training rows) it lost on 5/5 origins (+0.02…+0.52). On iid data
    the staged fit still wins with only 60 validation rows (R62), so
    validation size alone is not the cause. Not enough evidence to change
    the default. *Cycle 17 (R65, 10 origins): elnino offers no mean gain
    (test RMSE staged ≈ joint, 4/10). The staged σ is sharper, and half the
    loss is one shock window (≈ the 1982–83 El Niño, +1.05 there; median
    +0.04). This is a sharpness-vs-shock trade-off, not a bug. Staged
    validation NLL was lower on only 3/10 origins, so selecting by it would
    not help.*
53. **A post-hoc location shift on the recent block (`loccal`) is not
    worth an option.** Normal: within ±0.015. Poisson on a drifting series:
    −0.076 mean, but median −0.026 (7/10) with a large se. NB: within
    ±0.003.
54. **Refitting on train + validation is a workflow choice, not a library
    default.** `NGBoost::refit_full` (Cycle 17) does it with the kept round
    count and the carried-over recalibration. Scaling the rounds by the
    extra rows was not better (R67). Poisson −0.01 / −0.05 (7–8/10); NB
    −0.002…−0.028; Normal joint neutral (R61, R67). Staged fits are refused:
    their per-stage round counts are not replayed. A deserialized model does
    not remember that it was staged.

## Cycle 17

55. **The staged fit depends on the scale recalibration.** Without it the
    staged σ is far too narrow: n = 3000 +0.04…+0.21 NLL on 4/5 datasets;
    n = 700 +0.8…+14 (cov90 0.42–0.67); 30 validation rows +0.11…+0.21 on
    3/5 (R65–R66). The default (`staged_fit = None`) therefore stages only
    when the recalibration runs (`auto_calibrate_scale`, ≥ 40 validation
    rows). `staged_fit = Some(true)` with the calibration off reproduces the
    hazard. *(Cycle 18: the cause is the in-sample location residuals;
    `staged_crossfit_folds` removes it, opt-in, see 58.)*
56. **`CountModel::fit_auto` tests predictive, not structural,
    over-dispersion.** λ̂ is out of sample, so estimation error alone can
    push z over 1.645 on true-Poisson data with hard-to-estimate rates
    (switched on 6–9/10 seeds there). NB won on those, by −0.006 and
    −0.014. It needs explicit validation rows; it costs a second fit only
    when it switches.

## Cycle 18 (1.3.0: leaf size, cross-fitting, new heads, search space)

57. **The `min_samples_leaf = 20` default of `HistogramLearner` is a
    large-leaf prior that small data cannot afford.** At 210 training rows
    it still won on all six datasets (−0.012…−0.12 NLL). At about 100
    training rows it was a wash on NLL (mean −0.015; +0.05…+0.07 on two
    datasets, n.s.) and lost RMSE on one (het_b +0.29, 3.9 se), where leaf
    5–10 was better — hence the adaptive rule of item 62. It was measured
    on unweighted fits at depth 3 only: not with sample weights and not on
    the survival wrappers, which inherit it through
    `default_base_learner()`. On the
    multiclass, hurdle and mixture heads the effect is small (−0.011…+0.001
    NLL in 12 cases, none significantly worse). The gain comes from the
    location trees; a separate leaf size per parameter is not exposed.
58. **Cross-fitted staged residuals (`staged_crossfit_folds`) lose on a
    heavy-tailed target on small data.** Protein, Normal head: +0.07 NLL
    (4/10 seeds) at 490 training rows, +0.09 with 30 validation rows,
    against wins on the other five datasets in both settings. With leaf 1
    the 490-row loss was +0.23, so part of it was small scale leaves; the
    rest sits in four seeds whose validation NLL improved while test NLL
    worsened — a sharper σ selected on 70 validation rows and punished by
    tail rows. Checked row by row (Cycle 20): 5 of 1410 test rows carry the
    whole +0.069 — targets in the far mode of protein's two-humped
    distribution (14–21 where the fit predicts 2–6), which the sharper
    cross-fitted σ puts at |z| 5–10 under a Gaussian tail. The median row
    is 0.010 *better*. With `StudentTFiniteVar` the loss is gone (+0.004,
    se 0.019) and the default fit is 0.04 better than Normal's: on a
    small sample with a second mode or heavy tails, change the head
    before turning on cross-fitting. Opt-in for that reason.
    **LogNormal means exploded under cross-fitting until the scale
    ceiling of item 63.** On kin8nm (a bounded target; its logarithm has a
    heavy lower tail) NLL improved by 0.12 on 10/10 seeds while the
    predictive mean exp(μ + σ²/2) blew up (RMSE above 0.2 on 9 of 10
    seeds, up to 10²⁵; CRPS likewise on some): the honest log-scale
    residuals contain a few very large values, the additive scale trees
    isolate those rows and fit σ ≈ |r|, and the mean is exponential in σ².
    The default staged LogNormal fit showed the same effect mildly (RMSE
    0.30–0.57 against 0.15 on 4 of 10 seeds). With the ceiling: −0.124 NLL
    and RMSE 0.142 on kin8nm; true log-normal data, a gamma target and a
    real cost target −0.018, −0.005 and −0.007, means unaffected.
    GammaMean: −0.005…−0.084 NLL on three datasets, means unaffected.
    StudentTFiniteVar: −0.048, −0.037, −0.040 NLL on het_a, tnoise, kin8nm
    (9–10 of 10 seeds), −0.008 (se 0.012) on protein; RMSE unchanged or
    better (protein −0.05), 90 % coverage 0.93 → 0.90 on the two synthetic
    sets where the default was too wide.
    Folds are random or, with `ValidationSplit::Chronological`, contiguous
    blocks; on autocorrelated series neither is leak-free for the location,
    though only the residuals fed to the scale stage are affected.
59. **`StudentTFiniteVar` bounds ν below, not the tails above.** ν > 2.1
    keeps the variance finite (≤ 21 σ²) and removed every undefined-mean /
    infinite-variance row in 80 staged fits at no NLL cost on average
    (−0.001; worst dataset, protein, +0.019). That average is not sharper
    than ±0.01: an arithmetically equivalent implementation (the chain
    factor computed as e/(2.1 + e) instead of (ν − 2.1)/ν) changed
    individual fits by up to 0.17 NLL and gave +0.010 on average, +0.065
    on protein (`reports/O3_fvulp_*.csv`). The shipped code is the variant
    that was measured first and reproduces those 80 fits digit for digit.
    Data whose best t fit really has ν < 2.1 is fitted slightly worse than
    by `StudentT`. A bound of exactly
    1 or 2 does not work: it is attained in floating point. The bound is a
    constant of the type, not a model option.
60. **`GammaMean` was compared with `Gamma`, not with LogNormal or a
    tuned alternative,** on six datasets (three true-Gamma synthetic). Its
    staged default needs ≥ 300 validation rows (the shape recalibration);
    below that it is a joint fit, which beat `Gamma` on five of six
    datasets (+0.014 on the sixth). The existing `Gamma` head showed
    single-seed mean blow-ups (RMSE 114 against a median 2.9); that defect
    is still there and has no regression test. CRPS training of `GammaMean`
    reuses `Gamma`'s finite-difference CRPS gradient and is tested for
    correctness only, not benchmarked.
61. **The `hyper_opt` default search space was derived on one head**
    (and checked end to end on three more, item 65).
    Normal, LogScore, six datasets of at most 12 000 rows, trials scored on
    a single 10 % holdout rather than the 5-fold default. The
    learning-rate-scaled fold patience makes searches at learning rates
    below 0.05 slower (up to 4× the patience window per fold). TPE itself
    is unchanged: it beat random search by 0.02–0.05 NLL at 40 trials, a
    random startup phase made no reliable difference, and scoring trials on
    the early-stopping fold selected as well as an untouched fold.

## Cycle 19 (1.3.0: adaptive leaf, exact learners, LogNormal ceiling, tuner end to end)

62. **The adaptive leaf rule helps one kind of small dataset.** A tree
    fitted on n < 200 rows uses a leaf of n/10 instead of 20. Against a
    fixed 20 (Normal head, six datasets × 20 seeds): pooled RMSE −5.0 %
    at 70 training rows, −1.7 % at 105, −1.2 % at 140, NLL lower on 207 of
    360 fits. Most of it is one dataset whose mean depends strongly on one
    feature (−4…−25 % RMSE); the other five are within noise both ways
    (largest adverse: +2.3 % RMSE, 1.4 se). On the ten seeds that had not
    been used to propose the rule, the NLL gain at 105 rows was not
    reproduced (+0.005). Mean NLL at these sizes is dominated by a few
    diverging fits (protein, se 0.3) and was not used as evidence. "Rows"
    means what the learner is handed: after the validation split, a
    minibatch, and four fifths of the data inside `hyper_opt` or cross-fit
    folds — so fold fits on fewer than 250 rows use a smaller leaf than
    the final fit. Divisor and cap were stated in advance, not searched.
    Only the histogram learner was measured below 200 rows.
    On count and classification heads (Cycle 20; Poisson,
    NegativeBinomial, Bernoulli, Categorical K = 3; nine datasets × 20
    seeds) the rule is a trade, not a general gain: NLL lower on 280 of
    540 fits. A strong nonlinear rate gains a lot (pois_nl −0.60 and
    −0.25 NLL at 70 and 105 rows); a weak or linear signal loses a little
    (pois_low +0.048 and +0.044, about 3 se; bern_lin +0.026 at 105
    rows); the rest is noise, and everything vanishes by 140 rows. With
    sample weights the Normal result repeats (pooled RMSE −5.0 % at 70
    rows, nothing at 105–140), and the leaf-20 default itself holds
    (lower NLL than leaf 1 on 45 of 60 weighted fits at 2100 rows). For
    small, weak-signal count data use
    `HistogramLearner::with_params(depth, 255, 20, 2)`.
63. **The LogNormal scale ceiling is relative to a constant that the
    recalibration moves.** σ(x) ≤ 4 × exp(`init_params[scale]`): after a
    staged fit that constant is the rms residual log-scale, and at most
    1/16 of the rows can truly exceed four times it. The end-of-fit
    recalibration (log σ ← a + b·log σ) maps the constant with everything
    else, so with slope b > 1 the ceiling is tighter than that by
    (b − 1)·ln 4 and with b < 1 looser. On the four benchmark datasets it
    either repairs the mean (kin8nm) or never binds. In a constructed
    extreme (6 % of rows with 60 times the noise of the rest, a short
    fit whose recalibration stretches the scale) it binds on 6 % of the
    rows and scores 0.09–0.21 NLL better than the unclamped
    sum. A small region whose true σ is above the ceiling would be
    under-dispersed: use `pred_param_unbounded`, or `Normal` on ln y. The
    ceiling bounds σ, not the mean — with a baseline log-scale near 1 it
    still allows mean/median ratios in the hundreds. Not applied to the
    survival wrapper or to the hurdle heads. The exact leaf step for the
    log-scale, tried as an alternative, did not stop the blow-up and was
    not adopted (`proposals/O8_exact_scale_step.md`).
64. **`DecisionTreeLearner::new` and `default_tree_learner()` now differ.**
    `new(depth)` follows the library's leaf default (20, adaptive);
    `default_sklearn()`, `default_tree_learner()` and
    `arena_tree_learner()` keep scikit-learn's leaf 1, because matching
    Python NGBoost is their contract. The exact learner was measured on
    the Normal head at 210, 490 and 2100 training rows, the arena learner
    at 2100 only (protein unchanged there).
65. **The end-to-end tuner comparison is four heads and one size.**
    Normal (five datasets) and, in Cycle 20, LogNormal, Poisson and
    NegativeBinomial (five datasets): there the default search beat the
    1.2.0 example by 0.045 NLL (31 of 40) and no tuning by 0.037 (34 of
    40), was never worse than not tuning, and gained nothing on a weakly
    varying Poisson rate or beyond noise on the NegativeBinomial dataset.
    Classification heads are unmeasured. For Normal: LogScore, 2400
    training rows, 30 trials, 5 folds, five datasets × 8 seeds. The default search beat the 1.2.0 example by 0.093 NLL (37 of
    40) and no tuning by 0.077 (36 of 40), but only by 0.011–0.015 on two
    of the five datasets. Its searches took 0.7–1.8× the old ones' time on
    a loaded machine (deeper trees and more rounds on kin8nm); no clean
    timing was taken. The 1.2.0 arm is a reconstruction: that release's
    space, leaf 1 and a fold patience of 50, refitted with early stopping
    like the other arms.

## Cycle 20 (1.3.0: other heads' means, protein rows, last-bit noise, tuner on more heads)

66. **`HurdleLogNormal` bounds its point prediction, not its
    distribution.** On the dataset that broke `LogNormal` (kin8nm with a
    fifth of the targets set to zero) it was sane in 10 of 10 fits, and so
    was the survival wrapper's `LogNormal` (20 fits; largest mean/median
    ratio 2.5): both are joint fits. But on a constructed bounded target
    with a point mass at its floor — marginal log-scale near 0.9 — one
    hurdle fit in four stopped early with σ ≈ 3.4 on a few rows and a mean
    of 31 against a largest target of 1.45, and the scale ceiling of item
    63 would not have caught it. Since Cycle 21 `predict` returns
    (1 − π)·E[min(Y, c)], c the largest training target
    (`NGBoost::target_max`, stored with the model). That is a bound, not a
    repair: the fitted σ on those rows is still 3.4, intervals and
    `pred_dist(x).mean()` still reflect it, and a prediction near c on such
    a row is still poor. A genuine future target above every training
    target is under-predicted by construction (the winsorised mean is
    below the mean by E[(Y − c)⁺], small unless c sits inside the bulk).
    Models saved before 1.3.0 have no stored maximum and keep the plain
    mean. Plain `LogNormal`, the survival wrapper and `Gamma` do not use
    the bound. `Gamma`'s recorded blow-ups (RMSE 114 against a median 2.9)
    were single-row leaves driving the rate to zero and do not occur with
    the 1.3.0 leaf default (60 fits); they can with a leaf-1 learner.
    `GammaMean` never showed them.
67. **Single fits are chaotic in the last bit.** Multiplying the training
    targets by 1 ± 10⁻¹² changed the early-stopping round of 359 of 360
    Normal fits and a single fit's held-out NLL by 0.018 on average (up
    to 0.10). A 10-seed paired difference on one dataset moves by about
    0.007 (up to 0.015) under such a change; the average over six
    datasets by 0.0015. Single-dataset differences below about 0.02 in
    these documents are therefore not findings, however small their
    seed-to-seed standard error; pooled ones are. The same holds for
    comparing two releases of this crate on one dataset.
## Cycle 23 (1.3.0: `max_bins` in the default search space)

68. **`max_bins` earns its place in the search on one kind of target.**
    Searching it (8–255, log) at the same 30 trials was 0.009 NLL better
    pooled over eight dataset-head pairs (41 of 64 searches). All of that
    is kin8nm — Normal −0.026 (7 of 8), LogNormal −0.051 (8 of 8), with
    winning models at 18–19 bins — a smooth target where coarse bins act
    as a regulariser. Six pairs are inside the noise (−0.010…+0.006), and
    het_b was +0.024 (1.5 se, 3 of 8): a fifth dimension dilutes a
    30-trial search a little where it has nothing to find. The untuned
    default stays at 255 bins; no rule for choosing the bin count without
    a search was looked for. k-means bin edges were tried instead of
    quantile edges and rejected (no gain at 255 bins, and not invariant
    to rescaling a feature).

## Cycle 24 (1.3.0: cross-fitting and the head as search choices)

69. **Searching cross-fitting improves the log score, not the point
    prediction, and is expensive.** `default_search_space_with_crossfit`
    was 0.043 NLL better than the default space at 30 trials (Normal and
    LogNormal, seven dataset-head pairs, 42 of 56 searches), but RMSE did
    not move (two datasets about 1 % worse, 1.6 se) and the search took
    several times longer. Measured at 2400 training rows only: on a few
    hundred rows with a heavy-tailed target, cross-fitting with the
    Normal head loses (item 58), and whether a search catches that was
    not tested. The key does nothing on heads without a staged fit.
70. **The tuner's CV score is not a safe way to choose a head.** Picking
    between Normal and StudentTFiniteVar, or LogNormal and GammaMean, by
    the better search score was 0.007 worse than one head at the same
    total budget and 0.004 better at twice the budget (neither
    significant). It was right where the gap was large (t-distributed
    noise, −0.075) and wrong on kin8nm, where it chose GammaMean over
    LogNormal in 8 of 8 searches and lost 0.037 on test. The fold score
    ranks configurations of one head well; across heads it also reflects
    how each head's early stopping and recalibration use the scoring
    fold. No head-selection helper is provided.

71. **Cross-fitting is not the default, by cost and by small data.**
    Tested as the default of the staged fit (Cycle 25; the shipped
    default model, five heads, 19 head × dataset pairs, seeds chosen
    fresh, a second set to confirm). Always on fails: below 40 validation
    rows the default is a joint fit, the option replaces it by four
    stages stopped on 10–30 rows, and although the average improves,
    single datasets lose 0.1–0.3 NLL (Normal on kin8nm at 200 rows
    +0.116, 2.4 se; Normal on protein at 200–300 rows +0.18 … +0.26). From
    40 validation rows up it is better on 18 of 19 pairs: −0.045 NLL
    (se 0.007), 1543 of 1900 fits, RMSE −1.1 %, coverage unchanged; with
    3 folds −0.043. The exception is the Normal head on protein (+0.005,
    RMSE +0.5 % at 4.9 se). The gain decays with size (−0.03 at 6000
    rows, −0.02 at 20 000). One fit at a time it costs 2.4–6.6× the
    default with 5 folds (median 5.6×) and 0.9–5.0× with 3 (median 3.2×),
    and a tuner search pays that on every fold. One cross-fitted
    LogNormal fit in 560 predicted a mean of 901 for a target of 46 on a
    row whose σ sat just under the ceiling of item 63. Not tested:
    explicit validation sets, more than 20 000 rows, the survival
    wrapper.

72. **On a weak-signal count target the fit options are worth almost
    nothing — measured on football goals with basic features only.** The
    features were market odds, recent form and Elo, with no feature
    engineering, so this is evidence about a low-signal target, not about
    sports data with strong features: where the rate varies strongly
    (two synthetic sets) the same options are worth 0.03–0.07, and no
    default was chosen from these data. Cycle 26, 14 019
    football matches (home and away goals, means 1.5 and 1.2, variance /
    mean 1.07–1.10), random splits and a rolling origin by season:
    NegativeBinomial is 0.001–0.007 NLL *worse* than Poisson in all eight
    comparisons (item 4's slow dispersion on nearly-Poisson data);
    `CountModel::fit_auto` stays on Poisson in 219 of 232 fits. The leaf
    default against leaf 1 is −0.000 … −0.006; a 30-trial tuner search on
    the Poisson head −0.0001 (se 0.0005). The staged NegativeBinomial fit
    with cross-fitted residuals was tested as that head's default and
    failed its rule: −0.0021 on eleven real dataset rows (1.7 se, 60 % of
    fits), a gain only with strong over-dispersion or a strongly varying
    rate (−0.036, −0.027 on two synthetic sets), at 4–36× the fit time; it
    stays opt-in (item 51). For scale, the features are worth 0.05–0.07
    over a constant rate there. On over-dispersed real counts (variance /
    mean 2.1–7.9) NegativeBinomial wins by 0.02–1.5 and the leaf default
    by 0.004–0.010 (Poisson). Not tested: larger count scales, the Normal
    head on totals or margins.

## Not pursued (reasoned, not measured)

10. Noncyclical updates: Thomas et al. (2018) report no accuracy gain.
    *(Per-parameter line search was measured in Cycle 13 via the fold
    trick — item 35: worse.)*
11. *(Superseded, Cycle 10: CRPScore now exists for NegativeBinomial and
    NormalMixture<K> — see items 21–22.)*
