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
    it was not measured.
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
    hazard.
56. **`CountModel::fit_auto` tests predictive, not structural,
    over-dispersion.** λ̂ is out of sample, so estimation error alone can
    push z over 1.645 on true-Poisson data with hard-to-estimate rates
    (switched on 6–9/10 seeds there). NB won on those, by −0.006 and
    −0.014. It needs explicit validation rows; it costs a second fit only
    when it switches.

## Not pursued (reasoned, not measured)

10. Noncyclical updates: Thomas et al. (2018) report no accuracy gain.
    *(Per-parameter line search was measured in Cycle 13 via the fold
    trick — item 35: worse.)*
11. *(Superseded, Cycle 10: CRPScore now exists for NegativeBinomial and
    NormalMixture<K> — see items 21–22.)*
