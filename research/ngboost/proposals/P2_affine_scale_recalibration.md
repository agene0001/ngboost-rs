# P2 — Post-hoc affine recalibration of the log-scale (validation-fitted)

**Observed gap.** NGBoost(Normal, LogScore), early-stopped on validation
NLL, is systematically overconfident out of sample on every dataset tested
(R1–R3): √(mean err²/mean σ̂²) = 1.09–1.30, 90 % intervals cover 84–89 %,
50 % intervals cover 39–47 %. Persists at lr 0.01 and at n = 12 000.
Mechanism (hypothesis, consistent with the data): log σ trees are fitted to
½(1 − z²) with in-sample residuals of a μ fitted to the same rows, so
E[err²] < σ² (leverage) and σ̂ is biased low; the σ trees' own optimism adds
to it. Early stopping on NLL does not remove the bias because both heads
share one stopping point.

**Existing remedy.** `calibrate_uncertainty` multiplies σ by
√(mean err²/mean σ̂²) (moment matching). On het_b it over-corrects
(cov90 0.948, NLL +0.017): the ratio of means is dominated by large-σ̂
rows and is not the NLL-optimal factor.

**Proposal.** Fit, on the validation split, the affine map
log σ' = a + b·log σ̂ by minimizing the validation score:
f(a,b) = Σ_i [a + b ℓ_i + ½ r_i² e^{−2(a + b ℓ_i)}] (Normal/LogScore, with
ℓ_i = log σ̂_i, r_i = y_i − μ̂_i), which is convex in (a, b) (each term is
u + ½r²e^{−2u} with u affine). b = 1 reduces to the NLL-optimal common factor
c² = mean(z²). In the library the objective is the generic
`Scorable::<S>::total_score`, so the same head applies to any distribution
whose column is a log-scale (Laplace, LogNormal, Cauchy, StudentT).

**Persistence without a format change.** p_col' = (a + b·init) +
Σ_t f_t·(b·T_t(x)): shift `init_params[col]` and scale every column-`col`
learner's leaves by b (`TrainedBaseLearner::scale_output`). Saved models
(bincode, not self-describing — any new field would break old files) keep
their layout; staged predictions and `partial_fit` are automatically
consistent.

**Falsifiable predictions.** (1) affine ≤ MLE factor ≤ moment factor in
validation NLL by construction (nesting); on test NLL affine should be ≥ as
good as both on every dataset, and strictly better on het_b. (2) RMSE
unchanged (μ untouched). (3) cov90 → ≈ 0.90 everywhere. (4) The library's
FD-Newton must agree with the closed-form Newton to < 1e-3 in test NLL.
Reject if affine loses to base on any dataset in mean over ≥ 5 seeds.

**Results.** R4 (10 seeds, lr 0.05): affine Δ test NLL vs base: het_a
−0.037, het_b −0.018 (moment: +0.017), homo −0.023, tnoise −0.059, kin8nm
−0.031, protein −0.033 — 9–10/10 seed wins on all six; cov90 0.90–0.92;
RMSE identical. R5 (lr 0.01, 5 seeds): −0.020 … −0.029, 5/5 wins on five
datasets, 4/5 on protein, ≥ moment/MLE everywhere. R6 (n = 12k): all three
recalibrations converge (Δ −0.003 … −0.051), affine never worse than base.
Predictions 1–3 confirmed; 4 → R7.

**Code.** `src/learners/mod.rs`: `TrainedBaseLearner::scale_output`
(default false; implemented for stump, ridge, histogram tree, exact tree,
arena tree, serializable enum). `src/ngboost.rs`: `calibrate_param_affine`,
`calibrate_scale_affine`, `minimize_2d`; `NGBRegressor::calibrate_scale_affine`.
Tests: `tests/test_research_options.rs`.
