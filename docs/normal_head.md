# The Normal head: staged fit, log-σ step guard, CRPS metric fix

Research Cycle 13 (2026-11-10). Reports R44–R48, R51–R52 in
`research/ngboost/reports/`. All comparisons are paired over the same
70/10/20 splits, 10 seeds, lr 0.05, depth-3 histogram trees, early stopping
patience 50, default scale recalibration.

## 1. Why warm-grown mixtures beat the Normal head on unimodal data

Cycle 12 found that a 2-component mixture *grown* from the fitted Normal head
beats it on unimodal targets (−0.07…−0.12 NLL). Three explanations were
possible: extra boosting rounds, non-Gaussian residuals, or something else.
Probe R44 (`--dist normprobe`) separates them:

| dataset | Normal | + same extra rounds | symmetric split | split (δ=0.8, σ×0.6) | moment-matched Normal of the split | RMSE Normal → split | μ-only boosting RMSE |
|---|---|---|---|---|---|---|---|
| kin8nm | −0.444 | −0.452 | −0.445 | **−0.512** | −0.481 | 0.173 → 0.166 | 0.152 |
| het_a | 1.524 | 1.504 | 1.520 | **1.408** | 1.407 | 1.364 → 1.177 | 1.142 |
| het_b | 0.748 | 0.745 | 0.747 | **0.725** | 0.716 | 0.977 → 0.983 | 0.984 |
| homo (σ = 1) | 1.711 | 1.707 | 1.707 | **1.626** | 1.625 | 1.440 → 1.263 | 1.147 |
| tnoise (t₄) | 1.540 | 1.528 | 1.535 | **1.442** | 1.467 | 1.413 → 1.292 | 1.224 |
| protein | 2.885 | 2.861 | 2.885 | **2.592** | 2.901 | 4.590 → 4.518 | 4.273 |

- **Not the extra rounds.** Continuing the Normal head for exactly the rounds
  the split adds (no early stopping) recovers only 4–17 % of the gain.
- **Not the shape, on Gaussian-noise data.** The Normal with the split
  mixture's own mean and variance scores the same as the mixture on het_a and
  homo (1.407 vs 1.408, 1.625 vs 1.626; on het_b it is even better). The
  shape matters only where the noise really is non-Gaussian: tnoise (t₄,
  0.025 of 0.10), kin8nm (about half) and protein (bimodal, all of it).
- **It is the mean.** The split mixture's mean is far better (homo RMSE 1.44
  → 1.26 with σ = 1). The same trees, boosting the mean alone, reach 1.15.
  The symmetric split (two identical halves) reproduces "continue the
  Normal" to 4 decimals. That control shows the reparametrisation itself does
  nothing; the asymmetric split is what lets the mean move.

**Mechanism.** Training the Normal without early stopping (`trace`) puts the
validation-RMSE optimum at 1 165–1 918 rounds, against the validation-NLL
optimum at 102–387. het_b is the exception (both near 100–280), and there
the split gains little. At the RMSE optimum the uncalibrated test NLL is 4.4
(het_a, homo, tnoise): σ has collapsed. The log σ trees fit *in-sample*
residuals, which shrink as the mean overfits the training rows. Validation
NLL therefore turns up because σ becomes overconfident, long before the mean
has converged, and early stopping freezes an under-fitted mean.

*Correction (Cycle 14).* An earlier version of this note named a second
factor: the joint line search settling at scale 1 for the Normal against 2
for μ-only boosting. That is a rounding artifact, not a mechanism. For a
least-squares tree the full-step loss at scale 2 equals the starting loss
exactly in exact arithmetic (the mirror point), so the binary search's
choice between 1 and 2 is decided by floating-point rounding.

Rejected fixes (R45):

- **Per-parameter line search** (`per_param_line_search`, research option):
  worse everywhere (+0.01…+0.10 NLL). The binary search returns the largest
  step that does not raise the loss, so a column searched on its own
  overshoots to about 2× its optimum.
- **Early stopping on the calibrated validation NLL** (monitor = NLL after
  the optimal log-σ shift): helps (−0.003…−0.068, 8–10/10), but at only
  about 40 % of the staged fit's gain.

## 2. Staged fit (default since 1.2.0 for Normal, LogNormal, StudentT)

`fit_with_validation` (and `fit` with `validation_fraction` and early
stopping) runs the staged fit automatically for these heads under LogScore,
whenever early stopping is on and there is validation data. Opt out with
`model.staged_fit = Some(false)` (or `NGBoostParams::staged_fit`). It can
also be forced (`Some(true)`) for any head with a location column (Laplace,
TFixedDf, Cauchy), or called directly as `fit_staged` / `fit_two_stage`.

```rust
let mut m = NGBRegressor::new(2000, 0.05);
m.fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val))?; // staged
```

Stages, all on the same model, with every column other than the active one
frozen (zero steps), each with early stopping on the same validation rows
(split once when none is given):

1. Location only; the other columns stay at their marginal constants.
2. The other columns' constants refitted given the location, then those
   columns boosted with the location frozen.
3. A fresh location fit weighted by the location's Fisher entry from stage 2
   (1/σ̂² for Normal; normalised to mean 1), i.e. feasible GLS.
4. As stage 2 on the stage-3 location, then the default recalibration.

The first implementation (Cycle 13, a separate NormalFixedVar /
NormalFixedMean fit per stage, measured in R46–R48) and this generic one
differ only through the rounding-decided line-search scale (§1). Their
results agree within noise (R54a below).

The result is an ordinary `Normal` model: stage-3 rounds are stored as
`[tree, 0]` and stage-4 rounds as `[0, tree]`. Format, prediction and
serialization are the same as for a joint fit. A bench variant built on the
library call reproduces the research implementation to 4 decimals.

**Results** (R48, both arms with the default log-σ guard; Δ = staged − joint):

| dataset | joint NLL | ΔNLL (se) | seed wins | ΔCRPS | ΔRMSE | trees fitted |
|---|---|---|---|---|---|---|
| kin8nm | −0.444 | −0.091 (0.011) | 10/10 | −0.011 | −0.020 | 2.9× |
| het_a | 1.524 | −0.164 (0.020) | 10/10 | −0.132 | −0.245 | 2.7× |
| het_b | 0.748 | −0.072 (0.008) | 10/10 | −0.008 | −0.004 | 1.3× |
| homo | 1.711 | −0.111 (0.025) | 9/10 | −0.130 | −0.277 | 2.6× |
| tnoise | 1.525 | −0.092 (0.031) | 7/10 | −0.106 | −0.195 | 2.9× |
| lognorm_het | 2.653 | −0.019 (0.119) | 5/10 | −0.014 | −0.235 | 2.2× |
| protein | 2.885 | −0.025 (0.008) | 9/10 | −0.145 | −0.271 | 3.7× |
| support log cost | 1.154 | −0.011 (0.002) | 10/10 | −0.006 | −0.010 | 2.8× |
| support log d.time | 1.836 | −0.003 (0.003) | 6/10 | −0.005 | −0.007 | 2.0× |
| flchain follow-up | 8.441 | +0.001 (0.003) | 6/10 | −2.3 days | +0.3 | 2.0× |

**Why stage 3 (FGLS) is needed.** Without it, the staged fit loses on het_b,
where σ shares the mean's dominant feature (+0.049, 2/10). The unweighted
mean spends its accuracy on high-noise rows, while the likelihood needs it on
low-noise rows. With the weights, het_b becomes −0.072 (10/10).

Two other designs were rejected:

- **Joint "polish"** (continue joint boosting from the staged model): ≈
  neutral, and it does not repair het_b.
- **Recalibrating the stage-2 scale:** not done, because it only shapes the
  normalised weights.

**When it does not help.** If the joint fit's mean has already converged
(simple mean function, plenty of data), there is nothing to recover: on an
easier variant of the homo benchmark (half the feature range, non-Gaussian
noise) the staged fit was within ±0.006 of the joint fit in 6 runs.

**The generic staged fit as the default (R54a,** staged vs joint, 10 × 10,
default guard): kin8nm −0.089, het_a −0.149, het_b −0.086, homo −0.083,
tnoise −0.098, protein −0.022, support log cost −0.010 (7–10/10 seeds);
neutral on lognorm_het (−0.009), support log d.time (−0.004) and flchain
(+0.005, se 0.008). Never significantly worse. Cost ≈ 1.3–3.7× the tree
fits.

**Other heads (R54b–d,** staged vs joint, both recalibrated):

| head | result | default |
|---|---|---|
| LogNormal | kin8nm −0.056 (9/10), lognorm_het −0.118 (10/10), raw support cost −0.011 (10/10); CRPS better on all three. kin8nm RMSE +0.094: a few inflated means exp(μ + σ²/2) | staged |
| StudentT (R54c+e, 10 seeds) | −0.06…−0.46 NLL on all 8 datasets, 79/80 seed wins (the loss: one homo seed, +0.09 with 90 % coverage 0.58); CRPS better on 73/80, median −0.02…−0.76 (lognorm_het −7.1). 15/80 joint runs, 1/80 staged, put ν ≤ 1 or close to it on some test rows, so their CRPS is undefined or > 10³ (NLL stays finite). 90 % coverage is lower (0.86–0.92 vs 0.92–0.94; 0.89 on homo without that seed), 50 % coverage closer to nominal | staged |
| Laplace | het_a −0.36, tnoise −0.35, homo −0.14, kin8nm −0.06, het_b −0.03; but protein +0.25 (RMSE +1.55), support log d.time +0.06, flchain +0.03 (0/10 each) | joint (opt-in) |

Laplace's location stage with a frozen scale is sign-gradient boosting at
one marginal b, which is crude where the conditional law is lumpy;
not investigated further. TFixedDf and Cauchy were not measured.

## 3. Log-σ step guard (default since 1.2.0)

LogScore's natural-gradient step for log σ is (z² − 1)/2. On an outlier this
overshoots the row's own optimum, ln|z|, by orders of magnitude. With
`min_samples_leaf = 1`, a single training row alone in a leaf drove ln σ to
40 on log-normal data: test CRPS was 4·10¹³ for the staged fit and up to 423
for the joint fit.

The Normal and NormalFixedMean LogScore paths now cap a row's log-σ step at
**25** by raising that row's metric entry above its Fisher value 2. This
only happens when |z| > √51 ≈ 7.1, so the guard is a no-op on well-behaved
data.

Results (R46 vs R48):

- **Bit-identical** on 8 of 10 datasets for the joint fit (6 of 10 for the
  staged fit).
- **Better where it engages:** lognorm_het −0.045 NLL, with worst-seed CRPS
  423 → 3.0; tnoise −0.015; staged fit on tnoise −0.021. The other cells are
  within ±0.005.
- **Cap 5 is too tight:** it also removed the blow-ups but cost +0.02 on
  het_a and tnoise (R47).

`NGBOOST_NORMAL_GUARD=0` disables the guard, and `=<cap>` sets a different
cap (research A/B).

## 4. CRPS metric of Normal / NormalFixedMean / LogNormal (bug fix)

The CRPS metric is 2∫∇F(t)∇F(t)ᵀ dt (equal to the expected Hessian of the
CRPS). For N(μ, σ) in (μ, log σ) it is

&nbsp;&nbsp;&nbsp;&nbsp;diag(1/(√π σ), σ/(2√π)).

Upstream NGBoost, and this crate until now, used diag(1/√π, σ²/(2√π)), which
is σ times too large in both entries. Every row's natural gradient was
therefore divided by σ:

- Where σ is small, steps exploded: test CRPS up to 2·10¹³ and NLL up to
  10³³ (6 of 10 datasets above 10⁷; R51).
- Where σ is large (flchain, in days), steps were 1/σ too small.

The fix is verified by quadrature for all three heads, and by unit
equivariance: with y → c·y, the μ step scales by c and the log σ step is
unchanged. Both tests fail on the old metric. `NGBOOST_CRPS_METRIC_LEGACY=1`
restores the upstream metric (A/B only).

**Effect** (R52 vs R51): the CRPS-trained Normal is now stable on all 10
datasets (8–10/10 seed wins over the legacy metric everywhere). Against the
LogScore Normal it has lower mean CRPS on all 10 datasets (het_b only
−0.001). Its NLL
is mixed, and terrible on the multimodal flchain target (13.8 vs 8.4): CRPS
barely penalises thin tails. Use LogScore when NLL or interval tails matter.
Use the staged LogScore fit for both.

## The staged fit needs the scale recalibration (Cycle 17)

The staged scale stage is fitted to the residuals of a location trained
for about 4× as many rounds as the joint fit's, so it comes out too narrow.
The end-of-fit recalibration on the validation set is what corrects it.

| setting | staged − joint, NLL |
|---|---|
| calibrated, n = 3000 (R54a) | −0.02…−0.15 |
| calibrated, n = 700 (70 validation rows) | −0.01…−0.07 (6–9/10) |
| uncalibrated, n = 3000 | +0.04…+0.21 on 4/5 (het_b −0.08) |
| uncalibrated, n = 700 | +0.8…+14 (cov90 0.42–0.67) |
| 30 validation rows (calibration cannot run) | +0.11…+0.21 on 3/5 |

CRPS favours the staged fit in every row: the location really is better.
So the default (`staged_fit = None`) stages only when the recalibration
will run: `auto_calibrate_scale` on and ≥ 40 validation rows.
`NGBoost::uses_staged_fit(n_rows, n_val)` reports the decision. On small,
shock-prone series (elnino) the calibrated staged fit can still lose: no
mean gain, sharper σ (known_limitations 52).
