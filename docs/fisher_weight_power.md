# `fisher_weight_power` — Fisher-power weighted base learners (research option)

**Status:** opt-in for all heads except **Poisson, NegativeBinomial and Categorical (Bernoulli and K > 2), where β = 1 is the default since 1.2.0** (`Distribution::default_fisher_weight_power`; force the old behaviour with `fisher_weight_power = Some(0.0)`). Elsewhere default `None` (unweighted). Measured 2026-11-09 on `ngboost-rs` 1.2.0
(`research/ngboost/`, runs R1–R3, R8). This document exists so that a
downstream project can decide, *on its own data*, whether to enable it.

## What it changes

NGBoost fits one base learner per distribution parameter to that parameter's
natural-gradient column by least squares. Upstream (and this crate's
default) does so **unweighted**: a leaf takes `mean(g/F)`. With
`fisher_weight_power = Some(β)` the fit is weighted by `w_i = (F_ii + λ)^β`
(times any sample weight), where `F_ii` is the per-row Fisher diagonal for
that parameter and `λ = tikhonov_reg`:

| β | leaf value for the location parameter of Normal(μ, log σ) | meaning |
|---|---|---|
| 0 (default) | `mean(μ − y)` over the leaf | plain L2 boosting on residuals; every row counts equally regardless of its σ̂ |
| 1 (`fisher_weighted_learner = true`) | `Σ (μ−y)/σ̂² / Σ 1/σ̂²` | GLS / Fisher-scoring (Newton) step; rows with small σ̂ dominate |
| 0.5 | weights `1/σ̂` | halfway; the analogue of β-NLL with β = 0.5 (Seitzer et al., ICLR 2022 — note their β equals 1 − ours) |

The weights change both the leaf values and the split selection. For the
log σ parameter of the Normal the Fisher entry is constant (2), so only the
μ tree is affected. Diagonal Fisher metrics only (Normal, Laplace, Poisson,
Bernoulli, …); for non-diagonal metrics the option is silently a no-op.
Training-time only: the model format and prediction code are unchanged.

```rust
let mut m = NGBoost::<Normal, LogScore, HistogramLearner>::new(500, 0.05, HistogramLearner::new(3));
m.fisher_weight_power = Some(0.5);           // or via NGBoostParams.fisher_weight_power
```

## Evidence (held-out test NLL, paired vs β = 0, lower is better)

Protocol: 70/10/20 train/val/test per seed, early stopping on val (patience
50), identical hyperparameters, `NGBoost<Normal, LogScore, HistogramLearner(depth 3)>`,
lr 0.05, 10 seeds. "wins" = seeds where β beat β = 0.

| dataset | what it is | β = 0.5 | β = 1 | notes |
|---|---|---|---|---|
| het_b   | synthetic, σ = exp(−1 + x₀) shares the mean's feature | −0.071 (10/10) | **−0.103** (10/10) | also −0.107 at lr 0.01, −0.063 at n = 12k |
| kin8nm  | UCI robot-arm, 8 192 rows | −0.028 (10/10) | **−0.042** (10/10) | CRPS −0.003; −0.054 at lr 0.01 |
| het_a   | synthetic, σ driven by features the mean ignores | −0.025 (6/10) | −0.021 (5/10) | grows with n: −0.068 (5/5) at n = 12k |
| homo    | synthetic, constant σ (control) | −0.002 | −0.000 | neutral (at lr 0.01: +0.020 for β = 0.5) |
| tnoise  | synthetic, Student-t(4) noise, Normal head | −0.015 | −0.014 | noisy, se ≈ 0.03 |
| protein | UCI RMSD ≥ 0, skewed, 12k-row subsample | **+0.051** (3/10) | **+0.191** (1/10) | RMSE +4 %, CRPS +0.07, seed sd ×5 |

Selecting β ∈ {0, 0.5, 1} **by validation NLL** (R8) recovers the gains on
the five favourable datasets (−0.018 … −0.103) but **still loses on protein**
(+0.049, 3/10): β = 1's test NLL varies ~5× more across splits there, so a
≈1 000-row validation set often picks a β that is worse on test.

## Poisson (count) heads — recommended ON

For Poisson(λ) the unweighted leaf is a *mean of ratios*, `1 − mean(yᵢ/λ̂ᵢ)`,
which weights low-rate rows ~λ_high/λ_low times more than high-rate rows;
the β = 1 leaf is the *ratio of sums* `1 − Σy/Σλ̂` (the Newton step), and
β = 1 is exactly XGBoost/LightGBM's second-order Poisson tree (leaf G/H,
gain G²/H). Measured on five synthetic count DGPs × {lr 0.05 (10 seeds),
lr 0.01 (5), n = 12k (5)} with an oracle (true λ) as reference:

| DGP (rates) | baseline bias Σy/Σλ̂ | baseline dispersion | Δ NLL β=1 | Δ RMSE β=1 |
|---|---|---|---|---|
| points-like, λ 1–40 | 1.02–1.06 (under-predicts totals) | 1.24–2.18 on true-Poisson data | **−0.08 … −0.10** (19/20 seeds) | **−7.5 %** |
| log-linear, λ 0.2–30 | 1.02–1.05 | 1.02–1.10 | −0.003 … −0.012 (19/20) | −1 % |
| strikeouts-like, λ 3–10 | 1.00 | 1.01 | 0.000 ± 0.001 | 0 |
| over-dispersed (NegBin r=4) | 1.04–1.08 | ≈ 2.1 for every β (oracle 1.85) | −0.01 … −0.04 | ≈ 0 |
| mildly over-dispersed (r=10) | 1.01 | ≈ 1.29 | 0.000 | 0 |

β = 1 was never worse on any count DGP in any configuration, removed the
total bias, and cut seed-to-seed variance 3–5× where rates are
heterogeneous; where rates are nearly homogeneous it is a no-op. This
reproduces the earlier real-data measurement (NBA points: gain; MLB
strikeouts: wash) and explains it by rate heterogeneity. A post-hoc rate
shift on validation does not substitute for it. **If the dispersion index
√mean((y−λ̂)²/λ̂) on a held-out block is ≫ 1, no β fixes that — the model
needs a Negative Binomial head.**

Evaluation protocol for counts: on a chronological holdout compute, for
β = 0 and β = 1, Poisson NLL, CRPS, RMSE, the bias ratio Σy/Σλ̂ and the
dispersion index; expect β = 1 to win whenever predicted rates span more
than ~3× across rows.

## Other heads (analytical classification, 2026-11-09)

| head | Fisher structure | β effect |
|---|---|---|
| Exponential, HalfNormal | constant (1, 2) | provably none (tested: β = 0 and 1 agree to 1e-9) |
| Gamma, Weibull | non-diagonal 2×2 | the weighted path does not apply (diagonal metrics only) |
| LogNormal, Laplace, Cauchy, StudentT, TFixedDf | Normal-like: loc column weighted by 1/σ² | same trade-off as Normal — keep 0 unless validated |
| NegativeBinomial | diagonal, mean column ∝ rμ/(r+μ) | default 1 (`docs/negative_binomial.md`) |
| Bernoulli | p(1−p) per row (1×1) | **default 1 since 1.2.0**, confirmed on 3 real binary datasets (R27) (R20/R23/R26): restores the base rate the unweighted head under-predicts by 6–15 % on imbalanced data (structural — persists at lr 0.01), log loss −0.001…−0.011, worst cell +0.0004 |
| Categorical (K > 2) | non-diagonal; each logit weighted by its diagonal entry p_k(1−p_k) (XGBoost's multiclass approximation) | **default 1 + per-round intercept step since 1.2.0**; real data (R33): support2 disease group (8 classes) −0.082 (10/10), digits (10 classes) −0.310 (10/10, accuracy 87.0 → 96.4 %); never worse on 3 synthetic + 2 real tasks: support2 disease class −0.022 log loss (10/10), −0.032 with `refit_intercept` (10/10), +1.2 pt accuracy; flchain cause of death a tie (R28/R31) |
| MultivariateNormal, Gamma, Weibull | non-diagonal | only with an explicit β (unmeasured) |

## When it helps / hurts (mechanism)

Weighting by `1/σ̂²` is the statistically efficient (GLS) way to fit the
conditional mean **when the predicted σ̂ tracks a genuinely Gaussian
conditional variance**. It hurts when the high-σ̂ region is where the mean
signal lives *and* the Normal is misspecified (bounded, skewed targets like
RMSD): the μ tree then starves those rows — the "vicious cycle" described
for β-NLL — and RMSE degrades, which the NLL cannot recover.

## Why it fails on protein (bimodal targets) and why capping does not help

Protein's RMSD target is bimodal (modes at 1–3 and 11–16). With β = 1 the
tight low mode receives 58 % of the μ-tree weight from 40 % of the rows;
the wide high mode (36 % of rows, 19 % of weight) is under-fit (RMSE
5.83 → 6.29 there) and carries essentially the whole NLL loss (+0.16), and
the lowest-σ̂ decile loses +1.26 NLL per row from over-confidence
(`examples/protein_probe.rs`, report R25). Before enabling β on your data,
**check the conditional target for multiple modes; if present, keep β = 0.**
It is not Normal misspecification in general: on a truly log-normal target
the misspecified Normal head gains from β (−0.23 NLL), and the LogNormal
head on protein still loses with β = 1 (R19).

`fisher_weight_cap = Some(κ)` (opt-in) clamps each column's weights to
[median/κ, median·κ] per round. κ = 3 cuts protein's loss to +0.03 NLL but
still costs +3 % RMSE and forfeits a third of the het_b/kin8nm gains;
κ ≥ 10 keeps the gains but protein stays +0.07…+0.10 (R18). Not a safe
default.

## How to evaluate it on your data (recommended protocol)

1. Use the split you deploy with (for time series: chronological
   train → validation → test blocks; never random).
2. Fit β ∈ {0, 0.5, 1} with identical hyperparameters and early stopping on
   the validation block; score each on the **test** block: NLL first, then
   CRPS and RMSE. Repeat over ≥ 5 seeds / rolling origins.
3. Enable β > 0 only if it wins on NLL *and* does not lose on RMSE/CRPS, on
   most seeds (e.g. ≥ 80 %). If the target is bounded/skewed, expect β = 0.
4. Keep `auto_calibrate_scale` (default) on in all arms; it is orthogonal
   (it fixes σ's level after training) and in the measurements the
   combination β + calibration was the best configuration wherever β helped,
   and neutral on protein.
5. Report: β chosen, Δ NLL / CRPS / RMSE with standard errors and seed wins.

Leave it `None` if in doubt — the default is never the worst arm.
