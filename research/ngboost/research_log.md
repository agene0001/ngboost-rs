# NGBoost-rs — autonomous algorithm research log

Scope: library-level improvements to out-of-sample probabilistic predictive
performance of `ngboost-rs`. Production defaults are never changed here;
every candidate is an opt-in flag with the upstream path preserved.

Conventions: NLL/CRPS/RMSE are **held-out test** values unless marked
`train`. `calib` = sqrt(mean(err²)/mean(σ̂²)) on test (>1 overconfident).
Harness: `examples/research_bench.rs` (70/10/20 split per seed, early stop
on val, identical hyperparameters across variants). Raw rows are appended to
`experiments.csv`; per-run outputs live in `reports/`.

## 2026-11-09 — Cycle 0: repository diagnosis

Algorithm as implemented (`src/ngboost.rs::fit_internal`, verified by reading):
1. `init_params = D::fit(y)` (marginal MLE).
2. Per round: `grads = F⁻¹ ∇S` per row (diagonal fast path; 1–3×3 closed
   form; LAPACK/pinv fallback), one base learner per parameter fitted to the
   natural-gradient column by **unweighted** least squares (Friedman-MSE
   histogram tree, depth 3, leaf = mean), one **global** scalar line search
   (binary: powers of two, same as Python), `params -= lr·scale·tree(x)`.
3. Early stopping on a validation score with truncation to the best round.
4. Opt-ins already present: Tikhonov `F + λI`, golden-section line search,
   LR schedules, minibatch/col subsampling, `fisher_weighted_learner`
   (Fisher-scoring leaves, commit aa7b4eb, measured only on Poisson heads),
   `calibrate_uncertainty` (post-hoc multiplicative σ recalibration on a
   validation set).

Normal/LogScore specifics (`src/dist/normal.rs`): parameters (μ, log σ);
∇ = ((μ−y)/σ², 1 − z²); F = diag(1/σ², 2); natural gradient
ñ = (μ − y, ½(1 − z²)). So the loc tree is fitted to **plain residuals with
equal weight regardless of σ̂** — the natural gradient removes the 1/σ²
"gradient starvation" of raw NLL (Seitzer et al. 2022, β-NLL with β=1), while
`fisher_weighted_learner` re-weights by 1/σ² (GLS / Newton leaf, β=0 in
Seitzer's convention).

Existing test coverage relevant to accuracy: `tests/accuracy_parity.rs`
(hist vs exact trees, NLL/RMSE floors), finite-difference gradient tests for
several distributions, closed-form solver tests. **No held-out calibration /
coverage diagnostics and no real-data benchmark existed** → built
`examples/research_bench.rs`. Real datasets available locally (no download):
UCI kin8nm (8192×8) and protein (45730×9) from the sibling Python repo.

### Independently generated hypotheses (before literature)
- H1 **Fisher-power weighting**: weights w_i = F_jj^β for the base-learner
  fit, β∈[0,1] interpolating upstream (β=0) and Fisher scoring (β=1).
  Rationale: efficiency (GLS) vs. under-fitting the mean where σ̂ is large.
- H2 **Per-parameter line search**: the single scalar couples μ and log σ
  step lengths; location-scale boosting literature reports systematically
  different optimal step lengths per parameter.
- H3 **Variance overconfidence**: log σ trees are fitted to in-sample
  residuals of a μ that is itself fitted → σ̂ too small OOS. Measure
  `calib` and coverage first; `calibrate_uncertainty` is the reference fix.
- H4 **Noncyclical updates** (update only the parameter whose tree step
  reduces the loss most per round).

### Literature (Lacuna MCP + known sources)
- Duan et al., NGBoost, ICML 2020 — https://lacuna.tiptreesystems.com/work/ngboost-natural-gradient-boosting-for-probabilistic/version/art_8bb559093868439e9df690e9dc7ebc1e
- Seitzer, Tavakoli, Antic, Martius, "On the Pitfalls of Heteroscedastic
  Uncertainty Estimation with Probabilistic Neural Networks", ICLR 2022 —
  β-NLL: loss weighted by stopgrad(σ^{2β}); β=0.5 best trade-off across 12
  UCI sets. https://lacuna.tiptreesystems.com/work/on-the-pitfalls-of-heteroscedastic-uncertainty-estimation/version/art_abef037adc544a84b33e6627fb3928d9
  → maps onto H1 with β_Seitzer = 1 − β_ours.
- Thomas, Mayr, Bischl, Schmid, Smith, Hofner, "Gradient boosting for
  distributional regression: faster tuning and improved variable selection
  via noncyclical updates", Stat. Comput. 2018 — noncyclical updates: same
  predictive accuracy, 10–20× cheaper tuning. https://lacuna.tiptreesystems.com/work/gradient-boosting-for-distributional-regression-faster/version/art_b2bb664e25e9477da5ee90522e50844c
  → H4 is unlikely to improve OOS accuracy per se; deprioritized.
- Adaptive step-length work for Gaussian location–scale boosting (Zhang,
  Hepp, Greven, Bergherr, 2022) — **not found in Lacuna**; cited from memory,
  unverified here. Relevant to H2.
- Sprangers et al., PGBM, KDD 2021 — https://lacuna.tiptreesystems.com/work/probabilistic-gradient-boosting-machines-for-large-scale/version/art_3901ff9f96e947c58ef6ae5f34d3598c (alternative design; not pursued).

### Decision for Cycle 1
Measure baseline + calibration diagnostics (H3 evidence), then test H1
(cheapest, strongest prior, one `Option<f64>` field) across 6 datasets.

## 2026-11-09 — Cycle 1: baseline diagnostics + H1 (Fisher-power weights)

Runs: `reports/R1_fisher_power_lr05.csv` (6 datasets × 10 seeds, lr 0.05,
n_est 2000, patience 50, depth 3, synthetic n = 3000, protein subsampled
to 12 000 rows), `R2_…lr01.csv` (lr 0.01, n_est 6000, patience 100, 5
seeds), `R3_…n12k.csv` (synthetic n = 12 000, 5 seeds). Library tests after
the change: 443 passed, 0 failed. `fwp0` (β = 0 through the weighted path)
reproduces `base` bit-for-bit (equivalence check).

### Observed gap 1 (H3 evidence): systematic out-of-sample overconfidence
Baseline (upstream algorithm, early-stopped on val NLL), test-set values:

| dataset | calib = √(mean err²/mean σ̂²) | cov50 | cov90 | train NLL → test NLL |
|---|---|---|---|---|
| het_a   | 1.22 | 0.39 | 0.86 | 1.26 → 1.56 |
| het_b   | 1.22 | 0.47 | 0.87 | – |
| homo    | 1.14 | 0.42 | 0.85 | – |
| tnoise  | 1.30 | 0.39 | 0.86 | – |
| kin8nm  | 1.20 | 0.39 | 0.84 | – |
| protein | 1.11 | 0.42 | 0.89 | – |

Same picture at lr 0.01 (calib 1.10–1.24) and at n = 12k (1.09–1.26): it is
not a small-sample artefact. Mechanism (hypothesis): the log σ trees are
fitted to ½(1 − z²) with **in-sample** residuals of a μ that was fitted to
the same rows (leverage shrinks them), plus the σ trees' own optimism.
Post-hoc moment recalibration (`calibrate_uncertainty`, "calib") brings
calib to ≈1.00 and improves test NLL on 5/6 datasets (10/10 seed wins on
het_a, kin8nm, protein; Δ −0.022 … −0.059), but **over-corrects on het_b**
(Δ +0.017, cov90 0.948): a single factor from mean(err²)/mean(σ̂²) is
dominated by the large-σ rows. The NLL-optimal common factor is
c² = mean(z²) (MLE), and an affine log σ' = a + b·log σ head (convex, Newton)
can also fix over-dispersed σ̂. → Cycle 2.

### H1 results (paired Δ test NLL vs base; negative = better)

| dataset | β=0.25 | β=0.5 | β=0.75 | β=1 (fw) | note |
|---|---|---|---|---|---|
| het_a (n=3k)  | −0.006 | −0.025 (se .010, 6/10) | −0.007 | −0.021 (se .019) | R3 n=12k: β=1 −0.068 (se .008, 5/5) |
| het_b         | −0.049 | −0.071 (se .004, 10/10) | −0.095 | **−0.103 (se .009, 10/10)** | R2 −0.107 (5/5); R3 −0.063 (5/5) |
| homo          | +0.010 | −0.002 | −0.012 | −0.000 | R2: β=0.5 +0.020 (se .007, 1/5) |
| tnoise        | −0.036 | −0.015 | −0.004 | −0.014 | noisy (se ≈ 0.03) |
| kin8nm        | −0.013 | −0.028 (se .003, 10/10) | −0.034 | **−0.042 (se .008, 10/10)** | R2 −0.054 (5/5); CRPS also better |
| protein       | +0.019 | +0.051 (se .026, 3/10) | +0.081 | **+0.191 (se .087, 1/10)** | RMSE +4 %, CRPS +0.07; seed sd ×5 |

Interpretation: where σ̂ tracks a genuinely Gaussian conditional variance
(het_a/het_b, kin8nm), 1/σ̂²-weighted (GLS/Newton) leaf fits are more
efficient and the gain is monotone in β and grows with n. On protein
(RMSD ≥ 0, skewed, Normal misspecified) the weighting starves the mean fit
in the high-σ̂ region (exactly the β-NLL "vicious cycle" of Seitzer et al.):
RMSE +4 %, NLL worse and unstable. Weighting also changes split selection,
not only leaf values. **No β dominates across datasets → not a default.**
β = 0.5 is a hedge (≈60–70 % of the β=1 gain on het_b/kin8nm, +0.05 on
protein) but still not uniformly safe.

### Decision
- Keep `fisher_weight_power: Option<f64>` as an opt-in research knob
  (default None → production behaviour unchanged, verified bit-exact at
  β = 0 and by the boolean path at β = 1). Document: select β ∈ {0, 0.5, 1}
  by validation NLL; expect gains for Gaussian-like heteroscedastic targets,
  losses for bounded/skewed targets.
- Next autonomous question (Cycle 2): the overconfidence gap is the most
  consistent effect (every dataset, every config). Test MLE and affine
  post-hoc σ recalibration against the moment-ratio method, alone and
  combined with β.

## 2026-11-09 — Cycle 2: H3 → P2, post-hoc affine recalibration of log σ

Runs: `reports/R4_posthoc_lr05.csv` (10 seeds), `R5_posthoc_lr01.csv`
(5 seeds, lr 0.01), `R6_posthoc_n12k.csv` (5 seeds, n = 12k). Variants:
`calib` = library moment matching, `mle` = c² = mean(z²), `affine` =
log σ' = a + b·log σ (closed-form Newton in the harness), each fitted on the
validation split and applied to the test distribution. Full derivation and
code map in `proposals/P2_affine_scale_recalibration.md`.

### Paired Δ test NLL vs base (negative = better; seed wins in parentheses)

| dataset | calib (moment) | mle | affine |
|---|---|---|---|
| het_a  | −0.036 (10/10) | −0.037 (10/10) | −0.037 (10/10) |
| het_b  | **+0.017 (4/10)** | −0.010 (8/10) | **−0.018 (9/10)** |
| homo   | −0.023 (8/10) | −0.024 (9/10) | −0.023 (10/10) |
| tnoise | −0.059 (9/10) | −0.058 (9/10) | −0.059 (10/10) |
| kin8nm | −0.030 (10/10) | −0.030 (10/10) | −0.031 (10/10) |
| protein| −0.022 (10/10) | −0.024 (9/10) | −0.033 (9/10) |

lr 0.01 (R5): affine −0.020 … −0.029, 5/5 wins on five datasets, 4/5 on
protein, never below moment/MLE. n = 12k (R6): methods converge; affine
−0.003 … −0.051, never worse than base. Coverage after affine: cov90
0.90–0.93, cov50 0.44–0.50, calib 0.97–1.03. RMSE unchanged by construction.

Combined with H1: `fw+affine` on het_b −0.117, kin8nm −0.078, het_a −0.065
(all 9–10/10) but protein +0.066 — the β regression is not a calibration
problem, so P2 does not rescue β = 1 there.

### Implementation (opt-in; defaults untouched)
`NGBoost::calibrate_param_affine(x_val, y_val, col, weights)` and
`calibrate_scale_affine(x_val, y_val)` (col = 1), plus
`NGBRegressor::calibrate_scale_affine`. Generic score-driven damped
FD-Newton (`minimize_2d`), never accepts a worse validation score, keeps
b > 0. Persisted by folding into `init_params[col]` and the stored
learners' leaf values via the new `TrainedBaseLearner::scale_output`
(bincode layout unchanged — adding a field to `SerializedNGBoost` would
have broken every saved model). Tests (`tests/test_research_options.rs`,
8 new): β = 0 ≡ default and β = 1 ≡ boolean flag bit-exact; folded model ≡
manual a + b·p (loc bit-exact, log σ to 1e-9); staged == full; bincode
round trip bit-exact; all four learner types; input validation leaves the
model untouched; `NGBRegressor` save/load. Full suite: 451 passed, 0 failed.

### Decision
Keep P2 as an opt-in method. It is the only candidate that improved
held-out NLL on every dataset in every configuration. Not wired into
`fit` (would change default behaviour and silently consume the validation
set). Cross-check of the generic optimizer vs the closed form → R7.

### R7 cross-check (10 seeds, lr 0.05): library vs closed form
`libaffine` (= `NGBoost::calibrate_scale_affine`, generic score-driven
FD-Newton, folded into init + leaves) equals the harness's closed-form
Newton `affine` to every printed digit (NLL, CRPS, cov50/90, calib) on all
six datasets — prediction (4) of P2 confirmed; the fold-into-model
persistence is exact in practice. `reports/R7_libaffine_crosscheck.csv`.

## 2026-11-09 — Cycle 3: is β usable as a tuned hyperparameter? (R8)

`fwsel` fits β ∈ {0, 0.5, 1} per split and keeps the best **validation**
NLL (3× fit cost); `reports/R8_beta_selection.csv`, 10 seeds, lr 0.05.

| dataset | fwsel | libaffine | fwsel+libaffine |
|---|---|---|---|
| het_a  | −0.041 (7/10) | −0.037 (10/10) | **−0.096 (10/10)** |
| het_b  | −0.103 (10/10) | −0.018 (9/10) | **−0.117 (10/10)** |
| homo   | −0.018 (7/10) | −0.023 (10/10) | **−0.067 (10/10)** |
| tnoise | −0.063 (5/10) | −0.059 (10/10) | **−0.129 (10/10)** |
| kin8nm | −0.044 (10/10) | −0.031 (10/10) | **−0.082 (10/10)** |
| protein| **+0.049 (3/10)** | −0.033 (9/10) | +0.007 (6/10, se 0.026) |

Negative finding: validation selection does **not** make β safe on protein.
β = 1 has ~5× the seed-to-seed NLL variance of β = 0 there (R1), so a
1 200-row validation split often prefers a β whose test NLL is worse. The
β knob should therefore be documented as "try on Gaussian-like
heteroscedastic targets; keep 0 for bounded/skewed targets unless a larger
validation set confirms it", not as something to tune blindly. With the
affine head on top the downside shrinks to neutral and the upside on the
other five datasets is the largest measured in this study.

### Session summary / decisions
- **P2 (affine σ recalibration): keep, opt-in.** Lower test NLL on 6/6
  datasets × 3 configurations; cov90 restored to ≈ 0.90; μ untouched;
  persists through save/load with no format change; library optimizer
  verified against the closed form (R7).
- **H1 (`fisher_weight_power`): keep, opt-in, documented caveat.** Large,
  robust gains on het_b/kin8nm/het_a (grow with n), neutral on homo/tnoise,
  regression on protein that validation selection does not remove.
- Not pursued (reasoned out, not measured): H2 per-parameter line search
  and H4 noncyclical updates would both need per-parameter step storage
  (bincode-incompatible without the fold trick) and the literature (Thomas
  et al. 2018) predicts no accuracy gain for H4.

### Next autonomous questions
1. In-training leverage correction for the log σ gradient
   (z² / (1 − ĥ_ii), with ĥ_ii ≈ 1 − Π_t(1 − lr·s_t/n_leaf_t(i)) tracked
   during boosting) — would remove the overconfidence without a validation
   set; compare against P2 on the same grid.
2. Why β = 1 fails on protein: test the misspecification hypothesis by
   refitting with LogNormal/Gamma heads (bounded support) and checking
   whether β > 0 then helps; and test a weight cap w_i = min(F_ii, κ·median F)
   as a robustified Fisher weighting.
3. Extend the harness and P2 evaluation to Laplace / StudentT heads (P2 is
   distribution-generic; only Normal was measured).

## 2026-11-09 — Cycle 4: making P2 the default (1.2.0)

User decision: enable the affine σ recalibration by default (parity with
Python ngboost is no longer a goal); add a chronological auto-split for
time series; keep β opt-in and document it for downstream evaluation.

### Design
- `Distribution::scale_param_index()` (None by default; Some(1) for Normal,
  Laplace, LogNormal, Cauchy, StudentT, TFixedDf; Some(0) NormalFixedMean).
- `NGBoost::auto_calibrate_scale` (default **true**): at the end of
  `fit`/`partial_fit`, after truncation, with validation data present →
  `calibrate_param_affine_impl` on that column; result in
  `scale_calibration: Option<(col, a, b)>`; failures never fail the fit.
  Not applied without validation data (training residuals would shrink σ).
- Guards from R9 (val-size sweep, 10 seeds, het_b/kin8nm/tnoise/protein):

  | val rows | shift-only (b=1) Δ NLL | affine Δ NLL | decision |
  |---|---|---|---|
  | 20  | +0.010 / −0.043 / +0.026 / **+0.067** | +0.002 / +0.160 / +0.371 / +0.560 | skip |
  | 40  | −0.008 / −0.035 / −0.069 / −0.011 (7–10/10) | −0.006 / −0.017 / −0.051 / **+0.118** | shift only |
  | 80  | −0.006 / −0.026 / −0.077 / −0.017 | −0.013 / −0.019 / −0.079 / **+0.021** | shift only |
  | 160 | −0.008 / −0.033 / −0.062 / −0.023 | −0.016 / −0.033 / −0.062 / **+0.005** | shift only |
  | 400 | (300) / −0.031 / (300) / −0.028 | (300) / −0.031 / (300) / **−0.033** | slope |
  | 800 | – / −0.028 / – / −0.027 | – / −0.029 / – / **−0.035** | slope |

  (synthetic sets have only 300 val rows, so 400/800 inform kin8nm and
  protein only.) → `AUTO_CALIBRATION_MIN_ROWS = 40`,
  `AUTO_CALIBRATION_SLOPE_MIN_ROWS = 300`. The slope's instability is a
  heavy-tail effect (protein), not a sample-size law per se; 300 is the
  conservative choice that is safe on every measured dataset.
- `ValidationSplit::{Random (default), Chronological}` for the
  `validation_fraction` auto-split; `Chronological` is bit-identical to an
  explicit tail split (test).
- `NGBoostParams` gained `auto_calibrate_scale`, `validation_split`,
  `fisher_weight_power` (serde defaults; old JSON parses).
- `hyper_opt` folds inherit the default (calibrated on the scoring fold —
  same "optimistic but consistent" trade-off as the existing fold-level
  early stopping); documented in the code.
- `NGBSurvival` (own fit loop) and non-log-scale second parameters (Gamma,
  Weibull, …) are untouched.

### Evidence for the default beyond Normal (R10, 10 seeds, lr 0.05)
| head | het_a | tnoise | kin8nm | protein |
|---|---|---|---|---|
| Laplace  | −0.011 (10/10) | −0.016 (10/10) | −0.014 (10/10) | −0.007 (10/10) |
| StudentT | −0.039 (10/10) | −0.028 (10/10) | −0.011 (9/10) | −0.024 (10/10) |

StudentT's baseline is under-confident (cov90 ≈ 0.95); the head corrects
in that direction too. R11: the default path equals the explicit call on
6/6 Normal datasets.

### Artifacts
`docs/scale_calibration.md`, `docs/fisher_weight_power.md` (evaluation
protocol for downstream projects), README section, version 1.2.0,
tests/test_research_options.rs (14 tests). No existing test needed
changes when the default flipped.

## 2026-11-09 — Cycle 5: Poisson head (user priority) — P3

Question: is the Poisson head biased/miscalibrated out of sample, does β
help, is post-hoc rate correction useful, and is over-dispersion the real
gap? Harness: `--dist poisson`, count DGPs `pois_lin` (λ 0.2–30),
`pois_nl` (points-like 1–40), `pois_low` (strikeouts-like 3–10), `nb_over`
(Gamma–Poisson, r = 4), `nb_low` (r = 10); metrics NLL, CRPS, RMSE,
randomized-PIT coverage, dispersion index √mean((y−λ̂)²/λ̂), bias Σy/Σλ̂,
plus an `oracle` row (true λ). Derivation in `proposals/P3_poisson_leaf_bias.md`.

### Results (R12: lr 0.05, 10 seeds; R13: lr 0.01, 5 seeds; R14: n = 12k, 5 seeds)
Oracle: bias 0.99–1.01, dispersion 1.00 on `pois_*` (sampler OK).

| DGP | base bias / disp | Δ NLL β=0.5 | Δ NLL β=1 | Δ RMSE β=1 | post-hoc `rate` | note |
|---|---|---|---|---|---|---|
| pois_nl  | 1.063 / 1.28 (n12k: 2.18!) | −0.075 (10/10) | **−0.098 (10/10)**; −0.077 (4/5); −0.098 (5/5) | −0.31 (−7.5 %) | −0.015 (10/10) | β=1 bias 1.04→1.00, sd ÷3–5 |
| pois_lin | 1.046 / 1.10 | −0.012 (10/10) | −0.012 (9/10); −0.010 (5/5); −0.003 (5/5) | −0.019 | +0.001 | |
| pois_low | 1.011 / 1.02 | −0.000 | −0.001 (4/10); +0.000; +0.000 | −0.004 | +0.001 | homogeneous rates: no effect, as derived |
| nb_over  | 1.083 / 2.19 | −0.060 (8/10) | −0.044 (6/10); −0.028; −0.010 | −0.07 | −0.012 | oracle disp 1.85: misspecified → NegBin |
| nb_low   | 1.011 / 1.29 | −0.002 | +0.002 (3/10); −0.001; +0.001 | +0.004 | +0.003 | |

Predictions 1–4 of P3 hold. β = 1 is never worse on any count DGP in any
configuration and is monotone in β where it matters; it reproduces and
explains the earlier real-data pattern (commit aa7b4eb: NBA points gain,
MLB strikeouts wash = pois_nl vs pois_low). Post-hoc rate correction is
not a default candidate. Over-dispersion is untouched by every variant.

### Decision / recommendation
- Recommend `fisher_weight_power = Some(1.0)` (≡ XGBoost second-order
  leaves) for Poisson heads; proposed as a distribution-level default
  pending user approval. Documented with the evaluation protocol in
  `docs/fisher_weight_power.md`.
- Next: a Negative Binomial distribution (log μ, log dispersion) is the
  remedy for dispersion index ≫ 1; the same leaf-bias argument applies to
  Bernoulli/Categorical heads (F = p(1−p) varies per row; LogitBoost uses
  Newton leaves) — unmeasured here.

## 2026-11-09 — Cycle 6: Negative Binomial head + β defaults for count heads

User decisions: β = 1 default for Poisson (done; `Distribution::
default_fisher_weight_power`), investigate the other rate heads before
defaulting them, implement NegativeBinomial.

### Other rate heads (analytical, verified where testable)
Exponential (F ≡ 1) and HalfNormal (F ≡ 2): constant Fisher ⇒ β is a no-op
(test: β = 0 vs 1 agree to 1e-9). Gamma, Weibull: non-diagonal Fisher ⇒ the
weighted path is not applied. LogNormal/Laplace/Cauchy/StudentT: Normal-like
loc weighting 1/σ² ⇒ same protein-type risk ⇒ keep None. Bernoulli/
Categorical: applicable (LogitBoost), unmeasured — future.

### NegativeBinomial (`src/dist/negative_binomial.rs`)
(log μ, log r), diagonal Fisher (rμ/(r+μ); r²E[s²] by exact log-domain
series), r clamp [1e-3, 1e6] with freezing at the cap, Fisher floor 1e-2 on
the dispersion entry (derivation: entry ~ μ²/(2r²), gradient ~ μ/r ⇒ natural
gradient ~ r in the Poisson limit), moments init clamped [0.1, 1e3].
Unit tests: pmf normalisation/moments, Poisson limit, FD gradients, Fisher
= E[score²] to 1e-6 incl. orthogonality, cdf/ppf, sampler moments.
Integration: beats Poisson by > 0.1 NLL on over-dispersed data; within
0.03 on true-Poisson; bincode round trip bit-exact. Suite: 470 tests.

### Results (R15 lr 0.05 / 10 seeds; R16 lr 0.01 / 5 seeds; paired with R12)
| DGP | Poisson β=1 | NB β=0 | NB β=1 | oracle |
|---|---|---|---|---|
| nb_over | 3.835 (cov90 .63, disp 2.09) | 3.036 (cov90 .87, disp 1.18, r̂ 3.7) | 3.024 (r̂ 4.0) | 2.879 |
| nb_low  | 2.568 (cov90 .81) | 2.498 (cov90 .89, r̂ 13) | 2.500 (r̂ 10) | 2.477 |
| pois_nl | 2.596 | 2.689 | **2.571** (−0.117, 10/10; lr .01 −0.113, 5/5) | 2.418 |
| pois_lin| 1.794 | 1.804 | 1.803 | 1.767 |
| pois_low| 2.289 | 2.292 | 2.295 (+0.003 2/10; lr .01 +0.005 0/5) | 2.275 |

NB closes 0.80 of the 0.96 NLL gap on nb_over; the remainder is mean bias
(Σy/Σμ̂ 1.05–1.07), not dispersion (r̂ ≈ true r). On true-Poisson data NB ≈
Poisson; r̂ stalls at 20–200 (floored dispersion step) costing ≤ 0.005.
β = 1 for NB: large gain with heterogeneous rates, neutral otherwise, tiny
loss on homogeneous low-rate true-Poisson data → **default β = 1 for NB**
(same Newton rationale; documented caveat).

### Next questions
1. NB dispersion dynamics: larger init cap / per-column step size so r̂ → ∞
   on Poisson data; measure whether that removes the +0.003–0.005.
2. Residual mean bias of count heads (1.02–1.07 even with β = 1): the
   leaf/gain are now Newton, so what remains is early stopping on NLL vs
   total-count calibration — test a bias-aware stopping metric or a final
   intercept step on validation (the `ratio` token was not enough).
3. Bernoulli/Categorical heads: measure β = 1 (LogitBoost) vs upstream.
4. CRPS for NegativeBinomial (closed form via regularized incomplete beta).

## 2026-11-09 — Cycle 7: the four open questions (R17–R20)

### Q1 — In-training leverage correction for log σ: **falsified by diagnosis, not implemented**
Measured on the Normal baseline (R17, 6 datasets × 10 seeds): in-sample
z-rms 0.915–0.986 (in-sample z² 0.84–0.97), test calib 1.11–1.30 (test z²
1.23–1.70), and the tractable mean hat-diagonal of the μ trees,
h̄ = 1 − Π_t(1 − lr·s_t·L_t/n) = 0.015–0.035. A correction z²/(1−ĥ) would
inflate σ² by 1.5–3.5 %, against the 23–70 % gap between in-sample and
out-of-sample z². σ̂ is nearly right *on the training rows*; the gap is the
adaptive-search optimism of both heads (split selection, σ trees chasing
noise), which leaf-size leverage cannot see. Held-out recalibration (P2)
remains the right tool. (Hypothesis in Cycle 2 retracted.)

### Q2 — Is β = 1's protein failure misspecification? **No.** Is a capped weight safe? **No.**
- R19c: on a truly log-normal, skewed DGP the *misspecified* Normal head
  with β = 1 improves (−0.23 NLL, 8/10; RMSE −0.43). R19a: the correctly
  specified LogNormal head gains too (β = 0.5 −0.028 8/10; kin8nm β = 1
  −0.048 10/10). R19b: LogNormal head on protein (y + 0.1) with β = 1 still
  loses (+0.052, 1/10; β = 0.5 +0.006). Misspecification is not the cause;
  protein's failure is specific to its structure (272 exact zeros, a mass of
  near-zero RMSD rows with tiny σ̂). Side hazard: the LogNormal head's mean
  exp(μ+σ²/2) exploded on one protein seed (RMSE 1 872) — use the median
  for point predictions with that head.
- R18 (`fisher_weight_cap` κ, weights clamped to [median/κ, median·κ]):

  | κ | het_a | het_b | kin8nm | protein (NLL / RMSE) |
  |---|---|---|---|---|
  | none (β=1) | −0.021 | −0.103 | −0.042 | **+0.191 / +0.176** |
  | 30 | −0.017 | −0.102 | −0.042 | +0.103 / +0.162 |
  | 10 | −0.006 | −0.099 | −0.042 | +0.074 / +0.152 |
  | 3  | −0.025 | −0.081 | −0.019 | +0.031 / +0.136 |

  Even a 3× dynamic range costs protein +3 % RMSE (1/10 wins) while giving
  back a third of het_b/kin8nm; the weighted split criterion and earlier
  stopping (154–164 vs 252 rounds), not extreme weights, drive the loss.
  `fisher_weight_cap` stays an opt-in knob, not a default.

### Q3 — P2 on Laplace / StudentT: done in Cycle 4 (R10): Laplace −0.007…
−0.016 (10/10 ×4), StudentT −0.011…−0.039 (9–10/10 ×4); default on.

### Q4 — Bernoulli β (LogitBoost leaves) — R20, three binary DGPs
Fisher weights now reach Bernoulli (1×1 metric; natural-gradient path
unchanged). Log loss Δ at lr 0.05: −0.002 (9/10), −0.006 (7/10), −0.001
(6/10); at lr 0.01: +0.000, −0.006, +0.001 — not a default. But on the
imbalanced DGP the baseline under-predicts the positive rate by 7 %
(mean p̂ / mean y = 0.93 → 1.007 with β = 1): the Poisson mean-of-ratios
bias again, visible as base-rate miscalibration rather than NLL.
Categorical (K > 2) is non-diagonal and remains unmeasured.

### Decisions
No new defaults. `fisher_weight_cap` kept opt-in. Negative results recorded
in `docs/known_limitations.md`. Next: (a) the residual total-count bias of
count heads; (b) why protein specifically (zero-inflation? try a hurdle /
zero-inflated head or measure β with the 272 zeros removed); (c) Bernoulli
base-rate bias as a cheap validation-fitted intercept.

## 2026-11-09 — Cycle 8: residual count bias, protein mechanism, intercept step

### Hypothesis (stated before measuring)
Partly converged log-link fits under-predict totals: with log λ̂ = log ȳ + ρ·dᵢ,
Σ exp(ρ dᵢ) is convex in ρ and equals Σy at ρ = 0 and ρ = 1, so it is below
Σy in between. Proposed fix: an unshrunk Fisher-scoring intercept step per
round, c_j = −Σwg_j/ΣwF_jj, folded into init_params (`refit_intercept`).

### Results (R21–R24, 10 seeds lr 0.05 unless noted)
- **Poisson, β = 1 default: the hypothesis is refuted for the shipped
  configuration.** Training totals are already exact (train Σy/Σλ̂ = 1.000
  on all five DGPs) — Newton leaves preserve each leaf's total. The step is
  neutral (pois_nl −0.005 6/10 at lr .05; +0.012 1/5 at lr .01). The
  mechanism *does* hold for the unweighted fit (train bias 1.031–1.058,
  fixed to 1.000 by the step) but forcing the total there raises test NLL
  (pois_nl +0.136 vs +0.098 unweighted).
- **The residual test bias is generalization, not optimization.** pois_nl
  test bias minus oracle: +0.031 (lr .05, n = 3k), +0.017 (lr .01), +0.007
  (n = 12k). It shrinks with smaller learning rate and more data; neither
  the intercept step nor a validation ratio removes it (val noise ≈ bias).
  Practical guidance: lr ≤ 0.01 when calibrated totals matter.
- **NegBin:** step neutral (−0.001…+0.006).
- **Normal: step harmful** (+0.019…+0.094 NLL on 5/6, 0–2/10 wins). It
  forces in-sample mean z² = 1, but in-sample σ̂ is slightly *wide* (z-rms
  0.94), so it narrows σ and worsens out-of-sample overconfidence (calib
  1.22 → 1.29). Consistent with Cycle 7: the optimism is out-of-sample.
- **Bernoulli: step helps the base rate.** The unweighted head
  under-predicts positives *in-sample* on imbalanced data (train 0.930,
  test 0.932) — the mean-of-ratios mechanism again. The step restores it
  (1.000 / 1.001); log loss −0.0002…−0.0016 (5–9/10), Brier ≤ 0, never
  worse. β = 1 fixes it too (1.005) with larger log-loss gains on bern_nl.

### Protein: where β = 1 loses (R25, `examples/protein_probe.rs`, 5 seeds)
Target is bimodal (RMSD mode at 1–3, second mode at 11–16).

| band | rows | β=1 weight share | ΔNLL contribution | RMSE β0 → β1 |
|---|---|---|---|---|
| y < 1 | 3.7 % | 5.1 % | −0.002 | 3.89 → 3.84 |
| 1–4 | 40.3 % | 58.1 % | −0.011 | 3.91 → 3.83 |
| 4–10 | 19.7 % | 18.1 % | +0.005 | 3.33 → 3.21 |
| ≥ 10 | 36.3 % | 18.6 % | **+0.160** | 5.83 → **6.29** |

By σ̂ decile the lowest-σ̂ decile alone loses +1.26 NLL per row with
unchanged RMSE (confident-but-wrong predictions become more confident).
β = 1 early-stops at about half the rounds (95–224 vs 175–388).
**Mechanism:** with a conditionally bimodal target, 1/σ̂² weights hand the
tight low mode most of the μ-tree weight; the wide high mode is under-fit
(the β-NLL "vicious cycle"), and σ trees fitted to the tightened residuals
make the low-σ̂ rows overconfident. Not skew, not zeros, not Normal
misspecification per se (Cycle 7). A mixture/two-component head would be
the principled remedy (not implemented).

### Library changes this cycle
- `refit_intercept: Option<bool>` (opt-in; `Distribution::
  default_refit_intercept`, false everywhere) with per-round step folded
  into init_params after the loop, undone for truncated rounds.
- **Bug fix:** `partial_fit` no longer re-fits `init_params` when models
  exist (Python does; that silently changed what existing trees predict
  and, since 1.2.0, discarded calibration/intercept shifts folded into
  init_params). Test: `partial_fit_preserves_folded_calibration_shift`.
- `examples/protein_probe.rs`.

### Decision
No new defaults except pending the user's call on Bernoulli (evidence:
3 synthetic DGPs × 10 seeds, one learning rate — weaker than the Poisson
case). Next questions: (1) a two-component (mixture) Normal head for
multimodal targets like protein; (2) Categorical K > 2 natural gradient
weighting (non-diagonal) — needs a block-weighted tree fit; (3) Bernoulli
intercept step at lr 0.01 and on a real binary dataset.

### Cycle 8b — Bernoulli validation before a default (R26, n = 12k; user asked to validate first)
DGPs add `bern_imb6` (~6 % positives) and `bern_rare` (~2–3 %).
- Base-rate under-prediction of the default (unweighted) head is
  **structural** and grows with imbalance: train mean p̂/mean y = 0.94 /
  0.90 / 0.85 (12 % / 6 % / 2–3 % positives), unchanged at lr 0.01.
- `refit_intercept`: train ratio 1.000 everywhere; test log loss never worse
  in 10/10 dataset × config cells (−0.0000…−0.0011; 9/10 wins on the three
  imbalanced DGPs at lr .05, 5/5 on bern_rare at lr .01); ECE equal or
  better everywhere. Test base-rate ratios are within noise (±5–8 %).
- β = 1 (Newton/LogitBoost leaves) also restores the train base rate
  (1.000–1.003) and gains more on the nonlinear DGP (−0.010 10/10; −0.011
  5/5); worst cell across R20/R23/R26 +0.0004 (bern_lin).
Decision deferred to the user: intercept step (strictly safe, tiny gain)
vs β = 1 (fixes the same bias, larger gains on nonlinear structure).
**User decision (2026-11-09): β = 1 default for Bernoulli** (`Categorical<2>`
→ `default_fisher_weight_power() = Some(1.0)`; K > 2 unchanged).

## 2026-11-09 — Cycle 9: real binary data, Categorical K > 2, mixture head

### Bernoulli default on real data (R27)
Binary tasks built from the survival CSVs (labels = event indicators; time
and post-outcome columns removed): sprint CVD (5.4 % positive), flchain
death (27.5 %), support2 in-hospital death (25.9 %; rows carry an unnamed
index column — aligned). Old unweighted head vs the β = 1 default: old
under-predicts the positive rate in-sample (0.874 sprint, 0.957 support2)
and is worse on flchain (+0.0021, 0/10; lr .01 +0.0055, 0/5) and support2
(+0.0068 0/10; +0.0090 0/5); sprint tie. **Default confirmed.**

### Categorical K > 2 (R28, R31)
Explicit `fisher_weight_power` now also applies to non-diagonal metrics,
weighting each logit's tree by its diagonal Fisher entry p_k(1−p_k)
(XGBoost multi:softprob's diagonal-Hessian approximation; the target stays
the full natural gradient). Synthetic: cat3_nl −0.0086 (9/10), cat5_nl
−0.0062 (7/10; β = 0.5 −0.0071 8/10), cat3_imb neutral. Real: support2
disease class (4 classes) −0.0219 (10/10), + intercept step −0.0318
(10/10), accuracy +1.1–1.3 pt; lr .01 −0.027 (5/5); flchain cause of death
(weak signal) −0.002…−0.004, n.s. Never worse in any cell.

### NormalMixture2 (R29, R30)
New head (complete-data Fisher; see `docs/normal_mixture.md`), unit-tested
(density normalisation, cdf/ppf, FD gradients, complete-data ≥ observed
Fisher by quadrature, EM init). Recalibration generalised to a column set
(`scale_param_columns`, `calibrate_params_affine`). Numerical CRPS in the
harness agrees with the closed-form Normal CRPS to 0.05 %.
Calibrated mixture vs calibrated Normal (10 seeds): protein ΔNLL −0.461
(10/10), ΔCRPS −0.096, ΔRMSE +0.22; bimodal −0.786 / −0.285; unimodal
het_a +0.000, homo −0.025, kin8nm +0.004, tnoise +0.038 (n.s.). lr .01:
protein −0.450, bimodal −0.774 (5/5). Protein's β = 1 failure (Cycle 8)
was a symptom of the wrong family; the mixture is the remedy.

### Open decisions / next
- Categorical K > 2 default (β = 1, optionally + intercept step) — user.
- Mixture: point forecasts (median vs mean), K > 2 components, CRPS score.
**User decision (2026-11-10): Categorical K > 2 default = β = 1 (diagonal
Fisher) + per-round intercept step**; Bernoulli stays β = 1 only.

## 2026-11-10 — Cycle 10: mixture point forecasts, CRPS training, K > 2 mixtures, more real multiclass data

### More real multiclass data (R33)
New tasks: support2 `dzgroup` (8 classes, 9.1k), digits (10 classes, 1.8k)
and wine (3 classes, 178) — the latter two exported from the installed
scikit-learn (`python3 -I -c "from sklearn.datasets import load_digits"`,
written to `research/ngboost/data/*.csv`; no download). Upstream vs default:
dzgroup −0.082 (10/10; lr .01 −0.094 5/5), acc 57.3 → 59.6 %; digits
**−0.310** (10/10; lr .01 −0.317 5/5), acc 87.0 → 96.4 %. Wine is separable
in 1–2 rounds and uninformative (all arms ±3.5 NLL across seeds), but it
exposed an upstream bug: the Categorical natural gradient (Sherman–Morrison,
1/p) overflowed on saturated rows ("training diverged"). Fixed with a 1e-6
probability floor inside the solve.

### NormalMixture<K> (generalised) + CRPScore for mixtures and NegBin
- K components, softmax weights; complete-data Fisher with the weight block
  inverted in closed form. CRPScore in closed form (Grimit et al. 2006),
  analytic gradients, closed-form diagonal CRPS metric. NegBin discrete CRPS
  with full 2×2 metric. Unit tests: normalisation, ppf, FD gradients (both
  scores, K = 2/3), CRPS vs integral, metric diag vs quadrature, Fisher
  dominance, SM == LAPACK, EM recovery (K = 3), mode, NB CRPS vs brute force,
  NB metric == expected CRPS Hessian.
- Harness: runners generic over the training score; NLL always LogScore;
  exact CRPS for mixtures (the numerical integrator broke on a σ ≈ 2·10⁴
  component). A first R34–R36 attempt silently ran with default flags (zsh
  `$VAR` not word-split; parser ignored the trailing argument) — moved to
  `reports/invalid/`, parser now panics on leftovers.

### Results
- **Point forecasts (R34):** mean for RMSE, median for MAE (10/10 each way on
  multimodal sets), mode worst. Protein: mixture median MAE 3.48 < Normal
  3.62; Normal RMSE 4.59 < mixture mean 4.81.
- **CRPS training (R35):** mixture worse on NLL *and* CRPS everywhere (0/10;
  K = 2 ΔCRPS +0.007…+0.12; K = 3 likewise), 6× slower, one component
  collapse (het_a NLL 6.8e6). NegBin: CRPS/RMSE better only on nb_over
  (−0.056 / −0.24), NLL worse everywhere. Neither is a default.
- **K = 3 (R36):** trimodal −0.288, protein −0.053 (10/10); unimodal +0.02…+0.05.

### Next
Why CRPS training underperforms for mixtures (test the full CRPS metric vs
the diagonal); component pruning/σ guard for large K or n; a data-driven K
choice (validation NLL over K = 1..3).

## 2026-11-10 — Cycle 11: full CRPS curvature, choosing K, the σ blow-up

### σ blow-up — diagnosed and fixed (R37, R37b, R38)
Probe (`examples/mixture_sigma_probe.rs`, full protein, K = 3): not the
recalibration (b = 1.00005) and not a population drift (σ median/p99 stable):
ONE row jumped to σ = 2·10⁸ in a single round (338, lr·scale 0.05), with that
component carrying 94.5 % of the row's weight; 2 of 32 010 training rows
affected. Mechanism: a training row far in a component's tail (r ≈ 1, z ≈ 20)
gives a log σ natural gradient (1 − z²)/2 ≈ −200 — a linearisation of a move
whose exact per-point optimum is ln|z| — and with min_samples_leaf = 1 it can
own a leaf; the line search accepts (that row's loss drops).
- Guard v1, metric max(π, r·max(z², 1)) / max(π, r): no blow-ups, but it damps
  every row with r_k > π_k (most rows) → +0.03…+0.12 NLL; rejected
  (`reports/superseded/`). My first test even claimed it was inactive on
  typical rows — false; corrected.
- Guard v2 (default): cap each row's log σ step at 5 (entry max(2π, |g|/5));
  engages only ≈ 3.3σ into a tail. Full protein: 0 rows σ > 100 (3 seeds;
  unguarded 3 train / 2 test), test NLL 2.167/2.186/2.212 vs 2.208/2.218/2.214.
  12k grid: K = 2 neutral; K = 3 protein −0.019 (8/10), kin8nm −0.014 (10/10),
  tnoise −0.047; K = 4 neutral/better; no cell significantly worse.

### Full CRPS metric (R39) — hypothesis refuted
Closed form (Gaussian-product moments, Φ-cross probabilities, Stein's lemma,
energy distances), verified entry-by-entry against quadrature for K = 2, 3.
vs diagonal: better on bimodal/trimodal (−0.08…−0.18), worse on homo
(+0.52), het_a, protein, collapse on kin8nm (NLL 5·10⁵). vs LogScore training:
0/10 on all nine cells. CRPS training stays not recommended; the diagonal is
restored as the CRPS default (`NGBOOST_MIX_CRPS_FULL=1` for the full one).
Reproducibility: with both switches off the code reproduces R34/R35
bit-for-bit.

### Choosing K by validation NLL (R38) → `NGBAutoMixtureRegressor`
`research/ngboost/select_k.py` over Normal + K = 2, 3, 4 (guarded): the
validation-chosen head matched or beat the best fixed K on 7/7 datasets
(−0.001…−0.043), beat every fixed K on unimodal data, within 0.001–0.010 of a
test oracle. Implemented as `NGBAutoMixtureRegressor` (candidates 1..=4,
default all four; `n_components()`, `validation_nll`, mean/median/score/
cdf/ppf/interval). Integration test: picks the min-validation-NLL head, K ≥ 2
on bimodal data, sane 90 % coverage.

### Next
Per-component σ guards for the CRPS path; a cheaper K search (warm-start the
K + 1 fit from the K fit); real multimodal datasets beyond protein.

## 2026-11-10 — Cycle 12: warm-started K search, real multimodal data, CRPS guard

### Real multimodal data (R40)
Searched the drive, home and sklearn's bundled data (no downloads; the
~/Downloads files — OJ, Loans, Favorita — were deliberately not used: personal
folder, Loans empty, Favorita multimodal mostly across a feature). Targets
from the survival CSVs with outcome-revealing columns removed: flchain
follow-up days (early deaths + administrative censoring), support2 ln(days to
death/censoring), control ln(total cost > 0). Mixture vs Normal: −0.90 and
−0.29 NLL (10/10), control −0.005. Validation selection: K = 4 in 10/10 on
both multimodal targets (= oracle). 17 zero costs: a component collapsed onto
the atom (NaN) — excluded, documented.

### CRPS step guard (R41)
Cause of CRPS collapses: metric diagonal ∝ π²σ → natural-gradient step ∝
1/(π²σ). Cap per row (log σ, logits ≤ 5; μ ≤ 5σ). Guard off reproduces R35
bit-for-bit. Result: collapses gone (het_a max NLL 6.8e6 → 1.7), homo −0.27,
kin8nm −0.15 (10/10), multimodal ≈ neutral. Guarded CRPS vs LogScore: better
on unimodal (homo −0.14, het_a −0.05), worse on multimodal (protein +0.39).

### Warm-started K search (R42, R43)
`grow` builds a K-mixture from the (K−1) fit exactly (trees copied round for
round + constant shifts, zero trees for the new column; no format change),
then `partial_fit`. Misfit placement (Verbeek et al. 2003) stayed trapped
near the single Normal; split placement (Ueda et al. 2000) escapes it. R42:
warm = 46 % of cold cost but a different model — multimodal +0.17…+0.56,
unimodal −0.07…−0.11; greedy stop 84 % cost, same quality. R43: `KSearch::
Both` (cold + warm in one selection) is never worse than cold (multimodal
picks identical, 0/10 warm) and −0.07…−0.10 on unimodal (9–10/10; warm picked
9–10/10), at 118 % rounds / 141 % time; Both + greedy at 101 % rounds / 118 %
time. Cold rows reproduced R42 100/100. Default: `search = Both`, `greedy =
false`.

### Next
Why warm-grown mixtures beat the Normal head on unimodal data (extra μ rounds
past the Normal's early stop vs genuine non-Gaussian residual structure —
test by continuing the Normal head for the same number of rounds); point-mass
+ continuous (hurdle) heads; Both-search for the CRPS-trained head on
unimodal targets.

## 2026-11-10 — Cycle 13: why warm mixtures win, hurdle heads, CRPS search

### Why warm-grown mixtures beat the Normal head (R44)
Probe `--dist normprobe` on the R43 splits. Continuing the Normal for the
rounds the split adds: 4–17 % of the gain. A symmetric split (two identical
halves) equals "continue the Normal" to 4 decimals (control). The
moment-matched Normal of the grown mixture scores the same as the mixture on
het_a/homo/het_b, so the gain is the *mean* (homo RMSE 1.44 → 1.26, σ = 1),
not the shape. The shape matters only on tnoise (0.025 of 0.10), kin8nm
(about half) and protein (all). Mechanism (`trace`, no early stopping): the
validation-RMSE optimum is at 1 165–1 918 rounds vs the NLL optimum at
102–387; at the RMSE optimum the uncalibrated test NLL is 4.4, because σ,
fitted to in-sample residuals, collapses. The joint line search also settles
at scale 1 vs 2 for μ-only boosting.

### Fixing the Normal head (R45–R48)
- Per-parameter line search: worse everywhere. Calibrated-NLL early
  stopping: −0.003…−0.068, about 40 % of the staged fit.
- Staged mean→σ fit: −0.08…−0.16 except het_b (+0.049).
- Adding FGLS (refit the mean weighted by 1/σ̂², then σ): never
  significantly worse; −0.07…−0.16 NLL on 5 synthetic sets + kin8nm,
  −0.025 protein, −0.011 support cost. Cost 1.3–3.7× the tree fits.
- Joint polish after the stages: no help.
- Implemented as `NGBoost::<Normal, LogScore, _>::fit_two_stage` and
  `NGBRegressor::fit_two_stage` (opt-in). It reproduces the bench to
  4 decimals and serializes as an ordinary Normal model.
- Found on the way: a single outlier row alone in a leaf drove ln σ to 40
  (test CRPS 4·10¹³), from the uncapped LogScore log-σ step (z² − 1)/2.
  Guard: |ñ_logσ| ≤ cap. Cap 5 was too tight (+0.02 het_a/tnoise); cap 25 is
  bit-identical on 8/10 sets and removes every blow-up (worst-seed CRPS
  423 → 3). Default on for Normal and NormalFixedMean
  (`NGBOOST_NORMAL_GUARD`).

### Normal CRPS metric bug (R51, R52)
The CRPS-trained Normal (existing `NGBoost<Normal, CRPScore>`) blew up on
6/10 sets (NLL up to 10³³). Cause: the metric diag(1/√π, σ²/(2√π)),
inherited from upstream, is σ × the true 2∫∇F∇Fᵀ = diag(1/(√πσ), σ/(2√π)),
so steps scale with 1/σ. Fixed for Normal, NormalFixedMean and LogNormal.
Verified by quadrature and by unit equivariance (both tests fail on the old
metric); `NGBOOST_CRPS_METRIC_LEGACY=1` restores it. Fixed vs legacy: 8–10/10
seed wins on every set. Against the LogScore Normal: lower CRPS on all 10,
mixed NLL (flchain 13.8 — thin tails).

### Hurdle heads (R49, R49c, R49d)
`Hurdle<D>`: atom at 0 (logit π) plus any continuous head; exact
block-diagonal Fisher; natural gradient = Bernoulli step + D's own ÷ (1 − π)
on non-zero rows. Unit tests: FD gradients, metric = E[∇∇ᵀ] by quadrature,
the three natural-gradient paths agree, cdf/ppf at the atom.

Data: zi_lognorm, zi_normal (synthetic), support2 `adlsc` (35 % zeros), raw
`totmcst` (1.6 % zeros, 2 negative entries dropped). CRPS by split
quadrature (matches the closed form to 3·10⁻⁵). Results:
- zi_lognorm: HurdleLogNormal 2.04 vs Normal 3.40 (10/10).
- zi_normal: HurdleNormal −0.092 (10/10).
- adlsc: −0.054…−0.065 (10/10); Brier of P(0) 0.347 → 0.206.
- totmcst: HurdleLogNormal −12 % CRPS (10/10); the continuous part matters
  more than the small atom.
- Auto mixture without the atom: CRPS 10⁵² / NaN (collapse onto the atom).

### CRPS-trained combined search (R50a–d)
`NGBAutoMixtureRegressor<S>` is generic over the training score (`new_crps`;
selection by validation CRPS; `crps()` added). 6 seeds (CRPS mixtures are
slow; the 10-seed run timed out and was cut). Against the LogScore `Both`
search:
- CRPS better on kin8nm, het_a, homo and tnoise (−0.006…−0.037, 5–6/6);
  neutral on het_b; worse on bimodal (+0.026 CRPS, +0.13 NLL).
- NLL mixed.
- Warm candidates add nothing under CRPS (|Δ| ≤ 0.0025): the CRPS-trained
  Normal does not under-fit its mean, which confirms the R44 mechanism.

### Process notes
- The 30-min background limit killed two runs (R49 after 3 of 4 datasets,
  R50). Partial files are kept (`R49_hurdle_partial.csv`,
  `invalid/R50_*`). The first R50 attempt ran a pre-fix binary (legacy
  CRPS metric) and is in `invalid/`.
- The existing tests' `Lcg::next` returns values in [0, 0.5); the first
  staged-fit test on that generator showed no gain. Corrected generator in
  `tests/test_cycle13.rs`.

### Next
- Default decision for `fit_two_stage` (user's call: cost and behaviour
  change).
- The same staged idea for other location–scale heads (Laplace, StudentT,
  LogNormal).
- Audit the CRPS metrics of the remaining heads (Gamma, Weibull,
  Exponential, HalfNormal, StudentT).
- A hurdle CRPS rule.

## 2026-11-11 — Cycle 14: staged default, staged for other heads, CRPS metric audit, hurdle CRPS

### Staged fit made generic and the default (R54a, R57)
`NGBoost::fit_staged` (any head with `Distribution::location_column()`):
frozen columns get zero gradient steps; one validation split for all
stages; stage 3 weights = the location's Fisher entry. The new
`staged_fit: Option<bool>` (also in `NGBoostParams`) defaults per head via
`default_staged_fit()` (LogScore only, with early stopping and validation
data). `fit_two_stage` now calls it.
- Generic vs the Cycle 13 Normal-only implementation: within noise. The
  difference traced to the binary line search choosing scale 1 or 2 by
  floating-point rounding: for an LS tree the full-step loss at scale 2
  equals the start exactly. This also **retracts** Cycle 13's "σ limits the
  joint scale to 1" factor.
- R54a, Normal staged vs joint, 10×10: −0.083…−0.149 on synthetic and
  kin8nm, −0.022 protein, −0.010 support log cost; neutral on the rest;
  never significantly worse.
- R57: `NGBAutoMixtureRegressor` (its K = 1 candidate is now staged):
  identical on multimodal sets, better on unimodal (kin8nm −0.025 5/5,
  homo −0.032 4/5).

### Other heads (R54b–d)
- LogNormal: staged −0.011…−0.118 (9–10/10) → **default**.
- StudentT: −0.05…−0.51, 49/49 seed wins; joint diverged in 11 runs →
  **default**. The joint StudentT run also crashed the process: statrs'
  t cdf panics on NaN; now NaN-safe (`CdfNan`).
- Laplace: big synthetic gains but protein +0.25, support log d.time
  +0.06, flchain +0.03 (0/10) → stays joint (opt-in).

### CRPS metric audit (tests/test_crps_audit.rs, docs/crps_metrics.md)
Score and gradient were correct for every head. The metric was wrong for
HalfNormal, Gamma, Weibull, StudentT, TFixedDf, TFixedDfFixedVar and
Poisson: upstream uses E_Y[g gᵀ], whose ratio to the true 2∫∇F∇Fᵀ scales
with σ (×10 scale → ×10 ratio) or with λ. Replaced with exact or
dimensionless-quadrature metrics, all ≤ 3·10⁻⁵ from the definition at every
scale. Effects:
- The documented "diverging Weibull CRPS, rescued by Tikhonov" parity case
  now converges unregularised (CRPS 0.368; legacy still diverges).
- R55 HalfNormal: raw costs NLL −0.60, CRPS −8 795, 90 % coverage
  0.83 → 0.92 (6/6); small sets −0.003…−0.074.
- Poisson: neutral at these rates.
- StudentT (4 seeds): tnoise −0.44, het_a −0.32 NLL (4/4); legacy diverged on
  2/4 het_a seeds. Gamma/Weibull training runs not completed (metric verified
  by the audit and by the Weibull parity case only).

### Hurdle CRPS (R56)
`Scorable<CRPScore>` for `Hurdle<Normal>` and `Hurdle<LogNormal>` via
y-scale pieces (`HurdleCrpsPart`; Baran & Lerch for LogNormal), with the
closed-form 3×3 metric. Verified by quadrature. As a training rule it loses
to LogScore hurdle training on CRPS in 5/7 cells and on NLL everywhere →
implemented, not recommended.

### Process
The laptop slept mid-run: three 2-hour jobs were killed with partial
output. Reruns were cut to 4–6 seeds; partial files kept or moved to
`invalid/`.

## Cycle 15 (2026-11-10): closing Cycle 14's evidence gaps

Request: the weakest evidence from Cycle 14, namely StudentT at 4–6 seeds,
Gamma/Weibull CRPS training never benchmarked, and the "each change has a
switch" parity claim never tested.

### Upstream parity (tests/test_upstream_parity.rs)
Built the 1.1.2 source (`git archive 7c91877`, target
`/Volumes/Build/ngboost-rs-v112`) and the current tree with the same probe:
26 head/score combinations (LogScore with/without validation, CRPS for
every head that has it, Poisson, Bernoulli, Categorical3), exact hash of
the test-set parameters.
- All six opt-outs set (`auto_calibrate_scale = false`,
  `fisher_weight_power = Some(0.0)`, `refit_intercept = Some(false)`,
  `staged_fit = Some(false)`, `NGBOOST_NORMAL_GUARD=0`,
  `NGBOOST_CRPS_METRIC_LEGACY=1`): **26/26 bit-identical** to 1.1.2.
- Current defaults: 20/26 differ. The 6 identical ones are where no new
  default applies (Normal without validation, where the guard never
  engaged on this data; NormalFixedMean/Var; Laplace and Exponential CRPS;
  Gamma/Weibull LogScore).
- Mutation check: dropping only `staged_fit = Some(false)` fails exactly
  the Normal (val, valfrac), LogNormal and StudentT validation cases.
Kept as a platform-gated test with the recorded hashes. README gained a
"Reproducing upstream (1.1.2) behaviour" section; the old "set
`auto_calibrate_scale = false` for identical defaults" line was stale.

### CRPS audit extended
NegativeBinomial, HurdleNormal, HurdleLogNormal and NormalMixture<2> added
to `crps_metrics_match_definition`: all match 2∫∇F∇Fᵀ (mixture: its
diagonal, by design) with ratio 1 at every scale. The reference grid now
puts 0 and y on cell boundaries. The hurdle cdf jump at 0 and the
indicator jump at y had cost O(h) accuracy (10⁻⁴ metric, 10⁻² t-family
score). Score tolerance tightened 1e-2 → 1e-3.

### R54c+e: StudentT staged vs joint at 10 seeds
- NLL −0.06…−0.46 on all 8 datasets, **79/80 seed wins**. The loss is
  homo seed 7: +0.09, 90 % coverage 0.58.
- CRPS better on 73/80.
- **Correction to Cycle 14:** the 11 joint "divergences" were not NaN
  fits; NLL is finite in all 160 runs. Those fits put ν ≤ 1, or close to
  it, on some test rows, so CRPS is undefined or > 10³. That happened in
  15/80 joint fits and 1/80 staged.
- Staged 90 % coverage is lower: 0.86–0.92 vs 0.92–0.94.
→ default unchanged.

### R58: Gamma/Weibull CRPS, fixed vs legacy metric (and vs LogScore)
- Gamma: fixed better everywhere.
  - lognorm_het: NLL −0.29, CRPS −19 % (10/10).
  - raw costs: NLL −0.39, CRPS −24 % (10/10).
  - kin8nm: NLL −0.13 (4/4).
  - Ties LogScore Gamma on NLL.
- Weibull: fixed wins CRPS on every seed (23/23) and removes the legacy
  blow-ups.
  - Legacy raw-cost fits never pass round 1.
  - Legacy test NLL ≥ 2 nats worse (up to ∞) on 5/10 lognorm_het seeds.
  - Legacy train NLL is ∞ on all kin8nm seeds.
  - But a CRPS Weibull loses NLL to LogScore Weibull (median +0.06…+0.10)
    and is overconfident (90 % coverage 0.82–0.89).
- Legacy kin8nm runs were cut to 3–4 seeds: 15 min per Weibull fit with
  the legacy metric.

### R55b: StudentT CRPS, fixed vs legacy metric
- Seeds: 10 per dataset (kin8nm legacy 5–9 added in R55c). Legacy kin8nm
  seeds 5 and 9 ran 2000 and 1944 rounds (48–52 min) without early
  stopping and still lost: NLL 0.021 vs −0.146 for fixed after 35 rounds.
  With σ-scaled metric errors the steps are too small to converge.
- NLL better on **30/30** seeds: median −0.37 tnoise, −0.34 het_a, −0.13
  kin8nm.
- CRPS better on 27/30: median −0.37, −0.41, −0.02.
- ν ≈ 1 rows (CRPS > 10³) occur in both arms: legacy 2, fixed 2. One legacy
  tnoise fit has test NLL 7.6·10⁴².
- Consistent with R55 at 4 seeds; all 9 overlapping R55 legacy rows
  reproduce exactly.

### Process
- Long serial legacy jobs (Gamma/Weibull kin8nm, StudentT CRPS) would have
  overrun the 2-hour background limit and taken the queued jobs with them.
  They were stopped and the remaining seeds rerun as parallel per-seed-range
  jobs. Completed seeds were kept; reruns are deterministic, and the
  9 overlapping R55 rows reproduce exactly.
- Bench gained LogScore `gamma` / `weibull` dists.

### Follow-up (2026-11-11): kin8nm completion attempt
- Goal: bring every legacy kin8nm arm to 10 seeds, one process per seed
  (`c15b_jobs.txt`).
- The laptop was on battery and slept, and another session held load at
  30–38. After 93 min wall the legacy fits had ~9.5 CPU-min each.
- Stopped to spare the battery. Only StudentT seeds 6–8 finished: 8/8 NLL
  wins, median −0.13, unchanged picture.
- Resumed (`c15c_ordered.txt`, 6 parallel, cheapest first, battery guard
  `c15_run_guarded.sh`: no new job below 25 % on battery). Gamma and
  Weibull kin8nm are now 10 seeds in every arm (R58c):
  - Gamma, fixed − legacy: NLL −0.145 (se 0.014), CRPS −0.021, 10/10.
    Fixed − LogScore: +0.007 (5/10), a tie.
  - Weibull, fixed − legacy: CRPS better 10/10 (median −0.012), but NLL
    only even (median −0.02, 5/10).
  - Legacy Weibull: train NLL ∞ on 4/10, test NLL ≥ 10¹² on 3/10. On the
    other 7 seeds fixed NLL is worse on 4 (up to 1.56 vs −0.33).
  - Fixed 90 % coverage drops 0.88 → 0.83.
  - The 3-seed reading ("fixed NLL worse on 2/3") was too pessimistic; the
    10-seed one is mixed.
  - Fixed Weibull − LogScore Weibull: NLL +0.072 median (2/10), CRPS
    −0.005 (10/10).
- StudentT legacy kin8nm seeds 5 and 9 also finished (R55c), so every
  Cycle 15 arm now has 10 seeds. Nothing is left to resume.
- Deleted the 1.1.2 build cache (`/Volumes/Build/ngboost-rs-v112`). The
  parity test needs only its recorded hashes.


## Cycle 16 (2026-11-11): Poisson, NegativeBinomial and Normal — head-level and time-series gaps

User scope: performance of the Poisson, NegativeBinomial and Normal heads
(their data is time series). Prior state: β = 1 default for counts;
staged fit + P2 default for Normal; the NB head had neither a staged fit
nor any validation recalibration. Every earlier benchmark used iid random
splits, so known-limitations item 9 ("random validation is too optimistic
on time series") was unmeasured.

Literature: Thomas et al. 2018 (Stat. Comput., doi 10.1007/s11222-017-9754-6,
via Lacuna) — cyclical vs noncyclical boosted GAMLSS; per-parameter stopping
iterations are the closest precedent to the staged fit. Chronological
partitioning (Lacuna direction 25268) is standard evaluation practice.

### Hypotheses
- H16a: NB's r, fitted to in-sample residuals, is too large out of sample
  (the σ story of Cycle 2), so a validation affine map on log r (P2 for
  dispersion) helps. Competing: r is weakly identified on Poisson-like data,
  so a noisy map could hurt.
- H16b: the staged fit helps NB as it helps Normal (joint stopping
  under-trains μ). Competing: r overfits the better μ and gets overconfident.
- H16c: on time series a random validation split misleads early stopping and
  P2; refitting on train + val recovers the most recent block; a post-hoc
  location shift on that block corrects level drift.

### Code
- `NegativeBinomial::location_column() = Some(0)` (staged opt-in) and
  `scale_param_index() = Some(1)` (default end-of-fit log-r calibration).
- `Distribution::auto_calibration_min_rows()` (default 40; NB 300).
- `fit_staged` stage 3: with Fisher-weighted learners (β > 0) the stage-2
  location Fisher weights are divided by the stage-3 learner's own
  F^β (others at their marginal constants) — otherwise the leaf is
  Σ F² (μ−y)/μ / Σ F², not the FGLS Newton step Σ F(r̂)(μ−y)/μ / Σ F(r̂).
  Unchanged for β = 0 (Normal default).
- `examples/ts_bench.rs`: synthetic lag-feature series (Normal AR(2)+ARCH-like
  σ, log-linear Poisson / NB(r = 5) autoregressions; `_drift` = latent
  random-walk level, plus σ growth for Normal; oracle = DGP), and statsmodels'
  co2 (weekly change), elnino (monthly), sunspots (yearly counts) copied to
  `data_ts/` with 5 rolling origins. Variant tokens chrono/random/joint/
  staged/nocal/refit/carry/loccal/rcal/oracle.
- `research_bench` NB tokens staged/joint/rcal/auto; `c16_pair.py`.
- Tests: `negbin_auto_calibrates_log_r_on_validation_only`,
  `negbin_staged_fit_is_opt_in_and_sound`. Suite 509 passed, 0 failed;
  upstream parity still bit-identical.

### Results (10 seeds unless noted; Δ = variant − reference, NLL)
(R59–R62 ran on binaries built before the NB calibration default, so their
NB `base`/`chrono` rows are uncalibrated NB; R63 uses the final library.)
- **H16a holds, with a row floor.** R59/R63 (iid, 300 val rows): −0.026
  (10/10) nb_over, −0.005 (10/10) nb_low, −0.022 (9/10) pois_nl, −0.002
  (7/10) pois_lin, −0.005 (10/10) pois_low; nb_over cov90 0.867 → 0.897,
  calib 1.18 → 1.05. Time series (R60, 400 val rows): −0.016 (10/10)
  tsnb_stat, −0.014 (10/10) tsnb_drift, −0.008 (10/10) tsp_stat. The default
  path (auto) reproduces the explicit map bit for bit. With 100 val rows
  (shift only) it lost on nb_low and pois_nl (+0.004, 3/10) → NB floor 300.
- **H16b: mixed → opt-in.** Staged NB alone: nb_over +0.061 (1/10), cov90
  0.79 (the competing explanation), CRPS −0.057. Staged + calibration ≈
  calibration alone on NLL (−0.025 vs −0.026; pois_lin −0.007 vs −0.002),
  CRPS −0.107 on nb_over, 2–3× rounds. Sunspots: +1.05 (0/5). Corrected vs
  raw stage-3 weights: same picture (R59r).
- **H16c, validation split:** random − chrono, synthetic: Normal +0.031
  (3/10) / +0.143 (1/10); Poisson +0.032 (3/10) / −0.000 (5/10); NB +0.010
  (4/10) / +0.002 (5/10) / +0.020 (2/10). co2 +0.031 (0/5). elnino and
  sunspots favoured random (−0.09, −0.25/−0.42), with 30–70 val rows and
  high variance.
- **P2 on time series:** nocal − default: tsn_stat +0.044 (0/10), tsn_drift
  +0.239 (0/10; cov90 0.75), co2 +0.022 (0/5), elnino +0.67 (0/5). Fitted
  on the recent block, P2 absorbs variance drift.
- **Refit on train + val (H16c):** Poisson −0.010 (7/10) / −0.049 (8/10,
  median −0.094); NB +0.015 / +0.002 / −0.008; Normal joint refit loses P2
  (+0.013 / +0.027 vs joint); with the calibration carried over (R61) it is
  neutral (+0.010, +0.002, co2 +0.005, elnino +0.002).
- **Location shift on the recent block (loccal):** Normal ±0.015; Poisson
  drift −0.076 mean, median −0.026 (7/10), se 0.045; NB ±0.003. Rejected.
- **Staged vs joint Normal on time series (R60–R62):** n = 600–4000:
  −0.04…+0.07 NLL, within noise; CRPS better in 7 of 8 cells. elnino: +0.13
  (0/5). On iid data with 60 or 150 val rows staged still wins (−0.04…−0.14,
  7–10/10; tnoise neutral at 60), so the elnino loss is not a val-size
  effect. Default unchanged; documented.
- Oracle gaps remain large for all three heads (Normal 0.21/0.30, Poisson
  0.10/0.35, NB 0.07/0.08 stationary/drift): the remaining error is mean
  estimation and unobserved level, not calibration.

### Decision
- **Default:** NB log-r validation recalibration (≥ 300 val rows).
  Opt-out `auto_calibrate_scale = false`.
- Opt-in: staged NB. Rejected: loccal. Workflow guidance (README): validate
  chronologically, keep P2 on, consider refit for Poisson.

### Next questions
- Poisson has no dispersion column to recalibrate; on drifting series
  its cov90 is 0.82 with dispersion index 1.26 — a validation-fitted
  Poisson → NB switch (or dispersion test) could be automated.
- Why the staged Normal fit loses on elnino: small n, or strong seasonal
  lag structure where joint stopping is not the bottleneck?
- Rolling-origin refits for Poisson as a library helper (`refit_full`)
  if the user wants it.

## Cycle 17 (2026-11-11): Poisson → NB switch, the elnino staged loss, a refit helper

User request: follow up Cycle 16's three next questions.

### Q1. Automatic Poisson → NegativeBinomial switch (R64)
Hypotheses: (a) a validation dispersion test on the Poisson fit picks NB
when it helps and stays Poisson on true-Poisson data; (b) competing:
choose the head by validation NLL. Rule (a) uses the Dean–Lawless score
statistic for NB2 over-dispersion (Dean & Lawless 1989, JASA 84;
reference from memory, literature tools not consulted for it),
z = Σ[(y − λ̂)² − y] / √(2 Σ λ̂²), on out-of-sample validation λ̂, so
estimation error in λ̂ also counts as (predictive) over-dispersion.
`ts_bench --dist countsel` fits both heads once and reports Poisson, NB,
sel_nll, sel_z95 (z > 1.645), sel_z99 (z > 2.326). New iid controls
(`iid_*` = research_bench's count DGPs).

| data | Poisson → NB | z95 picks NB | z95 Δ vs Poisson |
|---|---|---|---|
| tsp_drift (Poisson + latent level drift) | −0.092 (9/10), cov90 0.82 → 0.88 | 10/10 | −0.092 |
| tsnb_stat / tsnb_drift | −0.63 / −0.55 (10/10) | 10/10 | same |
| iid_nb_over / iid_nb_low | −0.80 / −0.072 (10/10) | 10/10 | same |
| sunspots (5 origins) | −1.94 (5/5), cov90 0.53 → 0.87 | 5/5 | same |
| tsp_stat (true Poisson) | −0.004 (6/10) | 6/10 | −0.006 |
| iid_pois_nl | −0.014 (8/10) | 9/10 | −0.014 |
| iid_pois_lin | +0.003 | 2/10 | +0.002 |
| iid_pois_low | +0.003 | 0/10 | 0.000 |

sel_nll was a coin flip on iid_pois_low (+0.0025, NB 5/10): NB's
dispersion is recalibrated on the same validation rows, so its val NLL is
optimistic. z99 ≈ z95. → **`CountModel::fit_auto`** (`src/count_auto.rs`):
Poisson fit, z on validation, NB with the same settings when z > 1.645;
`fit_auto_with_threshold`, `dispersion_z`, pmf/cdf/ppf/interval/score,
`refit_full`. Reproduces the bench's sel_z95 exactly on 12 fits (z equal to
1e-12; summation order differs by 2 ulp).

### Q2. Why the staged Normal fit loses on elnino (R65, R66)
Four explanations, tested on 10 origins (5 % test blocks, 72 val rows):
- Small training set? **No**: iid at n = 700 (490 train, 70 val), staged +
  calibration wins −0.012…−0.067 (6–9/10).
- No mean gain to collect? **Yes**: elnino test RMSE staged vs joint equal
  (staged lower on 4/10); co2 also equal. Lag/seasonal structure is fitted
  well by the joint fit's early stop.
- Shift-only calibration can't fix the staged σ? **Partly**. Uncalibrated
  staged is far too narrow: elnino +0.72 (0/10, cov90 0.68) vs joint
  unaffected by calibration (+0.0006). Calibrated, the mean loss is +0.12,
  median +0.04 (2/10).
- Validation overfitting from four early stops on 72 rows? **No**: staged
  val NLL lower on only 3/10 origins.
Half the mean loss is one window (origin 9, ≈ the 1982–83 El Niño):
+1.05, where the staged fit's sharper σ (0.42 vs 0.50) is punished. A
sharpness-vs-shock trade-off with no mean gain to pay for it.

**Hazard found on the way.** Staged without calibration (R65/R66):
iid n = 700: +0.8…+14 NLL (0–2/10, cov90 0.42–0.67); n = 3000: +0.04…+0.21
on 4/5 (het_b −0.08); 30 val rows (calibration cannot run): +0.11…+0.21 on
3/5. CRPS was better in every case, so the location is better and the
uncorrected σ is the problem: the scale stage fits residuals of a location
trained ~4× longer. The default only staged when validation data existed,
not when the calibration would run. → **`uses_staged_fit(n_rows, n_val)`**:
the default (`staged_fit = None`) now also requires `auto_calibrate_scale`,
a scale column and ≥ `auto_calibration_min_rows` (40) validation rows;
`Some(true)` still forces it. Unchanged for every earlier benchmark (all had
≥ 60 val rows and calibration on); upstream parity bit-identical.

### Q3. Refit helper (R67)
`NGBoost::refit_full(x, y, w, rounds)`: same hyperparameters
(`NGBoost::unfitted_like::<D2>()`, new), the kept round count (or
`rounds`), no early stopping, the source's (a, b) recalibration folded in;
refuses staged fits (new private `last_fit_staged`). Bit-identical to
Cycle 16's manual refit (Poisson) and carry (Normal) on spot checks.
Kept vs scaled rounds ((train + val)/train):

| head / data | refit (kept) | refit (scaled) |
|---|---|---|
| Poisson tsp_stat / tsp_drift | −0.010 (7/10) / −0.049 (8/10) | +0.011 / −0.052 |
| Poisson iid nl / lin / low | −0.001 / −0.004 (7/10) / +0.002 (4/10) | −0.004 / −0.005 / +0.002 |
| Poisson sunspots | −0.63 (4/5) | −0.68 (5/5) |
| NB (now calibrated) tsnb_stat / tsnb_drift / tsp_drift | −0.002 (7/10) / −0.003 / −0.028 (5/10) | −0.004 / −0.005 / −0.033 |
| Normal joint tsn_stat / tsn_drift | +0.010 / +0.002 | +0.002 / +0.011 |

Scaling the rounds is not consistently better (tsp_stat flips to a loss)
→ default = kept rounds. With NB's calibration carried over, NB refit is
now mildly positive (it was mixed without calibration in R60).

### Decisions
- New API: `CountModel::fit_auto` (opt-in helper), `refit_full`,
  `unfitted_like`.
- Default change (safety): the staged default requires the calibration
  to run.
- elnino: documented trade-off; staged default unchanged where the
  calibration runs.

### Next questions
- A staged refit (replaying per-stage round counts) — only if a Normal
  user needs refit; Normal joint refit was neutral.
- Shock robustness of sharp σ on small series: a σ floor relative to the
  joint fit, or a tail-weighted calibration, measured on more real series.
