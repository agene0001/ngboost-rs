# Optimization-algorithm research log

Continues `research/ngboost/research_log.md` (Cycles 0–17). Same harness
conventions: `examples/research_bench.rs` data, 70/10/20 split per seed,
early stopping on the validation block, lr 0.05, depth 3, round cap 2000,
patience 50, protein subsampled to 12 000 rows, 10 seeds unless noted.
Δ = candidate − reference on held-out test, negative is better; "(k/10)"
is the number of seeds the candidate wins on NLL.

Commands and files: `reports/README.md`. Proposals with the mathematics:
`proposals/O1…O4`.

## 2026-11-13 — Cycle 18

Branch `research/cycle18`, base commit 70b314b (1.2.0). Nothing here
changes a default, a public API of an existing type, or serialization.

Four questions, chosen from the open items and from what Cycles 0–17
never varied:

| | Question | Outcome |
|---|---|---|
| O1 | Does fitting the scale to cross-fitted location residuals remove the overconfidence at its source? | **Yes.** Beats the calibrated default with no calibration at all; one small-n loss on protein |
| O2 | Are the trees under-regularised at `min_samples_leaf = 1`? | **Yes**, for the location trees. Leaf 20 was never significantly worse in 38 (head, fit, dataset, n) cases |
| O3 | Does a ν floor fix StudentT's undefined mean? | **Yes, at ν_min = 2.1** (not at exactly 1 or 2): 22/80 undefined means and 60/80 infinite variances → 0, mean ΔNLL −0.001, worst +0.019 |
| O4 | Does the Gamma head lose accuracy to its coordinates? | **Yes.** `GammaMean` wins on NLL everywhere it was tested once staged; base head has mean blow-ups |

### O2 — leaf regularisation (`proposals/O2_leaf_regularisation.md`)

Baseline: unweighted leaf means of the natural gradient, depth 3, no lower
bound on leaf size. Candidate: `min_samples_leaf = m`.

**Sweep at the default shape, Normal head, 1.2.0 default (staged) fit**
(`reports/O2_leaf_sweep.csv`), ΔNLL vs m = 1:

| dataset | m = 5 | m = 10 | m = 20 | m = 50 | m = 100 | m = 200 |
|---|---|---|---|---|---|---|
| het_a   | −0.010 (6/10) | −0.023 (7/10) | −0.061 (9/10) | −0.110 (10/10) | −0.160 (10/10) | −0.134 (10/10) |
| het_b   | −0.003 (5/10) | −0.013 (9/10) | −0.014 (9/10) | −0.012 (7/10) | **+0.038 (0/10)** | **+0.156 (0/10)** |
| homo    | −0.014 (7/10) | −0.061 (10/10) | −0.048 (9/10) | −0.092 (10/10) | −0.121 (10/10) | −0.118 (10/10) |
| tnoise  | −0.011 (5/10) | −0.034 (8/10) | −0.062 (9/10) | −0.139 (10/10) | −0.176 (10/10) | −0.152 (10/10) |
| kin8nm  | −0.014 (8/10) | −0.017 (9/10) | −0.025 (10/10) | −0.043 (10/10) | −0.060 (10/10) | −0.073 (10/10) |
| protein | −0.014 (7/10) | −0.010 (8/10) | −0.016 (8/10) | +0.010 (8/10, se 0.038) | −0.010 (7/10) | −0.019 (7/10) |

RMSE moves with NLL (m = 50: −4…−7 % on het_a, homo, tnoise). 90 %
coverage is unchanged or closer to 0.90 (homo 0.82 → 0.88).

P1 holds: monotone gain up to a dataset-dependent m*, then under-fitting
(het_b from m = 100; its noise scale shares the mean's dominant feature and
needs fine resolution along it).

**Joint fit** (`staged_fit = Some(false)`), same sweep: m = 20 −0.012 on
average (het_a +0.012, 4/10, se 0.019; the other five −0.002…−0.045);
m = 50 −0.02…−0.10 on all six; m = 200 −0.09…−0.21 on four, +0.178 on
het_b. Not specific to the staged fit, though the joint fit wants larger
leaves before it responds.

**Small n** (700 rows = 490 train / 70 validation,
`reports/O2_leaf_sweep_n700.csv`): m = 20: het_a −0.143 (10/10), het_b
−0.126 (8/10), homo −0.075 (7/10), tnoise −0.177 (10/10), kin8nm −0.017
(7/10), protein −0.001 (6/10). m = 5, 10 and 50 each lost on protein there
(+0.06…+0.12, 2–3/10, se 0.04–0.09).

**Which trees?** (`reports/O2_leafsplit.csv`; manual staged recipe with
separate leaf sizes, m = 50 vs 1):

| | location m = 50, scale m = 1 | location m = 1, scale m = 50 | both 50 |
|---|---|---|---|
| mean ΔNLL over 6 datasets | **−0.057** | −0.003 | −0.057 |
| mean ΔRMSE | −0.038 | −0.002 | −0.035 |

P2's first branch is false: it is the **location** trees that over-fit.
The scale trees are indifferent. This also explains the mechanism behind
limitations 12/33: single-row and few-row location leaves fit those rows'
noise, the in-sample residuals shrink, and σ̂ follows them down.

**Other heads** (`reports/O2_heads_*`; library defaults for each head,
m = 20 vs 1, 10 seeds):

| head | dataset: ΔNLL (wins) |
|---|---|
| Poisson | pois_lin −0.0001 (7), pois_nl −0.043 (9), pois_low −0.0005 (5), nb_over −0.061 (8) |
| NegativeBinomial | nb_over −0.005 (6), nb_low −0.002 (6), pois_nl −0.014 (8) |
| Bernoulli (real) | sprint_cvd +0.0003 (5), flchain_death −0.0010 (7), support_hospdead −0.0002 (5) |
| Laplace | het_a −0.100 (8), tnoise −0.094 (7), kin8nm −0.025 (7), support_logcost −0.001 (5) |
| LogNormal | lognorm_het −0.027 (8), kin8nm −0.036 (8), support_cost +0.002 (4; se 0.0014) |
| GammaMean (staged) | gamma_het −0.014 (6), lognorm_het −0.037 (10), kin8nm −0.049 (10) |

P3 holds. Over all 38 cases (Normal staged at two sample sizes, Normal
joint, and the 20 above) the largest unfavourable Δ is +0.012 (0.7 se;
joint Normal on het_a), the next +0.002 (1.5 se; LogNormal on
support_cost).

**Decision.** By the rule declared in the proposal, m = 20 qualifies as a
default candidate: never significantly worse, on 7 heads, 14 datasets and
two sample sizes; mean −0.04 NLL at n ≈ 2–8 k and −0.09 at n = 490 for the
Normal head. **Not changed** — it is a default of `HistogramLearner::new`
and of the fitted models, so it needs approval, and three things are
untested: n below ≈ 500 training rows, weighted fits where a "row" is not
a unit of information, and the exact (non-histogram) tree learner. Until
then: `min_samples_leaf` is the first knob to tune, range ≈ 5–200 on a log
scale.

Not implemented: per-parameter leaf sizes in the library (unnecessary:
the scale trees do not care) and L2 leaf shrinkage (candidate C).

### O1 — cross-fitted residuals for the scale stage (`proposals/O1_crossfit_scale.md`)

Harness-only prototype (`c18_bench --mode oof`): the four staged steps as
separate `NormalFixedVar` / `NormalFixedMean` fits; the scale steps see
5-fold out-of-fold location residuals. `ts_in` / `ts_in_cal` are the same
recipe with in-sample residuals, without / with the affine recalibration.

*Emulation check:* `ts_in_cal` − library staged default = −0.005 NLL on
average (within noise on all six), `ts_in` ≈ library staged without
calibration (+0.23 vs +0.21). The manual recipe reproduces the library.

**n ≈ 2–8 k** (`reports/O1_oof_n.csv`), Δ vs the library default (staged +
calibration):

| dataset | in-sample, no cal | **cross-fit, no cal** | cross-fit + cal | cov90: in-sample no cal → cross-fit no cal |
|---|---|---|---|---|
| het_a   | +0.294 (0/10) | **−0.087 (10/10)** | −0.087 (10/10) | 0.75 → 0.89 |
| het_b   | +0.037 (2/10) | +0.009 (4/10) | +0.011 (3/10) | 0.85 → 0.89 |
| homo    | +0.251 (0/10) | **−0.062 (10/10)** | −0.062 (9/10) | 0.71 → 0.91 |
| tnoise  | +0.549 (0/10) | **−0.081 (9/10)** | −0.095 (9/10) | 0.75 → 0.89 |
| kin8nm  | +0.188 (0/10) | **−0.049 (9/10)** | −0.054 (10/10) | 0.76 → 0.88 |
| protein | +0.057 (1/10) | −0.017 (8/10) | −0.033 (8/10) | 0.83 → 0.90 |

P1 and P2 hold: with no validation-fitted correction at all the
cross-fitted scale is calibrated (z-rms 0.97–1.16) and beats the
*calibrated* default on five of six datasets. P3: recalibration on top
adds little (−0.005 on average), so the residual optimism it leaves in
the in-sample fit is the x-dependent part a two-parameter map cannot
reach. RMSE also moves (het_a −3.4 %, protein +0.6 %): stage 3's
feasible-GLS weights come from the better σ̂.

**30 validation rows** (n = 3000; `reports/O1_oof_val30*.csv`). Here the
library default falls back to the joint, uncalibrated fit (its 40-row
threshold). Cross-fit, no cal, vs that default: het_a −0.273 (10/10),
het_b −0.081 (10/10), homo −0.223 (10/10), tnoise −0.254 (9/10), kin8nm
−0.106 (10/10), protein −0.085 (8/10); coverage 0.87–0.90 vs 0.82–0.86.
The validation rows are used for early stopping only.

**n = 700** (490 train / 70 validation; `reports/O1_oof_n700*.csv`), vs
the library default: het_a −0.130 (9/10), het_b −0.153 (10/10), homo
−0.008 (7/10), tnoise −0.146 (10/10), kin8nm +0.028 (5/10, se 0.028),
**protein +0.230 (4/10, se 0.10)**; with calibration on top protein is
still +0.198 (3/10). Coverage is fine there (0.87–0.90), so the loss is
sharpness, not calibration: on the bimodal target, out-of-fold residuals
of a 392-row location fit are dominated by wrong-mode rows.

**With leaf 20** (`reports/O1_oof_leaf20.csv`; reference = library staged
at leaf 20): cross-fit + cal −0.045 het_a (9/10), +0.001 het_b, −0.032
homo (9/10), −0.104 tnoise (9/10), −0.060 kin8nm (10/10), −0.018 protein
(7/10). The two changes stack: leaf 20 + cross-fit vs today's default is
−0.106 het_a, −0.013 het_b, −0.080 homo, −0.166 tnoise, −0.085 kin8nm,
−0.034 protein.

So the "simpler explanation" (it is only regularisation) is not the whole
story: O2 removes part of the in-sample optimism by fitting the location
less greedily, and cross-fitting removes more on top.

**Decision.** Keep as an opt-in candidate; worth a library implementation
(a `crossfit_folds` option on the staged fit). Before that: (1) understand
the protein n = 700 loss — try more folds, or cross-fit only stage 4;
(2) measure it for LogNormal and StudentT, which share the staged default;
(3) measure the no-validation case, where each stage needs a fixed or
cross-validated round count. Cost here: 2K extra location fits (K = 5:
wall time 2–4× the staged fit in this unoptimised harness).

### O4 — Gamma in (log mean, log shape) (`proposals/O4_gamma_mean.md`)

New research head `GammaMean`; `Gamma` untouched. Unit tests: identical
density, finite-difference gradient, Fisher metric by quadrature with a
zero cross term. `reports/O4_*.csv`; reference = `Gamma` (LogScore, its only
configuration).

| dataset | GammaMean, plain | + recalibration, Fisher weights | **+ recalibration, staged** | median ΔRMSE (staged) |
|---|---|---|---|---|
| gamma_homo (true Gamma, α = 4) | −0.023 (7/10) | −0.047 (9/10) | **−0.103 (10/10)** | −14 % |
| gamma_het (α from another feature) | −0.079 (10/10) | −0.087 (10/10) | **−0.128 (10/10)** | −7 % |
| gamma_shared (α from the mean's feature) | +0.014 (4/10) | −0.015 (6/10) | **−0.066 (10/10)** | −7 % |
| lognorm_het | −0.115 (10/10) | −0.173 (10/10) | **−0.264 (10/10)** | −12 % |
| kin8nm | −0.228 (10/10) | −0.317 (10/10) | **−0.324 (10/10)** | −31 % |
| support_cost (real, raw costs) | −0.013 (9/10) | −0.019 (10/10) | **−0.029 (10/10)** | −2 % |

P1 holds on five of six for the plain reparameterisation (same family,
same optimiser, only the coordinates differ); gamma_shared is the
predicted hard case and is a tie. P3 holds: the recalibration of log α
never lost and restores coverage (0.85–0.86 → 0.90), Fisher weights help
where α varies, and the staged fit — with the recalibration, as for
NegativeBinomial — is the best variant on every dataset.

The baseline head also shows what the proposal predicted for a mean that
is the difference of two trees: single-seed **mean blow-ups** (test RMSE
114 on gamma_shared against a median of 2.9; 4.6 on kin8nm against 0.22;
12.2 on lognorm_het against 5.4) with unremarkable NLL. `GammaMean` had
none in 300 fits. On kin8nm the `Gamma` head's early stopping is erratic
(49–65 rounds on some seeds, 950–1430 on others).

Cross-head context: on lognorm_het the staged GammaMean (2.034) is within
0.013 of the true-model LogNormal head (2.021); on kin8nm the Normal head
is still better (−0.533 vs −0.455).

**Decision.** Keep the head; recommend it over `Gamma` for LogScore with
`staged_fit = Some(true)` and the default recalibration (≥ 300 validation
rows). Open before it could replace anything: CRPScore is not implemented
for it; the 300-row threshold is NegativeBinomial's, not measured; the
`Gamma` mean blow-up should be reproduced as a regression test
(adversarial-testing) whatever happens to this head; Weibull has the same
non-orthogonal structure and was not looked at.

### O3 — StudentT ν floor (`proposals/O3_studentt_nu_floor.md`)

Prototype: `NGBOOST_STUDENTT_NU_MIN` (ν = ν_min + exp θ₃). `reports/O3_*`.
Counts are fits with at least one test row in the stated condition.

Staged fit (the 1.2.0 default for StudentT), 8 datasets × 10 seeds = 80
fits per floor:

| ν_min | undefined mean | CRPS undefined or > 10³ | infinite variance | mean ΔNLL vs 0 over datasets |
|---|---|---|---|---|
| 0 (today) | 22 | 1 | 60 | — |
| 1 | 35 | 1 | 68 | +0.011 |
| 1.1 | 0 | 0 | 69 | +0.017 |
| 2 | 0 | 0 | 64 | +0.006 |
| **2.1** | **0** | **0** | **0** | **−0.001** |

Per dataset, ν_min = 2.1 vs 0 (ΔNLL, wins): het_a −0.012 (7/10), het_b
−0.010 (6/10), homo −0.019 (4/10), tnoise +0.012 (4/10), lognorm_het
+0.002 (7/10), kin8nm +0.001 (5/10), support_logcost +0.001 (4/10),
protein +0.019 (3/10). Median CRPS within ±0.9 % except lognorm_het
(+8.9 %, on the fits where it was defined before); 90 % coverage unchanged
(protein 0.893 → 0.910). Predictive standard deviations are finite in
every fit.

Joint fit (opt-in; floors 0, 1, 2 only; 80 fits): ν_min = 2 gives 25 → 0
undefined means, 15 → 2 bad CRPS, and **−0.11…−0.28 NLL (7–10/10)** on
het_a, het_b, homo, tnoise and kin8nm, −0.03 (9/10) on support_logcost,
−0.06 (6/10) on protein; lognorm_het +0.054 (4/10). The
floor stabilises the fragile joint fit (limitation 43) far more than it
changes the staged one.

What the first run (ν_min = 1 and 2) got wrong: rows whose unconstrained
optimum lies below the bound run to it exponentially fast
(Δθ₃ = Δν/(ν − ν_min)), and in floating point ν_min + exp(θ₃) is *exactly*
ν_min for θ₃ < −37. So the bound is attained: at 1 the mean is undefined
on more fits than with no floor, at 2 the variance is still infinite on
64. P1 as stated was false. The bound has to sit strictly inside the
region where the quantity is defined; 2.1 does that. ν_min = 1.1 keeps the
mean but is the worst on NLL (het_b +0.039, 0/10; lognorm_het +0.080): a
looser bound is not a gentler one, because rows parked at ν = 1.1 are far
wider than rows parked at 2.1.

P2 holds for 2.1 on the mean (−0.001), with two datasets paying up to
+0.02 (protein, tnoise): those do use tails heavier than t(2.1) on some
rows.

**Decision.** ν_min = 2.1 is a workable fix for limitation 47: every
point forecast, variance and CRPS is defined, at ≤ +0.02 NLL on the worst
dataset. Kept as a research switch. To ship it the bound must be stored
with the model (θ₃ means something different under each ν_min) — a field
or a separate head, not an environment variable — and the CRPS gradient
and metric need the same factor c. An infinite-variance stress case
(P3) is still missing from the benchmark set.

## 2026-11-13 — Implemented after approval

The user approved all five candidate changes ("yes you can do all of
these"). What was built, and what the extra measurements done on the way
showed. Details and tables are in the proposals' "Library change"
sections; commands in `reports/README.md`.

| Item | Change | New evidence gathered while implementing |
|---|---|---|
| O2 | `HistogramLearner` default `min_samples_leaf` 1 → 20 (`DEFAULT_MIN_SAMPLES_LEAF`); opt-out `with_params(d, 255, 1, 2)` | Below 500 training rows: wins on 6/6 at 210 rows (mean −0.078); a wash at 105 rows (mean −0.015, het_b RMSE +0.29), where leaf 5–10 is better. Documented, default kept at 20 |
| O4 | `GammaMean` documented; staged by default (LogScore, ≥ 300 validation rows); `CRPScore` by the chain rule on `Gamma` | CRPS derivation verified by finite differences and natural-gradient invariance; not benchmarked |
| O3 | New head `StudentTFiniteVar` (ν > 2.1) instead of an env switch or a model field; `StudentT` restored to its 1.2.0 source | Reproduces the prototype's 80 fits exactly. **An arithmetically equivalent variant differs by up to 0.17 NLL per fit** (mean vs StudentT +0.010 instead of −0.001; protein +0.065 instead of +0.019) |
| O1 | `staged_crossfit_folds` (opt-in) in `fit_staged`; lifts the recalibration condition on the default staged fit | Library numbers for Normal (table in O1); protein small-data loss two-thirds explained by leaf size; **LogNormal means explode on kin8nm (8/10 seeds)**; GammaMean fine; StudentTFiniteVar −0.048/−0.037/−0.040/−0.008 NLL (het_a, tnoise, kin8nm, protein), means unaffected. The protein run was executed twice in separate processes: all 20 rows identical (the duplicate file was not kept) |
| A3/A1 | `hyper_opt`: fold patience scales with the learning rate; `default_search_space`, `default_histogram_model`, `default_round_cap` | none (see `research/automl-hyperparameter/research_log.md`) |

Two things in that table change how earlier entries of this log should be
read:

1. **Implementation noise.** Staged StudentT fits are chaotic in the last
   bit: reordering a multiplication moved single fits by up to 0.17 NLL
   and a ten-seed dataset mean by up to 0.05. Differences of ≤ 0.02
   between StudentT variants in O3 are inside that band, whatever their
   seed standard error says. It was not measured for the Normal head,
   whose staged fit has no ν coordinate; the O1/O2 effects there are
   several times larger and consistent in sign across datasets and
   settings.
2. **NLL can improve while the mean breaks.** Cross-fitted LogNormal on
   kin8nm is the cleanest case so far: −0.12 NLL on 10/10 seeds, RMSE up
   by many orders of magnitude on 8/10. Any future change to a scale
   stage should be judged on RMSE/CRPS of every head it touches, not on
   NLL alone. (The O1 Normal tables do report RMSE: −0.02…+0.02.)

A second one: the first implementation computed the stage-3 Fisher
weights (1/σ̂²) from the *clamped* prediction, so on kin8nm the ceiling
changed the fit itself (1821 rounds instead of 1214 on seed 0) while the
documentation said the fit was untouched. It was found because a rerun of
the emulation did not reproduce its own `k = inf` row. Every fit-time
quantity now reads the unclamped sum, the shipped output equals the
emulation digit for digit, and `tests/test_cycle19.rs` fits a twin head
without a ceiling and requires a bit-identical model. (The accidental
variant scored −0.009 NLL better on kin8nm with cross-fitting; one
dataset, not pursued.)

Verification on the final tree: `cargo test --release --features
accelerate --no-fail-fast` — 21 binaries, 536 passed, 0 failed, 8 ignored. The 1.1.2 bit-for-bit parity test
passes with the leaf-1 opt-out added to its recipe. `cargo doc` and
clippy report nothing new in the changed code.

Not done, deliberately: no version bump in `Cargo.toml`, no commit (the
CHANGELOG entry is headed "Unreleased (1.3.0)"); no guard against the
LogNormal mean blow-up; no change to `DecisionTreeLearner`, the sampler
or `Gamma`.

## 2026-11-14 — Cycle 19: four follow-ups, approved in advance

The user approved four items from the "next questions" below: a guard for
the LogNormal mean blow-up, an end-to-end benchmark of the shipped tuner
defaults (in `research/automl-hyperparameter/research_log.md`, A6), a
leaf default that adapts to small data, and the exact tree learners'
leaf default. Each was measured before anything was changed.

| | Question | Result | Decision |
|---|---|---|---|
| O5 | Why do LogNormal means explode under cross-fitting, and what bounds them? | Not extrapolation and not a step overshoot: the additive scale trees isolate single extreme log-residuals and fit σ ≈ \|r\|, which the score rewards. A ceiling of 4 × the model's own scale constant repairs kin8nm (RMSE max 10²⁵ → 0.148, NLL −0.003) and binds on no row of three well-specified datasets | **Implemented**: `Distribution::scale_ceiling`, `LogNormal` = 4, prediction-time only, no new stored state; `pred_param_unbounded` as the opt-out |
| O6 | Does a leaf of min(20, n/10) beat a fixed 20 below 200 rows? (rule stated before the sweep) | Pooled RMSE −5.0 %, −1.7 %, −1.2 % at 70, 105, 140 fitted rows; NLL lower on 207 of 360 fits; one dataset carries it, none significantly worse; at 210 rows 20 is best | **Implemented**: `adaptive_leaf` on the default constructors |
| O7 | Do the exact tree learners want leaf 20 too? | Yes: −0.047…−0.098 mean NLL at 210–2100 rows (exact), −0.038 at 2100 (arena); 7–10 of 10 seeds in 23 of 24 cases | **Implemented** for `new(depth)`; `default_sklearn` / `default_tree_learner` keep leaf 1 by contract |
| O8 | Is the natural-gradient leaf value for log σ an overshoot worth fixing? | No effect on the default Normal fit (−0.0001), −0.006 on cross-fitted fits, −0.018 on LogNormal/kin8nm; does not stop the blow-up | **Rejected**, not in the tree (`proposals/O8_exact_scale_step.patch`) |

What changed in how earlier entries should be read:

1. O1's LogNormal hazard is closed for `NGBoost` regression models, by a
   bound rather than by a better fit. The diagnosis matters more than the
   guard: a proper score is indifferent to what a large σ on one row does
   to the mean, so *any* head whose mean is steeply nonlinear in a scale
   parameter (LogNormal, the hurdle log-normal, Gamma with a small shape)
   can show this, and cross-fitting makes scale fits honest enough to
   expose it. The hurdle heads and the survival wrapper are not covered.
2. O2's "leaf 20, except on tiny data" is now one rule. The small-data
   evidence is thin on NLL (medians and win counts, not means) and solid
   on RMSE for one kind of dataset.
3. The first attempt at O5 looked for an algorithmic cause (O8) and found
   none. That negative result is why the fix is a prediction-time bound.

A wrong turn worth recording: an early unit test compared the bounded
LogNormal model with a separately fitted `Normal` on ln y and seemed to
show the ceiling *costing* 0.11 NLL. The two models were not the same
fit. Compared with its own unclamped sum, the bounded model was 0.09–0.21
*better*. Comparisons of the ceiling use `pred_param_unbounded` on the
same model since.

Verification on the final tree: `cargo test --release --features
accelerate --no-fail-fast` — 22 binaries, 540 passed, 0 failed, 8 ignored. New tests:
`tests/test_cycle19.rs` (4). The 1.1.2 parity test passes unchanged.

Not done, deliberately: no version bump, no commit; no ceiling for the
hurdle heads or the survival wrapper; no change to `default_sklearn`.

### Next questions (after Cycle 19)

1. The same blow-up check for `HurdleLogNormal` and for `Gamma` with a
   small fitted shape (the single-seed RMSE 114 noted under O4), with and
   without cross-fitting.
2. Whether the ceiling should be stored in final units once the model
   format next changes (it would make it independent of the
   recalibration slope).
3. Adaptive leaf on the count and classification heads below 200 rows,
   and with sample weights.
4. O8 again if cross-fitting is ever proposed as a default.
5. Still open from before: protein under cross-fitting on small data,
   row by row; a perturbed-build repeat to size implementation noise.

## 2026-11-14 — Cycle 20: closing the open questions

The user asked for the five items left open after Cycle 19 (the tuner
part is A7 in `research/automl-hyperparameter/research_log.md`).

| | Question | Result | Decision |
|---|---|---|---|
| O9 | Do `Gamma`, `HurdleLogNormal` and the survival `LogNormal` blow up like `LogNormal` did? | `Gamma`: yes at leaf 1 (a single-row leaf drives the rate to e⁻⁷), **not at the 1.3.0 leaf default** (60 fits). Survival and hurdle: no, on kin8nm (30 fits) — both are joint fits. One hurdle failure on a constructed floor-atom target (mean 31 against a largest target of 1.45) | No head changed. Regression tests (`tests/test_cycle20.rs`). The hurdle case is documented, not fixed: no bound relative to the model's own scale reaches it |
| O10 | Does the adaptive leaf carry over to count and classification heads and to weighted fits? | A trade: NLL lower on 280 of 540 fits; pois_nl −0.60/−0.25, pois_low +0.048/+0.044 (3 se), bern_lin +0.026 (2 se), rest noise. Weighted Normal: as unweighted. Leaf 20 against 1 under weights: lower on 45 of 60 | Keep the rule; caveat and opt-out documented |
| O11 | Where does cross-fitting lose on protein at 490 rows? | 5 of 1410 test rows carry the whole +0.069: far-mode targets at \|z\| 5–10 under the sharper σ. Median row −0.010. With `StudentTFiniteVar` the loss is +0.004 | No change; guidance: change the head |
| O12 | How much last-bit noise is in a 10-seed comparison? | 359 of 360 fits change their round count under a 10⁻¹² perturbation; one fit's NLL sd 0.018 (max 0.10); a 10-seed paired effect sd 0.007 (max 0.015) per dataset, 0.0015 pooled over six | Reporting rule: single-dataset differences below 0.02 are "no difference" |

Three things worth keeping from this cycle:

1. **Leaf size was behind more than one "head defect".** The `Gamma`
   blow-up recorded under O4 as a property of the (α, β) coordinates was a
   leaf-1 artefact. It supported the case for `GammaMean` at the time;
   `GammaMean`'s NLL advantage does not depend on it, but the blow-up
   argument does not hold at the current default.
2. **The regression test found what the benchmark did not.** kin8nm with
   zeros was clean for the hurdle head; a synthetic target with an atom
   at its floor was not. The benchmark datasets under-represent point
   masses and detection limits.
3. **O12 re-grades this log.** Every single-dataset effect of 0.01–0.02
   recorded since Cycle 18 is inside the noise of one dataset. The pooled
   numbers move by 0.0015 and are safe; the cross-fit effect on six
   datasets was −0.037, −0.034, −0.036, −0.036 in four replicates.

Verification on the final tree: `cargo test --release --features
accelerate --no-fail-fast` — 23 binaries, 542 passed, 0 failed, 8 ignored. New tests:
`tests/test_cycle20.rs` (2).

### Next questions (after Cycle 20)

1. A point prediction for the log-normal family that does not hang on
   the upper tail (median, or a mean truncated at the largest training
   target), which would also cover `HurdleLogNormal`.
2. A leaf rule that looks at signal strength, not only at n — e.g. the
   validation gain of the first location trees at leaf n/10 against 20 —
   since the adaptive rule's wins and losses split along that line.
3. Benchmark datasets with point masses and detection limits.
4. O12 on the tree builder and the line search (a perturbation inside
   the fit, not in the data).

## 2026-11-14 — Cycle 21: a bounded point prediction for HurdleLogNormal

Approved by the user, format change included
(`proposals/O13_hurdle_capped_prediction.md`).

`NGBoost::predict` of a `HurdleLogNormal` model now returns
(1 − π)·E[min(Y, c)], c the largest target seen in training, stored as
`target_max` — the one new field of `SerializedNGBoost`. Files from
before 1.3.0 load through `load_model` / `from_bincode` and keep the
plain mean; 1.2.0 reads 1.3.0 files.

Result: the constructed failure of O9 (mean 31 against a largest target
of 1.45) is bounded by construction; on three benchmark datasets the fits
are identical and RMSE moves by −0.8 %, −0.6 % and +0.1 % (no cost; by
O12 not a gain). Plain `LogNormal`, `HurdleNormal` and the survival
wrapper are untouched.

A side question from the user: the crate has no k-means or any other
clustering step, so the e-means paper (Touvron, Douze, Jégou et al.,
arXiv 2611.09501, 2026-11-06 — mini-batch k-means with EMA centroid
updates, a cosine step schedule, dead-cluster revival and fitted
batch-size rules, for billions of high-dimensional vectors) has nothing
to replace here. The two nearest things — quantile bin edges per feature
and the quantile-group initialisation of the mixture head's EM — are
one-dimensional, exact and cheap, and the paper itself notes its scaling
rules are not tuned for low-dimensional or small data.

Verification on the final tree: `cargo test --release --features
accelerate --no-fail-fast` — 23 binaries, 544 passed, 0 failed, 8 ignored.

## 2026-11-14 — Cycle 22: is there a k-means to replace? (O14, O15)

The user asked, after Meta FAIR's e-means paper, whether any crate in the
build uses k-means and whether e-means would suit it better, and to look
at the two places where a clustering step could sit.

- **Audit.** None of the 100 dependency crates (normal, build and dev)
  and nothing in this crate contains k-means or any clustering code.
- **O14, bin edges.** 1-D k-means edges against the current quantile
  edges: no difference at 255 bins (±0.003…0.021, n.s.), and k-means
  edges are not invariant to a monotone rescaling of a feature — after
  x → exp(1.5·x) quantile fits are bit-identical (60 of 60) while k-means
  at 8 bins loses 0.15–0.34 NLL. Rejected.
- **O15, mixture start.** A k-means start and the quantile start reach
  the same EM fit on eight targets (largest mean difference 0.001).
  Several starts find a better marginal optimum in one constructed case
  (two components, three modes: +0.075) and change nothing end to end.
  No change.
- **Why e-means does not apply either way.** It is an optimiser for
  k-means on billions of high-dimensional vectors. Both candidate places
  are one-dimensional, where a Lloyd pass is k prefix-sum lookups; the
  question was only whether the k-means objective is the better
  criterion, and it is not.

By-product (O14): `max_bins` moves held-out NLL by more than the edge
rule does and in a data-dependent direction — kin8nm −0.527, −0.584,
−0.603 at 255, 32, 8 bins; protein 2.988, 3.050, 3.109. A candidate for
the tuner's search space.

Nothing in `src/` was changed by this cycle; the multi-start prototype is
kept as a patch.

## 2026-11-14 — Cycle 25: cross-fitting as the default? (O16)

The user asked for the test after A9, where the tuner switched
cross-fitting on in 52 of 56 winning models. Plan, decision rule and
results: `proposals/O16_crossfit_default.md` (the rule was written before
the first run; the fallback rule before the confirmation run).

Design: the shipped default-shape model (adaptive leaf, own 10 %
validation split), `staged_crossfit_folds` 0 against 5 and 3, the five
heads that stage by default on 19 head × dataset pairs, pools of 100 to
6000 rows, 2000 held-out test rows, seeds 100–109 (stage 1) and 200–209
(stage 2). 7980 fits in the two stages, none failed.

| question | answer |
|---|---|
| Always on (the proposal as made) | **Fails the no-harm rule.** −0.047 NLL pooled, but Normal on kin8nm at 200 rows +0.116 (2.4 se) and a coin toss at 100 rows (−0.019, se 0.038, cells −0.31 … +0.30) |
| On from 40 validation rows (≈ 400 rows with the default split), chosen after stage 1 | **Passes, and is confirmed on fresh seeds:** −0.044 (0.007), lower on 763 of 950 fits, RMSE −1.0 %, coverage 0.899 → 0.895. Both stages: −0.045, 1543 of 1900, 18 of 19 pairs better |
| K = 3 instead of 5 | −0.043 (0.007), 738 of 950, RMSE −0.7 %: within the 0.005 band, passes the rule |
| Cost (one fit at a time) | K = 5: 2.4–6.6× the default, median 5.6×; K = 3: 0.9–5.0×, median 3.2× |
| Where it does not help | Normal on protein: +0.005 NLL (0.008), RMSE +0.5 % (4.9 se) |
| Large data | the gain decays: about −0.03 at 6000 rows, −0.02 at 20 000 |
| GammaMean | joint today below 300 validation rows; staged and cross-fitted from 40 it gains −0.01 … −0.19 at 400–1600 rows in all 18 cells of the two stages |
| Sample weights, chronological split | −0.042 and −0.040 pooled, lower on 66 and 68 of 80 |

One cross-fitted LogNormal fit of 560 (K = 3) had a point-prediction
blow-up — RMSE 19.9 against 4.6 with a better NLL: one test row with σ at
3.6 times the scale constant, just under the ceiling of 4 (O5).

**Decision.** None taken; no library default was changed. The rule that
passed is "heads that stage by default cross-fit when the staged fit has
at least 40 validation rows", K = 3 by the pre-registered tie-break. What
it needs in the library: an automatic setting distinct from "off", a
fallback instead of an error when a fold would have fewer than two rows,
and GammaMean staging from 40 validation rows when cross-fitting.

### Next questions (after Cycle 25)

1. An upper size limit: at 20 000 rows the gain is −0.02 for 2–4× the
   time; where does it stop paying?
2. Explicit validation sets and the survival wrapper under the rule.
3. Between 20 and 40 validation rows the average gain is larger
   (−0.07 … −0.09) but single datasets lose 0.1–0.3: is the cause the
   tiny validation block (early stopping of four stages) or the tiny
   folds? Separating them needs a large-train / small-validation run.
4. The LogNormal ceiling at 4 let one σ at 3.6× through: a ceiling of 3,
   or the median as the point prediction, re-measured on O5's data.

## 2026-11-14 — Cycle 26: count heads on real low-count data (O17)

The user's own models are sports-betting models on the Poisson,
NegativeBinomial and Normal heads, and asked for two things before
pushing: the staged NegativeBinomial fit with cross-fitting, and real
low-count data. Plan, rule and full results:
`proposals/O17_count_heads_real_data.md` (rule written before the runs).

Data: 14 019 football matches (six leagues, 2019/20–2024/25; home and
away goals, means 1.50 and 1.24; with market odds and without), three
public count sets (doctor visits, articles per biochemist, physician
visits; var/mean 2.1–7.9) and two new goals-rate synthetic sets. Random
splits with 10 fresh seeds, and for football a rolling origin by season,
pooled and per league. 3216 fits, none failed.

| question | answer |
|---|---|
| Staged + cross-fitted NegativeBinomial as its default | **Rule fails.** Real data −0.0021 NLL (1.7 se), 60 % of fits; it helps only with strong over-dispersion or a strongly varying rate (nb_over −0.036, pois_nl −0.027), at 4–36× the fit time |
| Leaf default against leaf 1 on real low counts | Poisson −0.0024 (2.7 se), NegativeBinomial −0.0017; football −0.000 … −0.006. Safe, worth little there |
| Poisson or NegativeBinomial on goals | **Poisson**, by 0.001–0.007 in all eight football rows (six at 2 se or more). NegativeBinomial wins by 0.02–1.5 once var/mean reaches 2 |
| Does `CountModel::fit_auto` choose correctly | Yes: NegativeBinomial on 13 of 232 football fits and on 63 of 70 fits of the three over-dispersed sets; no measurable loss against the better head |
| Does the tuner help on goals | No: −0.0001 (0.0005) over four datasets × 8 seeds |

The scale that matters: on football the features are worth 0.05–0.07 NLL
over a constant rate and market odds another 0.014, while the seven fit
variants differ by at most 0.007.

**Decision.** Nothing changed in the library. With these basic features
(odds, form, Elo; no feature engineering) the fit options are tied on
goals; that is a statement about a low-signal target, not a general one —
on strongly varying synthetic rates the same options are worth 0.03–0.07.
No default was chosen from the football data; the cycle's one decision
was a rejection (see "Scope of these conclusions" in the proposal).

### Next questions (after Cycle 26)

1. NegativeBinomial's 0.001–0.007 cost on nearly-Poisson data
   (limitation 4): a larger dispersion step or init would close it, but
   `fit_auto` already avoids it.
2. Counts at other scales (basketball points, runs), where the Normal
   head competes with the count heads.
3. A time-decayed or rolling-window fit for sports data, where the
   newest seasons matter most — a data-weighting question, not a fit one.

## Literature

- Barber, Candès, Ramdas, Tibshirani, "Predictive inference with the
  jackknife+", Ann. Statist. 2021 — leave-one-out / K-fold (CV+) residuals
  as the honest basis for predictive spread; O1 uses the same residuals as
  a regression target for σ(x) instead of as a conformal score.
  https://lacuna.tiptreesystems.com/work/predictive-inference-with-the-jackknife/version/art_0ad06ef1e461448a87ce07354e93faed
- Cross-fitting of nuisance fits (Chernozhukov et al. 2018, double
  machine learning) and parameter orthogonality (Cox & Reid 1987) — cited
  from memory, not checked this cycle. Relevant to O1 and O4.
- Leaf-size and leaf-weight regularisation of boosted trees is standard
  (XGBoost `min_child_weight` / λ, LightGBM `min_data_in_leaf` = 20); the
  finding here is only that this crate's default of 1 costs accuracy for
  natural-gradient targets. Not a new method.
- No claim of novelty is made for O1 or O4; neither was searched for
  exhaustively in the literature.

## Next questions (after the implementation)

1. LogNormal mean blow-up: a cap on σ in log space, or reporting the
   median as the point prediction; then re-measure the default staged
   LogNormal (4 of 10 kin8nm seeds are already mildly affected).
2. A leaf size that adapts to n (≈ n/10, capped at 20) — the 105-row
   result suggests it; needs its own sweep before any change.
3. Cross-fitting on protein-like targets: per-row scores to confirm the
   tail explanation; whether the Fisher-weighted stage 3 matters there.
4. How much of any ≤ 0.02 NLL effect in this log is implementation noise:
   repeat one Normal experiment with a perturbed build.
5. Weibull in an orthogonal parameterisation; a regression test for the
   `Gamma` head's mean blow-up; CRPS-trained `GammaMean` benchmark.

## Next questions (from this cycle's evidence, before the approval)

1. O1 in the library: `fit_staged` with cross-fitted residuals; the
   protein small-n loss; LogNormal and StudentT; no-validation operation.
2. O2 as a default: n < 500, weighted fits, the exact tree learner; then
   ask for approval with the 38-case table.
3. Location trees over-fit at leaf 1 — is the same true of *depth*? The
   tuner study found deeper trees better on kin8nm and protein, shallower
   on the synthetic sets (`research/automl-hyperparameter/`).
4. Weibull in an orthogonal parameterisation; CRPScore for `GammaMean`.
5. Regression test for the `Gamma` head's mean blow-up.
