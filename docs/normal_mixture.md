# `NormalMixture<K>` — K-component Normal head for multimodal targets (added 1.2.0)

**When to use it.** When the conditional target can have two modes — e.g.
protein RMSD (a tight mode at 1–3 Å and a wide one at 11–16 Å). A
single-Normal head must then place mass between the modes; with Fisher
weighting (β = 1) it even starves the wide mode (`docs/known_limitations.md`
#14). `NGBMixtureRegressor` = `NGBoost<NormalMixture2, LogScore,
HistogramLearner>`.

**Parameterization.** `NormalMixture<K>` (aliases `NormalMixture2`,
`NormalMixture3`): K−1 weight logits (softmax, last component as
reference), then (μ_k, log σ_k) per component; for K = 2 that is `logit π`,
`μ₁`, `log σ₁`, `μ₂`, `log σ₂`. Initialised by EM on the marginal target
(K equal-count quantile groups; components ordered by mean). LogScore and
CRPScore.

**Natural gradient.** The observed Fisher information of a mixture has no
closed form; the head uses the **complete-data Fisher** (component labels
treated as observed): weight logits diag(p) − ppᵀ (for K = 2: π(1−π)), μ_k
π_k/σ_k², log σ_k 2π_k. The weight block is inverted in closed form
(Sherman–Morrison: D⁻¹ + 11ᵀ/p_K), so the step is O(K) per row. It
dominates the observed Fisher (verified by quadrature in the unit tests), so
steps are conservative and EM-like: with responsibilities r₁ = πφ₁/f the
natural gradient is exactly the responsibility-weighted residual, e.g.
(r₁/π)(μ₁ − y). Reference: Xu & Jordan, "On convergence properties of the EM
algorithm for Gaussian mixtures", Neural Computation 8 (1996). Component
weights in the metric are floored at 1e-3.

**Default recalibration** applies to both component scales with one shared
(a, b) (`Distribution::scale_param_columns` = [2, 4]), preserving their
relative widths; it raised 90 % coverage from 0.83–0.87 to 0.90–0.93 and
lowered NLL 10/10 on five of six datasets (R30).

**Evidence** (10 seeds, lr 0.05; both heads with their default calibration;
paired Δ = mixture − Normal; CRPS by numerical integration):

| dataset | ΔNLL | ΔCRPS | ΔRMSE | cov90 |
|---|---|---|---|---|
| protein (UCI) | **−0.461** (10/10) | **−0.096** | +0.219 (+5 %) | 0.924 → 0.930 |
| bimodal (synthetic) | **−0.786** (10/10) | **−0.285** | −0.038 | 0.924 → 0.908 |
| het_a | +0.000 | −0.019 | −0.034 | |
| homo | −0.025 | −0.034 | −0.071 | |
| kin8nm (UCI) | +0.004 (2/10) | +0.003 | +0.006 | |
| tnoise | +0.038 | +0.027 | +0.053 | |

Confirmed at lr 0.01 (protein −0.450, bimodal −0.774, 5/5). On unimodal
data the extra flexibility costs at most ≈ 0.04 NLL, within noise except
kin8nm's small CRPS loss.

**Point forecasts (R34, 10 seeds).** Pick the summary that matches your loss:
`predict()`/`mean()` for squared error, `median()` for absolute error;
`mode()` (implemented by mean-shift from every component) was the worst
point forecast on every multimodal dataset.

| dataset | Normal RMSE / MAE | mixture mean RMSE | median RMSE | mode RMSE | mean MAE | **median MAE** | mode MAE |
|---|---|---|---|---|---|---|---|
| protein | 4.59 / 3.62 | 4.81 | 5.07 | 6.30 | 3.77 | **3.48** | 4.04 |
| bimodal | 4.33 / 3.44 | 4.29 | 4.59 | 5.95 | 3.39 | **2.92** | 3.75 |
| trimodal | 5.69 / 4.56 | 5.64 | 6.01 | 7.68 | 4.54 | **4.10** | 5.09 |
| kin8nm | 0.173 / 0.136 | 0.179 | 0.183 | 0.202 | 0.140 | 0.139 | 0.147 |

The median beat the mean on MAE 10/10 on every multimodal set, and the mean
beat the median on RMSE 10/10. For RMSE on protein the single-Normal head
remains best (4.59).

**K = 3 (`NormalMixture3`, R36).** vs K = 2, both calibrated: trimodal
−0.288 NLL (10/10), protein −0.053 (10/10), bimodal −0.001; unimodal costs
+0.02…+0.05 (kin8nm +0.050, 0/10). Choose K from the target's shape or by
validation NLL.

**Choosing K automatically: `NGBAutoMixtureRegressor` (Cycle 11).** Fits a
single Normal (K = 1) and K = 2, 3, 4 mixtures on the same data — each
early-stopped and recalibrated on the validation set — and keeps the lowest
validation NLL. Over 7 datasets × 10 seeds (R38), the chosen head was at
least as good as the best *fixed* K on every dataset (−0.001…−0.043 NLL),
better than every fixed K on unimodal data, and within 0.001–0.010 of a
test-set oracle:

| dataset | K=1 | K=2 | K=3 | K=4 | **auto** | oracle | picks K=1..4 |
|---|---|---|---|---|---|---|---|
| protein | 2.885 | 2.423 | 2.352 | 2.337 | **2.332** | 2.330 | 0/0/3/7 |
| trimodal | 3.133 | 2.445 | 2.156 | 2.154 | **2.153** | 2.152 | 0/0/4/6 |
| bimodal | 2.873 | 2.086 | 2.087 | 2.105 | **2.085** | 2.084 | 0/6/4/0 |
| het_a | 1.524 | 1.537 | 1.565 | 1.586 | **1.481** | 1.480 | 4/4/0/2 |
| homo | 1.711 | 1.710 | 1.705 | 1.813 | **1.661** | 1.661 | 3/3/3/1 |
| tnoise | 1.540 | 1.577 | 1.592 | 1.671 | **1.510** | 1.500 | 4/2/3/1 |
| kin8nm | −0.444 | −0.443 | −0.404 | −0.395 | **−0.446** | −0.451 | 5/4/1/0 |

It costs one fit per candidate. The same validation set drives early
stopping, recalibration and selection, so the validation NLL itself is
mildly optimistic — report on a test set.

**Warm-grown candidates and the search modes (Cycle 12, R42–R43).** A K-component
fit can be *grown* from the (K−1)-component one (`search = KSearch::Warm`):
the source's trees are copied round for round and its reference component
is split in two at μ ∓ 0.8σ̄ with σ × 0.6 (split-and-merge EM, Ueda et al.
2000), then boosting continues. That costs 46 % of a cold search but is a
*different* model — it keeps the Normal head's conditional mean: +0.17…+0.56
NLL worse on multimodal targets (it never leaves the between-modes mean
within early-stopping patience), −0.07…−0.11 better on unimodal ones.
Adding the new component at the worst-fitted rows instead (greedy mixture
learning; `GrowStrategy::Misfit`) was worse everywhere multimodal.
The default `KSearch::Both` puts cold and warm candidates into the same
validation-NLL selection:

| dataset | cold auto | Both | Δ | warm picked | cost (rounds) |
|---|---|---|---|---|---|
| protein / bimodal / trimodal | 2.332 / 2.085 / 2.153 | same | 0 | 0/10 | 116–143 % |
| flchain futime / support log d.time | 7.322 / 1.456 | same | 0 | 0/10 | 131–133 % |
| kin8nm | −0.446 | −0.518 | **−0.072 (10/10)** | 10/10 | 109 % |
| het_a | 1.481 | 1.378 | **−0.103 (9/10)** | 9/10 | 113 % |
| homo | 1.661 | 1.590 | **−0.072 (9/10)** | 10/10 | 108 % |
| tnoise | 1.510 | 1.411 | **−0.099 (9/10)** | 9/10 | 111 % |
| support log cost | 1.151 | 1.148 | −0.003 | 10/10 | 114 % |

Overall 118 % of the cold search's boosting rounds (141 % wall time); with
`greedy = true` (each family stops at the first K that does not improve
validation NLL) 101 % of the rounds / 118 % time with the same quality
except tnoise (−0.072 instead of −0.099). On unimodal data the selected
warm model also beats the plain Normal head clearly (het_a 1.524 → 1.378,
homo 1.711 → 1.590, kin8nm −0.444 → −0.518). `tried` lists every candidate
(K, warm?, validation NLL, rounds); `selected_is_warm()` reports the pick.

**Real multimodal targets beyond protein (Cycle 12, R40).** From the survival
CSVs, using censored follow-up times as plain targets (a test of whether the
head can represent multimodality, not survival analysis): flchain follow-up
days — mixture vs Normal −0.90 NLL, CRPS −78 days (10/10); support2 ln(days
to death/censoring) — −0.29 NLL (10/10); validation selection picked K = 4
in every seed on both (= the test oracle). Control, support2 ln(total cost):
−0.005, mixed K picks. 17 exact-zero costs made a component collapse onto
the point mass (NaN NLL) — a continuous mixture cannot represent atoms; they
were excluded from the control.

**Step guard (Cycle 11).** Unguarded, one training row far in a component's
tail and alone in a leaf drove that component's σ to 2·10⁸ in a single round
on full protein (K = 3; R37): the log σ step (1 − z²)/2 linearises a move whose
exact per-point optimum is only ln|z|. The LogScore natural gradient now caps
every row's log σ step at 5 (metric entry max(2π_k, |g|/5) — never below the
expected Fisher, engaged only ≈ 3.3σ into a tail). On full protein: no row
with σ > 100 over 3 seeds (unguarded: 3 train / 2 test rows) and better test
NLL (2.167–2.212 vs 2.208–2.218). On the 12k grid: neutral for K = 2, better
for K = 3 (protein −0.019, kin8nm −0.014 10/10, tnoise −0.047), no cell
significantly worse (R38). A broader guard that damped every row with
r_k > π_k removed the drift too but cost +0.03…+0.12 NLL and was rejected.
`NGBOOST_MIX_GUARD=0` disables the guard (research only).

**CRPS step guard (Cycle 12, R41).** The CRPS metric diagonal scales like
π_k²σ_k, so as a component shrinks its CRPS natural-gradient step grows like
1/(π_k²σ_k) — the cause of the R35 collapse. The CRPS training path now caps
each row's step (log σ and weight logits at 5, μ_k at 5σ_k). Collapses are
gone (het_a worst test NLL 6.8·10⁶ → 1.7) and CRPS training improved a lot on
unimodal data (homo −0.27, kin8nm −0.15, 10/10), ≈ neutral on multimodal.
With the guard, CRPS training *beats* LogScore training on unimodal targets
(homo −0.14 NLL 10/10, RMSE −0.28; het_a −0.05; kin8nm tie) but still loses
on multimodal ones (protein +0.39, trimodal +0.20, bimodal +0.14) — so for
the multimodal use case of this head, LogScore remains the recommendation.

**CRPS training (`CRPScore`, R35, R39) — implemented, not recommended for multimodal targets.** Closed
form (Grimit et al. 2006) with analytic gradients and the closed-form
diagonal of the CRPS metric 2∫∇F∇Fᵀ (all unit-tested). Trained with CRPS,
the mixture was worse on test NLL **and on test CRPS** on every dataset
(K = 2: ΔNLL +0.13…+0.39, ΔCRPS +0.007…+0.12, 0/10 seed wins everywhere;
K = 3 the same), 6× slower, and one het_a seed collapsed a component (test
NLL 6.8·10⁶ — CRPS barely penalises a near-zero-width component). The
full CRPS metric (closed form, verified entry-by-entry against quadrature;
`NGBOOST_MIX_CRPS_FULL=1`) was tested as the fix in Cycle 11 and **does not
rescue it**: better than the diagonal on bimodal/trimodal (−0.08…−0.18 NLL),
worse on homo/het_a/protein and collapsing on kin8nm (test NLL 5·10⁵), and
still 0/10 against LogScore training on all nine dataset × K cells (R39).

**Why warm-grown candidates win on unimodal data (Cycle 13, R44).** Not
extra rounds: continuing the Normal for the same rounds recovers 4–17 % of
the gain. Not shape either, on Gaussian-noise data: the moment-matched Normal
of the grown mixture scores the same. The grown mixture simply learns a
better **mean** than the joint Normal fit, whose early stopping is triggered
by σ overconfidence before the mean converges. The Normal head's own fix is
the staged fit `fit_two_stage` (docs/normal_head.md). It matches or beats the
warm-grown mixtures on Gaussian-noise targets (het_a 1.360 vs 1.378 for the
`Both` search; het_b 0.676 vs 0.725 for the grown K = 2). Real shape gains remain on t-noise,
kin8nm and multimodal targets.

**CRPS-trained search: `NGBAutoMixtureRegressor::new_crps` (Cycle 13,
R50).** Candidates are trained with CRPScore and selected by validation CRPS.
The K = 1 candidate uses the corrected Normal CRPS metric (the upstream
metric blew up; see docs/normal_head.md). On 5 unimodal targets × 6 seeds,
against the LogScore `Both` search:

- **CRPS:** better on four (kin8nm −0.006, het_a −0.017, homo −0.029, tnoise
  −0.037; 5–6/6 seeds); neutral on het_b.
- **NLL:** mixed (homo −0.033, others +0.04…+0.07).
- **Multimodal (bimodal):** worse on both (+0.13 NLL, +0.026 CRPS, 0/6).

Warm-grown candidates add nothing under CRPS training (`crps+both` −
`crps+cold`: |ΔCRPS| ≤ 0.0025, no consistent wins, 3–28 % more rounds),
because the CRPS-trained Normal does not under-fit its mean. With
`new_crps`, `search = KSearch::Cold` is enough. The CRPS-trained cold search
costs 0.8–2.5× the rounds and 1.2–3.4× the fit time of the LogScore `Both`
search.

**Point masses.** Exact repeated values (zeros) make a component collapse
onto them. Use `Hurdle<D>` (docs/hurdle.md), including
`Hurdle<NormalMixture2>`.

**Caveats.** Not a default for `NGBRegressor`: the head is a modelling
choice — check the target's shape first, or let `NGBAutoMixtureRegressor`
choose.
