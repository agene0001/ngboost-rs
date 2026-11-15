# O2 — Leaf regularisation of the natural-gradient trees

Status: measured with existing knobs only (`HistogramLearner::
min_samples_leaf`); harness `examples/c18_bench.rs --mode leaf|leafsplit`
and `research_bench --min-leaf`. No library change. Results:
`../research_log.md`, Cycle 18.

## Observed limitation

Every run of Cycles 1–17 used depth-3 trees with `min_samples_leaf = 1`.
Two step guards (Normal log σ cap 25, mixture cap 5) were added to stop one
mechanism: a single tail row in a leaf of its own receiving its full
linearised step (research_log Cycle 11: "with min_samples_leaf = 1 it can
own a leaf; the line search accepts"). The guards cap the step; they do not
stop the split.

## Baseline mathematics

Round t, parameter j: the tree partitions the rows into leaves L and
predicts the unweighted mean of the natural gradient ñᵢⱼ = (F⁻¹g)ᵢⱼ,

  v_L = (1/|L|) Σ_{i∈L} ñᵢⱼ,

with splits chosen by variance reduction (Friedman MSE) and no lower bound
on |L|. For the Normal log σ column ñ = ½(1 − z²): Var(ñ) = ½ under the
model, but the distribution is a shifted χ²₁ with a long left tail, and
variance-reduction splitting is rewarded for isolating exactly those rows.
A leaf of m rows estimates E[ñ | leaf] with standard error √(Var ñ / m);
at m = 1 the "estimate" is the row's own noise.

## Candidates (all existing or one-line)

A. `min_samples_leaf = m` for every tree: no split may create a leaf with
   fewer than m rows. Var(v_L) ≤ Var(ñ)/m. Capacity is reduced only at the
   resolution of m rows; the number of leaves is unchanged where data is
   dense.
B. The same, separately for location and scale trees (location m_μ, scale
   m_σ) — to learn which trees benefit. Only possible today by fitting the
   parameters as separate models (the manual staged recipe).
C. (not implemented) L2 leaf shrinkage v_L = Σñ / (|L| + λ): the
   XGBoost-style continuous analogue of A.

Proper score, family, links and the optimiser are untouched; this is a
base-learner constraint.

## Predictions (falsifiable)

P1. Test NLL improves monotonically from m = 1 up to some m* and then
    degrades (under-fitting); m* scales with n and inversely with signal
    resolution, so no single m is optimal everywhere.
P2. If the mechanism is noisy scale leaves, the gain comes mostly from
    m_σ, and RMSE is unchanged. If the gain shows in RMSE too, the
    location trees were over-fitting as well.
P3. The effect is not specific to the staged fit or to the Normal head.

## Simpler competing explanations to rule out

- "Larger m just means more rounds before early stopping": compare round
  counts; and the pool run varies learning rate and depth independently.
- "It is only the synthetic 10-feature DGPs": check kin8nm and protein.

## Decision rule (declared before the full results)

Recommend a changed default only if a single m is never significantly
worse than m = 1 on any dataset and head tested. Otherwise recommend it as
the first knob to tune, with a suggested range.

## Library change (approved by the user and implemented 2026-11-13)

`HistogramLearner::new`, `default_histogram` and `histogram_learner()` now
use `min_samples_leaf = HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF = 20`.
`with_params(depth, 255, 1, 2)` is the opt-out; `DecisionTreeLearner` is
untouched. `tests/test_upstream_parity.rs` builds its learner with leaf 1
and still reproduces 1.1.2 bit for bit.

One of the open points was closed before the change: sample sizes below
500 training rows (`reports/O2_leaf_sweep_n{150,300}.csv`, default fit,
10 seeds, Δ NLL against leaf 1):

| total rows (train) | leaf 5 | leaf 10 | leaf 20 | note |
|---|---|---|---|---|
| 300 (210) | −0.016 | −0.031 | **−0.078** | leaf 20 wins on all six: −0.114 het_a (10/10), −0.109 het_b, −0.062 homo, −0.122 tnoise, −0.012 kin8nm, −0.050 protein |
| 150 (105) | −0.038 | **−0.050** | −0.015 | leaf 20: het_b RMSE +0.288 (3.9 se), NLL +0.072 homo and +0.054 protein (n.s.); −0.156 tnoise, −0.061 kin8nm |

So the default holds down to about 200 training rows and is a wash at
about 100, where leaf 5–10 is the better choice (documented on the
constant and in `docs/known_limitations.md` 57). At 105 training rows a
validation block has 15 rows, so these fits are joint and uncalibrated.

The default learner is also what `NGBClassifier`, `NGBMultiClassifier`,
the hurdle and mixture heads use, which the 38 cases above did not cover.
Checked after the change (`reports/O2_heads_{cat3,cat5,cat8,hurdle,mix2}_ml{1,20}.csv`,
10 seeds, leaf 20 − leaf 1, paired by `leaf_pair.py`):

| head | dataset | Δ NLL (se) | seeds | Δ RMSE (se) |
|---|---|---|---|---|
| Categorical3 | cat3_nl | −0.011 (0.002) | 10/10 | — |
| Categorical3 | cat3_imb | −0.001 (0.001) | 6/10 | — |
| Categorical5 | cat5_nl | −0.002 (0.001) | 6/10 | — |
| Categorical8 | support_dzgroup | −0.006 (0.002) | 10/10 | — |
| HurdleNormal | zi_lognorm | −0.014 (0.027) | 6/10 | −0.056 (0.078) |
| HurdleLogNormal | zi_lognorm | −0.005 (0.005) | 6/10 | −0.025 (0.078) |
| HurdleNormal | zi_normal | −0.007 (0.009) | 6/10 | −0.002 (0.012) |
| HurdleNormal | support_adlsc | −0.003 (0.001) | 9/10 | −0.003 (0.001) |
| HurdleLogNormal | support_adlsc | −0.001 (0.001) | 6/10 | −0.002 (0.001) |
| NormalMixture2 | bimodal | −0.004 (0.003) | 5/10 | +0.000 (0.003) |
| NormalMixture2 | trimodal | −0.007 (0.002) | 9/10 | −0.003 (0.005) |
| NormalMixture2 | het_a | +0.001 (0.024) | 6/10 | +0.020 (0.033) |

(HurdleLogNormal on zi_normal is not a valid case: the target has
negative values.) Twelve more cases, none significantly worse, effects an
order of magnitude smaller than for the location–scale heads: 50 cases in
all.

Still not measured: weighted fits, the exact tree learner, the survival
wrappers (which inherit the default), a per-parameter leaf size.
