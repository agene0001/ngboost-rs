# O3 — A lower bound on the StudentT degrees of freedom

Status: prototype behind `NGBOOST_STUDENTT_NU_MIN` (process-wide, default 0
= unchanged, upstream parity test still passes). Derivative checks:
`tests/test_c18_nu_floor.rs`. Results: `../research_log.md`, Cycle 18.

## Observed limitation

Known limitation 47: the 3-parameter StudentT can put ν ≤ 1 on some rows.
The mean is then undefined (`predict()` and RMSE are NaN) and CRPS is
undefined or > 10³, while NLL stays finite. Re-counting
`reports/c15/R54ce_studentt_staged_10seeds.csv` with "RMSE not finite, or
CRPS not finite or > 10³" as the criterion: 26 of 80 joint fits and 22 of
80 staged (default) fits. ν ≤ 2 (infinite variance, so `variance()` and
`std()` are infinite) is more common still. "A floor would be cheap; it
was not measured."

## Baseline

θ = (μ, log σ, log ν), ν = exp(θ₃) capped at 1e8. LogScore gradient
g = ∇_θ S and closed-form Fisher I(θ) (Lange, Little & Taylor 1989 in log
coordinates); update Δθ = −η·s·tree(I⁻¹g).

## Candidate

Restrict the family to ν > ν_min by a smooth reparameterisation:

  θ₃ = ln(ν − ν_min),   ν = ν_min + exp(θ₃),   ν_min ∈ {1, 2}.

Chain rule: ∂/∂θ₃ = (ν − ν_min)·∂/∂ν = c·∂/∂ln ν with c = (ν − ν_min)/ν ∈
(0, 1). So, with the existing log-ν expressions g₃, I₂₃, I₃₃:

  g₃′ = c·g₃,   I₂₃′ = c·I₂₃,   I₃₃′ = c²·I₃₃,

everything else unchanged. I′ = D I D with D = diag(1, 1, c), c > 0, so I′
stays symmetric positive definite and the natural gradient I′⁻¹g′ = D⁻¹
I⁻¹ g is the same tangent vector in the new coordinate. The marginal
initialisation keeps ν₀ = 3 (θ₃ = ln(3 − ν_min)). Cost: one multiply per
row. With ν_min = 0, c = 1 exactly and every expression is bit-identical.

This is not a clamp: no row is frozen at the bound (g₃′ → 0 only as
ν → ν_min, where the row is asymptotically at the boundary of the allowed
family).

It *does* change the model family: tails heavier than t(ν_min) can no
longer be represented. ν_min = 1 keeps the mean defined; ν_min = 2 the
variance and a finite CRPS.

## Verification

`tests/test_c18_nu_floor.rs` (ν_min = 2): ν > ν_min for all θ₃; `params()`
inverts `from_params`; gradient = centred finite differences in all three
coordinates, including the ν > 100 branch; Fisher = E[g gᵀ] by midpoint
quadrature over the probability integral transform (400 k points, 0.5 %).

## Predictions (falsifiable)

P1. With ν_min = 2 no fit has an undefined mean, variance or CRPS.
P2. Median test NLL moves by < 0.01 on data whose noise is not heavier
    than t(2) (all current benchmark targets); mean NLL may improve where
    single rows with ν ≈ 1 carried large losses.
P3. On genuinely infinite-variance noise the floored head must lose NLL.
    Not in the current benchmark set — a stress case to add before any
    default is considered.

## Not done

CRPS training is not rescaled (LogScore only). A process-wide switch is a
research device: a shippable version needs the bound stored with the model
(a new head type or a serialised field), since θ₃ means something
different under each ν_min.

## Outcome (Cycle 18)

P1 failed at ν_min = 1 and 2: the bound is attained in floating point, so
the quantity that is undefined *at* the bound stays undefined. With the
bound strictly inside the defined region (ν_min = 2.1) P1 holds — 0 of 80
default fits with an undefined mean, variance or CRPS — and the mean NLL
change is −0.001 (worst dataset +0.019). See the log for the table.

## Library change (approved by the user and implemented 2026-11-13)

The approved item was "a stored model field replacing the environment
variable". A model field cannot reach the head: `Distribution::from_params`
is a static function of the parameter matrix, so a per-model ν_min would
need either process-global state (what the env switch was) or a new
argument on every trait method. The bound is therefore part of the
**type**: `StudentTFiniteVar` (`src/dist/studentt_fv.rs`), ν > 2.1,
θ₂ = ln(ν − 2.1). This meets the purpose of the request — the bound
travels with the model and cannot differ between fit and load — without a
serialization change. `StudentT` is restored to its 1.2.0 source (only
`DF_MAX` became `pub(crate)`); `NGBOOST_STUDENTT_NU_MIN` and
`tests/test_c18_nu_floor.rs` are gone.

The head wraps a `StudentT` and applies J = diag(1, 1, c), c = (ν − 2.1)/ν,
to its gradient (Jᵀ g) and metric (Jᵀ M J) for both LogScore and CRPScore
(the prototype rescaled LogScore only). Unit tests: ν ≥ bound and exact
`params()` inverse, marginal fit at ν = 3, same density/CRPS/cdf as
`StudentT`, finite-difference gradients for both scores, Fisher = E[g gᵀ]
by PIT quadrature, natural-gradient invariance. Integration test
`student_t_finite_var_never_predicts_an_undefined_moment`.

**Reproduction** (`reports/O3_fv_*.csv`, `RB --dist studenttfv --variants
auto+staged`, same datasets and seeds as the ν_min = 2.1 prototype run):
all 80 fits identical to the prototype in every reported column; 0
undefined-mean, 0 bad-CRPS, 0 infinite-variance fits (ν_min = 0: 22 / 1 /
60).

**Rounding sensitivity** (`reports/O3_fvulp_*.csv`). The first version of
the head computed c = e/(2.1 + e) and θ₂⁰ = ln(exp(ln 3) − 2.1): the same
mathematics, different last bits. Over the same 80 fits:

| dataset | shipped − StudentT (se), wins | variant − StudentT (se), wins |
|---|---|---|
| het_a | −0.012 (0.007), 7/10 | −0.000 (0.017), 5/10 |
| het_b | −0.010 (0.010), 6/10 | −0.010 (0.010), 6/10 |
| homo | −0.019 (0.025), 4/10 | −0.012 (0.027), 3/10 |
| tnoise | +0.012 (0.017), 4/10 | +0.020 (0.017), 2/10 |
| lognorm_het | +0.002 (0.018), 7/10 | −0.001 (0.018), 6/10 |
| kin8nm | +0.001 (0.002), 5/10 | +0.015 (0.007), 3/10 |
| support_logcost | +0.001 (0.002), 4/10 | +0.000 (0.002), 4/10 |
| protein | +0.019 (0.009), 3/10 | +0.065 (0.027), 2/10 |
| mean | −0.001 | +0.010 |

Single fits moved by up to 0.17 NLL. Neither variant produced an
undefined row, so the conclusion about the bound stands; the NLL cost is
"0 to 0.01 on average, a few hundredths on protein", not "−0.001". The
seed standard errors in this table do not include this implementation
noise, and the same caution applies to other ≤ 0.02 differences between
StudentT variants in this log. The shipped code is the prototype's
arithmetic, so the published numbers are those of the shipped head.

