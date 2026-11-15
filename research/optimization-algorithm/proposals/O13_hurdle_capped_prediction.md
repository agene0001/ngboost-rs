# O13 — A point prediction for HurdleLogNormal that cannot exceed the data

Cycle 21, 2026-11-14. Approved by the user after O9, including the change
to the saved-model format it needs.

## Observed limitation

O9: on a bounded target with a point mass at its floor, one
`HurdleLogNormal` fit in four predicted a mean of 31 against a largest
target of 1.45 (σ ≈ 3.4 on a few rows after an early stop at 80 rounds).
The hurdle's fit is joint, so the scale ceiling of O5 has only the
marginal log-scale to measure against (0.9 here) and would sit above the
offending σ. No bound relative to the model's own scale reaches this
case; the mean exp(μ + σ²/2) is the fragile object, not σ.

## Baseline and candidate

Baseline point prediction: the mean, (1 − π)·exp(μ + σ²/2).

Candidate: the mean of the predictive distribution winsorised at the
largest target c seen in training,

    (1 − π)·E[min(Y, c)],
    E[min(Y, c)] = exp(μ + σ²/2)·Φ((ln c − μ − σ²)/σ) + c·(1 − Φ((ln c − μ)/σ)).

Properties: 0 ≤ value ≤ min(mean, c); increasing in c; equal to the mean
as c → ∞; the shortfall against the mean is E[(Y − c)⁺], the part of the
mean carried by values no training row reached. It is the mean of
min(Y, c), i.e. the squared-error-optimal prediction of a target known not
to exceed c. Alternatives considered: clamping the mean at c (crude: every
blown-up row predicts exactly c) and the median (changes the functional
for every row, including the well-behaved ones).

Numerics: the first factor overflows exactly where Φ underflows, so both
terms are evaluated on the log scale with the stable `norm_ln_sf`; a σ of
40, whose plain mean is infinite, gives a finite value.

Verification (`tests/test_cycle20.rs`): the closed form against a 200 000
point quantile rule for five (μ, σ, c) within 0.2 %; monotone in c; equal
to the mean at c = ∞ and within 10⁻⁹ at c = 10⁹.

## What is stored

`NGBoost::target_max`: the largest finite target passed to `fit`
(training and validation rows); a `partial_fit` keeps the larger of the
old and the new value. It is the one new number in the saved model:
`SerializedNGBoost::target_max`, the last field, with a serde default.

bincode is positional, so the format question is real:

| writer → reader | result |
|---|---|
| ≤ 1.2.0 → 1.3.0, `load_model` or `SerializedNGBoost::from_bincode` | loads; `target_max = None`; plain mean |
| ≤ 1.2.0 → 1.3.0, a caller's own `bincode::deserialize::<SerializedNGBoost>` | **fails** (the bytes end one field early): switch to `from_bincode` |
| 1.3.0 → 1.2.0 | loads; bincode's default reader ignores trailing bytes |
| JSON or another self-describing format, either way | loads |

`from_bincode` retries a failed decode with the one-byte encoding of
`None` appended; truncated garbage still fails. All four rows are tested.

## Scope

`Distribution::winsorized_mean` (the formula; `LogNormal`) and
`Distribution::predict_capped` (what `NGBoost::predict` and
`staged_predict` consult; `Hurdle<D>` when D has a winsorised mean).
Plain `LogNormal` does **not** implement `predict_capped`: its `predict`
is unchanged, as are `HurdleNormal`, the survival wrapper and every
`pred_dist(x).mean()`.

## Result

Constructed floor-atom target (`tests/test_cycle20.rs`): the plain mean
exceeds five times the largest target on at least one of three seeds;
`predict` never exceeds the largest target.

Benchmark (`reports/O13_hurdle_capped.csv` against the plain-mean files
`O2_heads_hurdle_ml20.csv` and `O9_hurdle_zi_kin8nm.csv`; 10 seeds, leaf
20). The fits are identical in all 60 cases (same NLL, same rounds); only
the point prediction differs:

| dataset | RMSE, plain mean | RMSE, shipped | change (se), seeds lower |
|---|---|---|---|
| zi_lognorm | 5.980 | 5.935 | −0.82 % (0.49), 9 of 10 |
| zi_kin8nm | 0.3395 | 0.3372 | −0.64 % (0.25), 10 of 10 |
| support_adlsc | 1.8766 | 1.8782 | +0.08 % (0.10), 4 of 10 |

`HurdleNormal` is identical to the last digit on all three.

## Limits

- A bound, not a repair: the fitted σ on the affected rows is unchanged,
  so intervals and `pred_dist(x).mean()` still reflect it, and a
  prediction close to c on such a row is still a poor one.
- A future target above every training target is under-predicted by
  construction.
- Models saved before 1.3.0 keep the plain mean until refitted.
- Three benchmark datasets and one constructed case; the small RMSE gains
  are below what O12 allows as a single-dataset finding, so the claim is
  "no cost", not "better".

## Decision

Implemented as described.
