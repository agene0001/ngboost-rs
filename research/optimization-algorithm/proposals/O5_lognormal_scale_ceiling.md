# O5 — A ceiling on the LogNormal scale

Cycle 19, 2026-11-14. Follow-up to the hazard recorded under O1: with
cross-fitted residuals the LogNormal head's NLL improved while its
predictive mean exploded on kin8nm.

## Observed limitation

LogNormal, default staged fit and cross-fitted fit, kin8nm, 10 seeds,
leaf 20 (`reports/O1_lib_xfit_lognormal.csv`):

| fit | NLL | RMSE of the mean: min / median / max | seeds with RMSE > 0.2 |
|---|---|---|---|
| default | −0.438 | 0.145 / 0.150 / 0.57 | 4 |
| cross-fitted | −0.560 | 0.137 / 3·10⁵ / 5.6·10²⁵ | 9 |

## Diagnosis

`research_bench` with `NGB_PARAM_DIAG=1` prints the range of every fitted
parameter and the rows that dominate the squared error.

- It is not extrapolation. The largest log σ on *training* rows is
  1.4–1.9 (σ = 4–7 in log space, against a median of 0.19); the largest
  on test rows is within −0.2…+0.7 of that.
- The test rows that carry the error are ordinary (y ≈ 0.54–0.62, fitted
  μ between −2.0 and −1.2). The reading — not checked row by row — is
  that each sits next to a training row near the target's lower bound
  (y ≥ 0.04, so log-residuals down to about −3).
- It is not a step overshoot. Replacing the natural-gradient leaf value by
  the leaf's exact minimiser (O8) leaves the largest log σ at 1.2–1.7 and
  the mean still explodes.

So the model is doing what the score asks: the additive trees isolate a
single extreme log-residual and fit σ ≈ |r| to it. NLL rewards that. The
mean exp(μ + σ²/2) is exponential in σ², and its right tail there is an
artefact of the log-normal shape: the target (bounded above, heavy lower
tail in log space) is not log-normal in its tails.

## Candidates

A guard has to bound σ; the question is against what.

1. A constant ceiling on σ. Rejected without a run: log-scale values of
   1.5–2.5 are honest for costs and claim sizes.
2. σ(x) ≤ k × the marginal scale of ln y. Clean reference, but the
   marginal scale is not stored in a fitted model, and adding a field to
   `SerializedNGBoost` breaks bincode files written by 1.2.0.
3. σ(x) ≤ k × the model's own scale constant, `init_params[scale]`. After
   a staged fit that constant is the log of the rms residual scale given
   the location (`staged_rest` refits it), after a joint fit the marginal
   one. Needs no new state.

For 2 and 3 there is a bound that does not depend on the data: the mean
of σ²(x) is the squared rms residual scale, so at most 1/k² of the rows
can have a true σ(x) above k times it (Markov), and the marginal scale is
larger still.

## Emulation (no library change)

The harness clamps the fitted scale column at reference + ln k and
rescores the same fitted models (`reports/O5_capdiag_marginal.csv`,
`reports/O5_capdiag_init.csv`, `cap_summary.py`). Rows at the ceiling per
fit and RMSE of the mean, cross-fitted fits, reference = the model's own
constant (candidate 3):

| k | kin8nm: test rows capped / RMSE median, max | lognorm_het rows | gamma_het rows | support_cost rows |
|---|---|---|---|---|
| 2 | 69 / 0.145, 0.151 | 9.1 | 0.3 | 0 |
| 3 | 31 / 0.142, 0.149 | 0.0 | 0.1 | 0 |
| 4 | 16 / 0.142, 0.148 | 0.0 | 0.0 | 0 |
| 6 | 4.9 / 0.165, 0.244 | 0 | 0 | 0 |
| 8 | 2.7 / 0.270, 1.09 | 0 | 0 | 0 |
| 12 | 1.2 / 29, 150 | 0 | 0 | 0 |
| ∞ | 0 / 3·10⁵, 5.6·10²⁵ | 0 | 0 | 0 |

NLL moves by at most 0.005 for any k ≥ 2, in the favourable direction on
kin8nm. Against the marginal scale (candidate 2) the picture is the same:
k = 2…4 flat on kin8nm and no row capped on the well-specified sets from
k = 3.

k = 4 is the smallest value at which no training or test row of the three
well-specified datasets (true heteroscedastic log-normal, gamma-distributed,
a real cost target) reaches the ceiling, and it is inside the range where
kin8nm is fully repaired. It allows a true σ(x) above the bound on at most
1/16 of the rows.

## Library change (approved by the user, implemented 2026-11-14)

- `Distribution::scale_ceiling() -> Option<f64>` (default `None`);
  `LogNormal` returns `Some(4.0)`.
- Every public prediction of `NGBoost` (`pred_param`, `pred_param_at`,
  `pred_param_best`, `pred_dist*`, `predict`, `staged_pred_dist`, `score`)
  clamps the scale column at `init_params[col] + ln k`.
- `NGBoost::pred_param_unbounded(x)` returns the unclamped sum; it is the
  opt-out. `Normal` fitted on ln y is the same model with no ceiling.
- The fit is untouched: training, early stopping, the staged stages and
  the recalibration all work on the unclamped sum, so the stored model is
  bit-identical to what 1.3.0 would have stored without the ceiling, and
  `SerializedNGBoost` is unchanged.

Measured on the shipped code (`reports/O5_ceiling_lognormal.csv`, 10
seeds, leaf 20), against the same fits without the ceiling:

| dataset, fit | NLL without → with | RMSE without → with |
|---|---|---|
| kin8nm, default | −0.4380 → −0.4388 | max 0.57 → 0.152 (median 0.150 → 0.147) |
| kin8nm, cross-fitted | −0.5597 → −0.5623 | max 5.6·10²⁵ → 0.148 (median 0.142) |
| lognorm_het, both fits | identical | identical |
| support_cost, both fits | identical | identical |

All 20 kin8nm fits have the round counts of the run made before the
ceiling existed, and the scores equal the k = 4 rows of the emulation.

Cross-fitting on LogNormal with the ceiling: −0.124 NLL (10/10 seeds) and
−0.005 RMSE on kin8nm, −0.018 on lognorm_het (9/10), −0.005 on gamma_het
(8/10), −0.007 on support_cost (10/10); means unaffected.

## Limits and failure modes

- The reference moves with the end-of-fit recalibration, which maps
  log σ ← a + b·log σ and with it the constant. With b > 1 the ceiling is
  tighter than "k times the rms scale" (by (b − 1)·ln k in log units),
  with b < 1 looser. In `tests/test_cycle19.rs` a deliberately extreme
  case (6 % of rows with 60 times the noise of the rest, a fit of some
  fifty rounds whose recalibration stretches the scale) has 6 % of the
  test rows at the ceiling — and scores 0.09–0.21 NLL
  *better* than its own unclamped sum, which predicts σ up to 10⁴ there.
- A region holding fewer than 1/16 of the rows whose true σ is more than
  four times the rms scale will be under-dispersed. Not observed; use
  `pred_param_unbounded` or `Normal` on ln y if it applies.
- The ceiling bounds σ, not the mean: with a baseline log-scale of 1 the
  bound still allows mean/median ratios in the hundreds. It removes the
  artefact, not the head's sensitivity.
- Saved LogNormal models from 1.2.0 load unchanged but now predict with
  the ceiling.
- Only `NGBoost` regression models. The survival wrapper has its own
  prediction path and the hurdle and mixture heads are separate
  distributions; none of them is clamped.
- k was read off four datasets. The emulation is flat in k = 3…4 on the
  failing one and silent on the others from k = 3, so the choice is not
  delicate there, but it is not a tuned constant either.
