# O8 — Exact leaf step for a Gaussian log-scale (rejected)

Cycle 19, 2026-11-14. A side question raised while diagnosing O5. Not
approved, not in the tree; the prototype is `O8_exact_scale_step.patch`
(applies to the working tree of 2026-11-14 with `git apply`).

## Theoretical question

For Normal and LogNormal under LogScore the scale parameter s = log σ
enters a row's score as s + r²·e^(−2s)/2. Its gradient is 1 − z² and its
Fisher entry is 2, so the natural gradient is (1 − z²)/2 and a leaf's
fitted value is v = (1 − m)/2 with m the leaf's mean z².

- Baseline step of the leaf: s ← s − η·c·v = s + η·c·(m − 1)/2, linear
  in m (η learning rate, c the line-search scale).
- The shift that minimises the leaf's own score is ½·ln m.

Since ln m ≤ m − 1, the natural-gradient value over-states the exact one
without bound: one row with z = 60 in a 20-row leaf gives m ≈ 180, an
exact shift of 2.6 and, at η = 0.05, an applied shift of 4.5. The step
back is at most η/2 per round. Could that overshoot be what makes
cross-fitted scale fits produce absurd σ on a few rows (O5)?

## Candidate

Replace the fitted leaf value v by φ(v) for the log-scale column:

    φ(v) = −½·ln(1 − (2 + τ)·v)   if v < 0   (the leaf wants a larger σ)
    φ(v) = v                        otherwise

(τ the Tikhonov term). φ depends on the leaf value alone, so it is applied
to the stored tree's leaves and the model needs no new state. The fixed
point is unchanged (v = 0 ⇔ m = 1, the maximum-likelihood condition), φ is
C¹ at 0, and shrinking steps are left as they are because there the
natural gradient is the more cautious of the two. Exact for weighted fits
whenever the tree's weights are the loss weights (the Fisher entry is a
constant, so Fisher weighting does not disturb this).

## Result

`research_bench`, staged default (`auto`) and cross-fitted (`auto+xfit`)
fits with and without the candidate (`+exs`), leaf 20, 10 seeds.

| | ΔNLL of `+exs` | seeds lower |
|---|---|---|
| Normal, default fit, six datasets | mean −0.0001 (−0.013…+0.005) | 3–6 of 10 |
| Normal, cross-fitted, six datasets | mean −0.006: het_a −0.017 (0.008), het_b −0.007 (0.004), homo −0.008 (0.008), others within ±0.002 | 8, 8, 8, 6, 5, 5 |
| LogNormal on kin8nm, default fit | −0.018 (0.008) | 8 of 10 |
| LogNormal on kin8nm, cross-fitted | −0.009 | — |

The prediction that mattered failed: with the candidate the largest fitted
log σ on kin8nm is still 1.2–1.7 and the predictive mean still explodes
(RMSE up to 2·10⁴). The large σ are not an overshoot. They are the
additive model isolating single rows whose log-residual is 2.5–3.5 and
fitting σ ≈ |r| to them — which is the score-optimal thing to do.

## Decision

Rejected as a fix for O5 and not pursued as a default: no effect on the
default fit, a small and only partly consistent gain on cross-fitted
Normal fits (−0.006 on average). It would be the first candidate to
revisit if cross-fitting became a default, since it costs nothing at
prediction time. Reports: `reports/O8_exs_normal.csv`,
`reports/O8_exs_lognormal_kin8nm.csv`.
