# O12 — How much of a small effect is last-bit noise?

Cycle 20, 2026-11-14. O3 found that an arithmetically equivalent rewrite
of the StudentT head changed single fits by up to 0.17 NLL. How large is
that sensitivity for the Normal head, on which most effects in this log
were measured — and what does it do to a 10-seed comparison?

## Design

`c18_bench --mode noise` (`reports/O12_noise_normal.csv`,
`noise_summary.py`). Six datasets × 10 seeds, 2100 training rows, leaf 20,
the default staged fit and the cross-fitted fit. Replicate 0 is the plain
fit. Replicates 1–3 multiply every **training** target by 1 ± 10⁻¹² with
a seeded random sign per row; validation and test targets are untouched,
so a different test score can only come from a different fit. The
perturbation is four orders of magnitude above the rounding unit and
eleven below anything a model could learn: it stands in for "the same
computation with different rounding". It is not a rebuild with another
compiler setting — Rust does not reorder floating-point arithmetic, and
the fits are bit-reproducible across builds.

## Result

| quantity | default fit | cross-fitted |
|---|---|---|
| fits whose round count changed | 179 of 180 | 180 of 180 |
| sd of one fit's test NLL across replicates | 0.018 (0.011–0.034 by dataset) | 0.018 (0.011–0.026) |
| largest change of one fit | 0.104 | 0.103 |
| sd of a 10-seed mean | 0.005 (up to 0.010) | 0.005 (up to 0.011) |

The paired effect "cross-fit − default", mean over 10 seeds, in the four
replicates:

| | rep 0 | rep 1 | rep 2 | rep 3 | sd |
|---|---|---|---|---|---|
| het_a | −0.055 | −0.054 | −0.048 | −0.059 | 0.004 |
| het_b | +0.002 | −0.008 | +0.001 | +0.001 | 0.005 |
| homo | −0.024 | −0.031 | −0.053 | −0.020 | 0.015 |
| tnoise | −0.087 | −0.060 | −0.060 | −0.076 | 0.013 |
| kin8nm | −0.042 | −0.039 | −0.039 | −0.047 | 0.004 |
| protein | −0.017 | −0.010 | −0.014 | −0.015 | 0.003 |
| mean of the six | −0.037 | −0.034 | −0.036 | −0.036 | 0.0015 |

## Reading

- The fit is chaotic at the level of single models: early stopping picks
  a round from a validation curve, and a change in the 12th digit moves
  that round in essentially every fit. This is not specific to StudentT.
- One dataset, 10 seeds: a paired difference carries about 0.007 of
  last-bit noise (0.003–0.015). **A single-dataset effect below about
  0.015–0.03 is not a finding**, whatever its seed-to-seed standard error
  says about that particular set of fits.
- Six datasets pooled: 0.0015. Averages over datasets are solid to the
  third decimal.
- The seed-to-seed standard errors in this log already contain this
  variability (each seed draws its own), so they are not too small; what
  they do not show is that the *same* seeds give a different mean after
  an equivalent rewrite.

What this changes in earlier entries: single-dataset claims of 0.01–0.02
(het_b ±0.002–0.007 under cross-fitting, the O8 candidate's −0.007…−0.017,
protein +0.019 for `StudentTFiniteVar`, the 0.009 "accidental variant" of
O5) are within the noise of one dataset. The pooled claims and every
effect above 0.03 stand.

## Decision

No code change. Reporting rule from here on: single-dataset differences
below 0.02 are reported as "no difference"; pooled differences are given
with the number of datasets.

## Limits

Normal head, one size, three perturbed replicates. The perturbation acts
on the training targets only; a change inside the tree builder or the
line search may behave differently.
