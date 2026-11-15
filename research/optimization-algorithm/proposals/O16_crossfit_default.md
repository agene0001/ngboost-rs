# O16 — Cross-fitting as the default of the staged fit

Status: test plan, written 2026-11-14 **before any run of this cycle**.
Results are appended below the line "Results" once the runs are in. No
library default is changed by this cycle; a change needs the user's
approval.

## Question

`staged_crossfit_folds` (O1) is opt-in. In the tuner comparison A9 the
search switched it on in 52 of 56 winning models and gained −0.043 NLL.
Should the staged fit cross-fit by default (K = 5), for every head that
stages by default (Normal, LogNormal, StudentT, StudentTFiniteVar,
GammaMean)?

What is already known (O1, O5, O11, A9; seeds 0–9, explicit 70/10/20
split, leaf 20 fixed, mostly 2100 training rows): −0.01…−0.09 NLL on most
datasets, a loss on protein at 490 rows (+0.069) and with 30 validation
rows (+0.094), 2.6–4.8× fit time.

What is not known and is tested here:

1. the model a user actually gets today (adaptive leaf, LogNormal ceiling,
   the fit's own 10 % validation split) rather than the research settings;
2. sizes from 100 to 6000 rows, including the sizes where the default
   does not stage today (fewer than 40 validation rows; 300 for
   GammaMean) and the option would make it stage;
3. the plain `StudentT` head (never run with cross-fitting);
4. fresh seeds: every earlier cross-fit number used seeds 0–9;
5. K = 3 as a cheaper variant; clean (uncontended) fit times;
6. sample weights and the chronological split under the option.

## Design

`c18_bench --mode xdef`. Model: `hyper_opt::default_histogram_model` with
an empty parameter map (learning rate 0.05, depth 3, leaf adaptive,
round cap 2000, patience 50), seeded, fitted by `fit(x, y)` on a pool of
`size` rows; the fit holds out 10 % itself. Scored on 2000 rows outside
the pool (synthetic: drawn with the pool; real: the next 2000 rows of the
seed's shuffle).

- Arms: `off` (the shipped default), `k5` (`staged_crossfit_folds = 5`,
  the candidate), `k3` (secondary).
- Head × dataset (19 pairs): Normal × {het_a, het_b, homo, tnoise, kin8nm,
  protein}; StudentT and StudentTFiniteVar × {het_a, tnoise, kin8nm,
  protein}; LogNormal × {lognorm_het, kin8nm}; GammaMean × {gamma_het,
  lognorm_het, kin8nm}.
- Pool sizes: 100, 200, 400, 800, 1600, 3200, 6000 (133 cells).
- Seeds 100–109 (stage 1). Seeds 200–209 are reserved for confirming any
  rule that is chosen after looking at stage 1.
- Supplementary: 20 000 rows (Normal on het_a, tnoise, protein, 5 seeds);
  serial timing; `--weights tri`; `--split chrono`.

Metrics per fit: held-out NLL (primary), RMSE of `predict`, 90 % interval
coverage, rounds, wall time, whether the fit staged, all outputs finite.
Paired differences are `k5 − off` on the same pool, test rows and seed.
Standard errors are over seeds within a cell, and over the 19
head × dataset means for pooled figures.

## Candidate rules

- **U** — cross-fit whenever a head stages by default; as the option
  works today, this also makes the fit stage below the 40-validation-row
  threshold.
- **S** — cross-fit only where the shipped default already stages (the
  recalibration condition is kept). Readable from the same runs: `k5`
  where `off` staged, `off` elsewhere.
- **G(N)** — as U, but only for pools of at least N rows.

U is the primary hypothesis. S and G(N) are fallbacks, to be looked at
only if U fails, and any of them chosen that way must be confirmed on
seeds 200–209 before it is recommended.

## Decision rule (fixed before the runs)

A rule is recommended as the default only if, over the cells where it
turns cross-fitting on:

- **R1 benefit.** Pooled NLL difference ≤ −0.02 and at least 2 standard
  errors below zero, and `k5` lower on at least 60 % of paired fits.
- **R2 no harm.** No head × dataset pair worse by more than 0.02 at more
  than 2 standard errors; no single cell worse by more than 0.05 at more
  than 2 standard errors. (0.02 is the last-bit noise band of O12.)
- **R3 point predictions.** Pooled relative RMSE change at most +0.5 %;
  no head × dataset pair worse by more than 1 % at more than 2 standard
  errors.
- **R4 robustness.** No fit error and no non-finite output that the `off`
  arm does not also have.

Fit time is reported, not gated: whether the accuracy is worth the time
is the user's decision. K = 3 replaces K = 5 in the recommendation only
if its pooled NLL is within 0.005 of K = 5's and it passes R1–R4 itself.

## What a default would need in the library (not done here)

- With fewer than 2K training rows the option is an error today; a
  default has to fall back to the in-sample fit instead.
- An explicit off switch (today 0 means off; a default-on rule needs
  "unset" to differ from 0, or a negative convention).
- `hyper_opt` folds would cross-fit too: A9 measured the search time at
  105–451 s against 30–63 s.

## Results

### Stage 1 (seeds 100–109, 1330 paired fits per arm, no failures)

Files: `reports/O16_xdef_<head>.csv`; summaries
`reports/O16_xdef_stage1_*.txt` (`xdef_summary.py`).

**U fails R2.** Pooled it looks fine (−0.047 NLL, se 0.010, lower on 79 %
of fits, RMSE −2.0 %), but one cell is a significant loss — Normal on
kin8nm at 200 rows, +0.116 (se 0.047, 3 of 10) — and at 100 rows the
whole comparison is a coin toss with large swings (−0.019, se 0.038;
single cells from −0.31 to +0.30). Below 400 rows the fit has fewer than
40 validation rows; the shipped default is then a joint fit, and the
option replaces it by four stages early-stopped on 10–20 rows.

| pool rows | NLL, k5 − off (se over 19 pairs) | fits lower | RMSE | default stages today |
|---|---|---|---|---|
| 100 | −0.019 (0.038) | 124/190 | −1.9 % | no |
| 200 | −0.080 (0.023) | 149/190 | −5.8 % | no |
| 400 | −0.040 (0.010) | 143/190 | −1.5 % | yes, except GammaMean |
| 800 | −0.058 (0.010) | 165/190 | −2.1 % | yes, except GammaMean |
| 1600 | −0.067 (0.013) | 165/190 | −1.8 % | yes, except GammaMean |
| 3200 | −0.036 (0.008) | 154/190 | −0.5 % | yes |
| 6000 | −0.032 (0.008) | 153/190 | −0.3 % | yes |

Fallbacks on the same data (looked at only because U failed):

| rule | pooled NLL (se) | fits lower | RMSE | R1–R4 |
|---|---|---|---|---|
| S (only where the default stages) | −0.040 (0.008) | 693/860 | −0.8 % | pass |
| G(400) | −0.046 (0.007) | 780/950 | −1.2 % | pass |
| G(200) | −0.052 (0.008) | 929/1140 | −2.0 % | fails R2 (the kin8nm cell) |
| G(400) with K = 3 | −0.044 (0.008) | 736/950 | −1.2 % | pass |

G(400) and S differ only for GammaMean at 400–1600 rows, where the
default is a joint fit today (its recalibration needs 300 validation
rows) and cross-fitting makes it stage: −0.013 … −0.186 NLL in all nine
cells, RMSE −0.6 … −10.6 %.

K = 3 at 400+ rows is within 0.003 NLL of K = 5 and passes the rule, at
3.0× instead of 3.9× the wall time of the default (contended timings).

### Stage 2 plan (written before the stage-2 runs)

The fallbacks were chosen after seeing stage 1, so they are re-tested on
seeds 200–209 (new synthetic draws, new shuffles of the real data), same
heads, datasets and arms, sizes 200, 300, 400, 800, 1600, 3200, 6000.

- **Rule V40** (= G(400) in library terms): heads that stage by default
  cross-fit when the staged fit has at least 40 validation rows — the
  general recalibration threshold — including GammaMean, which then
  stages from 40 validation rows instead of 300. Below that, nothing
  changes. Evaluated on sizes ≥ 400.
- **Rule S** as the conservative alternative, same sizes.
- V40 is recommended if it passes R1–R4 on stage 2; otherwise S if S
  passes; otherwise neither.
- K: K = 3 is recommended over K = 5 if, on stage 2 under the recommended
  rule, its pooled NLL is within 0.005 of K = 5's and it passes R1–R4.
- Sizes 200 and 300 are run for information only (does the boundary sit
  in a sensible place?) and cannot change the recommendation.

Supplementary runs (information, not part of the rule): 20 000 rows;
random sample weights; the chronological split; serial timings.

### Stage 2 (seeds 200–209, 950 paired fits per arm at 400+ rows, no failures)

Files: `reports/O16_xdef2_<head>.csv`; summaries
`reports/O16_xdef_stage2_*.txt`, both stages together
`reports/O16_xdef_both_V40_*.txt`.

| rule, K | pooled NLL (se over 19 pairs) | fits lower | RMSE (se) | coverage off → on | R1–R4 |
|---|---|---|---|---|---|
| **V40, K = 5** | −0.044 (0.007) | 763/950 (80 %) | −1.0 % (0.3) | 0.899 → 0.895 | pass |
| **V40, K = 3** | −0.043 (0.007) | 738/950 (78 %) | −0.7 % (0.5) | 0.899 → 0.895 | pass |
| S, K = 5 | −0.038 (0.008) | 677/860 (79 %) | −0.6 % (0.2) | 0.902 → 0.897 | pass |

**V40 is confirmed**, and K = 3 is within 0.002 NLL of K = 5 and passes
the rule itself, so by the rule fixed above the recommendation is V40
with K = 3. Both stages together (1900 paired fits per arm): K = 5
−0.045 (0.007), 81 % lower, RMSE −1.1 %; K = 3 −0.043 (0.007), 78 %,
RMSE −0.9 %.

By size, stage 2, K = 5: −0.031, −0.062, −0.059, −0.042, −0.029 at 400,
800, 1600, 3200, 6000 rows. By head × dataset over both stages (K = 5,
400+ rows): 18 of 19 pairs better, from −0.008 (Normal, homo) to −0.094
(GammaMean, kin8nm); the exception is **Normal on protein, +0.005
(0.008)**, with RMSE +0.5 % (0.1; 4.9 standard errors, below the 1 %
bound of R3). Normal on kin8nm also gives up 0.5 % RMSE (2.1 se) for
−0.042 NLL.

GammaMean at 400–1600 rows (joint today, staged under V40), stage 2:
−0.029 … −0.169 NLL in all nine cells (all significant), RMSE −2.0 …
−9.1 %.

Below the threshold (information only): 200 rows −0.068 (0.028), 300 rows
−0.094 (0.024), no cell a significant loss, but Normal on protein
+0.26 and +0.18 (1.9 and 1.7 se). With stage 1's kin8nm cell this is the
same picture: larger average gain, heavier single losses. The threshold
stays at 40 validation rows.

### Supplementary

- **20 000 rows** (`O16_xdef_large_*.csv`, 5 seeds): the gain shrinks —
  K = 5 −0.015 (het_a), −0.023 (tnoise), −0.029 (protein) for Normal,
  −0.004 for StudentTFiniteVar on protein; pooled −0.018 (K = 5), −0.021
  (K = 3).
- **Random sample weights** (`O16_xdef_tri_*.csv`): −0.042 pooled, lower
  on 66 of 80; Normal on protein at 800 rows +0.066 (1.5 se).
- **Chronological split** (`O16_xdef_chrono_*.csv`; contiguous folds):
  −0.040 pooled, lower on 68 of 80; one protein seed loses 1.04 (the
  default's own score there is 3.34 against a typical 2.9).
- **Wall time, one fit at a time** (`O16_xdef_serial_*.csv`, 3 seeds, ten
  head × dataset × size cases): K = 5 2.4–6.6× the default (median
  5.6×), K = 3 0.9–5.0× (median 3.2×). The earlier 2.6–4.8× for K = 5
  was measured with a fixed leaf of 20 on the research split.
- **Point-prediction outliers.** Of 5320 cross-fitted fits in stages 1
  and 2, one has an RMSE more than 1.5× the default's: LogNormal on
  lognorm_het, 800 rows, seed 206, K = 3 — 19.9 against 4.6, with a
  *better* NLL. One test row carries 92 % of the squared error: log σ =
  1.00 against a scale constant of −0.29, a ratio of 3.6, just under the
  ceiling of 4 (O5), so exp(μ + σ²/2) = 901 for a target of 46. K = 5 on
  the same data is normal. It is the O5 hazard, not a new one, and one
  event in 560 cross-fitted LogNormal fits cannot say whether K matters.

### Reading

1. Always-on, as first proposed, fails: below 40 validation rows it
   replaces a joint fit by four stages stopped on a handful of rows, and
   single datasets lose badly.
2. From 40 validation rows up, cross-fitting is a consistent gain for all
   five staged heads: about −0.045 NLL, RMSE about 1 % better, coverage
   unchanged, on 18 of 19 head × dataset pairs, confirmed on seeds that
   were not used to choose the rule.
3. The price is fit time: about 3× with K = 3 and 5–6× with K = 5, for
   every default fit and every fold of a tuner search.
4. The Normal head on a heavy-tailed target (protein) gains nothing and
   loses 0.5 % RMSE; StudentTFiniteVar on the same data gains.
5. The gain decays with n: −0.03 at 6000 rows, −0.02 at 20 000.

Not tested: explicit validation sets (the rule is stated in validation
rows, so it would apply), more than 20 000 rows, the survival wrapper,
tuned (non-default) shapes beyond A9, and an upper size limit above
which the default could switch itself off again.

**Decision: none taken.** The default is unchanged; the recommendation
(V40, K = 3) is with the user.
