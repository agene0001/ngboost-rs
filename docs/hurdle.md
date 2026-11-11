# `Hurdle<D>`: a point mass at zero plus a continuous head

Research Cycle 13 (2026-11-10). Reports R49 (partial), R49c, R49d.

## What it is

Y = 0 with probability π(x); otherwise Y ~ D(x), for any continuous head D
in the crate. Aliases:

- `HurdleNormal` = `Hurdle<Normal>` (zeros *inside* the support).
- `HurdleLogNormal` = `Hurdle<LogNormal>` (zeros plus a positive, skewed
  part).

`Hurdle<NormalMixture2>` and other combinations work too. Typical targets:
intermittent demand, rainfall, claims, costs, and scores with a floor —
anything with **exact** zeros next to a continuous spread. For an atom at
another value c, model y − c.

```rust
use ngboost_rs::dist::HurdleLogNormal;
let mut m: NGBoost<HurdleLogNormal, LogScore, HistogramLearner> =
    NGBoost::new(2000, 0.05, HistogramLearner::new(3));
m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)?;
let d = m.pred_dist(&x_test);
// d.pi = P(Y = 0); d.inner = the continuous part; mean/cdf/ppf/interval include the atom
```

## Maths

- **Parameters:** [logit π, D's parameters…].
- **Log score** (w.r.t. counting measure at 0 plus Lebesgue measure
  elsewhere): −ln π at y = 0, and −ln(1 − π) − ln f_D(y) otherwise.
- **Fisher information** (expected over Y, exact) is block diagonal:
  π(1 − π) for the logit and (1 − π)·I_D for D's block.
- **Natural gradient:**
  - logit: the Bernoulli step;
  - D's parameters: on non-zero rows, D's own natural gradient ÷ (1 − π);
    on zero rows, 0. This is unbiased for D's natural gradient in
    expectation.
- D's own metric, guards and fused fast paths are reused unchanged.
- **Distribution functions:**
  - cdf = (1 − π)F_D(y) + π·1{y ≥ 0};
  - ppf is the generalised inverse (quantiles inside the atom's band are 0);
  - mean = (1 − π)E_D[Y].
- **Recalibration:** the default scale recalibration acts on D's scale
  columns.

**Tests:** finite-difference gradients; metric = E[∇∇ᵀ] by quadrature plus
the atom; dense, diagonal and generic natural gradients agree; cdf/ppf at
the atom; smoothed init; target validation; and end-to-end training
(`tests/test_cycle13.rs`).

**Comparing heads.** The hurdle NLL is w.r.t. a mixed measure, so it is
comparable only between hurdle heads. Compare against continuous heads with
the **CRPS**, which is proper for mixed laws. The bench computes it by
quadrature split at 0, at y and at the 1 %/99 % quantiles; this matches the
closed-form Normal CRPS to a relative 3·10⁻⁵.

## Results (10 seeds, default log-σ guard; ΔCRPS vs the Normal head)

| target | zeros | Normal CRPS | HurdleNormal | HurdleLogNormal | Hurdle<Mix2> | Brier of P(0): Normal → best hurdle |
|---|---|---|---|---|---|---|
| zi_lognorm (synthetic) | 38 % | 3.40 (median 2.52) | median 2.34 | **2.04 (−1.36, 10/10)** | 2.18 (10/10) | 0.377 → 0.155 |
| zi_normal (synthetic) | 37 % | 0.917 | **−0.092 (10/10)** | n/a (y < 0) | −0.057 (10/10) | 0.372 → 0.150 |
| support2 ADL score `adlsc` (real) | 35 % | 1.028 | −0.054 (10/10) | −0.053 (10/10) | **−0.065 (10/10)** | 0.347 → 0.206 |
| support2 micro-cost `totmcst` (real) | 1.6 % | 13 860 | −73 (7/10) | **−1 645 (−12 %, 10/10)** | −1 107 (10/10) | 0.016 → 0.015 |

- The **continuous part matters as much as the atom.** On `totmcst` the atom
  is tiny and the gain comes from the skewed positive part (HurdleLogNormal);
  HurdleNormal ≈ Normal there.
- `adlsc` also has integer atoms (1, 2, 6, … where the ADL score was
  observed, not imputed). The hurdle handles only the zero; a mixture as the
  continuous part helps a little more.
- **Normal mixtures without the atom collapse onto it.** On the R49 run,
  `NGBAutoMixtureRegressor` had test NLL −55 and CRPS 10⁵² on zi_lognorm,
  and NaN on `adlsc`.

## Caveats

- The atom is at **exactly** 0 (`y == 0.0`). Values like 1e-12 count as
  continuous.
- `HurdleLogNormal` needs y ≥ 0. Negative values fail target validation
  (`totmcst` has 2 negative accounting entries; the bench drops them).
- "NLL on non-zero rows" is reported by the bench but is not a fair
  cross-head comparison: it ignores the zero rows the hurdle gains on and
  charges it for the mass it puts on 0.
- CRPS training exists (below) but is not recommended.

## CRPS training rule (Cycle 14, R56) — implemented, not recommended

`Hurdle<D>: Scorable<CRPScore>` for D = Normal and D = LogNormal (trait
`HurdleCrpsPart`, which supplies D's **y-scale** CRPS pieces; for
LogNormal this is the Baran & Lerch (2015) closed form, not the crate's
ln-y CRPS). With F_H = (1 − π)F_D + π·1{t ≥ 0}:

- **Score:** CRPS_H(y) = (1 − π)²C(y) + π²|y| + 2π(1 − π)G(y), where
  C = CRPS_D and G = ∫₀^y F_D (y > 0) or ∫_y^0 (1 − F_D) (y < 0).
- **Metric** 2∫∇F_H∇F_Hᵀ dt in (logit π, θ):
  - ℓℓ: 2π²(1 − π)²C(0);
  - ℓθ: −π(1 − π)² ∂C(0)/∂θ;
  - θθ: (1 − π)² M_D;
  - solved per row (3×3).

The score, gradient and metric match quadrature (split at the atom and at
y) to ≤ 10⁻⁵.

As a training rule it loses to LogScore training of the same head: CRPS is
worse in 5 of 7 dataset/head cells (0–2/10 seed wins), and NLL is far worse
everywhere (one HurdleLogNormal seed exploded on `adlsc`, CRPS ≈ 4·10⁸).
The single CRPS win, HurdleNormal on zi_lognorm, beats a LogScore fit that
blew up on some seeds. This is the same pattern as CRPS training of
mixtures: a point mass plus a spread is not single-mode. Use LogScore.
