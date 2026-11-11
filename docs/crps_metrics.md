# CRPS metrics: audit and fix (research Cycles 13–14)

NGBoost's natural gradient under CRPScore uses the **CRPS metric**

&nbsp;&nbsp;&nbsp;&nbsp;M(θ) = 2∫ ∇_θF(t) ∇_θF(t)ᵀ dt  (Σ_k for discrete heads),

the expected Hessian of the CRPS. It has the units of y, like the CRPS and
its gradient, so the natural-gradient step is dimensionless: changing the
units of y must not change the fit.

## Audit

`tests/test_crps_audit.rs` checks every head with a `CRPScore`
numerically against its own cdf: the score against ∫(F − 1{t ≥ y})² dt, the
gradient against finite differences, and the metric against 2∫∇F∇Fᵀ dt. It
also checks that the implemented/reference metric ratio is 1 at scales
×0.1, ×1 and ×10 (and λ = 0.5, 3.2, 30 for Poisson).

| head | score, gradient | metric before | ratio vs truth across scale ×0.1 / ×1 / ×10 | now |
|---|---|---|---|---|
| Normal, NormalFixedMean, LogNormal (ln y) | ok | σ × the true metric (upstream) | ∝ σ | fixed (Cycle 13) |
| NormalFixedVar, Laplace, Exponential | ok | correct | 1 | — |
| HalfNormal | ok | E_Y[g²] = 0.197σ² | 0.035 / 0.35 / 3.5 | σ/√π |
| Gamma | ok | E_Y[g gᵀ] by quadrature | 0.068 / 0.68 / 6.8 | K(α)/β by quadrature in w = βt |
| Weibull | ok | E_Y[g gᵀ] by quadrature | 0.019 / 0.19 / 1.9 (and 0.063 / 0.63 / 6.3) | (λ/k)·K(k), closed-form ∇F |
| StudentT (μ, ln σ, ln ν) | ok | E_Y[g gᵀ] by quadrature | 0.046 / 0.46 / 4.6 (…) | ν-only integrals × σ^{±1} (below) |
| TFixedDf | ok | loc 1/3 (= E[(2F − 1)²]), scale c·σ² | 0.051 / 0.51 / 5.1 | 2A/σ, 2Bσ |
| TFixedDfFixedVar | ok | 1/3 | (−27 %) | 2A |
| Poisson | ok | E_Y[g²] | 0.50 (λ = 0.5), 1.08 (3.2), 3.24 (30) | 2λ² Σ pmf(k)² |
| NegativeBinomial (this crate) | ok | 2Σ∂F∂Fᵀ by construction | 1 (μ = 0.5, 30) | — |
| HurdleNormal, HurdleLogNormal (this crate) | ok | 3×3, by construction | 1 (×0.1 / ×1 / ×10) | — |
| NormalMixture (this crate) | ok | diagonal of 2∫∇F∇Fᵀ (by design, R39) | diagonal 1 (×0.1 / ×1 / ×10) | — |

All metrics match the definition to ≤ 3·10⁻⁵ relative (the mixture: its
diagonal), and have ratio 1 at every scale. Scores match to ≤ 7·10⁻⁵ and
gradients to ≤ 6·10⁻⁶. `crps_metrics_match_definition` asserts all of this.
Cauchy has no CRPScore: its CRPS is infinite.

The reference integrals use a midpoint grid with both 0 and y on cell
boundaries. Without that alignment, the hurdle heads' jump at 0 and the
indicator's jump at y cost O(h) accuracy: up to 10⁻⁴ for the metric and
10⁻² for the heavy-tailed t scores, where h ≈ 0.03.

**t family.** With F = T_ν((t − μ)/σ), σ factors out:
M = [[2A/σ, 0, 0], [0, 2σB, −2σC], [0, −2σC, 2σD]] with A = ∫f², B = ∫z²f²,
C = ∫z f ∂T/∂ln ν and D = ∫(∂T/∂ln ν)², all functions of ν alone:
- ν ≤ 200: tabulated once on 400 log-spaced points and interpolated;
- ν > 200: A and B by quadrature, and C ≈ 5/(32√π ν), D ≈ 31/(256√π ν²)
  from T_ν ≈ Φ − φ(z³ + z)/(4ν).

`NGBOOST_CRPS_METRIC_LEGACY=1` restores every upstream metric (A/B only).

## Side finding: the documented Weibull divergence

`tests/accuracy_parity.rs::tikhonov_reg_rescues_diverging_weibull_crps`
documented that a censored-CRPS Weibull fit (seed 17) diverges without
Tikhonov regularisation.

- With the legacy metric it still diverges.
- With the CRPS metric it converges unregularised to a healthy CRPS (0.368).

The divergence was the metric's units bug. The test now asserts
convergence.

## Effect on CRPS-trained models (R55; legacy → fixed, same splits)

| head | data | legacy → fixed |
|---|---|---|
| HalfNormal | raw support costs (y ~ 10⁴) | NLL 11.66 → 11.06 (−0.60), CRPS −8 795, 90 % coverage 0.83 → 0.92 (6/6 seeds) |
| HalfNormal | lognorm_het, kin8nm | −0.074, −0.003 NLL (5/6 seeds each) |
| StudentT (R55b–c) | tnoise, het_a, kin8nm (10 seeds each) | NLL better on 30/30 seeds: median −0.37, −0.34, −0.13; CRPS median −0.37, −0.41, −0.02 (27/30). Legacy: one tnoise test NLL 7.6·10⁴²; two het_a CRPS ≈ 10⁷ (ν ≈ 1 rows, known_limitations 47); two kin8nm fits never early-stopped (≈ 2000 rounds) and still lost. Fixed: one ν ≈ 1 row each on tnoise and het_a |
| Poisson | 4 count sets, λ ≈ 1–20 | neutral (all within ≈ 1 se) |
| Gamma (R58) | lognorm_het | NLL −0.29 (se 0.02), CRPS −0.49 (−19 %), 10/10 seeds |
| Gamma (R58) | raw support costs | NLL −0.39 (se 0.01), CRPS −4 241 (−24 %), 10/10 |
| Gamma (R58) | kin8nm | NLL −0.145 (se 0.014), CRPS −0.021, 10/10 |
| Weibull (R58) | lognorm_het | CRPS better 10/10 (median −0.27); legacy test NLL blew up (≥ +2 vs fixed, up to ∞) on 5/10, fixed on 1/10; median ΔNLL −8.0 |
| Weibull (R58) | raw support costs | the legacy fit never gets past round 1 (no step lowers the loss); fixed: CRPS −5 125 (median, −28 %) 10/10, NLL −0.29 (median) 9/10, one seed's test NLL 17.5 |
| Weibull (R58) | kin8nm (10 seeds) | legacy train NLL ∞ on 4/10 and test NLL ≥ 10¹² on 3/10; fixed CRPS better 10/10 (median −0.012), but NLL only even (median −0.02, 5/10): on the 7 seeds where legacy did not blow up, fixed NLL is worse on 4 (up to 1.56 vs −0.33), and 90 % coverage drops 0.88 → 0.83 |
| Weibull (censored) | parity seed 17 | legacy diverges, fixed converges unregularised |

The size of the effect tracks the units: the further σ (or λ) is from 1,
the larger the legacy error. The legacy Weibull metric was also
13–26× slower per fit (a quadrature over Y per row); Gamma 0.8–4×. All arms have 10 seeds.

**CRPS vs LogScore training (R58,** both fixed, 10 seeds each):
- A CRPS-trained Gamma is close to the LogScore Gamma: worse NLL on
  lognorm_het (+0.068, 1/10), even on raw costs and kin8nm (+0.005, +0.007).
  Its CRPS is −67 on raw costs and otherwise within noise.
- A CRPS-trained Weibull wins CRPS on every seed (30/30; median −0.16
  lognorm_het, −162 raw costs, −0.005 kin8nm). It loses NLL (median
  +0.07…+0.10, 5/30 wins) and is overconfident: 90 % coverage 0.83–0.89 vs
  0.84–0.93. As with Normal: train
on CRPS only when CRPS is the target metric.
