# P1 — Fisher-power weighted base learners (β-interpolation)

**Setting.** Normal(μ, log σ), LogScore. Natural gradient per row:
ñ_μ = μ − y, ñ_{log σ} = ½(1 − z²), z = (y − μ)/σ. Base learner j is fitted to
column ñ_j by weighted least squares with weights w_ij.

**Current paths.**
- Upstream (default): w_ij = 1 → leaf = mean(ñ_j); split by unweighted
  variance reduction. For μ this is L2 boosting on residuals.
- `fisher_weighted_learner` (opt-in): w_ij = F_jj = (1/σ_i², 2) → leaf for μ
  is Σ(err/σ²)/Σ(1/σ²) (GLS / Fisher scoring), log σ unchanged (constant F).

**Proposal.** w_ij = (F_jj + λ)^β, β ∈ [0, 1] (`fisher_weight_power`). β = 0
reproduces upstream, β = 1 reproduces the boolean opt-in bit-for-bit.
For Normal: μ-tree weights σ_i^{−2β}; log σ-tree weights 2^β (a constant →
no effect). Correspondence to β-NLL (Seitzer et al. 2022): β_S = 1 − β.

**Assumptions / failure cases.** Weighting is efficient only when σ̂ is
informative; early in training σ̂ ≈ marginal σ → weights ≈ constant, so the
effect appears as σ̂ differentiates. With early stopping, β → 1 risks
under-fitting the mean in high-noise regions (the Seitzer "vicious cycle").
Weights also change the split criterion, not only leaf values.

**Falsifiable predictions (held-out NLL, ≥5 seeds, paired vs β=0).**
1. het_a / het_b (strong heteroscedasticity): β = 0.5 < β = 0 (lower NLL).
2. homo: |Δ| within noise (σ̂ nearly constant).
3. tnoise (misspecified heavy tails): β > 0 ≤ β = 0 (down-weights the
   leverage of large-σ̂ rows' residuals).
4. kin8nm / protein: direction unknown; report.
If prediction 1 fails on both DGPs, reject P1.

**Code path.** `src/ngboost.rs`: field `fisher_weight_power: Option<f64>`;
`fit_internal` weight construction. No change to defaults, prediction,
serialization.
