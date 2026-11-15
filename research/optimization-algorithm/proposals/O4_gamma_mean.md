# O4 — Gamma in Fisher-orthogonal (log mean, log shape) coordinates

Status: research head `GammaMean` (`src/dist/gamma_mean.rs`, additive, no
existing head touched). Results: `../research_log.md`, Cycle 18.

## Theoretical question

`Gamma` boosts θ = (log α, log β) (shape, rate). Its Fisher matrix is not
diagonal, which is why 1.2.0's Fisher-weighted learners, staged fit and
end-of-fit recalibration were never applied to it (known limitations 6, 8).
Is the coordinate system itself costing accuracy?

## Baseline mathematics

ℓ = α ln β + (α − 1) ln y − β y − ln Γ(α).
I(θ) = [[α² ψ′(α), −α], [−α, α]]. Mean μ = α/β.

## Candidate mathematics

φ = (log μ, log α) = Aθ with A = [[1, −1], [1, 0]] (a *linear* map).

  ℓ = α ln α − α ln μ + (α − 1) ln y − α y/μ − ln Γ(α)
  ∂ℓ/∂log μ = α (y/μ − 1)
  ∂ℓ/∂log α = α [ln α + 1 − ln μ + ln y − y/μ − ψ(α)]
  I(φ) = diag(α, α² ψ′(α) − α)

E[y/μ − 1] = 0 and E[ln y] = ψ(α) − ln(α/μ) give the zero cross term.
α² ψ′(α) − α > 0 for all α > 0 (ψ′(α) > 1/α); it tends to ½ as α → ∞
(series ½ + 1/(6α) used above α = 10³, where the exact form cancels) and to
1 as α → 0. Both entries are positive, so I(φ) is positive definite and the
"solve" is two divisions.

Because A is linear, the natural-gradient vector field is the same in both
coordinate systems: v_φ = A v_θ, i.e.

  v_θ[log α] = v_φ[log α]                 (shape step)
  v_θ[log β] = v_φ[log α] − v_φ[log μ]    (shape step − mean step)

So the two heads would be identical if base learners were linear operators.
They are not. In the baseline, the log β tree must fit the *sum* of the
mean signal and the shape signal, and the mean is recovered as the
difference of two independently approximated trees:
Δlog μ = tree₁(shape step) − tree₂(shape step − mean step). The shape
step, α[…]/(α²ψ′(α) − α), is a heavy-tailed function of ln y; whatever part
of it the two trees approximate differently is injected into the mean. In
the candidate the mean tree sees only −(y/μ − 1).

Natural gradients (what the trees fit), candidate:
  log μ:  1 − y/μ               (relative residual; weight-α Newton leaf
                                 under Fisher weighting, β_F = 1)
  log α:  −[ln α + 1 + ln(y/μ) − y/μ − ψ(α)] / (α ψ′(α) − 1)

Complexity: unchanged per round. Domain: y > 0; μ, α > 0 by the log links.

## Verification

Unit tests in `src/dist/gamma_mean.rs`: same density as `Gamma` at mapped
parameters; gradient = centred finite differences; metric = E[g gᵀ] by
midpoint quadrature over the probability integral transform (0.5 %), with
a zero cross term; large-α series joins the exact form;
marginal fit equals `Gamma::fit` re-expressed.

## Predictions (falsifiable)

P1. Same family, so on homoscedastic-shape data with a non-trivial mean
    the candidate has lower test RMSE and NLL (cleaner mean), with the gap
    growing when the shape signal is noisy (small α).
P2. If instead the two heads tie everywhere, the non-diagonal coordinates
    cost nothing and the head is only worth having for the features it
    enables (P3).
P3. With the diagonal metric the NegativeBinomial playbook applies: the
    log α recalibration (≥ 300 validation rows) should not lose; Fisher
    weighting (β_F = 1) matters only where α varies across rows (weights
    are α, constant otherwise); a staged fit is expected to need the
    recalibration, as for NB (limitation 51).

## Failure modes

- A shared scalar line search couples the two coordinates in both heads;
  if the baseline's coupling happens to act as useful damping of the mean,
  the candidate could overfit the mean sooner.
- Real "Gamma-like" targets are usually misspecified; gains on true-Gamma
  synthetic data must be confirmed on real positive targets before any
  recommendation.

## Library change (approved by the user and implemented 2026-11-13)

`GammaMean` is a documented head (README table, CHANGELOG, module docs):

- `default_staged_fit() = true`: under LogScore with early stopping and at
  least 300 validation rows (its `auto_calibration_min_rows`) the default
  fit is the `auto+staged` variant measured above. Below 300 validation
  rows the default is the joint fit (the `base` variant).
- `Scorable<CRPScore>`: the score is `Gamma`'s; gradient and metric by the
  chain rule with J = ∂(log α, log β)/∂(log μ, log α) = [[0, 1], [−1, 1]],
  g' = Jᵀ g, M' = Jᵀ M J = [[d, −(b+d)], [−(b+d), a+2b+d]] for
  M = [[a, b], [b, d]]. Unit test `crps_is_gamma_crps_in_new_coordinates`:
  equal score, finite-difference gradient, positive-definite metric, and
  the natural gradient mapped back through J equals `Gamma`'s. CRPS
  training is tested for correctness only (it trains and lowers CRPS
  against the marginal fit), not benchmarked.
- Integration test `gamma_mean_fits_staged_by_default_and_matches_or_beats_gamma`.

`Gamma` is unchanged; its single-seed mean blow-ups still have no
regression test.
