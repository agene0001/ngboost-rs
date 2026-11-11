# P3 — Poisson head: the unweighted natural-gradient leaf is a biased rate estimator

**Setting.** Poisson(λ), θ = log λ. S = λ − y log λ + ln Γ(y+1); ∇_θ S = λ − y;
Fisher F = λ; natural gradient g̃ = (λ − y)/λ = 1 − y/λ.

**Upstream NGBoost leaf (unweighted LS on g̃).**
    c_LS = mean_{i∈L}(1 − y_i/λ_i) = 1 − (1/n_L) Σ y_i/λ_i      (mean of ratios)
**Fisher-weighted leaf (`fisher_weight_power = 1`, weights F_i = λ_i).**
    c_F  = Σ(λ_i − y_i)/Σ λ_i = 1 − Σ y_i / Σ λ_i                 (ratio of sums)
Both coincide when λ is constant inside the leaf. When it is not, c_LS
weights each row's relative error y_i/λ_i by 1/λ_i, so low-rate rows pull
the leaf ~λ_high/λ_low times harder than high-rate rows. High-rate rows —
which carry most of Σy and most of the squared error — are under-corrected.
Prediction: **Σ y / Σ λ̂ > 1 out of sample** (totals under-predicted), worse
RMSE, and the under-fit shows up as apparent over-dispersion
(mean((y−λ̂)²/λ̂) > 1) even on a correctly specified Poisson DGP.

**Equivalence.** Weighted LS of g̃ = g/F with weights F minimises
Σ F_i (g_i/F_i − c)² = Σ (g_i − c F_i)²/F_i, whose optimum is c = Σg/ΣF and
whose impurity reduction for a split is (Σ_L g)²/Σ_L F + (Σ_R g)²/Σ_R F −
(Σg)²/ΣF. For the Poisson objective g = λ − y and the Hessian h = λ = F, so
β = 1 is **exactly** the second-order tree of XGBoost/LightGBM (`count:poisson`):
leaf G/H, gain G²/H (Chen & Guestrin, KDD 2016, eqs. 5–7). Upstream NGBoost
instead takes mean(g/h) with unweighted variance-reduction splits. β ∈ (0, 1)
interpolates; β = 0.5 uses weights √λ.

**Falsifiable predictions** (R12–R14; 5 count DGPs, held-out test):
1. Base: Σy/Σλ̂ > 1 by several %, dispersion index > 1 on `pois_*`; oracle
   (true λ) ≈ 1.00 on both (sampler sanity check).
2. β = 1 reduces the bias ratio toward 1, lowers NLL/CRPS/RMSE on all `pois_*`
   DGPs across seeds and learning rates; β = 0.5 in between.
3. On over-dispersed DGPs (`nb_*`) the Poisson head's dispersion index stays
   ≫ 1 for every β (misspecification cannot be fixed by weighting) — the
   remedy there is a Negative-Binomial head, not β.
4. A post-hoc common shift of log λ fitted on validation (closed form:
   λ̂ × Σy_val/Σλ̂_val; or the affine `calibrate_param_affine(col = 0)`)
   recovers part of the base model's NLL gap but not the RMSE gap (it cannot
   re-rank rows).

**Decision rule.** If 1–2 hold, recommend `fisher_weight_power = Some(1.0)`
for Poisson heads (documented; downstream projects confirm on their counts,
cf. the earlier NBA/MLB measurement in commit aa7b4eb) and consider a
distribution-level default. If 3 holds, open a NegBin-head proposal.
