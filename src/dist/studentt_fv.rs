//! Student's t with a lower bound on the degrees of freedom (since 1.3.0;
//! research Cycle 18, `research/optimization-algorithm/proposals/
//! O3_studentt_nu_floor.md`).
//!
//! [`StudentT`] boosts θ₂ = ln ν, so ν can drift to any positive value: on
//! heavy-tailed data the fitted model routinely predicts rows with ν ≤ 2
//! (infinite variance, `std()` = ∞) and ν ≤ 1 (no mean, CRPS undefined) —
//! 60 and 22 of 80 staged fits in the benchmark had at least one such test
//! row. [`StudentTFiniteVar`] is the same family restricted to
//! ν > [`StudentTFiniteVar::NU_MIN`] = 2.1 through the smooth
//! reparameterization
//!
//!   ν = ν_min + exp(θ₂),      θ₂ = ln(ν − ν_min).
//!
//! It is not a clamp: no row is ever frozen at the bound. By the chain rule
//! every ∂/∂θ₂ is c·∂/∂ln ν with c = (ν − ν_min)/ν ∈ (0, 1), so the score
//! gradient's third entry is scaled by c and the metric's third row and
//! column by c (c² on the diagonal); both scores reuse [`StudentT`]'s
//! closed forms. The bound sits above 2 because a bound of exactly 2 is
//! attained in floating point (exp(θ₂) underflows against it) and the
//! variance ν/(ν − 2)·σ² is then still infinite.
//!
//! Measured against [`StudentT`] (staged fit, 8 datasets × 10 seeds): rows
//! with an undefined mean or variance went from 22 / 60 of 80 fits to 0 / 0,
//! with held-out NLL unchanged on average (−0.001; worst dataset, protein,
//! +0.019). These fits are sensitive to last-bit rounding: an arithmetically
//! equivalent variant of this code gave +0.010 on average and +0.065 on
//! protein over the same 80 fits, again with no undefined row — read the
//! NLL cost as "between 0 and 0.01 on average, a few hundredths on a
//! heavy-tailed target".
//! The bound is part of the type, so a saved model cannot be loaded with a
//! different one.

use crate::dist::studentt::DF_MAX;
use crate::dist::{Distribution, DistributionMethods, RegressionDistn, StudentT};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3};

/// Student's t with ν > 2.1: internal parameters (loc, log σ, ln(ν − 2.1)).
/// See the [module documentation](self).
#[derive(Debug, Clone)]
pub struct StudentTFiniteVar {
    /// The same distribution as a [`StudentT`] (ν already includes the bound).
    inner: StudentT,
    /// θ₂ = ln(ν − ν_min), kept so `params()` is exact near the bound.
    log_excess: Array1<f64>,
    /// c = (ν − ν_min)/ν = ∂ ln ν / ∂θ₂ per row.
    chain: Array1<f64>,
}

impl StudentTFiniteVar {
    /// Lower bound on ν (exclusive): every prediction has a finite mean and
    /// variance (variance ≤ 21·σ²).
    pub const NU_MIN: f64 = 2.1;

    /// The equivalent unconstrained-parameterization [`StudentT`].
    pub fn as_student_t(&self) -> &StudentT {
        &self.inner
    }

    /// Location per row.
    pub fn loc(&self) -> &Array1<f64> {
        &self.inner.loc
    }

    /// Scale σ per row (not the standard deviation, which is
    /// σ·√(ν/(ν − 2))).
    pub fn scale(&self) -> &Array1<f64> {
        &self.inner.scale
    }

    /// Degrees of freedom ν per row (> [`Self::NU_MIN`]).
    pub fn df(&self) -> &Array1<f64> {
        &self.inner.df
    }
}

impl Distribution for StudentTFiniteVar {
    fn location_column() -> Option<usize> {
        Some(0)
    }

    /// Staged by default, as [`StudentT`].
    fn default_staged_fit() -> bool {
        true
    }

    fn scale_param_index() -> Option<usize> {
        Some(1)
    }

    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = crate::vmath::exp_column(&params.column(1));
        let var = &scale * &scale;
        let log_excess = params.column(2).to_owned();
        // `>` (not f64::min) so a NaN θ₂ stays NaN and is still caught.
        let df = crate::vmath::exp_column(&params.column(2)).mapv_into(|e| {
            let v = Self::NU_MIN + e;
            if v > DF_MAX { DF_MAX } else { v }
        });
        // c = (ν − ν_min)/ν from the (capped) ν, exactly as in the measured
        // prototype (research Cycle 18, O3). The subtraction loses relative
        // accuracy only where exp(θ₂) < 1e-10, i.e. where c — and with it
        // the whole third coordinate — is already negligible; the absolute
        // error is below 1e-16. At the DF_MAX cap StudentT's third gradient
        // entry is exactly 0 and c ≈ 1.
        let chain = df.mapv(|v| (v - Self::NU_MIN) / v);
        StudentTFiniteVar {
            inner: StudentT {
                loc,
                scale,
                var,
                df,
            },
            log_excess,
            chain,
        }
    }

    /// [`StudentT::fit`] (location and scale by t₃ maximum likelihood, ν = 3)
    /// re-expressed: θ₂ = ln(3 − ν_min).
    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mut p = StudentT::fit(y);
        // StudentT::fit always starts at ν = 3 (its third entry is ln 3)
        p[2] = (3.0_f64 - Self::NU_MIN).ln();
        p
    }

    fn n_params(&self) -> usize {
        3
    }

    fn predict(&self) -> Array1<f64> {
        self.inner.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let mut p = self.inner.params();
        p.column_mut(2).assign(&self.log_excess);
        p
    }
}

impl RegressionDistn for StudentTFiniteVar {}

impl DistributionMethods for StudentTFiniteVar {
    fn mean(&self) -> Array1<f64> {
        self.inner.mean()
    }
    fn variance(&self) -> Array1<f64> {
        self.inner.variance()
    }
    fn std(&self) -> Array1<f64> {
        self.inner.std()
    }
    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.inner.pdf(y)
    }
    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.inner.logpdf(y)
    }
    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.inner.cdf(y)
    }
    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        self.inner.ppf(q)
    }
    fn sample(&self, n_samples: usize) -> Array2<f64> {
        DistributionMethods::sample(&self.inner, n_samples)
    }
    fn interval(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        self.inner.interval(alpha)
    }
    fn sf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.inner.sf(y)
    }
    fn median(&self) -> Array1<f64> {
        self.inner.median()
    }
    fn mode(&self) -> Array1<f64> {
        self.inner.mode()
    }
}

/// g_new = Jᵀ g with J = diag(1, 1, c).
fn chain_gradient(mut g: Array2<f64>, chain: &Array1<f64>) -> Array2<f64> {
    g.column_mut(2).zip_mut_with(chain, |v, &c| *v *= c);
    g
}

/// M_new = Jᵀ M J with J = diag(1, 1, c).
fn chain_metric(mut m: Array3<f64>, chain: &Array1<f64>) -> Array3<f64> {
    for (i, &c) in chain.iter().enumerate() {
        for j in 0..2 {
            m[[i, j, 2]] *= c;
            m[[i, 2, j]] *= c;
        }
        m[[i, 2, 2]] = m[[i, 2, 2]] * c * c;
    }
    m
}

impl Scorable<LogScore> for StudentTFiniteVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Scorable::<LogScore>::score(&self.inner, y)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        chain_gradient(Scorable::<LogScore>::d_score(&self.inner, y), &self.chain)
    }

    fn metric(&self) -> Array3<f64> {
        chain_metric(Scorable::<LogScore>::metric(&self.inner), &self.chain)
    }
}

impl Scorable<CRPScore> for StudentTFiniteVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Scorable::<CRPScore>::score(&self.inner, y)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        chain_gradient(Scorable::<CRPScore>::d_score(&self.inner, y), &self.chain)
    }

    fn metric(&self) -> Array3<f64> {
        chain_metric(Scorable::<CRPScore>::metric(&self.inner), &self.chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const FLOOR: f64 = StudentTFiniteVar::NU_MIN;

    fn dist(loc: f64, log_scale: f64, theta: f64) -> StudentTFiniteVar {
        StudentTFiniteVar::from_params(
            &Array2::from_shape_vec((1, 3), vec![loc, log_scale, theta]).unwrap(),
        )
    }

    #[test]
    fn nu_stays_above_the_bound_and_params_invert() {
        for theta in [-800.0, -40.0, -3.0, 0.0, 2.5, 12.0] {
            let d = dist(0.4, -0.2, theta);
            assert!(d.df()[0] >= FLOOR, "theta {theta}: nu {}", d.df()[0]);
            assert!(d.variance()[0].is_finite(), "theta {theta}");
            assert!(d.mean()[0].is_finite(), "theta {theta}");
            assert_eq!(d.params()[[0, 2]], theta);
        }
        // a strictly positive excess is strictly above the bound
        assert!(dist(0.0, 0.0, -3.0).df()[0] > FLOOR);
        // the DF_MAX cap still applies and freezes the third coordinate
        let d = dist(0.0, 0.0, 800.0);
        assert_eq!(d.df()[0], DF_MAX);
        let g = Scorable::<LogScore>::d_score(&d, &array![0.3]);
        assert_eq!(g[[0, 2]], 0.0);
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn marginal_fit_starts_at_nu_three() {
        let y: Array1<f64> = (0..200)
            .map(|i| ((i * 37) % 101) as f64 / 10.0 - 5.0)
            .collect();
        let (p, q) = (StudentTFiniteVar::fit(&y), StudentT::fit(&y));
        assert_eq!(p[0], q[0]);
        assert_eq!(p[1], q[1]);
        let d = StudentTFiniteVar::from_params(&p.insert_axis(ndarray::Axis(0)));
        assert!((d.df()[0] - 3.0).abs() < 1e-12, "{}", d.df()[0]);
    }

    #[test]
    fn same_density_as_student_t() {
        let (theta, y) = (0.3_f64, array![1.7]);
        let d = dist(0.4, -0.2, theta);
        let t = StudentT::from_params(
            &Array2::from_shape_vec((1, 3), vec![0.4, -0.2, (FLOOR + theta.exp()).ln()]).unwrap(),
        );
        for (a, b) in [
            (
                Scorable::<LogScore>::score(&d, &y)[0],
                Scorable::<LogScore>::score(&t, &y)[0],
            ),
            (
                Scorable::<CRPScore>::score(&d, &y)[0],
                Scorable::<CRPScore>::score(&t, &y)[0],
            ),
            (d.cdf(&y)[0], t.cdf(&y)[0]),
        ] {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        let h = 1e-6;
        for &(loc, ls, theta, yv) in &[
            (0.0, 0.0, 0.0, 0.8),
            (1.0, -0.5, -3.0, 2.5),
            (-2.0, 0.7, 2.0, -6.0),
            (0.3, 0.1, -8.0, 0.2),
        ] {
            let y = array![yv];
            let d = dist(loc, ls, theta);
            let log = Scorable::<LogScore>::d_score(&d, &y);
            let crps = Scorable::<CRPScore>::d_score(&d, &y);
            for j in 0..3 {
                let at = |e: f64| {
                    let mut p = [loc, ls, theta];
                    p[j] += e;
                    dist(p[0], p[1], p[2])
                };
                let fd_log = (Scorable::<LogScore>::score(&at(h), &y)[0]
                    - Scorable::<LogScore>::score(&at(-h), &y)[0])
                    / (2.0 * h);
                let fd_crps = (Scorable::<CRPScore>::score(&at(h), &y)[0]
                    - Scorable::<CRPScore>::score(&at(-h), &y)[0])
                    / (2.0 * h);
                assert!(
                    (log[[0, j]] - fd_log).abs() < 1e-5 * (1.0 + fd_log.abs()),
                    "LogScore ({loc},{ls},{theta}) col {j}: {} vs {fd_log}",
                    log[[0, j]]
                );
                assert!(
                    (crps[[0, j]] - fd_crps).abs() < 1e-4 * (1.0 + fd_crps.abs()),
                    "CRPS ({loc},{ls},{theta}) col {j}: {} vs {fd_crps}",
                    crps[[0, j]]
                );
            }
        }
    }

    /// Fisher information = E[g gᵀ], by deterministic midpoint quadrature
    /// over the probability integral transform.
    #[test]
    fn log_metric_is_expected_outer_product() {
        let n = 400_000;
        let u: Array1<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        for &(ls, theta) in &[(0.0, 0.0), (-0.4, -2.0), (0.5, 1.5)] {
            let mut p = Array2::zeros((n, 3));
            p.column_mut(1).fill(ls);
            p.column_mut(2).fill(theta);
            let d = StudentTFiniteVar::from_params(&p);
            let y = d.ppf(&u);
            let g = Scorable::<LogScore>::d_score(&d, &y);
            let m = Scorable::<LogScore>::metric(&dist(0.0, ls, theta));
            for a in 0..3 {
                for b in a..3 {
                    let e = (0..n).map(|i| g[[i, a]] * g[[i, b]]).sum::<f64>() / n as f64;
                    let scale = (m[[0, a, a]] * m[[0, b, b]]).sqrt();
                    assert!(
                        (e - m[[0, a, b]]).abs() < 5e-3 * scale,
                        "I[{a},{b}] at ({ls},{theta}): {e} vs {}",
                        m[[0, a, b]]
                    );
                    assert_eq!(m[[0, a, b]], m[[0, b, a]]);
                }
            }
        }
    }

    /// The natural gradient is coordinate-free: mapped back through
    /// J = diag(1, 1, c) it is StudentT's own natural gradient.
    #[test]
    fn natural_gradient_is_invariant() {
        let (theta, y) = (-1.0_f64, array![2.2]);
        let d = dist(0.5, 0.2, theta);
        let nu = FLOOR + theta.exp();
        let t = StudentT::from_params(
            &Array2::from_shape_vec((1, 3), vec![0.5, 0.2, nu.ln()]).unwrap(),
        );
        let c = theta.exp() / nu;
        let (a, b) = (
            Scorable::<LogScore>::grad(&d, &y, true),
            Scorable::<LogScore>::grad(&t, &y, true),
        );
        for j in 0..3 {
            let back = if j == 2 { a[[0, j]] * c } else { a[[0, j]] };
            assert!(
                (back - b[[0, j]]).abs() < 1e-9 * (1.0 + b[[0, j]].abs()),
                "col {j}: {back} vs {}",
                b[[0, j]]
            );
        }
    }
}
