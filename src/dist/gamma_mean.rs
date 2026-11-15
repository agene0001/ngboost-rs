//! The Gamma distribution in the Fisher-orthogonal (log mean, log shape)
//! parameterization (since 1.3.0; research Cycle 18,
//! `research/optimization-algorithm/proposals/O4_gamma_mean.md`).
//!
//! [`Gamma`] boosts (log α, log β) with rate β. Its Fisher matrix is not
//! diagonal, and the mean μ = α/β is moved by *both* trees: the natural
//! gradient step of log β is (shape step − mean step), so the mean tree has
//! to fit a sum of the mean signal and the heavy-tailed shape signal, and
//! the mean is the difference of two separately approximated trees.
//!
//! Here the internal parameters are θ = (log μ, log α), with
//!   ln f(y) = α ln α − α ln μ + (α − 1) ln y − α y/μ − ln Γ(α),
//!   ∂ℓ/∂log μ = α (y/μ − 1),
//!   ∂ℓ/∂log α = α [ln α + 1 − ln μ + ln y − y/μ − ψ(α)],
//!   Fisher     = diag(α, α² ψ′(α) − α)          (the cross term has mean 0).
//! Same family, same likelihood; only the coordinates the trees work in
//! differ. With a diagonal Fisher matrix the head also gets the staged fit
//! (log μ is the location; the default under `LogScore` when the
//! recalibration runs, i.e. from 300 validation rows) and the end-of-fit
//! recalibration (of log α, as NegativeBinomial does for log r).
//!
//! Measured against [`Gamma`] on six datasets × 10 seeds (true-Gamma
//! synthetic, LogNormal-noise synthetic, kin8nm, a real cost target):
//! held-out NLL −0.03…−0.32 with the staged fit and recalibration (10/10
//! seeds on every dataset), −0.01…−0.23 on five of six as a plain joint
//! fit (+0.014 on the sixth). `CRPScore` is supported through the chain
//! rule on [`Gamma`]'s CRPS (same score values; joint fit only).

use crate::dist::gamma::trigamma;
use crate::dist::{Distribution, DistributionMethods, Gamma, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use statrs::function::gamma::{digamma, ln_gamma};

/// Gamma with internal parameters (log mean, log shape).
#[derive(Debug, Clone)]
pub struct GammaMean {
    /// Mean μ.
    pub mu: Array1<f64>,
    /// Shape α (variance = μ²/α).
    pub shape: Array1<f64>,
    /// The same distribution as a (shape, rate) [`Gamma`], for the
    /// distribution functions.
    inner: Gamma,
}

impl GammaMean {
    /// The equivalent (shape, rate) Gamma.
    pub fn as_gamma(&self) -> &Gamma {
        &self.inner
    }
}

/// Fisher information of log α: α² ψ′(α) − α. The two terms cancel to
/// ½ + 1/(6α) − 1/(30α³) + … for large α, so switch to the series there.
fn shape_fisher(a: f64) -> f64 {
    if a > 1e3 {
        0.5 + 1.0 / (6.0 * a)
    } else {
        a * a * trigamma(a) - a
    }
}

impl Distribution for GammaMean {
    fn from_params(params: &Array2<f64>) -> Self {
        let mu = crate::vmath::exp_column(&params.column(0));
        let shape = crate::vmath::exp_column(&params.column(1));
        let inner = Gamma {
            shape: shape.clone(),
            rate: &shape / &mu,
        };
        GammaMean { mu, shape, inner }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // the marginal Gamma MLE, re-expressed: log μ = log α − log β
        let p = Gamma::fit(y);
        array![p[0] - p[1], p[0]]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn validate_targets(y: &Array1<f64>) -> Result<(), &'static str> {
        Gamma::validate_targets(y)
    }

    /// log μ is the location for the staged fit.
    fn location_column() -> Option<usize> {
        Some(0)
    }

    /// Staged fit by default (LogScore, early stopping, and at least
    /// [`Self::auto_calibration_min_rows`] validation rows so the log α
    /// recalibration runs): against the joint fit of this head it lowered
    /// held-out NLL by 0.02–0.15 on all six benchmark datasets (research
    /// Cycle 18, O4). `staged_fit = Some(false)` gives the joint fit.
    fn default_staged_fit() -> bool {
        true
    }

    /// log α is recalibrated by the end-of-fit validation map: an affine
    /// map on it changes the spread and leaves the mean alone.
    fn scale_param_index() -> Option<usize> {
        Some(1)
    }

    /// A shape, like NegativeBinomial's dispersion, is harder to pin down
    /// than a Normal σ; keep that head's threshold until measured.
    fn auto_calibration_min_rows() -> usize {
        300
    }

    fn predict(&self) -> Array1<f64> {
        self.mu.clone()
    }

    fn params(&self) -> Array2<f64> {
        let mut p = Array2::zeros((self.mu.len(), 2));
        p.column_mut(0).assign(&self.mu.mapv(f64::ln));
        p.column_mut(1).assign(&self.shape.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for GammaMean {}

impl DistributionMethods for GammaMean {
    fn mean(&self) -> Array1<f64> {
        self.mu.clone()
    }
    fn variance(&self) -> Array1<f64> {
        &self.mu * &self.mu / &self.shape
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
    fn median(&self) -> Array1<f64> {
        self.inner.median()
    }
    fn mode(&self) -> Array1<f64> {
        self.inner.mode()
    }
}

impl Scorable<LogScore> for GammaMean {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let mut d = Array2::zeros((self.mu.len(), 2));
        Zip::from(d.rows_mut())
            .and(&self.shape)
            .for_each(|mut row, &a| {
                row[0] = a;
                row[1] = shape_fisher(a);
            });
        d
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Zip::from(y)
            .and(&self.mu)
            .and(&self.shape)
            .map_collect(|&yi, &m, &a| {
                -(a * (a / m).ln() + (a - 1.0) * yi.ln() - a * yi / m - ln_gamma(a))
            })
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let mut d = Array2::zeros((y.len(), 2));
        Zip::from(d.rows_mut())
            .and(y)
            .and(&self.mu)
            .and(&self.shape)
            .for_each(|mut row, &yi, &m, &a| {
                let r = yi / m;
                row[0] = -a * (r - 1.0);
                row[1] = -a * (a.ln() + 1.0 + r.ln() - r - digamma(a));
            });
        d
    }

    fn metric(&self) -> Array3<f64> {
        let n = self.mu.len();
        let mut fi = Array3::zeros((n, 2, 2));
        for i in 0..n {
            fi[[i, 0, 0]] = self.shape[i];
            fi[[i, 1, 1]] = shape_fisher(self.shape[i]);
        }
        fi
    }
}

/// CRPS in (log μ, log α) coordinates by the chain rule on [`Gamma`]'s
/// (log α, log β) implementation: log β = log α − log μ, so with
/// J = ∂(log α, log β)/∂(log μ, log α) = [[0, 1], [−1, 1]] the gradient is
/// Jᵀ g and the metric Jᵀ M J. The metric is not diagonal under CRPS.
impl Scorable<CRPScore> for GammaMean {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Scorable::<CRPScore>::score(&self.inner, y)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let g = Scorable::<CRPScore>::d_score(&self.inner, y);
        let mut d = Array2::zeros((y.len(), 2));
        Zip::from(d.rows_mut())
            .and(g.rows())
            .for_each(|mut row, gi| {
                row[0] = -gi[1];
                row[1] = gi[0] + gi[1];
            });
        d
    }

    fn metric(&self) -> Array3<f64> {
        let m = Scorable::<CRPScore>::metric(&self.inner);
        let n = self.mu.len();
        let mut fi = Array3::zeros((n, 2, 2));
        for i in 0..n {
            let (a, b, d) = (m[[i, 0, 0]], m[[i, 0, 1]], m[[i, 1, 1]]);
            fi[[i, 0, 0]] = d;
            fi[[i, 0, 1]] = -(b + d);
            fi[[i, 1, 0]] = -(b + d);
            fi[[i, 1, 1]] = a + 2.0 * b + d;
        }
        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crps_is_gamma_crps_in_new_coordinates() {
        let h = 1e-5;
        for &(lm, la, yv) in &[(0.7, 1.3, 2.0), (-1.0, -0.5, 0.2), (2.0, 3.0, 9.0)] {
            let y = array![yv];
            let d = dist(lm, la);
            let g = Gamma::from_params(&Array2::from_shape_vec((1, 2), vec![la, la - lm]).unwrap());
            let f = |a: f64, b: f64| Scorable::<CRPScore>::score(&dist(a, b), &y)[0];
            // same score as the (shape, rate) head
            assert!((f(lm, la) - Scorable::<CRPScore>::score(&g, &y)[0]).abs() < 1e-12);
            // gradient against finite differences in (log μ, log α)
            let gr = Scorable::<CRPScore>::d_score(&d, &y);
            let fd0 = (f(lm + h, la) - f(lm - h, la)) / (2.0 * h);
            let fd1 = (f(lm, la + h) - f(lm, la - h)) / (2.0 * h);
            assert!(
                (gr[[0, 0]] - fd0).abs() < 1e-4 * (1.0 + fd0.abs()),
                "{lm} {la}: {} {fd0}",
                gr[[0, 0]]
            );
            assert!(
                (gr[[0, 1]] - fd1).abs() < 1e-4 * (1.0 + fd1.abs()),
                "{lm} {la}: {} {fd1}",
                gr[[0, 1]]
            );
            // metric: Jᵀ M J, symmetric positive definite; the natural
            // gradient is invariant: J·(M_new⁻¹ g_new) = M_old⁻¹ g_old
            let m = Scorable::<CRPScore>::metric(&d);
            let (a, b, c) = (m[[0, 0, 0]], m[[0, 0, 1]], m[[0, 1, 1]]);
            assert_eq!(b, m[[0, 1, 0]]);
            let det = a * c - b * b;
            assert!(a > 0.0 && det > 0.0, "not PD: {a} {b} {c}");
            let n_new = [
                (c * gr[[0, 0]] - b * gr[[0, 1]]) / det,
                (a * gr[[0, 1]] - b * gr[[0, 0]]) / det,
            ];
            let n_old = Scorable::<CRPScore>::grad(&g, &y, true);
            // old coordinates (log α, log β): δlog α = n_new[1], δlog β = n_new[1] − n_new[0]
            assert!(
                (n_new[1] - n_old[[0, 0]]).abs() < 1e-6 * (1.0 + n_old[[0, 0]].abs()),
                "{} vs {}",
                n_new[1],
                n_old[[0, 0]]
            );
            assert!(
                (n_new[1] - n_new[0] - n_old[[0, 1]]).abs() < 1e-6 * (1.0 + n_old[[0, 1]].abs()),
                "{} vs {}",
                n_new[1] - n_new[0],
                n_old[[0, 1]]
            );
        }
    }

    fn dist(lm: f64, la: f64) -> GammaMean {
        GammaMean::from_params(&Array2::from_shape_vec((1, 2), vec![lm, la]).unwrap())
    }

    #[test]
    fn same_density_as_gamma() {
        // (log μ, log α) ↔ (log α, log β = log α − log μ)
        let (lm, la) = (0.7, 1.3);
        let g = Gamma::from_params(&Array2::from_shape_vec((1, 2), vec![la, la - lm]).unwrap());
        let d = dist(lm, la);
        for yv in [0.05, 0.9, 2.0, 11.0] {
            let y = array![yv];
            let a = Scorable::<LogScore>::score(&d, &y)[0];
            let b = Scorable::<LogScore>::score(&g, &y)[0];
            assert!((a - b).abs() < 1e-10, "y {yv}: {a} vs {b}");
        }
        assert!((d.mean()[0] - g.mean()[0]).abs() < 1e-12);
        assert!((d.variance()[0] - g.variance()[0]).abs() < 1e-10);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let h = 1e-6;
        for &(lm, la, yv) in &[
            (0.7, 1.3, 2.0),
            (-1.0, -0.5, 0.2),
            (2.0, 4.0, 9.0),
            (0.0, 8.0, 1.01),
        ] {
            let y = array![yv];
            let g = Scorable::<LogScore>::d_score(&dist(lm, la), &y);
            let f = |a: f64, b: f64| Scorable::<LogScore>::score(&dist(a, b), &y)[0];
            let fd0 = (f(lm + h, la) - f(lm - h, la)) / (2.0 * h);
            let fd1 = (f(lm, la + h) - f(lm, la - h)) / (2.0 * h);
            assert!(
                (g[[0, 0]] - fd0).abs() < 1e-5 * (1.0 + fd0.abs()),
                "{lm} {la}: {} {fd0}",
                g[[0, 0]]
            );
            assert!(
                (g[[0, 1]] - fd1).abs() < 1e-5 * (1.0 + fd1.abs()),
                "{lm} {la}: {} {fd1}",
                g[[0, 1]]
            );
        }
    }

    #[test]
    fn metric_is_expected_outer_product_and_diagonal() {
        // E[g gᵀ] by deterministic midpoint quadrature over the probability
        // integral transform (y = F⁻¹(u) on a uniform grid of u)
        let n = 400_000;
        let u: Array1<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        for &(lm, la) in &[(0.3, 0.2), (1.0, 1.5), (-0.5, 3.0)] {
            let mut p = Array2::zeros((n, 2));
            p.column_mut(0).fill(lm);
            p.column_mut(1).fill(la);
            let d = GammaMean::from_params(&p);
            let y = d.ppf(&u);
            let g = Scorable::<LogScore>::d_score(&d, &y);
            let e =
                |a: usize, b: usize| (0..n).map(|i| g[[i, a]] * g[[i, b]]).sum::<f64>() / n as f64;
            let m = Scorable::<LogScore>::metric(&dist(lm, la));
            let (i00, i11) = (m[[0, 0, 0]], m[[0, 1, 1]]);
            assert!(
                (e(0, 0) - i00).abs() < 5e-3 * i00,
                "I00 {lm} {la}: {} vs {i00}",
                e(0, 0)
            );
            assert!(
                (e(1, 1) - i11).abs() < 5e-3 * i11,
                "I11 {lm} {la}: {} vs {i11}",
                e(1, 1)
            );
            assert!(
                e(0, 1).abs() < 5e-3 * (i00 * i11).sqrt(),
                "I01 {lm} {la}: {}",
                e(0, 1)
            );
            assert_eq!(m[[0, 0, 1]], 0.0);
            assert_eq!(m[[0, 1, 0]], 0.0);
        }
        // the large-α series joins the exact form
        let (a, b) = (shape_fisher(999.9), shape_fisher(1000.1));
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn marginal_fit_matches_gamma() {
        let y: Array1<f64> = (1..300)
            .map(|i| 0.1 + ((i * 53) % 97) as f64 / 9.0)
            .collect();
        let (p, q) = (GammaMean::fit(&y), Gamma::fit(&y));
        let (d, g) = (
            GammaMean::from_params(&p.clone().insert_axis(ndarray::Axis(0))),
            Gamma::from_params(&q.clone().insert_axis(ndarray::Axis(0))),
        );
        assert!((d.mu[0] - g.mean()[0]).abs() < 1e-10);
        assert!((d.shape[0] - g.shape[0]).abs() < 1e-10);
    }
}
