//! Negative Binomial (NB2) distribution for over-dispersed counts.
//!
//! Parameters (internal, unconstrained): `log μ` (mean) and `log r`
//! (dispersion / "size"); Var(Y) = μ + μ²/r, so r → ∞ is the Poisson limit.
//! pmf: P(Y = y) = Γ(y+r)/(Γ(r) y!) · (r/(r+μ))^r · (μ/(r+μ))^y.
//!
//! LogScore and CRPScore. Fisher information is diagonal: the mean and the
//! dispersion are orthogonal parameters in NB2.
//!
//! CRPScore uses the discrete CRPS Σ_k (F(k) − 1{k ≥ y})², its gradient
//! 2 Σ_k (F(k) − 1{k ≥ y}) ∂F(k) and the full 2×2 metric 2 Σ_k ∂F(k) ∂F(k)ᵀ,
//! all from one pass over the pmf with ∂F(k) = Σ_{j≤k} p(j) ∇log p(j).
//!   F_{log μ}  = r μ / (r + μ)
//!   F_{log r}  = r² · E[s(Y)²],  s(y) = ψ(y+r) − ψ(r) + ln(r/(r+μ)) + (μ − y)/(r + μ)
//! The second entry has no closed form; it is summed over the pmf in the log
//! domain (ψ(y+r) − ψ(r) = Σ_{k<y} 1/(r+k) accumulated incrementally, so no
//! digamma calls and no cancellation). For r ≫ μ it decays like μ²/(2r²)
//! while the gradient decays only like μ/r, so the natural gradient would grow
//! like r; `DISP_FISHER_FLOOR` turns the step into a plain small gradient step
//! in that weakly identified regime (it is far below the ≈0.1–1 values of the
//! identifiable regime r ≲ μ). At `R_MAX` the log r coordinate is frozen
//! (d_score = 0, unit metric), as StudentT does for its df.
use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use statrs::function::gamma::{digamma, ln_gamma};

/// Dispersion cap: numerically Poisson beyond this; log r is frozen there.
pub const R_MAX: f64 = 1e6;
const R_MIN: f64 = 1e-3;
const DISP_FISHER_FLOOR: f64 = 1e-2;
/// Method-of-moments init range: near-Poisson data starts at a finite,
/// movable r instead of the cap.
const R_INIT_MIN: f64 = 0.1;
const R_INIT_MAX: f64 = 1e3;
/// pmf series cap (cdf / ppf / Fisher sums).
const SERIES_CAP: usize = 200_000;

/// The Negative Binomial distribution (mean / dispersion parameterization).
#[derive(Debug, Clone)]
pub struct NegativeBinomial {
    /// Mean μ.
    pub mu: Array1<f64>,
    /// Dispersion r (size); variance = μ + μ²/r.
    pub r: Array1<f64>,
    /// Raw parameter log μ.
    pub log_mu: Array1<f64>,
    /// Raw parameter log r (after the `R_MAX` clamp).
    pub log_r: Array1<f64>,
}

#[inline]
fn ln_pmf(y: f64, mu: f64, r: f64) -> f64 {
    if y < 0.0 {
        return f64::NEG_INFINITY;
    }
    if r >= R_MAX {
        // Poisson limit (avoids lnΓ cancellation at huge r)
        return y * mu.ln() - mu - ln_gamma(y + 1.0);
    }
    // ln(r/(r+μ)) = −ln1p(μ/r); ln(μ/(r+μ)) = ln μ − ln(r+μ)
    ln_gamma(y + r) - ln_gamma(r) - ln_gamma(y + 1.0) - r * (mu / r).ln_1p()
        + y * (mu.ln() - (r + mu).ln())
}

/// ∂(−ln pmf)/∂ log r (0 when frozen at the cap).
#[inline]
fn d_log_r(y: f64, mu: f64, r: f64) -> f64 {
    if r >= R_MAX {
        return 0.0;
    }
    let s = digamma(y + r) - digamma(r) - (mu / r).ln_1p() + (mu - y) / (r + mu);
    -r * s
}

/// Fisher entry for log r: r² · Σ_y p(y) s(y)², summed in the log domain.
fn disp_fisher(mu: f64, r: f64) -> f64 {
    if r >= R_MAX {
        return 1.0;
    }
    let lnq = mu.ln() - (r + mu).ln(); // ln(μ/(r+μ))
    let ln_1mq = -(mu / r).ln_1p(); // ln(r/(r+μ))
    let mut lp = r * ln_1mq; // ln P(Y = 0)
    let mut h = 0.0; // ψ(y+r) − ψ(r)
    let (mut acc, mut cum) = (0.0f64, 0.0f64);
    let sd = (mu + mu * mu / r).sqrt();
    let cap = ((mu + 60.0 * sd + 200.0) as usize).min(SERIES_CAP);
    for y in 0..=cap {
        let yf = y as f64;
        let p = lp.exp();
        let s = h + ln_1mq + (mu - yf) / (r + mu);
        acc += p * s * s;
        cum += p;
        if cum > 1.0 - 1e-13 {
            break;
        }
        h += 1.0 / (r + yf);
        lp += ((yf + r) / (yf + 1.0)).ln() + lnq;
    }
    (r * r * acc).max(DISP_FISHER_FLOOR)
}

/// Σ_{k ≤ floor(y)} pmf(k), log-domain recursion.
fn cdf_scalar(y: f64, mu: f64, r: f64) -> f64 {
    if y < 0.0 {
        return 0.0;
    }
    if r >= R_MAX {
        // Poisson cdf via the regularized upper incomplete gamma: Q(k+1, μ)
        return statrs::function::gamma::gamma_ur(y.floor() + 1.0, mu);
    }
    let k_max = (y.floor() as usize).min(SERIES_CAP);
    let lnq = mu.ln() - (r + mu).ln();
    let mut lp = -r * (mu / r).ln_1p();
    let mut cum = 0.0;
    for k in 0..=k_max {
        cum += lp.exp();
        lp += ((k as f64 + r) / (k as f64 + 1.0)).ln() + lnq;
    }
    cum.min(1.0)
}

impl Distribution for NegativeBinomial {
    /// Newton / Fisher-scoring leaves by default, as for Poisson: the mean
    /// column's Fisher rμ/(r+μ) varies with the rate, so the unweighted leaf
    /// is a mean of ratios. Measured −0.12 NLL / −15 % RMSE on heterogeneous
    /// rates, neutral on over-dispersed data, +0.003…0.005 on homogeneous
    /// low-rate true-Poisson counts (`research/ngboost/reports/R15_*`).
    fn default_fisher_weight_power() -> Option<f64> {
        Some(1.0)
    }

    /// log μ is the location for the staged fit (`staged_fit = Some(true)`;
    /// opt-in, research Cycle 16): the mean first, then log r given the mean.
    fn location_column() -> Option<usize> {
        Some(0)
    }

    /// log r is recalibrated by the default end-of-fit validation map
    /// (`auto_calibrate_scale`, research Cycle 16): r fitted to in-sample
    /// residuals is too large out of sample (over-confident), exactly as σ is
    /// too small for Normal. An affine map on log r leaves the mean μ alone.
    fn scale_param_index() -> Option<usize> {
        Some(1)
    }

    /// Dispersion is harder to pin down than a Normal σ: with 100 validation
    /// rows the shift-only map lost on 2 of 5 count DGPs (+0.004 NLL, 3/10),
    /// with 300 (shift + slope) it never lost (R63).
    fn auto_calibration_min_rows() -> usize {
        300
    }

    fn from_params(params: &Array2<f64>) -> Self {
        let log_mu = params.column(0).to_owned();
        let mu = crate::vmath::exp_array(&log_mu);
        // `>` (not f64::min) so a NaN log r stays NaN and is still caught.
        let r = crate::vmath::exp_column(&params.column(1)).mapv_into(|v| {
            if v > R_MAX {
                R_MAX
            } else if v < R_MIN {
                R_MIN
            } else {
                v
            }
        });
        let log_r = r.mapv(f64::ln);
        NegativeBinomial {
            mu,
            r,
            log_mu,
            log_r,
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let n = y.len();
        let mean = y.mean().unwrap_or(1.0).max(1e-6);
        let var = if n > 1 {
            y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0)
        } else {
            mean
        };
        // Method of moments: Var = μ + μ²/r ⇒ r = μ² / (Var − μ)
        let r = if var > mean * (1.0 + 1e-6) {
            mean * mean / (var - mean)
        } else {
            R_INIT_MAX
        };
        array![mean.ln(), r.clamp(R_INIT_MIN, R_INIT_MAX).ln()]
    }

    fn validate_targets(y: &Array1<f64>) -> Result<(), &'static str> {
        if y.iter().any(|&v| !(v >= 0.0)) {
            return Err("NegativeBinomial targets must be non-negative");
        }
        Ok(())
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        self.mu.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.mu.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.log_mu);
        p.column_mut(1).assign(&self.log_r);
        p
    }
}

impl RegressionDistn for NegativeBinomial {}

impl DistributionMethods for NegativeBinomial {
    fn mean(&self) -> Array1<f64> {
        self.mu.clone()
    }

    fn variance(&self) -> Array1<f64> {
        Zip::from(&self.mu)
            .and(&self.r)
            .map_collect(|&m, &r| m + m * m / r)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.logpdf(y).mapv(f64::exp)
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        Zip::from(y)
            .and(&self.mu)
            .and(&self.r)
            .map_collect(|&yi, &m, &r| ln_pmf(yi, m, r))
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        Zip::from(y)
            .and(&self.mu)
            .and(&self.r)
            .map_collect(|&yi, &m, &r| cdf_scalar(yi, m, r))
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        Zip::from(q)
            .and(&self.mu)
            .and(&self.r)
            .map_collect(|&qi, &m, &r| {
                if !(0.0..=1.0).contains(&qi) {
                    return f64::NAN;
                }
                if qi >= 1.0 {
                    return f64::INFINITY;
                }
                let lnq = m.ln() - (r + m).ln();
                let mut lp = if r >= R_MAX { -m } else { -r * (m / r).ln_1p() };
                let mut cum = 0.0;
                for k in 0..=SERIES_CAP {
                    cum += lp.exp();
                    if cum >= qi {
                        return k as f64;
                    }
                    lp += if r >= R_MAX {
                        m.ln() - (k as f64 + 1.0).ln()
                    } else {
                        ((k as f64 + r) / (k as f64 + 1.0)).ln() + lnq
                    };
                }
                SERIES_CAP as f64
            })
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        // Gamma–Poisson mixture: λ ~ Gamma(r, scale μ/r), Y | λ ~ Poisson(λ).
        let n_obs = self.mu.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();
        let mut normal = crate::vmath::StdNormalSampler::new();
        for i in 0..n_obs {
            let (m, r) = (self.mu[i], self.r[i]);
            if !(m.is_finite() && m > 0.0 && r > 0.0) {
                continue;
            }
            for s in 0..n_samples {
                let lam = if r >= R_MAX {
                    m
                } else {
                    m * crate::vmath::sample_gamma_rate1(&mut rng, &mut normal, r) / r
                };
                samples[[s, i]] = sample_poisson(&mut rng, lam);
            }
        }
        samples
    }
}

/// Poisson(λ) draw: Knuth's product method for λ < 300, normal approximation
/// (rounded, non-negative) above — used only by `sample`.
fn sample_poisson<R: rand::RngExt + ?Sized>(rng: &mut R, lam: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam < 300.0 {
        let l = (-lam).exp();
        let (mut k, mut p) = (0u64, 1.0f64);
        loop {
            k += 1;
            p *= rng.random_range(0.0..1.0f64);
            if p <= l {
                return (k - 1) as f64;
            }
        }
    }
    let mut z = crate::vmath::StdNormalSampler::new();
    (lam + lam.sqrt() * z.next(rng)).round().max(0.0)
}

impl Scorable<LogScore> for NegativeBinomial {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let n = self.mu.len();
        let mut diag = Array2::zeros((n, 2));
        Zip::from(diag.rows_mut())
            .and(&self.mu)
            .and(&self.r)
            .for_each(|mut row, &m, &r| {
                row[0] = r * m / (r + m);
                row[1] = disp_fisher(m, r);
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Zip::from(y)
            .and(&self.mu)
            .and(&self.r)
            .map_collect(|&yi, &m, &r| -ln_pmf(yi, m, r))
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n = y.len();
        let mut d = Array2::zeros((n, 2));
        Zip::from(d.rows_mut())
            .and(y)
            .and(&self.mu)
            .and(&self.r)
            .for_each(|mut row, &yi, &m, &r| {
                row[0] = r * (m - yi) / (r + m);
                row[1] = d_log_r(yi, m, r);
            });
        d
    }

    fn metric(&self) -> Array3<f64> {
        let n = self.mu.len();
        let mut fi = Array3::zeros((n, 2, 2));
        let diag = Scorable::<LogScore>::diagonal_metric(self);
        for i in 0..n {
            fi[[i, 0, 0]] = diag[[i, 0]];
            fi[[i, 1, 1]] = diag[[i, 1]];
        }
        fi
    }
}

/// One pass over the pmf for the discrete CRPS: returns (CRPS, ∇CRPS
/// [log μ, log r], metric [m00, m01, m11]). At the `R_MAX` cap the log r
/// coordinate is frozen (zero gradient, unit metric entry), as for LogScore.
fn crps_row(y: f64, mu: f64, r: f64) -> (f64, [f64; 2], [f64; 3]) {
    let frozen = r >= R_MAX;
    let lnq = mu.ln() - (r + mu).ln();
    let ln_1mq = -(mu / r).ln_1p();
    let mut lp = if frozen { -mu } else { r * ln_1mq };
    let mut h = 0.0; // ψ(k + r) − ψ(r)
    let (mut cum, mut df0, mut df1) = (0.0f64, 0.0f64, 0.0f64);
    let (mut score, mut g0, mut g1) = (0.0f64, 0.0f64, 0.0f64);
    let (mut m00, mut m01, mut m11) = (0.0f64, 0.0f64, 0.0f64);
    let sd = (mu + mu * mu / r.min(R_MAX)).sqrt();
    let cap = ((mu.max(y) + 60.0 * sd + 200.0) as usize).min(SERIES_CAP);
    for k in 0..=cap {
        let kf = k as f64;
        let p = lp.exp();
        // ∇ log p(k): log μ → r(k − μ)/(r + μ) (Poisson limit: k − μ);
        // log r → r(ψ(k+r) − ψ(r) + ln(r/(r+μ)) + (μ − k)/(r + μ))
        let s0 = if frozen {
            kf - mu
        } else {
            r * (kf - mu) / (r + mu)
        };
        let s1 = if frozen {
            0.0
        } else {
            r * (h + ln_1mq + (mu - kf) / (r + mu))
        };
        cum += p;
        df0 += p * s0;
        df1 += p * s1;
        let f = cum.min(1.0);
        let resid = f - if kf >= y { 1.0 } else { 0.0 };
        score += resid * resid;
        g0 += 2.0 * resid * df0;
        g1 += 2.0 * resid * df1;
        m00 += 2.0 * df0 * df0;
        m01 += 2.0 * df0 * df1;
        m11 += 2.0 * df1 * df1;
        if kf >= y && 1.0 - cum < 1e-13 {
            break;
        }
        if !frozen {
            h += 1.0 / (r + kf);
            lp += ((kf + r) / (kf + 1.0)).ln() + lnq;
        } else {
            lp += mu.ln() - (kf + 1.0).ln();
        }
    }
    if frozen {
        m11 = 1.0;
        m01 = 0.0;
    }
    (score, [g0, g1], [m00.max(1e-12), m01, m11.max(1e-12)])
}

impl Scorable<CRPScore> for NegativeBinomial {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| crps_row(y[i], self.mu[i], self.r[i]).0)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let mut d = Array2::zeros((y.len(), 2));
        for i in 0..y.len() {
            let (_, g, _) = crps_row(y[i], self.mu[i], self.r[i]);
            d[[i, 0]] = g[0];
            d[[i, 1]] = g[1];
        }
        d
    }

    /// Metric 2Σ_k ∂F(k)∂F(k)ᵀ does not depend on y; evaluated at y = 0.
    fn metric(&self) -> Array3<f64> {
        let n = self.mu.len();
        let mut m = Array3::zeros((n, 2, 2));
        for i in 0..n {
            let (_, _, mm) = crps_row(0.0, self.mu[i], self.r[i]);
            m[[i, 0, 0]] = mm[0];
            m[[i, 0, 1]] = mm[1];
            m[[i, 1, 0]] = mm[1];
            m[[i, 1, 1]] = mm[2];
        }
        m
    }

    /// Fused single pass: gradient and metric together, 2×2 closed-form solve
    /// (+ reg·I), no LAPACK.
    fn dense_natural_grad(&self, y: &Array1<f64>, reg: f64) -> Array2<f64> {
        let mut out = Array2::zeros((y.len(), 2));
        for i in 0..y.len() {
            let (_, g, m) = crps_row(y[i], self.mu[i], self.r[i]);
            let (a, b, d) = (m[0] + reg, m[1], m[2] + reg);
            let det = a * d - b * b;
            if det > 0.0 && det.is_finite() {
                out[[i, 0]] = (d * g[0] - b * g[1]) / det;
                out[[i, 1]] = (a * g[1] - b * g[0]) / det;
            } else {
                out[[i, 0]] = g[0] / a;
                out[[i, 1]] = g[1] / d;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pmf_sums_to_one_and_moments_match() {
        for &(m, r) in &[(0.5, 0.3), (3.0, 1.5), (20.0, 4.0), (50.0, 100.0)] {
            let (mut tot, mut mean, mut m2) = (0.0, 0.0, 0.0);
            for k in 0..5000 {
                let p = ln_pmf(k as f64, m, r).exp();
                tot += p;
                mean += k as f64 * p;
                m2 += (k as f64) * (k as f64) * p;
            }
            assert!((tot - 1.0).abs() < 1e-9, "μ={m} r={r}: total {tot}");
            assert!((mean - m).abs() < 1e-6 * m.max(1.0), "mean {mean} vs {m}");
            let var = m2 - mean * mean;
            assert!(
                (var - (m + m * m / r)).abs() < 1e-5 * (m + m * m / r),
                "var {var}"
            );
        }
    }

    #[test]
    fn poisson_limit() {
        let (m, r) = (7.0, R_MAX);
        for k in 0..30 {
            let nb = ln_pmf(k as f64, m, r);
            let po = k as f64 * m.ln() - m - ln_gamma(k as f64 + 1.0);
            assert!((nb - po).abs() < 1e-9);
        }
        // just below the cap the two agree to O(1/r)
        let r2 = 1e5;
        let nb = ln_pmf(7.0, m, r2);
        let po = 7.0 * m.ln() - m - ln_gamma(8.0);
        assert!((nb - po).abs() < 1e-3, "{nb} vs {po}");
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let eps = 1e-6;
        let cases: [(f64, f64, f64); 4] = [
            (2.0, 0.7, 0.0),
            (5.0, 2.0, 3.0),
            (30.0, 5.0, 45.0),
            (1.0, 50.0, 2.0),
        ];
        for &(m, r, y) in &cases {
            let f = |lm: f64, lr: f64| -ln_pmf(y, lm.exp(), lr.exp());
            let (lm, lr) = (m.ln(), r.ln());
            let g0 = (f(lm + eps, lr) - f(lm - eps, lr)) / (2.0 * eps);
            let g1 = (f(lm, lr + eps) - f(lm, lr - eps)) / (2.0 * eps);
            let a0 = r * (m - y) / (r + m);
            let a1 = d_log_r(y, m, r);
            assert!(
                (g0 - a0).abs() < 1e-6 * (1.0 + a0.abs()),
                "d/dlogμ {a0} vs {g0}"
            );
            assert!(
                (g1 - a1).abs() < 1e-6 * (1.0 + a1.abs()),
                "d/dlogr {a1} vs {g1}"
            );
        }
    }

    /// Fisher entries equal E[score²] under the model (exact sums over the pmf).
    #[test]
    fn fisher_matches_expected_squared_score() {
        for &(m, r) in &[(3.0, 1.5), (20.0, 4.0), (8.0, 0.5)] {
            let (mut e00, mut e11, mut e01) = (0.0, 0.0, 0.0);
            for k in 0..20000 {
                let y = k as f64;
                let p = ln_pmf(y, m, r).exp();
                let g0 = r * (m - y) / (r + m);
                let g1 = d_log_r(y, m, r);
                e00 += p * g0 * g0;
                e11 += p * g1 * g1;
                e01 += p * g0 * g1;
            }
            assert!((e00 - r * m / (r + m)).abs() < 1e-6 * e00, "F00 {e00}");
            let f11 = disp_fisher(m, r);
            assert!(
                (e11 - f11).abs() < 1e-6 * e11,
                "F11 {f11} vs {e11} (μ={m}, r={r})"
            );
            assert!(e01.abs() < 1e-8, "orthogonality {e01}");
        }
    }

    #[test]
    fn cdf_ppf_consistency_and_init() {
        let params =
            Array2::from_shape_vec((2, 2), vec![10f64.ln(), 2f64.ln(), 3f64.ln(), 50f64.ln()])
                .unwrap();
        let d = NegativeBinomial::from_params(&params);
        let ys = Array1::from_vec(vec![12.0, 2.0]);
        let c = d.cdf(&ys);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)));
        let back = d.ppf(&c);
        assert!(
            (back[0] - 12.0).abs() < 1e-9 && (back[1] - 2.0).abs() < 1e-9,
            "{back}"
        );
        // moments init on an over-dispersed sample
        let y = Array1::from_vec(vec![
            0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 2.0, 4.0, 0.0,
        ]);
        let p = NegativeBinomial::fit(&y);
        let mean = y.mean().unwrap();
        assert!((p[0].exp() - mean).abs() < 1e-12);
        assert!(p[1].exp() > R_INIT_MIN && p[1].exp() < R_INIT_MAX);
        // Poisson-like sample → capped init r
        let yp = Array1::from_vec(vec![3.0, 4.0, 3.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 3.0]);
        assert!((NegativeBinomial::fit(&yp)[1].exp() - R_INIT_MAX).abs() < 1e-9);
    }

    #[test]
    fn sample_moments() {
        let params = Array2::from_shape_vec((1, 2), vec![6f64.ln(), 3f64.ln()]).unwrap();
        let d = NegativeBinomial::from_params(&params);
        let s = d.sample(200_000);
        let col = s.column(0);
        let n = col.len() as f64;
        let mean = col.sum() / n;
        let var = col.mapv(|v| (v - mean) * (v - mean)).sum() / n;
        assert!((mean - 6.0).abs() < 0.1, "mean {mean}");
        assert!((var - (6.0 + 12.0)).abs() < 0.6, "var {var}");
    }

    #[test]
    fn crps_matches_direct_sum_and_gradient_matches_fd() {
        let cases: [(f64, f64, f64); 4] = [
            (3.0, 2.0, 0.0),
            (8.0, 1.5, 5.0),
            (20.0, 4.0, 31.0),
            (6.0, 50.0, 6.0),
        ];
        for &(m, r, y) in &cases {
            let direct: f64 = (0..5000)
                .map(|k| {
                    let f: f64 = (0..=k).map(|j| ln_pmf(j as f64, m, r).exp()).sum();
                    (f.min(1.0) - if k as f64 >= y { 1.0 } else { 0.0 }).powi(2)
                })
                .sum();
            let (s, g, _) = crps_row(y, m, r);
            assert!(
                (s - direct).abs() < 1e-9 * (1.0 + direct),
                "μ={m} r={r} y={y}: {s} vs {direct}"
            );
            let eps = 1e-6;
            let f = |lm: f64, lr: f64| crps_row(y, lm.exp(), lr.exp()).0;
            let (lm, lr) = (m.ln(), r.ln());
            let fd0 = (f(lm + eps, lr) - f(lm - eps, lr)) / (2.0 * eps);
            let fd1 = (f(lm, lr + eps) - f(lm, lr - eps)) / (2.0 * eps);
            assert!(
                (g[0] - fd0).abs() < 1e-5 * (1.0 + fd0.abs()),
                "d/dlogμ {} vs {fd0}",
                g[0]
            );
            assert!(
                (g[1] - fd1).abs() < 1e-5 * (1.0 + fd1.abs()),
                "d/dlogr {} vs {fd1}",
                g[1]
            );
        }
    }

    /// The CRPS metric is E_Y[∇CRPS ∇CRPSᵀ]/2-style curvature: check it equals
    /// the expected Hessian of the CRPS under the model (Fisher-type identity
    /// for proper scores: E[∇²S] = 2Σ∂F∂Fᵀ), by finite differences of the
    /// expected gradient.
    #[test]
    fn crps_metric_equals_expected_hessian() {
        let (m, r) = (7.0f64, 3.0f64);
        let exp_grad = |lm: f64, lr: f64| {
            // E_{Y ~ NB(m, r)}[∇ CRPS(NB(e^lm, e^lr), Y)]
            let mut acc = [0.0f64; 2];
            for y in 0..400 {
                let p = ln_pmf(y as f64, m, r).exp();
                let (_, g, _) = crps_row(y as f64, lm.exp(), lr.exp());
                acc[0] += p * g[0];
                acc[1] += p * g[1];
            }
            acc
        };
        let eps = 1e-5;
        let (lm, lr) = (m.ln(), r.ln());
        let (gp0, gm0) = (exp_grad(lm + eps, lr), exp_grad(lm - eps, lr));
        let (gp1, gm1) = (exp_grad(lm, lr + eps), exp_grad(lm, lr - eps));
        let h00 = (gp0[0] - gm0[0]) / (2.0 * eps);
        let h01 = (gp1[0] - gm1[0]) / (2.0 * eps);
        let h11 = (gp1[1] - gm1[1]) / (2.0 * eps);
        let (_, _, mm) = crps_row(0.0, m, r);
        assert!(
            (mm[0] - h00).abs() < 1e-4 * (1.0 + h00.abs()),
            "m00 {} vs {h00}",
            mm[0]
        );
        assert!(
            (mm[1] - h01).abs() < 1e-4 * (1.0 + h01.abs()),
            "m01 {} vs {h01}",
            mm[1]
        );
        assert!(
            (mm[2] - h11).abs() < 1e-4 * (1.0 + h11.abs()),
            "m11 {} vs {h11}",
            mm[2]
        );
    }
}
