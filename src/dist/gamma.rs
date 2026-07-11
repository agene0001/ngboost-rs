use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, Gamma as GammaDist};
use statrs::function::gamma::digamma;

/// The Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    pub shape: Array1<f64>, // alpha
    pub rate: Array1<f64>,  // beta
}

impl Distribution for Gamma {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = crate::vmath::exp_column(&params.column(0));
        let rate = crate::vmath::exp_column(&params.column(1));
        Gamma { shape, rate }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // MLE for Gamma with fixed loc=0, matching scipy.stats.gamma.fit(Y, floc=0).
        // Uses Newton's method on: log(shape) - digamma(shape) = log(mean(Y)) - mean(log(Y))
        let n = y.len() as f64;
        let mean = y.mean().unwrap_or(1.0);
        let mean_log: f64 = y.iter().map(|&v| v.max(1e-300).ln()).sum::<f64>() / n;
        let s = mean.max(1e-300).ln() - mean_log; // s = log(mean(Y)) - mean(log(Y))

        // Initial estimate using Minka's approximation
        let mut shape = if s > 0.0 {
            (3.0 - s + ((s - 3.0) * (s - 3.0) + 24.0 * s).sqrt()) / (12.0 * s)
        } else {
            1.0
        };

        // Newton iterations: f(a) = log(a) - digamma(a) - s, f'(a) = 1/a - trigamma(a)
        for _ in 0..50 {
            let f = shape.ln() - digamma(shape) - s;
            let f_prime = 1.0 / shape - trigamma(shape);
            if f_prime.abs() < 1e-15 {
                break;
            }
            let step = f / f_prime;
            shape -= step;
            shape = shape.max(1e-10);
            if step.abs() < 1e-10 {
                break;
            }
        }

        let scale = mean / shape; // MLE scale = mean(Y) / shape
        let rate = 1.0 / scale;
        array![shape.ln(), rate.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean is shape / rate
        &self.shape / &self.rate
    }

    fn params(&self) -> Array2<f64> {
        let n = self.shape.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.shape.mapv(f64::ln));
        p.column_mut(1).assign(&self.rate.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for Gamma {}

impl DistributionMethods for Gamma {
    fn mean(&self) -> Array1<f64> {
        // Mean of Gamma is shape / rate
        &self.shape / &self.rate
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of Gamma is shape / rate^2
        &self.shape / (&self.rate * &self.rate)
    }

    fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inline formula: f(x; k, β) = (β^k * x^(k-1) * e^(-β*x)) / Γ(k)
        // Compute via exp(logpdf) to avoid overflow
        self.logpdf(y).mapv(f64::exp)
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inline formula: ln(f(x)) = k*ln(β) + (k-1)*ln(x) - β*x - ln(Γ(k))
        use statrs::function::gamma::ln_gamma;
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.shape)
            .and(&self.rate)
            .for_each(|r, &y_i, &k, &beta| {
                if y_i > 0.0 && k > 0.0 && beta > 0.0 {
                    *r = k * beta.ln() + (k - 1.0) * y_i.ln() - beta * y_i - ln_gamma(k);
                } else {
                    *r = f64::NEG_INFINITY;
                }
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized CDF using statrs library for accuracy
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.shape)
            .and(&self.rate)
            .for_each(|r, &y_i, &shape, &rate| {
                if let Ok(d) = GammaDist::new(shape, rate) {
                    *r = d.cdf(y_i);
                }
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Vectorized PPF using statrs library for accuracy
        let mut result = Array1::zeros(q.len());
        Zip::from(&mut result)
            .and(q)
            .and(&self.shape)
            .and(&self.rate)
            .for_each(|r, &q_i, &shape, &rate| {
                if let Ok(d) = GammaDist::new(shape, rate) {
                    let q_clamped = q_i.clamp(1e-15, 1.0 - 1e-15);
                    *r = d.inverse_cdf(q_clamped);
                }
            });
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.shape.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = GammaDist::new(self.shape[i], self.rate[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // For Gamma, median has no closed form. Use ppf(0.5)
        let q = Array1::from_elem(self.shape.len(), 0.5);
        self.ppf(&q)
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Gamma is (shape - 1) / rate for shape >= 1, else 0
        let mut result = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            if self.shape[i] >= 1.0 {
                result[i] = (self.shape[i] - 1.0) / self.rate[i];
            }
        }
        result
    }
}

impl Scorable<LogScore> for Gamma {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -ln_pdf: -(k*ln(β) + (k-1)*ln(y) - β*y - ln_gamma(k))
        use statrs::function::gamma::ln_gamma;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.shape)
            .and(&self.rate)
            .for_each(|s, &y_i, &k, &beta| {
                *s = -(k * beta.ln() + (k - 1.0) * y_i.max(1e-300).ln() - beta * y_i - ln_gamma(k));
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Vectorized gradient w.r.t. log(shape) and log(rate)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.shape)
            .and(&self.rate)
            .for_each(|mut row, &y_i, &shape_i, &rate_i| {
                // d/d(log(shape)) - matches Python: log(eps + beta * Y) with eps=1e-10
                row[0] = shape_i * (digamma(shape_i) - (1e-10 + y_i * rate_i).ln());
                // d/d(log(rate))
                row[1] = y_i * rate_i - shape_i;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Vectorized Fisher Information Matrix
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        Zip::from(fi.outer_iter_mut())
            .and(&self.shape)
            .for_each(|mut fi_i, &shape_i| {
                fi_i[[0, 0]] = shape_i * shape_i * trigamma(shape_i);
                fi_i[[1, 1]] = shape_i;
                fi_i[[0, 1]] = -shape_i;
                fi_i[[1, 0]] = -shape_i;
            });

        fi
    }
}

impl Scorable<CRPScore> for Gamma {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Gamma distribution
        // Based on: Gneiting, T. and Raftery, A.E. (2007)
        // CRPS(F, y) = y * (2*F(y) - 1) - shape/rate * (2*F_alpha+1(y) - 1)
        //              + shape / (rate * Beta(0.5, shape))
        // where F is the CDF and F_alpha+1 is the CDF of Gamma(shape+1, rate)
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let shape = self.shape[i];
            let rate = self.rate[i];
            let y_i = y[i];

            // CDF of Gamma(shape, rate) at y
            let f_y = if let Ok(d) = GammaDist::new(shape, rate) {
                d.cdf(y_i)
            } else {
                0.5
            };

            // CDF of Gamma(shape+1, rate) at y
            let f_alpha1_y = if let Ok(d) = GammaDist::new(shape + 1.0, rate) {
                d.cdf(y_i)
            } else {
                0.5
            };

            // Beta(0.5, shape) term using gamma functions
            // Beta(0.5, a) = Gamma(0.5) * Gamma(a) / Gamma(a + 0.5)
            // = sqrt(pi) * Gamma(a) / Gamma(a + 0.5)
            let beta_term = beta(0.5, shape);

            // CRPS formula (Gneiting & Raftery 2007, scoringRules R package):
            // CRPS = y*(2*F(y) - 1) - (shape/rate)*(2*F_{shape+1}(y) - 1) - 1/(rate*B(0.5, shape))
            let mean = shape / rate;
            scores[i] = y_i * (2.0 * f_y - 1.0)
                - mean * (2.0 * f_alpha1_y - 1.0)
                - 1.0 / (rate * beta_term);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Numerical gradient for CRPS using central differences
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-5;

        for i in 0..n_obs {
            let shape_i = self.shape[i];
            let rate_i = self.rate[i];
            let y_i = y[i];

            // Derivative w.r.t. log(shape) via central finite difference
            let shape_plus = shape_i * (1.0 + eps);
            let shape_minus = shape_i * (1.0 - eps);
            let score_shape_plus = self.crps_single(y_i, shape_plus, rate_i);
            let score_shape_minus = self.crps_single(y_i, shape_minus, rate_i);
            d_params[[i, 0]] = (score_shape_plus - score_shape_minus) / (2.0 * eps);

            // Derivative w.r.t. log(rate) via central finite difference
            let rate_plus = rate_i * (1.0 + eps);
            let rate_minus = rate_i * (1.0 - eps);
            let score_rate_plus = self.crps_single(y_i, shape_i, rate_plus);
            let score_rate_minus = self.crps_single(y_i, shape_i, rate_minus);
            d_params[[i, 1]] = (score_rate_plus - score_rate_minus) / (2.0 * eps);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Deterministic quadrature for E[g * gᵀ] over Y ~ Gamma(α, β).
        //
        // No inverse CDF (the previous PIT rule used statrs `inverse_cdf`
        // root-finding per node — same hang class observed for StudentT, and
        // ~100µs/node). Instead:
        //  - α ≥ 1: single log-domain panel, s = ln(w), w ~ Gamma(α,1):
        //      E[h] = ∫ h(e^s/β) exp(αs − e^s − lnΓ(α)) ds
        //    on s ∈ [(ln 1e-13 + lnΓ(α+1))/α, ln(α + 12√α + 40)] (non-iterative
        //    truncation bounds; both tails decay (super)exponentially in s).
        //  - α < 1: hybrid. Deep-left tail uses the asymptotic inverse CDF
        //    F(w) ≈ w^α/Γ(1+α) (exact as w→0), whose quantile is closed-form:
        //      w(u) = exp((ln u + lnΓ(1+α))/α),  node weight e^{−w}·Δu.
        //    Remainder (w > min(α,1)e⁻⁴) uses the log-domain panel.
        // Validated vs 2M-point references: rel err ≤ 2.3e-5 over
        // α ∈ [0.005, 1000] — everywhere 1.5-4 orders better than the old
        // 200-pt PIT rule (whose tail truncation gave 2.5e-5..1.3e-1).
        use statrs::function::gamma::ln_gamma;
        let n_obs = self.shape.len();
        let n_params = 2;
        let n_points: usize = 200;
        let eps = 1e-5;

        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            let shape_i = self.shape[i];
            let rate_i = self.rate[i];
            if !(shape_i > 0.0 && rate_i > 0.0) || !shape_i.is_finite() || !rate_i.is_finite() {
                continue;
            }

            // Gradient kernel at a point y (finite differences, same as d_score).
            let grads_at = |y_i: f64| -> [f64; 2] {
                let g0 = (self.crps_single(y_i, shape_i * (1.0 + eps), rate_i)
                    - self.crps_single(y_i, shape_i * (1.0 - eps), rate_i))
                    / (2.0 * eps);
                let g1 = (self.crps_single(y_i, shape_i, rate_i * (1.0 + eps))
                    - self.crps_single(y_i, shape_i, rate_i * (1.0 - eps)))
                    / (2.0 * eps);
                [g0, g1]
            };

            let mut acc = [[0.0f64; 2]; 2];
            fn accumulate(acc: &mut [[f64; 2]; 2], w: f64, g: [f64; 2]) {
                if w > 0.0 && w.is_finite() && g[0].is_finite() && g[1].is_finite() {
                    for a in 0..2 {
                        for b in 0..2 {
                            acc[a][b] += w * g[a] * g[b];
                        }
                    }
                }
            }

            let ln_gamma_a = ln_gamma(shape_i);
            let ln_gamma_a1 = ln_gamma(shape_i + 1.0);
            let s_hi = (shape_i + 12.0 * shape_i.sqrt() + 40.0).ln();

            // Log-domain panel over s ∈ [s_lo, s_hi] with n nodes.
            let log_panel =
                |acc: &mut [[f64; 2]; 2], s_lo: f64, s_hi: f64, n: usize| {
                    if n == 0 || s_hi <= s_lo {
                        return;
                    }
                    let ds = (s_hi - s_lo) / n as f64;
                    for j in 0..n {
                        let s = s_lo + ds * (j as f64 + 0.5);
                        let w_node = s.exp();
                        // pdf(w)·w·Δs computed in log space (no y^(α−1) overflow)
                        let w = (shape_i * s - w_node - ln_gamma_a).exp() * ds;
                        accumulate(acc, w, grads_at(w_node / rate_i));
                    }
                };

            if shape_i >= 1.0 {
                let s_lo = ((1e-13f64).ln() + ln_gamma_a1) / shape_i;
                log_panel(&mut acc, s_lo, s_hi, n_points);
            } else {
                // Breakpoint below which the CRPS kernels are ~constant.
                let sb = (shape_i.min(1.0).ln() - 4.0).max((1e-280f64).ln() / shape_i.max(1e-3));
                // Left-tail mass under the asymptotic CDF (closed form).
                let p_b = (shape_i * sb - ln_gamma_a1).exp();
                let n1 = n_points / 8;
                // Asymptotic-inverse-CDF panel: u uniform on (0, p_b),
                // w(u) = exp((ln u + lnΓ(1+α))/α), weight = e^{−w}·Δu.
                let du = p_b / n1 as f64;
                for j in 0..n1 {
                    let u = du * (j as f64 + 0.5);
                    let w_node = ((u.ln() + ln_gamma_a1) / shape_i).exp();
                    let w = (-w_node).exp() * du;
                    accumulate(&mut acc, w, grads_at(w_node / rate_i));
                }
                log_panel(&mut acc, sb, s_hi, n_points - n1);
            }

            for a in 0..2 {
                for b in 0..2 {
                    fi[[i, a, b]] = acc[a][b];
                }
            }
        }

        fi
    }
}

impl Gamma {
    /// Helper function to compute CRPS for a single observation.
    fn crps_single(&self, y: f64, shape: f64, rate: f64) -> f64 {
        // CDF of Gamma(shape, rate) at y
        let f_y = if let Ok(d) = GammaDist::new(shape, rate) {
            d.cdf(y)
        } else {
            0.5
        };

        // CDF of Gamma(shape+1, rate) at y
        let f_alpha1_y = if let Ok(d) = GammaDist::new(shape + 1.0, rate) {
            d.cdf(y)
        } else {
            0.5
        };

        let beta_term = beta(0.5, shape);
        let mean = shape / rate;

        y * (2.0 * f_y - 1.0) - mean * (2.0 * f_alpha1_y - 1.0) - 1.0 / (rate * beta_term)
    }
}

/// Trigamma function (second derivative of log gamma).
/// Also used by StudentT's closed-form Fisher (`dist::studentt`).
pub(crate) fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // Use recurrence relation trigamma(x) = trigamma(x+1) + 1/x^2
    // to shift argument to > 10 for asymptotic expansion accuracy
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion: 1/x + 1/2x^2 + 1/6x^3 - 1/30x^5 + 1/42x^7
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x2 * x3;
    let x7 = x2 * x5;

    result += 1.0 / x + 0.5 / x2 + 1.0 / (6.0 * x3) - 1.0 / (30.0 * x5) + 1.0 / (42.0 * x7);

    result
}

/// Beta function B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)
fn beta(a: f64, b: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// CRPS metric E[ggᵀ] pinned to 2M-point PIT-quadrature references
    /// (scipy, dumped 2026-07-09). Covers the hybrid small-shape branch
    /// (α<1: asymptotic-inverse-CDF tail + log panel) and the log-panel
    /// branch (α≥1). New-rule validation errors were ≤2.3e-5 relative.
    #[test]
    fn test_gamma_crps_metric_matches_reference() {
        // (shape, rate, m00, m11, m01)
        let cases = [
            (0.05, 2.5, 4.452588158e-5, 1.999268456e-5, -2.866218337e-5),
            (0.3, 2.5, 4.226957972e-3, 2.557937774e-3, -3.194849769e-3),
            (3.0, 2.5, 4.848041975e-1, 4.460508116e-1, -4.623662583e-1),
        ];
        for &(shape, rate, m00, m11, m01) in &cases {
            let params = Array2::from_shape_vec(
                (1, 2),
                vec![(shape as f64).ln(), (rate as f64).ln()],
            )
            .unwrap();
            let d = Gamma::from_params(&params);
            let fi = Scorable::<CRPScore>::metric(&d);
            assert_relative_eq!(fi[[0, 0, 0]], m00, max_relative = 1e-3);
            assert_relative_eq!(fi[[0, 1, 1]], m11, max_relative = 1e-3);
            assert_relative_eq!(fi[[0, 0, 1]], m01, max_relative = 1e-3);
            assert_relative_eq!(fi[[0, 1, 0]], m01, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_gamma_distribution_methods() {
        // shape=2, rate=1 -> mean=2, var=2
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Gamma::from_params(&params);

        // Test mean: shape / rate = 2
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 2.0, epsilon = 1e-10);

        // Test variance: shape / rate^2 = 2
        let var = dist.variance();
        assert_relative_eq!(var[0], 2.0, epsilon = 1e-10);

        // Test mode: (shape - 1) / rate = 1
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_cdf_ppf() {
        let params = Array2::from_shape_vec((1, 2), vec![1.0_f64.ln(), 0.0]).unwrap();
        let dist = Gamma::from_params(&params);

        // For shape=1, rate=1, Gamma is Exponential(1)
        // CDF at 1 should be 1 - exp(-1) ≈ 0.632
        let y = Array1::from_vec(vec![1.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 1.0 - (-1.0_f64).exp(), epsilon = 1e-6);

        // PPF inverse test
        let q = Array1::from_vec(vec![0.5]);
        let ppf = dist.ppf(&q);
        let cdf_of_ppf = dist.cdf(&ppf);
        assert_relative_eq!(cdf_of_ppf[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_gamma_sample() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.5_f64.ln()]).unwrap();
        let dist = Gamma::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be non-negative
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Check that sample mean is close to shape/rate = 2/0.5 = 4
        let sample_mean: f64 = samples.column(0).mean().unwrap();
        assert!((sample_mean - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_gamma_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Gamma::fit(&y);
        assert_eq!(params.len(), 2);
        // Should return log(shape) and log(rate)
    }

    #[test]
    fn test_gamma_logscore() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Gamma::from_params(&params);

        let y = Array1::from_vec(vec![2.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score should be finite and positive
        assert!(score[0].is_finite());
        assert!(score[0] > 0.0);
    }

    #[test]
    fn test_gamma_crps() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Gamma::from_params(&params);

        let y = Array1::from_vec(vec![2.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS should be finite and non-negative
        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);
    }

    #[test]
    fn test_gamma_crps_d_score() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Gamma::from_params(&params);

        let y = Array1::from_vec(vec![2.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradients should be finite
        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
    }

    #[test]
    fn test_trigamma() {
        // trigamma(1) = pi^2 / 6 ≈ 1.6449
        assert_relative_eq!(
            trigamma(1.0),
            std::f64::consts::PI.powi(2) / 6.0,
            epsilon = 1e-6
        );

        // trigamma(2) = pi^2 / 6 - 1 ≈ 0.6449
        assert_relative_eq!(
            trigamma(2.0),
            std::f64::consts::PI.powi(2) / 6.0 - 1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_beta_function() {
        // Beta(1, 1) = 1
        assert_relative_eq!(beta(1.0, 1.0), 1.0, epsilon = 1e-10);

        // Beta(0.5, 0.5) = pi
        assert_relative_eq!(beta(0.5, 0.5), std::f64::consts::PI, epsilon = 1e-10);
    }
}
