use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Gamma as GammaDist};
use statrs::function::gamma::digamma;

/// The Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    pub shape: Array1<f64>, // alpha
    pub rate: Array1<f64>,  // beta
    _params: Array2<f64>,
}

impl Distribution for Gamma {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = params.column(0).mapv(f64::exp);
        let rate = params.column(1).mapv(f64::exp);
        Gamma {
            shape,
            rate,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // This is a simplification, MLE for Gamma is complex.
        // Using method of moments.
        let mean = y.mean().unwrap_or(1.0);
        let var = y.var(0.0);
        let shape = mean * mean / var.max(1e-9);
        let scale = var / mean.max(1e-9);
        let rate: f64 = 1.0 / scale;
        array![shape.ln(), rate.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean is shape / rate
        &self.shape / &self.rate
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
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
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = GammaDist::new(self.shape[i], self.rate[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = GammaDist::new(self.shape[i], self.rate[i]) {
                result[i] = d.ln_pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = GammaDist::new(self.shape[i], self.rate[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            if let Ok(d) = GammaDist::new(self.shape[i], self.rate[i]) {
                let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
                result[i] = d.inverse_cdf(q_clamped);
            }
        }
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
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = GammaDist::new(self.shape[i], self.rate[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let shape_i = self.shape[i];
            let rate_i = self.rate[i];

            // d/d(log(shape))
            let d_log_shape = shape_i * (digamma(shape_i) - (y[i] * rate_i).max(1e-9).ln());
            d_params[[i, 0]] = d_log_shape;

            // d/d(log(rate))
            let d_log_rate = y[i] * rate_i - shape_i;
            d_params[[i, 1]] = d_log_rate;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let shape_i = self.shape[i];

            // We use our local helper function for trigamma
            fi[[i, 0, 0]] = shape_i * shape_i * trigamma(shape_i);
            fi[[i, 1, 1]] = shape_i;
            fi[[i, 0, 1]] = -shape_i;
            fi[[i, 1, 0]] = -shape_i;
        }

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

            // CRPS formula
            let mean = shape / rate;
            scores[i] = y_i * (2.0 * f_y - 1.0) - mean * (2.0 * f_alpha1_y - 1.0)
                + mean / (std::f64::consts::PI.sqrt() * beta_term);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Numerical gradient for CRPS (analytical form is complex)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-6;

        for i in 0..n_obs {
            let shape_i = self.shape[i];
            let rate_i = self.rate[i];
            let y_i = y[i];

            // Compute score at current params
            let score_center = self.crps_single(y_i, shape_i, rate_i);

            // Derivative w.r.t. log(shape) via finite difference
            let shape_plus = shape_i * (1.0 + eps);
            let score_shape_plus = self.crps_single(y_i, shape_plus, rate_i);
            d_params[[i, 0]] = (score_shape_plus - score_center) / (shape_i * eps);

            // Derivative w.r.t. log(rate) via finite difference
            let rate_plus = rate_i * (1.0 + eps);
            let score_rate_plus = self.crps_single(y_i, shape_i, rate_plus);
            d_params[[i, 1]] = (score_rate_plus - score_center) / (rate_i * eps);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Use identity matrix scaled by estimated variance as a simple metric
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            // Use a simple diagonal metric
            let mean = self.shape[i] / self.rate[i];
            fi[[i, 0, 0]] = mean;
            fi[[i, 1, 1]] = mean;
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

        y * (2.0 * f_y - 1.0) - mean * (2.0 * f_alpha1_y - 1.0)
            + mean / (std::f64::consts::PI.sqrt() * beta_term)
    }
}

/// Trigamma function (second derivative of log gamma).
fn trigamma(x: f64) -> f64 {
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
