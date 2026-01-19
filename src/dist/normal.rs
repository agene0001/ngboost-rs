use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal as NormalDist};
use statrs::statistics::Statistics;

/// Minimum scale (standard deviation) to avoid numerical issues.
const MIN_SCALE: f64 = 1e-6;
/// Maximum scale to prevent overflow in variance calculations.
const MAX_SCALE: f64 = 1e6;

/// The Normal (Gaussian) distribution.
#[derive(Debug, Clone)]
pub struct Normal {
    /// The mean of the distribution (loc).
    pub loc: Array1<f64>,
    /// The standard deviation of the distribution (scale).
    pub scale: Array1<f64>,
    /// The variance of the distribution.
    pub var: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Normal {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        // Clamp scale to [MIN_SCALE, MAX_SCALE] for numerical stability
        let scale = params
            .column(1)
            .mapv(|p| f64::exp(p).clamp(MIN_SCALE, MAX_SCALE));
        let var = &scale * &scale;
        Normal {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean();
        let std_dev = if y.len() <= 1 {
            1.0 // Fallback when we can't compute std dev (matches scipy behavior)
        } else {
            y.std(0.0)
        };
        // The parameters are loc and log(scale)
        // Handle edge case where std_dev is 0 or very small - match scipy's robust behavior
        let safe_std_dev = if std_dev <= 0.0 { 1.0 } else { std_dev };
        array![mean, safe_std_dev.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Normal {}

impl DistributionMethods for Normal {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.ln_pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.inverse_cdf(q[i]);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn mode(&self) -> Array1<f64> {
        // For Normal, mode = mean
        self.loc.clone()
    }
}

impl Scorable<LogScore> for Normal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y) with enhanced numerical stability and uncertainty handling
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            // Handle edge cases to avoid NaN/Inf
            let safe_loc = if self.loc[i].is_finite() {
                self.loc[i]
            } else {
                0.0
            };
            let safe_scale = if self.scale[i] >= MIN_SCALE && self.scale[i].is_finite() {
                self.scale[i]
            } else {
                1.0
            };

            // Use the original scale for normal operation
            if let Ok(d) = NormalDist::new(safe_loc, safe_scale) {
                let pdf = d.ln_pdf(y_i);
                scores[i] = if pdf.is_finite() { -pdf } else { f64::MAX };
            } else {
                scores[i] = f64::MAX;
            }
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Derivative wrt loc and log(scale)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        let err = &self.loc - y;

        // d/d(loc)
        d_params.column_mut(0).assign(&(&err / &self.var));

        // d/d(log(scale))
        let term2 = (&err * &err) / &self.var;
        d_params.column_mut(1).assign(&(1.0 - term2));

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / self.var[i];
            fi[[i, 1, 1]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for Normal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Normal distribution
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);

            // d/d(loc)
            d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);

            // d/d(log(scale)) - need to compute score first
            let pdf_z = std_normal.pdf(z);
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let score_i = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            d_params[[i, 1]] = score_i + (y[i] - self.loc[i]) * d_params[[i, 0]];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Normal
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
            fi[[i, 1, 1]] = self.var[i];
        }

        // Scale by 1/(2*sqrt(pi))
        fi.mapv_inplace(|x| x / (2.0 * sqrt_pi));
        fi
    }
}

// ============================================================================
// NormalFixedVar - Normal distribution with fixed variance = 1
// ============================================================================

/// Normal distribution with variance fixed at 1.
///
/// Has one parameter: loc (mean).
#[derive(Debug, Clone)]
pub struct NormalFixedVar {
    /// The location parameter (mean).
    pub loc: Array1<f64>,
    /// The scale parameter (fixed at 1.0).
    pub scale: Array1<f64>,
    /// The variance (fixed at 1.0).
    pub var: Array1<f64>,
    /// The parameters of the distribution.
    _params: Array2<f64>,
}

impl Distribution for NormalFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        NormalFixedVar {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean();
        array![mean]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for NormalFixedVar {}

impl DistributionMethods for NormalFixedVar {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.pdf(y[i]);
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.cdf(y[i]);
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.inverse_cdf(q[i]);
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            for s in 0..n_samples {
                let u: f64 = rng.random();
                samples[[s, i]] = d.inverse_cdf(u);
            }
        }
        samples
    }
}

impl Scorable<LogScore> for NormalFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            // d/d(loc) = (loc - y) / var
            d_params[[i, 0]] = (self.loc[i] - y[i]) / self.var[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / self.var[i] + 1e-5;
        }

        fi
    }
}

impl Scorable<CRPScore> for NormalFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);
            d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0 / (2.0 * sqrt_pi);
        }

        fi
    }
}

// ============================================================================
// NormalFixedMean - Normal distribution with fixed mean = 0
// ============================================================================

/// Normal distribution with mean fixed at 0.
///
/// Has one parameter: log(scale).
#[derive(Debug, Clone)]
pub struct NormalFixedMean {
    /// The location parameter (fixed at 0.0).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The variance.
    pub var: Array1<f64>,
    /// The parameters of the distribution.
    _params: Array2<f64>,
}

impl Distribution for NormalFixedMean {
    fn from_params(params: &Array2<f64>) -> Self {
        let scale = params.column(0).mapv(f64::exp);
        let var = &scale * &scale;
        let n = scale.len();
        let loc = Array1::zeros(n);
        NormalFixedMean {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let std_dev = y.std(0.0).max(1e-6);
        array![std_dev.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for NormalFixedMean {}

impl DistributionMethods for NormalFixedMean {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.pdf(y[i]);
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.cdf(y[i]);
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            result[i] = d.inverse_cdf(q[i]);
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            for s in 0..n_samples {
                let u: f64 = rng.random();
                samples[[s, i]] = d.inverse_cdf(u);
            }
        }
        samples
    }
}

impl Scorable<LogScore> for NormalFixedMean {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let err = self.loc[i] - y[i];
            // d/d(log(scale)) = 1 - (loc - y)^2 / var
            d_params[[i, 0]] = 1.0 - (err * err) / self.var[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for NormalFixedMean {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            let score_i = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            let d_loc = -(2.0 * cdf_z - 1.0);
            d_params[[i, 0]] = score_i + (y[i] - self.loc[i]) * d_loc;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = self.var[i] / (2.0 * sqrt_pi);
        }

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_distribution_methods() {
        let params =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);

        // Test mean
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(mean[2], 2.0, epsilon = 1e-10);

        // Test variance
        let var = dist.variance();
        assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[2], 1.0, epsilon = 1e-10);

        // Test CDF
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cdf[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cdf[2], 0.5, epsilon = 1e-10);

        // Test PPF (inverse CDF)
        let q = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let ppf = dist.ppf(&q);
        assert_relative_eq!(ppf[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(ppf[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ppf[2], 2.0, epsilon = 1e-10);

        // Test interval
        let (lower, upper) = dist.interval(0.05);
        assert!(lower[0] < 0.0);
        assert!(upper[0] > 0.0);
    }

    #[test]
    fn test_normal_sample() {
        let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 2]);

        // Check that samples have approximately correct mean
        let sample_mean_0: f64 = samples.column(0).iter().sum::<f64>() / samples.nrows() as f64;
        let sample_mean_1: f64 = samples.column(1).iter().sum::<f64>() / samples.nrows() as f64;

        assert!((sample_mean_0 - 0.0).abs() < 0.2);
        assert!((sample_mean_1 - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_normal_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Normal::fit(&y);
        assert_eq!(params.len(), 2);
        assert_relative_eq!(params[0], 3.0, epsilon = 1e-10); // mean
    }

    #[test]
    fn test_normal_fixed_var_distribution_methods() {
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 5.0]).unwrap();
        let dist = NormalFixedVar::from_params(&params);

        assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.mean()[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(dist.variance()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dist.variance()[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_fixed_mean_distribution_methods() {
        // params = [log(scale)] for NormalFixedMean
        // First obs: log(scale) = 0, so scale = 1
        // Second obs: log(scale) = 1, so scale = e
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let dist = NormalFixedMean::from_params(&params);

        assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.mean()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.std()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dist.std()[1], std::f64::consts::E, epsilon = 1e-10);
    }
}
