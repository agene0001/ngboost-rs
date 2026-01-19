use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;

/// The Laplace distribution.
#[derive(Debug, Clone)]
pub struct Laplace {
    /// The location parameter (mean/median).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Laplace {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        Laplace {
            loc,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // For Laplace, the MLE for loc is the median and scale is mean absolute deviation
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }

        // Compute median
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Scale is mean absolute deviation from median
        let mad: f64 = y.iter().map(|&x| (x - median).abs()).sum::<f64>() / n as f64;

        array![median, mad.max(1e-6).ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Laplace is loc
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Laplace {}

impl DistributionMethods for Laplace {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of Laplace is 2 * scale^2
        2.0 * &self.scale * &self.scale
    }

    fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // pdf = 1/(2*scale) * exp(-|y - loc| / scale)
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            result[i] = (-abs_diff / self.scale[i]).exp() / (2.0 * self.scale[i]);
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // log(pdf) = -|y - loc| / scale - log(2 * scale)
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            result[i] = -abs_diff / self.scale[i] - (2.0 * self.scale[i]).ln();
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // CDF for Laplace:
        // if y < loc: 0.5 * exp((y - loc) / scale)
        // if y >= loc: 1 - 0.5 * exp(-(y - loc) / scale)
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let diff = y[i] - self.loc[i];
            if diff < 0.0 {
                result[i] = 0.5 * (diff / self.scale[i]).exp();
            } else {
                result[i] = 1.0 - 0.5 * (-diff / self.scale[i]).exp();
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Inverse CDF (quantile function) for Laplace:
        // if q < 0.5: loc + scale * ln(2*q)
        // if q >= 0.5: loc - scale * ln(2*(1-q))
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
            if q_clamped < 0.5 {
                result[i] = self.loc[i] + self.scale[i] * (2.0 * q_clamped).ln();
            } else {
                result[i] = self.loc[i] - self.scale[i] * (2.0 * (1.0 - q_clamped)).ln();
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            for s in 0..n_samples {
                // Use inverse CDF method
                let u: f64 = rng.random();
                if u < 0.5 {
                    samples[[s, i]] = self.loc[i] + self.scale[i] * (2.0 * u).ln();
                } else {
                    samples[[s, i]] = self.loc[i] - self.scale[i] * (2.0 * (1.0 - u)).ln();
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of Laplace is loc
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Laplace is loc
        self.loc.clone()
    }
}

impl Scorable<LogScore> for Laplace {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y) = |y - loc| / scale + log(2 * scale)
        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            scores[i] = abs_diff / self.scale[i] + (2.0 * self.scale[i]).ln();
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let diff = self.loc[i] - y[i];
            // d/d(loc) = sign(loc - y) / scale
            d_params[[i, 0]] = diff.signum() / self.scale[i];
            // d/d(log(scale)) = 1 - |loc - y| / scale
            d_params[[i, 1]] = 1.0 - diff.abs() / self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Laplace
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let scale_sq = self.scale[i] * self.scale[i];
            fi[[i, 0, 0]] = 1.0 / scale_sq;
            fi[[i, 1, 1]] = 1.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for Laplace {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Laplace distribution:
        // CRPS(F, y) = |y - loc| + scale * exp(-|y - loc|/scale) - 0.75 * scale
        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            let exp_term = (-abs_diff / self.scale[i]).exp();
            scores[i] = abs_diff + self.scale[i] * exp_term - 0.75 * self.scale[i];
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let diff = self.loc[i] - y[i];
            let abs_diff = diff.abs();
            let exp_term = (-abs_diff / self.scale[i]).exp();

            // d/d(loc)
            d_params[[i, 0]] = diff.signum() * (1.0 - exp_term);

            // d/d(log(scale)) - multiply by scale due to chain rule for log(scale)
            d_params[[i, 1]] = exp_term * (self.scale[i] + abs_diff) - 0.75 * self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Laplace
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 0.5 / self.scale[i];
            fi[[i, 1, 1]] = 0.25 * self.scale[i];
        }

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_laplace_distribution_methods() {
        // params: [loc, log(scale)]
        // First obs: loc=0, scale=1 (log(scale)=0)
        // Second obs: loc=5, scale=2 (log(scale)=ln(2))
        let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 2.0_f64.ln()]).unwrap();
        let dist = Laplace::from_params(&params);

        // Test mean
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 5.0, epsilon = 1e-10);

        // Test variance: 2 * scale^2
        let var = dist.variance();
        assert_relative_eq!(var[0], 2.0, epsilon = 1e-10); // 2 * 1^2 = 2
        assert_relative_eq!(var[1], 8.0, epsilon = 1e-10); // 2 * 2^2 = 8

        // Test median = loc
        let median = dist.median();
        assert_relative_eq!(median[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(median[1], 5.0, epsilon = 1e-10);

        // Test mode = loc
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mode[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_cdf_ppf() {
        // Create distribution with 3 observations to test ppf inverse
        let params = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        // CDF at loc should be 0.5
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.5, epsilon = 1e-10);

        // PPF at 0.5 should be loc
        let q = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let ppf = dist.ppf(&q);
        assert_relative_eq!(ppf[0], 0.0, epsilon = 1e-10);

        // PPF inverse test
        let q = Array1::from_vec(vec![0.25, 0.5, 0.75]);
        let ppf = dist.ppf(&q);
        let cdf_of_ppf = dist.cdf(&ppf);
        assert_relative_eq!(cdf_of_ppf[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(cdf_of_ppf[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cdf_of_ppf[2], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_pdf() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        // PDF at loc should be 1/(2*scale) = 0.5 for scale=1
        let y = Array1::from_vec(vec![0.0]);
        let pdf = dist.pdf(&y);
        assert_relative_eq!(pdf[0], 0.5, epsilon = 1e-10);

        // logpdf should match
        let logpdf = dist.logpdf(&y);
        assert_relative_eq!(logpdf[0], 0.5_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_sample() {
        let params = Array2::from_shape_vec((1, 2), vec![5.0, 1.0_f64.ln()]).unwrap();
        let dist = Laplace::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // Check that sample mean is close to loc = 5
        let sample_mean: f64 = samples.column(0).mean().unwrap();
        assert!((sample_mean - 5.0).abs() < 0.3);

        // Check that sample median is close to loc = 5
        let mut sample_vec: Vec<f64> = samples.column(0).to_vec();
        sample_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sample_median = sample_vec[500];
        assert!((sample_median - 5.0).abs() < 0.3);
    }

    #[test]
    fn test_laplace_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Laplace::fit(&y);
        assert_eq!(params.len(), 2);
        // Median should be 3.0
        assert_relative_eq!(params[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_logscore() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score at loc should be log(2*scale) = log(2) for scale=1
        assert_relative_eq!(score[0], 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_crps() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS at loc = |0| + scale * exp(0) - 0.75*scale = 0 + 1 - 0.75 = 0.25 for scale=1
        assert_relative_eq!(score[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_crps_d_score() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradients should be finite
        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
    }

    #[test]
    fn test_laplace_interval() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        let (lower, upper) = dist.interval(0.1);
        assert!(lower[0] < 0.0);
        assert!(upper[0] > 0.0);
        // Should be symmetric around loc
        assert_relative_eq!(lower[0], -upper[0], epsilon = 1e-10);
    }

    #[test]
    fn test_laplace_survival_function() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Laplace::from_params(&params);

        // SF at loc should be 0.5
        let y = Array1::from_vec(vec![0.0]);
        let sf = dist.sf(&y);
        assert_relative_eq!(sf[0], 0.5, epsilon = 1e-10);

        // SF + CDF should equal 1
        let cdf = dist.cdf(&y);
        assert_relative_eq!(sf[0] + cdf[0], 1.0, epsilon = 1e-10);
    }
}
