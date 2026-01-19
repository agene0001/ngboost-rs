use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{ContinuousCDF, Exp};

/// The Exponential distribution.
#[derive(Debug, Clone)]
pub struct Exponential {
    /// The rate parameter (1/scale).
    pub rate: Array1<f64>,
    /// The scale parameter (1/rate).
    pub scale: Array1<f64>,
    _params: Array2<f64>,
}

impl Distribution for Exponential {
    fn from_params(params: &Array2<f64>) -> Self {
        // param = log(scale), scale = exp(param), rate = 1/scale = exp(-param)
        let scale = params.column(0).mapv(f64::exp);
        let rate = 1.0 / &scale;
        Exponential {
            rate,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean().unwrap_or(1.0);
        // mean = 1/rate = scale, so log(scale) = log(mean)
        array![mean.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        // Mean is 1/rate
        1.0 / &self.rate
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Exponential {}

impl DistributionMethods for Exponential {
    fn mean(&self) -> Array1<f64> {
        // Mean of exponential is 1/rate = scale
        self.scale.clone()
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of exponential is 1/rate^2 = scale^2
        &self.scale * &self.scale
    }

    fn std(&self) -> Array1<f64> {
        // Std of exponential is 1/rate = scale
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] >= 0.0 {
                // pdf = rate * exp(-rate * y)
                result[i] = self.rate[i] * (-self.rate[i] * y[i]).exp();
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] >= 0.0 {
                // log(pdf) = log(rate) - rate * y
                result[i] = self.rate[i].ln() - self.rate[i] * y[i];
            } else {
                result[i] = f64::NEG_INFINITY;
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] >= 0.0 {
                // cdf = 1 - exp(-rate * y)
                result[i] = 1.0 - (-self.rate[i] * y[i]).exp();
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            // ppf = -log(1 - q) / rate = -log(1 - q) * scale
            let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
            result[i] = -(1.0 - q_clamped).ln() / self.rate[i];
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.scale.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = Exp::new(self.rate[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of exponential is ln(2) / rate = ln(2) * scale
        self.scale.mapv(|s| s * std::f64::consts::LN_2)
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of exponential is 0
        Array1::zeros(self.scale.len())
    }
}

impl Scorable<LogScore> for Exponential {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -log_pdf(y) = -ln(rate) + rate * y = ln(scale) + y/scale
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            scores[i] = self.scale[i].ln() + y_i / self.scale[i];
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // d/d(log_scale) of (ln(scale) + y/scale)
        // = d/d(log_scale) ln(scale) + d/d(log_scale) (y/scale)
        // = 1 + y * d/d(log_scale) (1/scale)
        // = 1 + y * (-1/scale) = 1 - y/scale
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            d_params[[i, 0]] = 1.0 - y[i] / self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for Exponential {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Exponential distribution:
        // CRPS(F, y) = y + scale * (2 * exp(-y/scale) - 1.5)
        // where F is Exp(scale) and scale = 1/rate
        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let exp_term = (-y[i] / self.scale[i]).exp();
            scores[i] = y[i] + self.scale[i] * (2.0 * exp_term - 1.5);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // d/d(log_scale) of CRPS = d/d(log_scale) [y + scale * (2*exp(-y/scale) - 1.5)]
        // = d/d(log_scale) [scale * (2*exp(-y/scale) - 1.5)]
        // = scale * d/d(log_scale) [2*exp(-y/scale)] + (2*exp(-y/scale) - 1.5) * d/d(log_scale) scale
        // = scale * 2 * exp(-y/scale) * y/scale^2 * scale + (2*exp(-y/scale) - 1.5) * scale
        // = 2 * exp(-y/scale) * y/scale * scale + (2*exp(-y/scale) - 1.5) * scale
        // = 2 * exp(-y/scale) * (y + scale) - 1.5 * scale
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let exp_term = (-y[i] / self.scale[i]).exp();
            d_params[[i, 0]] = 2.0 * exp_term * (y[i] + self.scale[i]) - 1.5 * self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Exponential - use 0.5 * scale as in Python
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 0.5 * self.scale[i];
        }

        fi
    }
}

// ============================================================================
// Censored LogScore for survival analysis
// ============================================================================

impl CensoredScorable<LogScoreCensored> for Exponential {
    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        let eps = 1e-10;
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let t = y.time[i];
            let e = y.event[i];
            // Exponential distribution with rate = 1/scale
            let d = Exp::new(self.rate[i]).unwrap();

            if e {
                // Uncensored: -log(pdf(t)) = -ln(rate) + rate*t = ln(scale) + t/scale
                scores[i] = self.scale[i].ln() + t / self.scale[i];
            } else {
                // Censored: -log(1 - cdf(t)) = -log(exp(-rate*t)) = rate*t = t/scale
                let survival = 1.0 - d.cdf(t) + eps;
                scores[i] = -survival.ln();
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let t = y.time[i];
            let e = y.event[i];

            if e {
                // Uncensored: d/d(log_scale) of (ln(scale) + t/scale) = 1 - t/scale
                d_params[[i, 0]] = 1.0 - t / self.scale[i];
            } else {
                // Censored: d/d(log_scale) of (t/scale) = -t/scale * d(scale)/d(log_scale) / scale
                // = -t/scale * scale / scale = -t/scale = t/scale (with proper sign)
                d_params[[i, 0]] = t / self.scale[i];
            }
            // Negate to match Python's convention
            d_params[[i, 0]] = -d_params[[i, 0]];
        }
        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0;
        }

        fi
    }
}

// ============================================================================
// Censored CRPScore for survival analysis
// ============================================================================

impl CensoredScorable<CRPScoreCensored> for Exponential {
    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let t = y.time[i];
            let e = y.event[i];
            let exp_term = (-t / self.scale[i]).exp();

            // Base CRPS: t + scale * (2*exp(-t/scale) - 1.5)
            scores[i] = t + self.scale[i] * (2.0 * exp_term - 1.5);

            if e {
                // Uncensored: subtract 0.5 * scale * exp(-2*t/scale)
                let exp_2t = (-2.0 * t / self.scale[i]).exp();
                scores[i] -= 0.5 * self.scale[i] * exp_2t;
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let t = y.time[i];
            let e = y.event[i];
            let exp_term = (-t / self.scale[i]).exp();

            // Base derivative: 2*exp(-t/scale)*(t + scale) - 1.5*scale
            d_params[[i, 0]] = 2.0 * exp_term * (t + self.scale[i]) - 1.5 * self.scale[i];

            if e {
                // Uncensored: subtract derivative of 0.5*scale*exp(-2*t/scale)
                let exp_2t = (-2.0 * t / self.scale[i]).exp();
                d_params[[i, 0]] -= exp_2t * (0.5 * self.scale[i] - t);
            }
        }
        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 0.5 * self.scale[i];
        }

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_exponential_distribution_methods() {
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 1.0_f64.ln()]).unwrap();
        let dist = Exponential::from_params(&params);

        // Test mean: scale
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 1.0, epsilon = 1e-10);

        // Test variance: scale^2
        let var = dist.variance();
        assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[1], 1.0, epsilon = 1e-10);

        // Test mode is 0
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mode[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_cdf_ppf() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = Exponential::from_params(&params);

        // CDF at scale (mean) for exp(rate) should be 1 - 1/e ≈ 0.632
        let y = Array1::from_vec(vec![1.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 1.0 - (-1.0_f64).exp(), epsilon = 1e-10);

        // PPF inverse test
        let q = Array1::from_vec(vec![0.5]);
        let ppf = dist.ppf(&q);
        let cdf_of_ppf = dist.cdf(&ppf);
        assert_relative_eq!(cdf_of_ppf[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_sample() {
        let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
        let dist = Exponential::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be non-negative
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Check that sample mean is close to scale = 2
        let sample_mean: f64 = samples.column(0).mean().unwrap();
        assert!((sample_mean - 2.0).abs() < 0.3);
    }

    #[test]
    fn test_exponential_median() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = Exponential::from_params(&params);

        // Median = ln(2) * scale = ln(2) for scale = 1
        let median = dist.median();
        assert_relative_eq!(median[0], std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_fit() {
        let y = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
        let params = Exponential::fit(&y);
        assert_eq!(params.len(), 1);
        // Mean of y is 1.5, so log(scale) should be log(1.5)
        assert_relative_eq!(params[0], 1.5_f64.ln(), epsilon = 1e-10);
    }
}
