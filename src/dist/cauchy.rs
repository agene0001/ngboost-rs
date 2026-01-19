use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Cauchy as CauchyDist, Continuous, ContinuousCDF};

/// The Cauchy distribution.
///
/// The Cauchy distribution is equivalent to the Student's T distribution with df=1.
/// It has two parameters: loc (median) and log(scale).
///
/// Note: The Cauchy distribution has no defined mean or variance.
/// The `predict` method returns the median (loc).
#[derive(Debug, Clone)]
pub struct Cauchy {
    /// The location parameter (median).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The variance (scale^2) - used in gradient computations.
    pub var: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

/// Fixed df=1 for Cauchy (T distribution with df=1)
const CAUCHY_DF: f64 = 1.0;

impl Distribution for Cauchy {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        let var = &scale * &scale;
        Cauchy {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // For Cauchy, use median and interquartile range for robust estimation
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }

        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Median
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // IQR-based scale estimate (half the IQR for Cauchy)
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let iqr = sorted[q3_idx] - sorted[q1_idx];
        let scale = (iqr / 2.0).max(1e-6);

        array![median, scale.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Return median (Cauchy has no mean)
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Cauchy {}

impl DistributionMethods for Cauchy {
    fn mean(&self) -> Array1<f64> {
        // Cauchy has no defined mean, return NaN
        Array1::from_elem(self.loc.len(), f64::NAN)
    }

    fn variance(&self) -> Array1<f64> {
        // Cauchy has no defined variance, return NaN
        Array1::from_elem(self.loc.len(), f64::NAN)
    }

    fn std(&self) -> Array1<f64> {
        // Cauchy has no defined std, return NaN
        Array1::from_elem(self.loc.len(), f64::NAN)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = CauchyDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = CauchyDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.ln_pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = CauchyDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Inverse CDF for Cauchy: loc + scale * tan(pi * (q - 0.5))
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
            result[i] =
                self.loc[i] + self.scale[i] * (std::f64::consts::PI * (q_clamped - 0.5)).tan();
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            for s in 0..n_samples {
                // Use inverse CDF method: loc + scale * tan(pi * (u - 0.5))
                let u: f64 = rng.random();
                samples[[s, i]] =
                    self.loc[i] + self.scale[i] * (std::f64::consts::PI * (u - 0.5)).tan();
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of Cauchy is loc
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Cauchy is loc
        self.loc.clone()
    }
}

impl Scorable<LogScore> for Cauchy {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = CauchyDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Same as TFixedDf with df=1
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let loc_i = self.loc[i];
            let var_i = self.var[i];
            let y_i = y[i];

            let diff = y_i - loc_i;
            let diff_sq = diff * diff;
            let denom = CAUCHY_DF * var_i + diff_sq;

            // d/d(loc): -(df + 1) * (y - loc) / (df * var + (y - loc)^2)
            d_params[[i, 0]] = -(CAUCHY_DF + 1.0) * diff / denom;

            // d/d(log(scale)): 1 - (df + 1) * (y - loc)^2 / (df * var + (y - loc)^2)
            d_params[[i, 1]] = 1.0 - (CAUCHY_DF + 1.0) * diff_sq / denom;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Cauchy (T with df=1)
        // FI[0, 0] = (df + 1) / ((df + 3) * var) = 2 / (4 * var) = 0.5 / var
        // FI[1, 1] = df / (2 * (df + 3) * var) = 1 / (8 * var)
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let var_i = self.var[i];
            fi[[i, 0, 0]] = (CAUCHY_DF + 1.0) / ((CAUCHY_DF + 3.0) * var_i);
            fi[[i, 1, 1]] = CAUCHY_DF / (2.0 * (CAUCHY_DF + 3.0) * var_i);
        }

        fi
    }
}

/// The Cauchy distribution with fixed variance=1.
///
/// Has one parameter: loc (median).
#[derive(Debug, Clone)]
pub struct CauchyFixedVar {
    /// The location parameter (median).
    pub loc: Array1<f64>,
    /// The scale parameter (fixed at 1.0).
    pub scale: Array1<f64>,
    /// The variance (fixed at 1.0).
    pub var: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for CauchyFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        CauchyFixedVar {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Median estimation
        let n = y.len();
        if n == 0 {
            return array![0.0];
        }

        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        array![median]
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

impl RegressionDistn for CauchyFixedVar {}

impl DistributionMethods for CauchyFixedVar {
    fn mean(&self) -> Array1<f64> {
        Array1::from_elem(self.loc.len(), f64::NAN)
    }

    fn variance(&self) -> Array1<f64> {
        Array1::from_elem(self.loc.len(), f64::NAN)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = CauchyDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = CauchyDist::new(self.loc[i], self.scale[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
            result[i] =
                self.loc[i] + self.scale[i] * (std::f64::consts::PI * (q_clamped - 0.5)).tan();
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            for s in 0..n_samples {
                let u: f64 = rng.random();
                samples[[s, i]] =
                    self.loc[i] + self.scale[i] * (std::f64::consts::PI * (u - 0.5)).tan();
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        self.loc.clone()
    }
}

impl Scorable<LogScore> for CauchyFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = CauchyDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let loc_i = self.loc[i];
            let var_i = self.var[i];
            let y_i = y[i];

            let diff = y_i - loc_i;
            let diff_sq = diff * diff;

            // d/d(loc) for fixed var case
            let num = (CAUCHY_DF + 1.0) * (2.0 / (CAUCHY_DF * var_i)) * diff;
            let den = 2.0 * (1.0 + (1.0 / (CAUCHY_DF * var_i)) * diff_sq);
            d_params[[i, 0]] = -num / den;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            let var_i = self.var[i];
            fi[[i, 0, 0]] = (CAUCHY_DF + 1.0) / ((CAUCHY_DF + 3.0) * var_i);
        }

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cauchy_distribution_methods() {
        let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 1.0_f64.ln()]).unwrap();
        let dist = Cauchy::from_params(&params);

        // Mean and variance should be NaN for Cauchy
        assert!(dist.mean()[0].is_nan());
        assert!(dist.variance()[0].is_nan());

        // Median should be loc
        let median = dist.median();
        assert_relative_eq!(median[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(median[1], 5.0, epsilon = 1e-10);

        // Mode should be loc
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mode[1], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_cdf_ppf() {
        // Create 3 observations for ppf inverse test
        let params = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let dist = Cauchy::from_params(&params);

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
        assert_relative_eq!(cdf_of_ppf[0], 0.25, epsilon = 1e-6);
        assert_relative_eq!(cdf_of_ppf[1], 0.5, epsilon = 1e-6);
        assert_relative_eq!(cdf_of_ppf[2], 0.75, epsilon = 1e-6);
    }

    #[test]
    fn test_cauchy_pdf() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Cauchy::from_params(&params);

        // PDF at loc should be 1/(pi*scale) = 1/pi for scale=1
        let y = Array1::from_vec(vec![0.0]);
        let pdf = dist.pdf(&y);
        assert_relative_eq!(pdf[0], 1.0 / std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_sample() {
        let params = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
        let dist = Cauchy::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // Check that sample median is close to loc = 5
        // (mean is not defined for Cauchy, so we check median)
        let mut sample_vec: Vec<f64> = samples.column(0).to_vec();
        sample_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sample_median = sample_vec[500];
        assert!((sample_median - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_cauchy_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Cauchy::fit(&y);
        assert_eq!(params.len(), 2);
        // Median should be 3.0
        assert_relative_eq!(params[0], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_logscore() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Cauchy::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score at loc should be -ln(1/(pi*scale)) = ln(pi) for scale=1
        assert_relative_eq!(score[0], std::f64::consts::PI.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_fixed_var() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let dist = CauchyFixedVar::from_params(&params);

        assert_relative_eq!(dist.median()[0], 5.0, epsilon = 1e-10);
        assert_relative_eq!(dist.scale[0], 1.0, epsilon = 1e-10);

        let y = Array1::from_vec(vec![5.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cauchy_interval() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Cauchy::from_params(&params);

        let (lower, upper) = dist.interval(0.5);
        // For Cauchy with loc=0, scale=1, the 25% and 75% quantiles are -1 and 1
        assert_relative_eq!(lower[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(upper[0], 1.0, epsilon = 1e-10);
    }
}
