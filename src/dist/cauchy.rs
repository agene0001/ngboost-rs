use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
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
}

/// Fixed df=1 for Cauchy (T distribution with df=1)
const CAUCHY_DF: f64 = 1.0;

impl Distribution for Cauchy {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        let var = &scale * &scale;
        Cauchy { loc, scale, var }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // MLE for Cauchy (T with df=1) using IRLS (iteratively reweighted least squares).
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }
        let nf = n as f64;
        let df = 1.0_f64; // Cauchy = T(df=1)

        // Initial estimates: median for loc, MAD for scale
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut loc = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let mut scale = {
            let mut abs_devs: Vec<f64> = y.iter().map(|&v| (v - loc).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = if n % 2 == 0 {
                (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
            } else {
                abs_devs[n / 2]
            };
            (mad * 1.4826).max(1e-6) // MAD to std conversion factor
        };

        // IRLS iterations for t(df=3) MLE
        for _ in 0..100 {
            // E-step: compute weights w_i = (df + 1) / (df + ((y_i - loc) / scale)^2)
            let weights: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let z = (yi - loc) / scale;
                    (df + 1.0) / (df + z * z)
                })
                .collect();
            let sum_w: f64 = weights.iter().sum();

            // M-step: weighted mean for loc
            let new_loc: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * yi)
                .sum::<f64>()
                / sum_w;

            // M-step: weighted scale
            let new_scale_sq: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * (yi - new_loc).powi(2))
                .sum::<f64>()
                / nf;

            let new_scale = new_scale_sq.sqrt().max(1e-6);

            if (new_loc - loc).abs() < 1e-10 && (new_scale - scale).abs() < 1e-10 {
                loc = new_loc;
                scale = new_scale;
                break;
            }
            loc = new_loc;
            scale = new_scale;
        }

        array![loc, scale.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Return median (Cauchy has no mean)
        self.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.loc);
        p.column_mut(1).assign(&self.scale.mapv(f64::ln));
        p
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
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Fisher Information diagonal for Cauchy (T with df=1), 2 params: loc, log_scale
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.var)
            .for_each(|mut row, &var| {
                row[0] = (CAUCHY_DF + 1.0) / ((CAUCHY_DF + 3.0) * var);
                row[1] = 2.0 * CAUCHY_DF / (CAUCHY_DF + 3.0);
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -ln_pdf for Cauchy: -ln(1/(π*σ*(1+z²))) = ln(π) + ln(σ) + ln(1+z²)
        let mut scores = Array1::zeros(y.len());
        const LN_PI: f64 = 1.1447298858494002; // ln(π)
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                *s = LN_PI + scale.ln() + (1.0 + z * z).ln();
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Same as TFixedDf with df=1
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.var)
            .for_each(|mut row, &y_i, &loc, &var| {
                let diff = y_i - loc;
                let diff_sq = diff * diff;
                let denom = CAUCHY_DF * var + diff_sq;
                row[0] = -(CAUCHY_DF + 1.0) * diff / denom;
                row[1] = 1.0 - (CAUCHY_DF + 1.0) * diff_sq / denom;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Cauchy (T with df=1)
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        Zip::from(fi.outer_iter_mut())
            .and(&self.var)
            .for_each(|mut fi_i, &var| {
                fi_i[[0, 0]] = (CAUCHY_DF + 1.0) / ((CAUCHY_DF + 3.0) * var);
                fi_i[[1, 1]] = 2.0 * CAUCHY_DF / (CAUCHY_DF + 3.0);
            });

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
}

impl Distribution for CauchyFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        CauchyFixedVar { loc, scale, var }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // MLE for CauchyFixedVar (T with df=1, scale=1) using IRLS.
        // Returns only the loc parameter (scale is fixed at 1.0).
        let n = y.len();
        if n == 0 {
            return array![0.0];
        }
        let df = 1.0_f64; // Cauchy = T(df=1)
        let scale = 1.0_f64; // fixed

        // Initial estimate: median
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut loc = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // IRLS iterations for t(df=3) MLE, loc only (scale fixed)
        for _ in 0..100 {
            let weights: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let z = (yi - loc) / scale;
                    (df + 1.0) / (df + z * z)
                })
                .collect();
            let sum_w: f64 = weights.iter().sum();

            let new_loc: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * yi)
                .sum::<f64>()
                / sum_w;

            if (new_loc - loc).abs() < 1e-10 {
                loc = new_loc;
                break;
            }
            loc = new_loc;
        }

        array![loc]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 1));
        p.column_mut(0).assign(&self.loc);
        p
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
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Constant FI for CauchyFixedVar: (df+1)/((df+3)*var) = 2/4 = 0.5 (1 param)
        Array2::from_elem((self.loc.len(), 1), 0.5)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -ln_pdf for Cauchy with fixed var (scale=1)
        // -ln(1/(π*(1+z²))) = ln(π) + ln(1+z²)
        const LN_PI: f64 = 1.1447298858494002;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .for_each(|s, &y_i, &loc| {
                let z = y_i - loc; // scale=1
                *s = LN_PI + (1.0 + z * z).ln();
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // d/d(loc) for Cauchy with fixed var=1
        // Simplified: -(df+1)*diff / (df*var + diff²) = -2*diff / (1 + diff²)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.loc)
            .for_each(|d, &y_i, &loc| {
                let diff = y_i - loc;
                let diff_sq = diff * diff;
                // (CAUCHY_DF+1) = 2, CAUCHY_DF*var = 1*1 = 1
                *d = -2.0 * diff / (1.0 + diff_sq);
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // var=1 so FI = (df+1)/((df+3)*var) = 2/4 = 0.5 for all observations
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| 0.5);
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
        // IRLS with df=3 should converge close to the median for symmetric data
        assert_relative_eq!(params[0], 3.0, epsilon = 0.1);
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

    #[test]
    fn test_cauchy_score_matches_library() {
        // Verify vectorized score matches statrs per-element computation
        let params = Array2::from_shape_vec((3, 2), vec![-1.0, 0.0, 0.0, 0.5, 2.0, -0.3]).unwrap();
        let dist = Cauchy::from_params(&params);
        let y = Array1::from_vec(vec![0.5, -0.3, 1.0]);

        let scores = Scorable::<LogScore>::score(&dist, &y);
        for i in 0..3 {
            let d = CauchyDist::new(dist.loc[i], dist.scale[i]).unwrap();
            assert_relative_eq!(scores[i], -d.ln_pdf(y[i]), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cauchy_fixedvar_score_matches_library() {
        // Verify vectorized score matches statrs per-element computation
        let params = Array2::from_shape_vec((3, 1), vec![-1.0, 0.0, 2.5]).unwrap();
        let dist = CauchyFixedVar::from_params(&params);
        let y = Array1::from_vec(vec![0.5, -0.3, 1.0]);

        let scores = Scorable::<LogScore>::score(&dist, &y);
        for i in 0..3 {
            let d = CauchyDist::new(dist.loc[i], dist.scale[i]).unwrap();
            assert_relative_eq!(scores[i], -d.ln_pdf(y[i]), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cauchy_fixedvar_d_score_numerical() {
        // Verify d_score via central finite differences
        let params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let dist = CauchyFixedVar::from_params(&params);
        let y = Array1::from_vec(vec![0.5]);

        let d_score = Scorable::<LogScore>::d_score(&dist, &y);

        let eps = 1e-6;
        let params_plus = Array2::from_shape_vec((1, 1), vec![1.0 + eps]).unwrap();
        let params_minus = Array2::from_shape_vec((1, 1), vec![1.0 - eps]).unwrap();
        let dist_plus = CauchyFixedVar::from_params(&params_plus);
        let dist_minus = CauchyFixedVar::from_params(&params_minus);
        let s_plus = Scorable::<LogScore>::score(&dist_plus, &y);
        let s_minus = Scorable::<LogScore>::score(&dist_minus, &y);
        let numerical = (s_plus[0] - s_minus[0]) / (2.0 * eps);

        assert_relative_eq!(d_score[[0, 0]], numerical, epsilon = 1e-4);
    }
}
