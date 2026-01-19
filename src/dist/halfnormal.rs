use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal as NormalDist};

/// The Half-Normal distribution.
///
/// The Half-Normal distribution is a Normal distribution folded at zero,
/// with loc fixed at 0. It has one parameter: scale (sigma).
#[derive(Debug, Clone)]
pub struct HalfNormal {
    /// The scale parameter (sigma).
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for HalfNormal {
    fn from_params(params: &Array2<f64>) -> Self {
        let scale = params.column(0).mapv(f64::exp);
        HalfNormal {
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // For half-normal, MLE for scale is sqrt(mean(y^2))
        // Since E[X^2] = sigma^2 for half-normal
        let n = y.len();
        if n == 0 {
            return array![0.0];
        }

        let sum_sq: f64 = y.iter().map(|&x| x * x).sum();
        let scale = (sum_sq / n as f64).sqrt().max(1e-6);

        array![scale.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of half-normal is scale * sqrt(2/pi)
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        &self.scale * sqrt_2_over_pi
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for HalfNormal {}

impl DistributionMethods for HalfNormal {
    fn mean(&self) -> Array1<f64> {
        // Mean of half-normal is scale * sqrt(2/pi)
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        &self.scale * sqrt_2_over_pi
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of half-normal is scale^2 * (1 - 2/pi)
        let one_minus_2_over_pi = 1.0 - 2.0 / std::f64::consts::PI;
        &self.scale * &self.scale * one_minus_2_over_pi
    }

    fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // pdf(y) = sqrt(2/pi) / scale * exp(-y^2 / (2 * scale^2)) for y >= 0
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let mut result = Array1::zeros(y.len());

        for i in 0..y.len() {
            if y[i] >= 0.0 {
                let scale_sq = self.scale[i] * self.scale[i];
                result[i] =
                    sqrt_2_over_pi / self.scale[i] * (-y[i] * y[i] / (2.0 * scale_sq)).exp();
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // logpdf = log(sqrt(2/pi)) - log(scale) - y^2 / (2 * scale^2)
        let log_sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt().ln();
        let mut result = Array1::zeros(y.len());

        for i in 0..y.len() {
            if y[i] >= 0.0 {
                let scale_sq = self.scale[i] * self.scale[i];
                result[i] =
                    log_sqrt_2_over_pi - self.scale[i].ln() - (y[i] * y[i]) / (2.0 * scale_sq);
            } else {
                result[i] = f64::NEG_INFINITY;
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // CDF of half-normal: 2 * Phi(y/scale) - 1 for y >= 0
        // where Phi is the standard normal CDF
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(y.len());

        for i in 0..y.len() {
            if y[i] >= 0.0 {
                result[i] = 2.0 * std_normal.cdf(y[i] / self.scale[i]) - 1.0;
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Inverse CDF: scale * Phi^{-1}((1 + q) / 2)
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(q.len());

        for i in 0..q.len() {
            let q_clamped = q[i].clamp(0.0, 1.0 - 1e-15);
            result[i] = self.scale[i] * std_normal.inverse_cdf((1.0 + q_clamped) / 2.0);
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.scale.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let std_normal = NormalDist::new(0.0, 1.0).unwrap();
            for s in 0..n_samples {
                // Sample from normal and take absolute value
                let u: f64 = rng.random();
                let z = std_normal.inverse_cdf(u);
                samples[[s, i]] = self.scale[i] * z.abs();
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median = scale * Phi^{-1}(0.75) ≈ scale * 0.6745
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let median_factor = std_normal.inverse_cdf(0.75);
        &self.scale * median_factor
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of half-normal is 0
        Array1::zeros(self.scale.len())
    }
}

impl Scorable<LogScore> for HalfNormal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y) for half-normal
        // logpdf = log(sqrt(2/pi)) - log(scale) - y^2 / (2 * scale^2)
        // -logpdf = -log(sqrt(2/pi)) + log(scale) + y^2 / (2 * scale^2)
        let log_sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt().ln();
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let scale_sq = self.scale[i] * self.scale[i];
            scores[i] = -log_sqrt_2_over_pi + self.scale[i].ln() + (y[i] * y[i]) / (2.0 * scale_sq);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let scale_sq = self.scale[i] * self.scale[i];
            // d/d(log(scale)) = scale * d/d(scale)
            // d(-logpdf)/d(scale) = 1/scale - y^2/scale^3
            // d(-logpdf)/d(log(scale)) = 1 - y^2/scale^2
            d_params[[i, 0]] = 1.0 - (y[i] * y[i]) / scale_sq;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information for half-normal is 2 (constant)
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for HalfNormal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Half-Normal distribution
        // For a half-normal with scale σ and observation y >= 0:
        // CRPS = σ * [z * (2*Φ(z) - 1) + 2*φ(z) - (1/√π) * (2*Φ(z*√2) - 1)]
        // where z = y/σ, Φ is standard normal CDF, φ is standard normal PDF
        //
        // Simplified form:
        // CRPS = σ * [z * erf(z/√2) + √(2/π) * (exp(-z²/2) - 1) + z]
        //      = y * erf(y/(σ*√2)) + σ * √(2/π) * (exp(-y²/(2σ²)) - 1) + y
        //
        // Alternative form using standard normal:
        // CRPS = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]   for y >= 0

        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let sqrt_2 = std::f64::consts::SQRT_2;

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let y_i = y[i].max(0.0); // Half-normal only supports y >= 0
            let sigma = self.scale[i];
            let z = y_i / sigma;

            // CRPS for folded/half-normal:
            // CRPS = σ * [z*(2Φ(z) - 1) + 2φ(z)] - σ/√π * (2Φ(z√2) - 1) - σ/√π
            // Simplified: CRPS = σ * [z*(2Φ(z)-1) + 2φ(z) - (2Φ(z√2)-1)/√π - 1/√π]

            let phi_z = std_normal.pdf(z);
            let big_phi_z = std_normal.cdf(z);
            let big_phi_z_sqrt2 = std_normal.cdf(z * sqrt_2);

            // CRPS for half-normal
            scores[i] = sigma
                * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z
                    - (2.0 * big_phi_z_sqrt2 - 1.0) / sqrt_pi
                    - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of CRPS w.r.t. log(scale)
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let sqrt_2 = std::f64::consts::SQRT_2;

        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let y_i = y[i].max(0.0);
            let sigma = self.scale[i];
            let z = y_i / sigma;

            let phi_z = std_normal.pdf(z);
            let big_phi_z = std_normal.cdf(z);
            let phi_z_sqrt2 = std_normal.pdf(z * sqrt_2);
            let big_phi_z_sqrt2 = std_normal.cdf(z * sqrt_2);

            // d(CRPS)/d(σ) then multiply by σ to get d(CRPS)/d(log(σ))
            // Let S = CRPS/σ = z*(2Φ(z)-1) + 2φ(z) - (2Φ(z√2)-1)/√π - 1/√π
            // d(CRPS)/d(σ) = S + σ * dS/dσ
            // dz/dσ = -z/σ
            // dS/dσ = dS/dz * dz/dσ = -z/σ * dS/dz

            // dS/dz = (2Φ(z)-1) + z*2φ(z) + 2*(-z*φ(z)) - 2*√2*φ(z√2)/√π
            //       = (2Φ(z)-1) - 2√2*φ(z√2)/√π
            let ds_dz = 2.0 * big_phi_z - 1.0 - 2.0 * sqrt_2 * phi_z_sqrt2 / sqrt_pi;

            let s = z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z
                - (2.0 * big_phi_z_sqrt2 - 1.0) / sqrt_pi
                - 1.0 / sqrt_pi;

            // d(CRPS)/d(log(σ)) = σ * d(CRPS)/d(σ) = σ * (S - z * dS/dz)
            d_params[[i, 0]] = sigma * (s - z * ds_dz);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information matrix for CRPS
        // For half-normal, this is approximately constant
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        // The metric is E[∇S ∇S^T] which for half-normal CRPS
        // is approximately 2/(π) based on the variance structure
        let metric_val = 2.0 / std::f64::consts::PI;

        for i in 0..n_obs {
            fi[[i, 0, 0]] = metric_val;
        }

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_halfnormal_distribution_methods() {
        // First obs: scale=1 (log(scale)=0)
        // Second obs: scale=2 (log(scale)=ln(2))
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 2.0_f64.ln()]).unwrap();
        let dist = HalfNormal::from_params(&params);

        // Test mean: scale * sqrt(2/pi)
        let mean = dist.mean();
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        assert_relative_eq!(mean[0], sqrt_2_over_pi, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 2.0 * sqrt_2_over_pi, epsilon = 1e-10);

        // Test variance: scale^2 * (1 - 2/pi)
        let var = dist.variance();
        let one_minus_2_over_pi = 1.0 - 2.0 / std::f64::consts::PI;
        assert_relative_eq!(var[0], one_minus_2_over_pi, epsilon = 1e-10);
        assert_relative_eq!(var[1], 4.0 * one_minus_2_over_pi, epsilon = 1e-10);

        // Test mode is 0
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mode[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_halfnormal_cdf_ppf() {
        // Create 3 observations for ppf inverse test
        let params = Array2::from_shape_vec((3, 1), vec![0.0, 0.0, 0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        // CDF at 0 should be 0
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.0, epsilon = 1e-10);

        // PPF at 0 should be 0
        let q = Array1::from_vec(vec![0.0, 0.0, 0.0]);
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
    fn test_halfnormal_pdf() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        // PDF at 0 should be sqrt(2/pi) for scale=1
        let y = Array1::from_vec(vec![0.0]);
        let pdf = dist.pdf(&y);
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        assert_relative_eq!(pdf[0], sqrt_2_over_pi, epsilon = 1e-10);

        // PDF should be 0 for negative values
        let y_neg = Array1::from_vec(vec![-1.0]);
        let pdf_neg = dist.pdf(&y_neg);
        assert_relative_eq!(pdf_neg[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_halfnormal_sample() {
        // scale = 2 (log(scale) = ln(2))
        let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
        let dist = HalfNormal::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be non-negative
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Check that sample mean is close to scale * sqrt(2/pi) = 2 * sqrt(2/pi)
        let sample_mean: f64 = samples.column(0).iter().sum::<f64>() / samples.nrows() as f64;
        let expected_mean = 2.0 * (2.0 / std::f64::consts::PI).sqrt();
        assert!((sample_mean - expected_mean).abs() / expected_mean < 0.15);
    }

    #[test]
    fn test_halfnormal_fit() {
        let y = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
        let params = HalfNormal::fit(&y);
        assert_eq!(params.len(), 1);
        // log(scale) should be based on sqrt(mean(y^2))
    }

    #[test]
    fn test_halfnormal_logscore() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score should be finite and positive
        assert!(score[0].is_finite());
        assert!(score[0] > 0.0);
    }

    #[test]
    fn test_halfnormal_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<LogScore>::d_score(&dist, &y);

        // Gradient should be finite
        assert!(d_score[[0, 0]].is_finite());
    }

    #[test]
    fn test_halfnormal_median() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        // Median ≈ 0.6745 for scale=1
        let median = dist.median();
        assert_relative_eq!(median[0], 0.6744897501960817, epsilon = 1e-6);
    }

    #[test]
    fn test_halfnormal_survival_function() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        // SF at 0 should be 1
        let y = Array1::from_vec(vec![0.0]);
        let sf = dist.sf(&y);
        assert_relative_eq!(sf[0], 1.0, epsilon = 1e-10);

        // SF + CDF should equal 1
        let y = Array1::from_vec(vec![1.0]);
        let sf = dist.sf(&y);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(sf[0] + cdf[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_halfnormal_crpscore() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(); // scale = 1
        let dist = HalfNormal::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS should be finite and non-negative
        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);

        // CRPS at y=0 should be σ * (2φ(0) - 1/√π - 1/√π) = σ * (2/√(2π) - 2/√π)
        let y_zero = Array1::from_vec(vec![0.0]);
        let score_zero = Scorable::<CRPScore>::score(&dist, &y_zero);
        assert!(score_zero[0].is_finite());
        assert!(score_zero[0] >= 0.0);
    }

    #[test]
    fn test_halfnormal_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradient should be finite
        assert!(d_score[[0, 0]].is_finite());
    }

    #[test]
    fn test_halfnormal_crpscore_metric() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = HalfNormal::from_params(&params);

        let metric = Scorable::<CRPScore>::metric(&dist);

        // Metric should be positive definite
        assert!(metric[[0, 0, 0]] > 0.0);
    }

    #[test]
    fn test_halfnormal_crpscore_scaling() {
        // CRPS should scale linearly with sigma
        let params1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(); // scale = 1
        let params2 = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap(); // scale = 2
        let dist1 = HalfNormal::from_params(&params1);
        let dist2 = HalfNormal::from_params(&params2);

        // For y=0, CRPS(σ) = σ * constant, so CRPS(2σ) / CRPS(σ) ≈ 2
        let y = Array1::from_vec(vec![0.0]);
        let score1 = Scorable::<CRPScore>::score(&dist1, &y);
        let score2 = Scorable::<CRPScore>::score(&dist2, &y);

        assert_relative_eq!(score2[0] / score1[0], 2.0, epsilon = 1e-10);
    }
}
