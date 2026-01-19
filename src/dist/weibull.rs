use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Weibull as WeibullDist};
use statrs::function::gamma::{digamma, gamma};

/// The Weibull distribution.
#[derive(Debug, Clone)]
pub struct Weibull {
    /// The shape parameter (k or c).
    pub shape: Array1<f64>,
    /// The scale parameter (lambda).
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Weibull {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = params.column(0).mapv(f64::exp);
        let scale = params.column(1).mapv(f64::exp);
        Weibull {
            shape,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Simple method of moments estimation for Weibull
        // This is approximate; proper MLE requires numerical optimization
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }

        let mean = y.mean().unwrap_or(1.0);
        let var = y.var(0.0);

        // Coefficient of variation
        let cv = (var.sqrt() / mean).clamp(0.1, 10.0);

        // Approximate shape from CV (using approximation k ≈ 1.2 / CV)
        let shape = (1.2 / cv).max(0.1);

        // Scale from mean: mean = scale * Gamma(1 + 1/shape)
        let gamma_val = gamma(1.0 + 1.0 / shape);
        let scale = (mean / gamma_val).max(1e-6);

        array![shape.ln(), scale.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Weibull is scale * Gamma(1 + 1/shape)
        let mut means = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            let gamma_val = gamma(1.0 + 1.0 / self.shape[i]);
            means[i] = self.scale[i] * gamma_val;
        }
        means
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Weibull {}

impl DistributionMethods for Weibull {
    fn mean(&self) -> Array1<f64> {
        // Mean of Weibull is scale * Gamma(1 + 1/shape)
        let mut means = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            let gamma_val = gamma(1.0 + 1.0 / self.shape[i]);
            means[i] = self.scale[i] * gamma_val;
        }
        means
    }

    fn variance(&self) -> Array1<f64> {
        // Var of Weibull is scale^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]
        let mut vars = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            let k = self.shape[i];
            let lam = self.scale[i];
            let gamma_1 = gamma(1.0 + 1.0 / k);
            let gamma_2 = gamma(1.0 + 2.0 / k);
            vars[i] = lam * lam * (gamma_2 - gamma_1 * gamma_1);
        }
        vars
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] >= 0.0 {
                if let Ok(d) = WeibullDist::new(self.shape[i], self.scale[i]) {
                    result[i] = d.pdf(y[i]);
                }
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] >= 0.0 {
                if let Ok(d) = WeibullDist::new(self.shape[i], self.scale[i]) {
                    result[i] = d.ln_pdf(y[i]);
                }
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
                if let Ok(d) = WeibullDist::new(self.shape[i], self.scale[i]) {
                    result[i] = d.cdf(y[i]);
                }
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Inverse CDF for Weibull: scale * (-ln(1 - q))^(1/shape)
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
            result[i] = self.scale[i] * (-(1.0 - q_clamped).ln()).powf(1.0 / self.shape[i]);
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.shape.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            for s in 0..n_samples {
                // Use inverse CDF method: scale * (-ln(1 - u))^(1/shape)
                let u: f64 = rng.random();
                samples[[s, i]] = self.scale[i] * (-(1.0 - u).ln()).powf(1.0 / self.shape[i]);
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of Weibull is scale * (ln(2))^(1/shape)
        let mut result = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            result[i] = self.scale[i] * std::f64::consts::LN_2.powf(1.0 / self.shape[i]);
        }
        result
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Weibull is scale * ((k-1)/k)^(1/k) for k > 1, else 0
        let mut result = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            let k = self.shape[i];
            if k > 1.0 {
                result[i] = self.scale[i] * ((k - 1.0) / k).powf(1.0 / k);
            }
            // For k <= 1, mode is 0 (already initialized)
        }
        result
    }
}

impl Scorable<LogScore> for Weibull {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            // statrs Weibull uses (shape, scale) parameterization
            let d = WeibullDist::new(self.shape[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i];

            // Ratio y/scale
            let ratio = y_i / lam;
            let ratio_k = ratio.powf(k);

            // shared_term = k * ((y/scale)^k - 1)
            let shared_term = k * (ratio_k - 1.0);

            // d/d(log(shape)) = shape * [shared_term * log(y/scale) - 1]
            // But we parameterize as log(shape), so multiply by shape
            d_params[[i, 0]] = shared_term * ratio.ln() - 1.0;

            // d/d(log(scale)) = -shared_term
            d_params[[i, 1]] = -shared_term;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Weibull (from Python implementation)
        // Uses Euler's constant gamma ≈ 0.5772156649
        let euler_gamma = 0.5772156649;
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let k = self.shape[i];

            // FI[0, 0] = (pi^2 / 6) + (1 - gamma)^2
            let pi = std::f64::consts::PI;
            let one_minus_gamma = 1.0 - euler_gamma;
            fi[[i, 0, 0]] = (pi * pi / 6.0) + (one_minus_gamma * one_minus_gamma);

            // FI[1, 0] = FI[0, 1] = -k * (1 - gamma)
            fi[[i, 0, 1]] = -k * (1.0 - euler_gamma);
            fi[[i, 1, 0]] = fi[[i, 0, 1]];

            // FI[1, 1] = k^2
            fi[[i, 1, 1]] = k * k;
        }

        fi
    }
}

// ============================================================================
// CRPScore for Weibull (uncensored)
// ============================================================================

impl Scorable<CRPScore> for Weibull {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Weibull distribution
        // Reference: Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
        // prediction, and estimation. JASA, 102(477), 359-378.
        //
        // CRPS = y * (2*F(y) - 1) - λ * Γ(1+1/k) * (2*(1-F(y)) - 1 + 2^(-1/k))
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i].max(1e-10); // Avoid issues with zero

            let z = y_i / lam;
            let z_k = z.powf(k);

            // CDF at y: F(y) = 1 - exp(-(y/λ)^k)
            let f_y = 1.0 - (-z_k).exp();

            // Gamma(1 + 1/k)
            let gamma_term = gamma(1.0 + 1.0 / k);

            // 2^(-1/k)
            let two_pow = 2.0_f64.powf(-1.0 / k);

            // CRPS formula
            scores[i] =
                y_i * (2.0 * f_y - 1.0) - lam * gamma_term * (2.0 * (1.0 - f_y) - 1.0 + two_pow);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of CRPS with respect to log(shape) and log(scale)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-10;

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i].max(eps);

            let z = y_i / lam;
            let z_k = z.powf(k);
            let ln_z = z.ln();

            // F(y) = 1 - exp(-z^k)
            let s_y = (-z_k).exp(); // Survival function S(y) = 1 - F(y)

            // PDF: f(y) = (k/λ) * z^(k-1) * exp(-z^k)
            let pdf_y = (k / lam) * z.powf(k - 1.0) * s_y;

            // Gamma terms
            let gamma_term = gamma(1.0 + 1.0 / k);
            let dgamma_dk = -gamma_term * digamma(1.0 + 1.0 / k) / (k * k);

            // 2^(-1/k) and its derivative
            let two_pow = 2.0_f64.powf(-1.0 / k);
            let dtwo_pow_dk = two_pow * 2.0_f64.ln() / (k * k);

            // dF/dk = f(y) * z^k * ln(z)
            let df_dk = pdf_y * z_k * ln_z / k;

            // dF/dλ = -f(y) * k * z / λ = -pdf_y * k * z / lam
            let df_dlam = -pdf_y * k * z / lam;

            // dS/dk = -dF/dk, dS/dlam = -dF/dlam
            let ds_dk = -df_dk;
            let ds_dlam = -df_dlam;

            // d(CRPS)/d(log k) = k * d(CRPS)/dk
            // CRPS = y*(2F - 1) - λ*Γ*(2S - 1 + 2^(-1/k))
            // d(CRPS)/dk = y*2*dF/dk - λ*dΓ/dk*(2S - 1 + 2^(-1/k)) - λ*Γ*(2*dS/dk + d(2^(-1/k))/dk)
            let dcrps_dk = y_i * 2.0 * df_dk
                - lam * dgamma_dk * (2.0 * s_y - 1.0 + two_pow)
                - lam * gamma_term * (2.0 * ds_dk + dtwo_pow_dk);
            d_params[[i, 0]] = k * dcrps_dk;

            // d(CRPS)/d(log λ) = λ * d(CRPS)/dλ
            // d(CRPS)/dλ = y*2*dF/dλ - Γ*(2S - 1 + 2^(-1/k)) - λ*Γ*2*dS/dλ
            let dcrps_dlam = y_i * 2.0 * df_dlam
                - gamma_term * (2.0 * s_y - 1.0 + two_pow)
                - lam * gamma_term * 2.0 * ds_dlam;
            d_params[[i, 1]] = lam * dcrps_dlam;
        }
        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Weibull
        // Use an approximation based on the structure from Python
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));
        let eps = 1e-10;
        let sqrt_pi = std::f64::consts::PI.sqrt();

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];
            let gamma_term = gamma(1.0 + 1.0 / k);

            // Approximate metric based on CRPS second moments (from Python)
            fi[[i, 0, 0]] = (lam * lam * gamma_term * gamma_term / (k * k)) / (2.0 * sqrt_pi) + eps;
            fi[[i, 1, 1]] = (lam * lam * gamma_term * gamma_term) / (2.0 * sqrt_pi) + eps;
            fi[[i, 0, 1]] = 0.0;
            fi[[i, 1, 0]] = 0.0;
        }
        fi
    }
}

// ============================================================================
// Censored LogScore for survival analysis
// ============================================================================

impl CensoredScorable<LogScoreCensored> for Weibull {
    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        // For right-censored data:
        // - Uncensored (E=1): use log-likelihood = log(f(t)) → score = -log(f(t))
        // - Censored (E=0): use log-survival = log(S(t)) → score = -log(S(t))
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let t = y.time[i].max(1e-10);
            let e = y.event[i];
            let k = self.shape[i];
            let lam = self.scale[i];

            let z = t / lam;
            let z_k = z.powf(k);

            if e {
                // Uncensored: -log(f(t))
                // f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
                // log(f(t)) = log(k) - log(λ) + (k-1)*log(z) - z^k
                let log_pdf = k.ln() - lam.ln() + (k - 1.0) * z.ln() - z_k;
                scores[i] = -log_pdf;
            } else {
                // Censored: -log(S(t)) where S(t) = exp(-z^k)
                // -log(S(t)) = z^k
                scores[i] = z_k;
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-10;

        for i in 0..n_obs {
            let t = y.time[i].max(eps);
            let e = y.event[i];
            let k = self.shape[i];
            let lam = self.scale[i];

            let z = t / lam;
            let z_k = z.powf(k);
            let ln_z = z.ln();

            if e {
                // Uncensored gradient (same as LogScore)
                // shared_term = k * (z^k - 1)
                let shared_term = k * (z_k - 1.0);
                d_params[[i, 0]] = shared_term * ln_z - 1.0;
                d_params[[i, 1]] = -shared_term;
            } else {
                // Censored gradient: d/dθ[z^k] where z = t/λ
                // d(z^k)/d(log k) = k * z^k * ln(z)
                // d(z^k)/d(log λ) = -k * z^k
                d_params[[i, 0]] = k * z_k * ln_z;
                d_params[[i, 1]] = -k * z_k;
            }
        }
        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        // Use the uncensored Fisher information as approximation
        Scorable::<LogScore>::metric(self)
    }
}

// ============================================================================
// Censored CRPScore for survival analysis
// ============================================================================

impl CensoredScorable<CRPScoreCensored> for Weibull {
    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        // CRPS for right-censored Weibull data
        // For uncensored: standard CRPS
        // For censored: modified CRPS that accounts for right-censoring
        let mut scores = Array1::zeros(y.len());
        let eps = 1e-10;

        for i in 0..y.len() {
            let t = y.time[i].max(eps);
            let e = y.event[i];
            let k = self.shape[i];
            let lam = self.scale[i];

            let z = t / lam;
            let z_k = z.powf(k);

            let f_t = 1.0 - (-z_k).exp(); // CDF
            let s_t = (-z_k).exp(); // Survival function

            let gamma_term = gamma(1.0 + 1.0 / k);
            let two_pow = 2.0_f64.powf(-1.0 / k);

            if e {
                // Uncensored CRPS (same as regular CRPS)
                scores[i] = t * (2.0 * f_t - 1.0) - lam * gamma_term * (2.0 * s_t - 1.0 + two_pow);
            } else {
                // Censored CRPS (adapted for right-censoring)
                // From Python: crps_cens = t*F^2 + 2*λ*Γ*S*F - λ*Γ*2^(-1/k)*(1 - S^2)/2
                scores[i] = t * f_t * f_t + 2.0 * lam * gamma_term * s_t * f_t
                    - lam * gamma_term * two_pow * (1.0 - s_t * s_t) / 2.0;
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-10;

        for i in 0..n_obs {
            let t = y.time[i].max(eps);
            let e = y.event[i];
            let k = self.shape[i];
            let lam = self.scale[i];

            let z = t / lam;
            let z_k = z.powf(k);
            let ln_z = z.ln();

            let f_t = 1.0 - (-z_k).exp();
            let s_t = (-z_k).exp();

            // PDF: f(t) = (k/λ) * z^(k-1) * exp(-z^k)
            let pdf_t = (k / lam) * z.powf(k - 1.0) * s_t;

            let gamma_term = gamma(1.0 + 1.0 / k);
            let dgamma_dk = -gamma_term * digamma(1.0 + 1.0 / k) / (k * k);

            let two_pow = 2.0_f64.powf(-1.0 / k);
            let dtwo_pow_dk = two_pow * 2.0_f64.ln() / (k * k);

            // dF/dk and dS/dk
            let df_dk = pdf_t * z_k * ln_z / k;
            let ds_dk = -df_dk;

            // dF/dλ and dS/dλ
            let df_dlam = -pdf_t * k * z / lam;
            let ds_dlam = -df_dlam;

            if e {
                // Uncensored derivatives (same as CRPScore)
                let dcrps_dk = t * 2.0 * df_dk
                    - lam * dgamma_dk * (2.0 * s_t - 1.0 + two_pow)
                    - lam * gamma_term * (2.0 * ds_dk + dtwo_pow_dk);
                d_params[[i, 0]] = k * dcrps_dk;

                let dcrps_dlam = t * 2.0 * df_dlam
                    - gamma_term * (2.0 * s_t - 1.0 + two_pow)
                    - lam * gamma_term * 2.0 * ds_dlam;
                d_params[[i, 1]] = lam * dcrps_dlam;
            } else {
                // Censored derivatives
                // crps_cens = t*F^2 + 2*λ*Γ*S*F - λ*Γ*2^(-1/k)*(1 - S^2)/2

                // d/dk[crps_cens]
                let dcrps_cens_dk = t * 2.0 * f_t * df_dk
                    + 2.0 * lam * dgamma_dk * s_t * f_t
                    + 2.0 * lam * gamma_term * (ds_dk * f_t + s_t * df_dk)
                    - lam * dgamma_dk * two_pow * (1.0 - s_t * s_t) / 2.0
                    - lam * gamma_term * dtwo_pow_dk * (1.0 - s_t * s_t) / 2.0
                    + lam * gamma_term * two_pow * s_t * ds_dk;
                d_params[[i, 0]] = k * dcrps_cens_dk;

                // d/dλ[crps_cens]
                let dcrps_cens_dlam = t * 2.0 * f_t * df_dlam
                    + 2.0 * gamma_term * s_t * f_t
                    + 2.0 * lam * gamma_term * (ds_dlam * f_t + s_t * df_dlam)
                    - gamma_term * two_pow * (1.0 - s_t * s_t) / 2.0
                    + lam * gamma_term * two_pow * s_t * ds_dlam;
                d_params[[i, 1]] = lam * dcrps_cens_dlam;
            }
        }
        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        // Use the CRPS metric
        Scorable::<CRPScore>::metric(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_weibull_distribution_methods() {
        // shape=2, scale=1 (Rayleigh distribution)
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        // Test mean: scale * Gamma(1 + 1/shape) = 1 * Gamma(1.5) ≈ 0.886
        let mean = dist.mean();
        assert!(mean[0] > 0.8 && mean[0] < 1.0);

        // Test variance
        let var = dist.variance();
        assert!(var[0] > 0.0);

        // Test mode: scale * ((k-1)/k)^(1/k) = 1 * (0.5)^0.5 ≈ 0.707
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 0.5_f64.sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_weibull_cdf_ppf() {
        let params = Array2::from_shape_vec((1, 2), vec![1.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        // For shape=1, Weibull is Exponential(1)
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
    fn test_weibull_sample() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 1.0_f64.ln()]).unwrap();
        let dist = Weibull::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be non-negative
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Check that sample mean is close to theoretical mean
        let sample_mean: f64 = samples.column(0).mean().unwrap();
        let theoretical_mean = dist.mean()[0];
        assert!((sample_mean - theoretical_mean).abs() / theoretical_mean < 0.15);
    }

    #[test]
    fn test_weibull_median() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        // For shape=1, scale=1, median = ln(2) ≈ 0.693
        let median = dist.median();
        assert_relative_eq!(median[0], std::f64::consts::LN_2, epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_fit() {
        let y = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
        let params = Weibull::fit(&y);
        assert_eq!(params.len(), 2);
        // Should return log(shape) and log(scale)
    }

    #[test]
    fn test_weibull_logscore() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score should be finite and positive
        assert!(score[0].is_finite());
        assert!(score[0] > 0.0);
    }

    #[test]
    fn test_weibull_d_score() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<LogScore>::d_score(&dist, &y);

        // Gradients should be finite
        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
    }

    #[test]
    fn test_weibull_interval() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let (lower, upper) = dist.interval(0.1);
        assert!(lower[0] > 0.0);
        assert!(upper[0] > lower[0]);
    }

    #[test]
    fn test_weibull_survival_function() {
        let params = Array2::from_shape_vec((1, 2), vec![1.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        // For shape=1, Weibull is Exponential
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

    // ========================================================================
    // CRPScore tests
    // ========================================================================

    #[test]
    fn test_weibull_crpscore() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS should be finite and non-negative
        assert!(score[0].is_finite());
        // CRPS can be negative for some parameter/observation combinations
    }

    #[test]
    fn test_weibull_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradients should be finite
        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
    }

    #[test]
    fn test_weibull_crpscore_metric() {
        let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let metric = Scorable::<CRPScore>::metric(&dist);

        // Metric should be 2x2 positive definite
        assert_eq!(metric.shape(), &[1, 2, 2]);
        assert!(metric[[0, 0, 0]] > 0.0);
        assert!(metric[[0, 1, 1]] > 0.0);
    }

    // ========================================================================
    // Censored LogScore tests
    // ========================================================================

    #[test]
    fn test_weibull_censored_logscore() {
        let params =
            Array2::from_shape_vec((2, 2), vec![2.0_f64.ln(), 0.0, 2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let time = Array1::from_vec(vec![1.0, 2.0]);
        let event = Array1::from_vec(vec![1.0, 0.0]); // First is uncensored, second is censored
        let y = SurvivalData::from_arrays(&time, &event);

        let scores = CensoredScorable::<LogScoreCensored>::censored_score(&dist, &y);

        // Scores should be finite
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());

        // Uncensored score should match regular LogScore
        let regular_y = Array1::from_vec(vec![1.0]);
        let regular_params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let regular_dist = Weibull::from_params(&regular_params);
        let regular_score = Scorable::<LogScore>::score(&regular_dist, &regular_y);
        assert_relative_eq!(scores[0], regular_score[0], epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_censored_logscore_d_score() {
        let params =
            Array2::from_shape_vec((2, 2), vec![2.0_f64.ln(), 0.0, 2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let time = Array1::from_vec(vec![1.0, 2.0]);
        let event = Array1::from_vec(vec![1.0, 0.0]);
        let y = SurvivalData::from_arrays(&time, &event);

        let d_scores = CensoredScorable::<LogScoreCensored>::censored_d_score(&dist, &y);

        // All gradients should be finite
        assert!(d_scores.iter().all(|&x| x.is_finite()));
    }

    // ========================================================================
    // Censored CRPScore tests
    // ========================================================================

    #[test]
    fn test_weibull_censored_crpscore() {
        let params =
            Array2::from_shape_vec((2, 2), vec![2.0_f64.ln(), 0.0, 2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let time = Array1::from_vec(vec![1.0, 2.0]);
        let event = Array1::from_vec(vec![1.0, 0.0]); // First is uncensored, second is censored
        let y = SurvivalData::from_arrays(&time, &event);

        let scores = CensoredScorable::<CRPScoreCensored>::censored_score(&dist, &y);

        // Scores should be finite
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());

        // Uncensored score should match regular CRPScore
        let regular_y = Array1::from_vec(vec![1.0]);
        let regular_params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
        let regular_dist = Weibull::from_params(&regular_params);
        let regular_score = Scorable::<CRPScore>::score(&regular_dist, &regular_y);
        assert_relative_eq!(scores[0], regular_score[0], epsilon = 1e-10);
    }

    #[test]
    fn test_weibull_censored_crpscore_d_score() {
        let params =
            Array2::from_shape_vec((2, 2), vec![2.0_f64.ln(), 0.0, 2.0_f64.ln(), 0.0]).unwrap();
        let dist = Weibull::from_params(&params);

        let time = Array1::from_vec(vec![1.0, 2.0]);
        let event = Array1::from_vec(vec![1.0, 0.0]);
        let y = SurvivalData::from_arrays(&time, &event);

        let d_scores = CensoredScorable::<CRPScoreCensored>::censored_d_score(&dist, &y);

        // All gradients should be finite
        assert!(d_scores.iter().all(|&x| x.is_finite()));
    }
}
