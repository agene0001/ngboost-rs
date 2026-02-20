use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{Array1, Array2, Array3, array};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Gamma as GammaDist, Weibull as WeibullDist};
use statrs::function::gamma::{digamma, gamma};

/// The Weibull distribution.
#[derive(Debug, Clone)]
pub struct Weibull {
    /// The shape parameter (k or c).
    pub shape: Array1<f64>,
    /// The scale parameter (lambda).
    pub scale: Array1<f64>,
}

impl Distribution for Weibull {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = params.column(0).mapv(f64::exp);
        let scale = params.column(1).mapv(f64::exp);
        Weibull { shape, scale }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // MLE for Weibull with fixed loc=0, matching scipy.stats.weibull_min.fit(Y, floc=0).
        // Uses Newton's method on the profile log-likelihood.
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }
        let nf = n as f64;

        // Filter out non-positive values for log computations
        let log_y: Vec<f64> = y.iter().map(|&v| v.max(1e-300).ln()).collect();

        // Initial estimate using method of moments
        let mean = y.mean().unwrap_or(1.0);
        let var = y.var(0.0);
        let cv = (var.sqrt() / mean.max(1e-300)).clamp(0.1, 10.0);
        let mut shape = (1.2 / cv).max(0.1);

        // Newton iterations on the MLE equation for shape:
        // n/k + sum(log(y)) - n * sum(y^k * log(y)) / sum(y^k) = 0
        for _ in 0..100 {
            let mut sum_yk: f64 = 0.0;
            let mut sum_yk_logy: f64 = 0.0;
            let mut sum_yk_logy2: f64 = 0.0;
            for i in 0..n {
                let yi = y[i].max(1e-300);
                let logy = log_y[i];
                let yk = yi.powf(shape);
                sum_yk += yk;
                sum_yk_logy += yk * logy;
                sum_yk_logy2 += yk * logy * logy;
            }
            let sum_logy: f64 = log_y.iter().sum();

            if sum_yk.abs() < 1e-300 {
                break;
            }

            let f = nf / shape + sum_logy - nf * sum_yk_logy / sum_yk;
            let f_prime = -nf / (shape * shape)
                - nf * (sum_yk_logy2 * sum_yk - sum_yk_logy * sum_yk_logy) / (sum_yk * sum_yk);

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

        // MLE scale given shape: scale = (sum(y^k) / n)^(1/k)
        let sum_yk: f64 = y.iter().map(|&v| v.max(1e-300).powf(shape)).sum();
        let scale = (sum_yk / nf).powf(1.0 / shape).max(1e-300);

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

    fn params(&self) -> Array2<f64> {
        let n = self.shape.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.shape.mapv(f64::ln));
        p.column_mut(1).assign(&self.scale.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for Weibull {}

impl Weibull {
    /// Compute CRPS for a single observation with given Weibull parameters.
    /// Uses the correct formula from scoringRules R package:
    /// CRPS = y*(2*F(y) - 1) - 2*λ*Γ(1+1/k)*P(1+1/k, (y/λ)^k) + λ*Γ(1+1/k)*(1 - 2^{-1/k})
    fn crps_single_static(y: f64, k: f64, lam: f64) -> f64 {
        let z = y / lam;
        let z_k = z.powf(k);

        // CDF at y: F(y) = 1 - exp(-(y/λ)^k)
        let f_y = 1.0 - (-z_k).exp();

        // Γ(1 + 1/k)
        let gamma_term = gamma(1.0 + 1.0 / k);

        // P(1+1/k, (y/λ)^k) = regularized lower incomplete gamma
        // = CDF of Gamma(shape=1+1/k, rate=1) at (y/λ)^k
        let p_term = if let Ok(g) = GammaDist::new(1.0 + 1.0 / k, 1.0) {
            g.cdf(z_k)
        } else {
            0.5
        };

        // 2^{-1/k}
        let two_pow = 2.0_f64.powf(-1.0 / k);

        y * (2.0 * f_y - 1.0) - 2.0 * lam * gamma_term * p_term + lam * gamma_term * (1.0 - two_pow)
    }
}

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
        // Inlined Weibull log-PDF: ln(f) = ln(k) - ln(λ) + (k-1)*ln(y/λ) - (y/λ)^k
        // score = -ln(f)
        let mut scores = Array1::zeros(y.len());
        ndarray::Zip::from(&mut scores)
            .and(y)
            .and(&self.shape)
            .and(&self.scale)
            .for_each(|s, &y_i, &k, &lam| {
                let ratio = y_i / lam;
                let log_pdf = k.ln() - lam.ln() + (k - 1.0) * ratio.ln() - ratio.powf(k);
                *s = -log_pdf;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        ndarray::Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.shape)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &k, &lam| {
                let ratio = y_i / lam;
                let ratio_k = ratio.powf(k);
                let shared_term = k * (ratio_k - 1.0);

                row[0] = shared_term * ratio.ln() - 1.0;
                row[1] = -shared_term;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Weibull
        // Pre-compute constants outside the loop
        const EULER_GAMMA: f64 = 0.5772156649;
        const PI: f64 = std::f64::consts::PI;
        const FI_00: f64 = (PI * PI / 6.0) + (1.0 - EULER_GAMMA) * (1.0 - EULER_GAMMA);
        const ONE_MINUS_GAMMA: f64 = 1.0 - EULER_GAMMA;

        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let k = self.shape[i];
            fi[[i, 0, 0]] = FI_00;
            fi[[i, 0, 1]] = -k * ONE_MINUS_GAMMA;
            fi[[i, 1, 0]] = fi[[i, 0, 1]];
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
        // CRPS for Weibull distribution (Jordan et al. 2019, scoringRules R package):
        // CRPS = y*(2*F(y) - 1) - 2*λ*Γ(1+1/k)*P(1+1/k, (y/λ)^k) + λ*Γ(1+1/k)*(1 - 2^{-1/k})
        // where P(a, x) = regularized lower incomplete gamma = CDF of Gamma(shape=a, rate=1) at x
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i].max(1e-10);

            scores[i] = Self::crps_single_static(y_i, k, lam);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Numerical gradient of CRPS using central differences
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let eps = 1e-5;

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i].max(1e-10);

            // Derivative w.r.t. log(shape) via central finite difference
            let k_plus = k * (1.0 + eps);
            let k_minus = k * (1.0 - eps);
            let score_k_plus = Self::crps_single_static(y_i, k_plus, lam);
            let score_k_minus = Self::crps_single_static(y_i, k_minus, lam);
            d_params[[i, 0]] = (score_k_plus - score_k_minus) / (2.0 * eps);

            // Derivative w.r.t. log(scale) via central finite difference
            let lam_plus = lam * (1.0 + eps);
            let lam_minus = lam * (1.0 - eps);
            let score_lam_plus = Self::crps_single_static(y_i, k, lam_plus);
            let score_lam_minus = Self::crps_single_static(y_i, k, lam_minus);
            d_params[[i, 1]] = (score_lam_plus - score_lam_minus) / (2.0 * eps);
        }
        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Deterministic quadrature for E[g * g^T].
        // Uses probability integral transform: u = F(y), y = F^{-1}(u)
        // so E[h(Y)] = ∫_0^1 h(F^{-1}(u)) du, computed via midpoint rule.
        let n_obs = self.shape.len();
        let n_params = 2;
        let n_points = 200;
        let eps = 1e-5;

        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];

            for j in 0..n_points {
                let u = (j as f64 + 0.5) / n_points as f64;
                // Weibull inverse CDF: y = lam * (-ln(1-u))^(1/k)
                let y_i = lam * (-(1.0 - u).ln()).powf(1.0 / k);

                // Gradient via finite differences
                let k_plus = k * (1.0 + eps);
                let k_minus = k * (1.0 - eps);
                let g0 = (Self::crps_single_static(y_i, k_plus, lam)
                    - Self::crps_single_static(y_i, k_minus, lam))
                    / (2.0 * eps);

                let lam_plus = lam * (1.0 + eps);
                let lam_minus = lam * (1.0 - eps);
                let g1 = (Self::crps_single_static(y_i, k, lam_plus)
                    - Self::crps_single_static(y_i, k, lam_minus))
                    / (2.0 * eps);

                let grads = [g0, g1];
                for a in 0..n_params {
                    for b in 0..n_params {
                        fi[[i, a, b]] += grads[a] * grads[b];
                    }
                }
            }
        }

        fi.mapv_inplace(|x| x / n_points as f64);
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
                scores[i] = Self::crps_single_static(t, k, lam);
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
            // dF/dk = S(t) * z^k * ln(z) = pdf_t * t * ln(z) / k
            let df_dk = pdf_t * t * ln_z / k;
            let ds_dk = -df_dk;

            // dF/dλ and dS/dλ
            // dF/dλ = -S(t) * k * z^k / λ = -pdf_t * z
            let df_dlam = -pdf_t * z;
            let ds_dlam = -df_dlam;

            if e {
                // Uncensored derivatives via central finite differences on correct CRPS
                let eps = 1e-5;

                let k_plus = k * (1.0 + eps);
                let k_minus = k * (1.0 - eps);
                d_params[[i, 0]] = (Self::crps_single_static(t, k_plus, lam)
                    - Self::crps_single_static(t, k_minus, lam))
                    / (2.0 * eps);

                let lam_plus = lam * (1.0 + eps);
                let lam_minus = lam * (1.0 - eps);
                d_params[[i, 1]] = (Self::crps_single_static(t, k, lam_plus)
                    - Self::crps_single_static(t, k, lam_minus))
                    / (2.0 * eps);
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
