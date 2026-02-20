use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{Array1, Array2, Array3, Zip, array};
use rand::prelude::*;
use statrs::distribution::{
    Continuous, ContinuousCDF, LogNormal as LogNormalDist, Normal as NormalDist,
};

/// The LogNormal distribution.
#[derive(Debug, Clone)]
pub struct LogNormal {
    pub loc: Array1<f64>,
    pub scale: Array1<f64>,
}

impl Distribution for LogNormal {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        LogNormal { loc, scale }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        if y.is_empty() {
            return array![0.0, 0.0];
        }
        let log_y: Array1<f64> = y.mapv(|v| v.ln());
        let mean = log_y.mean().unwrap_or(0.0);
        let std_dev = log_y.std(0.0);
        array![mean, std_dev.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of lognormal is exp(loc + scale^2 / 2)
        (&self.loc + &(&self.scale.mapv(|s| s.powi(2)) / 2.0)).mapv(f64::exp)
    }

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.loc);
        p.column_mut(1).assign(&self.scale.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for LogNormal {}

impl DistributionMethods for LogNormal {
    fn mean(&self) -> Array1<f64> {
        // Mean of lognormal is exp(mu + sigma^2 / 2)
        (&self.loc + &(&self.scale.mapv(|s| s.powi(2)) / 2.0)).mapv(f64::exp)
    }

    fn variance(&self) -> Array1<f64> {
        // Var of lognormal is (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        let sigma_sq = self.scale.mapv(|s| s.powi(2));
        let exp_sigma_sq = sigma_sq.mapv(f64::exp);
        let two_mu_plus_sigma_sq = 2.0 * &self.loc + &sigma_sq;
        (&exp_sigma_sq - 1.0) * two_mu_plus_sigma_sq.mapv(f64::exp)
    }

    fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inline formula: f(x) = (1 / (x * sigma * sqrt(2*pi))) * exp(-((ln(x) - mu)^2) / (2 * sigma^2))
        const INV_SQRT_2PI: f64 = 0.3989422804014327; // 1 / sqrt(2 * pi)
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &mu, &sigma| {
                if y_i > 0.0 {
                    let ln_y = y_i.ln();
                    let z = (ln_y - mu) / sigma;
                    *r = INV_SQRT_2PI / (y_i * sigma) * (-0.5 * z * z).exp();
                }
            });
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inline formula: ln(f(x)) = -ln(x) - ln(sigma) - 0.5*ln(2*pi) - ((ln(x) - mu)^2) / (2 * sigma^2)
        const LN_SQRT_2PI: f64 = 0.9189385332046727; // 0.5 * ln(2 * pi)
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &mu, &sigma| {
                if y_i > 0.0 {
                    let ln_y = y_i.ln();
                    let z = (ln_y - mu) / sigma;
                    *r = -ln_y - sigma.ln() - LN_SQRT_2PI - 0.5 * z * z;
                } else {
                    *r = f64::NEG_INFINITY;
                }
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // LogNormal CDF: Φ((ln(x) - μ) / σ)
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &mu, &sigma| {
                if y_i > 0.0 {
                    let z = (y_i.ln() - mu) / sigma;
                    *r = std_normal.cdf(z);
                }
                // For y_i <= 0, result stays 0 (CDF of lognormal at 0 is 0)
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // PPF requires inverse error function - keep using statrs for accuracy
        let mut result = Array1::zeros(q.len());
        Zip::from(&mut result)
            .and(q)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &q_i, &loc, &scale| {
                if let Ok(d) = LogNormalDist::new(loc, scale) {
                    *r = d.inverse_cdf(q_i);
                }
            });
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = LogNormalDist::new(self.loc[i], self.scale[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of lognormal is exp(mu)
        self.loc.mapv(f64::exp)
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of lognormal is exp(mu - sigma^2)
        (&self.loc - &self.scale.mapv(|s| s.powi(2))).mapv(f64::exp)
    }
}

impl Scorable<LogScore> for LogNormal {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Diagonal of Fisher Information Matrix (2 params: loc, log_scale)
        let eps = 1e-5;
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 1.0 / (scale * scale) + eps;
                row[1] = 2.0;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -ln_pdf for LogNormal:
        // -ln_pdf(y) = ln(y) + 0.5*ln(2π) + ln(σ) + 0.5*((ln(y)-μ)/σ)²
        const HALF_LN_2PI: f64 = 0.9189385332046727;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &loc, &scale| {
                let log_y = y_i.max(1e-300).ln();
                let z = (log_y - loc) / scale;
                *s = log_y + HALF_LN_2PI + scale.ln() + 0.5 * z * z;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Vectorized gradient w.r.t. loc and log(scale)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &loc, &scale| {
                let log_y = y_i.ln();
                let var = scale * scale;
                let err = loc - log_y;
                row[0] = err / var;
                row[1] = 1.0 - (err * err) / var;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Vectorized Fisher Information Matrix (diagonal)
        let eps = 1e-5;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        Zip::from(fi.outer_iter_mut())
            .and(&self.scale)
            .for_each(|mut fi_i, &scale| {
                fi_i[[0, 0]] = 1.0 / (scale * scale) + eps;
                fi_i[[1, 1]] = 2.0;
            });

        fi
    }
}

impl Scorable<CRPScore> for LogNormal {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // CRPS metric diagonal (2 params: loc, log_scale)
        const INV_2_SQRT_PI: f64 = 0.28209479177387814;
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 2.0 * INV_2_SQRT_PI;
                row[1] = scale * scale * INV_2_SQRT_PI;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized CRPS for LogNormal
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &loc, &scale| {
                let log_y = y_i.ln();
                let z = (log_y - loc) / scale;
                let pdf_z = std_normal.pdf(z);
                let cdf_z = std_normal.cdf(z);
                *s = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI);
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Vectorized gradient of CRPS for LogNormal
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &loc, &scale| {
                let log_y = y_i.ln();
                let z = (log_y - loc) / scale;
                let cdf_z = std_normal.cdf(z);
                let pdf_z = std_normal.pdf(z);

                // d/d(loc)
                let d_loc = -(2.0 * cdf_z - 1.0);
                row[0] = d_loc;

                // d/d(log(scale))
                let score_i = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI);
                row[1] = score_i + (log_y - loc) * d_loc;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for LogNormal — vectorized
        const INV_2_SQRT_PI: f64 = 0.28209479177387814;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        Zip::from(fi.outer_iter_mut())
            .and(&self.scale)
            .for_each(|mut fi_i, &scale| {
                fi_i[[0, 0]] = 2.0 * INV_2_SQRT_PI;
                fi_i[[1, 1]] = scale * scale * INV_2_SQRT_PI;
            });

        fi
    }
}

// ============================================================================
// Censored LogScore for survival analysis
// ============================================================================

impl CensoredScorable<LogScoreCensored> for LogNormal {
    fn is_diagonal_censored_metric(&self) -> bool {
        true
    }

    fn diagonal_censored_metric(&self) -> Array2<f64> {
        let eps = 1e-5;
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 1.0 / (scale * scale) + eps;
                row[1] = 2.0;
            });
        diag
    }

    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        let eps = 1e-5;
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let t = y.time[i];
            let e = y.event[i];
            let d = LogNormalDist::new(self.loc[i], self.scale[i]).unwrap();

            if e {
                // Uncensored: -log(pdf(t))
                scores[i] = -d.ln_pdf(t);
            } else {
                // Censored: -log(1 - cdf(t))
                let survival = 1.0 - d.cdf(t) + eps;
                scores[i] = -survival.ln();
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let eps = 1e-5;
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let t = y.time[i];
            let e = y.event[i];
            let log_t = t.ln();
            let z = (log_t - self.loc[i]) / self.scale[i];
            let var = self.scale[i].powi(2);
            let d = LogNormalDist::new(self.loc[i], self.scale[i]).unwrap();

            if e {
                // Uncensored gradient (same as regular LogScore)
                d_params[[i, 0]] = (self.loc[i] - log_t) / var;
                d_params[[i, 1]] = 1.0 - ((self.loc[i] - log_t).powi(2)) / var;
            } else {
                // Censored gradient
                let survival = 1.0 - d.cdf(t) + eps;
                let norm_pdf = std_normal.pdf(z);

                d_params[[i, 0]] = -norm_pdf / (self.scale[i] * survival);
                d_params[[i, 1]] = -z * norm_pdf / survival;
            }
        }
        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        // Use the same metric as uncensored LogScore (Fisher Information)
        let eps = 1e-5;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));
        let var = self.scale.mapv(|s| s.powi(2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / var[i] + eps;
            fi[[i, 1, 1]] = 2.0;
        }

        fi
    }
}

// ============================================================================
// Censored CRPScore for survival analysis
// ============================================================================

impl CensoredScorable<CRPScoreCensored> for LogNormal {
    fn is_diagonal_censored_metric(&self) -> bool {
        true
    }

    fn diagonal_censored_metric(&self) -> Array2<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let inv_2_sqrt_pi = 1.0 / (2.0 * sqrt_pi);
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 2.0 * inv_2_sqrt_pi;
                row[1] = scale * scale * inv_2_sqrt_pi;
            });
        diag
    }

    fn censored_score(&self, y: &SurvivalData) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let sqrt_2 = 2.0_f64.sqrt();

        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let t = y.time[i];
            let e = y.event[i];
            let log_t = t.ln();
            let z = (log_t - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);
            let pdf_z = std_normal.pdf(z);

            if e {
                // Uncensored CRPS (same as regular)
                scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            } else {
                // Censored CRPS
                let cdf_sqrt2_z = std_normal.cdf(sqrt_2 * z);
                scores[i] = self.scale[i]
                    * (z * cdf_z.powi(2) + 2.0 * cdf_z * pdf_z - cdf_sqrt2_z / sqrt_pi);
            }
        }
        scores
    }

    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let sqrt_2 = 2.0_f64.sqrt();
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let t = y.time[i];
            let e = y.event[i];
            let log_t = t.ln();
            let z = (log_t - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);
            let pdf_z = std_normal.pdf(z);
            let pdf_sqrt2_z = std_normal.pdf(sqrt_2 * z);

            if e {
                // Uncensored gradient: d/d(mu) = -(2*Phi(z) - 1)
                d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);
            } else {
                // Censored gradient: d/d(mu) = -h'(z) where
                // h(z) = z*Phi(z)^2 + 2*Phi(z)*phi(z) - Phi(sqrt(2)*z)/sqrt(pi)
                // h'(z) = Phi(z)^2 + 2*phi(z)^2 - sqrt(2/pi)*phi(sqrt(2)*z)
                // (the 2*z*Phi*phi terms from the first two parts cancel exactly)
                d_params[[i, 0]] =
                    -(cdf_z.powi(2) + 2.0 * pdf_z.powi(2) - sqrt_2_over_pi * pdf_sqrt2_z);
            }

            // d/d(log(scale)) = score + (log_t - loc) * d/d(loc)
            let score_i = if e {
                self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi)
            } else {
                let cdf_sqrt2_z = std_normal.cdf(sqrt_2 * z);
                self.scale[i] * (z * cdf_z.powi(2) + 2.0 * cdf_z * pdf_z - cdf_sqrt2_z / sqrt_pi)
            };
            d_params[[i, 1]] = score_i + (log_t - self.loc[i]) * d_params[[i, 0]];
        }

        d_params
    }

    fn censored_metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
            fi[[i, 1, 1]] = self.scale[i].powi(2);
        }

        fi.mapv_inplace(|x| x / (2.0 * sqrt_pi));
        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lognormal_distribution_methods() {
        let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.5_f64.ln()]).unwrap();
        let dist = LogNormal::from_params(&params);

        // Test mean: exp(mu + sigma^2/2)
        let mean = dist.mean();
        assert_relative_eq!(mean[0], (0.5_f64).exp(), epsilon = 1e-6);

        // Test median: exp(mu)
        let median = dist.median();
        assert_relative_eq!(median[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(median[1], std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_lognormal_cdf_ppf() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = LogNormal::from_params(&params);

        // CDF at median should be 0.5
        let y = Array1::from_vec(vec![1.0]); // exp(0) = 1 is the median
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.5, epsilon = 1e-10);

        // PPF at 0.5 should return median
        let q = Array1::from_vec(vec![0.5]);
        let ppf = dist.ppf(&q);
        assert_relative_eq!(ppf[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lognormal_sample() {
        let params = Array2::from_shape_vec((1, 2), vec![1.0, 0.5_f64.ln()]).unwrap();
        let dist = LogNormal::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be positive (lognormal is always positive)
        assert!(samples.iter().all(|&x| x > 0.0));

        // Check that sample median is close to exp(mu)
        let mut sample_vec: Vec<f64> = samples.column(0).to_vec();
        sample_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let sample_median = sample_vec[500];
        let expected_median = std::f64::consts::E; // exp(1.0)
        assert!((sample_median - expected_median).abs() / expected_median < 0.15);
    }

    #[test]
    fn test_lognormal_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = LogNormal::fit(&y);
        assert_eq!(params.len(), 2);
        // Should fit log-mean and log-std
    }

    #[test]
    fn test_lognormal_survival_function() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = LogNormal::from_params(&params);

        // SF at median should be 0.5
        let y = Array1::from_vec(vec![1.0]);
        let sf = dist.sf(&y);
        assert_relative_eq!(sf[0], 0.5, epsilon = 1e-10);
    }
}
