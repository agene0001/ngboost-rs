use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{
    Continuous, ContinuousCDF, LogNormal as LogNormalDist, Normal as NormalDist,
};

/// The LogNormal distribution.
#[derive(Debug, Clone)]
pub struct LogNormal {
    pub loc: Array1<f64>,
    pub scale: Array1<f64>,
    _params: Array2<f64>,
}

impl Distribution for LogNormal {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        LogNormal {
            loc,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        if y.is_empty() {
            return array![0.0, 0.0];
        }
        let log_y: Array1<f64> = y.mapv(|v| v.max(1e-9).ln());
        let mean = log_y.mean().unwrap_or(0.0);
        let std_dev = log_y.std(0.0);
        array![mean, std_dev.max(1e-6).ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of lognormal is exp(loc + scale^2 / 2)
        (&self.loc + &(&self.scale.mapv(|s| s.powi(2)) / 2.0)).mapv(f64::exp)
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for LogNormal {}

impl Scorable<LogScore> for LogNormal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = LogNormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        let log_y = y.mapv(|v| v.ln());
        let err = &self.loc - &log_y;
        let var = self.scale.mapv(|s| s.powi(2));

        // d/d(loc)
        d_params.column_mut(0).assign(&(&err / &var));

        // d/d(log(scale))
        let term2 = (&err * &err) / &var;
        d_params.column_mut(1).assign(&(1.0 - term2));

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));
        let var = self.scale.mapv(|s| s.powi(2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / var[i];
            fi[[i, 1, 1]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for LogNormal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let log_y = y[i].ln();
            let z = (log_y - self.loc[i]) / self.scale[i];
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
            let log_y = y[i].ln();
            let z = (log_y - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);

            // d/d(loc)
            d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);

            // d/d(log(scale))
            let pdf_z = std_normal.pdf(z);
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let score_i = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            d_params[[i, 1]] = score_i + (log_y - self.loc[i]) * d_params[[i, 0]];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));
        let var = self.scale.mapv(|s| s.powi(2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
            fi[[i, 1, 1]] = var[i];
        }

        // Scale by 1/(2*sqrt(pi))
        fi.mapv_inplace(|x| x / (2.0 * sqrt_pi));
        fi
    }
}

// ============================================================================
// Censored LogScore for survival analysis
// ============================================================================

impl CensoredScorable<LogScoreCensored> for LogNormal {
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
            fi[[i, 0, 0]] = 1.0 / (var[i] + eps);
            fi[[i, 1, 1]] = 2.0;
        }

        fi
    }
}

// ============================================================================
// Censored CRPScore for survival analysis
// ============================================================================

impl CensoredScorable<CRPScoreCensored> for LogNormal {
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
                // Uncensored gradient
                d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);
            } else {
                // Censored gradient
                d_params[[i, 0]] = -(cdf_z.powi(2) + 2.0 * z * cdf_z * pdf_z + 2.0 * pdf_z.powi(2)
                    - 2.0 * cdf_z * pdf_z.powi(2)
                    - sqrt_2_over_pi * pdf_sqrt2_z);
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
