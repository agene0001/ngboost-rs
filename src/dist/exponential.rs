use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{
    CRPScore, CRPScoreCensored, CensoredScorable, LogScore, LogScoreCensored, Scorable,
    SurvivalData,
};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Exp;

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
