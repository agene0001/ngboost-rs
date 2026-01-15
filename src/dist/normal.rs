use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{Continuous, ContinuousCDF, Normal as NormalDist};
use statrs::statistics::Statistics;

/// The Normal (Gaussian) distribution.
#[derive(Debug, Clone)]
pub struct Normal {
    /// The mean of the distribution (loc).
    pub loc: Array1<f64>,
    /// The standard deviation of the distribution (scale).
    pub scale: Array1<f64>,
    /// The variance of the distribution.
    pub var: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Normal {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        let var = &scale * &scale;
        Normal {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean();
        let std_dev = y.std_dev();
        // The parameters are loc and log(scale)
        array![mean, std_dev.max(1e-6).ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Normal {}

impl Scorable<LogScore> for Normal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y)
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Derivative wrt loc and log(scale)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        let err = &self.loc - y;

        // d/d(loc)
        d_params.column_mut(0).assign(&(&err / &self.var));

        // d/d(log(scale))
        let term2 = (&err * &err) / &self.var;
        d_params.column_mut(1).assign(&(1.0 - term2));

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / self.var[i];
            fi[[i, 1, 1]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for Normal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Normal distribution
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
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
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);

            // d/d(loc)
            d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);

            // d/d(log(scale)) - need to compute score first
            let pdf_z = std_normal.pdf(z);
            let sqrt_pi = std::f64::consts::PI.sqrt();
            let score_i = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            d_params[[i, 1]] = score_i + (y[i] - self.loc[i]) * d_params[[i, 0]];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Normal
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
            fi[[i, 1, 1]] = self.var[i];
        }

        // Scale by 1/(2*sqrt(pi))
        fi.mapv_inplace(|x| x / (2.0 * sqrt_pi));
        fi
    }
}

// ============================================================================
// NormalFixedVar - Normal distribution with fixed variance = 1
// ============================================================================

/// Normal distribution with variance fixed at 1.
///
/// Has one parameter: loc (mean).
#[derive(Debug, Clone)]
pub struct NormalFixedVar {
    /// The location parameter (mean).
    pub loc: Array1<f64>,
    /// The scale parameter (fixed at 1.0).
    pub scale: Array1<f64>,
    /// The variance (fixed at 1.0).
    pub var: Array1<f64>,
    /// The parameters of the distribution.
    _params: Array2<f64>,
}

impl Distribution for NormalFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        NormalFixedVar {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean();
        array![mean]
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

impl RegressionDistn for NormalFixedVar {}

impl Scorable<LogScore> for NormalFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            // d/d(loc) = (loc - y) / var
            d_params[[i, 0]] = (self.loc[i] - y[i]) / self.var[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / self.var[i] + 1e-5;
        }

        fi
    }
}

impl Scorable<CRPScore> for NormalFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let cdf_z = std_normal.cdf(z);
            d_params[[i, 0]] = -(2.0 * cdf_z - 1.0);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0 / (2.0 * sqrt_pi);
        }

        fi
    }
}

// ============================================================================
// NormalFixedMean - Normal distribution with fixed mean = 0
// ============================================================================

/// Normal distribution with mean fixed at 0.
///
/// Has one parameter: log(scale).
#[derive(Debug, Clone)]
pub struct NormalFixedMean {
    /// The location parameter (fixed at 0.0).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The variance.
    pub var: Array1<f64>,
    /// The parameters of the distribution.
    _params: Array2<f64>,
}

impl Distribution for NormalFixedMean {
    fn from_params(params: &Array2<f64>) -> Self {
        let scale = params.column(0).mapv(f64::exp);
        let var = &scale * &scale;
        let n = scale.len();
        let loc = Array1::zeros(n);
        NormalFixedMean {
            loc,
            scale,
            var,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let std_dev = y.std(0.0).max(1e-6);
        array![std_dev.ln()]
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

impl RegressionDistn for NormalFixedMean {}

impl Scorable<LogScore> for NormalFixedMean {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let err = self.loc[i] - y[i];
            // d/d(log(scale)) = 1 - (loc - y)^2 / var
            d_params[[i, 0]] = 1.0 - (err * err) / self.var[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for NormalFixedMean {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();

        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            scores[i] = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let z = (y[i] - self.loc[i]) / self.scale[i];
            let pdf_z = std_normal.pdf(z);
            let cdf_z = std_normal.cdf(z);
            let score_i = self.scale[i] * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / sqrt_pi);
            let d_loc = -(2.0 * cdf_z - 1.0);
            d_params[[i, 0]] = score_i + (y[i] - self.loc[i]) * d_loc;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = self.var[i] / (2.0 * sqrt_pi);
        }

        fi
    }
}
