use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{Cauchy as CauchyDist, Continuous};

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
