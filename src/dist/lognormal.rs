use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};

use statrs::distribution::{Continuous, LogNormal as LogNormalDist};

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
