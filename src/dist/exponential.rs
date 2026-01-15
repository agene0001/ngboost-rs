use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};

/// The Exponential distribution.
#[derive(Debug, Clone)]
pub struct Exponential {
    pub rate: Array1<f64>,
    _params: Array2<f64>,
}

impl Distribution for Exponential {
    fn from_params(params: &Array2<f64>) -> Self {
        // scale = exp(param), rate = 1/scale = exp(-param)
        let rate = params.column(0).mapv(|p| (-p).exp());
        Exponential {
            rate,
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
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            // ln_pdf = ln(rate) - rate * x
            scores[i] = -(self.rate[i].ln() - self.rate[i] * y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        let scale = 1.0 / &self.rate;
        let grad = 1.0 - y / &scale;

        d_params.column_mut(0).assign(&grad);
        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.rate.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0;
        }

        fi
    }
}
