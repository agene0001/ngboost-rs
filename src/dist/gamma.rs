use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{Continuous, Gamma as GammaDist};
use statrs::function::gamma::digamma;

/// The Gamma distribution.
#[derive(Debug, Clone)]
pub struct Gamma {
    pub shape: Array1<f64>, // alpha
    pub rate: Array1<f64>,  // beta
    _params: Array2<f64>,
}

impl Distribution for Gamma {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = params.column(0).mapv(f64::exp);
        let rate = params.column(1).mapv(f64::exp);
        Gamma {
            shape,
            rate,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // This is a simplification, MLE for Gamma is complex.
        // Using method of moments.
        let mean = y.mean().unwrap_or(1.0);
        let var = y.var(0.0);
        let shape = mean * mean / var.max(1e-9);
        let scale = var / mean.max(1e-9);
        let rate: f64 = 1.0 / scale;
        array![shape.ln(), rate.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean is shape / rate
        &self.shape / &self.rate
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Gamma {}

impl Scorable<LogScore> for Gamma {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = GammaDist::new(self.shape[i], self.rate[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let shape_i = self.shape[i];
            let rate_i = self.rate[i];

            // d/d(log(shape))
            let d_log_shape = shape_i * (digamma(shape_i) - (y[i] * rate_i).max(1e-9).ln());
            d_params[[i, 0]] = d_log_shape;

            // d/d(log(rate))
            let d_log_rate = y[i] * rate_i - shape_i;
            d_params[[i, 1]] = d_log_rate;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let shape_i = self.shape[i];

            // We use our local helper function for trigamma
            fi[[i, 0, 0]] = shape_i * shape_i * trigamma(shape_i);
            fi[[i, 1, 1]] = shape_i;
            fi[[i, 0, 1]] = -shape_i;
            fi[[i, 1, 0]] = -shape_i;
        }

        fi
    }
}
fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut result = 0.0;

    // Use recurrence relation trigamma(x) = trigamma(x+1) + 1/x^2
    // to shift argument to > 10 for asymptotic expansion accuracy
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion: 1/x + 1/2x^2 + 1/6x^3 - 1/30x^5 + 1/42x^7
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x2 * x3;
    let x7 = x2 * x5;

    result += 1.0 / x
        + 0.5 / x2
        + 1.0 / (6.0 * x3)
        - 1.0 / (30.0 * x5)
        + 1.0 / (42.0 * x7);

    result
}
