use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{Discrete, Poisson as PoissonDist};

/// The Poisson distribution.
#[derive(Debug, Clone)]
pub struct Poisson {
    pub rate: Array1<f64>,
    _params: Array2<f64>,
}

impl Distribution for Poisson {
    fn from_params(params: &Array2<f64>) -> Self {
        let rate = params.column(0).mapv(f64::exp);
        Poisson {
            rate,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean().unwrap_or(1.0);
        array![mean.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.rate.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Poisson {}

impl Scorable<LogScore> for Poisson {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = PoissonDist::new(self.rate[i]).unwrap();
            scores[i] = -d.ln_pmf(y_i as u64);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        let grad = &self.rate - y;

        d_params.column_mut(0).assign(&grad);
        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.rate.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = self.rate[i];
        }

        fi
    }
}
