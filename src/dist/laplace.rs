use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};

/// The Laplace distribution.
#[derive(Debug, Clone)]
pub struct Laplace {
    /// The location parameter (mean/median).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Laplace {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        Laplace {
            loc,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // For Laplace, the MLE for loc is the median and scale is mean absolute deviation
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }

        // Compute median
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Scale is mean absolute deviation from median
        let mad: f64 = y.iter().map(|&x| (x - median).abs()).sum::<f64>() / n as f64;

        array![median, mad.max(1e-6).ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Laplace is loc
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Laplace {}

impl Scorable<LogScore> for Laplace {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y) = |y - loc| / scale + log(2 * scale)
        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            scores[i] = abs_diff / self.scale[i] + (2.0 * self.scale[i]).ln();
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let diff = self.loc[i] - y[i];
            // d/d(loc) = sign(loc - y) / scale
            d_params[[i, 0]] = diff.signum() / self.scale[i];
            // d/d(log(scale)) = 1 - |loc - y| / scale
            d_params[[i, 1]] = 1.0 - diff.abs() / self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Laplace
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let scale_sq = self.scale[i] * self.scale[i];
            fi[[i, 0, 0]] = 1.0 / scale_sq;
            fi[[i, 1, 1]] = 1.0;
        }

        fi
    }
}

impl Scorable<CRPScore> for Laplace {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Laplace distribution
        let mut scores = Array1::zeros(y.len());
        for i in 0..y.len() {
            let abs_diff = (y[i] - self.loc[i]).abs();
            let exp_term = (-abs_diff / self.scale[i]).exp();
            scores[i] = abs_diff + self.scale[i] * exp_term - 0.75 * self.scale[i];
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let diff = self.loc[i] - y[i];
            let abs_diff = diff.abs();
            let exp_term = (-abs_diff / self.scale[i]).exp();

            // d/d(loc)
            d_params[[i, 0]] = diff.signum() * (1.0 - exp_term);

            // d/d(log(scale)) - multiply by scale due to chain rule for log(scale)
            d_params[[i, 1]] = exp_term * (self.scale[i] + abs_diff) - 0.75 * self.scale[i];
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Laplace
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 0.5 / self.scale[i];
            fi[[i, 1, 1]] = 0.25 * self.scale[i];
        }

        fi
    }
}
