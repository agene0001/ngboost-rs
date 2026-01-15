use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};

/// The Half-Normal distribution.
///
/// The Half-Normal distribution is a Normal distribution folded at zero,
/// with loc fixed at 0. It has one parameter: scale (sigma).
#[derive(Debug, Clone)]
pub struct HalfNormal {
    /// The scale parameter (sigma).
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for HalfNormal {
    fn from_params(params: &Array2<f64>) -> Self {
        let scale = params.column(0).mapv(f64::exp);
        HalfNormal {
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // For half-normal, MLE for scale is sqrt(mean(y^2))
        // Since E[X^2] = sigma^2 for half-normal
        let n = y.len();
        if n == 0 {
            return array![0.0];
        }

        let sum_sq: f64 = y.iter().map(|&x| x * x).sum();
        let scale = (sum_sq / n as f64).sqrt().max(1e-6);

        array![scale.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of half-normal is scale * sqrt(2/pi)
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        &self.scale * sqrt_2_over_pi
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for HalfNormal {}

impl Scorable<LogScore> for HalfNormal {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -logpdf(y) for half-normal
        // logpdf = log(sqrt(2/pi)) - log(scale) - y^2 / (2 * scale^2)
        // -logpdf = -log(sqrt(2/pi)) + log(scale) + y^2 / (2 * scale^2)
        let log_sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt().ln();
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let scale_sq = self.scale[i] * self.scale[i];
            scores[i] = -log_sqrt_2_over_pi + self.scale[i].ln() + (y[i] * y[i]) / (2.0 * scale_sq);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let scale_sq = self.scale[i] * self.scale[i];
            // d/d(log(scale)) = scale * d/d(scale)
            // d(-logpdf)/d(scale) = 1/scale - y^2/scale^3
            // d(-logpdf)/d(log(scale)) = 1 - y^2/scale^2
            d_params[[i, 0]] = (scale_sq - y[i] * y[i]) / scale_sq;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information for half-normal is 2 (constant)
        let n_obs = self.scale.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 2.0;
        }

        fi
    }
}
