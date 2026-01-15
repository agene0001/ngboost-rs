use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use statrs::distribution::{Continuous, Weibull as WeibullDist};
use statrs::function::gamma::gamma;

/// The Weibull distribution.
#[derive(Debug, Clone)]
pub struct Weibull {
    /// The shape parameter (k or c).
    pub shape: Array1<f64>,
    /// The scale parameter (lambda).
    pub scale: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for Weibull {
    fn from_params(params: &Array2<f64>) -> Self {
        let shape = params.column(0).mapv(f64::exp);
        let scale = params.column(1).mapv(f64::exp);
        Weibull {
            shape,
            scale,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Simple method of moments estimation for Weibull
        // This is approximate; proper MLE requires numerical optimization
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }

        let mean = y.mean().unwrap_or(1.0);
        let var = y.var(0.0);

        // Coefficient of variation
        let cv = (var.sqrt() / mean).clamp(0.1, 10.0);

        // Approximate shape from CV (using approximation k ≈ 1.2 / CV)
        let shape = (1.2 / cv).max(0.1);

        // Scale from mean: mean = scale * Gamma(1 + 1/shape)
        let gamma_val = gamma(1.0 + 1.0 / shape);
        let scale = (mean / gamma_val).max(1e-6);

        array![shape.ln(), scale.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Weibull is scale * Gamma(1 + 1/shape)
        let mut means = Array1::zeros(self.shape.len());
        for i in 0..self.shape.len() {
            let gamma_val = gamma(1.0 + 1.0 / self.shape[i]);
            means[i] = self.scale[i] * gamma_val;
        }
        means
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for Weibull {}

impl Scorable<LogScore> for Weibull {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            // statrs Weibull uses (shape, scale) parameterization
            let d = WeibullDist::new(self.shape[i], self.scale[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let k = self.shape[i];
            let lam = self.scale[i];
            let y_i = y[i];

            // Ratio y/scale
            let ratio = y_i / lam;
            let ratio_k = ratio.powf(k);

            // shared_term = k * ((y/scale)^k - 1)
            let shared_term = k * (ratio_k - 1.0);

            // d/d(log(shape)) = shape * [shared_term * log(y/scale) - 1]
            // But we parameterize as log(shape), so multiply by shape
            d_params[[i, 0]] = shared_term * ratio.ln() - 1.0;

            // d/d(log(scale)) = -shared_term
            d_params[[i, 1]] = -shared_term;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for Weibull (from Python implementation)
        // Uses Euler's constant gamma ≈ 0.5772156649
        let euler_gamma = 0.5772156649;
        let n_obs = self.shape.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let k = self.shape[i];

            // FI[0, 0] = (pi^2 / 6) + (1 - gamma)^2
            let pi = std::f64::consts::PI;
            let one_minus_gamma = 1.0 - euler_gamma;
            fi[[i, 0, 0]] = (pi * pi / 6.0) + (one_minus_gamma * one_minus_gamma);

            // FI[1, 0] = FI[0, 1] = -k * (1 - gamma)
            fi[[i, 0, 1]] = -k * (1.0 - euler_gamma);
            fi[[i, 1, 0]] = fi[[i, 0, 1]];

            // FI[1, 1] = k^2
            fi[[i, 1, 1]] = k * k;
        }

        fi
    }
}
