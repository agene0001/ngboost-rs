use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal as NormalDist};
use statrs::statistics::Statistics;

/// Pre-computed constant for vectorized normal distribution computations.
const INV_SQRT_2PI: f64 = 0.3989422804014327; // 1 / sqrt(2 * PI)

/// The Normal (Gaussian) distribution.
#[derive(Debug, Clone)]
pub struct Normal {
    /// The mean of the distribution (loc).
    pub loc: Array1<f64>,
    /// The standard deviation of the distribution (scale).
    pub scale: Array1<f64>,
}

impl Normal {
    /// Returns the variance of the distribution (scale^2).
    /// Computed lazily to avoid redundant storage.
    #[inline]
    pub fn var(&self) -> Array1<f64> {
        &self.scale * &self.scale
    }
}

impl Distribution for Normal {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        Normal { loc, scale }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean();
        let std_dev = if y.len() <= 1 {
            1.0 // Fallback when we can't compute std dev (matches scipy behavior)
        } else {
            y.std(0.0)
        };
        // The parameters are loc and log(scale)
        // Handle edge case where std_dev is 0 or very small - match scipy's robust behavior
        let safe_std_dev = if std_dev <= 0.0 { 1.0 } else { std_dev };
        array![mean, safe_std_dev.ln()]
    }

    fn n_params(&self) -> usize {
        2
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 2));
        p.column_mut(0).assign(&self.loc);
        p.column_mut(1).assign(&self.scale.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for Normal {}

impl DistributionMethods for Normal {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized PDF: f(x) = (1 / (σ * sqrt(2π))) * exp(-0.5 * ((x - μ) / σ)²)
        // Using pre-computed constants for efficiency
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                *r = INV_SQRT_2PI / scale * (-0.5 * z * z).exp();
            });
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized log PDF: ln(f(x)) = -0.5 * ln(2π) - ln(σ) - 0.5 * ((x - μ) / σ)²
        const HALF_LN_2PI: f64 = 0.9189385332046727; // 0.5 * ln(2π)
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                *r = -HALF_LN_2PI - scale.ln() - 0.5 * z * z;
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                *r = std_normal.cdf(z);
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(q.len());
        Zip::from(&mut result)
            .and(q)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &q_i, &loc, &scale| {
                let z = std_normal.inverse_cdf(q_i);
                *r = loc + scale * z;
            });
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = NormalDist::new(self.loc[i], self.scale[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn mode(&self) -> Array1<f64> {
        // For Normal, mode = mean
        self.loc.clone()
    }
}

impl Scorable<LogScore> for Normal {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                let var = scale * scale;
                row[0] = 1.0 / var;
                row[1] = 2.0;
            });
        diag
    }

    fn d_score_and_diagonal_metric(&self, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let mut diag = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(diag.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut d_row, mut m_row, &y_i, &loc, &scale| {
                let var = scale * scale;
                let err = loc - y_i;
                d_row[0] = err / var;
                d_row[1] = 1.0 - (err * err) / var;
                m_row[0] = 1.0 / var;
                m_row[1] = 2.0;
            });

        (d_params, diag)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -logpdf(y)
        // -ln(f(y)) = 0.5*ln(2π) + ln(σ) + 0.5*((y-μ)/σ)²
        const HALF_LN_2PI: f64 = 0.9189385332046727;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                *s = HALF_LN_2PI + scale.ln() + 0.5 * z * z;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Vectorized derivative wrt loc and log(scale)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        // Compute in a single pass using Zip
        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &loc, &scale| {
                let var = scale * scale;
                let err = loc - y_i;
                // d/d(loc) = (loc - y) / var
                row[0] = err / var;
                // d/d(log(scale)) = 1 - (loc - y)² / var
                row[1] = 1.0 - (err * err) / var;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix - diagonal for Normal distribution
        // This is a performance optimization: we only compute diagonal elements
        // since the off-diagonal elements are zero
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        // Vectorized computation of diagonal elements
        Zip::from(fi.outer_iter_mut())
            .and(&self.scale)
            .for_each(|mut fi_i, &scale| {
                let var = scale * scale;
                fi_i[[0, 0]] = 1.0 / var;
                fi_i[[1, 1]] = 2.0;
                // Off-diagonal elements are 0 (already initialized)
            });

        fi
    }
}

impl Scorable<CRPScore> for Normal {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        const INV_2_SQRT_PI: f64 = 0.28209479177387814;
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 2.0 * INV_2_SQRT_PI;
                row[1] = scale * scale * INV_2_SQRT_PI;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Normal distribution — vectorized with Zip
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563; // 1/sqrt(pi)

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                let pdf_z = std_normal.pdf(z);
                let cdf_z = std_normal.cdf(z);
                *s = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI);
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Vectorized derivative — single pass computing both partials
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &loc, &scale| {
                let z = (y_i - loc) / scale;
                let cdf_z = std_normal.cdf(z);
                let pdf_z = std_normal.pdf(z);
                let d_loc = -(2.0 * cdf_z - 1.0);
                row[0] = d_loc;
                row[1] = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI)
                    + (y_i - loc) * d_loc;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // CRPS metric for Normal — vectorized, avoids allocating self.var()
        const INV_2_SQRT_PI: f64 = 0.28209479177387814; // 1/(2*sqrt(pi))
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        Zip::from(fi.outer_iter_mut())
            .and(&self.scale)
            .for_each(|mut fi_i, &scale| {
                fi_i[[0, 0]] = 2.0 * INV_2_SQRT_PI;
                fi_i[[1, 1]] = scale * scale * INV_2_SQRT_PI;
            });

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
}

impl Distribution for NormalFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        NormalFixedVar { loc, scale, var }
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

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 1));
        p.column_mut(0).assign(&self.loc);
        p
    }
}

impl RegressionDistn for NormalFixedVar {}

impl DistributionMethods for NormalFixedVar {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized PDF for fixed variance (scale = 1)
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .for_each(|r, &y_i, &loc| {
                let z = y_i - loc;
                *r = INV_SQRT_2PI * (-0.5 * z * z).exp();
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .for_each(|r, &y_i, &loc| {
                let z = y_i - loc;
                *r = std_normal.cdf(z);
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(q.len());
        Zip::from(&mut result)
            .and(q)
            .and(&self.loc)
            .for_each(|r, &q_i, &loc| {
                let z = std_normal.inverse_cdf(q_i);
                *r = loc + z;
            });
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            for s in 0..n_samples {
                let u: f64 = rng.random();
                samples[[s, i]] = d.inverse_cdf(u);
            }
        }
        samples
    }
}

impl Scorable<LogScore> for NormalFixedVar {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let n_obs = self.loc.len();
        Array2::from_elem((n_obs, 1), 1.0 + 1e-5)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -logpdf for fixed variance (scale=1, var=1)
        // -ln(f(y)) = 0.5*ln(2π) + ln(σ) + 0.5*((y-μ)/σ)²
        // With σ=1: = 0.5*ln(2π) + 0.5*(y-μ)²
        const HALF_LN_2PI: f64 = 0.9189385332046727;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .for_each(|s, &y_i, &loc| {
                let z = y_i - loc;
                *s = HALF_LN_2PI + 0.5 * z * z;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        // d/d(loc) = (loc - y) / var; var=1 so just loc - y
        Zip::from(d_params.column_mut(0))
            .and(&self.loc)
            .and(y)
            .for_each(|d, &loc, &y_i| {
                *d = loc - y_i;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // var=1 so FI = 1/var + eps = 1 + 1e-5 for all observations
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| 1.0 + 1e-5);
        fi
    }
}

impl Scorable<CRPScore> for NormalFixedVar {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        const INV_SQRT_PI: f64 = 0.5641895835477563;
        let n_obs = self.loc.len();
        Array2::from_elem((n_obs, 1), INV_SQRT_PI)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Normal with fixed variance (scale=1)
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .for_each(|s, &y_i, &loc| {
                let z = y_i - loc; // scale=1
                let pdf_z = std_normal.pdf(z);
                let cdf_z = std_normal.cdf(z);
                *s = z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.loc)
            .for_each(|d, &y_i, &loc| {
                let z = y_i - loc; // scale=1
                let cdf_z = std_normal.cdf(z);
                *d = -(2.0 * cdf_z - 1.0);
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Constant for all observations: 1/sqrt(pi)
        const INV_SQRT_PI: f64 = 0.5641895835477563;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| INV_SQRT_PI);
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
}

impl Distribution for NormalFixedMean {
    fn from_params(params: &Array2<f64>) -> Self {
        let scale = params.column(0).mapv(f64::exp);
        let var = &scale * &scale;
        let n = scale.len();
        let loc = Array1::zeros(n);
        NormalFixedMean { loc, scale, var }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let std_dev = y.std(0.0);
        // from_params expects log(scale), so return log(std_dev).
        // Note: Python's NormalFixedMean.fit() has a bug here — it returns raw s
        // instead of log(s), causing from_params to compute exp(s) instead of s.
        let safe_std_dev = if std_dev <= 0.0 { 1.0 } else { std_dev };
        array![safe_std_dev.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.scale.len();
        let mut p = Array2::zeros((n, 1));
        p.column_mut(0).assign(&self.scale.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for NormalFixedMean {}

impl DistributionMethods for NormalFixedMean {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.var.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.scale.clone()
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized PDF for fixed mean (loc = 0)
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.scale)
            .for_each(|r, &y_i, &scale| {
                let z = y_i / scale;
                *r = INV_SQRT_2PI / scale * (-0.5 * z * z).exp();
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(y.len());
        Zip::from(&mut result)
            .and(y)
            .and(&self.scale)
            .for_each(|r, &y_i, &scale| {
                let z = y_i / scale;
                *r = std_normal.cdf(z);
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        let mut result = Array1::zeros(q.len());
        Zip::from(&mut result)
            .and(q)
            .and(&self.scale)
            .for_each(|r, &q_i, &scale| {
                let z = std_normal.inverse_cdf(q_i);
                *r = scale * z;
            });
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let d = NormalDist::new(self.loc[i], self.scale[i]).unwrap();
            for s in 0..n_samples {
                let u: f64 = rng.random();
                samples[[s, i]] = d.inverse_cdf(u);
            }
        }
        samples
    }
}

impl Scorable<LogScore> for NormalFixedMean {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let n_obs = self.loc.len();
        Array2::from_elem((n_obs, 1), 2.0)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Vectorized -logpdf for fixed mean (loc=0)
        // -ln(f(y)) = 0.5*ln(2π) + ln(σ) + 0.5*(y/σ)²
        const HALF_LN_2PI: f64 = 0.9189385332046727;
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.scale)
            .for_each(|s, &y_i, &scale| {
                let z = y_i / scale;
                *s = HALF_LN_2PI + scale.ln() + 0.5 * z * z;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        // d/d(log(scale)) = 1 - y^2 / var (loc=0, so err = -y)
        Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.var)
            .for_each(|d, &y_i, &var| {
                *d = 1.0 - (y_i * y_i) / var;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Constant: fi[[i,0,0]] = 2.0 for all i
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| 2.0);
        fi
    }
}

impl Scorable<CRPScore> for NormalFixedMean {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        const INV_2_SQRT_PI: f64 = 0.28209479177387814;
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 1));
        Zip::from(diag.column_mut(0))
            .and(&self.var)
            .for_each(|d, &var| {
                *d = var * INV_2_SQRT_PI;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Normal with fixed mean (loc=0)
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.scale)
            .for_each(|s, &y_i, &scale| {
                let z = y_i / scale; // loc=0
                let pdf_z = std_normal.pdf(z);
                let cdf_z = std_normal.cdf(z);
                *s = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI);
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let std_normal = NormalDist::new(0.0, 1.0).unwrap();
        const INV_SQRT_PI: f64 = 0.5641895835477563;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.scale)
            .for_each(|d, &y_i, &scale| {
                let z = y_i / scale; // loc=0
                let pdf_z = std_normal.pdf(z);
                let cdf_z = std_normal.cdf(z);
                let score_i = scale * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - INV_SQRT_PI);
                let d_loc = -(2.0 * cdf_z - 1.0);
                // d/d(log(scale)) = score + (y - loc) * d_loc; loc=0
                *d = score_i + y_i * d_loc;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // var / (2*sqrt(pi)) for each observation
        const INV_2_SQRT_PI: f64 = 0.28209479177387814;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        Zip::from(fi.outer_iter_mut())
            .and(&self.var)
            .for_each(|mut fi_i, &var| {
                fi_i[[0, 0]] = var * INV_2_SQRT_PI;
            });

        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vectorized_pdf_matches_scalar() {
        let params =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let pdf_vec = dist.pdf(&y);

        // Compare with statrs scalar implementation
        for i in 0..3 {
            let d = NormalDist::new(dist.loc[i], dist.scale[i]).unwrap();
            let pdf_scalar = d.pdf(y[i]);
            assert_relative_eq!(pdf_vec[i], pdf_scalar, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vectorized_cdf_matches_scalar() {
        let params =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);

        let cdf_vec = dist.cdf(&y);

        // Compare with statrs scalar implementation
        for i in 0..3 {
            let d = NormalDist::new(dist.loc[i], dist.scale[i]).unwrap();
            let cdf_scalar = d.cdf(y[i]);
            assert_relative_eq!(cdf_vec[i], cdf_scalar, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normal_distribution_methods() {
        let params =
            Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);

        // Test mean
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(mean[2], 2.0, epsilon = 1e-10);

        // Test variance
        let var = dist.variance();
        assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[2], 1.0, epsilon = 1e-10);

        // Test CDF
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let cdf = dist.cdf(&y);
        assert_relative_eq!(cdf[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cdf[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cdf[2], 0.5, epsilon = 1e-10);

        // Test PPF (inverse CDF)
        let q = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        let ppf = dist.ppf(&q);
        assert_relative_eq!(ppf[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(ppf[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ppf[2], 2.0, epsilon = 1e-10);

        // Test interval
        let (lower, upper) = dist.interval(0.05);
        assert!(lower[0] < 0.0);
        assert!(upper[0] > 0.0);
    }

    #[test]
    fn test_normal_sample() {
        let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 1.0_f64.ln()]).unwrap();
        let dist = Normal::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 2]);

        // Check that samples have approximately correct mean
        let sample_mean_0: f64 = samples.column(0).iter().sum::<f64>() / samples.nrows() as f64;
        let sample_mean_1: f64 = samples.column(1).iter().sum::<f64>() / samples.nrows() as f64;

        assert!((sample_mean_0 - 0.0).abs() < 0.2);
        assert!((sample_mean_1 - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_normal_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Normal::fit(&y);
        assert_eq!(params.len(), 2);
        assert_relative_eq!(params[0], 3.0, epsilon = 1e-10); // mean
    }

    #[test]
    fn test_normal_fixed_var_distribution_methods() {
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 5.0]).unwrap();
        let dist = NormalFixedVar::from_params(&params);

        assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.mean()[1], 5.0, epsilon = 1e-10);
        assert_relative_eq!(dist.variance()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dist.variance()[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_fixed_mean_distribution_methods() {
        // params = [log(scale)] for NormalFixedMean
        // First obs: log(scale) = 0, so scale = 1
        // Second obs: log(scale) = 1, so scale = e
        let params = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let dist = NormalFixedMean::from_params(&params);

        assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.mean()[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist.std()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dist.std()[1], std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_normalfixedvar_score_matches_library() {
        // Verify vectorized score matches statrs per-element computation
        let params = Array2::from_shape_vec((3, 1), vec![-1.0, 0.0, 2.5]).unwrap();
        let dist = NormalFixedVar::from_params(&params);
        let y = Array1::from_vec(vec![0.5, -0.3, 1.0]);

        let scores = Scorable::<LogScore>::score(&dist, &y);
        for i in 0..3 {
            let d = NormalDist::new(dist.loc[i], dist.scale[i]).unwrap();
            assert_relative_eq!(scores[i], -d.ln_pdf(y[i]), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalfixedmean_score_matches_library() {
        // Verify vectorized score matches statrs per-element computation
        let params = Array2::from_shape_vec((3, 1), vec![0.0, 0.5, 1.0]).unwrap();
        let dist = NormalFixedMean::from_params(&params);
        let y = Array1::from_vec(vec![0.5, -0.3, 1.0]);

        let scores = Scorable::<LogScore>::score(&dist, &y);
        for i in 0..3 {
            let d = NormalDist::new(dist.loc[i], dist.scale[i]).unwrap();
            assert_relative_eq!(scores[i], -d.ln_pdf(y[i]), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalfixedvar_crp_gradient_numerical() {
        // Verify CRP d_score via central finite differences
        let params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let dist = NormalFixedVar::from_params(&params);
        let y = Array1::from_vec(vec![0.5]);

        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Numerical gradient
        let eps = 1e-5;
        let params_plus = Array2::from_shape_vec((1, 1), vec![1.0 + eps]).unwrap();
        let params_minus = Array2::from_shape_vec((1, 1), vec![1.0 - eps]).unwrap();
        let dist_plus = NormalFixedVar::from_params(&params_plus);
        let dist_minus = NormalFixedVar::from_params(&params_minus);
        let s_plus = Scorable::<CRPScore>::score(&dist_plus, &y);
        let s_minus = Scorable::<CRPScore>::score(&dist_minus, &y);
        let numerical = (s_plus[0] - s_minus[0]) / (2.0 * eps);

        assert_relative_eq!(d_score[[0, 0]], numerical, epsilon = 1e-4);
    }

    #[test]
    fn test_normalfixedmean_crp_gradient_numerical() {
        // Verify CRP d_score via central finite differences
        let params = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let dist = NormalFixedMean::from_params(&params);
        let y = Array1::from_vec(vec![1.0]);

        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Numerical gradient
        let eps = 1e-5;
        let params_plus = Array2::from_shape_vec((1, 1), vec![0.5 + eps]).unwrap();
        let params_minus = Array2::from_shape_vec((1, 1), vec![0.5 - eps]).unwrap();
        let dist_plus = NormalFixedMean::from_params(&params_plus);
        let dist_minus = NormalFixedMean::from_params(&params_minus);
        let s_plus = Scorable::<CRPScore>::score(&dist_plus, &y);
        let s_minus = Scorable::<CRPScore>::score(&dist_minus, &y);
        let numerical = (s_plus[0] - s_minus[0]) / (2.0 * eps);

        assert_relative_eq!(d_score[[0, 0]], numerical, epsilon = 1e-4);
    }
}
