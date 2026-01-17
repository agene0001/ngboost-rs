use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, StudentsT as StudentsTDist};
use statrs::function::gamma::digamma;

/// The Student's T distribution with learnable degrees of freedom.
///
/// Has three parameters: loc (mean), log(scale), and log(df).
#[derive(Debug, Clone)]
pub struct StudentT {
    /// The location parameter (mean).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The variance (scale^2).
    pub var: Array1<f64>,
    /// The degrees of freedom.
    pub df: Array1<f64>,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl Distribution for StudentT {
    fn from_params(params: &Array2<f64>) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        let var = &scale * &scale;
        let df = params.column(2).mapv(f64::exp);
        StudentT {
            loc,
            scale,
            var,
            df,
            _params: params.clone(),
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Simple estimation using method of moments
        let mean = y.mean().unwrap_or(0.0);
        let std_dev = y.std(0.0).max(1e-6);
        // Default df to 3.0 (common choice for robust estimation)
        let df = 3.0_f64;
        array![mean, std_dev.ln(), df.ln()]
    }

    fn n_params(&self) -> usize {
        3
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Student's T is loc (for df > 1)
        self.loc.clone()
    }

    fn params(&self) -> &Array2<f64> {
        &self._params
    }
}

impl RegressionDistn for StudentT {}

impl StudentT {
    /// Sample from the distribution using inverse CDF method.
    /// Returns an array of shape (n_samples, n_obs) where each column is samples for one observation.
    pub fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let d = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]).unwrap();
            for s in 0..n_samples {
                // Use inverse CDF sampling
                let u: f64 = rng.random();
                samples[[s, i]] = d.inverse_cdf(u);
            }
        }
        samples
    }
}

impl Scorable<LogScore> for StudentT {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 3));

        for i in 0..n_obs {
            let loc_i = self.loc[i];
            let var_i = self.var[i];
            let df_i = self.df[i];
            let y_i = y[i];

            let diff = y_i - loc_i;
            let diff_sq = diff * diff;

            // Denominator: df * var + (y - loc)^2
            let denom = df_i * var_i + diff_sq;

            // d/d(loc): -(df + 1) * (y - loc) / (df * var + (y - loc)^2)
            d_params[[i, 0]] = -(df_i + 1.0) * diff / denom;

            // d/d(log(scale)): 1 - (df + 1) * (y - loc)^2 / (df * var + (y - loc)^2)
            d_params[[i, 1]] = 1.0 - (df_i + 1.0) * diff_sq / denom;

            // d/d(log(df)) is more complex
            let term_1 = (df_i / 2.0) * digamma((df_i + 1.0) / 2.0);
            let term_2 = (-df_i / 2.0) * digamma(df_i / 2.0);
            let term_3 = -0.5;
            let term_4_1 = (-df_i / 2.0) * (1.0 + diff_sq / (df_i * var_i)).ln();
            let term_4_2_num = (df_i + 1.0) * diff_sq;
            let term_4_2_den = 2.0 * (df_i * var_i) * (1.0 + diff_sq / (df_i * var_i));

            d_params[[i, 2]] = -(term_1 + term_2 + term_3 + term_4_1 + term_4_2_num / term_4_2_den);
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Python's TLogScore does NOT implement metric(), so it falls back to
        // the default LogScore.metric() which returns identity matrix.
        // We match that behavior here.
        let n_obs = self.loc.len();
        let n_params = 3;

        let mut fi = Array3::zeros((n_obs, n_params, n_params));
        for i in 0..n_obs {
            for j in 0..n_params {
                fi[[i, j, j]] = 1.0;
            }
        }
        fi
    }
}

/// The Student's T distribution with fixed degrees of freedom.
///
/// Has two parameters: loc (mean) and log(scale).
/// Degrees of freedom is fixed at construction time (default: 3.0).
#[derive(Debug, Clone)]
pub struct TFixedDf {
    /// The location parameter (mean).
    pub loc: Array1<f64>,
    /// The scale parameter.
    pub scale: Array1<f64>,
    /// The variance (scale^2).
    pub var: Array1<f64>,
    /// The degrees of freedom (fixed).
    pub df: Array1<f64>,
    /// The fixed df value used for all observations.
    pub fixed_df: f64,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl TFixedDf {
    /// The default fixed degrees of freedom.
    pub const DEFAULT_FIXED_DF: f64 = 3.0;

    /// Creates a new TFixedDf distribution from parameters with a custom fixed df.
    pub fn from_params_with_df(params: &Array2<f64>, fixed_df: f64) -> Self {
        let loc = params.column(0).to_owned();
        let scale = params.column(1).mapv(f64::exp);
        let var = &scale * &scale;
        let n = loc.len();
        let df = Array1::from_elem(n, fixed_df);
        TFixedDf {
            loc,
            scale,
            var,
            df,
            fixed_df,
            _params: params.clone(),
        }
    }
}

impl Distribution for TFixedDf {
    fn from_params(params: &Array2<f64>) -> Self {
        Self::from_params_with_df(params, Self::DEFAULT_FIXED_DF)
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean().unwrap_or(0.0);
        let std_dev = y.std(0.0).max(1e-6);
        array![mean, std_dev.ln()]
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

impl RegressionDistn for TFixedDf {}

impl Scorable<LogScore> for TFixedDf {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]).unwrap();
            scores[i] = -d.ln_pdf(y_i);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let loc_i = self.loc[i];
            let var_i = self.var[i];
            let df_i = self.df[i];
            let y_i = y[i];

            let diff = y_i - loc_i;
            let diff_sq = diff * diff;
            let denom = df_i * var_i + diff_sq;

            // d/d(loc)
            d_params[[i, 0]] = -(df_i + 1.0) * diff / denom;

            // d/d(log(scale))
            d_params[[i, 1]] = 1.0 - (df_i + 1.0) * diff_sq / denom;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for T with fixed df
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let df_i = self.df[i];
            let var_i = self.var[i];

            // FI[0, 0] = (df + 1) / ((df + 3) * var)
            fi[[i, 0, 0]] = (df_i + 1.0) / ((df_i + 3.0) * var_i);

            // FI[1, 1] = df / (2 * (df + 3) * var)
            // Note: Python uses var in denominator but this seems wrong dimensionally
            // Matching Python behavior exactly:
            fi[[i, 1, 1]] = df_i / (2.0 * (df_i + 3.0) * var_i);
        }

        fi
    }
}

/// The Student's T distribution with fixed df=3 and fixed variance=1.
///
/// Has one parameter: loc (mean).
#[derive(Debug, Clone)]
pub struct TFixedDfFixedVar {
    /// The location parameter (mean).
    pub loc: Array1<f64>,
    /// The scale parameter (fixed at 1.0).
    pub scale: Array1<f64>,
    /// The variance (fixed at 1.0).
    pub var: Array1<f64>,
    /// The degrees of freedom (fixed).
    pub df: Array1<f64>,
    /// The fixed df value.
    pub fixed_df: f64,
    /// The parameters of the distribution, stored as a 2D array.
    _params: Array2<f64>,
}

impl TFixedDfFixedVar {
    /// The default fixed degrees of freedom.
    pub const DEFAULT_FIXED_DF: f64 = 3.0;

    /// Creates a new TFixedDfFixedVar distribution from parameters with a custom fixed df.
    pub fn from_params_with_df(params: &Array2<f64>, fixed_df: f64) -> Self {
        let loc = params.column(0).to_owned();
        let n = loc.len();
        let scale = Array1::ones(n);
        let var = Array1::ones(n);
        let df = Array1::from_elem(n, fixed_df);
        TFixedDfFixedVar {
            loc,
            scale,
            var,
            df,
            fixed_df,
            _params: params.clone(),
        }
    }
}

impl Distribution for TFixedDfFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        Self::from_params_with_df(params, Self::DEFAULT_FIXED_DF)
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean().unwrap_or(0.0);
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

impl RegressionDistn for TFixedDfFixedVar {}

impl Scorable<LogScore> for TFixedDfFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let d = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]).unwrap();
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
            let df_i = self.df[i];
            let y_i = y[i];

            let diff = y_i - loc_i;
            let diff_sq = diff * diff;

            // d/d(loc) for fixed var case
            let num = (df_i + 1.0) * (2.0 / (df_i * var_i)) * diff;
            let den = 2.0 * (1.0 + (1.0 / (df_i * var_i)) * diff_sq);
            d_params[[i, 0]] = -num / den;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            let df_i = self.df[i];
            let var_i = self.var[i];
            fi[[i, 0, 0]] = (df_i + 1.0) / ((df_i + 3.0) * var_i);
        }

        fi
    }
}
