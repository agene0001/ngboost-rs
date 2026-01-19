use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, StudentsT as StudentsTDist};
use statrs::function::beta::ln_beta;
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

impl DistributionMethods for StudentT {
    fn mean(&self) -> Array1<f64> {
        // Mean of Student's T is loc for df > 1, undefined otherwise
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            if self.df[i] > 1.0 {
                result[i] = self.loc[i];
            } else {
                result[i] = f64::NAN;
            }
        }
        result
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of Student's T is scale^2 * df / (df - 2) for df > 2
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            if self.df[i] > 2.0 {
                result[i] = self.var[i] * self.df[i] / (self.df[i] - 2.0);
            } else if self.df[i] > 1.0 {
                result[i] = f64::INFINITY;
            } else {
                result[i] = f64::NAN;
            }
        }
        result
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.ln_pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
                result[i] = d.inverse_cdf(q_clamped);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of Student's T is loc (symmetric distribution)
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Student's T is loc
        self.loc.clone()
    }
}

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

impl Scorable<CRPScore> for StudentT {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Student's T distribution
        // CRPS(F, y) = σ * [z * (2*F_ν(z) - 1) + 2*f_ν(z) * (ν + z²)/(ν - 1)
        //              - 2*sqrt(ν) * B(0.5, ν - 0.5) / ((ν - 1) * B(0.5, ν/2)²)]
        // where z = (y - μ)/σ, F_ν is standard t CDF, f_ν is standard t PDF,
        // and B is the beta function.
        //
        // For ν = 1 (Cauchy), CRPS is undefined (infinite).
        // For ν ≤ 1, we return a large penalty.

        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            // For df <= 1, CRPS is not well-defined
            if nu <= 1.0 {
                scores[i] = 1e10; // Large penalty
                continue;
            }

            let z = (y_i - mu) / sigma;

            // Standard t distribution with df = nu
            if let Ok(std_t) = StudentsTDist::new(0.0, 1.0, nu) {
                let f_z = std_t.pdf(z);
                let big_f_z = std_t.cdf(z);

                // Term 1: z * (2*F(z) - 1)
                let term1 = z * (2.0 * big_f_z - 1.0);

                // Term 2: 2*f(z) * (ν + z²)/(ν - 1)
                let term2 = 2.0 * f_z * (nu + z * z) / (nu - 1.0);

                // Term 3: 2*sqrt(ν) * B(0.5, ν - 0.5) / ((ν - 1) * B(0.5, ν/2)²)
                // Using ln_beta for numerical stability
                let ln_b1 = ln_beta(0.5, nu - 0.5);
                let ln_b2 = ln_beta(0.5, nu / 2.0);
                let term3 = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);

                scores[i] = sigma * (term1 + term2 - term3);
            } else {
                scores[i] = 1e10;
            }
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of CRPS for Student T w.r.t. (loc, log(scale), log(df))
        // This is complex, so we use numerical approximation for simplicity
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 3));
        let eps = 1e-6;

        for i in 0..n_obs {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                continue; // Skip for invalid df
            }

            let z = (y_i - mu) / sigma;

            if let Ok(std_t) = StudentsTDist::new(0.0, 1.0, nu) {
                let big_f_z = std_t.cdf(z);

                // d(CRPS)/d(μ) = -σ * d(CRPS_std)/dz / σ = -(2*F(z) - 1)
                d_params[[i, 0]] = -(2.0 * big_f_z - 1.0);

                // d(CRPS)/d(log(σ)) = CRPS + (y - μ) * d(CRPS)/d(μ)
                // Compute CRPS at this point
                let f_z = std_t.pdf(z);
                let term1 = z * (2.0 * big_f_z - 1.0);
                let term2 = 2.0 * f_z * (nu + z * z) / (nu - 1.0);
                let ln_b1 = ln_beta(0.5, nu - 0.5);
                let ln_b2 = ln_beta(0.5, nu / 2.0);
                let term3 = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);
                let crps_std = term1 + term2 - term3;

                d_params[[i, 1]] = sigma * crps_std + (y_i - mu) * d_params[[i, 0]];

                // d(CRPS)/d(log(ν)) - use numerical differentiation
                let nu_plus = nu * (1.0 + eps);
                let nu_minus = nu * (1.0 - eps);

                let crps_plus = compute_t_crps_std(z, nu_plus);
                let crps_minus = compute_t_crps_std(z, nu_minus);

                d_params[[i, 2]] = sigma * nu * (crps_plus - crps_minus) / (2.0 * eps * nu);
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Approximate metric using identity matrix (matching LogScore behavior for T)
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

/// Helper function to compute standardized CRPS for t-distribution
fn compute_t_crps_std(z: f64, nu: f64) -> f64 {
    if nu <= 1.0 {
        return 1e10;
    }

    if let Ok(std_t) = StudentsTDist::new(0.0, 1.0, nu) {
        let f_z = std_t.pdf(z);
        let big_f_z = std_t.cdf(z);

        let term1 = z * (2.0 * big_f_z - 1.0);
        let term2 = 2.0 * f_z * (nu + z * z) / (nu - 1.0);
        let ln_b1 = ln_beta(0.5, nu - 0.5);
        let ln_b2 = ln_beta(0.5, nu / 2.0);
        let term3 = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);

        term1 + term2 - term3
    } else {
        1e10
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

impl DistributionMethods for TFixedDf {
    fn mean(&self) -> Array1<f64> {
        // Mean is loc for df > 1 (df=3 by default, so always defined)
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        // Variance is scale^2 * df / (df - 2) for df > 2
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            if self.df[i] > 2.0 {
                result[i] = self.var[i] * self.df[i] / (self.df[i] - 2.0);
            } else {
                result[i] = f64::INFINITY;
            }
        }
        result
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
                result[i] = d.inverse_cdf(q_clamped);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        self.loc.clone()
    }
}

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

impl Scorable<CRPScore> for TFixedDf {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Student's T with fixed df
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                scores[i] = 1e10;
                continue;
            }

            let z = (y_i - mu) / sigma;
            scores[i] = sigma * compute_t_crps_std(z, nu);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient w.r.t. (loc, log(scale))
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        for i in 0..n_obs {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                continue;
            }

            let z = (y_i - mu) / sigma;

            if let Ok(std_t) = StudentsTDist::new(0.0, 1.0, nu) {
                let big_f_z = std_t.cdf(z);

                // d(CRPS)/d(μ)
                d_params[[i, 0]] = -(2.0 * big_f_z - 1.0);

                // d(CRPS)/d(log(σ))
                let crps_std = compute_t_crps_std(z, nu);
                d_params[[i, 1]] = sigma * crps_std + (y_i - mu) * d_params[[i, 0]];
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Use similar metric to LogScore
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            let df_i = self.df[i];
            let var_i = self.var[i];
            fi[[i, 0, 0]] = (df_i + 1.0) / ((df_i + 3.0) * var_i);
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

impl DistributionMethods for TFixedDfFixedVar {
    fn mean(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn variance(&self) -> Array1<f64> {
        // For df=3, variance = 1 * 3 / (3 - 2) = 3
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            result[i] = self.var[i] * self.df[i] / (self.df[i] - 2.0);
        }
        result
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.pdf(y[i]);
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                result[i] = d.cdf(y[i]);
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                let q_clamped = q[i].clamp(1e-15, 1.0 - 1e-15);
                result[i] = d.inverse_cdf(q_clamped);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            if let Ok(d) = StudentsTDist::new(self.loc[i], self.scale[i], self.df[i]) {
                for s in 0..n_samples {
                    let u: f64 = rng.random();
                    samples[[s, i]] = d.inverse_cdf(u);
                }
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        self.loc.clone()
    }

    fn mode(&self) -> Array1<f64> {
        self.loc.clone()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_studentt_crpscore() {
        // Student T with df=5, loc=0, scale=1
        let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 5.0_f64.ln()]).unwrap();
        let dist = StudentT::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS should be finite and non-negative
        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);
    }

    #[test]
    fn test_studentt_crpscore_at_mean() {
        // CRPS at the mean should be relatively small
        let params = Array2::from_shape_vec((1, 3), vec![5.0, 0.0, 3.0_f64.ln()]).unwrap();
        let dist = StudentT::from_params(&params);

        let y = Array1::from_vec(vec![5.0]); // At the mean
        let score = Scorable::<CRPScore>::score(&dist, &y);

        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);
        assert!(score[0] < 1.0); // Should be small at the mean
    }

    #[test]
    fn test_studentt_crpscore_df_1_penalty() {
        // For df=1 (Cauchy), CRPS should return large penalty
        let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(); // df = exp(0) = 1
        let dist = StudentT::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // Should be very large penalty
        assert!(score[0] > 1e9);
    }

    #[test]
    fn test_studentt_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 3.0_f64.ln()]).unwrap();
        let dist = StudentT::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradients should be finite
        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
        assert!(d_score[[0, 2]].is_finite());
    }

    #[test]
    fn test_tfixeddf_crpscore() {
        // TFixedDf with default df=3
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = TFixedDf::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);
    }

    #[test]
    fn test_tfixeddf_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist = TFixedDf::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        assert!(d_score[[0, 0]].is_finite());
        assert!(d_score[[0, 1]].is_finite());
    }

    #[test]
    fn test_tfixeddfixedvar_crpscore() {
        // TFixedDfFixedVar with default df=3, var=1
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = TFixedDfFixedVar::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);
    }

    #[test]
    fn test_tfixeddfixedvar_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = TFixedDfFixedVar::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        assert!(d_score[[0, 0]].is_finite());
    }

    #[test]
    fn test_studentt_crps_converges_to_normal() {
        // As df -> infinity, Student T converges to Normal
        // CRPS should be similar to Normal CRPS for large df
        let params_t = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 100.0_f64.ln()]).unwrap();
        let dist_t = StudentT::from_params(&params_t);

        let y = Array1::from_vec(vec![1.0]);
        let score_t = Scorable::<CRPScore>::score(&dist_t, &y);

        // Compare to Normal CRPS (Student T converges to Normal as df -> infinity)
        use crate::dist::Normal;
        let params_n = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let dist_n = Normal::from_params(&params_n);
        let score_n = Scorable::<CRPScore>::score(&dist_n, &y);

        // Student T with large df should be very close to Normal
        assert!(score_t[0].is_finite());
        assert!((score_t[0] - score_n[0]).abs() < 0.01);
    }
}

impl Scorable<CRPScore> for TFixedDfFixedVar {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Student's T with fixed df and fixed variance
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let mu = self.loc[i];
            let sigma = self.scale[i]; // Fixed at 1.0
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                scores[i] = 1e10;
                continue;
            }

            let z = (y_i - mu) / sigma;
            scores[i] = sigma * compute_t_crps_std(z, nu);
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient w.r.t. loc only
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                continue;
            }

            let z = (y_i - mu) / sigma;

            if let Ok(std_t) = StudentsTDist::new(0.0, 1.0, nu) {
                let big_f_z = std_t.cdf(z);
                // d(CRPS)/d(μ) = -(2*F(z) - 1)
                d_params[[i, 0]] = -(2.0 * big_f_z - 1.0);
            }
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
