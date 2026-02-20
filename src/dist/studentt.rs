use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
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
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Match Python: T.fit() calls scipy.stats.t.fit(Y, fdf=TFixedDf.fixed_df)
        // which fixes df=3.0 and does MLE for loc and scale.
        // We use IRLS to approximate the MLE, matching scipy's optimizer.
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0, 3.0_f64.ln()];
        }
        let nf = n as f64;
        let df = 3.0_f64;

        // Initial estimates: median for loc, MAD for scale
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut loc = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let mut scale = {
            let mut abs_devs: Vec<f64> = y.iter().map(|&v| (v - loc).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = if n % 2 == 0 {
                (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
            } else {
                abs_devs[n / 2]
            };
            (mad * 1.4826).max(1e-6)
        };

        // IRLS iterations for t(df=3) MLE
        for _ in 0..100 {
            let weights: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let z = (yi - loc) / scale;
                    (df + 1.0) / (df + z * z)
                })
                .collect();
            let sum_w: f64 = weights.iter().sum();

            let new_loc: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * yi)
                .sum::<f64>()
                / sum_w;

            let new_scale_sq: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * (yi - new_loc).powi(2))
                .sum::<f64>()
                / nf;

            let new_scale = new_scale_sq.sqrt().max(1e-6);

            if (new_loc - loc).abs() < 1e-10 && (new_scale - scale).abs() < 1e-10 {
                loc = new_loc;
                scale = new_scale;
                break;
            }
            loc = new_loc;
            scale = new_scale;
        }

        // Python returns [m, log(s), log(df)] where df is the MLE df.
        // Since we fix df=3 for the initial fit, return log(3).
        array![loc, scale.ln(), df.ln()]
    }

    fn n_params(&self) -> usize {
        3
    }

    fn predict(&self) -> Array1<f64> {
        // Mean of Student's T is loc (for df > 1)
        self.loc.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.loc.len();
        let mut p = Array2::zeros((n, 3));
        p.column_mut(0).assign(&self.loc);
        p.column_mut(1).assign(&self.scale.mapv(f64::ln));
        p.column_mut(2).assign(&self.df.mapv(f64::ln));
        p
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
        // Inlined Student-t log-PDF to avoid per-element StudentsTDist construction
        use statrs::function::gamma::ln_gamma;
        let mut scores = Array1::zeros(y.len());

        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .and(&self.var)
            .and(&self.df)
            .for_each(|s, &y_i, &loc, &scale, &var, &df| {
                let half_df_plus_1 = (df + 1.0) / 2.0;
                let log_normalizer = ln_gamma(half_df_plus_1)
                    - ln_gamma(df / 2.0)
                    - 0.5 * (df * std::f64::consts::PI).ln();
                let diff = y_i - loc;
                let z_sq = diff * diff / (df * var);
                let log_pdf = log_normalizer - scale.ln() - half_df_plus_1 * (1.0 + z_sq).ln();
                *s = -log_pdf;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 3));

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.var)
            .and(&self.df)
            .for_each(|mut row, &y_i, &loc, &var, &df| {
                let diff = y_i - loc;
                let diff_sq = diff * diff;
                let denom = df * var + diff_sq;
                let df_plus_1 = df + 1.0;

                row[0] = -df_plus_1 * diff / denom;
                row[1] = 1.0 - df_plus_1 * diff_sq / denom;

                let term_1 = (df / 2.0) * digamma(df_plus_1 / 2.0);
                let term_2 = (-df / 2.0) * digamma(df / 2.0);
                let term_3 = -0.5;
                let df_var = df * var;
                let term_4_1 = (-df / 2.0) * (1.0 + diff_sq / df_var).ln();
                let term_4_2_num = df_plus_1 * diff_sq;
                let term_4_2_den = 2.0 * df_var * (1.0 + diff_sq / df_var);

                row[2] = -(term_1 + term_2 + term_3 + term_4_1 + term_4_2_num / term_4_2_den);
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Deterministic quadrature for E[g * g^T].
        // Uses probability integral transform: u = F(z), z = F^{-1}(u)
        // so E[h(Z)] = ∫_0^1 h(F^{-1}(u)) du, computed via midpoint rule.
        let n_obs = self.loc.len();
        let n_params = 3;
        let n_points = 200;

        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            let loc_i = self.loc[i];
            let scale_i = self.scale[i];
            let var_i = self.var[i];
            let df_i = self.df[i];

            let std_t = match StudentsTDist::new(0.0, 1.0, df_i) {
                Ok(d) => d,
                Err(_) => continue,
            };

            for j in 0..n_points {
                let u = (j as f64 + 0.5) / n_points as f64;
                let z = std_t.inverse_cdf(u);
                let y_i = loc_i + scale_i * z;

                let diff = y_i - loc_i;
                let diff_sq = diff * diff;
                let denom = df_i * var_i + diff_sq;

                let g0 = -(df_i + 1.0) * diff / denom;
                let g1 = 1.0 - (df_i + 1.0) * diff_sq / denom;

                let term_1 = (df_i / 2.0) * digamma((df_i + 1.0) / 2.0);
                let term_2 = (-df_i / 2.0) * digamma(df_i / 2.0);
                let term_3 = -0.5;
                let term_4_1 = (-df_i / 2.0) * (1.0 + diff_sq / (df_i * var_i)).ln();
                let term_4_2_num = (df_i + 1.0) * diff_sq;
                let term_4_2_den = 2.0 * (df_i * var_i) * (1.0 + diff_sq / (df_i * var_i));
                let g2 = -(term_1 + term_2 + term_3 + term_4_1 + term_4_2_num / term_4_2_den);

                let grads = [g0, g1, g2];
                for a in 0..n_params {
                    for b in 0..n_params {
                        fi[[i, a, b]] += grads[a] * grads[b];
                    }
                }
            }
        }

        fi.mapv_inplace(|x| x / n_points as f64);
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
        // Deterministic quadrature for E[g * g^T].
        // Uses probability integral transform: u = F(z), z = F^{-1}(u)
        let n_obs = self.loc.len();
        let n_params = 3;
        let n_points = 200;
        let eps = 1e-6;

        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            let sigma = self.scale[i];
            let nu = self.df[i];

            if nu <= 1.0 {
                fi[[i, 0, 0]] = 1.0;
                fi[[i, 1, 1]] = 1.0;
                fi[[i, 2, 2]] = 1.0;
                continue;
            }

            let std_t = match StudentsTDist::new(0.0, 1.0, nu) {
                Ok(d) => d,
                Err(_) => continue,
            };

            for j in 0..n_points {
                let u = (j as f64 + 0.5) / n_points as f64;
                let z = std_t.inverse_cdf(u);

                // g0 = d(CRPS)/d(μ) = -(2F(z)-1)
                let g0 = -(2.0 * u - 1.0);

                // g1 = d(CRPS)/d(log(σ))
                let crps_std = compute_t_crps_std(z, nu);
                let g1 = sigma * crps_std + sigma * z * g0;

                // g2 = d(CRPS)/d(log(ν)) via finite difference on nu
                let nu_plus = nu * (1.0 + eps);
                let nu_minus = nu * (1.0 - eps);
                let crps_plus = compute_t_crps_std(z, nu_plus);
                let crps_minus = compute_t_crps_std(z, nu_minus);
                let g2 = sigma * nu * (crps_plus - crps_minus) / (2.0 * eps * nu);

                let grads = [g0, g1, g2];
                for a in 0..n_params {
                    for b in 0..n_params {
                        fi[[i, a, b]] += grads[a] * grads[b];
                    }
                }
            }
        }

        fi.mapv_inplace(|x| x / n_points as f64);
        fi
    }
}

/// Compute the CRPScore scale metric constant C_t(nu) = E_Z[h(Z)^2]
/// where h(z) = CRPS_std(z) - z*(2F(z)-1) (the "scale gradient kernel")
/// and Z ~ t_nu (standard t-distribution).
///
/// Uses Gauss-Legendre quadrature on a tanh-sinh transformed domain.
/// The result is independent of loc and scale.
fn compute_crps_scale_metric_constant(nu: f64) -> f64 {
    if nu <= 1.0 {
        return 1.0; // Fallback for Cauchy (CRPS undefined)
    }

    let std_t = match StudentsTDist::new(0.0, 1.0, nu) {
        Ok(d) => d,
        Err(_) => return 1.0,
    };

    // C_nu constant
    let ln_b1 = ln_beta(0.5, nu - 0.5);
    let ln_b2 = ln_beta(0.5, nu / 2.0);
    let c_nu = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);

    // Integrate h(z)^2 * f_nu(z) over (-inf, inf)
    // Use substitution z = sinh(t) to map to finite domain, then Gauss-Legendre.
    // Actually, simpler: use the t-distribution CDF to transform.
    // Let u = F_nu(z), then z = F_nu^{-1}(u), dz = 1/f_nu(z) du
    // Integral = ∫_0^1 h(F_nu^{-1}(u))^2 du
    //
    // This avoids density evaluation since f_nu(z)*dz = du.

    let n_points = 200;
    let mut integral = 0.0;

    for j in 0..n_points {
        // Gauss-Legendre on [0, 1]: use midpoint rule with many points
        let u = (j as f64 + 0.5) / n_points as f64;
        let z = std_t.inverse_cdf(u);
        let f_z = std_t.pdf(z);

        // h(z) = CRPS_std(z) - z*(2F(z)-1) = 2*f(z)*(nu+z^2)/(nu-1) - C_nu
        let h = 2.0 * f_z * (nu + z * z) / (nu - 1.0) - c_nu;

        // The integrand is h(z)^2 * f_nu(z) dz = h(F^{-1}(u))^2 du
        integral += h * h;
    }

    integral / n_points as f64
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
        }
    }
}

impl Distribution for TFixedDf {
    fn from_params(params: &Array2<f64>) -> Self {
        Self::from_params_with_df(params, Self::DEFAULT_FIXED_DF)
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Match Python: TFixedDf.fit() calls scipy.stats.t.fit(Y, fdf=TFixedDf.fixed_df)
        // which fixes df=3.0 and does MLE for loc and scale via IRLS.
        let n = y.len();
        if n == 0 {
            return array![0.0, 0.0];
        }
        let nf = n as f64;
        let df = Self::DEFAULT_FIXED_DF; // 3.0

        // Initial estimates: median for loc, MAD for scale
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut loc = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let mut scale = {
            let mut abs_devs: Vec<f64> = y.iter().map(|&v| (v - loc).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = if n % 2 == 0 {
                (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
            } else {
                abs_devs[n / 2]
            };
            (mad * 1.4826).max(1e-6)
        };

        // IRLS iterations for t(df=3) MLE
        for _ in 0..100 {
            let weights: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let z = (yi - loc) / scale;
                    (df + 1.0) / (df + z * z)
                })
                .collect();
            let sum_w: f64 = weights.iter().sum();

            let new_loc: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * yi)
                .sum::<f64>()
                / sum_w;

            let new_scale_sq: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * (yi - new_loc).powi(2))
                .sum::<f64>()
                / nf;

            let new_scale = new_scale_sq.sqrt().max(1e-6);

            if (new_loc - loc).abs() < 1e-10 && (new_scale - scale).abs() < 1e-10 {
                loc = new_loc;
                scale = new_scale;
                break;
            }
            loc = new_loc;
            scale = new_scale;
        }

        array![loc, scale.ln()]
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
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Fisher Information diagonal for T with fixed df (2 params: loc, log_scale)
        let df = self.fixed_df;
        let fi_00_factor = (df + 1.0) / (df + 3.0);
        let fi_11 = 2.0 * df / (df + 3.0);
        let n_obs = self.loc.len();
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.var)
            .for_each(|mut row, &var| {
                row[0] = fi_00_factor / var;
                row[1] = fi_11;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inlined Student-t log-PDF to avoid per-element StudentsTDist construction.
        // ln f(y|loc,scale,df) = lnΓ((df+1)/2) - lnΓ(df/2) - 0.5*ln(df*π) - ln(scale)
        //                        - ((df+1)/2)*ln(1 + (y-loc)²/(df*scale²))
        let mut scores = Array1::zeros(y.len());
        // For fixed df, precompute the constant terms once
        let df = self.fixed_df;
        let half_df_plus_1 = (df + 1.0) / 2.0;
        let log_normalizer = statrs::function::gamma::ln_gamma(half_df_plus_1)
            - statrs::function::gamma::ln_gamma(df / 2.0)
            - 0.5 * (df * std::f64::consts::PI).ln();

        ndarray::Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .and(&self.var)
            .for_each(|s, &y_i, &loc, &scale, &var| {
                let diff = y_i - loc;
                let z_sq = diff * diff / (df * var);
                let log_pdf = log_normalizer - scale.ln() - half_df_plus_1 * (1.0 + z_sq).ln();
                *s = -log_pdf;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));
        let df = self.fixed_df;
        let df_plus_1 = df + 1.0;

        ndarray::Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.var)
            .for_each(|mut row, &y_i, &loc, &var| {
                let diff = y_i - loc;
                let diff_sq = diff * diff;
                let denom = df * var + diff_sq;

                row[0] = -df_plus_1 * diff / denom;
                row[1] = 1.0 - df_plus_1 * diff_sq / denom;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for T with fixed df
        // Pre-compute df-dependent constants once
        let df = self.fixed_df;
        let fi_11 = 2.0 * df / (df + 3.0);
        let fi_00_factor = (df + 1.0) / (df + 3.0);

        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = fi_00_factor / self.var[i];
            fi[[i, 1, 1]] = fi_11;
        }

        fi
    }
}

impl Scorable<CRPScore> for TFixedDf {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // CRPScore metric diagonal for T with fixed df (2 params: loc, log_scale)
        // M[0,0] = 1/3, M[1,1] = C_t(nu) * sigma^2
        let n_obs = self.loc.len();
        let c_t = compute_crps_scale_metric_constant(self.fixed_df);
        let mut diag = Array2::zeros((n_obs, 2));
        Zip::from(diag.rows_mut())
            .and(&self.scale)
            .for_each(|mut row, &scale| {
                row[0] = 1.0 / 3.0;
                row[1] = c_t * scale * scale;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Student's T with fixed df (vectorized)
        // Precompute df-dependent constants once since df is fixed
        let nu = self.fixed_df;
        if nu <= 1.0 {
            return Array1::from_elem(y.len(), 1e10);
        }

        let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();
        let ln_b1 = ln_beta(0.5, nu - 0.5);
        let ln_b2 = ln_beta(0.5, nu / 2.0);
        let c_nu = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);
        let inv_nu_minus_1 = 1.0 / (nu - 1.0);

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|s, &y_i, &mu, &sigma| {
                let z = (y_i - mu) / sigma;
                let f_z = std_t.pdf(z);
                let big_f_z = std_t.cdf(z);
                let term1 = z * (2.0 * big_f_z - 1.0);
                let term2 = 2.0 * f_z * (nu + z * z) * inv_nu_minus_1;
                *s = sigma * (term1 + term2 - c_nu);
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient w.r.t. (loc, log(scale)) — vectorized
        let nu = self.fixed_df;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 2));

        if nu <= 1.0 {
            return d_params;
        }

        let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();
        let ln_b1 = ln_beta(0.5, nu - 0.5);
        let ln_b2 = ln_beta(0.5, nu / 2.0);
        let c_nu = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);
        let inv_nu_minus_1 = 1.0 / (nu - 1.0);

        Zip::from(d_params.rows_mut())
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|mut row, &y_i, &mu, &sigma| {
                let z = (y_i - mu) / sigma;
                let f_z = std_t.pdf(z);
                let big_f_z = std_t.cdf(z);

                // d(CRPS)/d(μ) = -(2F(z)-1)
                let g0 = -(2.0 * big_f_z - 1.0);
                row[0] = g0;

                // d(CRPS)/d(log(σ)) = σ * crps_std + (y-μ) * g0
                let crps_std =
                    z * (2.0 * big_f_z - 1.0) + 2.0 * f_z * (nu + z * z) * inv_nu_minus_1 - c_nu;
                row[1] = sigma * crps_std + (y_i - mu) * g0;
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Analytical CRPScore metric for T with fixed df.
        // M[0,0] = 1/3 (universal for any continuous distribution)
        // M[0,1] = M[1,0] = 0 (by symmetry: odd × even integrand)
        // M[1,1] = C_t(nu) * sigma^2 (computed via quadrature)
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 2, 2));

        let c_t = compute_crps_scale_metric_constant(self.fixed_df);

        for i in 0..n_obs {
            fi[[i, 0, 0]] = 1.0 / 3.0;
            fi[[i, 1, 1]] = c_t * self.scale[i] * self.scale[i];
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
        }
    }
}

impl Distribution for TFixedDfFixedVar {
    fn from_params(params: &Array2<f64>) -> Self {
        Self::from_params_with_df(params, Self::DEFAULT_FIXED_DF)
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Match Python: TFixedDfFixedVar.fit() calls
        // scipy.stats.t.fit(Y, fdf=TFixedDfFixedVar.fixed_df)
        // which fixes df=3.0 and scale, returning MLE loc.
        // We use IRLS with fixed scale=1.0 to match.
        let n = y.len();
        if n == 0 {
            return array![0.0];
        }
        let df = Self::DEFAULT_FIXED_DF; // 3.0
        let scale = 1.0;

        // Initial estimate: median
        let mut sorted: Vec<f64> = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut loc = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // IRLS iterations for t(df=3, scale=1) MLE of loc only
        for _ in 0..100 {
            let weights: Vec<f64> = y
                .iter()
                .map(|&yi| {
                    let z = (yi - loc) / scale;
                    (df + 1.0) / (df + z * z)
                })
                .collect();
            let sum_w: f64 = weights.iter().sum();

            let new_loc: f64 = y
                .iter()
                .zip(weights.iter())
                .map(|(&yi, &wi)| wi * yi)
                .sum::<f64>()
                / sum_w;

            if (new_loc - loc).abs() < 1e-10 {
                loc = new_loc;
                break;
            }
            loc = new_loc;
        }

        array![loc]
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
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Constant FI for fixed df and var=1: (df+1)/(df+3) (1 param)
        let df = self.fixed_df;
        let fi_val = (df + 1.0) / (df + 3.0);
        Array2::from_elem((self.loc.len(), 1), fi_val)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Inlined Student-t log-PDF for fixed df and fixed var (scale=1)
        // Precompute constants once since df is fixed
        let df = self.fixed_df;
        let half_df_plus_1 = (df + 1.0) / 2.0;
        let log_normalizer = statrs::function::gamma::ln_gamma(half_df_plus_1)
            - statrs::function::gamma::ln_gamma(df / 2.0)
            - 0.5 * (df * std::f64::consts::PI).ln();
        // scale=1, so -ln(scale) = 0

        let mut scores = Array1::zeros(y.len());
        ndarray::Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .for_each(|s, &y_i, &loc| {
                let diff = y_i - loc;
                let z_sq = diff * diff / df; // var=1
                let log_pdf = log_normalizer - half_df_plus_1 * (1.0 + z_sq).ln();
                *s = -log_pdf;
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // d/d(loc) for fixed df, fixed var=1
        // Simplified: -(df+1)*diff / (df + diff²)
        let df = self.fixed_df;
        let df_plus_1 = df + 1.0;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        ndarray::Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.loc)
            .for_each(|d, &y_i, &loc| {
                let diff = y_i - loc;
                // df*var = df*1 = df
                *d = -df_plus_1 * diff / (df + diff * diff);
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Constant for fixed df and var=1: (df+1)/((df+3)*var) = (df+1)/(df+3)
        let df = self.fixed_df;
        let fi_val = (df + 1.0) / (df + 3.0);
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| fi_val);
        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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

    #[test]
    fn test_studentt_logscore_matches_library() {
        // Verify inlined Student-t log-PDF matches statrs library computation
        let test_cases: Vec<(f64, f64, f64)> = vec![
            (0.0, 1.0, 3.0),   // standard
            (2.0, 0.5, 5.0),   // shifted, narrow, higher df
            (-1.0, 2.0, 2.5),  // negative loc, wide, low df
            (0.0, 1.0, 100.0), // near-normal
        ];
        let y_vals: Vec<f64> = vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0];

        for (loc, scale, df) in test_cases {
            let params = Array2::from_shape_vec((1, 3), vec![loc, scale.ln(), df.ln()]).unwrap();
            let dist = StudentT::from_params(&params);

            for &y_val in &y_vals {
                let y = Array1::from_vec(vec![y_val]);
                let score = Scorable::<LogScore>::score(&dist, &y);

                // Reference: library-based computation
                let d = StudentsTDist::new(loc, scale, df).unwrap();
                let expected = -d.ln_pdf(y_val);

                assert_relative_eq!(score[0], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_studentt_d_score_numerical() {
        // Verify StudentT LogScore d_score via numerical finite differences
        let eps = 1e-6;
        let loc = 1.0_f64;
        let log_scale = 0.5_f64;
        let log_df = 3.0_f64.ln();
        let y_val = 2.5;

        let params = Array2::from_shape_vec((1, 3), vec![loc, log_scale, log_df]).unwrap();
        let dist = StudentT::from_params(&params);
        let y = Array1::from_vec(vec![y_val]);
        let d_score = Scorable::<LogScore>::d_score(&dist, &y);

        // Numerical gradient for each parameter
        for p_idx in 0..3 {
            let mut p_plus = vec![loc, log_scale, log_df];
            let mut p_minus = vec![loc, log_scale, log_df];
            p_plus[p_idx] += eps;
            p_minus[p_idx] -= eps;

            let params_plus = Array2::from_shape_vec((1, 3), p_plus).unwrap();
            let params_minus = Array2::from_shape_vec((1, 3), p_minus).unwrap();
            let dist_plus = StudentT::from_params(&params_plus);
            let dist_minus = StudentT::from_params(&params_minus);
            let score_plus = Scorable::<LogScore>::score(&dist_plus, &y);
            let score_minus = Scorable::<LogScore>::score(&dist_minus, &y);
            let numerical = (score_plus[0] - score_minus[0]) / (2.0 * eps);

            assert_relative_eq!(d_score[[0, p_idx]], numerical, epsilon = 1e-4);
        }
    }
}

impl Scorable<CRPScore> for TFixedDfFixedVar {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // E[(2F(Z)-1)^2] = 1/3 universally for any continuous distribution (1 param)
        Array2::from_elem((self.loc.len(), 1), 1.0 / 3.0)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Student's T with fixed df and fixed var (scale=1) — vectorized
        let nu = self.fixed_df;
        if nu <= 1.0 {
            return Array1::from_elem(y.len(), 1e10);
        }

        let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();
        let ln_b1 = ln_beta(0.5, nu - 0.5);
        let ln_b2 = ln_beta(0.5, nu / 2.0);
        let c_nu = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);
        let inv_nu_minus_1 = 1.0 / (nu - 1.0);

        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .for_each(|s, &y_i, &mu| {
                let z = y_i - mu; // scale=1
                let f_z = std_t.pdf(z);
                let big_f_z = std_t.cdf(z);
                let term1 = z * (2.0 * big_f_z - 1.0);
                let term2 = 2.0 * f_z * (nu + z * z) * inv_nu_minus_1;
                *s = term1 + term2 - c_nu; // sigma=1
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient w.r.t. loc only — vectorized
        let nu = self.fixed_df;
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        if nu <= 1.0 {
            return d_params;
        }

        let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();

        Zip::from(d_params.column_mut(0))
            .and(y)
            .and(&self.loc)
            .for_each(|d, &y_i, &mu| {
                let z = y_i - mu; // scale=1
                let big_f_z = std_t.cdf(z);
                *d = -(2.0 * big_f_z - 1.0);
            });

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // E[(2F(Z)-1)^2] = 1/3 universally for any continuous distribution.
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        fi.mapv_inplace(|_| 1.0 / 3.0);
        fi
    }
}
