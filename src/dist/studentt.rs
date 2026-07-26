use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use statrs::distribution::{
    Continuous, ContinuousCDF, Normal as NormalDist, StudentsT as StudentsTDist,
};
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
        let scale = crate::vmath::exp_column(&params.column(1));
        let var = &scale * &scale;
        let df = crate::vmath::exp_column(&params.column(2));
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
        // Inlined log-pdf (TLogPdf): no per-element StudentsTDist. The guard
        // reproduces the old constructor accept-set (rejects NaN loc, NaN or
        // non-positive scale/df), leaving those rows 0 as before.
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !loc.is_nan() && scale > 0.0 && df > 0.0 {
                result[i] = TLogPdf::new(df).pdf(y[i], loc, scale);
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Same inlined formula as Scorable<LogScore>::score (see TLogPdf);
        // invalid rows stay 0 exactly as the old StudentsTDist::new guard did.
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !loc.is_nan() && scale > 0.0 && df > 0.0 {
                result[i] = TLogPdf::new(df).ln_pdf(y[i], loc, scale);
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
        // Bounded quantile (see t_ppf_std): statrs inverse_cdf's unbounded
        // inv_beta_reg Newton loop is gone from the last user-facing surface.
        // Same [1e-15, 1−1e-15] q clamp (inside t_ppf_std), same
        // loc + scale·t_std structure; invalid rows stay 0 as before. df is
        // per-row here, so nothing can be hoisted (unlike the fixed-df dists).
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !loc.is_nan() && scale > 0.0 && df > 0.0 {
                result[i] = loc + scale * t_ppf_std(q[i], df);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        // Bailey polar draws: no per-draw statrs inverse_cdf (unbounded
        // root-finding, observed to hang on pathological df). Same validity
        // guard as the old StudentsTDist::new path — invalid params leave
        // that column zeroed.
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0 && df > 0.0) {
                continue;
            }
            for s in 0..n_samples {
                samples[[s, i]] = loc + scale * crate::vmath::sample_std_t(&mut rng, df);
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
    /// Sample from the distribution (Bailey's polar method).
    /// Returns an array of shape (n_samples, n_obs) where each column is samples for one observation.
    pub fn sample(&self, n_samples: usize) -> Array2<f64> {
        DistributionMethods::sample(self, n_samples)
    }
}

impl Scorable<LogScore> for StudentT {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // TLogPdf inlines the same formula (same per-row ln_gamma cost) and
        // adds the df ≥ 1e8 normal-limit branch — without it score() drifted
        // from -logpdf() by ~2e-4 at df = 1e12 (ln_gamma cancellation).
        let mut scores = Array1::zeros(y.len());

        Zip::from(&mut scores)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .and(&self.df)
            .for_each(|s, &y_i, &loc, &scale, &df| {
                *s = -TLogPdf::new(df).ln_pdf(y_i, loc, scale);
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
        // Closed-form Fisher information in the internal (loc, log σ, log ν)
        // parameterization (Lange, Little & Taylor 1989, transformed to
        // log-scale coordinates):
        //   I[0,0] = (ν+1) / ((ν+3) σ²)
        //   I[1,1] = 2ν / (ν+3)
        //   I[1,2] = I[2,1] = −2ν / ((ν+1)(ν+3))
        //   I[2,2] = ν² [ψ′(ν/2) − ψ′((ν+1)/2)] / 4 − ν(ν+5) / (2(ν+1)(ν+3))
        //   I[0,1] = I[0,2] = 0 (odd integrands: t is symmetric about loc)
        // Verified against 2M-point PIT quadrature and 4M-sample MC of E[ggᵀ]
        // (max diff ~5e-6, the references' own error). Replaces a 200-point
        // midpoint quadrature that underestimated I[2,2] by 16–37% from tail
        // truncation and cost ~200 inverse-CDF calls per row.
        //
        // I[2,2] is a difference of two ~0.5 quantities decaying like ~3.5/ν²,
        // so the exact form loses to catastrophic cancellation for large ν
        // (≥1.6% wrong at ν=1e5, garbage at 1e6). Above ν=1e4 switch to the
        // asymptotic (3.5 − 13/ν)/ν², which agrees with the exact form to
        // ~3e-5 at the boundary and stays positive (Fisher is PD).
        use super::gamma::trigamma;
        let n_obs = self.loc.len();
        let mut fi = Array3::zeros((n_obs, 3, 3));

        for i in 0..n_obs {
            let var_i = self.var[i];
            let df = self.df[i];

            let i22 = if df > 1e4 {
                (3.5 - 13.0 / df) / (df * df)
            } else {
                df * df * 0.25 * (trigamma(df / 2.0) - trigamma((df + 1.0) / 2.0))
                    - df * (df + 5.0) / (2.0 * (df + 1.0) * (df + 3.0))
            };

            fi[[i, 0, 0]] = (df + 1.0) / ((df + 3.0) * var_i);
            fi[[i, 1, 1]] = 2.0 * df / (df + 3.0);
            fi[[i, 1, 2]] = -2.0 * df / ((df + 1.0) * (df + 3.0));
            fi[[i, 2, 1]] = fi[[i, 1, 2]];
            fi[[i, 2, 2]] = i22;
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

        for i in 0..n_obs {
            let mu = self.loc[i];
            let sigma = self.scale[i];
            let nu = self.df[i];
            let y_i = y[i];

            if nu <= 1.0 {
                // CRPS is undefined here and score() returns a flat 1e10
                // penalty. A zero gradient would strand the row permanently
                // (the boosting step could never move df back above 1), so
                // emit a small restoring gradient on log(df). The metric for
                // these rows is identity, so this passes through the natural
                // gradient unchanged.
                d_params[[i, 2]] = -1.0;
                continue;
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

                // d(CRPS)/d(log(ν)): FD below ν=1000, Richardson-extrapolated
                // A(z)/ν tail above (the raw FD sign-flips there; see
                // t_crps_dlognu).
                d_params[[i, 2]] = t_crps_dlognu(z, nu, sigma);
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Deterministic quadrature for E[g * gᵀ] over Z ~ t_ν.
        //
        // Change of variables z = tan(θ), θ midpoint-uniform on (−π/2, π/2):
        //   E[h(Z)] = ∫ h(z) f_ν(z) dz ≈ (π/n) Σ h(tanθ_j) f_ν(tanθ_j) sec²θ_j
        // No inverse CDF: the previous PIT rule (z = F⁻¹(u)) used statrs
        // `inverse_cdf`, whose root-finding can fail to converge on drifted ν
        // (observed: a fit stuck 30+ CPU-min inside ONE call) and cost ~100µs
        // per node. The tan rule needs only pdf/cdf and is 1-7 orders of
        // magnitude MORE accurate (validated vs 2M-point references:
        // rel err 1.6e-6 at ν=2.7, ~1e-11 for ν≥5; old rule ~2.5e-5 tail bias).
        // Accuracy degrades only as ν→1 (integrand tail ~z^{1−ν}): ~1e-3 at
        // ν=1.5, ~2e-2 at ν=1.2 — comparable to the old rule there, and the
        // score itself is guarded for ν≤1.
        let n_obs = self.loc.len();
        let n_params = 3;
        let n_points = 200;
        let pi = std::f64::consts::PI;

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
                let theta = -pi / 2.0 + pi * (j as f64 + 0.5) / n_points as f64;
                let z = theta.tan();
                let w = std_t.pdf(z) * (1.0 + z * z) * (pi / n_points as f64);
                if w == 0.0 || !w.is_finite() {
                    continue; // underflowed tail node contributes nothing
                }

                // g0 = d(CRPS)/d(μ) = -(2F(z)-1)
                let g0 = -(2.0 * std_t.cdf(z) - 1.0);

                // g1 = d(CRPS)/d(log(σ))
                let crps_std = compute_t_crps_std(z, nu);
                let g1 = sigma * crps_std + sigma * z * g0;

                // g2 = d(CRPS)/d(log(ν)): same guarded gradient as d_score
                let g2 = t_crps_dlognu(z, nu, sigma);

                let grads = [g0, g1, g2];
                for a in 0..n_params {
                    for b in 0..n_params {
                        fi[[i, a, b]] += w * grads[a] * grads[b];
                    }
                }
            }
        }

        fi
    }
}

/// Compute the CRPScore scale metric constant C_t(nu) = E_Z[h(Z)^2]
/// where h(z) = CRPS_std(z) - z*(2F(z)-1) (the "scale gradient kernel")
/// and Z ~ t_nu (standard t-distribution).
///
/// Quantile-free tan-rule quadrature (z = tan θ, midpoint over (−π/2, π/2)) —
/// the same scheme as the 3-param CRPS metric. The previous PIT rule called
/// statrs `inverse_cdf` 200× per invocation: the exact root-finding
/// hang/latency class eliminated everywhere else (reachable here through
/// `from_params_with_df` with a drifting custom df), and it carried a
/// 1.1e-3 (ν=3) .. 6.4e-3 (ν=1.5) tail-truncation bias the tan rule removes.
/// df is fixed for a TFixedDf model but metric() runs every boosting
/// iteration, so the last (df, C_t) pair is memoized.
fn compute_crps_scale_metric_constant(nu: f64) -> f64 {
    if nu <= 1.0 {
        return 1.0; // Fallback for Cauchy (CRPS undefined)
    }

    static CACHE: std::sync::Mutex<Option<(f64, f64)>> = std::sync::Mutex::new(None);
    if let Ok(guard) = CACHE.lock() {
        if let Some((cached_nu, cached_c)) = *guard {
            if cached_nu == nu {
                return cached_c;
            }
        }
    }

    let std_t = match StudentsTDist::new(0.0, 1.0, nu) {
        Ok(d) => d,
        Err(_) => return 1.0,
    };

    // C_nu constant
    let ln_b1 = ln_beta(0.5, nu - 0.5);
    let ln_b2 = ln_beta(0.5, nu / 2.0);
    let c_nu = 2.0 * nu.sqrt() * (ln_b1 - 2.0 * ln_b2).exp() / (nu - 1.0);

    let n_points = 200;
    let pi = std::f64::consts::PI;
    let mut integral = 0.0;
    for j in 0..n_points {
        let theta = -pi / 2.0 + pi * (j as f64 + 0.5) / n_points as f64;
        let z = theta.tan();
        let f_z = std_t.pdf(z);
        let w = f_z * (1.0 + z * z) * (pi / n_points as f64);
        if w == 0.0 || !w.is_finite() {
            continue; // underflowed tail node contributes nothing
        }

        // h(z) = CRPS_std(z) - z*(2F(z)-1) = 2*f(z)*(nu+z^2)/(nu-1) - C_nu
        let h = 2.0 * f_z * (nu + z * z) / (nu - 1.0) - c_nu;
        integral += w * h * h;
    }

    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some((nu, integral));
    }
    integral
}

/// d(CRPS)/d(log ν) for the t CRPS, = σ · d(CRPS_std)/d(log ν).
///
/// For ν ≤ 1000: central finite difference with relative step 1e-6
/// (bit-identical to the previous inline code; validated ≤ 1.5e-3 relative
/// vs 50-digit mpmath). Above that the FD is noise-dominated: crps_std's
/// constant term is `ln_beta(½, ν−½) − 2·ln_beta(½, ν/2)` — a difference of
/// ln_gamma values of magnitude ~ν·ln ν carrying ~1e-12..1e-10 absolute
/// rounding error, which divided by the 2e-6·ν step exceeds the true gradient
/// (which decays as A(z)/ν): at ν=1e4 the FD SIGN-FLIPS (−2.7e-7 vs true
/// +9.7e-7 at z=−1), at ν=1e6 it is ~1e4× too large. Since
/// d/dlogν = A(z)/ν + B(z)/ν² + ..., Richardson extrapolation from the
/// well-conditioned points ν=500, 1000 recovers A(z) exactly through the B
/// term (u(ν) = ν·g(ν) = A + B/ν ⇒ A = 2·u(1000) − u(500)), then g = A/ν.
/// Validated vs mpmath over z ∈ [−3, 2]: rel err ≤ 1e-2 at ν=1500 falling to
/// ≤ 9e-4 for ν ≥ 1e4 (old FD: 1.3 at ν=1e4, 8.5e4 at ν=1e6).
fn t_crps_dlognu(z: f64, nu: f64, sigma: f64) -> f64 {
    let eps = 1e-6;
    // Preserves the exact fp expression order of the previous inline code.
    let fd = |nu_c: f64| {
        let crps_plus = compute_t_crps_std(z, nu_c * (1.0 + eps));
        // The lower leg must not cross the ν ≤ 1 sentinel: for
        // ν ∈ (1, 1/(1−ε)] the central difference evaluated the 1e10 penalty
        // and returned ≈ −5e15·σ garbage — sometimes with the wrong sign —
        // while score() stayed finite (2026-07-23 audit). One-sided there.
        if nu_c * (1.0 - eps) <= 1.0 {
            let crps_at = compute_t_crps_std(z, nu_c);
            sigma * (crps_plus - crps_at) / eps
        } else {
            let crps_minus = compute_t_crps_std(z, nu_c * (1.0 - eps));
            sigma * nu_c * (crps_plus - crps_minus) / (2.0 * eps * nu_c)
        }
    };

    const NU_SWITCH: f64 = 1000.0;
    if nu <= NU_SWITCH {
        fd(nu)
    } else {
        let u1 = 500.0 * fd(500.0);
        let u2 = 1000.0 * fd(1000.0);
        (2.0 * u2 - u1) / nu
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

/// Bounded, guaranteed-terminating standard Student-t quantile.
///
/// Replaces statrs `StudentsT::inverse_cdf`, whose `inv_beta_reg` Newton loop
/// is UNBOUNDED and has been observed to hang on pathological df (ppf was the
/// last user-facing surface still calling it; the fit loop and sample() were
/// cured earlier). Strategy:
///   * clamp q to [1e-15, 1 − 1e-15] (this file's long-standing ppf clamp),
///   * symmetry: solve for q ≤ 0.5 only and reflect the upper half
///     (1 − q is exact for q ∈ [0.5, 1], Sterbenz),
///   * seed with the better of the Cornish–Fisher expansion (A&S 26.7.5,
///     around the closed-form erf-based normal quantile) and the power-law
///     tail asymptote,
///   * refine with Newton on the statrs t `cdf` (analytic `pdf` as the
///     derivative — both root-finding-free), safeguarded by a maintained
///     bisection bracket with downward expansion, HARD-capped at 100
///     iterations after which the bisection midpoint is returned.
/// The result is therefore the numerical inverse of the exact same cdf users
/// evaluate — with two guards added after the 2026-07-23 audit:
///   * statrs' t cdf carries a noise floor (~1e-13 at ν≈1e3, ~1e-9 at ν≈1e6).
///     On a one-signed noise plateau Newton micro-steps forever without ever
///     moving the upper bracket, and the old exhaustion fallback
///     `0.5*(lo+hi)` averaged with the untouched hi = 0.0 sentinel —
///     silently returning HALF the (already converged) iterate. Now the
///     iterate with the smallest |cdf(x)−p| seen is returned instead, and an
///     exactly-repeated cdf residual exits early (the step is below cdf
///     resolution, so no further progress is possible).
///   * For ν ≥ 1e5 the Cornish–Fisher seed (error O(ν⁻⁵)) is already far
///     more accurate than anything Newton against the noisy cdf can deliver
///     (statrs cdf error ~1e-9 there) — it is returned directly.
/// Accuracy: ≤ 6.7e-13 rel for ν ≤ 1e4 (audit grid vs scipy); above that,
/// bounded by statrs cdf noise / pdf (≲1e-9 abs) until the ν ≥ 1e5 seed
/// takeover restores near-machine accuracy.
pub(crate) fn t_ppf_std(q: f64, nu: f64) -> f64 {
    let q = q.clamp(1e-15, 1.0 - 1e-15);
    if nu.is_infinite() && nu > 0.0 {
        // statrs treats df = +inf as exactly normal in cdf/pdf; invert
        // consistently (Normal::inverse_cdf is closed-form erf-based — safe).
        return NormalDist::new(0.0, 1.0).unwrap().inverse_cdf(q);
    }
    if q == 0.5 {
        0.0
    } else if q > 0.5 {
        -t_ppf_std_neg_half(1.0 - q, nu)
    } else {
        t_ppf_std_neg_half(q, nu)
    }
}

/// Lower-half worker for [`t_ppf_std`]: 0 < p ≤ 0.5, finite ν; returns t ≤ 0.
fn t_ppf_std_neg_half(p: f64, nu: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;
    let std_t = match StudentsTDist::new(0.0, 1.0, nu) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };

    // Seed 1: Cornish–Fisher expansion about the normal quantile (A&S 26.7.5).
    let z = NormalDist::new(0.0, 1.0).unwrap().inverse_cdf(p);
    let z2 = z * z;
    let g1 = z * (z2 + 1.0) / 4.0;
    let g2 = z * ((5.0 * z2 + 16.0) * z2 + 3.0) / 96.0;
    let g3 = z * (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) / 384.0;
    let g4 = z * ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) / 92160.0;
    let inv_nu = 1.0 / nu;
    let mut x = (z + inv_nu * (g1 + inv_nu * (g2 + inv_nu * (g3 + inv_nu * g4)))).min(0.0);

    // statrs' t cdf noise reaches ~1e-9 by ν≈5e5 while the CF expansion's own
    // error is O(ν⁻⁵): beyond ν = 1e5 Newton against the noisy cdf can only
    // damage the seed (and its noise plateaus used to burn all 100 iterations
    // into the broken midpoint fallback — 2026-07-23 audit).
    if nu >= 1e5 {
        return x;
    }

    // Seed 2 (tail): p ≈ (C/ν)·ν^{(ν+1)/2}·|t|^{−ν} with C the t pdf
    // normalizer ⇒ ln|t| = (ln C + ((ν−1)/2)·ln ν − ln p) / ν.
    if p < 0.25 {
        let ln_c = ln_gamma((nu + 1.0) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * (nu * std::f64::consts::PI).ln();
        let ln_t = (ln_c + 0.5 * (nu - 1.0) * nu.ln() - p.ln()) / nu;
        if ln_t >= 354.0 {
            // |t|² overflows f64, so the cdf itself saturates and cannot be
            // inverted numerically; the asymptote (rel err O(ν/t²)) is the
            // best f64 answer. ln_t ≥ ~709.8 would overflow exp: saturate.
            return if ln_t < 709.0 {
                -ln_t.exp()
            } else {
                f64::NEG_INFINITY
            };
        }
        if ln_t.is_finite() {
            let tail = -ln_t.exp();
            if (std_t.cdf(tail) - p).abs() < (std_t.cdf(x) - p).abs() {
                x = tail;
            }
        }
    }

    // Safeguarded Newton inside a maintained bracket; HARD-capped at 100
    // iterations. F is convex on t < 0, so plain Newton already converges
    // monotonically from either side — the bracket, expansion, and cap are
    // the no-hang guarantee, not the usual path (measured ≤ 6 iterations).
    let eps4 = 4.0 * f64::EPSILON;
    let mut lo = f64::NEG_INFINITY; // no lower bracket endpoint known yet
    let mut hi = 0.0; // F(0) = 0.5 ≥ p always
    let mut best_x = x;
    let mut best_af = f64::INFINITY;
    let mut prev_f = f64::NAN;
    for _ in 0..100 {
        let f = std_t.cdf(x) - p;
        if f == 0.0 {
            return x;
        }
        if f.abs() < best_af {
            best_af = f.abs();
            best_x = x;
        }
        // An exactly-repeated residual means the step fell below the cdf's
        // resolution (noise plateau): no further progress is possible.
        if f == prev_f {
            return best_x;
        }
        prev_f = f;
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let d = std_t.pdf(x);
        let mut xn = if d > 0.0 && d.is_finite() {
            x - f / d
        } else {
            f64::NAN
        };
        // Converged when the Newton step is negligible (f ≈ 0 at x). Checked
        // BEFORE the bracket safeguard: at x == hi (or lo) a zero-length step
        // is "outside" the open bracket and would otherwise trigger a
        // pointless bisection restart costing ~50 iterations.
        if xn.is_finite() && (xn - x).abs() <= eps4 * xn.abs() {
            return xn;
        }
        if !(xn > lo && xn < hi) {
            // NaN or outside the bracket: bisect, or expand downward while no
            // lower endpoint is known yet.
            xn = if lo.is_finite() {
                0.5 * (lo + hi)
            } else if x >= -1.0 {
                -2.0
            } else {
                2.0 * x
            };
            if xn == x {
                return x; // bracket collapsed to adjacent floats
            }
        }
        x = xn;
    }
    // Exhaustion: the best evaluated iterate always beats an unevaluated
    // bisection midpoint (and NEVER average with the hi = 0.0 sentinel —
    // that returned half the true quantile on cdf-noise plateaus).
    if std_t.cdf(x).is_finite() && (std_t.cdf(x) - p).abs() < best_af {
        x
    } else {
        best_x
    }
}

/// df-dependent constants for the inlined Student-t log-pdf: the same formula
/// as `Scorable<LogScore>::score`, plus statrs' own edge branches (y = ±inf →
/// −inf log-density; df ≥ 1e8 → normal limit) so values match the previous
/// per-element `StudentsTDist` pdf/ln_pdf path to ulp level. Hoist one of
/// these per distribution when df is fixed: that removes the two `ln_gamma`
/// calls the old code paid per row.
#[derive(Clone, Copy)]
struct TLogPdf {
    df: f64,
    half_df_plus_1: f64,
    log_normalizer: f64,
}

impl TLogPdf {
    fn new(df: f64) -> Self {
        use statrs::function::gamma::ln_gamma;
        let half_df_plus_1 = (df + 1.0) / 2.0;
        let log_normalizer = if df >= 1e8 {
            0.0 // unused: ln_pdf takes the normal-limit branch
        } else {
            ln_gamma(half_df_plus_1) - ln_gamma(df / 2.0) - 0.5 * (df * std::f64::consts::PI).ln()
        };
        TLogPdf {
            df,
            half_df_plus_1,
            log_normalizer,
        }
    }

    #[inline]
    fn ln_pdf(&self, y: f64, loc: f64, scale: f64) -> f64 {
        if y.is_infinite() {
            return f64::NEG_INFINITY;
        }
        let diff = y - loc;
        if self.df >= 1e8 {
            // Normal limit, exactly as statrs' StudentsT::ln_pdf above this df
            // (the ln_gamma difference cancels catastrophically there).
            let d = diff / scale;
            return -0.5 * d * d - statrs::consts::LN_SQRT_2PI - scale.ln();
        }
        let z_sq = diff * diff / (self.df * (scale * scale));
        self.log_normalizer - scale.ln() - self.half_df_plus_1 * (1.0 + z_sq).ln()
    }

    #[inline]
    fn pdf(&self, y: f64, loc: f64, scale: f64) -> f64 {
        self.ln_pdf(y, loc, scale).exp()
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
        let scale = crate::vmath::exp_column(&params.column(1));
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
        // Mean is loc for df > 1; undefined for df <= 1 — we return NaN (the
        // mathematically honest answer; note scipy.stats.t returns inf there).
        // df=3 by default, but from_params_with_df allows any df.
        let mut result = self.loc.clone();
        for i in 0..result.len() {
            if self.df[i] <= 1.0 {
                result[i] = f64::NAN;
            }
        }
        result
    }

    fn variance(&self) -> Array1<f64> {
        // Variance is scale^2 * df / (df - 2) for df > 2, infinite for
        // 1 < df <= 2, undefined for df <= 1 — we return NaN (scipy returns inf).
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            result[i] = if self.df[i] > 2.0 {
                self.var[i] * self.df[i] / (self.df[i] - 2.0)
            } else if self.df[i] > 1.0 {
                f64::INFINITY
            } else {
                f64::NAN
            };
        }
        result
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Fixed df ⇒ the two ln_gamma calls are hoisted out of the row loop
        // (the old per-element StudentsTDist paid them per row).
        let mut result = Array1::zeros(y.len());
        if !(self.fixed_df > 0.0) {
            return result; // old behavior: constructor error left zeros
        }
        let lp = TLogPdf::new(self.fixed_df);
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                if !loc.is_nan() && scale > 0.0 {
                    *r = lp.pdf(y_i, loc, scale);
                }
            });
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Direct log-pdf with hoisted ln_gamma constants (no exp→ln
        // round-trip through the default pdf().ln()). Rows the old default
        // mapped to ln(0) = −inf (invalid params) stay −inf.
        let mut result = Array1::zeros(y.len());
        if !(self.fixed_df > 0.0) {
            result.fill(f64::NEG_INFINITY);
            return result;
        }
        let lp = TLogPdf::new(self.fixed_df);
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                *r = if !loc.is_nan() && scale > 0.0 {
                    lp.ln_pdf(y_i, loc, scale)
                } else {
                    f64::NEG_INFINITY
                };
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Hoisted standard-t: statrs cdf standardizes internally, so
        // evaluating the (0, 1, df) dist at z = (y − loc)/scale is the same
        // fp computation without a per-row constructor.
        let mut result = Array1::zeros(y.len());
        let std_t = match StudentsTDist::new(0.0, 1.0, self.fixed_df) {
            Ok(d) => d,
            Err(_) => return result,
        };
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                if !loc.is_nan() && scale > 0.0 {
                    *r = std_t.cdf((y_i - loc) / scale);
                }
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Bounded quantile (see t_ppf_std): no statrs inverse_cdf. Same
        // [1e-15, 1−1e-15] clamp (inside t_ppf_std); invalid rows stay 0.
        let mut result = Array1::zeros(q.len());
        if !(self.fixed_df > 0.0) {
            return result;
        }
        for i in 0..q.len() {
            let (loc, scale) = (self.loc[i], self.scale[i]);
            if !loc.is_nan() && scale > 0.0 {
                result[i] = loc + scale * t_ppf_std(q[i], self.fixed_df);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        // Bailey polar draws: no per-draw statrs inverse_cdf (unbounded
        // root-finding; see StudentT::sample). Invalid params leave that
        // column zeroed, matching the old StudentsTDist::new guard.
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0 && df > 0.0) {
                continue;
            }
            for s in 0..n_samples {
                samples[[s, i]] = loc + scale * crate::vmath::sample_std_t(&mut rng, df);
            }
        }
        samples
    }

    fn interval(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        // Hoisted quantiles (mirrors Normal::interval): fixed df ⇒ one
        // standard-t quantile per bound instead of one per row per bound.
        // Bit-identical to the default ppf-based path: same t_ppf_std
        // argument (clamp included), same loc + scale·z arithmetic, same
        // invalid-row zeros.
        let n = self.loc.len();
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);
        if self.fixed_df > 0.0 {
            let z_lo = t_ppf_std(alpha / 2.0, self.fixed_df);
            let z_hi = t_ppf_std(1.0 - alpha / 2.0, self.fixed_df);
            Zip::from(&mut lower)
                .and(&mut upper)
                .and(&self.loc)
                .and(&self.scale)
                .for_each(|lo, hi, &loc, &scale| {
                    if !loc.is_nan() && scale > 0.0 {
                        *lo = loc + scale * z_lo;
                        *hi = loc + scale * z_hi;
                    }
                });
        }
        (lower, upper)
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
        // Mean is loc for df > 1; undefined for df <= 1 — we return NaN (the
        // mathematically honest answer; note scipy.stats.t returns inf there).
        let mut result = self.loc.clone();
        for i in 0..result.len() {
            if self.df[i] <= 1.0 {
                result[i] = f64::NAN;
            }
        }
        result
    }

    fn variance(&self) -> Array1<f64> {
        // For df=3 (default), variance = 1 * 3 / (3 - 2) = 3. Guards mirror
        // TFixedDf: the unguarded formula returned NEGATIVE variance for
        // df < 2 via from_params_with_df (e.g. -1.0 at df = 1).
        let mut result = Array1::zeros(self.loc.len());
        for i in 0..self.loc.len() {
            result[i] = if self.df[i] > 2.0 {
                self.var[i] * self.df[i] / (self.df[i] - 2.0)
            } else if self.df[i] > 1.0 {
                f64::INFINITY
            } else {
                f64::NAN
            };
        }
        result
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Fixed df and scale ⇒ ln_gamma constants hoisted (see TLogPdf).
        let mut result = Array1::zeros(y.len());
        if !(self.fixed_df > 0.0) {
            return result; // old behavior: constructor error left zeros
        }
        let lp = TLogPdf::new(self.fixed_df);
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                if !loc.is_nan() && scale > 0.0 {
                    *r = lp.pdf(y_i, loc, scale);
                }
            });
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Direct log-pdf with hoisted constants; invalid rows −inf, matching
        // the old default pdf().ln() = ln(0).
        let mut result = Array1::zeros(y.len());
        if !(self.fixed_df > 0.0) {
            result.fill(f64::NEG_INFINITY);
            return result;
        }
        let lp = TLogPdf::new(self.fixed_df);
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                *r = if !loc.is_nan() && scale > 0.0 {
                    lp.ln_pdf(y_i, loc, scale)
                } else {
                    f64::NEG_INFINITY
                };
            });
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Hoisted standard-t; identical fp computation to the old per-row
        // constructed dist (statrs standardizes internally anyway).
        let mut result = Array1::zeros(y.len());
        let std_t = match StudentsTDist::new(0.0, 1.0, self.fixed_df) {
            Ok(d) => d,
            Err(_) => return result,
        };
        Zip::from(&mut result)
            .and(y)
            .and(&self.loc)
            .and(&self.scale)
            .for_each(|r, &y_i, &loc, &scale| {
                if !loc.is_nan() && scale > 0.0 {
                    *r = std_t.cdf((y_i - loc) / scale);
                }
            });
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Bounded quantile (see t_ppf_std): no statrs inverse_cdf. Same
        // [1e-15, 1−1e-15] clamp (inside t_ppf_std); invalid rows stay 0.
        let mut result = Array1::zeros(q.len());
        if !(self.fixed_df > 0.0) {
            return result;
        }
        for i in 0..q.len() {
            let (loc, scale) = (self.loc[i], self.scale[i]);
            if !loc.is_nan() && scale > 0.0 {
                result[i] = loc + scale * t_ppf_std(q[i], self.fixed_df);
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        // Bailey polar draws: no per-draw statrs inverse_cdf (unbounded
        // root-finding; see StudentT::sample). Invalid params leave that
        // column zeroed, matching the old StudentsTDist::new guard.
        let n_obs = self.loc.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let (loc, scale, df) = (self.loc[i], self.scale[i], self.df[i]);
            if !(loc.is_finite() && scale.is_finite() && scale > 0.0 && df > 0.0) {
                continue;
            }
            for s in 0..n_samples {
                samples[[s, i]] = loc + scale * crate::vmath::sample_std_t(&mut rng, df);
            }
        }
        samples
    }

    fn interval(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        // Hoisted quantiles (mirrors Normal::interval and TFixedDf::interval):
        // one standard-t quantile per bound. Bit-identical to the default
        // ppf-based path (same t_ppf_std argument, same loc + scale·z with
        // scale ≡ 1, same invalid-row zeros).
        let n = self.loc.len();
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);
        if self.fixed_df > 0.0 {
            let z_lo = t_ppf_std(alpha / 2.0, self.fixed_df);
            let z_hi = t_ppf_std(1.0 - alpha / 2.0, self.fixed_df);
            Zip::from(&mut lower)
                .and(&mut upper)
                .and(&self.loc)
                .and(&self.scale)
                .for_each(|lo, hi, &loc, &scale| {
                    if !loc.is_nan() && scale > 0.0 {
                        *lo = loc + scale * z_lo;
                        *hi = loc + scale * z_hi;
                    }
                });
        }
        (lower, upper)
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

    /// 2026-07-23 audit regression: on statrs t-cdf noise plateaus (ν ≳ 800)
    /// the Newton loop's upper bracket never moved off its hi = 0.0 sentinel
    /// and the exhaustion fallback `0.5*(lo+hi)` returned HALF the true
    /// quantile (0.10522 instead of 0.21041 at ν=806.92, q=0.5833). Near-
    /// median quantiles at large ν were also noise-dominated (abs err 4.6e-6
    /// at ν=1e5). Pins scipy.stats.t.ppf truths across both failing regimes
    /// plus surrounding coverage; all previously failed, all must now agree.
    #[test]
    fn test_t_ppf_std_noise_plateau_regression() {
        let cases: [(f64, f64, f64); 12] = [
            (806.92, 0.5833, 0.21041103450421875),
            (1713791.56, 0.29001844, -0.5533309555558769),
            (6.8e6, 0.25, -0.6744897862747489),
            (1e5, 0.500001, 2.5066345412842254e-6),
            (1e6, 0.499999, -2.5066289012237065e-6),
            (1e8, 0.4, -0.2533471038098203),
            (1e5, 0.01, -2.3263851653552683),
            (2e5, 0.999, 3.0902730573051986),
            (900.0, 0.42, -0.20195185609511526),
            (1200.0, 0.75, 0.6746942511662576),
            (850.0, 1e-6, -4.786612562082759),
            (5e4, 0.3, -0.5244038557675486),
        ];
        for (nu, q, truth) in cases {
            let got = t_ppf_std(q, nu);
            let rel = ((got - truth) / truth).abs();
            assert!(
                rel < 1e-7,
                "t_ppf_std({q}, {nu}) = {got:e}, scipy = {truth:e}, rel = {rel:e}"
            );
        }
    }

    /// 2026-07-23 audit regression: for ν ∈ (1, ≈1.0000010] the central-FD
    /// lower leg crossed the ν ≤ 1 CRPS sentinel (1e10) and the "gradient"
    /// became ≈ −5.0e15·σ (true value ~0.35, sign sometimes flipped) while
    /// score() stayed finite. The guarded one-sided FD must stay sane.
    #[test]
    fn test_studentt_crps_df_gradient_sentinel_window() {
        for nu in [1.0000005_f64, 1.0000009, 1.0000011, 1.000002] {
            for z in [-3.0_f64, -1.0, 0.5] {
                let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, nu.ln()]).unwrap();
                let dist = StudentT::from_params(&params);
                let y = Array1::from_vec(vec![z]);
                let d = Scorable::<CRPScore>::d_score(&dist, &y);
                let got = d[[0, 2]];
                assert!(
                    got.is_finite() && got.abs() < 10.0,
                    "nu={nu}, z={z}: d/dlognu = {got:e} (sentinel contamination)"
                );
            }
        }
    }

    /// Pins 50-digit mpmath truths for d(CRPS)/d(log ν) at σ=1, z=−1.
    /// The old raw FD SIGN-FLIPS at ν=1e4 (−2.68e-7 vs true +9.71e-7) and
    /// returns −1.35e-6 vs true +9.72e-8 at ν=1e5 — both fail these bounds
    /// decisively; the Richardson tail is within ~1e-3.
    #[test]
    fn test_studentt_crps_df_gradient_large_nu() {
        // (nu, truth at z = -1, rel tol)
        let cases: [(f64, f64, f64); 3] = [
            (2.7, -0.022967266589000517, 1e-6), // moderate nu: FD path, no regression
            (1e4, 9.7091040050627367e-7, 2e-2),
            (1e5, 9.7229033972779943e-8, 2e-2),
        ];
        for (nu, truth, tol) in cases {
            // params: [loc, log(scale), log(df)]; y = loc - scale => z = -1
            let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, nu.ln()]).unwrap();
            let dist = StudentT::from_params(&params);
            let y = Array1::from_vec(vec![-1.0]);
            let d = Scorable::<CRPScore>::d_score(&dist, &y);
            let got = d[[0, 2]];
            assert!(
                ((got - truth) / truth).abs() < tol,
                "nu={nu}: d/dlognu = {got:e}, expected {truth:e}"
            );
        }
    }

    #[test]
    fn test_fixed_df_moment_guards() {
        // Non-default fixed df via from_params_with_df must follow scipy
        // moment conventions: df <= 1 -> mean/variance NaN; 1 < df <= 2 ->
        // variance +inf (the unguarded TFixedDfFixedVar formula previously
        // returned variance = -1.0 at df = 1).
        let params = Array2::from_shape_vec((1, 2), vec![0.5, 0.0]).unwrap();

        let d = TFixedDf::from_params_with_df(&params, 1.0);
        assert!(d.mean()[0].is_nan());
        assert!(d.variance()[0].is_nan());

        let d = TFixedDf::from_params_with_df(&params, 1.5);
        assert_relative_eq!(d.mean()[0], 0.5, epsilon = 1e-15);
        assert_eq!(d.variance()[0], f64::INFINITY);

        let d = TFixedDf::from_params_with_df(&params, 3.0);
        assert_relative_eq!(d.variance()[0], 3.0, epsilon = 1e-12);

        let params1 = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let d = TFixedDfFixedVar::from_params_with_df(&params1, 1.0);
        assert!(d.mean()[0].is_nan());
        assert!(d.variance()[0].is_nan());

        let d = TFixedDfFixedVar::from_params_with_df(&params1, 1.5);
        assert_eq!(d.variance()[0], f64::INFINITY);

        let d = TFixedDfFixedVar::from_params_with_df(&params1, 3.0);
        assert_relative_eq!(d.variance()[0], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_studentt_crps_df_below_one_has_restoring_gradient() {
        // Rows whose predicted df drifts ≤ 1 get the flat 1e10 CRPS penalty;
        // the gradient must push log(df) back up (negative d/dlog(df)) so the
        // boosting step can recover instead of stranding the row forever.
        // params: [loc, log(scale), log(df)] — df = 0.5 for row 0, 3.0 for row 1.
        let params = Array2::from_shape_vec(
            (2, 3),
            vec![0.0, 0.0, (0.5_f64).ln(), 0.0, 0.0, (3.0_f64).ln()],
        )
        .unwrap();
        let dist = StudentT::from_params(&params);
        let y = ndarray::array![0.3, 0.3];

        let scores = Scorable::<CRPScore>::score(&dist, &y);
        assert_eq!(scores[0], 1e10);
        assert!(scores[1] < 1.0, "healthy row must have a real CRPS");

        let d = Scorable::<CRPScore>::d_score(&dist, &y);
        assert_eq!(d[[0, 0]], 0.0);
        assert_eq!(d[[0, 1]], 0.0);
        assert!(
            d[[0, 2]] < 0.0,
            "df≤1 row needs a negative log-df gradient (descent step raises df), got {}",
            d[[0, 2]]
        );
        // Healthy row unaffected: gradient is finite and not the sentinel
        assert!(d.row(1).iter().all(|v| v.is_finite()));

        // Metric for the penalized row is identity, so the natural gradient
        // preserves the restoring direction.
        let m = Scorable::<CRPScore>::metric(&dist);
        for a in 0..3 {
            for b in 0..3 {
                let expect = if a == b { 1.0 } else { 0.0 };
                assert_eq!(m[[0, a, b]], expect);
            }
        }
    }

    /// Reference E[ggᵀ] for the 3-param StudentT LogScore via high-resolution
    /// PIT midpoint quadrature (the method the closed form replaced, at a
    /// point count where its tail-truncation error is ~1e-3 instead of ~30%).
    fn studentt_fisher_quadrature(loc: f64, scale: f64, df: f64, n_points: usize) -> [[f64; 3]; 3] {
        let var = scale * scale;
        let std_t = StudentsTDist::new(0.0, 1.0, df).unwrap();
        let mut fi = [[0.0f64; 3]; 3];
        for j in 0..n_points {
            let u = (j as f64 + 0.5) / n_points as f64;
            let z = std_t.inverse_cdf(u);
            let diff = scale * z;
            let diff_sq = diff * diff;
            let denom = df * var + diff_sq;
            let g0 = -(df + 1.0) * diff / denom;
            let g1 = 1.0 - (df + 1.0) * diff_sq / denom;
            let term_1 = (df / 2.0) * digamma((df + 1.0) / 2.0);
            let term_2 = (-df / 2.0) * digamma(df / 2.0);
            let term_4_1 = (-df / 2.0) * (1.0 + diff_sq / (df * var)).ln();
            let term_4_2 = (df + 1.0) * diff_sq / (2.0 * (df * var) * (1.0 + diff_sq / (df * var)));
            let g2 = -(term_1 + term_2 - 0.5 + term_4_1 + term_4_2);
            let g = [g0, g1, g2];
            for a in 0..3 {
                for b in 0..3 {
                    fi[a][b] += g[a] * g[b];
                }
            }
        }
        for row in fi.iter_mut() {
            for v in row.iter_mut() {
                *v /= n_points as f64;
            }
        }
        let _ = loc; // Fisher is translation invariant; loc kept for clarity
        fi
    }

    #[test]
    fn test_studentt_logscore_metric_closed_form_vs_quadrature() {
        // Closed-form Fisher must match a 100k-point quadrature of E[ggᵀ]
        // (reference tail-truncation error ~1.5e-3) on every entry.
        for &(loc, scale, df) in &[(0.5, 1.3, 2.7), (-1.0, 0.6, 8.0), (2.0, 1.0, 20.0)] {
            let params =
                Array2::from_shape_vec((1, 3), vec![loc, (scale as f64).ln(), (df as f64).ln()])
                    .unwrap();
            let dist = StudentT::from_params(&params);
            let fi = Scorable::<LogScore>::metric(&dist);
            let reference = studentt_fisher_quadrature(loc, scale, df, 100_000);
            for a in 0..3 {
                for b in 0..3 {
                    let closed = fi[[0, a, b]];
                    let quad = reference[a][b];
                    if quad.abs() < 1e-8 {
                        // Odd integrands: exactly zero in closed form,
                        // ~fp-cancellation-zero in quadrature.
                        assert!(closed.abs() < 1e-12, "[{a},{b}] expected 0, got {closed}");
                    } else {
                        assert_relative_eq!(closed, quad, max_relative = 1e-2);
                    }
                }
            }
        }
    }

    /// CRPS metric E[ggᵀ] pinned to 2M-point PIT-quadrature references
    /// (scipy, dumped 2026-07-09). The tan-rule replacement's validation
    /// errors were 1.6e-6 (ν=2.7) down to ~1e-11 (ν≥5).
    /// Note E[g0²] = E[(2U−1)²] = 1/3 exactly.
    #[test]
    fn test_studentt_crps_metric_matches_reference() {
        // (nu, m11, m22, m12) at sigma = 0.7; m00 = 1/3 always.
        let cases = [
            (2.7, 3.9400582e-2, 3.670059478e-3, -1.132702258e-2),
            (5.0, 3.0838072e-2, 5.988928621e-4, -4.110146721e-3),
            (20.0, 2.5558790e-2, 2.358913637e-5, -7.509118801e-4),
        ];
        for &(nu, m11, m22, m12) in &cases {
            let params =
                Array2::from_shape_vec((1, 3), vec![0.0, 0.7_f64.ln(), (nu as f64).ln()]).unwrap();
            let d = StudentT::from_params(&params);
            let fi = Scorable::<CRPScore>::metric(&d);
            assert_relative_eq!(fi[[0, 0, 0]], 1.0 / 3.0, max_relative = 1e-4);
            assert_relative_eq!(fi[[0, 1, 1]], m11, max_relative = 1e-3);
            assert_relative_eq!(fi[[0, 2, 2]], m22, max_relative = 1e-3);
            assert_relative_eq!(fi[[0, 1, 2]], m12, max_relative = 1e-3);
        }
    }

    #[test]
    fn test_studentt_logscore_metric_known_values() {
        // I[1,1] = 2ν/(ν+3): classic value 1/2 at ν=1 (Cauchy log-scale Fisher).
        // I[0,0] = (ν+1)/((ν+3)σ²).
        let params = Array2::from_shape_vec((1, 3), vec![0.0, 2.0_f64.ln(), 1.0_f64.ln()]).unwrap();
        let dist = StudentT::from_params(&params);
        let fi = Scorable::<LogScore>::metric(&dist);
        assert_relative_eq!(fi[[0, 1, 1]], 0.5, max_relative = 1e-12);
        assert_relative_eq!(fi[[0, 0, 0]], 2.0 / (4.0 * 4.0), max_relative = 1e-12);
    }

    #[test]
    fn test_studentt_logscore_metric_large_df_branch() {
        // The asymptotic I[2,2] branch must join continuously at ν=1e4 and
        // stay strictly positive where the exact form would be cancellation
        // noise (possibly negative → indefinite "Fisher").
        let metric_at = |df: f64| {
            let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, df.ln()]).unwrap();
            let dist = StudentT::from_params(&params);
            Scorable::<LogScore>::metric(&dist)[[0, 2, 2]]
        };
        let below = metric_at(9.999e3);
        let above = metric_at(1.0001e4);
        assert_relative_eq!(below, above, max_relative = 1e-3);
        for &df in &[1e5, 1e6, 1e8] {
            let i22 = metric_at(df);
            assert!(i22 > 0.0, "I[2,2] must stay positive at df={df}, got {i22}");
            assert_relative_eq!(i22, 3.5 / (df * df), max_relative = 1e-3);
        }
    }

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

    // ---- bounded t quantile (t_ppf_std) ----

    /// At benign points statrs inverse_cdf converges fine; the bounded
    /// replacement must agree to 1e-9 relative.
    #[test]
    fn test_t_ppf_std_matches_statrs_at_benign_points() {
        for &(nu, q) in &[
            (3.0, 0.25),
            (3.0, 0.9),
            (5.0, 0.1),
            (10.0, 0.75),
            (2.0, 0.05),
        ] {
            let statrs_val = StudentsTDist::new(0.0, 1.0, nu).unwrap().inverse_cdf(q);
            let ours = t_ppf_std(q, nu);
            assert_relative_eq!(ours, statrs_val, max_relative = 1e-9);
        }
        // Median: exactly 0, as statrs returns exactly loc.
        assert_eq!(t_ppf_std(0.5, 3.0), 0.0);
    }

    /// Full-distribution ppf (loc + scale·t_std) must agree with the old
    /// statrs inverse_cdf path at benign points for all three t variants.
    #[test]
    fn test_ppf_matches_statrs_all_t_variants_benign() {
        let (loc, scale, df) = (1.5_f64, 0.8_f64, 3.0_f64);
        let q = array![0.25, 0.5, 0.9];

        let p3 = Array2::from_shape_vec((1, 3), vec![loc, scale.ln(), df.ln()]).unwrap();
        let d3 = StudentT::from_params(&p3);
        let p2 = Array2::from_shape_vec((1, 2), vec![loc, scale.ln()]).unwrap();
        let d2 = TFixedDf::from_params(&p2);
        let p1 = Array2::from_shape_vec((1, 1), vec![loc]).unwrap();
        let d1 = TFixedDfFixedVar::from_params(&p1);

        for &q_i in q.iter() {
            let q_arr = Array1::from_elem(1, q_i);
            let ref3 = StudentsTDist::new(d3.loc[0], d3.scale[0], d3.df[0])
                .unwrap()
                .inverse_cdf(q_i);
            assert_relative_eq!(d3.ppf(&q_arr)[0], ref3, max_relative = 1e-9);
            let ref2 = StudentsTDist::new(d2.loc[0], d2.scale[0], d2.fixed_df)
                .unwrap()
                .inverse_cdf(q_i);
            assert_relative_eq!(d2.ppf(&q_arr)[0], ref2, max_relative = 1e-9);
            let ref1 = StudentsTDist::new(d1.loc[0], 1.0, d1.fixed_df)
                .unwrap()
                .inverse_cdf(q_i);
            assert_relative_eq!(d1.ppf(&q_arr)[0], ref1, max_relative = 1e-9);
        }
    }

    /// ppf must invert the exact statrs cdf users evaluate: round-trip across
    /// the full validation grid (lower half via cdf, upper half via sf so the
    /// comparison is not limited by 1 − cdf cancellation).
    #[test]
    fn test_t_ppf_cdf_roundtrip_grid() {
        let nus = [1.01, 1.5, 2.0, 3.0, 10.0, 100.0, 1e4, 1e6];
        let qs = [
            1e-15,
            1e-12,
            1e-6,
            0.01,
            0.25,
            0.5,
            0.75,
            0.99,
            1.0 - 1e-6,
            1.0 - 1e-12,
        ];
        for &nu in &nus {
            let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();
            for &q in &qs {
                let x = t_ppf_std(q, nu);
                assert!(x.is_finite(), "nu={nu} q={q} -> {x}");
                if q == 0.5 {
                    assert_eq!(x, 0.0);
                    continue;
                }
                let (got, want) = if q < 0.5 {
                    (std_t.cdf(x), q)
                } else {
                    (std_t.sf(x), 1.0 - q)
                };
                assert_relative_eq!(got, want, max_relative = 1e-9);
            }
        }
    }

    /// scipy.stats.t.ppf pins from the 2026-07-16 prototype validation
    /// (worst rel err there: 1.7e-15; tolerance here also absorbs any
    /// statrs-vs-scipy cdf discrepancy).
    #[test]
    fn test_t_ppf_std_matches_scipy_reference() {
        let cases: &[(f64, f64, f64)] = &[
            (1.01, 1e-12, -2.4536101608566061e11),
            (1.01, 0.25, -0.99589217292454912),
            (1.5, 1e-6, -5.2194693247069436e3),
            (2.0, 0.75, 0.81649658092772615),
            (3.0, 0.01, -4.5407028585681335),
            (3.0, 1.0 - 1e-6, 1.0329946777942898e2),
            (10.0, 1e-15, -8.1040890880039342e1),
            (100.0, 1e-6, -5.0488308772283457),
            (1e4, 0.99, 2.3267208386694755),
            (1e6, 0.25, -0.67448999553108724),
        ];
        for &(nu, q, truth) in cases {
            let got = t_ppf_std(q, nu);
            assert_relative_eq!(got, truth, max_relative = 1e-9);
        }
    }

    /// The whole point of the rewrite: df values in the regime where statrs
    /// inverse_cdf's unbounded Newton has been observed to hang or
    /// misconverge must terminate AND still invert the cdf.
    #[test]
    fn test_t_ppf_pathological_df_terminates_and_inverts() {
        for &nu in &[0.2, 0.7, 1.0, 1.000001, 37.3, 12345.6] {
            let std_t = StudentsTDist::new(0.0, 1.0, nu).unwrap();
            for &q in &[1e-15, 1e-4, 0.31, 0.5, 0.77, 1.0 - 1e-10] {
                let x = t_ppf_std(q, nu);
                assert!(!x.is_nan(), "nu={nu} q={q} -> NaN");
                if q == 0.5 {
                    assert_eq!(x, 0.0);
                    continue;
                }
                let (got, want) = if q < 0.5 {
                    (std_t.cdf(x), q)
                } else {
                    (std_t.sf(x), 1.0 - q)
                };
                assert_relative_eq!(got, want, max_relative = 1e-8);
            }
        }
        // Saturating regime: |t|² exceeds f64, the cdf can no longer be
        // inverted numerically — the tail asymptote is returned (finite or
        // −inf), never NaN, never an unbounded loop.
        let x = t_ppf_std(1e-15, 0.06);
        assert!(x.is_finite() && x < -1e200, "got {x}");
        assert_eq!(t_ppf_std(1e-15, 0.02), f64::NEG_INFINITY);
        // df = +inf: statrs cdf is exactly normal there; so is the quantile.
        let z = t_ppf_std(0.025, f64::INFINITY);
        assert_relative_eq!(z, -1.9599639845400545, max_relative = 1e-9);
    }

    /// The hoisted-quantile interval() overrides must be bit-identical to the
    /// default ppf-based path (mirrors Normal's interval test).
    #[test]
    fn test_tfixeddf_interval_override_matches_ppf_path() {
        let params = Array2::from_shape_vec((3, 2), vec![0.5, 0.2, -1.0, -0.4, 2.0, 0.9]).unwrap();
        let dist = TFixedDf::from_params(&params);
        for &alpha in &[0.05, 0.1, 0.5] {
            let (lo, hi) = dist.interval(alpha);
            let lo_ref = dist.ppf(&Array1::from_elem(3, alpha / 2.0));
            let hi_ref = dist.ppf(&Array1::from_elem(3, 1.0 - alpha / 2.0));
            for i in 0..3 {
                assert_eq!(lo[i], lo_ref[i], "alpha={alpha} row={i}");
                assert_eq!(hi[i], hi_ref[i], "alpha={alpha} row={i}");
            }
        }
    }

    #[test]
    fn test_tfixeddffixedvar_interval_override_matches_ppf_path() {
        let params = Array2::from_shape_vec((3, 1), vec![0.5, -1.0, 2.0]).unwrap();
        let dist = TFixedDfFixedVar::from_params(&params);
        for &alpha in &[0.05, 0.1, 0.5] {
            let (lo, hi) = dist.interval(alpha);
            let lo_ref = dist.ppf(&Array1::from_elem(3, alpha / 2.0));
            let hi_ref = dist.ppf(&Array1::from_elem(3, 1.0 - alpha / 2.0));
            for i in 0..3 {
                assert_eq!(lo[i], lo_ref[i], "alpha={alpha} row={i}");
                assert_eq!(hi[i], hi_ref[i], "alpha={alpha} row={i}");
            }
        }
    }

    /// The inlined bulk pdf/logpdf/cdf must match the old per-element statrs
    /// StudentsTDist values to 1e-12, including the df ≥ 1e8 normal-limit
    /// branch and infinite y.
    #[test]
    fn test_inlined_pdf_logpdf_cdf_match_statrs() {
        let y = array![-3.0, -0.5, 0.0, 1.2, 4.0, f64::INFINITY, f64::NEG_INFINITY];

        // 3-param StudentT over varied (loc, scale, df)
        for &(loc, scale, df) in &[
            (0.5, 1.3, 2.7),
            (-1.0, 0.6, 8.0),
            (2.0, 1.0, 3.0),
            (0.0, 2.0, 1e9), // normal-limit branch
        ] {
            let params =
                Array2::from_shape_vec((1, 3), vec![loc, (scale as f64).ln(), (df as f64).ln()])
                    .unwrap();
            let dist = StudentT::from_params(&params);
            // exp∘ln round-trips of the params: use the struct's own values
            let d = StudentsTDist::new(dist.loc[0], dist.scale[0], dist.df[0]).unwrap();
            for &y_i in y.iter() {
                let y_arr = Array1::from_elem(1, y_i);
                assert_relative_eq!(
                    dist.pdf(&y_arr)[0],
                    d.pdf(y_i),
                    max_relative = 1e-12,
                    epsilon = 1e-300
                );
                let (got, want) = (dist.logpdf(&y_arr)[0], d.ln_pdf(y_i));
                if want.is_infinite() {
                    assert_eq!(got, want);
                } else {
                    assert_relative_eq!(got, want, max_relative = 1e-12, epsilon = 1e-12);
                }
                assert_relative_eq!(
                    dist.cdf(&y_arr)[0],
                    d.cdf(y_i),
                    max_relative = 1e-12,
                    epsilon = 1e-300
                );
            }
        }

        // Fixed-df variants (hoisted constants), incl. a non-default df
        for &fdf in &[3.0, 2.5, 1e9] {
            let p2 = Array2::from_shape_vec((1, 2), vec![0.7, -0.3]).unwrap();
            let d2 = TFixedDf::from_params_with_df(&p2, fdf);
            let ref2 = StudentsTDist::new(d2.loc[0], d2.scale[0], fdf).unwrap();
            let p1 = Array2::from_shape_vec((1, 1), vec![0.7]).unwrap();
            let d1 = TFixedDfFixedVar::from_params_with_df(&p1, fdf);
            let ref1 = StudentsTDist::new(d1.loc[0], 1.0, fdf).unwrap();
            for &y_i in y.iter() {
                let y_arr = Array1::from_elem(1, y_i);
                for (got, want) in [
                    (d2.pdf(&y_arr)[0], ref2.pdf(y_i)),
                    (d2.cdf(&y_arr)[0], ref2.cdf(y_i)),
                    (d1.pdf(&y_arr)[0], ref1.pdf(y_i)),
                    (d1.cdf(&y_arr)[0], ref1.cdf(y_i)),
                ] {
                    assert_relative_eq!(got, want, max_relative = 1e-12, epsilon = 1e-300);
                }
                for (got, want) in [
                    (d2.logpdf(&y_arr)[0], ref2.ln_pdf(y_i)),
                    (d1.logpdf(&y_arr)[0], ref1.ln_pdf(y_i)),
                ] {
                    if want.is_infinite() {
                        assert_eq!(got, want);
                    } else {
                        assert_relative_eq!(got, want, max_relative = 1e-12, epsilon = 1e-12);
                    }
                }
            }
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
