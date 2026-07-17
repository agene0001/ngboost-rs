use crate::dist::{Distribution, RegressionDistn};
use crate::scores::{LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, s};
use ndarray_linalg::Inverse;

/// Get the lower triangular indices for a p x p matrix.
/// Returns (row_indices, col_indices, diagonal_mask).
fn tril_indices(p: usize) -> (Vec<usize>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for i in 0..p {
        for j in 0..=i {
            rows.push(i);
            cols.push(j);
        }
    }
    (rows, cols)
}

/// Get the number of lower triangular elements for a p x p matrix.
fn tril_size(p: usize) -> usize {
    p * (p + 1) / 2
}

/// Build the Cholesky factor L from flattened lower triangular values.
/// The diagonal elements are exponentiated to ensure positive definiteness.
fn build_cholesky_factor(tril_vals: &Array2<f64>, p: usize) -> Array3<f64> {
    // tril_vals is (lower_size, N)
    let n = tril_vals.ncols();
    let mut l = Array3::zeros((n, p, p));

    let (rows, cols) = tril_indices(p);

    for (par_idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
        for i in 0..n {
            let val = tril_vals[[par_idx, i]];
            if row == col {
                // Diagonal: exponentiate to ensure positive
                l[[i, row, col]] = val.exp() + 1e-6;
            } else {
                l[[i, row, col]] = val;
            }
        }
    }

    l
}

/// A P-dimensional Multivariate Normal distribution.
///
/// Uses the parameterization Σ^(-1) = L * L^T where L is lower triangular.
/// The diagonal of L is modeled on the log scale.
///
/// Number of parameters: P + P*(P+1)/2 = P*(P+3)/2
/// - First P parameters are the mean
/// - Remaining P*(P+1)/2 are the lower triangle of L
#[derive(Debug, Clone)]
pub struct MultivariateNormal<const P: usize> {
    /// The mean vector (N x P).
    pub loc: Array2<f64>,
    /// The Cholesky factor L (N x P x P).
    pub l: Array3<f64>,
    /// The precision matrix (inverse covariance): Σ^(-1) = L * L^T (N x P x P).
    pub cov_inv: Array3<f64>,
    /// Cached covariance matrix (computed lazily).
    cov: Option<Array3<f64>>,
    /// Number of observations.
    n_obs: usize,
}

impl<const P: usize> MultivariateNormal<P> {
    /// Number of parameters for this distribution.
    pub const N_PARAMS: usize = P * (P + 3) / 2;

    /// PDF normalization constant: -P/2 * log(2*pi)
    fn pdf_constant() -> f64 {
        -(P as f64) / 2.0 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Compute the log determinant of L (sum of log of diagonal elements).
    fn log_det_l(&self, obs_idx: usize) -> f64 {
        let mut log_det = 0.0;
        for i in 0..P {
            log_det += self.l[[obs_idx, i, i]].ln();
        }
        log_det
    }

    /// Get the covariance matrix (computed lazily).
    pub fn cov(&mut self) -> &Array3<f64> {
        if self.cov.is_none() {
            self.cov = Some(crate::dist::MultivariateDistributionMethods::covariance(
                self,
            ));
        }
        self.cov.as_ref().unwrap()
    }

    /// Compute residual and eta = L^T * (loc - y) for each observation.
    fn summaries(&self, y: &Array2<f64>) -> (Array3<f64>, Array2<f64>) {
        // y is (N, P)
        // diff: (N, P, 1)
        // eta: (N, P) = L^T @ diff
        let n = self.n_obs;
        let mut diff = Array3::zeros((n, P, 1));
        let mut eta = Array2::zeros((n, P));

        for i in 0..n {
            for j in 0..P {
                diff[[i, j, 0]] = self.loc[[i, j]] - y[[i, j]];
            }

            // eta = L^T @ diff
            for j in 0..P {
                let mut sum = 0.0;
                for k in 0..P {
                    sum += self.l[[i, k, j]] * diff[[i, k, 0]];
                }
                eta[[i, j]] = sum;
            }
        }

        (diff, eta)
    }

    /// Compute log PDF for multivariate normal.
    pub fn logpdf(&self, y: &Array2<f64>) -> Array1<f64> {
        let (_, eta) = self.summaries(y);
        let mut logpdf = Array1::zeros(self.n_obs);

        for i in 0..self.n_obs {
            // Quadratic form: -0.5 * ||eta||^2
            let mut quad = 0.0;
            for j in 0..P {
                quad += eta[[i, j]] * eta[[i, j]];
            }
            let p1 = -0.5 * quad;

            // Log determinant term
            let p2 = self.log_det_l(i);

            logpdf[i] = p1 + p2 + Self::pdf_constant();
        }

        logpdf
    }
}

impl<const P: usize> Distribution for MultivariateNormal<P> {
    fn from_params(params: &Array2<f64>) -> Self {
        // params is (N, n_params) where n_params = P + P*(P+1)/2
        let n_obs = params.nrows();

        // Extract mean (first P columns)
        let loc = params.slice(s![.., 0..P]).to_owned();

        // Extract lower triangular values (remaining columns)
        // Need to transpose to (tril_len, N) for build_cholesky_factor
        let tril_params = params.slice(s![.., P..]).t().to_owned();
        let l = build_cholesky_factor(&tril_params, P);

        // Compute precision matrix: cov_inv = L @ L^T
        let mut cov_inv = Array3::zeros((n_obs, P, P));
        for i in 0..n_obs {
            for r in 0..P {
                for c in 0..P {
                    let mut sum = 0.0;
                    for k in 0..P {
                        sum += l[[i, r, k]] * l[[i, c, k]];
                    }
                    cov_inv[[i, r, c]] = sum;
                }
            }
        }

        MultivariateNormal {
            loc,
            l,
            cov_inv,
            cov: None,
            n_obs,
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let total_len = y.len();
        if total_len == 0 || total_len % P != 0 {
            return Array1::zeros(Self::N_PARAMS);
        }

        let n = total_len / P;
        if n < 2 {
            return Array1::zeros(Self::N_PARAMS);
        }
        // Reshape to (N, P)
        let mut y_2d = Array2::zeros((n, P));
        for i in 0..n {
            for j in 0..P {
                y_2d[[i, j]] = y[i * P + j];
            }
        }

        // Compute sample mean
        let mut mean = Array1::zeros(P);
        for j in 0..P {
            let mut sum = 0.0;
            for i in 0..n {
                sum += y_2d[[i, j]];
            }
            mean[j] = sum / n as f64;
        }

        // Compute sample covariance (biased, using N like Python)
        let mut cov = Array2::zeros((P, P));
        for i in 0..n {
            for r in 0..P {
                for c in 0..P {
                    cov[[r, c]] += (y_2d[[i, r]] - mean[r]) * (y_2d[[i, c]] - mean[c]);
                }
            }
        }
        cov /= n as f64; // Python uses N, not N-1

        // Compute precision matrix (inverse of covariance)
        // Add small regularization only if inversion fails
        let cov_inv = match cov.inv() {
            Ok(inv) => inv,
            Err(_) => {
                // Add regularization and retry
                for j in 0..P {
                    cov[[j, j]] += 1e-6;
                }
                cov.inv().unwrap_or_else(|_| Array2::eye(P))
            }
        };

        // Cholesky decomposition of precision: cov_inv = L @ L^T
        let l = match cholesky_lower(&cov_inv) {
            Some(l) => l,
            None => Array2::eye(P),
        };

        // Build parameter vector: [mean..., tril(L) with log-diagonal...]
        let tril_len = tril_size(P);
        let mut params = Array1::zeros(P + tril_len);

        // Mean parameters
        for j in 0..P {
            params[j] = mean[j];
        }

        // Lower triangular parameters (diagonal stored as log, matching Python exactly)
        let (rows, cols) = tril_indices(P);
        for (par_idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            if row == col {
                // Diagonal: store log(L_ii) directly like Python
                params[P + par_idx] = l[[row, col]].ln();
            } else {
                params[P + par_idx] = l[[row, col]];
            }
        }

        params
    }
    fn n_params(&self) -> usize {
        Self::N_PARAMS
    }

    fn predict(&self) -> Array1<f64> {
        // Return the first dimension of the mean for each observation
        // (For multi-output, this is a simplification)
        self.loc.column(0).to_owned()
    }

    fn params(&self) -> Array2<f64> {
        let n_params = Self::N_PARAMS;
        let mut p = Array2::zeros((self.n_obs, n_params));
        // First P columns: loc (identity transform)
        for j in 0..P {
            for i in 0..self.n_obs {
                p[[i, j]] = self.loc[[i, j]];
            }
        }
        // Remaining columns: lower triangle of L
        // Diagonal elements are log-transformed (inverse of exp in build_cholesky_factor)
        let (rows, cols) = tril_indices(P);
        for (par_idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            for i in 0..self.n_obs {
                let val = self.l[[i, row, col]];
                if row == col {
                    // Diagonal: apply log (inverse of exp + 1e-6)
                    p[[i, P + par_idx]] = (val - 1e-6).ln();
                } else {
                    p[[i, P + par_idx]] = val;
                }
            }
        }
        p
    }
}

impl<const P: usize> RegressionDistn for MultivariateNormal<P> {}

impl<const P: usize> crate::dist::MultivariateDistributionMethods for MultivariateNormal<P> {
    fn mean(&self) -> Array2<f64> {
        self.loc.clone()
    }

    fn covariance(&self) -> Array3<f64> {
        // Σ = (L Lᵀ)⁻¹, inverted per observation; identity fallback on
        // singular precision (mirrors the lazy `cov()` cache path).
        let mut cov = Array3::zeros((self.n_obs, P, P));
        for i in 0..self.n_obs {
            let cov_inv_i = self.cov_inv.slice(s![i, .., ..]).to_owned();
            if let Ok(cov_i) = cov_inv_i.inv() {
                cov.slice_mut(s![i, .., ..]).assign(&cov_i);
            } else {
                for r in 0..P {
                    cov[[i, r, r]] = 1.0;
                }
            }
        }
        cov
    }

    fn sample(&self, n_samples: usize) -> Array3<f64> {
        // Mirrors Python ngboost's MVN.rv(): x = loc + chol(Σ) · z with
        // z ~ N(0, I). chol(Σ) is computed once per observation.
        use crate::dist::MultivariateDistributionMethods;
        let cov = MultivariateDistributionMethods::covariance(self);
        let mut chols: Vec<Array2<f64>> = Vec::with_capacity(self.n_obs);
        for i in 0..self.n_obs {
            let cov_i = cov.slice(s![i, .., ..]).to_owned();
            // Identity fallback matches covariance()'s singular handling.
            chols.push(cholesky_lower(&cov_i).unwrap_or_else(|| Array2::eye(P)));
        }

        let mut rng = rand::rng();
        let mut std_z = crate::vmath::StdNormalSampler::new();
        let mut samples = Array3::zeros((n_samples, self.n_obs, P));
        let mut z = [0.0f64; 32]; // P ≤ 32 in practice; assert below
        assert!(P <= 32, "MVN sampling supports up to 32 dimensions");
        for s_idx in 0..n_samples {
            for i in 0..self.n_obs {
                for zj in z.iter_mut().take(P) {
                    // Direct polar Box-Muller draw (no per-draw inverse-erf).
                    *zj = std_z.next(&mut rng);
                }
                let lc = &chols[i];
                for r in 0..P {
                    let mut v = self.loc[[i, r]];
                    for (c, zc) in z.iter().enumerate().take(r + 1) {
                        v += lc[[r, c]] * zc;
                    }
                    samples[[s_idx, i, r]] = v;
                }
            }
        }
        samples
    }
}

impl<const P: usize> Scorable<LogScore> for MultivariateNormal<P> {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Reshape y from (N*P,) to (N, P) if needed
        // For now, assume y is already (N,) where N is n_obs
        // and we need 2D y for MVN

        // This is a simplification - in practice, y should be 2D
        // We'll reshape assuming y contains all P dimensions for each observation
        let n = self.n_obs;
        let mut y_2d = Array2::zeros((n, P));

        if y.len() == n * P {
            for i in 0..n {
                for j in 0..P {
                    y_2d[[i, j]] = y[i * P + j];
                }
            }
        } else if y.len() == n {
            // Single dimension - replicate
            for i in 0..n {
                y_2d[[i, 0]] = y[i];
            }
        }

        -self.logpdf(&y_2d)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Reshape y
        let n = self.n_obs;
        let mut y_2d = Array2::zeros((n, P));

        if y.len() == n * P {
            for i in 0..n {
                for j in 0..P {
                    y_2d[[i, j]] = y[i * P + j];
                }
            }
        } else if y.len() == n {
            for i in 0..n {
                y_2d[[i, 0]] = y[i];
            }
        }

        let (diff, eta) = self.summaries(&y_2d);
        let tril_len = tril_size(P);
        let n_params = P + tril_len;
        let mut gradient = Array2::zeros((n, n_params));

        let (rows, cols) = tril_indices(P);

        for i in 0..n {
            // Gradient of the mean: L @ eta
            for j in 0..P {
                let mut sum = 0.0;
                for k in 0..P {
                    sum += self.l[[i, j, k]] * eta[[i, k]];
                }
                gradient[[i, j]] = sum;
            }

            // Gradient of the lower triangular elements
            for (par_idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
                if row == col {
                    // Diagonal: L_ii = exp(θ) + 1e-6 (singularity guard in
                    // get_chol_factor), so ∂L_ii/∂θ = exp(θ) = L_ii − 1e-6.
                    // Differentiate THROUGH the guard:
                    //   d/dθ [½‖η‖²]     = diff_i · η_i · (L_ii − 1e-6)
                    //   d/dθ [−log L_ii] = −(L_ii − 1e-6)/L_ii
                    // Python uses L_ii for both (treats the guard as constant),
                    // making its gradient inconsistent with its own loss by
                    // ~1e-6·|η_i·diff_i| on every row — deliberate deviation.
                    let l_ii = self.l[[i, row, row]];
                    let dl_dtheta = l_ii - 1e-6;
                    gradient[[i, P + par_idx]] =
                        diff[[i, row, 0]] * eta[[i, row]] * dl_dtheta - dl_dtheta / l_ii;
                } else {
                    // Off-diagonal: d/d(L_ij) = eta_j * diff_i
                    gradient[[i, P + par_idx]] = eta[[i, col]] * diff[[i, row, 0]];
                }
            }
        }

        gradient
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for MVN
        // Formulas obtained by taking the expectation of the Hessian
        // (as noted in the Python implementation)
        let tril_len = tril_size(P);
        let n_params = P + tril_len;
        let mut fi = Array3::zeros((self.n_obs, n_params, n_params));

        let (rows, cols) = tril_indices(P);

        // Identify diagonal and off-diagonal indices
        let mut diags = Vec::new();
        let mut off_diags = Vec::new();
        for (par_idx, (&row, &col)) in rows.iter().zip(cols.iter()).enumerate() {
            if row == col {
                diags.push(par_idx);
            } else {
                off_diags.push(par_idx);
            }
        }

        for obs in 0..self.n_obs {
            // Initialize diagonal to 1 for stability
            for j in 0..n_params {
                fi[[obs, j, j]] = 1.0;
            }

            // FI of the location: L @ L^T = cov_inv
            for r in 0..P {
                for c in 0..P {
                    fi[[obs, r, c]] = self.cov_inv[[obs, r, c]];
                }
            }

            // Compute covariance matrix for this observation (inverse of cov_inv)
            let cov_inv_obs = self.cov_inv.slice(s![obs, .., ..]).to_owned();
            let cov_obs = match cov_inv_obs.inv() {
                Ok(inv) => inv,
                Err(_) => {
                    // Fallback to identity if singular
                    Array2::eye(P)
                }
            };

            // Compute L^T @ cov for the variance component FI
            // cov_sum[i,j] = sum_k L[k,i] * cov[k,j]
            let mut cov_sum = Array2::zeros((P, P));
            for i in 0..P {
                for j in 0..P {
                    let mut sum = 0.0;
                    for k in 0..P {
                        sum += self.l[[obs, k, i]] * cov_obs[[k, j]];
                    }
                    cov_sum[[i, j]] = sum;
                }
            }

            // Variance component of FI (VarComp in Python)
            // E[d^2l / dlog(a_ii) dlog(a_kk)] and E[d^2l / dlog(a_ii) da_kq]
            for &diag_idx in &diags {
                let i = rows[diag_idx];
                let l_ii = self.l[[obs, i, i]];
                // Diagonal-diagonal: L_ii^2 * cov_ii + cov_sum_ii * L_ii
                let value = l_ii * l_ii * cov_obs[[i, i]] + cov_sum[[i, i]] * l_ii;
                fi[[obs, P + diag_idx, P + diag_idx]] = value;

                // Diagonal-offdiagonal interactions
                for &par_idx in &off_diags {
                    let q = rows[par_idx];
                    let k = cols[par_idx];
                    if i == k {
                        let value = cov_obs[[q, i]] * l_ii;
                        fi[[obs, P + diag_idx, P + par_idx]] = value;
                        fi[[obs, P + par_idx, P + diag_idx]] = value;
                    }
                }
            }

            // Off-diagonal w.r.t. off-diagonal
            for &par_idx in &off_diags {
                let j = rows[par_idx];
                let i = cols[par_idx];
                for &par_idx2 in &off_diags {
                    let k = rows[par_idx2];
                    let q = cols[par_idx2];
                    if i == q {
                        let value = cov_obs[[k, j]];
                        fi[[obs, P + par_idx, P + par_idx2]] = value;
                        fi[[obs, P + par_idx2, P + par_idx]] = value;
                    }
                }
            }
        }

        fi
    }
}

/// Type alias for 2-dimensional MVN.
pub type MultivariateNormal2 = MultivariateNormal<2>;

/// Type alias for 3-dimensional MVN.
pub type MultivariateNormal3 = MultivariateNormal<3>;

/// Type alias for 4-dimensional MVN.
pub type MultivariateNormal4 = MultivariateNormal<4>;
/// Compute the lower Cholesky factor of a positive definite matrix.
/// Returns None if the matrix is not positive definite.
fn cholesky_lower(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag = a[[j, j]] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[[j, j]] = diag.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]].abs() < 1e-10 {
                    return None;
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Some(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::MultivariateDistributionMethods;
    use crate::scores::{LogScore, Scorable};

    /// Build a 2-obs MVN<2> with a known covariance: params place loc and
    /// the precision Cholesky L directly (diag entries are log-transformed).
    /// L = [[1,0],[−0.5,2]] ⇒ Σ⁻¹ = LLᵀ, Σ = (LLᵀ)⁻¹ known analytically.
    fn known_mvn2() -> (MultivariateNormal<2>, Array2<f64>) {
        // param layout: loc0, loc1, l00(log), l10, l11(log)
        // exp(log 1)+1e-6 ≈ 1, exp(log 2)+1e-6 ≈ 2 (eps-guard shifts by 1e-6)
        let row = [0.7, -1.2, 0.0_f64, -0.5, (2.0_f64).ln()];
        let vals: Vec<f64> = row.iter().chain(row.iter()).cloned().collect();
        let params = Array2::from_shape_vec((2, 5), vals).unwrap();
        let d = MultivariateNormal::<2>::from_params(&params);

        // Σ = inv(L Lᵀ) with the eps-guarded diagonal
        let l00 = 1.0 + 1e-6;
        let l11 = 2.0 + 1e-6;
        let l = ndarray::array![[l00, 0.0], [-0.5, l11]];
        let prec = l.dot(&l.t());
        let det = prec[[0, 0]] * prec[[1, 1]] - prec[[0, 1]] * prec[[1, 0]];
        let sigma = ndarray::array![
            [prec[[1, 1]] / det, -prec[[0, 1]] / det],
            [-prec[[1, 0]] / det, prec[[0, 0]] / det]
        ];
        (d, sigma)
    }

    #[test]
    fn test_mvn_mean_and_covariance() {
        let (d, sigma) = known_mvn2();
        let mean = MultivariateDistributionMethods::mean(&d);
        assert_eq!(mean.shape(), &[2, 2]);
        assert!((mean[[0, 0]] - 0.7).abs() < 1e-12);
        assert!((mean[[1, 1]] - (-1.2)).abs() < 1e-12);

        let cov = MultivariateDistributionMethods::covariance(&d);
        assert_eq!(cov.shape(), &[2, 2, 2]);
        for i in 0..2 {
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (cov[[i, r, c]] - sigma[[r, c]]).abs() < 1e-9,
                        "cov[{i},{r},{c}] = {} vs analytic {}",
                        cov[[i, r, c]],
                        sigma[[r, c]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_mvn_sample_moments_match() {
        let (d, sigma) = known_mvn2();
        let n = 60_000;
        let samples = MultivariateDistributionMethods::sample(&d, n);
        assert_eq!(samples.shape(), &[n, 2, 2]);

        // Empirical mean and covariance of obs 0's draws vs loc/Σ
        let loc = [0.7, -1.2];
        let mut m = [0.0f64; 2];
        for s in 0..n {
            for r in 0..2 {
                m[r] += samples[[s, 0, r]];
            }
        }
        m.iter_mut().for_each(|v| *v /= n as f64);
        for r in 0..2 {
            assert!(
                (m[r] - loc[r]).abs() < 0.05,
                "mean[{r}] = {} vs {}",
                m[r],
                loc[r]
            );
        }

        let mut c = [[0.0f64; 2]; 2];
        for s in 0..n {
            for r in 0..2 {
                for q in 0..2 {
                    c[r][q] += (samples[[s, 0, r]] - m[r]) * (samples[[s, 0, q]] - m[q]);
                }
            }
        }
        for r in 0..2 {
            for q in 0..2 {
                let emp = c[r][q] / n as f64;
                assert!(
                    (emp - sigma[[r, q]]).abs() < 0.05,
                    "cov[{r},{q}] = {emp} vs {}",
                    sigma[[r, q]]
                );
            }
        }
    }

    /// Regression test for the eps-guard gradient bug: the analytic d_score
    /// must be the exact derivative of the (eps-guarded) score. Python's MVN
    /// gradient treats the `exp(θ)+1e-6` diagonal guard as constant and is
    /// off by ~1e-6·|η·diff| on every row; ours differentiates through it.
    #[test]
    fn test_mvn_d_score_matches_own_score_fd() {
        // 4 obs, k=3 → 9 params: loc(3) + tril(6), diag log-scale.
        let vals = [
            0.5_f64, -1.0, 2.0, 0.3, -0.4, 0.2, 0.5, -0.3, -0.6, // obs 0
            -0.2, 0.8, 1.0, -0.5, 0.6, 0.4, -0.2, 0.1, 0.3, // obs 1
            1.5, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // obs 2 (identity-ish L)
            0.0, 0.0, 0.0, 0.6, 0.5, -0.6, 0.4, -0.5, 0.6, // obs 3
        ];
        let params = Array2::from_shape_vec((4, 9), vals.to_vec()).unwrap();
        let d = MultivariateNormal::<3>::from_params(&params);
        // y flattened row-major (n_obs * 3)
        let y = Array1::from_vec(vec![
            0.9, -0.4, 2.5, 0.1, 1.2, 0.2, 2.0, -0.8, -1.0, 0.5, -0.5, 0.7,
        ]);

        let analytic = Scorable::<LogScore>::d_score(&d, &y);

        let h = 1e-6;
        for j in 0..9 {
            let mut pp = params.clone();
            pp.column_mut(j).mapv_inplace(|v| v + h);
            let sp = Scorable::<LogScore>::score(&MultivariateNormal::<3>::from_params(&pp), &y);
            let mut pm = params.clone();
            pm.column_mut(j).mapv_inplace(|v| v - h);
            let sm = Scorable::<LogScore>::score(&MultivariateNormal::<3>::from_params(&pm), &y);
            for i in 0..4 {
                let fd = (sp[i] - sm[i]) / (2.0 * h);
                let diff = (analytic[[i, j]] - fd).abs();
                assert!(
                    diff < 1e-6,
                    "d_score[{i},{j}]={} vs FD {} (diff {diff:.2e})",
                    analytic[[i, j]],
                    fd
                );
            }
        }
    }
}
