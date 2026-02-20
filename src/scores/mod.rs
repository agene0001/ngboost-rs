use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_linalg::{SVD, Solve};
use rayon::prelude::*;

/// A marker trait for scoring rules.
pub trait Score: 'static + Clone + Copy + Default {}

/// The Logarithmic Score (or Log-Likelihood).
#[derive(Clone, Copy, Default)]
pub struct LogScore;
impl Score for LogScore {}

/// The Continuous Ranked Probability Score.
#[derive(Clone, Copy, Default)]
pub struct CRPScore;
impl Score for CRPScore {}

/// Censored version of LogScore for survival analysis.
#[derive(Clone, Copy, Default)]
pub struct LogScoreCensored;
impl Score for LogScoreCensored {}

/// Censored version of CRPScore for survival analysis.
#[derive(Clone, Copy, Default)]
pub struct CRPScoreCensored;
impl Score for CRPScoreCensored {}

/// Survival data structure containing event indicators and times.
#[derive(Debug, Clone)]
pub struct SurvivalData {
    /// Event indicator: true if event occurred, false if censored
    pub event: Array1<bool>,
    /// Time to event or censoring
    pub time: Array1<f64>,
}

impl SurvivalData {
    /// Create new survival data from event indicators and times.
    pub fn new(event: Array1<bool>, time: Array1<f64>) -> Self {
        SurvivalData { event, time }
    }

    /// Create from separate arrays (converting f64 event to bool).
    pub fn from_arrays(time: &Array1<f64>, event: &Array1<f64>) -> Self {
        let event_bool = event.mapv(|e| e > 0.5);
        SurvivalData {
            event: event_bool,
            time: time.clone(),
        }
    }

    /// Create survival data assuming all events are observed (no censoring).
    pub fn uncensored(time: Array1<f64>) -> Self {
        let event = Array1::from_elem(time.len(), true);
        SurvivalData { event, time }
    }

    /// Get the number of observations.
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// A trait for censored scoring rules used in survival analysis.
pub trait CensoredScorable<S: Score> {
    /// Calculates the censored score for each observation.
    fn censored_score(&self, y: &SurvivalData) -> Array1<f64>;

    /// Calculates the gradient of the censored score with respect to the distribution's parameters.
    fn censored_d_score(&self, y: &SurvivalData) -> Array2<f64>;

    /// Calculates the Riemannian metric tensor for censored data.
    fn censored_metric(&self) -> Array3<f64>;

    /// Calculates the total censored score, averaged over all observations.
    fn total_censored_score(&self, y: &SurvivalData, sample_weight: Option<&Array1<f64>>) -> f64 {
        let scores = self.censored_score(y);
        if let Some(weights) = sample_weight {
            (scores * weights).sum() / weights.sum()
        } else {
            scores.mean().unwrap_or(0.0)
        }
    }

    /// Returns true if the censored metric is always diagonal.
    fn is_diagonal_censored_metric(&self) -> bool {
        false
    }

    /// Returns the diagonal of the censored metric as a flat Array2.
    /// Shape: (n_obs, n_params). Override for diagonal distributions.
    fn diagonal_censored_metric(&self) -> Array2<f64> {
        let metric = self.censored_metric();
        let n_obs = metric.shape()[0];
        let n_params = metric.shape()[1];
        let mut diag = Array2::zeros((n_obs, n_params));
        for j in 0..n_params {
            ndarray::Zip::from(diag.column_mut(j))
                .and(metric.slice(ndarray::s![.., j, j]))
                .for_each(|d, &m| {
                    *d = m;
                });
        }
        diag
    }

    /// Calculates the gradient for censored data, optionally the natural gradient.
    fn censored_grad(&self, y: &SurvivalData, natural: bool) -> Array2<f64>
    where
        Self: Sized,
    {
        let grad = self.censored_d_score(y);
        if !natural {
            return grad;
        }

        // Fast path: diagonal metric — use flat Array2
        if self.is_diagonal_censored_metric() {
            let diag_metric = self.diagonal_censored_metric();
            return diagonal_natural_gradient_flat(&grad, &diag_metric);
        }

        // Non-diagonal: delegate to parallelized solver
        let metric = self.censored_metric();
        natural_gradient_regularized(&grad, &metric, 0.0)
    }
}

/// Compute the natural gradient when the metric (Fisher Information) is diagonal.
/// Instead of solving a linear system per observation, we simply divide element-wise:
/// nat_grad[i,j] = grad[i,j] / metric[i,j,j]
///
/// This is dramatically faster than LU decomposition for diagonal matrices.
fn diagonal_natural_gradient(grad: &Array2<f64>, metric: &Array3<f64>) -> Array2<f64> {
    let n_params = grad.ncols();
    let mut nat_grad = Array2::zeros(grad.raw_dim());

    for j in 0..n_params {
        ndarray::Zip::from(nat_grad.column_mut(j))
            .and(grad.column(j))
            .and(metric.slice(ndarray::s![.., j, j]))
            .for_each(|ng, &g, &m| {
                *ng = g / m;
            });
    }

    nat_grad
}

/// Compute the natural gradient using a flat diagonal metric (Array2 instead of Array3).
/// The diagonal_metric has shape (n_obs, n_params) where each element is the diagonal
/// of the Fisher Information for that observation and parameter.
///
/// This avoids allocating the full (n_obs, n_params, n_params) Array3 when only
/// diagonal elements are needed, saving memory and improving cache performance.
fn diagonal_natural_gradient_flat(grad: &Array2<f64>, diag_metric: &Array2<f64>) -> Array2<f64> {
    let mut nat_grad = Array2::zeros(grad.raw_dim());

    ndarray::Zip::from(&mut nat_grad)
        .and(grad)
        .and(diag_metric)
        .for_each(|ng, &g, &m| {
            *ng = g / m;
        });

    nat_grad
}

/// Compute the natural gradient using a flat diagonal metric with Tikhonov regularization.
/// Adds `reg` to each diagonal element before dividing.
pub fn diagonal_natural_gradient_flat_regularized(
    grad: &Array2<f64>,
    diag_metric: &Array2<f64>,
    reg: f64,
) -> Array2<f64> {
    let mut nat_grad = Array2::zeros(grad.raw_dim());

    ndarray::Zip::from(&mut nat_grad)
        .and(grad)
        .and(diag_metric)
        .for_each(|ng, &g, &m| {
            *ng = g / (m + reg);
        });

    nat_grad
}

/// Compute the natural gradient for a single observation.
/// This is the core computation factored out for parallelization.
/// Optimized to minimize allocations by reusing workspace arrays where possible.
#[inline]
fn compute_natural_gradient_single(
    g_i: &ndarray::ArrayView1<f64>,
    metric_i: &ndarray::ArrayView2<f64>,
    reg: f64,
    n_params: usize,
) -> Array1<f64> {
    // Only clone if we need to modify (when regularization is applied)
    let metric_owned: Array2<f64> = if reg > 0.0 {
        let mut metric_reg = metric_i.to_owned();
        for j in 0..n_params {
            metric_reg[[j, j]] += reg;
        }
        metric_reg
    } else {
        metric_i.to_owned()
    };
    let g_owned = g_i.to_owned();

    // Try direct solve first (matches Python's np.linalg.solve)
    // Match Python: do NOT check is_finite after solve — Python keeps
    // whatever np.linalg.solve returns (including NaN/Inf)
    if let Ok(ng_i) = metric_owned.solve_into(g_owned.clone()) {
        return ng_i;
    }

    // Fall back to pseudo-inverse (matches Python's np.linalg.pinv)
    if let Some(pinv_metric_i) = pinv(&metric_owned) {
        let result = pinv_metric_i.dot(&g_owned);
        return result;
    }

    // Last resort: return zeros (Python would have raised an error here)
    Array1::zeros(g_owned.len())
}

/// Compute the natural gradient with optional Tikhonov regularization.
/// This function adds `reg * I` to the metric before solving, which stabilizes
/// the solution for ill-conditioned Fisher Information Matrices.
///
/// Uses rayon to parallelize the computation across observations for better
/// performance on multi-core systems.
///
/// Optimized to use array views instead of cloning, reducing memory allocations.
///
/// # Arguments
/// * `grad` - The standard gradient (n_obs x n_params)
/// * `metric` - The Fisher Information Matrix (n_obs x n_params x n_params)
/// * `reg` - Tikhonov regularization parameter (0.0 to disable)
///
/// # Returns
/// The natural gradient (n_obs x n_params)
pub fn natural_gradient_regularized(
    grad: &Array2<f64>,
    metric: &Array3<f64>,
    reg: f64,
) -> Array2<f64> {
    let n_obs = grad.nrows();
    let n_params = grad.ncols();

    // Parallel computation using rayon
    // Optimization: Use views instead of cloning where possible
    let results: Vec<Array1<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let g_i = grad.row(i);
            let metric_i = metric.index_axis(Axis(0), i);
            compute_natural_gradient_single(&g_i, &metric_i, reg, n_params)
        })
        .collect();

    // Assemble results into output array
    let mut natural_grad = Array2::zeros(grad.raw_dim());
    for (i, ng_i) in results.into_iter().enumerate() {
        natural_grad.row_mut(i).assign(&ng_i);
    }
    natural_grad
}

/// Compute the Moore-Penrose pseudo-inverse of a matrix using SVD.
/// This matches numpy's np.linalg.pinv behavior with default rcond.
fn pinv(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    // Match numpy's default rcond: max(M, N) * machine_epsilon
    let (m, n_cols) = (matrix.nrows(), matrix.ncols());
    let rcond = m.max(n_cols) as f64 * f64::EPSILON;

    // Perform SVD: A = U * S * V^T
    let (u, s, vt) = matrix.svd(true, true).ok()?;
    let u = u?;
    let vt = vt?;

    // Determine the cutoff for small singular values
    let max_sv = s.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = rcond * max_sv;

    // Compute S^+ (pseudo-inverse of singular values)
    let s_pinv: Array1<f64> = s.mapv(|sv| if sv > cutoff { 1.0 / sv } else { 0.0 });

    // Compute A^+ = V * S^+ * U^T
    // First compute S^+ * U^T by scaling rows of U^T
    let n = s_pinv.len();
    let mut result = Array2::zeros((vt.ncols(), u.nrows()));

    for i in 0..n {
        for j in 0..u.nrows() {
            for k in 0..vt.ncols() {
                result[[k, j]] += vt[[i, k]] * s_pinv[i] * u[[j, i]];
            }
        }
    }

    Some(result)
}

/// A trait that connects a Distribution to a Score.
pub trait Scorable<S: Score> {
    /// Calculates the score for each observation.
    fn score(&self, y: &Array1<f64>) -> Array1<f64>;

    /// Calculates the gradient of the score with respect to the distribution's parameters.
    fn d_score(&self, y: &Array1<f64>) -> Array2<f64>;

    /// Calculates the Riemannian metric tensor of the score for each observation.
    fn metric(&self) -> Array3<f64>;

    /// Returns true if the metric (Fisher Information Matrix) is always diagonal.
    /// When true, the natural gradient can be computed via element-wise division
    /// instead of a full linear solve, which is dramatically faster.
    fn is_diagonal_metric(&self) -> bool {
        false
    }

    /// Returns the diagonal of the metric (Fisher Information Matrix) as a flat Array2.
    /// Shape: (n_obs, n_params) — each row contains only the diagonal elements.
    ///
    /// Override this for diagonal distributions to avoid allocating the full
    /// (n_obs, n_params, n_params) Array3. The default implementation extracts
    /// diagonals from the full metric.
    fn diagonal_metric(&self) -> Array2<f64> {
        let metric = self.metric();
        let n_obs = metric.shape()[0];
        let n_params = metric.shape()[1];
        let mut diag = Array2::zeros((n_obs, n_params));
        for j in 0..n_params {
            ndarray::Zip::from(diag.column_mut(j))
                .and(metric.slice(ndarray::s![.., j, j]))
                .for_each(|d, &m| {
                    *d = m;
                });
        }
        diag
    }

    /// Calculates the total score, averaged over all observations.
    fn total_score(&self, y: &Array1<f64>, sample_weight: Option<&Array1<f64>>) -> f64 {
        let scores = self.score(y);
        if let Some(weights) = sample_weight {
            (scores * weights).sum() / weights.sum()
        } else {
            scores.mean().unwrap_or(0.0)
        }
    }

    /// Computes d_score and diagonal_metric in a single fused pass.
    /// Override this for distributions where both share intermediate computations
    /// (e.g., z = (y - loc) / scale) to avoid redundant memory traversals.
    ///
    /// Default implementation calls them separately.
    fn d_score_and_diagonal_metric(&self, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        (self.d_score(y), self.diagonal_metric())
    }

    /// Calculates the gradient, optionally the natural gradient.
    /// Uses the same fallback strategy as Python's NGBoost:
    /// 1. Try to solve the linear system directly
    /// 2. Fall back to matrix inverse
    /// 3. Fall back to pseudo-inverse (pinv) for singular/ill-conditioned matrices
    fn grad(&self, y: &Array1<f64>, natural: bool) -> Array2<f64> {
        if !natural {
            return self.d_score(y);
        }

        // Fast path: diagonal metric — fused d_score + diagonal_metric
        if self.is_diagonal_metric() {
            return self.diagonal_natural_grad(y, 0.0);
        }

        // Non-diagonal: separate d_score + full metric + solve
        let grad = self.d_score(y);
        let metric = self.metric();
        natural_gradient_regularized(&grad, &metric, 0.0)
    }

    /// Calculates the natural gradient using the fused d_score + diagonal_metric path.
    /// This is used by the training loop when tikhonov_reg > 0 and the metric is diagonal.
    fn diagonal_natural_grad(&self, y: &Array1<f64>, reg: f64) -> Array2<f64> {
        let (grad, diag_metric) = self.d_score_and_diagonal_metric(y);
        if reg > 0.0 {
            diagonal_natural_gradient_flat_regularized(&grad, &diag_metric, reg)
        } else {
            diagonal_natural_gradient_flat(&grad, &diag_metric)
        }
    }
}
