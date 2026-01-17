use ndarray::{Array1, Array2, Array3};
use ndarray_linalg::{Inverse, Solve, SVD};

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

    /// Calculates the gradient for censored data, optionally the natural gradient.
    fn censored_grad(&self, y: &SurvivalData, natural: bool) -> Array2<f64>
    where
        Self: Sized,
    {
        let grad = self.censored_d_score(y);
        if !natural {
            return grad;
        }

        let metric = self.censored_metric();
        let n_obs = grad.nrows();
        let mut natural_grad = Array2::zeros(grad.raw_dim());

        for i in 0..n_obs {
            let g_i = grad.row(i).to_owned();
            let metric_i = metric.index_axis(ndarray::Axis(0), i).to_owned();

            // Try direct solve first
            if let Ok(ng_i) = metric_i.solve_into(g_i.clone()) {
                if ng_i.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&ng_i);
                    continue;
                }
            }

            // Fall back to inverse
            if let Ok(inv_metric_i) = metric_i.inv() {
                let result = inv_metric_i.dot(&grad.row(i));
                if result.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&result);
                    continue;
                }
            }

            // Fall back to pseudo-inverse
            if let Some(pinv_metric_i) = pinv(&metric_i) {
                let result = pinv_metric_i.dot(&grad.row(i));
                if result.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&result);
                    continue;
                }
            }

            // Last resort: use regular gradient
            natural_grad.row_mut(i).assign(&(&grad.row(i) * 0.99));
        }
        natural_grad
    }
}

/// Compute the Moore-Penrose pseudo-inverse of a matrix using SVD.
/// This matches numpy's np.linalg.pinv behavior with default rcond.
fn pinv(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    // Use default rcond like numpy (machine epsilon times max dimension)
    let rcond = 1e-15; // This matches numpy's default behavior for typical matrices

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

    /// Calculates the total score, averaged over all observations.
    fn total_score(&self, y: &Array1<f64>, sample_weight: Option<&Array1<f64>>) -> f64 {
        let scores = self.score(y);
        if let Some(weights) = sample_weight {
            (scores * weights).sum() / weights.sum()
        } else {
            scores.mean().unwrap_or(0.0)
        }
    }

    /// Calculates the gradient, optionally the natural gradient.
    /// Uses the same fallback strategy as Python's NGBoost:
    /// 1. Try to solve the linear system directly
    /// 2. Fall back to matrix inverse
    /// 3. Fall back to pseudo-inverse (pinv) for singular/ill-conditioned matrices
    fn grad(&self, y: &Array1<f64>, natural: bool) -> Array2<f64> {
        let grad = self.d_score(y);
        if !natural {
            return grad;
        }

        let metric = self.metric();
        let n_obs = grad.nrows();
        let mut natural_grad = Array2::zeros(grad.raw_dim());

        for i in 0..n_obs {
            let g_i = grad.row(i).to_owned();
            let metric_i = metric.index_axis(ndarray::Axis(0), i).to_owned();

            // Try direct solve first (fastest) - matches Python's np.linalg.solve
            if let Ok(ng_i) = metric_i.solve_into(g_i.clone()) {
                // Check if solution is reasonable
                if ng_i.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&ng_i);
                    continue;
                }
            }

            // Fall back to inverse
            if let Ok(inv_metric_i) = metric_i.inv() {
                let result = inv_metric_i.dot(&grad.row(i));
                if result.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&result);
                    continue;
                }
            }

            // Fall back to pseudo-inverse (matches Python's np.linalg.pinv)
            if let Some(pinv_metric_i) = pinv(&metric_i) {
                let result = pinv_metric_i.dot(&grad.row(i));
                if result.iter().all(|&v| v.is_finite()) {
                    natural_grad.row_mut(i).assign(&result);
                    continue;
                }
            }

            // Last resort: use regular gradient (should rarely happen)
            // Add small regularization to avoid numerical issues
            natural_grad.row_mut(i).assign(&(&grad.row(i) * 0.99));
        }
        natural_grad
    }
}
