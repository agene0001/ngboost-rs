use ndarray::{Array1, Array2, Array3};
use ndarray_linalg::{Inverse, Solve};

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

            if let Ok(ng_i) = metric_i.solve_into(g_i) {
                natural_grad.row_mut(i).assign(&ng_i);
            } else if let Ok(inv_metric_i) = metric_i.inv() {
                natural_grad
                    .row_mut(i)
                    .assign(&inv_metric_i.dot(&grad.row(i)));
            } else {
                natural_grad.row_mut(i).assign(&grad.row(i));
            }
        }
        natural_grad
    }
}
