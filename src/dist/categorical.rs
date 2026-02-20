use crate::dist::{ClassificationDistn, Distribution};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Axis};

/// Softmax function applied along axis 0 with numerical stability improvements.
/// Optimized with in-place operations to reduce memory allocations.
#[inline]
fn softmax_axis0(logits: &Array2<f64>) -> Array2<f64> {
    // Compute max values for numerical stability (log-sum-exp trick)
    let max_vals = logits.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));

    // Pre-allocate result array
    let mut probs = Array2::zeros(logits.raw_dim());

    // Compute exp(logits - max) and sum in one pass per column
    let (k, n) = (logits.nrows(), logits.ncols());
    for j in 0..n {
        let max_j = max_vals[j];
        let mut sum_exp = 0.0;

        // First pass: compute exp values and their sum
        for i in 0..k {
            let exp_val = (logits[[i, j]] - max_j).exp();
            probs[[i, j]] = exp_val;
            sum_exp += exp_val;
        }

        // Second pass: normalize
        let inv_sum = 1.0 / sum_exp;
        for i in 0..k {
            probs[[i, j]] = probs[[i, j]] * inv_sum;
        }
    }

    probs
}

/// A K-class Categorical distribution for classification.
///
/// This is a generic struct that can represent any K-class categorical.
/// The number of parameters is K-1 (the 0th class logit is fixed at 0).
#[derive(Debug, Clone)]
pub struct Categorical<const K: usize> {
    /// The logits (K x N), where K is the number of classes and N is the number of observations.
    pub logits: Array2<f64>,
    /// The probabilities (K x N), computed via softmax.
    pub probs: Array2<f64>,
    /// Number of observations.
    n_obs: usize,
}

impl<const K: usize> Distribution for Categorical<K> {
    fn from_params(params: &Array2<f64>) -> Self {
        // params is (N, K-1) - each row is one observation's parameters
        let n_obs = params.nrows();

        // Build logits: (K, N) with first row as zeros
        let mut logits = Array2::zeros((K, n_obs));
        for i in 0..n_obs {
            for j in 0..(K - 1) {
                logits[[j + 1, i]] = params[[i, j]];
            }
        }

        let probs = softmax_axis0(&logits);

        Categorical {
            logits,
            probs,
            n_obs,
        }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        // Count occurrences of each class
        let n = y.len();
        let mut counts = vec![0usize; K];
        for &y_i in y.iter() {
            let class = y_i as usize;
            if class < K {
                counts[class] += 1;
            }
        }

        // Convert to probabilities (no smoothing, matching Python)
        let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

        // Return logits relative to class 0: log(p_k) - log(p_0)
        let log_p0 = probs[0].ln();
        let mut init_params = Array1::zeros(K - 1);
        for k in 1..K {
            init_params[k - 1] = probs[k].ln() - log_p0;
        }

        init_params
    }

    fn n_params(&self) -> usize {
        K - 1
    }

    fn predict(&self) -> Array1<f64> {
        // Return the most likely class for each observation
        let mut predictions = Array1::zeros(self.n_obs);
        for i in 0..self.n_obs {
            let mut max_prob = f64::NEG_INFINITY;
            let mut max_class = 0;
            for k in 0..K {
                if self.probs[[k, i]] > max_prob {
                    max_prob = self.probs[[k, i]];
                    max_class = k;
                }
            }
            predictions[i] = max_class as f64;
        }
        predictions
    }

    fn params(&self) -> Array2<f64> {
        // Reconstruct params (N, K-1) from logits (K, N)
        let mut p = Array2::zeros((self.n_obs, K - 1));
        for i in 0..self.n_obs {
            for j in 0..(K - 1) {
                p[[i, j]] = self.logits[[j + 1, i]];
            }
        }
        p
    }
}

impl<const K: usize> ClassificationDistn for Categorical<K> {
    fn class_probs(&self) -> Array2<f64> {
        // Return (N, K) probabilities
        self.probs.t().to_owned()
    }
}

impl<const K: usize> Scorable<LogScore> for Categorical<K> {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -log(p[y_i]) for each observation
        // Match Python: no probability clipping, -log(p) directly
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let class = y_i as usize;
            scores[i] = -self.probs[[class, i]].ln();
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient: probs - one_hot(y), but only for classes 1..K (not class 0)
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, K - 1));

        for i in 0..n_obs {
            let y_i = y[i] as usize;
            for k in 1..K {
                // d/d(logit_k) = p_k - 1{y == k}
                let indicator = if y_i == k { 1.0 } else { 0.0 };
                d_params[[i, k - 1]] = self.probs[[k, i]] - indicator;
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information Matrix for categorical
        // FI[j,k] = -p_j * p_k for j != k
        // FI[j,j] = p_j * (1 - p_j) = p_j - p_j^2
        let n_obs = self.n_obs;
        let n_params = K - 1;
        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            for j in 0..n_params {
                let p_j = self.probs[[j + 1, i]];
                for k in 0..n_params {
                    let p_k = self.probs[[k + 1, i]];
                    if j == k {
                        fi[[i, j, k]] = p_j * (1.0 - p_j);
                    } else {
                        fi[[i, j, k]] = -p_j * p_k;
                    }
                }
            }
        }

        fi
    }
}

impl<const K: usize> Scorable<CRPScore> for Categorical<K> {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // For categorical distributions, the CRPS equivalent is the Brier score:
        // BS = Σ_k (p_k - 1{y == k})^2
        // This is the sum of squared errors between predicted probs and one-hot encoding
        //
        // Note: For ordinal categories, one would use the Ranked Probability Score (RPS)
        // which is CRPS applied to the cumulative distribution. Here we use Brier score
        // since categorical typically implies unordered classes.

        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let true_class = y_i as usize;
            let mut brier = 0.0;
            for k in 0..K {
                let p_k = self.probs[[k, i]];
                let indicator = if k == true_class { 1.0 } else { 0.0 };
                brier += (p_k - indicator).powi(2);
            }
            scores[i] = brier;
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of Brier score w.r.t. logits (parameters)
        // BS = Σ_k (p_k - 1{y == k})^2
        // d(BS)/d(logit_j) = Σ_k 2*(p_k - 1{y==k}) * d(p_k)/d(logit_j)
        //
        // For softmax: d(p_k)/d(logit_j) = p_k * (1{k==j} - p_j)
        //
        // d(BS)/d(logit_j) = 2 * Σ_k (p_k - 1{y==k}) * p_k * (1{k==j} - p_j)
        //                  = 2 * [(p_j - 1{y==j}) * p_j * (1 - p_j)
        //                        - p_j * Σ_{k≠j} (p_k - 1{y==k}) * p_k]
        //                  = 2 * p_j * [(p_j - 1{y==j}) * (1 - p_j)
        //                              - Σ_{k≠j} (p_k - 1{y==k}) * p_k]
        //
        // Simplified: d(BS)/d(logit_j) = 2 * p_j * [Σ_k (p_k - 1{y==k}) * (1{k==j} - p_k)]

        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, K - 1));

        for i in 0..n_obs {
            let y_i = y[i] as usize;

            // Compute residuals: r_k = p_k - 1{y == k}
            let mut residuals = vec![0.0; K];
            for k in 0..K {
                let indicator = if k == y_i { 1.0 } else { 0.0 };
                residuals[k] = self.probs[[k, i]] - indicator;
            }

            // For each parameter (logit_j for j = 1..K)
            for j in 1..K {
                let p_j = self.probs[[j, i]];

                // d(BS)/d(logit_j) = 2 * Σ_k r_k * p_k * (1{k==j} - p_j)
                let mut grad = 0.0;
                for k in 0..K {
                    let p_k = self.probs[[k, i]];
                    let delta_kj = if k == j { 1.0 } else { 0.0 };
                    grad += residuals[k] * p_k * (delta_kj - p_j);
                }
                d_params[[i, j - 1]] = 2.0 * grad;
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Metric for Brier score: M_{jl} = E_Y[g_j * g_l]
        // where g_j = 2 * Σ_k (p_k - 1{Y==k}) * p_k * (δ_{kj} - p_j)
        // and the expectation is over Y ~ Categorical(p).
        let n_obs = self.n_obs;
        let n_params = K - 1;
        let mut fi = Array3::zeros((n_obs, n_params, n_params));

        for i in 0..n_obs {
            // For each possible outcome c (with probability p_c),
            // compute the gradient vector, then accumulate g * g^T weighted by p_c.
            for c in 0..K {
                let p_c = self.probs[[c, i]];

                // Compute residuals for outcome Y = c
                let mut residuals = vec![0.0; K];
                for k in 0..K {
                    let indicator = if k == c { 1.0 } else { 0.0 };
                    residuals[k] = self.probs[[k, i]] - indicator;
                }

                // Compute gradient for each parameter
                let mut grad = vec![0.0; n_params];
                for j in 0..n_params {
                    let p_j = self.probs[[j + 1, i]];
                    let mut g = 0.0;
                    for k in 0..K {
                        let p_k = self.probs[[k, i]];
                        let delta_kj = if k == j + 1 { 1.0 } else { 0.0 };
                        g += residuals[k] * p_k * (delta_kj - p_j);
                    }
                    grad[j] = 2.0 * g;
                }

                // Accumulate p_c * grad * grad^T
                for j in 0..n_params {
                    for l in 0..n_params {
                        fi[[i, j, l]] += p_c * grad[j] * grad[l];
                    }
                }
            }
        }

        fi
    }
}

/// Type alias for binary classification (Bernoulli distribution).
pub type Bernoulli = Categorical<2>;

/// Type alias for 3-class classification.
pub type Categorical3 = Categorical<3>;

/// Type alias for 4-class classification.
pub type Categorical4 = Categorical<4>;

/// Type alias for 5-class classification.
pub type Categorical5 = Categorical<5>;

/// Type alias for 10-class classification (e.g., digit recognition).
pub type Categorical10 = Categorical<10>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_categorical_crpscore_bernoulli() {
        // Binary classification with Bernoulli
        // params: logit for class 1 (class 0 logit is fixed at 0)
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(); // p = [0.5, 0.5]
        let dist = Bernoulli::from_params(&params);

        // With equal probs [0.5, 0.5], Brier score for y=0:
        // (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5
        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);
        assert_relative_eq!(score[0], 0.5, epsilon = 1e-6);

        // Same for y=1
        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);
        assert_relative_eq!(score[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_categorical_crpscore_perfect_prediction() {
        // Perfect prediction should have Brier score of 0
        // Use large logit to get probability close to 1
        let params = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap(); // p ≈ [0, 1]
        let dist = Bernoulli::from_params(&params);

        // Predicting class 1 when true class is 1
        let y = Array1::from_vec(vec![1.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);
        assert!(score[0] < 0.01); // Should be very small
    }

    #[test]
    fn test_categorical_crpscore_worst_prediction() {
        // Worst prediction (confident wrong answer)
        let params = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap(); // p ≈ [0, 1]
        let dist = Bernoulli::from_params(&params);

        // Predicting class 1 when true class is 0
        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);
        // Brier score ≈ (0 - 1)^2 + (1 - 0)^2 = 2
        assert!(score[0] > 1.9);
    }

    #[test]
    fn test_categorical_crpscore_multiclass() {
        // 3-class classification
        let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap(); // equal probs
        let dist = Categorical3::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // With equal probs [1/3, 1/3, 1/3], Brier for y=0:
        // (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        assert_relative_eq!(score[0], 2.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_categorical_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = Bernoulli::from_params(&params);

        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradient should be finite
        assert!(d_score[[0, 0]].is_finite());
    }

    #[test]
    fn test_categorical_crpscore_metric() {
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let dist = Bernoulli::from_params(&params);

        let metric = Scorable::<CRPScore>::metric(&dist);

        // Metric should be positive (for diagonal)
        assert!(metric[[0, 0, 0]] > 0.0);
    }

    #[test]
    fn test_categorical_logscore() {
        // Basic LogScore test
        let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(); // p = [0.5, 0.5]
        let dist = Bernoulli::from_params(&params);

        let y = Array1::from_vec(vec![0.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // -log(0.5) ≈ 0.693
        assert_relative_eq!(score[0], 0.5_f64.ln().abs(), epsilon = 1e-6);
    }
}
