use crate::dist::{ClassificationDistn, Distribution};
use crate::scores::{LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Axis};

/// Softmax function applied along axis 0.
fn softmax_axis0(logits: &Array2<f64>) -> Array2<f64> {
    let max_vals = logits.fold_axis(Axis(0), f64::NEG_INFINITY, |&a, &b| a.max(b));
    let shifted = logits - &max_vals;
    let exp_vals = shifted.mapv(f64::exp);
    let sum_exp = exp_vals.sum_axis(Axis(0));
    exp_vals / &sum_exp
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
    /// The parameters of the distribution (K-1 x N).
    _params: Array2<f64>,
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
            _params: params.clone(),
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

        // Convert to probabilities
        let probs: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

        // Return logits relative to class 0: log(p_k) - log(p_0)
        let log_p0 = (probs[0].max(1e-10)).ln();
        let mut init_params = Array1::zeros(K - 1);
        for k in 1..K {
            init_params[k - 1] = (probs[k].max(1e-10)).ln() - log_p0;
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

    fn params(&self) -> &Array2<f64> {
        &self._params
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
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            let class = y_i as usize;
            scores[i] = -self.probs[[class, i]].max(1e-10).ln();
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
