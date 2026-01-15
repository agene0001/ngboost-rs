use ndarray::{Array1, Array2};

pub trait BaseLearner {
    fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str>;
}

pub trait TrainedBaseLearner: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;
}

#[derive(Clone)]
pub struct StumpLearner;

impl BaseLearner for StumpLearner {
    fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }

        let (best_feature, best_threshold, _best_mse) = find_best_split(x, y);

        if best_threshold.is_nan() {
            return Ok(Box::new(TrainedStumpLearner {
                feature_index: 0,
                threshold: 0.0,
                left_value: y.mean().unwrap_or(0.0),
                right_value: y.mean().unwrap_or(0.0),
            }));
        }

        let feature_col = x.column(best_feature);

        let mut left_y = Vec::new();
        let mut right_y = Vec::new();
        for (i, &y_val) in y.iter().enumerate() {
            if feature_col[i] < best_threshold {
                left_y.push(y_val);
            } else {
                right_y.push(y_val);
            }
        }

        let left_mean = if left_y.is_empty() {
            y.mean().unwrap_or(0.0)
        } else {
            Array1::from(left_y).mean().unwrap()
        };
        let right_mean = if right_y.is_empty() {
            y.mean().unwrap_or(0.0)
        } else {
            Array1::from(right_y).mean().unwrap()
        };

        Ok(Box::new(TrainedStumpLearner {
            feature_index: best_feature,
            threshold: best_threshold,
            left_value: left_mean,
            right_value: right_mean,
        }))
    }
}

pub struct TrainedStumpLearner {
    feature_index: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl TrainedBaseLearner for TrainedStumpLearner {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        if x.shape()[1] <= self.feature_index {
            return Array1::from_elem(x.nrows(), self.left_value);
        }
        x.column(self.feature_index).mapv(|val| {
            if val < self.threshold {
                self.left_value
            } else {
                self.right_value
            }
        })
    }
}

fn find_best_split(x: &Array2<f64>, y: &Array1<f64>) -> (usize, f64, f64) {
    let n_features = x.shape()[1];
    let mut best_feature = 0;
    let mut best_threshold = f64::NAN;
    let mut best_mse = f64::INFINITY;

    for feature_index in 0..n_features {
        let feature_values = x.column(feature_index);
        let mut unique_values = feature_values.to_vec();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_values.dedup();

        for &threshold in &unique_values {
            let mut left_y = Vec::new();
            let mut right_y = Vec::new();

            for i in 0..y.len() {
                if feature_values[i] < threshold {
                    left_y.push(y[i]);
                } else {
                    right_y.push(y[i]);
                }
            }

            if left_y.is_empty() || right_y.is_empty() {
                continue;
            }

            let left_y_arr = Array1::from(left_y);
            let right_y_arr = Array1::from(right_y);

            let mean_left = left_y_arr.mean().unwrap();
            let mean_right = right_y_arr.mean().unwrap();

            let mse_left = left_y_arr.mapv(|v| (v - mean_left).powi(2)).mean().unwrap();
            let mse_right = right_y_arr
                .mapv(|v| (v - mean_right).powi(2))
                .mean()
                .unwrap();

            let n_left = left_y_arr.len() as f64;
            let n_right = right_y_arr.len() as f64;
            let n_total = y.len() as f64;

            let mse = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right;

            if mse < best_mse {
                best_mse = mse;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
    }
    (best_feature, best_threshold, best_mse)
}

pub fn default_tree_learner() -> StumpLearner {
    StumpLearner
}
