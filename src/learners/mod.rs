use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

pub trait BaseLearner: Send + Sync {
    fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        self.fit_with_weights(x, y, None)
    }

    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str>;
}

pub trait TrainedBaseLearner: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Returns feature importances if the learner supports it.
    /// Returns None for learners that don't support feature importances.
    /// For tree-based learners, returns an array of length n_features where
    /// each value represents the importance of that feature.
    fn feature_importances(&self) -> Option<Array1<f64>> {
        None
    }

    /// Returns the feature index used for splitting (for single-split learners like stumps)
    fn split_feature(&self) -> Option<usize> {
        None
    }

    /// Returns all feature indices used in splits (for multi-level trees)
    fn split_features(&self) -> Option<Vec<usize>> {
        self.split_feature().map(|f| vec![f])
    }

    /// Convert to a serializable representation.
    /// Returns None if the learner type doesn't support serialization.
    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        None
    }
}

/// Serializable wrapper enum for trained learners.
/// This allows saving and loading trained models with their base learners.
#[derive(Clone, Serialize, Deserialize)]
pub enum SerializableTrainedLearner {
    Stump(TrainedStumpLearner),
    DecisionTree(TrainedDecisionTree),
    HistogramTree(TrainedHistogramTree),
    Ridge(TrainedRidgeLearner),
    ArenaTree(ArenaDecisionTree),
}

impl SerializableTrainedLearner {
    /// Convert a trait object to a serializable enum.
    /// Use the `to_serializable()` method on TrainedBaseLearner instead.
    #[allow(dead_code)]
    pub fn from_trait_object(_learner: &dyn TrainedBaseLearner) -> Option<Self> {
        // Use to_serializable() method on the trait object instead
        None
    }

    /// Convert back to a boxed trait object
    pub fn to_trait_object(self) -> Box<dyn TrainedBaseLearner> {
        match self {
            SerializableTrainedLearner::Stump(s) => Box::new(s),
            SerializableTrainedLearner::DecisionTree(t) => Box::new(t),
            SerializableTrainedLearner::HistogramTree(h) => Box::new(h),
            SerializableTrainedLearner::Ridge(r) => Box::new(r),
            SerializableTrainedLearner::ArenaTree(a) => Box::new(a),
        }
    }
}

impl TrainedBaseLearner for SerializableTrainedLearner {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.predict(x),
            SerializableTrainedLearner::DecisionTree(t) => t.predict(x),
            SerializableTrainedLearner::HistogramTree(h) => h.predict(x),
            SerializableTrainedLearner::Ridge(r) => r.predict(x),
            SerializableTrainedLearner::ArenaTree(a) => a.predict(x),
        }
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.feature_importances(),
            SerializableTrainedLearner::DecisionTree(t) => t.feature_importances(),
            SerializableTrainedLearner::HistogramTree(h) => h.feature_importances(),
            SerializableTrainedLearner::Ridge(r) => r.feature_importances(),
            SerializableTrainedLearner::ArenaTree(a) => a.feature_importances(),
        }
    }

    fn split_feature(&self) -> Option<usize> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.split_feature(),
            SerializableTrainedLearner::DecisionTree(t) => t.split_feature(),
            SerializableTrainedLearner::HistogramTree(h) => h.split_feature(),
            SerializableTrainedLearner::Ridge(_) => None, // Linear models don't split
            SerializableTrainedLearner::ArenaTree(a) => a.split_feature(),
        }
    }

    fn split_features(&self) -> Option<Vec<usize>> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.split_features(),
            SerializableTrainedLearner::DecisionTree(t) => t.split_features(),
            SerializableTrainedLearner::HistogramTree(h) => h.split_features(),
            SerializableTrainedLearner::Ridge(_) => None, // Linear models don't split
            SerializableTrainedLearner::ArenaTree(a) => a.split_features(),
        }
    }
}

#[derive(Clone)]
pub struct StumpLearner;

impl BaseLearner for StumpLearner {
    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }

        let (best_feature, best_threshold, _best_mse) = if let Some(weights) = sample_weight {
            find_best_split_weighted(x, y, weights)
        } else {
            find_best_split(x, y)
        };

        if best_threshold.is_nan() {
            let mean_val = if let Some(weights) = sample_weight {
                weighted_mean(y, weights)
            } else {
                y.mean().unwrap_or(0.0)
            };

            return Ok(Box::new(TrainedStumpLearner {
                feature_index: 0,
                threshold: 0.0,
                left_value: mean_val,
                right_value: mean_val,
            }));
        }

        let feature_col = x.column(best_feature);

        // Pre-allocate with estimated capacity for better performance
        let estimated_size = y.len() / 2;
        let mut left_y = Vec::with_capacity(estimated_size);
        let mut left_weights = Vec::with_capacity(if sample_weight.is_some() {
            estimated_size
        } else {
            0
        });
        let mut right_y = Vec::with_capacity(estimated_size);
        let mut right_weights = Vec::with_capacity(if sample_weight.is_some() {
            estimated_size
        } else {
            0
        });

        for (i, &y_val) in y.iter().enumerate() {
            if feature_col[i] < best_threshold {
                left_y.push(y_val);
                if let Some(weights) = sample_weight {
                    left_weights.push(weights[i]);
                }
            } else {
                right_y.push(y_val);
                if let Some(weights) = sample_weight {
                    right_weights.push(weights[i]);
                }
            }
        }

        let left_mean = if left_y.is_empty() {
            if let Some(w) = sample_weight {
                weighted_mean(y, w)
            } else {
                y.mean().unwrap_or(0.0)
            }
        } else if sample_weight.is_some() {
            weighted_mean(&Array1::from(left_y), &Array1::from(left_weights))
        } else {
            Array1::from(left_y).mean().unwrap()
        };

        let right_mean = if right_y.is_empty() {
            if let Some(w) = sample_weight {
                weighted_mean(y, w)
            } else {
                y.mean().unwrap_or(0.0)
            }
        } else if sample_weight.is_some() {
            weighted_mean(&Array1::from(right_y), &Array1::from(right_weights))
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

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedStumpLearner {
    pub feature_index: usize,
    pub threshold: f64,
    pub left_value: f64,
    pub right_value: f64,
}

impl TrainedBaseLearner for TrainedStumpLearner {
    #[inline]
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

    #[inline]
    fn split_feature(&self) -> Option<usize> {
        // Only return the feature if there's an actual split (non-NaN threshold)
        if self.threshold.is_nan() {
            None
        } else {
            Some(self.feature_index)
        }
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::Stump(self.clone()))
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
            // Pre-allocate with estimated capacity
            let estimated_size = y.len() / 2;
            let mut left_y = Vec::with_capacity(estimated_size);
            let mut right_y = Vec::with_capacity(estimated_size);

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

// ============================================================================
// Ridge (L2-regularized Linear) Learner
// ============================================================================

/// A Ridge regression learner (L2-regularized least squares).
/// Matches sklearn's Ridge with alpha parameter.
///
/// Solves: min_w ||y - Xw||^2 + alpha * ||w||^2
#[derive(Clone)]
pub struct RidgeLearner {
    /// Regularization strength. Larger values specify stronger regularization.
    pub alpha: f64,
    /// Whether to fit an intercept term.
    pub fit_intercept: bool,
}

impl RidgeLearner {
    /// Create a new Ridge learner with the given regularization strength.
    pub fn new(alpha: f64) -> Self {
        RidgeLearner {
            alpha,
            fit_intercept: true,
        }
    }

    /// Create a Ridge learner with alpha=0 (equivalent to OLS).
    /// This matches Python's default_linear_learner = Ridge(alpha=0.0)
    pub fn default_linear() -> Self {
        RidgeLearner {
            alpha: 0.0,
            fit_intercept: true,
        }
    }

    /// Create a Ridge learner with custom options.
    pub fn with_options(alpha: f64, fit_intercept: bool) -> Self {
        RidgeLearner {
            alpha,
            fit_intercept,
        }
    }
}

impl BaseLearner for RidgeLearner {
    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }
        if x.nrows() != y.len() {
            return Err("X and y must have the same number of samples");
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Apply sample weights by scaling X and y
        let (x_weighted, y_weighted) = if let Some(weights) = sample_weight {
            // Scale by sqrt(weight) so that (X'WX) becomes (X_w'X_w)
            let sqrt_weights = weights.mapv(|w| w.sqrt());
            let mut x_w = x.clone();
            let mut y_w = y.clone();
            for i in 0..n_samples {
                let sw = sqrt_weights[i];
                for j in 0..n_features {
                    x_w[[i, j]] *= sw;
                }
                y_w[i] *= sw;
            }
            (x_w, y_w)
        } else {
            (x.clone(), y.clone())
        };

        // Handle intercept by centering
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x_weighted.mean_axis(ndarray::Axis(0)).unwrap();
            let y_mean = y_weighted.mean().unwrap_or(0.0);

            let x_centered = &x_weighted - &x_mean;
            let y_centered = &y_weighted - y_mean;

            (x_centered, y_centered, Some(x_mean), y_mean)
        } else {
            (x_weighted, y_weighted, None, 0.0)
        };

        // Solve Ridge regression: (X'X + alpha*I)^{-1} X'y
        // Using the normal equations approach
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        // Add regularization: X'X + alpha * I
        let mut xtx_reg = xtx;
        for i in 0..n_features {
            xtx_reg[[i, i]] += self.alpha;
        }

        // Solve the linear system
        use ndarray_linalg::Solve;
        let coefficients = match xtx_reg.solve_into(xty) {
            Ok(coef) => coef,
            Err(_) => {
                // If solve fails, try with more regularization
                for i in 0..n_features {
                    xtx_reg[[i, i]] += 1e-6;
                }
                match xtx_reg.solve_into(x_centered.t().dot(&y_centered)) {
                    Ok(coef) => coef,
                    Err(_) => {
                        // Fall back to zero coefficients
                        Array1::zeros(n_features)
                    }
                }
            }
        };

        // Compute intercept
        let intercept = if self.fit_intercept {
            let x_mean = x_mean.unwrap();
            y_mean - x_mean.dot(&coefficients)
        } else {
            0.0
        };

        Ok(Box::new(TrainedRidgeLearner {
            coefficients: coefficients.to_vec(),
            intercept,
            n_features,
        }))
    }
}

/// A trained Ridge regression model.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedRidgeLearner {
    /// Coefficient weights for each feature (stored as Vec for serialization).
    pub coefficients: Vec<f64>,
    /// Intercept term.
    pub intercept: f64,
    /// Number of features the model was trained on.
    pub n_features: usize,
}

impl TrainedRidgeLearner {
    /// Get coefficients as an Array1.
    pub fn coefficients_array(&self) -> Array1<f64> {
        Array1::from(self.coefficients.clone())
    }
}

impl TrainedBaseLearner for TrainedRidgeLearner {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        // y = X @ w + b
        let coef = Array1::from(self.coefficients.clone());
        x.dot(&coef) + self.intercept
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        // For linear models, use absolute coefficient values as importance
        let abs_coef: Vec<f64> = self.coefficients.iter().map(|c| c.abs()).collect();
        let sum: f64 = abs_coef.iter().sum();
        if sum > 0.0 {
            Some(Array1::from(
                abs_coef.iter().map(|c| c / sum).collect::<Vec<_>>(),
            ))
        } else {
            Some(Array1::zeros(self.n_features))
        }
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::Ridge(self.clone()))
    }
}

/// Returns the default linear learner matching Python's sklearn default:
/// Ridge(alpha=0.0) which is equivalent to OLS
pub fn default_linear_learner() -> RidgeLearner {
    RidgeLearner::default_linear()
}

/// Returns a Ridge learner with specified regularization
pub fn ridge_learner(alpha: f64) -> RidgeLearner {
    RidgeLearner::new(alpha)
}

// ============================================================================
// Histogram-based Gradient Boosting Learner
// ============================================================================

/// A histogram-based decision tree regressor for faster training.
/// This bins continuous features into discrete buckets (like LightGBM/XGBoost).
///
/// Benefits over standard decision trees:
/// - O(n_bins) split finding instead of O(n_samples)
/// - Better cache efficiency
/// - Faster training, especially for large datasets
/// - Same or similar accuracy
#[derive(Clone)]
pub struct HistogramLearner {
    pub max_depth: usize,
    pub max_bins: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
}

impl HistogramLearner {
    pub fn new(max_depth: usize) -> Self {
        HistogramLearner {
            max_depth,
            max_bins: 255, // Like LightGBM default
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
    }

    /// Create with default settings matching typical gradient boosting defaults
    pub fn default_histogram() -> Self {
        HistogramLearner {
            max_depth: 3,
            max_bins: 255,
            min_samples_leaf: 1,
            min_samples_split: 2,
        }
    }

    pub fn with_params(
        max_depth: usize,
        max_bins: usize,
        min_samples_leaf: usize,
        min_samples_split: usize,
    ) -> Self {
        HistogramLearner {
            max_depth,
            max_bins,
            min_samples_leaf,
            min_samples_split,
        }
    }
}

impl BaseLearner for HistogramLearner {
    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }

        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Build bin edges for each feature
        let bin_edges: Vec<Vec<f64>> = (0..n_features)
            .map(|f| {
                let col_vec: Vec<f64> = x.column(f).to_vec();
                compute_bin_edges(&col_vec, self.max_bins)
            })
            .collect();

        // Bin the data (convert continuous to discrete bin indices)
        let binned_x: Vec<Vec<u8>> = (0..n_features)
            .map(|f| {
                let col = x.column(f);
                let edges = &bin_edges[f];
                col.iter().map(|&v| find_bin(v, edges)).collect()
            })
            .collect();

        // Build the tree using histograms
        let indices: Vec<usize> = (0..n_samples).collect();
        let root = build_histogram_tree_node(
            &binned_x,
            &bin_edges,
            y,
            sample_weight,
            &indices,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.max_bins,
        );

        Ok(Box::new(TrainedHistogramTree { root, n_features }))
    }
}

/// Compute bin edges for a feature column using quantiles
fn compute_bin_edges(values: &[f64], max_bins: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().cloned().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return vec![0.0];
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Remove duplicates and get unique values
    sorted.dedup();

    let n_unique = sorted.len();
    let n_bins = max_bins.min(n_unique);

    if n_bins <= 1 {
        return vec![sorted[0]];
    }

    // Use quantile-based binning for even distribution
    let mut edges = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let idx = (i * (n_unique - 1)) / (n_bins - 1);
        edges.push(sorted[idx]);
    }

    // Ensure edges are unique
    edges.dedup();
    edges
}

/// Find which bin a value belongs to (binary search)
fn find_bin(value: f64, edges: &[f64]) -> u8 {
    if edges.is_empty() || value <= edges[0] {
        return 0;
    }

    // Binary search for the right bin
    match edges.binary_search_by(|e| e.partial_cmp(&value).unwrap()) {
        Ok(i) => i.min(254) as u8,
        Err(i) => (i.saturating_sub(1)).min(254) as u8,
    }
}

/// Histogram structure for efficient split finding
struct Histogram {
    count: Vec<f64>,  // Count (or sum of weights) per bin
    sum: Vec<f64>,    // Sum of targets per bin
    sum_sq: Vec<f64>, // Sum of squared targets per bin (for variance)
}

impl Histogram {
    fn new(n_bins: usize) -> Self {
        Histogram {
            count: vec![0.0; n_bins],
            sum: vec![0.0; n_bins],
            sum_sq: vec![0.0; n_bins],
        }
    }

    fn add(&mut self, bin: u8, y: f64, weight: f64) {
        let b = bin as usize;
        if b < self.count.len() {
            self.count[b] += weight;
            self.sum[b] += y * weight;
            self.sum_sq[b] += y * y * weight;
        }
    }

    #[allow(dead_code)]
    fn total_count(&self) -> f64 {
        self.count.iter().sum()
    }

    #[allow(dead_code)]
    fn total_sum(&self) -> f64 {
        self.sum.iter().sum()
    }
}

/// Internal histogram tree node
#[derive(Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum HistTreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature_index: usize,
        bin_threshold: u8,    // Split on bin index (used during tree building)
        threshold_value: f64, // Actual threshold value for prediction
        left: Box<HistTreeNode>,
        right: Box<HistTreeNode>,
    },
}

impl HistTreeNode {
    /// Predict for a single row using direct array indexing (no allocation)
    #[inline]
    fn predict_row(&self, x: &Array2<f64>, row_idx: usize) -> f64 {
        match self {
            HistTreeNode::Leaf { value } => *value,
            HistTreeNode::Split {
                feature_index,
                threshold_value,
                left,
                right,
                ..
            } => {
                if x[[row_idx, *feature_index]] < *threshold_value {
                    left.predict_row(x, row_idx)
                } else {
                    right.predict_row(x, row_idx)
                }
            }
        }
    }

    fn collect_split_features(&self, features: &mut Vec<usize>) {
        match self {
            HistTreeNode::Leaf { .. } => {}
            HistTreeNode::Split {
                feature_index,
                left,
                right,
                ..
            } => {
                features.push(*feature_index);
                left.collect_split_features(features);
                right.collect_split_features(features);
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedHistogramTree {
    pub root: HistTreeNode,
    pub n_features: usize,
}

impl TrainedBaseLearner for TrainedHistogramTree {
    #[inline]
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            predictions[i] = self.root.predict_row(x, i);
        }
        predictions
    }

    fn split_features(&self) -> Option<Vec<usize>> {
        let mut features = Vec::new();
        self.root.collect_split_features(&mut features);
        if features.is_empty() {
            None
        } else {
            Some(features)
        }
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        let features = self.split_features()?;
        let mut importances = Array1::zeros(self.n_features);
        for &f in &features {
            if f < self.n_features {
                importances[f] += 1.0;
            }
        }
        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances.mapv_inplace(|v| v / sum);
        }
        Some(importances)
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::HistogramTree(self.clone()))
    }
}

/// Build histogram tree node recursively
fn build_histogram_tree_node(
    binned_x: &[Vec<u8>],
    bin_edges: &[Vec<f64>],
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_bins: usize,
) -> HistTreeNode {
    // Calculate weighted mean for this node
    let node_value = if let Some(weights) = sample_weight {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        for &i in indices {
            sum += y[i] * weights[i];
            weight_sum += weights[i];
        }
        if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        }
    } else {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        sum / indices.len() as f64
    };

    // Check stopping conditions
    if depth >= max_depth || indices.len() < min_samples_split {
        return HistTreeNode::Leaf { value: node_value };
    }

    // Find best split using histograms
    let (best_feature, best_bin, best_improvement, best_threshold_value) =
        find_best_histogram_split(
            binned_x,
            bin_edges,
            y,
            sample_weight,
            indices,
            min_samples_leaf,
            max_bins,
        );

    // If no valid split found
    if best_improvement <= 0.0 {
        return HistTreeNode::Leaf { value: node_value };
    }

    // Partition indices - pre-allocate with estimated capacity
    let estimated_size = indices.len() / 2;
    let mut left_indices = Vec::with_capacity(estimated_size);
    let mut right_indices = Vec::with_capacity(estimated_size);
    for &i in indices {
        if binned_x[best_feature][i] < best_bin {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    // Check min_samples_leaf
    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        return HistTreeNode::Leaf { value: node_value };
    }

    // Recursively build children
    let left_child = build_histogram_tree_node(
        binned_x,
        bin_edges,
        y,
        sample_weight,
        &left_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_bins,
    );
    let right_child = build_histogram_tree_node(
        binned_x,
        bin_edges,
        y,
        sample_weight,
        &right_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_bins,
    );

    HistTreeNode::Split {
        feature_index: best_feature,
        bin_threshold: best_bin,
        threshold_value: best_threshold_value,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Find best split using histogram-based approach
fn find_best_histogram_split(
    binned_x: &[Vec<u8>],
    bin_edges: &[Vec<f64>],
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    min_samples_leaf: usize,
    max_bins: usize,
) -> (usize, u8, f64, f64) {
    let n_features = binned_x.len();
    let mut best_feature = 0;
    let mut best_bin: u8 = 0;
    let mut best_improvement = 0.0;
    let mut best_threshold_value = 0.0;

    // Calculate parent statistics
    let (parent_sum, parent_count) = if let Some(weights) = sample_weight {
        let mut sum = 0.0;
        let mut count = 0.0;
        for &i in indices {
            sum += y[i] * weights[i];
            count += weights[i];
        }
        (sum, count)
    } else {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        (sum, indices.len() as f64)
    };

    if parent_count == 0.0 {
        return (
            best_feature,
            best_bin,
            best_improvement,
            best_threshold_value,
        );
    }

    let _parent_mean = parent_sum / parent_count;

    for feature_index in 0..n_features {
        // Build histogram for this feature
        let mut hist = Histogram::new(max_bins);
        for &i in indices {
            let bin = binned_x[feature_index][i];
            let weight = sample_weight.map_or(1.0, |w| w[i]);
            hist.add(bin, y[i], weight);
        }

        // Scan through bins to find best split
        let mut left_count = 0.0;
        let mut left_sum = 0.0;

        for bin in 0..(max_bins - 1) as u8 {
            left_count += hist.count[bin as usize];
            left_sum += hist.sum[bin as usize];

            let right_count = parent_count - left_count;
            let right_sum = parent_sum - left_sum;

            // Check min_samples_leaf (approximate with counts)
            if left_count < min_samples_leaf as f64 || right_count < min_samples_leaf as f64 {
                continue;
            }

            if left_count <= 0.0 || right_count <= 0.0 {
                continue;
            }

            let left_mean = left_sum / left_count;
            let right_mean = right_sum / right_count;

            // Friedman MSE improvement
            let improvement =
                (left_count * right_count / parent_count) * (left_mean - right_mean).powi(2);

            if improvement > best_improvement {
                best_improvement = improvement;
                best_feature = feature_index;
                best_bin = bin + 1; // Split point is just after this bin

                // Get actual threshold value from bin edges
                let edges = &bin_edges[feature_index];
                best_threshold_value = if (best_bin as usize) < edges.len() {
                    edges[best_bin as usize]
                } else if !edges.is_empty() {
                    edges[edges.len() - 1]
                } else {
                    0.0
                };
            }
        }
    }

    (
        best_feature,
        best_bin,
        best_improvement,
        best_threshold_value,
    )
}

/// Returns the default histogram learner (faster alternative to DecisionTreeLearner)
pub fn histogram_learner() -> HistogramLearner {
    HistogramLearner::default_histogram()
}

// ============================================================================
// Decision Tree Learner (configurable depth, matching sklearn's default behavior)
// ============================================================================

/// A decision tree regressor with configurable max depth.
/// This matches sklearn's DecisionTreeRegressor with criterion="friedman_mse".
#[derive(Clone)]
pub struct DecisionTreeLearner {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

impl DecisionTreeLearner {
    pub fn new(max_depth: usize) -> Self {
        DecisionTreeLearner {
            max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Create a decision tree learner matching Python's default settings
    pub fn default_sklearn() -> Self {
        DecisionTreeLearner {
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    pub fn with_params(
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        DecisionTreeLearner {
            max_depth,
            min_samples_split,
            min_samples_leaf,
        }
    }
}

impl BaseLearner for DecisionTreeLearner {
    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }

        let indices: Vec<usize> = (0..x.nrows()).collect();
        // Pre-allocate sort buffer once — reused across all recursive calls
        let mut sort_buf: Vec<(f64, usize)> = Vec::with_capacity(x.nrows());
        let root = build_tree_node(
            x,
            y,
            sample_weight,
            &indices,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            &mut sort_buf,
        );

        Ok(Box::new(TrainedDecisionTree {
            root,
            n_features: x.ncols(),
        }))
    }
}

// ============================================================================
// Arena-style tree nodes using flat vector storage for cache efficiency
// ============================================================================

/// Sentinel value indicating no child (used for leaf nodes)
const NO_CHILD: u32 = u32::MAX;

/// Compact tree node using indices instead of Box pointers.
/// All nodes are stored in a contiguous Vec for better cache locality.
/// This is an "arena-style" allocation pattern without lifetime complexity.
#[derive(Clone, Serialize, Deserialize)]
pub struct ArenaTreeNode {
    /// Feature index for split nodes, unused for leaves
    pub feature_index: u16,
    /// Left child index (NO_CHILD for leaves)
    pub left: u32,
    /// Right child index (NO_CHILD for leaves)
    pub right: u32,
    /// Split threshold for split nodes, prediction value for leaves
    pub threshold_or_value: f64,
}

impl ArenaTreeNode {
    #[inline]
    fn is_leaf(&self) -> bool {
        self.left == NO_CHILD
    }

    #[inline]
    fn new_leaf(value: f64) -> Self {
        ArenaTreeNode {
            feature_index: 0,
            left: NO_CHILD,
            right: NO_CHILD,
            threshold_or_value: value,
        }
    }

    #[inline]
    fn new_split(feature_index: usize, threshold: f64, left: u32, right: u32) -> Self {
        ArenaTreeNode {
            feature_index: feature_index as u16,
            left,
            right,
            threshold_or_value: threshold,
        }
    }
}

/// Legacy tree node enum for backward compatibility with serialization.
/// New code should use ArenaTreeNode for better performance.
#[derive(Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        value: f64,
    },
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    /// Predict for a single row using direct array indexing (no allocation)
    #[inline]
    fn predict_row(&self, x: &Array2<f64>, row_idx: usize) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if x[[row_idx, *feature_index]] < *threshold {
                    left.predict_row(x, row_idx)
                } else {
                    right.predict_row(x, row_idx)
                }
            }
        }
    }

    fn collect_split_features(&self, features: &mut Vec<usize>) {
        match self {
            TreeNode::Leaf { .. } => {}
            TreeNode::Split {
                feature_index,
                left,
                right,
                ..
            } => {
                features.push(*feature_index);
                left.collect_split_features(features);
                right.collect_split_features(features);
            }
        }
    }

    /// Convert legacy TreeNode to arena format
    fn to_arena_nodes(&self, nodes: &mut Vec<ArenaTreeNode>) -> u32 {
        let idx = nodes.len() as u32;
        match self {
            TreeNode::Leaf { value } => {
                nodes.push(ArenaTreeNode::new_leaf(*value));
                idx
            }
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                // Reserve space for this node
                nodes.push(ArenaTreeNode::new_leaf(0.0)); // placeholder
                let left_idx = left.to_arena_nodes(nodes);
                let right_idx = right.to_arena_nodes(nodes);
                nodes[idx as usize] =
                    ArenaTreeNode::new_split(*feature_index, *threshold, left_idx, right_idx);
                idx
            }
        }
    }
}

/// Arena-based decision tree with flat node storage.
/// Provides better cache locality and fewer allocations than Box-based trees.
#[derive(Clone, Serialize, Deserialize)]
pub struct ArenaDecisionTree {
    /// All nodes stored contiguously. Root is always at index 0.
    pub nodes: Vec<ArenaTreeNode>,
    pub n_features: usize,
}

impl ArenaDecisionTree {
    /// Predict for a single row using direct array indexing (no allocation)
    #[inline]
    fn predict_row(&self, x: &Array2<f64>, row_idx: usize) -> f64 {
        let mut node_idx = 0usize;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.threshold_or_value;
            }
            let feature_val = x[[row_idx, node.feature_index as usize]];
            if feature_val < node.threshold_or_value {
                node_idx = node.left as usize;
            } else {
                node_idx = node.right as usize;
            }
        }
    }

    fn collect_split_features(&self, node_idx: usize, features: &mut Vec<usize>) {
        let node = &self.nodes[node_idx];
        if !node.is_leaf() {
            features.push(node.feature_index as usize);
            self.collect_split_features(node.left as usize, features);
            self.collect_split_features(node.right as usize, features);
        }
    }

    /// Convert from legacy TreeNode format
    pub fn from_tree_node(root: &TreeNode, n_features: usize) -> Self {
        let mut nodes = Vec::new();
        root.to_arena_nodes(&mut nodes);
        ArenaDecisionTree { nodes, n_features }
    }
}

impl TrainedBaseLearner for ArenaDecisionTree {
    #[inline]
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            predictions[i] = self.predict_row(x, i);
        }
        predictions
    }

    fn split_features(&self) -> Option<Vec<usize>> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut features = Vec::new();
        self.collect_split_features(0, &mut features);
        if features.is_empty() {
            None
        } else {
            Some(features)
        }
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        let features = self.split_features()?;
        let mut importances = Array1::zeros(self.n_features);
        for &f in &features {
            if f < self.n_features {
                importances[f] += 1.0;
            }
        }
        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances.mapv_inplace(|v| v / sum);
        }
        Some(importances)
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::ArenaTree(self.clone()))
    }
}

/// Arena-based decision tree learner - faster than DecisionTreeLearner.
/// Uses flat vector storage for better cache locality and fewer allocations.
#[derive(Clone)]
pub struct ArenaDecisionTreeLearner {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
}

impl ArenaDecisionTreeLearner {
    pub fn new(max_depth: usize) -> Self {
        ArenaDecisionTreeLearner {
            max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Create an arena tree learner matching Python's default settings
    pub fn default_sklearn() -> Self {
        ArenaDecisionTreeLearner {
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    pub fn with_params(
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        ArenaDecisionTreeLearner {
            max_depth,
            min_samples_split,
            min_samples_leaf,
        }
    }
}

impl BaseLearner for ArenaDecisionTreeLearner {
    fn fit_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }

        let indices: Vec<usize> = (0..x.nrows()).collect();

        // Pre-allocate nodes vector based on max possible tree size
        // A complete binary tree of depth d has 2^(d+1) - 1 nodes
        let max_nodes = (1usize << (self.max_depth + 1)).saturating_sub(1);
        let mut nodes = Vec::with_capacity(max_nodes.min(indices.len() * 2));
        // Pre-allocate sort buffer once — reused across all recursive calls
        let mut sort_buf: Vec<(f64, usize)> = Vec::with_capacity(x.nrows());

        build_arena_tree_node(
            &mut nodes,
            x,
            y,
            sample_weight,
            &indices,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            &mut sort_buf,
        );

        Ok(Box::new(ArenaDecisionTree {
            nodes,
            n_features: x.ncols(),
        }))
    }
}

/// Build arena tree node recursively, returning the index of the created node
fn build_arena_tree_node(
    nodes: &mut Vec<ArenaTreeNode>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    sort_buf: &mut Vec<(f64, usize)>,
) -> u32 {
    // Calculate parent_sum and parent_weight once — used for both node_value and split finding
    let (parent_sum, parent_weight) = if let Some(weights) = sample_weight {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for &i in indices {
            sum += y[i] * weights[i];
            weight += weights[i];
        }
        (sum, weight)
    } else {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        (sum, indices.len() as f64)
    };

    let node_value = if parent_weight > 0.0 {
        parent_sum / parent_weight
    } else {
        0.0
    };

    // Check stopping conditions
    if depth >= max_depth || indices.len() < min_samples_split {
        let idx = nodes.len() as u32;
        nodes.push(ArenaTreeNode::new_leaf(node_value));
        return idx;
    }

    // Find the best split using Friedman MSE improvement — pass pre-computed parent stats
    let (best_feature, best_threshold, best_improvement) = find_best_split_friedman(
        x,
        y,
        sample_weight,
        indices,
        min_samples_leaf,
        parent_sum,
        parent_weight,
        sort_buf,
    );

    // If no valid split found, return a leaf
    if best_improvement <= 0.0 || best_threshold.is_nan() {
        let idx = nodes.len() as u32;
        nodes.push(ArenaTreeNode::new_leaf(node_value));
        return idx;
    }

    // Partition the data - pre-allocate with estimated capacity
    let estimated_size = indices.len() / 2;
    let mut left_indices = Vec::with_capacity(estimated_size);
    let mut right_indices = Vec::with_capacity(estimated_size);
    for &i in indices {
        if x[[i, best_feature]] < best_threshold {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    // Check min_samples_leaf constraint
    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        let idx = nodes.len() as u32;
        nodes.push(ArenaTreeNode::new_leaf(node_value));
        return idx;
    }

    // Reserve space for this node (will update with child indices later)
    let node_idx = nodes.len() as u32;
    nodes.push(ArenaTreeNode::new_leaf(0.0)); // placeholder

    // Recursively build children
    let left_idx = build_arena_tree_node(
        nodes,
        x,
        y,
        sample_weight,
        &left_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        sort_buf,
    );
    let right_idx = build_arena_tree_node(
        nodes,
        x,
        y,
        sample_weight,
        &right_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        sort_buf,
    );

    // Update the placeholder with actual split info
    nodes[node_idx as usize] =
        ArenaTreeNode::new_split(best_feature, best_threshold, left_idx, right_idx);

    node_idx
}

/// Returns the default arena tree learner - faster alternative to default_tree_learner
pub fn arena_tree_learner() -> ArenaDecisionTreeLearner {
    ArenaDecisionTreeLearner::default_sklearn()
}

/// Legacy trained decision tree (kept for backward compatibility)
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedDecisionTree {
    pub root: TreeNode,
    pub n_features: usize,
}

impl TrainedBaseLearner for TrainedDecisionTree {
    #[inline]
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            predictions[i] = self.root.predict_row(x, i);
        }
        predictions
    }

    fn split_features(&self) -> Option<Vec<usize>> {
        let mut features = Vec::new();
        self.root.collect_split_features(&mut features);
        if features.is_empty() {
            None
        } else {
            Some(features)
        }
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        let features = self.split_features()?;
        let mut importances = Array1::zeros(self.n_features);
        for &f in &features {
            if f < self.n_features {
                importances[f] += 1.0;
            }
        }
        let sum: f64 = importances.sum();
        if sum > 0.0 {
            importances.mapv_inplace(|v| v / sum);
        }
        Some(importances)
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::DecisionTree(self.clone()))
    }
}

/// Build a tree node recursively
fn build_tree_node(
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    sort_buf: &mut Vec<(f64, usize)>,
) -> TreeNode {
    // Calculate parent_sum and parent_weight once — used for both node_value and split finding
    let (parent_sum, parent_weight) = if let Some(weights) = sample_weight {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for &i in indices {
            sum += y[i] * weights[i];
            weight += weights[i];
        }
        (sum, weight)
    } else {
        let sum: f64 = indices.iter().map(|&i| y[i]).sum();
        (sum, indices.len() as f64)
    };

    let node_value = if parent_weight > 0.0 {
        parent_sum / parent_weight
    } else {
        0.0
    };

    // Check stopping conditions
    if depth >= max_depth || indices.len() < min_samples_split {
        return TreeNode::Leaf { value: node_value };
    }

    // Find the best split using Friedman MSE improvement — pass pre-computed parent stats
    let (best_feature, best_threshold, best_improvement) = find_best_split_friedman(
        x,
        y,
        sample_weight,
        indices,
        min_samples_leaf,
        parent_sum,
        parent_weight,
        sort_buf,
    );

    // If no valid split found, return a leaf
    if best_improvement <= 0.0 || best_threshold.is_nan() {
        return TreeNode::Leaf { value: node_value };
    }

    // Partition the data - pre-allocate with estimated capacity
    let estimated_size = indices.len() / 2;
    let mut left_indices = Vec::with_capacity(estimated_size);
    let mut right_indices = Vec::with_capacity(estimated_size);
    for &i in indices {
        if x[[i, best_feature]] < best_threshold {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    // Check min_samples_leaf constraint
    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        return TreeNode::Leaf { value: node_value };
    }

    // Recursively build children
    let left_child = build_tree_node(
        x,
        y,
        sample_weight,
        &left_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        sort_buf,
    );
    let right_child = build_tree_node(
        x,
        y,
        sample_weight,
        &right_indices,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        sort_buf,
    );

    TreeNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Evaluate a single feature for the best split using Friedman MSE criterion.
/// Returns (threshold, improvement) for this feature.
#[inline]
fn evaluate_feature_split(
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    feature_index: usize,
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
    sort_buf: &mut Vec<(f64, usize)>,
) -> (f64, f64) {
    let feature_col = x.column(feature_index);

    // Build (feature_value, index) pairs — sort comparisons then access values
    // directly instead of doing indirect lookups through feature_col[idx]
    sort_buf.clear();
    sort_buf.extend(indices.iter().map(|&i| (feature_col[i], i)));
    sort_buf.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;
    let mut left_sum = 0.0;
    let mut left_weight = 0.0;
    let n = indices.len();

    // Split into weighted vs unweighted paths to avoid per-element branch
    if let Some(weights) = sample_weight {
        for pos in 0..n {
            let (feat_i, i) = sort_buf[pos];
            let w = weights[i];
            left_sum += y[i] * w;
            left_weight += w;

            let left_count = pos + 1;
            let right_count = n - left_count;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            if pos + 1 < n && (feat_i - sort_buf[pos + 1].0).abs() < 1e-10 {
                continue;
            }

            let right_sum = parent_sum - left_sum;
            let right_weight = parent_weight - left_weight;

            if left_weight <= 0.0 || right_weight <= 0.0 {
                continue;
            }

            let left_mean = left_sum / left_weight;
            let right_mean = right_sum / right_weight;
            let diff = left_mean - right_mean;
            let improvement = (left_weight * right_weight / parent_weight) * diff * diff + 1e-10;

            if improvement > best_improvement {
                best_improvement = improvement;
                best_threshold = if pos + 1 < n {
                    (feat_i + sort_buf[pos + 1].0) * 0.5
                } else {
                    feat_i
                };
            }
        }
    } else {
        // Unweighted path — no per-element weight lookup
        for pos in 0..n {
            let (feat_i, i) = sort_buf[pos];
            left_sum += y[i];
            left_weight += 1.0;

            let left_count = pos + 1;
            let right_count = n - left_count;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            if pos + 1 < n && (feat_i - sort_buf[pos + 1].0).abs() < 1e-10 {
                continue;
            }

            let right_sum = parent_sum - left_sum;
            let right_weight = parent_weight - left_weight;

            if right_weight <= 0.0 {
                continue;
            }

            let left_mean = left_sum / left_weight;
            let right_mean = right_sum / right_weight;
            let diff = left_mean - right_mean;
            let improvement = (left_weight * right_weight / parent_weight) * diff * diff + 1e-10;

            if improvement > best_improvement {
                best_improvement = improvement;
                best_threshold = if pos + 1 < n {
                    (feat_i + sort_buf[pos + 1].0) * 0.5
                } else {
                    feat_i
                };
            }
        }
    }

    (best_threshold, best_improvement)
}

/// Find the best split using Friedman MSE criterion (variance reduction weighted by sample counts)
/// This matches sklearn's "friedman_mse" criterion.
/// `sort_buf` is a reusable scratch buffer to avoid per-feature allocations.
/// Uses parallel feature evaluation when the workload is large enough.
fn find_best_split_friedman(
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
    sort_buf: &mut Vec<(f64, usize)>,
) -> (usize, f64, f64) {
    let n_features = x.ncols();

    if parent_weight == 0.0 {
        return (0, f64::NAN, 0.0);
    }

    // Parallel path: when the total work (n_features * n_samples * log(n_samples)) is large
    // enough to justify rayon overhead. Rayon has ~5-10µs per task overhead, so we need each
    // feature's work to be substantial. With 10 features and 500 samples, each feature sort
    // takes ~10µs which is too small. We need at least ~50µs per feature to see benefit.
    // n_samples >= 1000 with 10+ features gives ~30µs/feature sort — marginal.
    // Only parallelize for genuinely large workloads.
    if n_features >= 10 && indices.len() >= 5000 {
        use rayon::prelude::*;

        let results: Vec<(f64, f64)> = (0..n_features)
            .into_par_iter()
            .map(|feature_index| {
                let mut local_sort_buf: Vec<(f64, usize)> = Vec::with_capacity(indices.len());
                evaluate_feature_split(
                    x,
                    y,
                    sample_weight,
                    indices,
                    feature_index,
                    min_samples_leaf,
                    parent_sum,
                    parent_weight,
                    &mut local_sort_buf,
                )
            })
            .collect();

        let mut best_feature = 0;
        let mut best_threshold = f64::NAN;
        let mut best_improvement = 0.0;
        for (feature_index, &(threshold, improvement)) in results.iter().enumerate() {
            if improvement > best_improvement {
                best_improvement = improvement;
                best_feature = feature_index;
                best_threshold = threshold;
            }
        }
        return (best_feature, best_threshold, best_improvement);
    }

    // Sequential path: reuse sort buffer for small workloads
    let mut best_feature = 0;
    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;

    for feature_index in 0..n_features {
        let (threshold, improvement) = evaluate_feature_split(
            x,
            y,
            sample_weight,
            indices,
            feature_index,
            min_samples_leaf,
            parent_sum,
            parent_weight,
            sort_buf,
        );
        if improvement > best_improvement {
            best_improvement = improvement;
            best_feature = feature_index;
            best_threshold = threshold;
        }
    }

    (best_feature, best_threshold, best_improvement)
}

/// Returns the default tree learner matching Python's sklearn default:
/// DecisionTreeRegressor with max_depth=3 and criterion="friedman_mse"
pub fn default_tree_learner() -> DecisionTreeLearner {
    DecisionTreeLearner::default_sklearn()
}

/// Returns a stump learner (depth-1 tree) for simpler/faster models
pub fn stump_learner() -> StumpLearner {
    StumpLearner
}

/// Calculate weighted mean
/// Computes weighted mean using ndarray's dot product (SIMD-enabled).
#[inline]
fn weighted_mean(y: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let total_weight = weights.sum();
    if total_weight > 0.0 {
        y.dot(weights) / total_weight
    } else {
        y.mean().unwrap_or(0.0)
    }
}

/// Find best split with weighted samples using Friedman MSE criterion.
/// Optimized to O(n log n) per feature using cumulative sums instead of O(n²).
fn find_best_split_weighted(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
) -> (usize, f64, f64) {
    let n_features = x.shape()[1];
    let n_samples = y.len();
    let mut best_feature = 0;
    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;

    // Pre-compute total weighted sum
    let total_weight: f64 = weights.sum();
    let total_sum: f64 = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();

    if total_weight == 0.0 {
        return (best_feature, best_threshold, f64::INFINITY);
    }

    // Reusable buffer for sorted indices
    let mut sorted_indices: Vec<usize> = (0..n_samples).collect();

    for feature_index in 0..n_features {
        // Sort indices by feature value
        sorted_indices.clear();
        sorted_indices.extend(0..n_samples);
        sorted_indices.sort_by(|&a, &b| {
            x[[a, feature_index]]
                .partial_cmp(&x[[b, feature_index]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scan through sorted samples, maintaining cumulative sums
        let mut left_weight = 0.0;
        let mut left_sum = 0.0;

        for pos in 0..(n_samples - 1) {
            let i = sorted_indices[pos];
            let yi = y[i];
            let wi = weights[i];

            left_weight += wi;
            left_sum += yi * wi;

            let right_weight = total_weight - left_weight;
            let right_sum = total_sum - left_sum;

            // Skip if either side has zero weight
            if left_weight <= 0.0 || right_weight <= 0.0 {
                continue;
            }

            // Skip if next sample has the same feature value (no valid split point)
            let next_i = sorted_indices[pos + 1];
            if (x[[i, feature_index]] - x[[next_i, feature_index]]).abs() < 1e-10 {
                continue;
            }

            let left_mean = left_sum / left_weight;
            let right_mean = right_sum / right_weight;

            // Friedman MSE improvement: weighted variance reduction
            let improvement =
                (left_weight * right_weight / total_weight) * (left_mean - right_mean).powi(2);

            if improvement > best_improvement {
                best_improvement = improvement;
                best_feature = feature_index;
                // Threshold is midpoint between current and next value
                best_threshold = (x[[i, feature_index]] + x[[next_i, feature_index]]) / 2.0;
            }
        }
    }

    // Convert improvement back to MSE for compatibility (lower is better)
    // Note: The actual MSE value isn't used by the caller, only the threshold matters
    let best_mse = if best_improvement > 0.0 {
        1.0 / best_improvement
    } else {
        f64::INFINITY
    };

    (best_feature, best_threshold, best_mse)
}
