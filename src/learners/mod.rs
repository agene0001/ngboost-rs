use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

pub trait BaseLearner {
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
        }
    }
}

impl TrainedBaseLearner for SerializableTrainedLearner {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.predict(x),
            SerializableTrainedLearner::DecisionTree(t) => t.predict(x),
            SerializableTrainedLearner::HistogramTree(h) => h.predict(x),
        }
    }

    fn feature_importances(&self) -> Option<Array1<f64>> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.feature_importances(),
            SerializableTrainedLearner::DecisionTree(t) => t.feature_importances(),
            SerializableTrainedLearner::HistogramTree(h) => h.feature_importances(),
        }
    }

    fn split_feature(&self) -> Option<usize> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.split_feature(),
            SerializableTrainedLearner::DecisionTree(t) => t.split_feature(),
            SerializableTrainedLearner::HistogramTree(h) => h.split_feature(),
        }
    }

    fn split_features(&self) -> Option<Vec<usize>> {
        match self {
            SerializableTrainedLearner::Stump(s) => s.split_features(),
            SerializableTrainedLearner::DecisionTree(t) => t.split_features(),
            SerializableTrainedLearner::HistogramTree(h) => h.split_features(),
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

        let mut left_y = Vec::new();
        let mut left_weights = Vec::new();
        let mut right_y = Vec::new();
        let mut right_weights = Vec::new();

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
            .map(|f| compute_bin_edges(x.column(f).as_slice().unwrap(), self.max_bins))
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
    fn predict_single(&self, row: &[f64]) -> f64 {
        match self {
            HistTreeNode::Leaf { value } => *value,
            HistTreeNode::Split {
                feature_index,
                threshold_value,
                left,
                right,
                ..
            } => {
                if row[*feature_index] < *threshold_value {
                    left.predict_single(row)
                } else {
                    right.predict_single(row)
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
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for (i, row) in x.rows().into_iter().enumerate() {
            let row_vec: Vec<f64> = row.iter().cloned().collect();
            predictions[i] = self.root.predict_single(&row_vec);
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

    // Partition indices
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
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
        let root = build_tree_node(
            x,
            y,
            sample_weight,
            &indices,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        );

        Ok(Box::new(TrainedDecisionTree {
            root,
            n_features: x.ncols(),
        }))
    }
}

/// Internal tree node structure
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
    fn predict_single(&self, row: &[f64]) -> f64 {
        match self {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if row[*feature_index] < *threshold {
                    left.predict_single(row)
                } else {
                    right.predict_single(row)
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
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedDecisionTree {
    pub root: TreeNode,
    pub n_features: usize,
}

impl TrainedBaseLearner for TrainedDecisionTree {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(x.nrows());
        for (i, row) in x.rows().into_iter().enumerate() {
            // Use to_vec() to handle non-contiguous rows
            let row_vec: Vec<f64> = row.iter().cloned().collect();
            predictions[i] = self.root.predict_single(&row_vec);
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
) -> TreeNode {
    // Calculate the weighted mean for this node
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
        return TreeNode::Leaf { value: node_value };
    }

    // Find the best split using Friedman MSE improvement
    let (best_feature, best_threshold, best_improvement) =
        find_best_split_friedman(x, y, sample_weight, indices, min_samples_leaf);

    // If no valid split found, return a leaf
    if best_improvement <= 0.0 || best_threshold.is_nan() {
        return TreeNode::Leaf { value: node_value };
    }

    // Partition the data
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
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
    );

    TreeNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Find the best split using Friedman MSE criterion (variance reduction weighted by sample counts)
/// This matches sklearn's "friedman_mse" criterion
fn find_best_split_friedman(
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    min_samples_leaf: usize,
) -> (usize, f64, f64) {
    let n_features = x.ncols();
    let mut best_feature = 0;
    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;

    // Calculate parent statistics
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

    if parent_weight == 0.0 {
        return (best_feature, best_threshold, best_improvement);
    }

    for feature_index in 0..n_features {
        // Sort indices by feature value
        let mut sorted_indices: Vec<usize> = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            x[[a, feature_index]]
                .partial_cmp(&x[[b, feature_index]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Use cumulative sums for efficient split evaluation
        let mut left_sum = 0.0;
        let mut left_weight = 0.0;

        for (pos, &i) in sorted_indices.iter().enumerate() {
            let y_val = y[i];
            let w = sample_weight.map_or(1.0, |weights| weights[i]);

            left_sum += y_val * w;
            left_weight += w;

            let left_count = pos + 1;
            let right_count = indices.len() - left_count;

            // Check min_samples_leaf
            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            // Skip if next sample has the same feature value (no split point here)
            if pos + 1 < sorted_indices.len() {
                let next_i = sorted_indices[pos + 1];
                if (x[[i, feature_index]] - x[[next_i, feature_index]]).abs() < 1e-10 {
                    continue;
                }
            }

            let right_sum = parent_sum - left_sum;
            let right_weight = parent_weight - left_weight;

            if left_weight <= 0.0 || right_weight <= 0.0 {
                continue;
            }

            let left_mean = left_sum / left_weight;
            let right_mean = right_sum / right_weight;

            // Friedman MSE improvement: weighted variance reduction
            // improvement = (n_left * n_right / n_total) * (mean_left - mean_right)^2
            // This exactly matches sklearn's friedman_mse criterion
            let improvement =
                (left_weight * right_weight / parent_weight) * (left_mean - right_mean).powi(2);

            // Add small epsilon to avoid numerical issues with identical means
            let improvement = improvement + 1e-10;

            if improvement > best_improvement {
                best_improvement = improvement;
                best_feature = feature_index;
                // Threshold is midpoint between current and next value
                if pos + 1 < sorted_indices.len() {
                    let next_i = sorted_indices[pos + 1];
                    best_threshold = (x[[i, feature_index]] + x[[next_i, feature_index]]) / 2.0;
                } else {
                    best_threshold = x[[i, feature_index]];
                }
            }
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
fn weighted_mean(y: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let weighted_sum: f64 = y
        .iter()
        .zip(weights.iter())
        .map(|(&y_i, &w_i)| y_i * w_i)
        .sum();
    let total_weight: f64 = weights.sum();
    if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        y.mean().unwrap_or(0.0)
    }
}

/// Find best split with weighted samples
fn find_best_split_weighted(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
) -> (usize, f64, f64) {
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
            let mut left_weights = Vec::new();
            let mut right_y = Vec::new();
            let mut right_weights = Vec::new();

            for i in 0..y.len() {
                if feature_values[i] < threshold {
                    left_y.push(y[i]);
                    left_weights.push(weights[i]);
                } else {
                    right_y.push(y[i]);
                    right_weights.push(weights[i]);
                }
            }

            if left_y.is_empty() || right_y.is_empty() {
                continue;
            }

            let left_y_arr = Array1::from(left_y);
            let right_y_arr = Array1::from(right_y);
            let left_weights_arr = Array1::from(left_weights);
            let right_weights_arr = Array1::from(right_weights);

            let mean_left = weighted_mean(&left_y_arr, &left_weights_arr);
            let mean_right = weighted_mean(&right_y_arr, &right_weights_arr);

            let mse_left = left_y_arr
                .iter()
                .zip(left_weights_arr.iter())
                .map(|(&v, &w)| w * (v - mean_left).powi(2))
                .sum::<f64>()
                / left_weights_arr.sum();
            let mse_right = right_y_arr
                .iter()
                .zip(right_weights_arr.iter())
                .map(|(&v, &w)| w * (v - mean_right).powi(2))
                .sum::<f64>()
                / right_weights_arr.sum();

            let n_left = left_weights_arr.sum();
            let n_right = right_weights_arr.sum();
            let n_total = weights.sum();

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
