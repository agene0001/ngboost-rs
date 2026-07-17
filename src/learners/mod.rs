use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Opaque, learner-specific preprocessing of X that can be reused across many
/// fits on the same X with different targets — exactly the access pattern of
/// gradient boosting, where every iteration refits trees on the same features.
/// Built via [`BaseLearner::build_fit_cache`]; consumed by
/// [`BaseLearner::fit_with_weights_cached`].
pub trait FitCache: Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
}

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

    /// Build a reusable preprocessing cache for fitting repeatedly on the same
    /// `x` (e.g. bin edges + binned matrix for histogram trees). Learners that
    /// don't benefit return `None` (the default), which costs nothing.
    fn build_fit_cache(&self, _x: &Array2<f64>) -> Option<std::sync::Arc<dyn FitCache>> {
        None
    }

    /// Fit using a cache previously built by [`BaseLearner::build_fit_cache`]
    /// **on the same `x`**. The default ignores the cache and fits normally.
    fn fit_with_weights_cached(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        _cache: &dyn FitCache,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        self.fit_with_weights(x, y, sample_weight)
    }

    /// Fit and return both the fitted learner and its predictions on the
    /// training `x`. The default fits, then predicts. Learners that can
    /// harvest the training predictions during fitting itself (a histogram
    /// tree already routes every row to its leaf while partitioning)
    /// override this to skip the second walk over the tree — the returned
    /// predictions must be bit-identical to `fitted.predict(x)`.
    fn fit_predict_cached(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        cache: Option<&dyn FitCache>,
    ) -> Result<(Box<dyn TrainedBaseLearner>, Array1<f64>), &'static str> {
        let fitted = match cache {
            Some(c) => self.fit_with_weights_cached(x, y, sample_weight, c)?,
            None => self.fit_with_weights(x, y, sample_weight)?,
        };
        let preds = fitted.predict(x);
        Ok((fitted, preds))
    }

    /// Fit on the row subset `rows` of `x` using a cache built on the FULL
    /// `x`, returning the fitted learner and its predictions for those rows
    /// (in `rows` order). `y` and `sample_weight` are indexed by the cache's
    /// row ids (full length); entries outside `rows` are never read.
    ///
    /// The default materializes the row subset and fits it from scratch.
    /// Learners with a usable cache (histogram trees) override this to fit
    /// directly on the subset indices, reusing the full-X bin edges instead
    /// of re-binning every minibatch — LightGBM-style bagging. NOTE: for
    /// such learners the full-X quantile edges are a deliberate behavioral
    /// difference from re-binning the minibatch.
    fn fit_predict_cached_rows(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        _cache: &dyn FitCache,
        rows: &[u32],
    ) -> Result<(Box<dyn TrainedBaseLearner>, Array1<f64>), &'static str> {
        let idx: Vec<usize> = rows.iter().map(|&r| r as usize).collect();
        let x_sub = x.select(ndarray::Axis(0), &idx);
        let y_sub = y.select(ndarray::Axis(0), &idx);
        let w_sub = sample_weight.map(|w| w.select(ndarray::Axis(0), &idx));
        self.fit_predict_cached(&x_sub, &y_sub, w_sub.as_ref(), None)
    }
}

pub trait TrainedBaseLearner: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Predict a block of rows given as one contiguous row-major slice
    /// (`out.len()` rows × `n_cols` columns). When `col_idxs` is given, the
    /// learner was fit on that column subset: its internal feature index `f`
    /// refers to full-matrix column `col_idxs[f]`.
    ///
    /// Default implementation materializes the (sub)matrix and delegates to
    /// `predict`; tree learners override this with an allocation-free walk.
    fn predict_rows(&self, xs: &[f64], n_cols: usize, col_idxs: Option<&[usize]>, out: &mut [f64]) {
        let n_rows = out.len();
        let preds = match col_idxs {
            None => {
                let x = ndarray::ArrayView2::from_shape((n_rows, n_cols), xs)
                    .expect("predict_rows: shape mismatch")
                    .to_owned();
                self.predict(&x)
            }
            Some(idxs) => {
                let mut sub = Array2::zeros((n_rows, idxs.len()));
                for r in 0..n_rows {
                    let row = &xs[r * n_cols..(r + 1) * n_cols];
                    for (j, &c) in idxs.iter().enumerate() {
                        sub[[r, j]] = row[c];
                    }
                }
                self.predict(&sub)
            }
        };
        out.copy_from_slice(preds.as_slice().expect("predict output not contiguous"));
    }

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

        // A stump is a depth-1 tree: reuse the same Friedman scan as the tree
        // learners (for a single split, minimizing weighted MSE and maximizing
        // the Friedman improvement select the same threshold). This replaces
        // the old O(n_unique·n) per-feature loop with one O(n log n) scan and
        // inherits midpoint thresholds and NaN-tolerant sorting.
        let n = x.nrows();
        let (parent_sum, parent_weight) = if let Some(weights) = sample_weight {
            let sum = y.iter().zip(weights.iter()).map(|(&yi, &wi)| yi * wi).sum();
            (sum, weights.sum())
        } else {
            (y.sum(), n as f64)
        };
        let overall_mean = if parent_weight > 0.0 {
            parent_sum / parent_weight
        } else {
            0.0
        };

        let indices: Vec<usize> = (0..n).collect();
        let mut sort_buf: Vec<SortedEntry> = Vec::with_capacity(n);
        let (best_feature, best_threshold, best_improvement) = find_best_split_friedman(
            x,
            y,
            sample_weight,
            &indices,
            1,
            parent_sum,
            parent_weight,
            &mut sort_buf,
        );

        if best_improvement <= 0.0 || best_threshold.is_nan() {
            // No valid split: a NaN threshold marks the no-split stump
            // (split_feature() reports None; predict returns the mean).
            return Ok(Box::new(TrainedStumpLearner {
                feature_index: 0,
                threshold: f64::NAN,
                left_value: overall_mean,
                right_value: overall_mean,
            }));
        }

        // Weighted means on each side; compare via the f32-cast value to
        // match the scan's quantized view of the feature (as the tree
        // builders' partitions do).
        let col = x.column(best_feature);
        let (mut left_sum, mut left_w, mut right_sum, mut right_w) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let w = sample_weight.map_or(1.0, |wts| wts[i]);
            if (col[i] as f32 as f64) < best_threshold {
                left_sum += y[i] * w;
                left_w += w;
            } else {
                right_sum += y[i] * w;
                right_w += w;
            }
        }
        let left_value = if left_w > 0.0 {
            left_sum / left_w
        } else {
            overall_mean
        };
        let right_value = if right_w > 0.0 {
            right_sum / right_w
        } else {
            overall_mean
        };

        Ok(Box::new(TrainedStumpLearner {
            feature_index: best_feature,
            threshold: best_threshold,
            left_value,
            right_value,
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
            // f32-quantized comparison, matching the fit-time partition (and
            // sklearn, which casts X to float32 at predict): a raw f64 equal
            // to the f64 midpoint of two adjacent f32s could otherwise be
            // partitioned left at fit but routed right at predict.
            if ((val as f32) as f64) < self.threshold {
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

        // sklearn semantics: center by the WEIGHTED means of the raw data
        // first (the intercept is unpenalized), THEN scale rows by
        // sqrt(weight). Scaling before centering would compute "means" of
        // sqrt(w)-scaled rows — neither the weighted nor the unweighted mean —
        // e.g. uniform weights of 4 would double the intercept.
        let (x_mean, y_mean) = if self.fit_intercept {
            if let Some(weights) = sample_weight {
                let w_sum: f64 = weights.sum();
                if w_sum <= 0.0 {
                    // All-zero weights carry no information; match the tree
                    // learners' degenerate behavior of predicting 0.
                    return Ok(Box::new(TrainedRidgeLearner {
                        coefficients: vec![0.0; n_features],
                        intercept: 0.0,
                        n_features,
                    }));
                }
                let mut xm = Array1::<f64>::zeros(n_features);
                let mut ym = 0.0;
                for i in 0..n_samples {
                    let w = weights[i];
                    ym += w * y[i];
                    for j in 0..n_features {
                        xm[j] += w * x[[i, j]];
                    }
                }
                (Some(xm / w_sum), ym / w_sum)
            } else {
                (
                    Some(x.mean_axis(ndarray::Axis(0)).unwrap()),
                    y.mean().unwrap_or(0.0),
                )
            }
        } else {
            (None, 0.0)
        };

        // Center (if fitting an intercept), then scale rows by sqrt(weight)
        // so ordinary penalized least squares on the transformed data solves
        // the weighted problem.
        let mut x_t = x.clone();
        let mut y_t = y.clone();
        if let Some(xm) = &x_mean {
            x_t -= xm;
            y_t.mapv_inplace(|v| v - y_mean);
        }
        if let Some(weights) = sample_weight {
            for i in 0..n_samples {
                let sw = weights[i].sqrt();
                for j in 0..n_features {
                    x_t[[i, j]] *= sw;
                }
                y_t[i] *= sw;
            }
        }
        let (x_centered, y_centered) = (x_t, y_t);

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
        let cache = build_histogram_cache(x, self.max_bins);
        self.fit_from_cache(&cache, y, sample_weight)
    }

    fn build_fit_cache(&self, x: &Array2<f64>) -> Option<std::sync::Arc<dyn FitCache>> {
        if x.nrows() == 0 {
            return None;
        }
        Some(std::sync::Arc::new(build_histogram_cache(x, self.max_bins)))
    }

    fn fit_with_weights_cached(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        cache: &dyn FitCache,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        match cache.as_any().downcast_ref::<HistogramFitCache>() {
            Some(c) if c.n_rows == x.nrows() && c.n_features == x.ncols() => {
                self.fit_from_cache(c, y, sample_weight)
            }
            // Wrong cache type or stale dims — fall back to the uncached path
            _ => self.fit_with_weights(x, y, sample_weight),
        }
    }

    fn fit_predict_cached(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        cache: Option<&dyn FitCache>,
    ) -> Result<(Box<dyn TrainedBaseLearner>, Array1<f64>), &'static str> {
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }
        let mut preds = Array1::zeros(x.nrows());
        let preds_slice = preds
            .as_slice_mut()
            .expect("freshly allocated Array1 is contiguous");
        let fitted = match cache.and_then(|c| c.as_any().downcast_ref::<HistogramFitCache>()) {
            Some(c) if c.n_rows == x.nrows() && c.n_features == x.ncols() => {
                self.fit_from_cache_scattering(c, y, sample_weight, Some(preds_slice))?
            }
            // No / wrong / stale cache — bin x fresh, still leaf-scatter
            _ => {
                let c = build_histogram_cache(x, self.max_bins);
                self.fit_from_cache_scattering(&c, y, sample_weight, Some(preds_slice))?
            }
        };
        Ok((fitted, preds))
    }

    fn fit_predict_cached_rows(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        cache: &dyn FitCache,
        rows: &[u32],
    ) -> Result<(Box<dyn TrainedBaseLearner>, Array1<f64>), &'static str> {
        match cache.as_any().downcast_ref::<HistogramFitCache>() {
            Some(c) if c.n_rows == x.nrows() && c.n_features == x.ncols() => {
                // Full-length scatter buffer indexed by cache row ids; only
                // the subset's entries get written, then gathered in `rows`
                // order for the caller.
                let mut preds_full = vec![0.0f64; c.n_rows];
                let fitted = self.fit_indices_from_cache(
                    c,
                    y,
                    sample_weight,
                    rows.to_vec(),
                    Some(&mut preds_full),
                )?;
                let preds =
                    Array1::from_iter(rows.iter().map(|&r| preds_full[r as usize]));
                Ok((fitted, preds))
            }
            // Wrong cache type or stale dims — materialize the subset and
            // fit it from scratch (the trait default's behavior)
            _ => {
                let idx: Vec<usize> = rows.iter().map(|&r| r as usize).collect();
                let x_sub = x.select(ndarray::Axis(0), &idx);
                let y_sub = y.select(ndarray::Axis(0), &idx);
                let w_sub = sample_weight.map(|w| w.select(ndarray::Axis(0), &idx));
                self.fit_predict_cached(&x_sub, &y_sub, w_sub.as_ref(), None)
            }
        }
    }
}

impl HistogramLearner {
    fn fit_from_cache(
        &self,
        cache: &HistogramFitCache,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        self.fit_from_cache_scattering(cache, y, sample_weight, None)
    }

    /// Like [`fit_from_cache`], but when `preds` is given, every row's leaf
    /// value is written into it during the build (leaf-scatter) — the tree
    /// partition already routes each row to exactly one leaf, so the training
    /// predictions come for free instead of via a second tree walk. The
    /// scattered values are bit-identical to `fitted.predict(x)` (tested).
    fn fit_from_cache_scattering(
        &self,
        cache: &HistogramFitCache,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        preds: Option<&mut [f64]>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        let indices: Vec<u32> = (0..cache.n_rows as u32).collect();
        self.fit_indices_from_cache(cache, y, sample_weight, indices, preds)
    }

    /// Fit a tree on an arbitrary row-index subset of the cache. `y`,
    /// `sample_weight`, and the `preds` scatter buffer are all indexed by the
    /// CACHE's row ids (full length); rows outside `indices` are never read
    /// or written. This is how row-subsampled boosting iterations reuse the
    /// full-X binning instead of re-binning every minibatch.
    fn fit_indices_from_cache(
        &self,
        cache: &HistogramFitCache,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        indices: Vec<u32>,
        preds: Option<&mut [f64]>,
    ) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
        if y.len() != cache.n_rows {
            return Err("y length does not match cached X");
        }

        let root_hists = FeatureHists::build(cache, &indices, y, sample_weight);
        let root = build_histogram_tree_node(
            cache,
            y,
            sample_weight,
            indices,
            root_hists,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            preds,
        );

        Ok(Box::new(TrainedHistogramTree::new(root, cache.n_features)))
    }
}

/// Precomputed binning of X for histogram tree fitting: bin edges per feature
/// plus the row-major binned matrix. Building this is the expensive part
/// (a sort per feature); reusing it across boosting iterations makes each
/// subsequent fit O(n·p) instead of O(n·p·log n).
pub struct HistogramFitCache {
    bin_edges: Vec<Vec<f64>>,
    /// Bin of sample i, feature f at `binned[i * n_features + f]` (row-major)
    binned: Vec<u8>,
    n_rows: usize,
    n_features: usize,
    /// Unweighted ROOT bin counts per (feature, bin) slot. They depend only on
    /// the binned matrix, so every tree fit from this cache shares them —
    /// computed once on the first root build (OnceLock: the n_params trees of
    /// a boosting iteration fit in parallel against the same Arc'd cache).
    root_counts: std::sync::OnceLock<Vec<f64>>,
}

impl FitCache for HistogramFitCache {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Compute bin edges and the row-major binned matrix for X.
fn build_histogram_cache(x: &Array2<f64>, max_bins: usize) -> HistogramFitCache {
    // Bin indices are stored as u8, so resolution silently caps at 255 bins;
    // clamp here so edges, histograms, and split scans all agree on that cap.
    let max_bins = max_bins.min(255);
    let n_rows = x.nrows();
    let n_features = x.ncols();

    let make_edges = |f: usize| {
        let col_vec: Vec<f64> = x.column(f).to_vec();
        compute_bin_edges(&col_vec, max_bins)
    };

    // The per-feature edge computation sorts the column — parallelize for
    // large workloads, mirroring the presorted exact-tree root sort.
    let bin_edges: Vec<Vec<f64>> = if n_features >= 4 && n_rows >= 2000 {
        use rayon::prelude::*;
        (0..n_features).into_par_iter().map(make_edges).collect()
    } else {
        (0..n_features).map(make_edges).collect()
    };

    // Row-major binned matrix: one cache-friendly row per sample
    let mut binned = vec![0u8; n_rows * n_features];
    for i in 0..n_rows {
        let row = &mut binned[i * n_features..(i + 1) * n_features];
        for (f, b) in row.iter_mut().enumerate() {
            *b = find_bin(x[[i, f]], &bin_edges[f]);
        }
    }

    HistogramFitCache {
        bin_edges,
        binned,
        n_rows,
        n_features,
        root_counts: std::sync::OnceLock::new(),
    }
}

/// Per-feature histograms for one tree node: weighted count and y-sum per bin,
/// stored flat as `[feature * n_bins + bin]`.
///
/// NEGATIVE RESULT (2026-07): interleaving count/sum as `[f64; 2]` pairs
/// (the LightGBM histogram layout) benched consistently SLOWER at n≤5000 and
/// slower than this layout at every size on Apple Silicon — the ~20KB arrays
/// both sit in L1/L2, so the halved cache-line traffic never materializes and
/// the doubled index arithmetic just costs. Do not retry the interleave here.
struct FeatureHists {
    count: Vec<f64>,
    sum: Vec<f64>,
    /// Raw (unweighted) per-bin sample counts, tracked only for weighted fits.
    /// `min_samples_leaf` is a raw-count constraint (sklearn semantics);
    /// checking it against weighted counts would reject every split when
    /// weights are small (e.g. normalized to sum to 1). For unweighted fits
    /// the count half of `cs` already holds raw counts and this stays `None`.
    raw: Option<Vec<f64>>,
    n_bins: usize,
}

impl FeatureHists {
    /// Build histograms for all features in a single pass over the node's rows
    /// (row-major binned matrix keeps the inner loop sequential in memory).
    fn build(
        cache: &HistogramFitCache,
        indices: &[u32],
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Self {
        let p = cache.n_features;
        // Per-feature bin counts differ; use the max so the flat layout is uniform
        let n_bins = cache
            .bin_edges
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut count = vec![0.0; p * n_bins];
        let mut sum = vec![0.0; p * n_bins];
        let mut raw: Option<Vec<f64>> = None;

        if let Some(weights) = sample_weight {
            let raw = raw.get_or_insert_with(|| vec![0.0; p * n_bins]);
            for &i in indices {
                let i = i as usize;
                let w = weights[i];
                let yw = y[i] * w;
                let row = &cache.binned[i * p..(i + 1) * p];
                for (f, &b) in row.iter().enumerate() {
                    let slot = f * n_bins + b as usize;
                    count[slot] += w;
                    sum[slot] += yw;
                    raw[slot] += 1.0;
                }
            }
        } else if indices.len() == cache.n_rows {
            // ROOT node, unweighted: bin counts depend only on the binned
            // matrix, so seed them from the cache (computed once per cache,
            // shared by every tree of every boosting iteration) and
            // accumulate only the y-sums. Counts are exact integer sums of
            // 1.0 — identical to accumulating them per row.
            let counts = cache.root_counts.get_or_init(|| {
                let mut c = vec![0.0; p * n_bins];
                for &i in indices {
                    let row = &cache.binned[i as usize * p..(i as usize + 1) * p];
                    for (f, &b) in row.iter().enumerate() {
                        c[f * n_bins + b as usize] += 1.0;
                    }
                }
                c
            });
            count.copy_from_slice(counts);
            for &i in indices {
                let i = i as usize;
                let yi = y[i];
                let row = &cache.binned[i * p..(i + 1) * p];
                for (f, &b) in row.iter().enumerate() {
                    sum[f * n_bins + b as usize] += yi;
                }
            }
        } else {
            for &i in indices {
                let i = i as usize;
                let yi = y[i];
                let row = &cache.binned[i * p..(i + 1) * p];
                for (f, &b) in row.iter().enumerate() {
                    let slot = f * n_bins + b as usize;
                    count[slot] += 1.0;
                    sum[slot] += yi;
                }
            }
        }

        FeatureHists {
            count,
            sum,
            raw,
            n_bins,
        }
    }

    /// Sibling-subtraction trick: this node's histograms minus a child's give
    /// the other child's, without a second data pass. Consumes self — the
    /// parent's histograms are dead after the split — so the ~20KB vecs are
    /// mutated in place instead of reallocated at every split node.
    fn subtract(mut self, child: &FeatureHists) -> FeatureHists {
        for (a, b) in self.count.iter_mut().zip(child.count.iter()) {
            *a = (*a - b).max(0.0);
        }
        for (a, b) in self.sum.iter_mut().zip(child.sum.iter()) {
            *a -= b;
        }
        if let (Some(a), Some(b)) = (self.raw.as_mut(), child.raw.as_ref()) {
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x = (*x - y).max(0.0);
            }
        }
        self
    }

    /// Total (count, sum) over all bins of one feature.
    fn totals(&self, feature: usize) -> (f64, f64) {
        let base = feature * self.n_bins;
        let mut c = 0.0;
        let mut s = 0.0;
        for b in 0..self.n_bins {
            c += self.count[base + b];
            s += self.sum[base + b];
        }
        (c, s)
    }
}

/// Build a histogram tree node recursively. `hists` are this node's
/// per-feature histograms; children reuse them via the subtraction trick
/// (scan only the smaller child, derive the larger by subtraction).
#[allow(clippy::too_many_arguments)]
fn build_histogram_tree_node(
    cache: &HistogramFitCache,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: Vec<u32>,
    hists: FeatureHists,
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    mut preds: Option<&mut [f64]>,
) -> HistTreeNode {
    let (parent_count, parent_sum) = hists.totals(0);
    let node_value = if parent_count > 0.0 {
        parent_sum / parent_count
    } else {
        0.0
    };

    // Leaf-scatter: every row of a leaf gets the leaf's value — the exact
    // f64 `predict` would return for it (partition and predict route rows
    // identically; see `leaf_scatter_matches_predict` tests).
    let make_leaf = |indices: &[u32], preds: &mut Option<&mut [f64]>| {
        if let Some(p) = preds.as_deref_mut() {
            for &i in indices {
                p[i as usize] = node_value;
            }
        }
        HistTreeNode::Leaf { value: node_value }
    };

    // Check stopping conditions
    if depth >= max_depth || indices.len() < min_samples_split || parent_count <= 0.0 {
        return make_leaf(&indices, &mut preds);
    }

    // Find the best Friedman-MSE split by scanning bin prefix sums per feature
    let mut best_feature = 0;
    let mut best_bin: u8 = 0;
    let mut best_improvement = 0.0;
    let mut best_threshold_value = 0.0;

    // min_samples_leaf counts SAMPLES (sklearn semantics), so it must be
    // checked against raw counts; with weights, `count` holds weighted totals.
    let parent_raw = indices.len() as f64;
    let raw_bins = hists.raw.as_deref();

    for f in 0..cache.n_features {
        let n_edges = cache.bin_edges[f].len();
        if n_edges < 2 {
            continue; // constant feature — no split possible
        }
        let base = f * hists.n_bins;
        let mut left_count = 0.0;
        let mut left_sum = 0.0;
        let mut left_raw = 0.0;

        // Candidate split after each bin except the last occupied one
        for bin in 0..(n_edges - 1) {
            // A bin holding NO rows leaves every accumulator unchanged, so its
            // candidate is an exact duplicate of the previous one — never taken
            // under strict `>` and with identical gate outcomes. Skip it. With
            // weights, count can be 0 while zero-weight rows still occupy the
            // bin (raw > 0) and DO move the min_samples_leaf gate — so the
            // emptiness test must use raw counts there.
            let empty = match raw_bins {
                Some(r) => r[base + bin] == 0.0,
                None => hists.count[base + bin] == 0.0,
            };
            if empty {
                continue;
            }
            left_count += hists.count[base + bin];
            left_sum += hists.sum[base + bin];
            let left_raw = match raw_bins {
                Some(r) => {
                    left_raw += r[base + bin];
                    left_raw
                }
                // Unweighted: `count` is already the raw count
                None => left_count,
            };

            let right_count = parent_count - left_count;
            let right_sum = parent_sum - left_sum;

            if left_raw < min_samples_leaf as f64
                || parent_raw - left_raw < min_samples_leaf as f64
            {
                continue;
            }
            if left_count <= 0.0 || right_count <= 0.0 {
                continue;
            }

            let left_mean = left_sum / left_count;
            let right_mean = right_sum / right_count;
            let diff = left_mean - right_mean;
            let improvement = (left_count * right_count / parent_count) * diff * diff;

            if improvement > best_improvement {
                best_improvement = improvement;
                best_feature = f;
                best_bin = (bin + 1) as u8; // split point is just after this bin
                best_threshold_value = cache.bin_edges[f][bin + 1];
            }
        }
    }

    // If no valid split found
    if best_improvement <= 0.0 {
        return make_leaf(&indices, &mut preds);
    }

    // Partition row ids by bin
    let p = cache.n_features;
    let estimated = indices.len() / 2;
    let mut left_indices = Vec::with_capacity(estimated);
    let mut right_indices = Vec::with_capacity(estimated);
    for &i in &indices {
        if cache.binned[i as usize * p + best_feature] < best_bin {
            left_indices.push(i);
        } else {
            right_indices.push(i);
        }
    }

    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        return make_leaf(&indices, &mut preds);
    }

    // Histogram subtraction: scan the smaller child, derive the larger
    let (left_hists, right_hists) = if left_indices.len() <= right_indices.len() {
        let lh = FeatureHists::build(cache, &left_indices, y, sample_weight);
        let rh = hists.subtract(&lh);
        (lh, rh)
    } else {
        let rh = FeatureHists::build(cache, &right_indices, y, sample_weight);
        let lh = hists.subtract(&rh);
        (lh, rh)
    };

    let left_child = build_histogram_tree_node(
        cache,
        y,
        sample_weight,
        left_indices,
        left_hists,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        preds.as_deref_mut(),
    );
    let right_child = build_histogram_tree_node(
        cache,
        y,
        sample_weight,
        right_indices,
        right_hists,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        preds,
    );

    HistTreeNode::Split {
        feature_index: best_feature,
        bin_threshold: best_bin,
        threshold_value: best_threshold_value,
        improvement: best_improvement,
        left: Box::new(left_child),
        right: Box::new(right_child),
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
    // NaN goes to the LAST bin: prediction routes a row right whenever
    // `value < threshold` is false, which NaN always is, so the last bin is
    // the only placement consistent between fit-time partitioning and
    // prediction (and it avoids a partial_cmp panic in the binary search).
    if value.is_nan() {
        return (edges.len().saturating_sub(1)).min(254) as u8;
    }
    if edges.is_empty() || value <= edges[0] {
        return 0;
    }

    // Binary search for the right bin
    match edges.binary_search_by(|e| e.partial_cmp(&value).unwrap()) {
        Ok(i) => i.min(254) as u8,
        Err(i) => (i.saturating_sub(1)).min(254) as u8,
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
        /// Friedman-MSE improvement of this split (= the node's weighted
        /// impurity decrease, N_t·Var_t − N_l·Var_l − N_r·Var_r), recorded for
        /// sklearn-style feature importances. `serde(skip)` keeps the bincode
        /// wire format unchanged: models saved before this field existed still
        /// load (their importances fall back to split counts).
        #[serde(skip)]
        improvement: f64,
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

    fn collect_split_importances(&self, out: &mut Vec<(usize, f64)>) {
        match self {
            HistTreeNode::Leaf { .. } => {}
            HistTreeNode::Split {
                feature_index,
                improvement,
                left,
                right,
                ..
            } => {
                out.push((*feature_index, *improvement));
                left.collect_split_importances(out);
                right.collect_split_importances(out);
            }
        }
    }
}

/// sklearn-style feature importances from per-split (feature, improvement)
/// pairs: each split contributes its weighted impurity decrease — for MSE
/// trees that equals the Friedman improvement `N_l·N_r/N_t·(m̄_l−m̄_r)²`
/// recorded at fit time — normalized to sum to 1 (matching sklearn's
/// `tree_.compute_feature_importances()`, which Python ngboost aggregates).
/// Trees deserialized from models saved before improvements were recorded
/// carry zeros; those fall back to the previous split-count behavior.
fn importances_from_splits(pairs: &[(usize, f64)], n_features: usize) -> Option<Array1<f64>> {
    if pairs.is_empty() {
        return None;
    }
    let mut importances = Array1::zeros(n_features);
    let total: f64 = pairs.iter().map(|(_, imp)| imp).sum();
    if total > 0.0 {
        for &(f, imp) in pairs {
            if f < n_features {
                importances[f] += imp;
            }
        }
    } else {
        for &(f, _) in pairs {
            if f < n_features {
                importances[f] += 1.0;
            }
        }
    }
    let sum: f64 = importances.sum();
    if sum > 0.0 {
        importances.mapv_inplace(|v| v / sum);
    }
    Some(importances)
}

/// One node of the flattened prediction tree. Leaves are marked with
/// `feature == u32::MAX` and carry the leaf value in `threshold_or_value`.
#[derive(Clone, Copy)]
struct FlatHistNode {
    threshold_or_value: f64,
    feature: u32,
    left: u32,
    right: u32,
}

/// Flatten the boxed tree into a contiguous array (depth-first). A depth-3
/// tree is ≤15 nodes ≈ 360 bytes, so traversal stays in L1 instead of
/// pointer-chasing heap-allocated `Box`es per row.
fn flatten_hist_tree(root: &HistTreeNode) -> Vec<FlatHistNode> {
    fn push(nodes: &mut Vec<FlatHistNode>, n: &HistTreeNode) -> u32 {
        let idx = nodes.len() as u32;
        match n {
            HistTreeNode::Leaf { value } => {
                nodes.push(FlatHistNode {
                    threshold_or_value: *value,
                    feature: u32::MAX,
                    left: 0,
                    right: 0,
                });
            }
            HistTreeNode::Split {
                feature_index,
                threshold_value,
                left,
                right,
                ..
            } => {
                nodes.push(FlatHistNode {
                    threshold_or_value: *threshold_value,
                    feature: *feature_index as u32,
                    left: 0,
                    right: 0,
                });
                let l = push(nodes, left);
                let r = push(nodes, right);
                nodes[idx as usize].left = l;
                nodes[idx as usize].right = r;
            }
        }
        idx
    }
    let mut nodes = Vec::new();
    push(&mut nodes, root);
    nodes
}

#[inline]
fn predict_row_flat(nodes: &[FlatHistNode], row: &[f64]) -> f64 {
    let mut idx = 0usize;
    loop {
        let n = nodes[idx];
        if n.feature == u32::MAX {
            return n.threshold_or_value;
        }
        idx = if row[n.feature as usize] < n.threshold_or_value {
            n.left as usize
        } else {
            n.right as usize
        };
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedHistogramTree {
    pub root: HistTreeNode,
    pub n_features: usize,
    /// Flattened copy of `root` for fast prediction. Built lazily so models
    /// serialized before this field existed still deserialize and predict.
    #[serde(skip)]
    flat: std::sync::OnceLock<Vec<FlatHistNode>>,
}

impl TrainedHistogramTree {
    pub fn new(root: HistTreeNode, n_features: usize) -> Self {
        TrainedHistogramTree {
            root,
            n_features,
            flat: std::sync::OnceLock::new(),
        }
    }
}

impl TrainedBaseLearner for TrainedHistogramTree {
    #[inline]
    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let nodes = self.flat.get_or_init(|| flatten_hist_tree(&self.root));
        let p = x.ncols();
        if let Some(xs) = x.as_slice() {
            // Row-major contiguous: traverse each row as a plain slice
            Array1::from_iter(xs.chunks_exact(p).map(|row| predict_row_flat(nodes, row)))
        } else {
            let mut predictions = Array1::zeros(x.nrows());
            for i in 0..x.nrows() {
                predictions[i] = self.root.predict_row(x, i);
            }
            predictions
        }
    }

    fn predict_rows(&self, xs: &[f64], n_cols: usize, col_idxs: Option<&[usize]>, out: &mut [f64]) {
        let nodes = self.flat.get_or_init(|| flatten_hist_tree(&self.root));
        match col_idxs {
            None => {
                for (o, row) in out.iter_mut().zip(xs.chunks_exact(n_cols)) {
                    *o = predict_row_flat(nodes, row);
                }
            }
            Some(idxs) => {
                for (o, row) in out.iter_mut().zip(xs.chunks_exact(n_cols)) {
                    // Walk with the feature index remapped through col_idxs
                    let mut idx = 0usize;
                    *o = loop {
                        let n = nodes[idx];
                        if n.feature == u32::MAX {
                            break n.threshold_or_value;
                        }
                        idx = if row[idxs[n.feature as usize]] < n.threshold_or_value {
                            n.left as usize
                        } else {
                            n.right as usize
                        };
                    };
                }
            }
        }
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
        let mut pairs = Vec::new();
        self.root.collect_split_importances(&mut pairs);
        importances_from_splits(&pairs, self.n_features)
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::HistogramTree(self.clone()))
    }
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

        // Presort each feature once at the root; child nodes inherit sorted
        // order via stable partition instead of re-sorting (sklearn-style).
        let sorted_features = build_presorted_features(x, y);
        let mut goes_left = vec![false; x.nrows()];
        let root = build_tree_node_presorted(
            sample_weight,
            sorted_features,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            &mut goes_left,
        );

        Ok(Box::new(TrainedDecisionTree {
            root,
            n_features: x.ncols(),
        }))
    }
}

/// One row's entry in a feature's presorted list.
///
/// The response `y` is stored inline so split scans stream this array
/// sequentially instead of gathering `y[row]` from random memory — the gather
/// is a full cache-line fetch per element once y falls out of cache, and it
/// blocks vectorization of the scan. The feature value is f32 (sklearn
/// quantization, halves value bytes); `row` is kept for partitioning and
/// sample-weight lookups. 16 bytes, 8-aligned.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct SortedEntry {
    pub y: f64,
    pub val: f32,
    pub row: u32,
}

/// Build per-feature entry lists sorted ascending by feature value.
/// Sorting happens once here; `build_tree_node_presorted` partitions the
/// lists down the tree, preserving sorted order, so nodes never re-sort.
fn build_presorted_features(x: &Array2<f64>, y: &Array1<f64>) -> Vec<Vec<SortedEntry>> {
    let n = x.nrows();
    let n_features = x.ncols();

    let sort_one = |f: usize| {
        let col = x.column(f);
        let mut entries: Vec<SortedEntry> = (0..n)
            .map(|i| SortedEntry {
                y: y[i],
                val: col[i] as f32,
                row: i as u32,
            })
            .collect();
        entries
            .sort_unstable_by(|a, b| a.val.partial_cmp(&b.val).unwrap_or(std::cmp::Ordering::Equal));
        entries
    };

    // Parallel root sort for large workloads (sorting dominates tree fitting);
    // small workloads stay sequential to avoid rayon task overhead.
    if n_features >= 4 && n >= 2000 {
        use rayon::prelude::*;
        (0..n_features).into_par_iter().map(sort_one).collect()
    } else {
        (0..n_features).map(sort_one).collect()
    }
}

/// Recursively build a tree from presorted per-feature pair lists.
/// Equivalent to `build_tree_node` but O(n·p) per split instead of
/// O(n·log n·p): the sorted order is inherited by stable partition.
/// `goes_left` is a scratch buffer indexed by row id; each node writes the
/// rows it owns before reading them, so a single shared buffer is safe.
#[allow(clippy::too_many_arguments)]
fn build_tree_node_presorted(
    sample_weight: Option<&Array1<f64>>,
    sorted_features: Vec<Vec<SortedEntry>>,
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    goes_left: &mut Vec<bool>,
) -> TreeNode {
    let n = sorted_features[0].len();

    // Parent stats over this node's rows (any feature list holds the same rows)
    let (parent_sum, parent_weight) = if let Some(weights) = sample_weight {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for e in &sorted_features[0] {
            let w = weights[e.row as usize];
            sum += e.y * w;
            weight += w;
        }
        (sum, weight)
    } else {
        let sum: f64 = sorted_features[0].iter().map(|e| e.y).sum();
        (sum, n as f64)
    };

    let node_value = if parent_weight > 0.0 {
        parent_sum / parent_weight
    } else {
        0.0
    };

    // Check stopping conditions
    if depth >= max_depth || n < min_samples_split || parent_weight == 0.0 {
        return TreeNode::Leaf { value: node_value };
    }

    // Scan every feature's presorted pairs for the best Friedman split.
    // Parallelized across features only for large nodes (see gate constants).
    let (best_feature, best_threshold, best_improvement) = find_best_presorted_split(
        &sorted_features,
        sample_weight,
        min_samples_leaf,
        parent_sum,
        parent_weight,
    );

    // If no valid split found, return a leaf
    if best_improvement <= 0.0 || best_threshold.is_nan() {
        return TreeNode::Leaf { value: node_value };
    }

    // Mark which rows go left using the winning feature's own values
    // (same comparison as `x[[i, best_feature]] < best_threshold`)
    let mut n_left = 0usize;
    for e in &sorted_features[best_feature] {
        let left = (e.val as f64) < best_threshold;
        goes_left[e.row as usize] = left;
        n_left += left as usize;
    }
    let n_right = n - n_left;

    // Check min_samples_leaf constraint
    if n_left < min_samples_leaf || n_right < min_samples_leaf {
        return TreeNode::Leaf { value: node_value };
    }

    // Stable-partition every feature's sorted pairs into left/right children
    let n_features = sorted_features.len();
    let mut left_sorted = Vec::with_capacity(n_features);
    let mut right_sorted = Vec::with_capacity(n_features);
    for entries in sorted_features {
        let mut left_entries = Vec::with_capacity(n_left);
        let mut right_entries = Vec::with_capacity(n_right);
        for e in entries {
            if goes_left[e.row as usize] {
                left_entries.push(e);
            } else {
                right_entries.push(e);
            }
        }
        left_sorted.push(left_entries);
        right_sorted.push(right_entries);
    }

    // Recursively build children
    let left_child = build_tree_node_presorted(
        sample_weight,
        left_sorted,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        goes_left,
    );
    let right_child = build_tree_node_presorted(
        sample_weight,
        right_sorted,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        goes_left,
    );

    TreeNode::Split {
        feature_index: best_feature,
        threshold: best_threshold,
        improvement: best_improvement,
        left: Box::new(left_child),
        right: Box::new(right_child),
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
    /// Friedman-MSE improvement for split nodes (sklearn-style importances);
    /// `serde(skip)` keeps the bincode wire format unchanged.
    #[serde(skip)]
    pub improvement: f64,
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
            improvement: 0.0,
        }
    }

    #[inline]
    fn new_split(
        feature_index: usize,
        threshold: f64,
        improvement: f64,
        left: u32,
        right: u32,
    ) -> Self {
        ArenaTreeNode {
            feature_index: feature_index as u16,
            left,
            right,
            threshold_or_value: threshold,
            improvement,
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
        /// Friedman-MSE improvement of this split, for sklearn-style feature
        /// importances. `serde(skip)` keeps the bincode wire format unchanged
        /// (old saved models load; their importances fall back to counts).
        #[serde(skip)]
        improvement: f64,
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
                ..
            } => {
                // f32-quantized comparison, matching the fit-time partition
                // (and sklearn's float32 predict-time cast); see
                // TrainedStumpLearner::predict.
                if ((x[[row_idx, *feature_index]] as f32) as f64) < *threshold {
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

    fn collect_split_importances(&self, out: &mut Vec<(usize, f64)>) {
        match self {
            TreeNode::Leaf { .. } => {}
            TreeNode::Split {
                feature_index,
                improvement,
                left,
                right,
                ..
            } => {
                out.push((*feature_index, *improvement));
                left.collect_split_importances(out);
                right.collect_split_importances(out);
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
                improvement,
                left,
                right,
            } => {
                // Reserve space for this node
                nodes.push(ArenaTreeNode::new_leaf(0.0)); // placeholder
                let left_idx = left.to_arena_nodes(nodes);
                let right_idx = right.to_arena_nodes(nodes);
                nodes[idx as usize] = ArenaTreeNode::new_split(
                    *feature_index,
                    *threshold,
                    *improvement,
                    left_idx,
                    right_idx,
                );
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
            // f32-quantized comparison, matching the fit-time partition (and
            // sklearn's float32 predict-time cast); see TrainedStumpLearner.
            let feature_val = (x[[row_idx, node.feature_index as usize]] as f32) as f64;
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
        let pairs: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .filter(|n| !n.is_leaf())
            .map(|n| (n.feature_index as usize, n.improvement))
            .collect();
        importances_from_splits(&pairs, self.n_features)
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
        // ArenaTreeNode stores the split feature as u16 (wire-format
        // compatibility); `feature_index as u16` would silently wrap for
        // wider matrices, splitting on the wrong column.
        if x.ncols() > u16::MAX as usize + 1 {
            return Err("ArenaDecisionTreeLearner supports at most 65536 features");
        }

        let indices: Vec<usize> = (0..x.nrows()).collect();

        // Pre-allocate nodes vector based on max possible tree size
        // A complete binary tree of depth d has 2^(d+1) - 1 nodes
        let max_nodes = (1usize << (self.max_depth + 1)).saturating_sub(1);
        let mut nodes = Vec::with_capacity(max_nodes.min(indices.len() * 2));
        // Pre-allocate sort buffer once — reused across all recursive calls
        let mut sort_buf: Vec<SortedEntry> = Vec::with_capacity(x.nrows());

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
    sort_buf: &mut Vec<SortedEntry>,
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
        // Compare via the f32-cast value, matching the sorted-pair view used
        // by split scanning (and the presorted builder's partition)
        if (x[[i, best_feature]] as f32 as f64) < best_threshold {
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
    nodes[node_idx as usize] = ArenaTreeNode::new_split(
        best_feature,
        best_threshold,
        best_improvement,
        left_idx,
        right_idx,
    );

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
        let mut pairs = Vec::new();
        self.root.collect_split_importances(&mut pairs);
        importances_from_splits(&pairs, self.n_features)
    }

    fn to_serializable(&self) -> Option<SerializableTrainedLearner> {
        Some(SerializableTrainedLearner::DecisionTree(self.clone()))
    }
}

/// Build a tree node recursively
/// Per-node-sorting tree builder. Superseded by `build_tree_node_presorted`
/// for `DecisionTreeLearner`; kept as the reference implementation for the
/// presorted-equivalence tests.
#[cfg_attr(not(test), allow(dead_code))]
fn build_tree_node(
    x: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    sort_buf: &mut Vec<SortedEntry>,
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
        // Compare via the f32-cast value, matching the sorted-pair view used
        // by split scanning (and the presorted builder's partition)
        if (x[[i, best_feature]] as f32 as f64) < best_threshold {
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
        improvement: best_improvement,
        left: Box::new(left_child),
        right: Box::new(right_child),
    }
}

/// Scan entries already sorted ascending by feature value and return the best
/// (threshold, improvement) under the Friedman MSE criterion.
/// Shared by the per-node-sorting path and the presorted path.
///
/// Feature values are f32 (sklearn quantization); the response is inline in
/// each entry so the unweighted scan is a pure sequential stream with no
/// random gathers. All accumulation (left_sum, means, improvement) and the
/// returned threshold stay f64, so there is no error accumulation.
#[inline]
fn scan_sorted_entries(
    entries: &[SortedEntry],
    sample_weight: Option<&Array1<f64>>,
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
) -> (f64, f64) {
    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;
    let mut left_sum = 0.0;
    let mut left_weight = 0.0;
    let n = entries.len();

    // Split into weighted vs unweighted paths to avoid per-element branch
    if let Some(weights) = sample_weight {
        for pos in 0..n {
            let e = entries[pos];
            let w = weights[e.row as usize];
            left_sum += e.y * w;
            left_weight += w;

            let left_count = pos + 1;
            let right_count = n - left_count;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            if pos + 1 < n && (e.val as f64 - entries[pos + 1].val as f64).abs() < 1e-10 {
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
                    (e.val as f64 + entries[pos + 1].val as f64) * 0.5
                } else {
                    e.val as f64
                };
            }
        }
    } else {
        // Unweighted path — sequential stream, no per-element weight or y gather
        for pos in 0..n {
            let e = entries[pos];
            left_sum += e.y;
            left_weight += 1.0;

            let left_count = pos + 1;
            let right_count = n - left_count;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            if pos + 1 < n && (e.val as f64 - entries[pos + 1].val as f64).abs() < 1e-10 {
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
                    (e.val as f64 + entries[pos + 1].val as f64) * 0.5
                } else {
                    e.val as f64
                };
            }
        }
    }

    (best_threshold, best_improvement)
}

// ----------------------------------------------------------------------------
// Best-split search over presorted features (used by `build_tree_node_presorted`).
//
// The per-feature scans are independent, so they can run in parallel. But after
// the presort rewrite each scan is a cheap, memory-bound linear pass, so rayon's
// per-task overhead only pays off for large nodes. The gate below is a function
// of total work (node rows × features), not rows alone — the `presort_scan`
// benchmark showed the seq/par crossover tracks n×p almost exactly. With f32
// pair storage (8 bytes/pair) the measured crossover is:
//
//   n×p elements    seq       par      winner
//   ~25k            ~47µs     ~58µs    sequential (rayon overhead dominates)
//   ~50k            ~94µs     ~59-95µs tie at p=5, parallel ~1.6× at p=10
//   ~500k           ~1.03ms   ~0.40ms  parallel  (~2.6×)
//   ~1M             ~2.08ms   ~0.63ms  parallel  (~3.3×)
//
// Speedups cap around 2–3× (not core-count) because the scans are memory-
// bandwidth bound, not CPU bound. Gate set at 50k: nothing loses there and
// higher feature counts already win ~1.6×.
// `presort_par_matches_seq` guarantees both variants produce identical splits.
// ----------------------------------------------------------------------------

/// Minimum total work (node rows × features) before parallelizing the
/// per-feature split scan. Below this, rayon's per-batch overhead exceeds the
/// savings from these cheap, memory-bound scans.
const PRESORT_PAR_MIN_WORK: usize = 50_000;
/// Need at least 2 features (tasks) for parallelism to do anything.
const PRESORT_PAR_MIN_FEATURES: usize = 2;

/// Sequentially scan all presorted features and return the best
/// (feature, threshold, improvement) under the Friedman MSE criterion.
/// Ties in improvement are broken toward the lowest feature index.
#[doc(hidden)]
pub fn find_best_presorted_split_seq(
    sorted_features: &[Vec<SortedEntry>],
    sample_weight: Option<&Array1<f64>>,
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
) -> (usize, f64, f64) {
    let mut best_feature = 0;
    let mut best_threshold = f64::NAN;
    let mut best_improvement = 0.0;
    for (feature_index, entries) in sorted_features.iter().enumerate() {
        let (threshold, improvement) = scan_sorted_entries(
            entries,
            sample_weight,
            min_samples_leaf,
            parent_sum,
            parent_weight,
        );
        if improvement > best_improvement {
            best_improvement = improvement;
            best_feature = feature_index;
            best_threshold = threshold;
        }
    }
    (best_feature, best_threshold, best_improvement)
}

/// Parallel (rayon) version of [`find_best_presorted_split_seq`].
/// Each feature is scanned on its own task; results are collected in feature
/// order and reduced with the identical argmax, so the output is bit-for-bit
/// the same as the sequential version (including tie-breaking).
#[doc(hidden)]
pub fn find_best_presorted_split_par(
    sorted_features: &[Vec<SortedEntry>],
    sample_weight: Option<&Array1<f64>>,
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
) -> (usize, f64, f64) {
    use rayon::prelude::*;

    let results: Vec<(f64, f64)> = sorted_features
        .par_iter()
        .map(|entries| {
            scan_sorted_entries(
                entries,
                sample_weight,
                min_samples_leaf,
                parent_sum,
                parent_weight,
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
    (best_feature, best_threshold, best_improvement)
}

/// Gated dispatch: parallel scan only for large nodes, sequential otherwise.
#[inline]
fn find_best_presorted_split(
    sorted_features: &[Vec<SortedEntry>],
    sample_weight: Option<&Array1<f64>>,
    min_samples_leaf: usize,
    parent_sum: f64,
    parent_weight: f64,
) -> (usize, f64, f64) {
    let n_features = sorted_features.len();
    let n_rows = sorted_features.first().map_or(0, |f| f.len());
    if n_features >= PRESORT_PAR_MIN_FEATURES
        && n_rows.saturating_mul(n_features) >= PRESORT_PAR_MIN_WORK
    {
        find_best_presorted_split_par(
            sorted_features,
            sample_weight,
            min_samples_leaf,
            parent_sum,
            parent_weight,
        )
    } else {
        find_best_presorted_split_seq(
            sorted_features,
            sample_weight,
            min_samples_leaf,
            parent_sum,
            parent_weight,
        )
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
    sort_buf: &mut Vec<SortedEntry>,
) -> (f64, f64) {
    let feature_col = x.column(feature_index);

    // Build entries (y inline) — the scan then streams sequentially with no
    // indirect lookups through feature_col[idx] or y[idx]
    sort_buf.clear();
    sort_buf.extend(indices.iter().map(|&i| SortedEntry {
        y: y[i],
        val: feature_col[i] as f32,
        row: i as u32,
    }));
    sort_buf.sort_unstable_by(|a, b| a.val.partial_cmp(&b.val).unwrap_or(std::cmp::Ordering::Equal));

    scan_sorted_entries(
        sort_buf,
        sample_weight,
        min_samples_leaf,
        parent_sum,
        parent_weight,
    )
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
    sort_buf: &mut Vec<SortedEntry>,
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
                let mut local_sort_buf: Vec<SortedEntry> = Vec::with_capacity(indices.len());
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
/// DecisionTreeRegressor with max_depth=3 and criterion="friedman_mse".
/// This is the *exact* split finder — bit-equivalent to sklearn's trees.
pub fn default_tree_learner() -> DecisionTreeLearner {
    DecisionTreeLearner::default_sklearn()
}

/// Returns the default base learner used by the high-level wrappers
/// (`NGBRegressor`, `NGBClassifier`, `NGBMultiClassifier`): a depth-3
/// histogram tree (255 bins).
///
/// Chosen over the exact tree because it is 2.5–2.7× faster end-to-end at
/// n ≥ 2000 (binning is cached across boosting iterations) with held-out
/// accuracy at parity or better across regression, classification, survival,
/// outlier, discrete-feature, and high-dimensional benchmarks
/// (see tests/accuracy_parity.rs). For exact sklearn-equivalent trees use
/// [`default_tree_learner`] with the generic `NGBoost` type.
pub fn default_base_learner() -> HistogramLearner {
    HistogramLearner::default_histogram()
}

/// Returns a stump learner (depth-1 tree) for simpler/faster models
pub fn stump_learner() -> StumpLearner {
    StumpLearner
}

#[cfg(test)]
mod presort_tests {
    use super::*;

    /// Deterministic pseudo-random data (no external rand dependency needed)
    fn synth(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        let x = Array2::from_shape_fn((n, p), |_| next());
        let y = Array1::from_shape_fn(n, |i| x[[i, 0]] * 3.0 - x[[i, p - 1]] + next() * 0.1);
        (x, y)
    }

    fn build_reference(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: Option<&Array1<f64>>,
        max_depth: usize,
    ) -> TreeNode {
        let indices: Vec<usize> = (0..x.nrows()).collect();
        let mut sort_buf: Vec<SortedEntry> = Vec::with_capacity(x.nrows());
        build_tree_node(x, y, w, &indices, 0, max_depth, 2, 1, &mut sort_buf)
    }

    fn build_presorted(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: Option<&Array1<f64>>,
        max_depth: usize,
    ) -> TreeNode {
        let sorted = build_presorted_features(x, y);
        let mut goes_left = vec![false; x.nrows()];
        build_tree_node_presorted(w, sorted, 0, max_depth, 2, 1, &mut goes_left)
    }

    fn assert_trees_equal(a: &TreeNode, b: &TreeNode) {
        match (a, b) {
            (TreeNode::Leaf { value: va }, TreeNode::Leaf { value: vb }) => {
                assert!(
                    (va - vb).abs() <= 1e-12 * va.abs().max(1.0),
                    "leaf mismatch: {va} vs {vb}"
                );
            }
            (
                TreeNode::Split {
                    feature_index: fa,
                    threshold: ta,
                    improvement: ia,
                    left: la,
                    right: ra,
                },
                TreeNode::Split {
                    feature_index: fb,
                    threshold: tb,
                    improvement: ib,
                    left: lb,
                    right: rb,
                },
            ) => {
                assert_eq!(fa, fb, "split feature mismatch");
                assert!((ta - tb).abs() <= 1e-12, "threshold mismatch: {ta} vs {tb}");
                assert!(
                    (ia - ib).abs() <= 1e-12 * ia.abs().max(1.0),
                    "improvement mismatch: {ia} vs {ib}"
                );
                assert_trees_equal(la, lb);
                assert_trees_equal(ra, rb);
            }
            _ => panic!("tree structure mismatch (leaf vs split)"),
        }
    }

    /// Brute-force reference for sklearn-style importances: walk the tree
    /// re-partitioning the training rows and accumulate each split's weighted
    /// variance decrease N_t·Var_t − N_l·Var_l − N_r·Var_r per feature.
    fn variance_decrease_walk(
        x: &Array2<f64>,
        y: &Array1<f64>,
        node: &TreeNode,
        indices: &[usize],
        importances: &mut Array1<f64>,
    ) {
        if let TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
            ..
        } = node
        {
            let (mut ls, mut lss, mut ln) = (0.0, 0.0, 0.0);
            let (mut rs, mut rss, mut rn) = (0.0, 0.0, 0.0);
            let mut li = Vec::new();
            let mut ri = Vec::new();
            for &i in indices {
                // Same f32-quantized comparison as the fit-time partition
                if (x[[i, *feature_index]] as f32 as f64) < *threshold {
                    ls += y[i];
                    lss += y[i] * y[i];
                    ln += 1.0;
                    li.push(i);
                } else {
                    rs += y[i];
                    rss += y[i] * y[i];
                    rn += 1.0;
                    ri.push(i);
                }
            }
            let (n, ts, tss) = (ln + rn, ls + rs, lss + rss);
            let nvar_parent = tss - ts * ts / n;
            let nvar_left = if ln > 0.0 { lss - ls * ls / ln } else { 0.0 };
            let nvar_right = if rn > 0.0 { rss - rs * rs / rn } else { 0.0 };
            importances[*feature_index] += nvar_parent - nvar_left - nvar_right;
            variance_decrease_walk(x, y, left, &li, importances);
            variance_decrease_walk(x, y, right, &ri, importances);
        }
    }

    /// The stored Friedman improvements must equal the per-split variance
    /// decrease (the identity sklearn's importances rely on), so
    /// feature_importances() must match the brute-force reference.
    #[test]
    fn importances_match_variance_decrease_reference() {
        let (x, y) = synth(500, 5, 77);
        let tree = TrainedDecisionTree {
            root: build_presorted(&x, &y, None, 3),
            n_features: x.ncols(),
        };

        let mut reference = Array1::zeros(x.ncols());
        let indices: Vec<usize> = (0..x.nrows()).collect();
        variance_decrease_walk(&x, &y, &tree.root, &indices, &mut reference);
        let ref_sum = reference.sum();
        assert!(ref_sum > 0.0, "test data must produce splits");
        reference.mapv_inplace(|v| v / ref_sum);

        let importances = tree.feature_importances().expect("tree has splits");
        for f in 0..x.ncols() {
            assert!(
                (importances[f] - reference[f]).abs() <= 1e-9,
                "feature {f}: stored-improvement importance {} vs variance-decrease reference {}",
                importances[f],
                reference[f]
            );
        }
        // Sanity: the dominant signal (3·x0) must dominate the importances
        let max_f = (0..x.ncols()).max_by(|&a, &b| importances[a].total_cmp(&importances[b]));
        assert_eq!(max_f, Some(0), "x0 carries the strongest signal");
    }

    /// Models saved before improvements were recorded deserialize with zeroed
    /// improvements (serde(skip)); importances must fall back to split counts
    /// rather than returning all-zeros.
    #[test]
    fn importances_fall_back_to_counts_for_legacy_trees() {
        let (x, y) = synth(300, 3, 9);
        let mut tree = TrainedDecisionTree {
            root: build_presorted(&x, &y, None, 2),
            n_features: x.ncols(),
        };
        // Simulate a legacy deserialized tree: zero out stored improvements
        fn zero_improvements(node: &mut TreeNode) {
            if let TreeNode::Split {
                improvement,
                left,
                right,
                ..
            } = node
            {
                *improvement = 0.0;
                zero_improvements(left);
                zero_improvements(right);
            }
        }
        zero_improvements(&mut tree.root);

        let imp = tree.feature_importances().expect("tree has splits");
        let sum: f64 = imp.sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "count-fallback importances must normalize to 1, got {sum}"
        );
    }

    /// A raw f64 exactly at the f64 midpoint of two adjacent f32 values casts
    /// DOWN across the threshold (ties-to-even): the fit-time partition (f32
    /// compare) put it in the LEFT leaf, so predict must route it left too.
    /// Before the predict-time f32 cast, predict compared the raw f64 and
    /// routed it right — returning a leaf the row never contributed to.
    #[test]
    fn exact_tree_fit_predict_f32_consistent() {
        let eps24 = (2.0f64).powi(-24); // half of the f32 ULP at 1.0
        let adversarial = 1.0 + eps24; // f32 cast = 1.0 (ties-to-even)
        let x = Array2::from_shape_vec(
            (3, 1),
            vec![1.0, adversarial, 1.0 + 2.0 * eps24],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 10.0]);
        // Split lands between f32 groups {1.0, 1.0} and {1 + 2^-23}:
        // threshold = 1 + 2^-24 == adversarial exactly.

        let tree = DecisionTreeLearner::new(1).fit(&x, &y).unwrap();
        let p = tree.predict(&x);
        assert_eq!(p[1], 0.0, "depth-1 tree routed the tie value to the wrong leaf");
        assert_eq!(p[0], 0.0);
        assert_eq!(p[2], 10.0);

        let stump = StumpLearner.fit(&x, &y).unwrap();
        let p = stump.predict(&x);
        assert_eq!(p[1], 0.0, "stump routed the tie value to the wrong leaf");

        let arena = ArenaDecisionTreeLearner::new(1).fit(&x, &y).unwrap();
        let p = arena.predict(&x);
        assert_eq!(p[1], 0.0, "arena tree routed the tie value to the wrong leaf");
    }

    /// A stump is a depth-1 tree: both must pick the same split and produce
    /// the same leaf means.
    #[test]
    fn stump_matches_depth1_tree() {
        for seed in [3u64, 41, 99] {
            let (x, y) = synth(300, 4, seed);
            let stump = StumpLearner.fit(&x, &y).unwrap();
            let tree = DecisionTreeLearner::new(1).fit(&x, &y).unwrap();

            assert_eq!(stump.split_feature(), tree.split_features().map(|f| f[0]));
            let ps = stump.predict(&x);
            let pt = tree.predict(&x);
            for i in 0..x.nrows() {
                assert!(
                    (ps[i] - pt[i]).abs() <= 1e-12 * pt[i].abs().max(1.0),
                    "seed {seed} row {i}: stump {} vs depth-1 tree {}",
                    ps[i],
                    pt[i]
                );
            }
        }
    }

    #[test]
    fn presorted_matches_reference_unweighted() {
        for (n, p, depth) in [(50, 3, 3), (500, 7, 3), (1200, 5, 4), (30, 2, 6)] {
            let (x, y) = synth(n, p, 42 + n as u64);
            let reference = build_reference(&x, &y, None, depth);
            let presorted = build_presorted(&x, &y, None, depth);
            assert_trees_equal(&reference, &presorted);
        }
    }

    #[test]
    fn presorted_matches_reference_weighted() {
        for (n, p, depth) in [(80, 4, 3), (600, 6, 4)] {
            let (x, y) = synth(n, p, 7 + n as u64);
            let mut s = 99u64;
            let w = Array1::from_shape_fn(n, |_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                0.1 + ((s >> 33) as f64) / (u32::MAX as f64)
            });
            let reference = build_reference(&x, &y, Some(&w), depth);
            let presorted = build_presorted(&x, &y, Some(&w), depth);
            assert_trees_equal(&reference, &presorted);
        }
    }

    #[test]
    fn presorted_handles_tied_feature_values() {
        // Heavy ties (quantized features): split decisions must agree; leaf
        // values may differ in the last ulp due to summation order within
        // tie runs, so compare predictions loosely.
        let (mut x, y) = synth(400, 4, 1234);
        x.mapv_inplace(|v| (v * 8.0).floor() / 8.0);

        let reference = build_reference(&x, &y, None, 3);
        let presorted = build_presorted(&x, &y, None, 3);

        let ref_tree = TrainedDecisionTree {
            root: reference,
            n_features: x.ncols(),
        };
        let new_tree = TrainedDecisionTree {
            root: presorted,
            n_features: x.ncols(),
        };
        let pa = ref_tree.predict(&x);
        let pb = new_tree.predict(&x);
        for i in 0..x.nrows() {
            assert!(
                (pa[i] - pb[i]).abs() <= 1e-9 * pa[i].abs().max(1.0),
                "prediction mismatch at {i}: {} vs {}",
                pa[i],
                pb[i]
            );
        }
    }

    #[test]
    fn presorted_constant_feature_yields_leaf() {
        // All-constant X → no valid split → single leaf with the mean
        let x = Array2::from_elem((20, 3), 1.0);
        let y = Array1::from_shape_fn(20, |i| i as f64);
        let tree = build_presorted(&x, &y, None, 3);
        match tree {
            TreeNode::Leaf { value } => assert!((value - 9.5).abs() < 1e-12),
            _ => panic!("expected a single leaf for constant features"),
        }
    }

    /// Build sorted feature lists + parent stats for a synthetic root node.
    fn presorted_root(
        x: &Array2<f64>,
        y: &Array1<f64>,
        w: Option<&Array1<f64>>,
    ) -> (Vec<Vec<SortedEntry>>, f64, f64) {
        let sorted = build_presorted_features(x, y);
        let (parent_sum, parent_weight) = match w {
            Some(weights) => {
                let mut s = 0.0;
                let mut wt = 0.0;
                for i in 0..y.len() {
                    s += y[i] * weights[i];
                    wt += weights[i];
                }
                (s, wt)
            }
            None => (y.sum(), y.len() as f64),
        };
        (sorted, parent_sum, parent_weight)
    }

    /// The parallel split scan must return exactly the same (feature, threshold,
    /// improvement) as the sequential one — including tie-breaking — across a
    /// range of sizes and feature counts, weighted and unweighted.
    #[test]
    fn presorted_par_matches_seq() {
        for (n, p) in [(500usize, 5usize), (5000, 10), (20000, 4), (3000, 30)] {
            let (x, y) = synth(n, p, 2025 + (n + p) as u64);

            // Unweighted
            let (sorted, ps, pw) = presorted_root(&x, &y, None);
            let seq = find_best_presorted_split_seq(&sorted, None, 1, ps, pw);
            let par = find_best_presorted_split_par(&sorted, None, 1, ps, pw);
            assert_eq!(seq.0, par.0, "feature mismatch (n={n}, p={p})");
            assert_eq!(
                seq.1.to_bits(),
                par.1.to_bits(),
                "threshold bits mismatch (n={n}, p={p})"
            );
            assert_eq!(
                seq.2.to_bits(),
                par.2.to_bits(),
                "improvement bits mismatch (n={n}, p={p})"
            );

            // Weighted
            let mut s = 13u64;
            let w = Array1::from_shape_fn(n, |_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                0.1 + ((s >> 33) as f64) / (u32::MAX as f64)
            });
            let (sorted, ps, pw) = presorted_root(&x, &y, Some(&w));
            let seq = find_best_presorted_split_seq(&sorted, Some(&w), 1, ps, pw);
            let par = find_best_presorted_split_par(&sorted, Some(&w), 1, ps, pw);
            assert_eq!(seq.0, par.0, "weighted feature mismatch (n={n}, p={p})");
            assert_eq!(seq.1.to_bits(), par.1.to_bits(), "weighted threshold mismatch");
            assert_eq!(seq.2.to_bits(), par.2.to_bits(), "weighted improvement mismatch");
        }
    }

    /// A node large enough to cross the parallel gate must build a tree
    /// identical to the per-node-sorting reference (exercises the par path).
    #[test]
    fn presorted_large_node_matches_reference() {
        // n×p = 60_000×3 = 180k ≥ PRESORT_PAR_MIN_WORK, so the root + shallow
        // nodes take the parallel scan path.
        let (x, y) = synth(60_000, 3, 777);
        let reference = build_reference(&x, &y, None, 3);
        let presorted = build_presorted(&x, &y, None, 3);
        assert_trees_equal(&reference, &presorted);
    }
}

#[cfg(test)]
mod histogram_tests {
    use super::*;

    fn synth(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        let x = Array2::from_shape_fn((n, p), |_| next());
        let y = Array1::from_shape_fn(n, |i| x[[i, 0]] * 3.0 - x[[i, p - 1]] + next() * 0.1);
        (x, y)
    }

    /// Leaf-scattered training predictions must be BIT-identical to
    /// `fitted.predict(x)` — the invariant `fit_predict_cached` relies on.
    /// Covers the partition/predict routing edge cases: values exactly on
    /// bin edges (integer features), NaN rows (both route right), weighted
    /// fits, and min_samples_leaf-forced leaf fallbacks.
    #[test]
    fn leaf_scatter_matches_predict() {
        // Continuous, unweighted
        let (x, y) = synth(1500, 6, 7);
        // Integer-valued feature column → every value sits exactly on a bin
        // edge, exercising the x == edges[bin] tie (routes right both ways)
        let mut x = x;
        for i in 0..x.nrows() {
            x[[i, 2]] = (x[[i, 2]] * 8.0).floor();
        }
        // A NaN row: rejected by NGBoost::fit but legal via the learner API
        x[[17, 4]] = f64::NAN;

        let learner = HistogramLearner::new(4);
        let cache = learner.build_fit_cache(&x).unwrap();

        for weights in [
            None,
            Some(Array1::from_shape_fn(x.nrows(), |i| {
                0.5 + ((i % 7) as f64) * 0.3
            })),
        ] {
            let (fitted, scattered) = learner
                .fit_predict_cached(&x, &y, weights.as_ref(), Some(cache.as_ref()))
                .unwrap();
            let walked = fitted.predict(&x);
            for i in 0..x.nrows() {
                assert_eq!(
                    scattered[i].to_bits(),
                    walked[i].to_bits(),
                    "scatter/predict mismatch at row {i} (weighted: {})",
                    weights.is_some()
                );
            }
        }

        // min_samples_leaf large enough to force no-split leaf fallbacks
        let learner = HistogramLearner {
            max_depth: 4,
            max_bins: 255,
            min_samples_leaf: 400,
            min_samples_split: 2,
        };
        let (fitted, scattered) = learner.fit_predict_cached(&x, &y, None, None).unwrap();
        let walked = fitted.predict(&x);
        for i in 0..x.nrows() {
            assert_eq!(scattered[i].to_bits(), walked[i].to_bits());
        }
    }

    /// Row-subset fitting against the full-X cache: with rows = ALL rows it
    /// must reproduce the full cached fit bit-for-bit, and with a proper
    /// subset its returned predictions must be bit-identical to walking the
    /// fitted tree over the materialized subset (in rows order).
    #[test]
    fn subset_fit_matches_full_and_predict() {
        let (x, y) = synth(1200, 6, 21);
        let learner = HistogramLearner::new(3);
        let cache = learner.build_fit_cache(&x).unwrap();

        // All rows == full fit
        let all_rows: Vec<u32> = (0..x.nrows() as u32).collect();
        let (_, p_full) = learner
            .fit_predict_cached(&x, &y, None, Some(cache.as_ref()))
            .unwrap();
        let (_, p_rows) = learner
            .fit_predict_cached_rows(&x, &y, None, cache.as_ref(), &all_rows)
            .unwrap();
        for i in 0..x.nrows() {
            assert_eq!(p_full[i].to_bits(), p_rows[i].to_bits());
        }

        // Proper subset (shuffled-ish order, includes duplicated coverage of
        // the row-id space via stride tricks): preds must equal a tree walk
        // over the materialized subset, in rows order
        let rows: Vec<u32> = (0..x.nrows() as u32).filter(|r| r % 3 != 0).collect();
        for weights in [
            None,
            Some(Array1::from_shape_fn(x.nrows(), |i| {
                0.25 + ((i % 5) as f64) * 0.5
            })),
        ] {
            let (fitted, preds) = learner
                .fit_predict_cached_rows(&x, &y, weights.as_ref(), cache.as_ref(), &rows)
                .unwrap();
            let idx: Vec<usize> = rows.iter().map(|&r| r as usize).collect();
            let x_sub = x.select(ndarray::Axis(0), &idx);
            let walked = fitted.predict(&x_sub);
            assert_eq!(preds.len(), rows.len());
            for k in 0..rows.len() {
                assert_eq!(
                    preds[k].to_bits(),
                    walked[k].to_bits(),
                    "subset pred mismatch at k={k} (weighted: {})",
                    weights.is_some()
                );
            }
        }
    }

    /// fit_with_weights (which builds its own cache) and fit_with_weights_cached
    /// (with an externally built cache on the same X) must produce identical trees.
    #[test]
    fn cached_fit_matches_uncached() {
        for (n, p) in [(300usize, 4usize), (2000, 8)] {
            let (x, y) = synth(n, p, 99 + n as u64);
            let learner = HistogramLearner::new(3);

            let uncached = learner.fit_with_weights(&x, &y, None).unwrap();
            let cache = learner.build_fit_cache(&x).unwrap();
            let cached = learner
                .fit_with_weights_cached(&x, &y, None, cache.as_ref())
                .unwrap();

            let pu = uncached.predict(&x);
            let pc = cached.predict(&x);
            for i in 0..n {
                assert_eq!(
                    pu[i].to_bits(),
                    pc[i].to_bits(),
                    "prediction mismatch at row {i} (n={n}, p={p})"
                );
            }
        }
    }

    /// Weighted fits must also agree between cached and uncached paths.
    #[test]
    fn cached_fit_matches_uncached_weighted() {
        let (x, y) = synth(800, 5, 7);
        let mut s = 5u64;
        let w = Array1::from_shape_fn(800, |_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.1 + ((s >> 33) as f64) / (u32::MAX as f64)
        });
        let learner = HistogramLearner::new(3);

        let uncached = learner.fit_with_weights(&x, &y, Some(&w)).unwrap();
        let cache = learner.build_fit_cache(&x).unwrap();
        let cached = learner
            .fit_with_weights_cached(&x, &y, Some(&w), cache.as_ref())
            .unwrap();

        let pu = uncached.predict(&x);
        let pc = cached.predict(&x);
        for i in 0..800 {
            assert_eq!(pu[i].to_bits(), pc[i].to_bits(), "row {i}");
        }
    }

    /// A stale cache (built on different X dims) must fall back gracefully
    /// rather than producing garbage.
    #[test]
    fn stale_cache_falls_back() {
        let (x_small, _) = synth(100, 3, 1);
        let (x, y) = synth(400, 3, 2);
        let learner = HistogramLearner::new(3);

        let stale = learner.build_fit_cache(&x_small).unwrap();
        let fitted = learner
            .fit_with_weights_cached(&x, &y, None, stale.as_ref())
            .unwrap();
        let direct = learner.fit_with_weights(&x, &y, None).unwrap();

        let pf = fitted.predict(&x);
        let pd = direct.predict(&x);
        for i in 0..400 {
            assert_eq!(pf[i].to_bits(), pd[i].to_bits(), "row {i}");
        }
    }

    /// The histogram learner must still split sensibly: tree predictions must
    /// correlate strongly with a linear signal.
    #[test]
    fn histogram_tree_learns_signal() {
        let (x, y) = synth(1500, 4, 33);
        let learner = HistogramLearner::new(3);
        let fitted = learner.fit_with_weights(&x, &y, None).unwrap();
        let preds = fitted.predict(&x);

        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(preds.iter())
            .map(|(v, p)| (v - p).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.7, "histogram tree failed to learn: R²={r2}");
    }

    /// Histogram-tree importances must also equal the brute-force variance
    /// decrease of each split's actual partition (training rows partition by
    /// `value < threshold_value` exactly as the binned fit does).
    #[test]
    fn hist_importances_match_variance_decrease_reference() {
        fn walk(
            x: &Array2<f64>,
            y: &Array1<f64>,
            node: &HistTreeNode,
            indices: &[usize],
            importances: &mut Array1<f64>,
        ) {
            if let HistTreeNode::Split {
                feature_index,
                threshold_value,
                left,
                right,
                ..
            } = node
            {
                let (mut ls, mut lss, mut ln) = (0.0, 0.0, 0.0);
                let (mut rs, mut rss, mut rn) = (0.0, 0.0, 0.0);
                let mut li = Vec::new();
                let mut ri = Vec::new();
                for &i in indices {
                    if x[[i, *feature_index]] < *threshold_value {
                        ls += y[i];
                        lss += y[i] * y[i];
                        ln += 1.0;
                        li.push(i);
                    } else {
                        rs += y[i];
                        rss += y[i] * y[i];
                        rn += 1.0;
                        ri.push(i);
                    }
                }
                let (n, ts, tss) = (ln + rn, ls + rs, lss + rss);
                let nvar_parent = tss - ts * ts / n;
                let nvar_left = if ln > 0.0 { lss - ls * ls / ln } else { 0.0 };
                let nvar_right = if rn > 0.0 { rss - rs * rs / rn } else { 0.0 };
                importances[*feature_index] += nvar_parent - nvar_left - nvar_right;
                walk(x, y, left, &li, importances);
                walk(x, y, right, &ri, importances);
            }
        }

        let (x, y) = synth(600, 4, 55);
        let cache = build_histogram_cache(&x, 255);
        let indices: Vec<u32> = (0..x.nrows() as u32).collect();
        let hists = FeatureHists::build(&cache, &indices, &y, None);
        let root = build_histogram_tree_node(&cache, &y, None, indices, hists, 0, 3, 2, 1, None);
        let tree = TrainedHistogramTree::new(root, x.ncols());

        let mut reference = Array1::zeros(x.ncols());
        let all: Vec<usize> = (0..x.nrows()).collect();
        walk(&x, &y, &tree.root, &all, &mut reference);
        let ref_sum = reference.sum();
        assert!(ref_sum > 0.0, "test data must produce splits");
        reference.mapv_inplace(|v| v / ref_sum);

        let importances = tree.feature_importances().expect("tree has splits");
        for f in 0..x.ncols() {
            assert!(
                (importances[f] - reference[f]).abs() <= 1e-9,
                "feature {f}: {} vs reference {}",
                importances[f],
                reference[f]
            );
        }
    }

    /// min_samples_leaf is a raw-count constraint (sklearn semantics). With
    /// normalized weights (summing to 1) every weighted bin count is < 1, and
    /// the old weighted-count check rejected every split, silently collapsing
    /// the tree to a single root leaf.
    #[test]
    fn small_normalized_weights_still_split() {
        let (x, y) = synth(600, 4, 21);
        let n = x.nrows();
        let w = Array1::from_elem(n, 1.0 / n as f64);
        let learner = HistogramLearner::new(3);

        let weighted = learner.fit_with_weights(&x, &y, Some(&w)).unwrap();
        let unweighted = learner.fit_with_weights(&x, &y, None).unwrap();

        // Uniform weights must not change the tree structure: the Friedman
        // improvement just scales by the weight, leaving the argmax intact.
        assert_eq!(
            weighted.split_features(),
            unweighted.split_features(),
            "uniform 1/n weights changed the tree structure"
        );

        let pw = weighted.predict(&x);
        let pu = unweighted.predict(&x);
        for i in 0..n {
            assert!(
                (pw[i] - pu[i]).abs() <= 1e-9 * pu[i].abs().max(1.0),
                "row {i}: weighted {} vs unweighted {}",
                pw[i],
                pu[i]
            );
        }
    }

    /// Direct learner use with NaN features must not panic, and NaN must route
    /// exactly like +inf both at fit time (last bin) and predict time
    /// (`value < threshold` is false → right child).
    #[test]
    fn nan_rows_route_like_positive_infinity() {
        let (mut x, y) = synth(400, 3, 5);
        x[[7, 1]] = f64::NAN;
        x[[123, 0]] = f64::NAN;
        let learner = HistogramLearner::new(3);
        let fitted = learner.fit_with_weights(&x, &y, None).unwrap();

        let x_nan = Array2::from_elem((1, 3), f64::NAN);
        let x_inf = Array2::from_elem((1, 3), f64::INFINITY);
        assert_eq!(
            fitted.predict(&x_nan)[0].to_bits(),
            fitted.predict(&x_inf)[0].to_bits(),
            "NaN must follow the same path as +inf"
        );
    }

    /// Bin indices are stored as u8, so max_bins silently caps at 255; a
    /// larger request must behave exactly like 255, not corrupt the binning.
    #[test]
    fn max_bins_above_u8_capacity_is_clamped() {
        let (x, y) = synth(2000, 3, 13);
        let big = HistogramLearner::with_params(3, 4096, 1, 2);
        let capped = HistogramLearner::with_params(3, 255, 1, 2);

        let pb = big.fit_with_weights(&x, &y, None).unwrap().predict(&x);
        let pc = capped.fit_with_weights(&x, &y, None).unwrap().predict(&x);
        for i in 0..x.nrows() {
            assert_eq!(pb[i].to_bits(), pc[i].to_bits(), "row {i}");
        }
    }
}

#[cfg(test)]
mod ridge_tests {
    use super::*;

    fn synth(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64)
        };
        let x = Array2::from_shape_fn((n, p), |_| next());
        // Nonzero mean so a wrong intercept actually shows up in predictions
        let y = Array1::from_shape_fn(n, |i| 5.0 + x[[i, 0]] * 3.0 - x[[i, p - 1]] + next() * 0.1);
        (x, y)
    }

    /// With alpha = 0 a uniform weight c cancels out of the normal equations
    /// entirely, so predictions must match the unweighted fit. The old
    /// scale-then-center order computed means of sqrt(c)-scaled data and
    /// returned an intercept inflated by sqrt(c) (2x for c = 4).
    #[test]
    fn uniform_weights_match_unweighted() {
        let (x, y) = synth(200, 3, 11);
        let w = Array1::from_elem(200, 4.0);
        let learner = RidgeLearner::new(0.0);

        let pw = learner
            .fit_with_weights(&x, &y, Some(&w))
            .unwrap()
            .predict(&x);
        let pu = learner.fit_with_weights(&x, &y, None).unwrap().predict(&x);
        for i in 0..x.nrows() {
            assert!(
                (pw[i] - pu[i]).abs() <= 1e-9 * pu[i].abs().max(1.0),
                "row {i}: weighted {} vs unweighted {}",
                pw[i],
                pu[i]
            );
        }
    }

    /// sklearn semantics: an integer weight w on a row is exactly equivalent
    /// to duplicating that row w times (holds for any alpha — both sides
    /// produce the same X'WX + alpha*I system and the same weighted means).
    #[test]
    fn weight_two_equals_duplicated_rows() {
        let (x, y) = synth(120, 3, 23);
        let n = x.nrows();
        // Even rows get weight 2, odd rows weight 1
        let w = Array1::from_shape_fn(n, |i| if i % 2 == 0 { 2.0 } else { 1.0 });

        // Build the equivalent duplicated dataset
        let n_dup = n + n / 2 + n % 2;
        let mut xd = Array2::zeros((n_dup, x.ncols()));
        let mut yd = Array1::zeros(n_dup);
        let mut r = 0;
        for i in 0..n {
            let reps = if i % 2 == 0 { 2 } else { 1 };
            for _ in 0..reps {
                for j in 0..x.ncols() {
                    xd[[r, j]] = x[[i, j]];
                }
                yd[r] = y[i];
                r += 1;
            }
        }
        assert_eq!(r, n_dup);

        let learner = RidgeLearner::new(0.5);
        let pw = learner
            .fit_with_weights(&x, &y, Some(&w))
            .unwrap()
            .predict(&x);
        let pd = learner.fit_with_weights(&xd, &yd, None).unwrap().predict(&x);
        for i in 0..n {
            assert!(
                (pw[i] - pd[i]).abs() <= 1e-9 * pd[i].abs().max(1.0),
                "row {i}: weighted {} vs duplicated {}",
                pw[i],
                pd[i]
            );
        }
    }
}
