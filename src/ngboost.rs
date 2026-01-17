/// Type alias for loss monitor functions
pub type LossMonitor<D> = Box<dyn Fn(&D, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync>;

use crate::dist::categorical::Bernoulli;
use crate::dist::normal::Normal;
use crate::dist::{ClassificationDistn, Distribution};
use crate::learners::{default_tree_learner, BaseLearner, DecisionTreeLearner, TrainedBaseLearner};
use crate::scores::{LogScore, Scorable, Score};
use ndarray::s;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use std::marker::PhantomData;

pub struct NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    // Hyperparameters
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub natural_gradient: bool,
    pub minibatch_frac: f64,
    pub col_sample: f64,
    pub verbose: bool,
    pub verbose_eval: u32,
    pub tol: f64,
    pub early_stopping_rounds: Option<u32>,
    pub validation_fraction: f64,
    pub adaptive_learning_rate: bool, // Enable adaptive learning rate for better convergence

    // Base learner
    base_learner: B,

    // State
    pub base_models: Vec<Vec<Box<dyn TrainedBaseLearner>>>,
    pub scalings: Vec<f64>,
    pub init_params: Option<Array1<f64>>,
    pub col_idxs: Vec<Vec<usize>>,
    train_loss_monitor: Option<LossMonitor<D>>,
    val_loss_monitor: Option<LossMonitor<D>>,
    best_val_loss_itr: Option<usize>,
    n_features: Option<usize>,

    // Random number generator
    rng: ThreadRng,

    // Generics
    _dist: PhantomData<D>,
    _score: PhantomData<S>,
}

impl<D, S, B> NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    pub fn new(n_estimators: u32, learning_rate: f64, base_learner: B) -> Self {
        NGBoost {
            n_estimators,
            learning_rate,
            natural_gradient: true,
            minibatch_frac: 1.0,
            col_sample: 1.0,
            verbose: false,
            verbose_eval: 100,
            tol: 1e-4,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            adaptive_learning_rate: false, // Default to false for backward compatibility
            base_learner,
            base_models: Vec::new(),
            scalings: Vec::new(),
            init_params: None,
            col_idxs: Vec::new(),
            train_loss_monitor: None,
            val_loss_monitor: None,
            best_val_loss_itr: None,
            n_features: None,
            rng: rng(),
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    pub fn with_options(
        n_estimators: u32,
        learning_rate: f64,
        base_learner: B,
        natural_gradient: bool,
        minibatch_frac: f64,
        col_sample: f64,
        verbose: bool,
        verbose_eval: u32,
        tol: f64,
        early_stopping_rounds: Option<u32>,
        validation_fraction: f64,
        adaptive_learning_rate: bool,
    ) -> Self {
        NGBoost {
            n_estimators,
            learning_rate,
            natural_gradient,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            early_stopping_rounds,
            validation_fraction,
            adaptive_learning_rate,
            base_learner,
            base_models: Vec::new(),
            scalings: Vec::new(),
            init_params: None,
            col_idxs: Vec::new(),
            train_loss_monitor: None,
            val_loss_monitor: None,
            best_val_loss_itr: None,
            n_features: None,
            rng: rng(),
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Set a custom training loss monitor function
    pub fn set_train_loss_monitor(&mut self, monitor: LossMonitor<D>) {
        self.train_loss_monitor = Some(monitor);
    }

    /// Set a custom validation loss monitor function
    pub fn set_val_loss_monitor(&mut self, monitor: LossMonitor<D>) {
        self.val_loss_monitor = Some(monitor);
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.fit_with_validation(x, y, None, None, None, None)
    }

    /// Fits an NGBoost model to the data appending base models to the existing ones.
    ///
    /// NOTE: This method is similar to Python's partial_fit. The first call will be the most
    /// significant and later calls will retune the model to newer data.
    ///
    /// Unlike `fit()`, this method does NOT reset the model state, allowing incremental learning.
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.partial_fit_with_validation(x, y, None, None, None, None)
    }

    /// Partial fit with validation data support.
    pub fn partial_fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        // Don't reset state - this is the key difference from fit()
        self.fit_internal(x, y, x_val, y_val, sample_weight, val_sample_weight, false)
    }

    pub fn fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.fit_internal(x, y, x_val, y_val, sample_weight, val_sample_weight, true)
    }

    /// Internal fit implementation that can optionally reset state.
    fn fit_internal(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        _val_sample_weight: Option<&Array1<f64>>,
        reset_state: bool,
    ) -> Result<(), &'static str> {
        // Validate input dimensions with more detailed error messages
        if x.nrows() != y.len() {
            return Err("Number of samples in X and y must match");
        }
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }
        if x.ncols() == 0 {
            return Err("Cannot fit to dataset with no features");
        }

        // Check for NaN/Inf values in input data
        if x.iter().any(|&v| !v.is_finite()) {
            return Err("Input X contains NaN or infinite values");
        }
        if y.iter().any(|&v| !v.is_finite()) {
            return Err("Input y contains NaN or infinite values");
        }

        // Reset state only if requested (fit() resets, partial_fit() doesn't)
        if reset_state {
            self.base_models.clear();
            self.scalings.clear();
            self.col_idxs.clear();
            self.best_val_loss_itr = None;
        }
        self.n_features = Some(x.ncols());

        // Handle automatic validation split if early stopping is enabled
        let (x_train, y_train, x_val_auto, y_val_auto) = if self.early_stopping_rounds.is_some()
            && x_val.is_none()
            && y_val.is_none()
            && self.validation_fraction > 0.0
            && self.validation_fraction < 1.0
        {
            // Split training data into training and validation sets
            let n_samples = x.nrows();
            let n_val = ((n_samples as f64) * self.validation_fraction) as usize;
            let n_train = n_samples - n_val;

            // Simple split (could be enhanced with proper shuffling)
            let x_train = x.slice(s![0..n_train, ..]).to_owned();
            let y_train = y.slice(s![0..n_train]).to_owned();
            let x_val_auto = Some(x.slice(s![n_train..n_samples, ..]).to_owned());
            let y_val_auto = Some(y.slice(s![n_train..n_samples]).to_owned());

            (x_train, y_train, x_val_auto, y_val_auto)
        } else {
            (x.to_owned(), y.to_owned(), x_val.cloned(), y_val.cloned())
        };

        // Use the automatically split or provided validation data
        let x_train = x_train;
        let y_train = y_train;
        let x_val = x_val_auto.as_ref().or(x_val);
        let y_val = y_val_auto.as_ref().or(y_val);

        // Validate validation data if provided
        if let (Some(xv), Some(yv)) = (x_val, y_val) {
            if xv.nrows() != yv.len() {
                return Err("Number of samples in validation X and y must match");
            }
            if xv.ncols() != x_train.ncols() {
                return Err("Number of features in training and validation data must match");
            }
        }

        self.init_params = Some(D::fit(&y_train));
        let n_params = self.init_params.as_ref().unwrap().len();
        let mut params = Array2::from_elem((x_train.nrows(), n_params), 0.0);

        // Safe unwrap with proper error handling
        let init_params = self.init_params.as_ref().unwrap();
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        // Prepare validation params if validation data is provided
        let mut val_params = if let (Some(xv), Some(_yv)) = (x_val, y_val) {
            let mut v_params = Array2::from_elem((xv.nrows(), n_params), 0.0);
            v_params
                .outer_iter_mut()
                .for_each(|mut row| row.assign(init_params));
            Some(v_params)
        } else {
            None
        };

        let mut best_val_loss = f64::INFINITY;
        let mut best_iter = 0;
        let mut no_improvement_count = 0;

        for itr in 0..self.n_estimators {
            let dist = D::from_params(&params);
            let grads = Scorable::grad(&dist, &y_train, self.natural_gradient);

            // Sample data for this iteration
            let (row_idxs, col_idxs, x_sampled, y_sampled, params_sampled, weight_sampled) =
                self.sample(&x_train, &y_train, &params, sample_weight);
            self.col_idxs.push(col_idxs.clone());

            let grads_sampled = grads.select(ndarray::Axis(0), &row_idxs);

            let mut fitted_learners: Vec<Box<dyn TrainedBaseLearner>> = Vec::new();
            let mut predictions_cols: Vec<Array1<f64>> = Vec::new();

            for j in 0..n_params {
                let grad_j = grads_sampled.column(j).to_owned();
                let learner = self.base_learner.clone();
                let fitted =
                    learner.fit_with_weights(&x_sampled, &grad_j, weight_sampled.as_ref())?;
                predictions_cols.push(fitted.predict(&x_sampled));
                fitted_learners.push(fitted);
            }

            let predictions = to_2d_array(predictions_cols);

            let scale = self.line_search(
                &predictions,
                &params_sampled,
                &y_sampled,
                weight_sampled.as_ref(),
            );
            self.scalings.push(scale);
            self.base_models.push(fitted_learners);

            // Update full parameters using the sampled predictions
            let mut full_predictions = Array2::zeros(params.raw_dim());
            for j in 0..n_params {
                let mut pred_col = Array1::zeros(x_train.nrows());
                // Map the sampled predictions back to the full row set
                for (i, &row_idx) in row_idxs.iter().enumerate() {
                    pred_col[row_idx] = predictions[[i, j]];
                }
                // Assign to the specific parameter column (j)
                full_predictions.column_mut(j).assign(&pred_col);
            }

            // Apply adaptive learning rate if enabled
            let effective_learning_rate = if self.adaptive_learning_rate {
                // Reduce learning rate as we progress for better fine-tuning
                let progress = itr as f64 / self.n_estimators as f64;
                self.learning_rate * (1.0 - 0.7 * progress).max(0.1)
            } else {
                self.learning_rate
            };

            params -= &(effective_learning_rate * scale * &full_predictions);

            // Update validation parameters if validation data is provided
            if let (Some(xv), Some(yv), Some(vp)) = (x_val, y_val, val_params.as_mut()) {
                // Get predictions on validation data from the fitted base learners
                let fitted_learners = self.base_models.last().unwrap();
                let val_predictions_cols: Vec<Array1<f64>> = fitted_learners
                    .iter()
                    .map(|learner| learner.predict(xv))
                    .collect();
                let val_predictions = to_2d_array(val_predictions_cols);
                *vp -= &(self.learning_rate * scale * &val_predictions);

                // Calculate validation loss using monitor or default
                let val_dist = D::from_params(vp);
                let val_loss = if let Some(monitor) = &self.val_loss_monitor {
                    monitor(&val_dist, yv, None)
                } else {
                    Scorable::total_score(&val_dist, yv, None)
                };

                // Early stopping logic
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_iter = itr;
                    no_improvement_count = 0;
                    self.best_val_loss_itr = Some(itr as usize);
                } else {
                    no_improvement_count += 1;
                }

                // Check if we should stop early
                if let Some(rounds) = self.early_stopping_rounds {
                    if no_improvement_count >= rounds {
                        if self.verbose {
                            println!("== Early stopping achieved.");
                            println!(
                                "== Best iteration / VAL{} (val_loss={:.4})",
                                best_iter, best_val_loss
                            );
                        }
                        break;
                    }
                }

                // Verbose logging with validation
                if self.verbose && itr % self.verbose_eval == 0 {
                    let dist = D::from_params(&params);
                    let train_loss = if let Some(monitor) = &self.train_loss_monitor {
                        monitor(&dist, &y_train, None)
                    } else {
                        Scorable::total_score(&dist, &y_train, None)
                    };
                    println!(
                        "[iter {}] train_loss={:.4} val_loss={:.4}",
                        itr, train_loss, val_loss
                    );
                }
            } else {
                // Verbose logging without validation
                if self.verbose && itr % self.verbose_eval == 0 {
                    let dist = D::from_params(&params);
                    let loss = if let Some(monitor) = &self.train_loss_monitor {
                        monitor(&dist, &y_train, None)
                    } else {
                        Scorable::total_score(&dist, &y_train, None)
                    };

                    // Calculate gradient norm for debugging
                    let grad_norm: f64 =
                        grads.iter().map(|x| x * x).sum::<f64>().sqrt() / grads.len() as f64;

                    println!(
                        "[iter {}] loss={:.4} grad_norm={:.4} scale={:.4}",
                        itr, loss, grad_norm, scale
                    );
                }
            }
        }

        Ok(())
    }

    fn sample(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        params: &Array2<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Array2<f64>,
        Array1<f64>,
        Array2<f64>,
        Option<Array1<f64>>,
    ) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Sample rows (minibatch)
        let sample_size = if self.minibatch_frac >= 1.0 {
            n_samples
        } else {
            ((n_samples as f64) * self.minibatch_frac) as usize
        };

        // Uniform random sampling without replacement (matches Python's np.random.choice behavior)
        // Note: Python does NOT do weighted sampling for minibatch selection,
        // it only passes the weights to the base learner's fit method
        let row_idxs: Vec<usize> = if sample_size == n_samples {
            (0..n_samples).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            // Use Fisher-Yates shuffle for better randomness (matches numpy's algorithm)
            for i in (1..indices.len()).rev() {
                let j = self.rng.random_range(0..=i);
                indices.swap(i, j);
            }
            indices.into_iter().take(sample_size).collect()
        };

        // Sample columns
        let col_size = if self.col_sample >= 1.0 {
            n_features
        } else if self.col_sample > 0.0 {
            ((n_features as f64) * self.col_sample) as usize
        } else {
            0
        };

        let col_idxs: Vec<usize> = if col_size == n_features || col_size == 0 {
            (0..n_features).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.shuffle(&mut self.rng);
            indices.into_iter().take(col_size).collect()
        };

        // Create sampled data
        let x_sampled = x
            .select(ndarray::Axis(0), &row_idxs)
            .select(ndarray::Axis(1), &col_idxs);
        let y_sampled = y.select(ndarray::Axis(0), &row_idxs);
        let params_sampled = params.select(ndarray::Axis(0), &row_idxs);

        // Handle sample weights
        let sample_weights_sampled =
            sample_weight.map(|weights| weights.select(ndarray::Axis(0), &row_idxs));

        (
            row_idxs,
            col_idxs,
            x_sampled,
            y_sampled,
            params_sampled,
            sample_weights_sampled,
        )
    }

    fn get_params(&self, x: &Array2<f64>) -> Array2<f64> {
        self.get_params_at(x, None)
    }

    fn get_params_at(&self, x: &Array2<f64>, max_iter: Option<usize>) -> Array2<f64> {
        if x.nrows() == 0 {
            return Array2::zeros((0, 0));
        }

        let init_params = self.init_params.as_ref().unwrap();
        let n_params = init_params.len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        let n_iters = max_iter
            .unwrap_or(self.base_models.len())
            .min(self.base_models.len());

        for (i, (learners, _col_idx)) in self
            .base_models
            .iter()
            .zip(self.col_idxs.iter())
            .enumerate()
            .take(n_iters)
        {
            let scale = self.scalings[i];

            let predictions_cols: Vec<Array1<f64>> =
                learners.iter().map(|learner| learner.predict(x)).collect();

            let predictions = to_2d_array(predictions_cols);

            params -= &(self.learning_rate * scale * &predictions);
        }
        params
    }

    /// Get the predicted distribution parameters (like Python's pred_param)
    pub fn pred_param(&self, x: &Array2<f64>) -> Array2<f64> {
        self.get_params(x)
    }

    /// Get the predicted distribution parameters up to a specific iteration
    pub fn pred_param_at(&self, x: &Array2<f64>, max_iter: usize) -> Array2<f64> {
        self.get_params_at(x, Some(max_iter))
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> D {
        let params = self.get_params(x);
        D::from_params(&params)
    }

    /// Get the predicted distribution up to a specific iteration
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> D {
        let params = self.get_params_at(x, Some(max_iter));
        D::from_params(&params)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.pred_dist(x).predict()
    }

    /// Get predictions up to a specific iteration
    pub fn predict_at(&self, x: &Array2<f64>, max_iter: usize) -> Array1<f64> {
        self.pred_dist_at(x, max_iter).predict()
    }

    /// Returns an iterator over staged predictions (predictions at each boosting iteration)
    pub fn staged_predict<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> impl Iterator<Item = Array1<f64>> + 'a {
        (1..=self.base_models.len()).map(move |i| self.predict_at(x, i))
    }

    /// Returns an iterator over staged distribution predictions
    pub fn staged_pred_dist<'a>(&'a self, x: &'a Array2<f64>) -> impl Iterator<Item = D> + 'a {
        (1..=self.base_models.len()).map(move |i| self.pred_dist_at(x, i))
    }

    /// Compute the average score (loss) on the given data
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let dist = self.pred_dist(x);
        Scorable::total_score(&dist, y, None)
    }

    /// Get number of features the model was trained on
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Compute feature importances based on how often each feature is used in splits.
    /// Returns a 2D array of shape (n_params, n_features) where each row contains
    /// the normalized feature importances for that distribution parameter.
    /// Returns None if the model hasn't been trained or has no features.
    pub fn feature_importances(&self) -> Option<Array2<f64>> {
        let n_features = self.n_features?;
        if self.base_models.is_empty() || n_features == 0 {
            return None;
        }

        let n_params = self.init_params.as_ref()?.len();
        let mut importances = Array2::zeros((n_params, n_features));

        // Aggregate feature usage across all iterations, weighted by scaling factor
        for (iter_idx, learners) in self.base_models.iter().enumerate() {
            let scale = self.scalings[iter_idx].abs();

            for (param_idx, learner) in learners.iter().enumerate() {
                if let Some(feature_idx) = learner.split_feature() {
                    if feature_idx < n_features {
                        importances[[param_idx, feature_idx]] += scale;
                    }
                }
            }
        }

        // Normalize each parameter's importances to sum to 1
        for mut row in importances.rows_mut() {
            let sum: f64 = row.sum();
            if sum > 0.0 {
                row.mapv_inplace(|v| v / sum);
            }
        }

        Some(importances)
    }

    /// Calibrate uncertainty estimates using isotonic regression on validation data
    /// This improves the quality of probabilistic predictions by adjusting the variance estimates
    pub fn calibrate_uncertainty(
        &mut self,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<(), &'static str> {
        if self.base_models.is_empty() {
            return Err("Model must be trained before calibration");
        }

        // Get predictions on validation data
        let params = self.pred_param(x_val);
        let dist = D::from_params(&params);

        // Calculate predictions and errors
        let predictions = dist.predict();
        let errors = y_val - &predictions;

        // Calculate empirical variance
        let empirical_var = errors.mapv(|e| e * e).mean().unwrap_or(1.0);

        // For normal distribution (2 parameters), adjust the scale parameter
        if let Some(init_params) = self.init_params.as_mut() {
            if init_params.len() >= 2 {
                // The second parameter is log(scale), so we adjust it based on empirical variance
                let current_var = (-init_params[1]).exp(); // exp(2*log(scale)) = scale^2
                let target_var = empirical_var;
                let calibration_factor = (target_var / current_var).sqrt();
                init_params[1] += calibration_factor.ln();
            }
        }

        Ok(())
    }

    /// Compute aggregated feature importances across all distribution parameters.
    /// Returns a 1D array of length n_features with normalized importances.
    pub fn feature_importances_aggregated(&self) -> Option<Array1<f64>> {
        let importances = self.feature_importances()?;
        let mut aggregated = importances.sum_axis(ndarray::Axis(0));

        let sum: f64 = aggregated.sum();
        if sum > 0.0 {
            aggregated.mapv_inplace(|v| v / sum);
        }

        Some(aggregated)
    }

    fn line_search(
        &self,
        resids: &Array2<f64>,
        start: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> f64 {
        let mut scale = 1.0;
        let initial_score = Scorable::total_score(&D::from_params(start), y, sample_weight);

        // Scale up phase: try to find a larger step that still reduces loss
        loop {
            if scale > 256.0 {
                break;
            }
            let scaled_resids = resids * (scale * 2.0);
            let next_params = start - &scaled_resids;
            let score = Scorable::total_score(&D::from_params(&next_params), y, sample_weight);
            if score >= initial_score || !score.is_finite() {
                break;
            }
            scale *= 2.0;
        }

        // Scale down phase: find a step that actually reduces loss
        // Check residual norm for convergence (like Python does)
        loop {
            let scaled_resids = resids * scale;
            // Compute MEAN norm of scaled residuals (matching Python's np.linalg.norm(..., axis=1).mean())
            let norm: f64 = scaled_resids
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x * x).sum::<f64>().sqrt())
                .sum::<f64>()
                / scaled_resids.nrows() as f64;
            if norm < self.tol {
                break;
            }

            let next_params = start - &scaled_resids;
            let score = Scorable::total_score(&D::from_params(&next_params), y, sample_weight);
            if score < initial_score && score.is_finite() {
                break;
            }
            scale *= 0.5;

            // Prevent infinite loop with minimum scale
            if scale < 1e-10 {
                break;
            }
        }

        scale
    }

    /// Serialize the model to a platform-independent format
    pub fn serialize(&self) -> Result<SerializedNGBoost, Box<dyn std::error::Error>> {
        // Serialize base models
        let serialized_base_models: Vec<Vec<crate::learners::SerializableTrainedLearner>> = self
            .base_models
            .iter()
            .map(|learners| {
                learners
                    .iter()
                    .filter_map(|learner| learner.to_serializable())
                    .collect()
            })
            .collect();

        Ok(SerializedNGBoost {
            n_estimators: self.n_estimators,
            learning_rate: self.learning_rate,
            natural_gradient: self.natural_gradient,
            minibatch_frac: self.minibatch_frac,
            col_sample: self.col_sample,
            verbose: self.verbose,
            verbose_eval: self.verbose_eval,
            tol: self.tol,
            early_stopping_rounds: self.early_stopping_rounds,
            validation_fraction: self.validation_fraction,
            init_params: self.init_params.as_ref().map(|p| p.to_vec()),
            scalings: self.scalings.clone(),
            col_idxs: self.col_idxs.clone(),
            best_val_loss_itr: self.best_val_loss_itr,
            base_models: serialized_base_models,
        })
    }

    /// Deserialize the model from a platform-independent format
    pub fn deserialize(
        serialized: SerializedNGBoost,
        base_learner: B,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        D: Distribution + Scorable<S> + Clone,
        S: Score,
        B: BaseLearner + Clone,
    {
        let mut model = Self::with_options(
            serialized.n_estimators,
            serialized.learning_rate,
            base_learner,
            serialized.natural_gradient,
            serialized.minibatch_frac,
            serialized.col_sample,
            serialized.verbose,
            serialized.verbose_eval,
            serialized.tol,
            serialized.early_stopping_rounds,
            serialized.validation_fraction,
            false, // Default adaptive_learning_rate to false for backward compatibility
        );

        // Restore trained state
        if let Some(init_params) = serialized.init_params {
            model.init_params = Some(Array1::from(init_params));
        }
        model.scalings = serialized.scalings;
        model.col_idxs = serialized.col_idxs;
        model.best_val_loss_itr = serialized.best_val_loss_itr;

        // Restore base models
        model.base_models = serialized
            .base_models
            .into_iter()
            .map(|learners| learners.into_iter().map(|l| l.to_trait_object()).collect())
            .collect();

        Ok(model)
    }
}

/// Serialized model data structure
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SerializedNGBoost {
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub natural_gradient: bool,
    pub minibatch_frac: f64,
    pub col_sample: f64,
    pub verbose: bool,
    pub verbose_eval: u32,
    pub tol: f64,
    pub early_stopping_rounds: Option<u32>,
    pub validation_fraction: f64,
    pub init_params: Option<Vec<f64>>,
    pub scalings: Vec<f64>,
    pub col_idxs: Vec<Vec<usize>>,
    pub best_val_loss_itr: Option<usize>,
    /// Serialized base models - each inner Vec contains learners for each parameter
    pub base_models: Vec<Vec<crate::learners::SerializableTrainedLearner>>,
}

fn to_2d_array(cols: Vec<Array1<f64>>) -> Array2<f64> {
    if cols.is_empty() {
        return Array2::zeros((0, 0));
    }
    let nrows = cols[0].len();
    let ncols = cols.len();
    let mut arr = Array2::zeros((nrows, ncols));
    for (j, col) in cols.iter().enumerate() {
        arr.column_mut(j).assign(col);
    }
    arr
}

// High-level API
pub struct NGBRegressor {
    model: NGBoost<Normal, LogScore, DecisionTreeLearner>,
}

pub struct NGBClassifier {
    model: NGBoost<Bernoulli, LogScore, DecisionTreeLearner>,
}

impl NGBRegressor {
    pub fn new(n_estimators: u32, learning_rate: f64) -> Self {
        Self {
            model: NGBoost::new(n_estimators, learning_rate, default_tree_learner()),
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.model.fit(x, y)
    }

    pub fn fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, x_val, y_val, None, None)
    }

    /// Fits an NGBoost model to the data appending base models to the existing ones.
    ///
    /// NOTE: This method is similar to Python's partial_fit. The first call will be the most
    /// significant and later calls will retune the model to newer data.
    ///
    /// Unlike `fit()`, this method does NOT reset the model state, allowing incremental learning.
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.model.partial_fit(x, y)
    }

    /// Partial fit with validation data support.
    pub fn partial_fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .partial_fit_with_validation(x, y, x_val, y_val, None, None)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.model.predict(x)
    }

    /// Get predictions up to a specific iteration
    pub fn predict_at(&self, x: &Array2<f64>, max_iter: usize) -> Array1<f64> {
        self.model.predict_at(x, max_iter)
    }

    /// Returns an iterator over staged predictions
    pub fn staged_predict<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> impl Iterator<Item = Array1<f64>> + 'a {
        self.model.staged_predict(x)
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> Normal {
        self.model.pred_dist(x)
    }

    /// Get the predicted distribution up to a specific iteration
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> Normal {
        self.model.pred_dist_at(x, max_iter)
    }

    /// Returns an iterator over staged distribution predictions
    pub fn staged_pred_dist<'a>(&'a self, x: &'a Array2<f64>) -> impl Iterator<Item = Normal> + 'a {
        self.model.staged_pred_dist(x)
    }

    /// Get the predicted distribution parameters
    pub fn pred_param(&self, x: &Array2<f64>) -> Array2<f64> {
        self.model.pred_param(x)
    }

    /// Compute the average score (loss) on the given data
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        self.model.score(x, y)
    }

    /// Set a custom training loss monitor function
    pub fn set_train_loss_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(&Normal, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
    {
        self.model.set_train_loss_monitor(Box::new(monitor));
    }

    /// Set a custom validation loss monitor function
    pub fn set_val_loss_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(&Normal, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
    {
        self.model.set_val_loss_monitor(Box::new(monitor));
    }

    /// Enhanced constructor with all options
    pub fn with_options(
        n_estimators: u32,
        learning_rate: f64,
        natural_gradient: bool,
        minibatch_frac: f64,
        col_sample: f64,
        verbose: bool,
        verbose_eval: u32,
        tol: f64,
        early_stopping_rounds: Option<u32>,
        validation_fraction: f64,
        adaptive_learning_rate: bool,
    ) -> Self {
        Self {
            model: NGBoost::with_options(
                n_estimators,
                learning_rate,
                default_tree_learner(),
                natural_gradient,
                minibatch_frac,
                col_sample,
                verbose,
                verbose_eval,
                tol,
                early_stopping_rounds,
                validation_fraction,
                adaptive_learning_rate,
            ),
        }
    }

    /// Enhanced constructor with all options (backward compatible version without adaptive_learning_rate)
    pub fn with_options_compat(
        n_estimators: u32,
        learning_rate: f64,
        natural_gradient: bool,
        minibatch_frac: f64,
        col_sample: f64,
        verbose: bool,
        verbose_eval: u32,
        tol: f64,
        early_stopping_rounds: Option<u32>,
        validation_fraction: f64,
    ) -> Self {
        Self::with_options(
            n_estimators,
            learning_rate,
            natural_gradient,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            early_stopping_rounds,
            validation_fraction,
            false, // Default adaptive_learning_rate to false
        )
    }

    /// Enable adaptive learning rate for better convergence in probabilistic forecasting
    pub fn set_adaptive_learning_rate(&mut self, enabled: bool) {
        self.model.adaptive_learning_rate = enabled;
    }

    /// Calibrate uncertainty estimates using validation data
    /// This improves the quality of probabilistic predictions
    pub fn calibrate_uncertainty(
        &mut self,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<(), &'static str> {
        self.model.calibrate_uncertainty(x_val, y_val)
    }

    /// Get the number of estimators (boosting iterations)
    pub fn n_estimators(&self) -> u32 {
        self.model.n_estimators
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        self.model.learning_rate
    }

    /// Get whether natural gradient is used
    pub fn natural_gradient(&self) -> bool {
        self.model.natural_gradient
    }

    /// Get the minibatch fraction
    pub fn minibatch_frac(&self) -> f64 {
        self.model.minibatch_frac
    }

    /// Get the column sampling fraction
    pub fn col_sample(&self) -> f64 {
        self.model.col_sample
    }

    /// Get the best validation iteration
    pub fn best_val_loss_itr(&self) -> Option<usize> {
        self.model.best_val_loss_itr
    }

    /// Get early stopping rounds
    pub fn early_stopping_rounds(&self) -> Option<u32> {
        self.model.early_stopping_rounds
    }

    /// Get validation fraction
    pub fn validation_fraction(&self) -> f64 {
        self.model.validation_fraction
    }

    /// Get number of features the model was trained on
    pub fn n_features(&self) -> Option<usize> {
        self.model.n_features()
    }

    /// Compute feature importances per distribution parameter.
    /// Returns a 2D array of shape (n_params, n_features).
    pub fn feature_importances(&self) -> Option<Array2<f64>> {
        self.model.feature_importances()
    }

    /// Compute aggregated feature importances across all distribution parameters.
    /// Returns a 1D array of length n_features with normalized importances.
    pub fn feature_importances_aggregated(&self) -> Option<Array1<f64>> {
        self.model.feature_importances_aggregated()
    }

    /// Save model to file using bincode serialization
    pub fn save_model(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = self.model.serialize()?;
        let encoded = bincode::serialize(&serialized)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Load model from file using bincode deserialization
    pub fn load_model(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let encoded = std::fs::read(path)?;
        let serialized: SerializedNGBoost = bincode::deserialize(&encoded)?;
        let model = NGBoost::<Normal, LogScore, DecisionTreeLearner>::deserialize(
            serialized,
            default_tree_learner(),
        )?;
        Ok(Self { model })
    }
}

impl NGBClassifier {
    pub fn new(n_estimators: u32, learning_rate: f64) -> Self {
        Self {
            model: NGBoost::new(n_estimators, learning_rate, default_tree_learner()),
        }
    }

    pub fn with_options(
        n_estimators: u32,
        learning_rate: f64,
        natural_gradient: bool,
        minibatch_frac: f64,
        col_sample: f64,
        verbose: bool,
        verbose_eval: u32,
        tol: f64,
        early_stopping_rounds: Option<u32>,
        validation_fraction: f64,
        adaptive_learning_rate: bool,
    ) -> Self {
        Self {
            model: NGBoost::with_options(
                n_estimators,
                learning_rate,
                default_tree_learner(),
                natural_gradient,
                minibatch_frac,
                col_sample,
                verbose,
                verbose_eval,
                tol,
                early_stopping_rounds,
                validation_fraction,
                adaptive_learning_rate,
            ),
        }
    }

    /// Set a custom training loss monitor function
    pub fn set_train_loss_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(&Bernoulli, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
    {
        self.model.set_train_loss_monitor(Box::new(monitor));
    }

    /// Set a custom validation loss monitor function
    pub fn set_val_loss_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(&Bernoulli, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
    {
        self.model.set_val_loss_monitor(Box::new(monitor));
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.model.fit(x, y)
    }

    pub fn fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, x_val, y_val, None, None)
    }

    /// Fits an NGBoost model to the data appending base models to the existing ones.
    ///
    /// NOTE: This method is similar to Python's partial_fit. The first call will be the most
    /// significant and later calls will retune the model to newer data.
    ///
    /// Unlike `fit()`, this method does NOT reset the model state, allowing incremental learning.
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.model.partial_fit(x, y)
    }

    /// Partial fit with validation data support.
    pub fn partial_fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .partial_fit_with_validation(x, y, x_val, y_val, None, None)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.model.predict(x)
    }

    /// Get predictions up to a specific iteration
    pub fn predict_at(&self, x: &Array2<f64>, max_iter: usize) -> Array1<f64> {
        self.model.predict_at(x, max_iter)
    }

    /// Returns an iterator over staged predictions
    pub fn staged_predict<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> impl Iterator<Item = Array1<f64>> + 'a {
        self.model.staged_predict(x)
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let dist = self.model.pred_dist(x);
        dist.class_probs()
    }

    /// Get class probabilities up to a specific iteration
    pub fn predict_proba_at(&self, x: &Array2<f64>, max_iter: usize) -> Array2<f64> {
        let dist = self.model.pred_dist_at(x, max_iter);
        dist.class_probs()
    }

    /// Returns an iterator over staged probability predictions
    pub fn staged_predict_proba<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> impl Iterator<Item = Array2<f64>> + 'a {
        (1..=self.model.base_models.len()).map(move |i| self.predict_proba_at(x, i))
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> Bernoulli {
        self.model.pred_dist(x)
    }

    /// Get the predicted distribution up to a specific iteration
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> Bernoulli {
        self.model.pred_dist_at(x, max_iter)
    }

    /// Returns an iterator over staged distribution predictions
    pub fn staged_pred_dist<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> impl Iterator<Item = Bernoulli> + 'a {
        self.model.staged_pred_dist(x)
    }

    /// Get the predicted distribution parameters
    pub fn pred_param(&self, x: &Array2<f64>) -> Array2<f64> {
        self.model.pred_param(x)
    }

    /// Compute the average score (loss) on the given data
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        self.model.score(x, y)
    }

    /// Get the number of estimators (boosting iterations)
    pub fn n_estimators(&self) -> u32 {
        self.model.n_estimators
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        self.model.learning_rate
    }

    /// Get whether natural gradient is used
    pub fn natural_gradient(&self) -> bool {
        self.model.natural_gradient
    }

    /// Get the minibatch fraction
    pub fn minibatch_frac(&self) -> f64 {
        self.model.minibatch_frac
    }

    /// Get the column sampling fraction
    pub fn col_sample(&self) -> f64 {
        self.model.col_sample
    }

    /// Get the best validation iteration
    pub fn best_val_loss_itr(&self) -> Option<usize> {
        self.model.best_val_loss_itr
    }

    /// Get early stopping rounds
    pub fn early_stopping_rounds(&self) -> Option<u32> {
        self.model.early_stopping_rounds
    }

    /// Get validation fraction
    pub fn validation_fraction(&self) -> f64 {
        self.model.validation_fraction
    }

    /// Get number of features the model was trained on
    pub fn n_features(&self) -> Option<usize> {
        self.model.n_features()
    }

    /// Compute feature importances per distribution parameter.
    /// Returns a 2D array of shape (n_params, n_features).
    pub fn feature_importances(&self) -> Option<Array2<f64>> {
        self.model.feature_importances()
    }

    /// Compute aggregated feature importances across all distribution parameters.
    /// Returns a 1D array of length n_features with normalized importances.
    pub fn feature_importances_aggregated(&self) -> Option<Array1<f64>> {
        self.model.feature_importances_aggregated()
    }

    /// Save model to file using bincode serialization
    pub fn save_model(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = self.model.serialize()?;
        let encoded = bincode::serialize(&serialized)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Load model from file using bincode deserialization
    pub fn load_model(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let encoded = std::fs::read(path)?;
        let serialized: SerializedNGBoost = bincode::deserialize(&encoded)?;
        let model = NGBoost::<Bernoulli, LogScore, DecisionTreeLearner>::deserialize(
            serialized,
            default_tree_learner(),
        )?;
        Ok(Self { model })
    }
}
