//! Survival analysis support for NGBoost.
//!
//! This module provides NGBSurvival, a wrapper for NGBoost that handles
//! right-censored survival data, similar to Python's NGBSurvival.

use crate::dist::exponential::Exponential;
use crate::dist::lognormal::LogNormal;
use crate::dist::weibull::Weibull;
use crate::dist::{Distribution, RegressionDistn};
use crate::learners::{default_tree_learner, BaseLearner, DecisionTreeLearner, TrainedBaseLearner};
use crate::scores::{CRPScoreCensored, CensoredScorable, LogScoreCensored, Score, SurvivalData};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rng;
use std::marker::PhantomData;

/// NGBoost for survival analysis with right-censored data.
///
/// This is the Rust equivalent of Python's NGBSurvival class.
/// It handles survival data where some observations may be censored
/// (i.e., we only know that the event occurred after a certain time).
///
/// # Example
///
/// ```no_run
/// use ngboost_rs::survival::NGBSurvivalLogNormal;
/// use ndarray::{Array1, Array2};
///
/// // Create sample data
/// let x = Array2::from_shape_vec((5, 2), vec![
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
/// ]).unwrap();
/// let time = Array1::from_vec(vec![1.0, 2.0, 1.5, 3.0, 2.5]);
/// let event = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]); // 1=observed, 0=censored
///
/// let mut model = NGBSurvivalLogNormal::lognormal(10, 0.1);
/// model.fit(&x, &time, &event).unwrap();
/// let predictions = model.predict(&x);
/// assert_eq!(predictions.len(), 5);
/// ```
pub struct NGBSurvival<D, S, B = DecisionTreeLearner>
where
    D: Distribution + RegressionDistn + CensoredScorable<S> + Clone,
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
    /// Interval for verbose output during training.
    /// - If >= 1.0: print every `verbose_eval` iterations (e.g., 100 means every 100 iterations)
    /// - If < 1.0 and > 0.0: print every `verbose_eval * n_estimators` iterations (e.g., 0.1 means every 10%)
    pub verbose_eval: f64,
    pub tol: f64,
    pub early_stopping_rounds: Option<u32>,
    pub validation_fraction: f64,

    // Base learner
    base_learner: B,

    // State
    pub base_models: Vec<Vec<Box<dyn TrainedBaseLearner>>>,
    pub scalings: Vec<f64>,
    pub init_params: Option<Array1<f64>>,
    pub col_idxs: Vec<Vec<usize>>,
    best_val_loss_itr: Option<usize>,
    n_features: Option<usize>,

    // Random number generator
    rng: ThreadRng,

    // Generics
    _dist: PhantomData<D>,
    _score: PhantomData<S>,
}

impl<D, S, B> NGBSurvival<D, S, B>
where
    D: Distribution + RegressionDistn + CensoredScorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    /// Create a new NGBSurvival model with default settings.
    pub fn new(n_estimators: u32, learning_rate: f64, base_learner: B) -> Self {
        NGBSurvival {
            n_estimators,
            learning_rate,
            natural_gradient: true,
            minibatch_frac: 1.0,
            col_sample: 1.0,
            verbose: false,
            verbose_eval: 100.0,
            tol: 1e-4,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            base_learner,
            base_models: Vec::new(),
            scalings: Vec::new(),
            init_params: None,
            col_idxs: Vec::new(),
            best_val_loss_itr: None,
            n_features: None,
            rng: rng(),
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Create a new NGBSurvival model with all options specified.
    pub fn with_options(
        n_estimators: u32,
        learning_rate: f64,
        base_learner: B,
        natural_gradient: bool,
        minibatch_frac: f64,
        col_sample: f64,
        verbose: bool,
        verbose_eval: f64,
        tol: f64,
        early_stopping_rounds: Option<u32>,
        validation_fraction: f64,
    ) -> Self {
        NGBSurvival {
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
            base_learner,
            base_models: Vec::new(),
            scalings: Vec::new(),
            init_params: None,
            col_idxs: Vec::new(),
            best_val_loss_itr: None,
            n_features: None,
            rng: rng(),
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Fit the survival model to the data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `time` - Time to event or censoring for each sample
    /// * `event` - Event indicator (1.0 = event occurred, 0.0 = censored)
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        time: &Array1<f64>,
        event: &Array1<f64>,
    ) -> Result<(), &'static str> {
        self.fit_with_validation(x, time, event, None, None, None, None, None)
    }

    /// Fit the survival model with sample weights.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `time` - Time to event or censoring for each sample
    /// * `event` - Event indicator (1.0 = event occurred, 0.0 = censored)
    /// * `sample_weight` - Optional weights for each sample
    pub fn fit_with_weights(
        &mut self,
        x: &Array2<f64>,
        time: &Array1<f64>,
        event: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.fit_with_validation(x, time, event, None, None, None, sample_weight, None)
    }

    /// Fit the survival model with validation data for early stopping.
    pub fn fit_with_validation(
        &mut self,
        x: &Array2<f64>,
        time: &Array1<f64>,
        event: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        time_val: Option<&Array1<f64>>,
        event_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        // Validate input dimensions
        if x.nrows() != time.len() || x.nrows() != event.len() {
            return Err("Number of samples in X, time, and event must match");
        }
        if x.nrows() == 0 {
            return Err("Cannot fit to empty dataset");
        }
        if x.ncols() == 0 {
            return Err("Cannot fit to dataset with no features");
        }

        // Check for NaN/Inf values
        if x.iter().any(|&v| !v.is_finite()) {
            return Err("Input X contains NaN or infinite values");
        }
        if time.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err("Time values must be positive and finite");
        }

        // Validate sample weights if provided
        if let Some(weights) = sample_weight {
            if weights.len() != x.nrows() {
                return Err("Sample weights length must match number of samples");
            }
            if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
                return Err("Sample weights must be non-negative and finite");
            }
        }
        if let Some(weights) = val_sample_weight {
            if let Some(xv) = x_val {
                if weights.len() != xv.nrows() {
                    return Err("Validation sample weights length must match validation samples");
                }
            }
        }

        // Reset state
        self.base_models.clear();
        self.scalings.clear();
        self.col_idxs.clear();
        self.best_val_loss_itr = None;
        self.n_features = Some(x.ncols());

        // Create survival data
        let y = SurvivalData::from_arrays(time, event);

        // Fit initial parameters to marginal distribution (using just the times)
        self.init_params = Some(D::fit(time));
        let n_params = self.init_params.as_ref().unwrap().len();

        // Initialize parameters
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        let init_params = self.init_params.as_ref().unwrap();
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        // Prepare validation data if provided
        let val_data = if let (Some(xv), Some(tv), Some(ev)) = (x_val, time_val, event_val) {
            if xv.nrows() != tv.len() || xv.nrows() != ev.len() {
                return Err("Validation data dimensions must match");
            }
            let mut v_params = Array2::from_elem((xv.nrows(), n_params), 0.0);
            v_params
                .outer_iter_mut()
                .for_each(|mut row| row.assign(init_params));
            Some((xv.clone(), SurvivalData::from_arrays(tv, ev), v_params))
        } else {
            None
        };

        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;

        for itr in 0..self.n_estimators {
            // Create distribution from current parameters
            let dist = D::from_params(&params);

            // Compute gradients using censored scoring rule
            let grads = CensoredScorable::censored_grad(&dist, &y, self.natural_gradient);

            // Sample data for this iteration
            let (row_idxs, col_idxs, x_sampled, y_sampled, params_sampled, weights_sampled) =
                self.sample(x, &y, &params, sample_weight);
            self.col_idxs.push(col_idxs.clone());

            let grads_sampled = grads.select(ndarray::Axis(0), &row_idxs);

            // Fit base learners for each parameter
            let mut fitted_learners: Vec<Box<dyn TrainedBaseLearner>> = Vec::new();
            let mut predictions_cols: Vec<Array1<f64>> = Vec::new();

            for j in 0..n_params {
                let grad_j = grads_sampled.column(j).to_owned();
                let learner = self.base_learner.clone();
                let fitted =
                    learner.fit_with_weights(&x_sampled, &grad_j, weights_sampled.as_ref())?;
                predictions_cols.push(fitted.predict(&x_sampled));
                fitted_learners.push(fitted);
            }

            let predictions = to_2d_array(predictions_cols);

            // Line search to find optimal step size
            let scale = self.line_search(&predictions, &params_sampled, &y_sampled);
            self.scalings.push(scale);
            self.base_models.push(fitted_learners);

            // Update parameters for ALL training samples by re-predicting on full X
            // This matches Python's behavior: after fitting base learners on minibatch,
            // we predict on the FULL training set to update all parameters
            let fitted_learners = self.base_models.last().unwrap();
            let full_predictions_cols: Vec<Array1<f64>> = if col_idxs.len() == x.ncols() {
                fitted_learners
                    .iter()
                    .map(|learner| learner.predict(x))
                    .collect()
            } else {
                let x_subset = x.select(ndarray::Axis(1), &col_idxs);
                fitted_learners
                    .iter()
                    .map(|learner| learner.predict(&x_subset))
                    .collect()
            };
            let full_predictions = to_2d_array(full_predictions_cols);
            params -= &(self.learning_rate * scale * &full_predictions);

            // Handle validation and early stopping
            if let Some((ref xv, ref yv, ref mut vp)) = val_data.clone() {
                // Apply column subsampling to match training
                let fitted_learners = self.base_models.last().unwrap();
                let val_predictions_cols: Vec<Array1<f64>> = if col_idxs.len() == xv.ncols() {
                    fitted_learners
                        .iter()
                        .map(|learner| learner.predict(xv))
                        .collect()
                } else {
                    let xv_subset = xv.select(ndarray::Axis(1), &col_idxs);
                    fitted_learners
                        .iter()
                        .map(|learner| learner.predict(&xv_subset))
                        .collect()
                };
                let val_predictions = to_2d_array(val_predictions_cols);
                *vp -= &(self.learning_rate * scale * &val_predictions);

                let val_dist = D::from_params(vp);
                let val_loss =
                    CensoredScorable::total_censored_score(&val_dist, yv, val_sample_weight);

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    self.best_val_loss_itr = Some(itr as usize);
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                }

                if let Some(rounds) = self.early_stopping_rounds {
                    if no_improvement_count >= rounds {
                        if self.verbose {
                            println!("== Early stopping achieved at iteration {}", itr);
                        }
                        break;
                    }
                }

                if self.should_print_verbose(itr) {
                    let train_loss =
                        CensoredScorable::total_censored_score(&dist, &y, sample_weight);
                    println!(
                        "[iter {}] train_loss={:.4} val_loss={:.4}",
                        itr, train_loss, val_loss
                    );
                }
            } else if self.should_print_verbose(itr) {
                let train_loss = CensoredScorable::total_censored_score(&dist, &y, sample_weight);
                println!("[iter {}] loss={:.4} scale={:.4}", itr, train_loss, scale);
            }
        }

        Ok(())
    }

    fn sample(
        &mut self,
        x: &Array2<f64>,
        y: &SurvivalData,
        params: &Array2<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Array2<f64>,
        SurvivalData,
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

        let row_idxs: Vec<usize> = if sample_size == n_samples {
            (0..n_samples).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut self.rng);
            indices.into_iter().take(sample_size).collect()
        };

        // Sample columns
        let col_size = if self.col_sample >= 1.0 {
            n_features
        } else if self.col_sample > 0.0 {
            ((n_features as f64) * self.col_sample) as usize
        } else {
            n_features
        };

        let col_idxs: Vec<usize> = if col_size == n_features {
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
        let y_sampled = SurvivalData {
            time: y.time.select(ndarray::Axis(0), &row_idxs),
            event: y.event.select(ndarray::Axis(0), &row_idxs),
        };
        let params_sampled = params.select(ndarray::Axis(0), &row_idxs);

        // Sample weights if provided
        let weights_sampled =
            sample_weight.map(|weights| weights.select(ndarray::Axis(0), &row_idxs));

        (
            row_idxs,
            col_idxs,
            x_sampled,
            y_sampled,
            params_sampled,
            weights_sampled,
        )
    }

    fn line_search(&self, resids: &Array2<f64>, start: &Array2<f64>, y: &SurvivalData) -> f64 {
        let mut scale = 1.0;
        let initial_score = CensoredScorable::total_censored_score(&D::from_params(start), y, None);

        // Scale up phase
        loop {
            if scale > 256.0 {
                break;
            }
            let scaled_resids = resids * (scale * 2.0);
            let next_params = start - &scaled_resids;
            let score =
                CensoredScorable::total_censored_score(&D::from_params(&next_params), y, None);
            if score >= initial_score || !score.is_finite() {
                break;
            }
            scale *= 2.0;
        }

        // Scale down phase
        loop {
            let scaled_resids = resids * scale;
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
            let score =
                CensoredScorable::total_censored_score(&D::from_params(&next_params), y, None);
            if score < initial_score && score.is_finite() {
                break;
            }
            scale *= 0.5;

            if scale < 1e-10 {
                break;
            }
        }

        scale
    }

    fn get_params(&self, x: &Array2<f64>) -> Array2<f64> {
        if x.nrows() == 0 {
            return Array2::zeros((0, 0));
        }

        let init_params = self.init_params.as_ref().unwrap();
        let n_params = init_params.len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        for (i, (learners, col_idx)) in self
            .base_models
            .iter()
            .zip(self.col_idxs.iter())
            .enumerate()
        {
            let scale = self.scalings[i];

            // Apply column subsampling during prediction to match training
            let predictions_cols: Vec<Array1<f64>> = if col_idx.len() == x.ncols() {
                learners.iter().map(|learner| learner.predict(x)).collect()
            } else {
                let x_subset = x.select(ndarray::Axis(1), col_idx);
                learners
                    .iter()
                    .map(|learner| learner.predict(&x_subset))
                    .collect()
            };

            let predictions = to_2d_array(predictions_cols);
            params -= &(self.learning_rate * scale * &predictions);
        }
        params
    }

    /// Get predicted distribution parameters.
    pub fn pred_param(&self, x: &Array2<f64>) -> Array2<f64> {
        self.get_params(x)
    }

    /// Get the predicted distribution.
    pub fn pred_dist(&self, x: &Array2<f64>) -> D {
        let params = self.get_params(x);
        D::from_params(&params)
    }

    /// Get point predictions (mean of the distribution).
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.pred_dist(x).predict()
    }

    /// Compute the survival function S(t) = P(T > t) at given times.
    pub fn predict_survival(&self, x: &Array2<f64>, _times: &Array1<f64>) -> Array2<f64> {
        // For now, just return predictions - full survival curves would need
        // distribution-specific implementation
        let preds = self.predict(x);
        let n_samples = x.nrows();
        let mut result = Array2::zeros((n_samples, 1));
        result.column_mut(0).assign(&preds);
        result
    }

    /// Get the best validation iteration (if early stopping was used).
    pub fn best_val_loss_itr(&self) -> Option<usize> {
        self.best_val_loss_itr
    }

    /// Get number of features the model was trained on.
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Determine if verbose output should be printed at the given iteration.
    /// Handles both integer intervals (verbose_eval >= 1.0) and percentage intervals (0 < verbose_eval < 1.0).
    fn should_print_verbose(&self, iteration: u32) -> bool {
        if !self.verbose || self.verbose_eval <= 0.0 {
            return false;
        }

        // Compute verbose_eval interval:
        // - If >= 1.0: use as integer iteration count (e.g., 100 = every 100 iterations)
        // - If 0 < x < 1.0: use as percentage of n_estimators (e.g., 0.1 = every 10%)
        let verbose_interval = if self.verbose_eval >= 1.0 {
            self.verbose_eval as u32
        } else {
            // Percentage of total iterations
            (self.n_estimators as f64 * self.verbose_eval).max(1.0) as u32
        };

        verbose_interval > 0 && iteration % verbose_interval == 0
    }
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

// ============================================================================
// Convenience type aliases
// ============================================================================

/// NGBSurvival with LogNormal distribution and LogScore (censored).
pub type NGBSurvivalLogNormal = NGBSurvival<LogNormal, LogScoreCensored, DecisionTreeLearner>;

/// NGBSurvival with Exponential distribution and LogScore (censored).
pub type NGBSurvivalExponential = NGBSurvival<Exponential, LogScoreCensored, DecisionTreeLearner>;

/// NGBSurvival with Weibull distribution and LogScore (censored).
pub type NGBSurvivalWeibull = NGBSurvival<Weibull, LogScoreCensored, DecisionTreeLearner>;

/// NGBSurvival with Weibull distribution and CRPScore (censored).
pub type NGBSurvivalWeibullCRPS = NGBSurvival<Weibull, CRPScoreCensored, DecisionTreeLearner>;

// ============================================================================
// Simplified constructors
// ============================================================================

impl NGBSurvival<LogNormal, LogScoreCensored, DecisionTreeLearner> {
    /// Create a new NGBSurvival with LogNormal distribution.
    pub fn lognormal(n_estimators: u32, learning_rate: f64) -> Self {
        Self::new(n_estimators, learning_rate, default_tree_learner())
    }
}

impl NGBSurvival<Exponential, LogScoreCensored, DecisionTreeLearner> {
    /// Create a new NGBSurvival with Exponential distribution.
    pub fn exponential(n_estimators: u32, learning_rate: f64) -> Self {
        Self::new(n_estimators, learning_rate, default_tree_learner())
    }
}

impl NGBSurvival<Weibull, LogScoreCensored, DecisionTreeLearner> {
    /// Create a new NGBSurvival with Weibull distribution and LogScore.
    pub fn weibull(n_estimators: u32, learning_rate: f64) -> Self {
        Self::new(n_estimators, learning_rate, default_tree_learner())
    }
}

impl NGBSurvival<Weibull, CRPScoreCensored, DecisionTreeLearner> {
    /// Create a new NGBSurvival with Weibull distribution and CRPScore.
    pub fn weibull_crps(n_estimators: u32, learning_rate: f64) -> Self {
        Self::new(n_estimators, learning_rate, default_tree_learner())
    }
}
