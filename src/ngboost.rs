/// Type alias for loss monitor functions
pub type LossMonitor<D> = Box<dyn Fn(&D, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync>;

use crate::dist::categorical::{Bernoulli, Categorical};
use crate::dist::normal::Normal;
use crate::dist::{ClassificationDistn, Distribution};
use crate::learners::{BaseLearner, DecisionTreeLearner, TrainedBaseLearner, default_tree_learner};
use crate::scores::{LogScore, Scorable, Score};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::borrow::Cow;
use std::marker::PhantomData;

use rayon::prelude::*;

/// Learning rate schedule for controlling step size during training.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate throughout training.
    #[default]
    Constant,
    /// Linear decay: lr * (1 - decay_rate * progress), clamped to min_lr.
    /// Default: decay_rate=0.7, min_lr=0.1
    Linear {
        decay_rate: f64,
        min_lr_fraction: f64,
    },
    /// Exponential decay: lr * exp(-decay_rate * progress).
    Exponential { decay_rate: f64 },
    /// Cosine annealing: lr * 0.5 * (1 + cos(pi * progress)).
    /// Proven effective for probabilistic models.
    Cosine,
    /// Cosine annealing with warm restarts.
    /// Restarts the schedule every `restart_period` iterations.
    CosineWarmRestarts { restart_period: u32 },
}

/// Line search method for finding optimal step size.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum LineSearchMethod {
    /// Binary search (original NGBoost method): scale up then scale down by 2x.
    /// Fast but may miss optimal step size.
    #[default]
    Binary,
    /// Golden section search: more accurate but slightly slower.
    /// Uses the golden ratio to efficiently narrow down the optimal step size.
    /// Generally finds better step sizes with fewer function evaluations.
    GoldenSection {
        /// Maximum number of iterations (default: 20)
        max_iters: usize,
    },
}

/// Golden ratio constant for golden section search.
const GOLDEN_RATIO: f64 = 1.618033988749895;

/// Training history result containing loss values at each iteration.
#[derive(Debug, Clone, Default)]
pub struct EvalsResult {
    /// Training loss at each iteration.
    pub train: Vec<f64>,
    /// Validation loss at each iteration (if validation data provided).
    pub val: Vec<f64>,
}

/// Hyperparameters for NGBoost models.
/// Used for `get_params()` and `set_params()` methods (sklearn-style interface).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NGBoostParams {
    /// Number of boosting iterations.
    pub n_estimators: u32,
    /// Learning rate (step size shrinkage).
    pub learning_rate: f64,
    /// Whether to use natural gradient (recommended).
    pub natural_gradient: bool,
    /// Fraction of samples to use per iteration (1.0 = all samples).
    pub minibatch_frac: f64,
    /// Fraction of features to use per iteration (1.0 = all features).
    pub col_sample: f64,
    /// Whether to print training progress.
    pub verbose: bool,
    /// Verbose evaluation interval.
    pub verbose_eval: f64,
    /// Tolerance for early stopping based on loss improvement.
    pub tol: f64,
    /// Number of rounds without improvement before stopping.
    pub early_stopping_rounds: Option<u32>,
    /// Fraction of training data to use for validation when no explicit validation set provided.
    pub validation_fraction: f64,
    /// Random seed for reproducibility.
    pub random_state: Option<u64>,
    /// Learning rate schedule.
    pub lr_schedule: LearningRateSchedule,
    /// Tikhonov regularization for Fisher matrix.
    pub tikhonov_reg: f64,
    /// Line search method.
    pub line_search_method: LineSearchMethod,
}

impl Default for NGBoostParams {
    fn default() -> Self {
        Self {
            n_estimators: 500,
            learning_rate: 0.01,
            natural_gradient: true,
            minibatch_frac: 1.0,
            col_sample: 1.0,
            verbose: false,
            verbose_eval: 1.0,
            tol: 1e-4,
            early_stopping_rounds: None,
            validation_fraction: 0.1,
            random_state: None,
            lr_schedule: LearningRateSchedule::Constant,
            tikhonov_reg: 0.0,
            line_search_method: LineSearchMethod::Binary,
        }
    }
}

pub struct NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    // ==========================================================================
    // Fields ordered by alignment for optimal memory layout (largest first)
    // ==========================================================================

    // --- 24-byte aligned (Vec = ptr + len + cap) ---
    /// Trained base models for each iteration and parameter.
    pub base_models: Vec<Vec<Box<dyn TrainedBaseLearner>>>,
    /// Scaling factors from line search at each iteration.
    pub scalings: Vec<f64>,
    /// Effective learning rates used at each iteration (for LR schedule support).
    pub effective_learning_rates: Vec<f64>,
    /// Column indices used at each iteration (for col_sample).
    pub col_idxs: Vec<Vec<usize>>,
    /// Training history containing loss values at each iteration.
    pub evals_result: EvalsResult,

    // --- 16-byte aligned (Option<Box<...>> with fat pointer) ---
    train_loss_monitor: Option<LossMonitor<D>>,
    val_loss_monitor: Option<LossMonitor<D>>,

    // --- 8-byte aligned (f64, usize, Option<usize>, Array1, StdRng, enums with f64) ---
    /// Learning rate (step size shrinkage).
    pub learning_rate: f64,
    /// Fraction of samples to use per iteration (1.0 = all samples).
    pub minibatch_frac: f64,
    /// Fraction of features to use per iteration (1.0 = all features).
    pub col_sample: f64,
    /// Interval for verbose output during training.
    /// - If >= 1.0: print every `verbose_eval` iterations (e.g., 100 means every 100 iterations)
    /// - If < 1.0 and > 0.0: print every `verbose_eval * n_estimators` iterations (e.g., 0.1 means every 10%)
    /// - If <= 0.0: no verbose output regardless of `verbose` setting
    pub verbose_eval: f64,
    /// Tolerance for early stopping based on gradient norm.
    pub tol: f64,
    /// Fraction of training data to use for validation when no explicit validation set provided.
    pub validation_fraction: f64,
    /// Tikhonov regularization parameter for stabilizing Fisher matrix inversion.
    /// Added to diagonal of Fisher Information Matrix: F + tikhonov_reg * I.
    /// Set to 0.0 to disable (default). Typical values: 1e-6 to 1e-3.
    pub tikhonov_reg: f64,
    /// Learning rate schedule for controlling step size during training.
    pub lr_schedule: LearningRateSchedule,
    /// Line search method for finding optimal step size.
    pub line_search_method: LineSearchMethod,
    /// Initial distribution parameters fitted from training data.
    pub init_params: Option<Array1<f64>>,
    /// Optional random seed for reproducibility. If None, uses entropy from the OS.
    random_state: Option<u64>,
    best_val_loss_itr: Option<usize>,
    n_features: Option<usize>,
    /// Random number generator (seeded for reproducibility).
    rng: StdRng,
    /// Base learner to use for fitting gradients.
    base_learner: B,

    // --- 4-byte aligned (u32) ---
    /// Number of boosting iterations.
    pub n_estimators: u32,
    /// Number of rounds without improvement before stopping.
    pub early_stopping_rounds: Option<u32>,

    // --- 1-byte aligned (bool) ---
    /// Whether to use natural gradient (recommended).
    pub natural_gradient: bool,
    /// Whether to print training progress.
    pub verbose: bool,
    /// Enable adaptive learning rate for better convergence (deprecated, use lr_schedule).
    pub adaptive_learning_rate: bool,

    // --- 0-byte (PhantomData) ---
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
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac: 1.0,
            col_sample: 1.0,
            verbose_eval: 100.0,
            tol: 1e-4,
            validation_fraction: 0.1,
            tikhonov_reg: 0.0,
            lr_schedule: LearningRateSchedule::Constant,
            line_search_method: LineSearchMethod::Binary,
            init_params: None,
            random_state: None,
            best_val_loss_itr: None,
            n_features: None,
            rng: StdRng::from_rng(&mut rand::rng()),
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds: None,
            // 1-byte aligned
            natural_gradient: true,
            verbose: false,
            adaptive_learning_rate: false,
            // 0-byte
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Create a new NGBoost model with a specific random seed for reproducibility.
    /// This is equivalent to Python's `random_state` parameter.
    pub fn with_seed(n_estimators: u32, learning_rate: f64, base_learner: B, seed: u64) -> Self {
        NGBoost {
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac: 1.0,
            col_sample: 1.0,
            verbose_eval: 100.0,
            tol: 1e-4,
            validation_fraction: 0.1,
            tikhonov_reg: 0.0,
            lr_schedule: LearningRateSchedule::Constant,
            line_search_method: LineSearchMethod::Binary,
            init_params: None,
            random_state: Some(seed),
            best_val_loss_itr: None,
            n_features: None,
            rng: StdRng::seed_from_u64(seed),
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds: None,
            // 1-byte aligned
            natural_gradient: true,
            verbose: false,
            adaptive_learning_rate: false,
            // 0-byte
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Set the random seed for reproducibility.
    /// Call this before `fit()` to ensure reproducible results.
    pub fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
        self.rng = StdRng::seed_from_u64(seed);
    }

    /// Get the current random state (seed), if set.
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }

    /// Returns a reference to the training history.
    pub fn evals_result(&self) -> &EvalsResult {
        &self.evals_result
    }

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
        adaptive_learning_rate: bool,
    ) -> Self {
        NGBoost {
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose_eval,
            tol,
            validation_fraction,
            tikhonov_reg: 0.0,
            lr_schedule: LearningRateSchedule::Constant,
            line_search_method: LineSearchMethod::Binary,
            init_params: None,
            random_state: None,
            best_val_loss_itr: None,
            n_features: None,
            rng: StdRng::from_rng(&mut rand::rng()),
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds,
            // 1-byte aligned
            natural_gradient,
            verbose,
            adaptive_learning_rate,
            // 0-byte
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Create NGBoost with all options including random seed for reproducibility.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options_seeded(
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
        adaptive_learning_rate: bool,
        random_state: Option<u64>,
    ) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        NGBoost {
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose_eval,
            tol,
            validation_fraction,
            tikhonov_reg: 0.0,
            lr_schedule: LearningRateSchedule::Constant,
            line_search_method: LineSearchMethod::Binary,
            init_params: None,
            random_state,
            best_val_loss_itr: None,
            n_features: None,
            rng,
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds,
            // 1-byte aligned
            natural_gradient,
            verbose,
            adaptive_learning_rate,
            // 0-byte
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Create NGBoost with advanced options including learning rate schedule and regularization.
    #[allow(clippy::too_many_arguments)]
    pub fn with_advanced_options(
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
        lr_schedule: LearningRateSchedule,
        tikhonov_reg: f64,
        line_search_method: LineSearchMethod,
    ) -> Self {
        NGBoost {
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose_eval,
            tol,
            validation_fraction,
            tikhonov_reg,
            lr_schedule,
            line_search_method,
            init_params: None,
            random_state: None,
            best_val_loss_itr: None,
            n_features: None,
            rng: StdRng::from_rng(&mut rand::rng()),
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds,
            // 1-byte aligned
            natural_gradient,
            verbose,
            adaptive_learning_rate: false,
            // 0-byte
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    /// Create NGBoost with all advanced options including random seed for reproducibility.
    #[allow(clippy::too_many_arguments)]
    pub fn with_full_options(
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
        lr_schedule: LearningRateSchedule,
        tikhonov_reg: f64,
        line_search_method: LineSearchMethod,
        random_state: Option<u64>,
    ) -> Self {
        let rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut rand::rng()),
        };
        NGBoost {
            // 24-byte aligned
            base_models: Vec::new(),
            scalings: Vec::new(),
            effective_learning_rates: Vec::new(),
            col_idxs: Vec::new(),
            evals_result: EvalsResult::default(),
            // 16-byte aligned
            train_loss_monitor: None,
            val_loss_monitor: None,
            // 8-byte aligned
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose_eval,
            tol,
            validation_fraction,
            tikhonov_reg,
            lr_schedule,
            line_search_method,
            init_params: None,
            random_state,
            best_val_loss_itr: None,
            n_features: None,
            rng,
            base_learner,
            // 4-byte aligned
            n_estimators,
            early_stopping_rounds,
            // 1-byte aligned
            natural_gradient,
            verbose,
            adaptive_learning_rate: false,
            // 0-byte
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

    /// Validates hyperparameters before fitting.
    /// Returns an error message if any hyperparameter is invalid.
    fn validate_hyperparameters(&self) -> Result<(), &'static str> {
        if self.n_estimators == 0 {
            return Err("n_estimators must be greater than 0");
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive");
        }
        if self.learning_rate > 10.0 {
            return Err("learning_rate > 10.0 is likely a mistake");
        }
        if self.minibatch_frac <= 0.0 || self.minibatch_frac > 1.0 {
            return Err("minibatch_frac must be in (0, 1]");
        }
        if self.col_sample <= 0.0 || self.col_sample > 1.0 {
            return Err("col_sample must be in (0, 1]");
        }
        if self.tol < 0.0 {
            return Err("tol must be non-negative");
        }
        if self.validation_fraction < 0.0 || self.validation_fraction >= 1.0 {
            return Err("validation_fraction must be in [0, 1)");
        }
        if self.tikhonov_reg < 0.0 {
            return Err("tikhonov_reg must be non-negative");
        }

        // Validate learning rate schedule parameters
        match self.lr_schedule {
            LearningRateSchedule::Linear {
                decay_rate,
                min_lr_fraction,
            } => {
                if decay_rate < 0.0 || decay_rate > 1.0 {
                    return Err("Linear schedule decay_rate must be in [0, 1]");
                }
                if min_lr_fraction < 0.0 || min_lr_fraction > 1.0 {
                    return Err("Linear schedule min_lr_fraction must be in [0, 1]");
                }
            }
            LearningRateSchedule::Exponential { decay_rate } => {
                if decay_rate < 0.0 {
                    return Err("Exponential schedule decay_rate must be non-negative");
                }
            }
            LearningRateSchedule::CosineWarmRestarts { restart_period } => {
                if restart_period == 0 {
                    return Err("CosineWarmRestarts restart_period must be > 0");
                }
            }
            _ => {}
        }

        // Validate line search parameters
        if let LineSearchMethod::GoldenSection { max_iters } = self.line_search_method {
            if max_iters == 0 {
                return Err("GoldenSection max_iters must be > 0");
            }
        }

        Ok(())
    }

    /// Internal fit implementation that can optionally reset state.
    fn fit_internal(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
        reset_state: bool,
    ) -> Result<(), &'static str> {
        // Validate hyperparameters first
        self.validate_hyperparameters()?;

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
            self.effective_learning_rates.clear();
            self.col_idxs.clear();
            self.best_val_loss_itr = None;
            self.evals_result = EvalsResult::default();
        }
        self.n_features = Some(x.ncols());

        // Handle automatic validation split if early stopping is enabled
        let (x_train, y_train, x_val_auto, y_val_auto, train_weight_auto, val_weight_auto) = if self
            .early_stopping_rounds
            .is_some()
            && x_val.is_none()
            && y_val.is_none()
            && self.validation_fraction > 0.0
            && self.validation_fraction < 1.0
        {
            // Split training data into training and validation sets
            // Shuffle indices first to match sklearn's train_test_split behavior
            let n_samples = x.nrows();
            let n_val = ((n_samples as f64) * self.validation_fraction) as usize;
            let n_train = n_samples - n_val;

            // Shuffle indices for random split (matches Python's train_test_split)
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..indices.len()).rev() {
                let j = self.rng.random_range(0..=i);
                indices.swap(i, j);
            }

            let train_indices: Vec<usize> = indices[0..n_train].to_vec();
            let val_indices: Vec<usize> = indices[n_train..].to_vec();

            let x_train = x.select(ndarray::Axis(0), &train_indices);
            let y_train = y.select(ndarray::Axis(0), &train_indices);
            let x_val_auto = Some(x.select(ndarray::Axis(0), &val_indices));
            let y_val_auto = Some(y.select(ndarray::Axis(0), &val_indices));

            // Split sample weights along with the data (matches Python's train_test_split)
            let train_weight_auto =
                sample_weight.map(|w| w.select(ndarray::Axis(0), &train_indices));
            let val_weight_auto = sample_weight.map(|w| w.select(ndarray::Axis(0), &val_indices));

            (
                Cow::Owned(x_train),
                Cow::Owned(y_train),
                x_val_auto,
                y_val_auto,
                train_weight_auto,
                val_weight_auto,
            )
        } else {
            (
                Cow::Borrowed(x),
                Cow::Borrowed(y),
                x_val.cloned(),
                y_val.cloned(),
                None,
                None,
            )
        };

        // Use the automatically split or provided data
        // For sample weights: if we auto-split, use the split weights;
        // otherwise use the original references passed in.
        let x_train = &*x_train;
        let y_train = &*y_train;
        let x_val = x_val_auto.as_ref().or(x_val);
        let y_val = y_val_auto.as_ref().or(y_val);
        let sample_weight: Option<&Array1<f64>> = train_weight_auto.as_ref().or(sample_weight);
        let val_sample_weight: Option<&Array1<f64>> =
            val_weight_auto.as_ref().or(val_sample_weight);

        // Validate validation data if provided
        if let (Some(xv), Some(yv)) = (x_val, y_val) {
            if xv.nrows() != yv.len() {
                return Err("Number of samples in validation X and y must match");
            }
            if xv.ncols() != x_train.ncols() {
                return Err("Number of features in training and validation data must match");
            }
        }

        // Match Python: always re-fit init_params from the current training data.
        // Python's partial_fit calls fit_init_params_to_marginal(Y) every time.
        self.init_params = Some(D::fit(&y_train));
        let n_params = self.init_params.as_ref().unwrap().len();

        // For partial_fit, replay all existing base models to reconstruct current params
        // (matches Python's pred_param(X) call at start of partial_fit)
        let mut params = if !reset_state && !self.base_models.is_empty() {
            self.get_params_at(&x_train, None)
        } else {
            let mut p = Array2::from_elem((x_train.nrows(), n_params), 0.0);
            let init_params = self.init_params.as_ref().unwrap();
            p.outer_iter_mut()
                .for_each(|mut row| row.assign(init_params));
            p
        };

        // Prepare validation params if validation data is provided
        let mut val_params = if let (Some(xv), Some(_yv)) = (x_val, y_val) {
            if !reset_state && !self.base_models.is_empty() {
                Some(self.get_params_at(xv, None))
            } else {
                let mut v_params = Array2::from_elem((xv.nrows(), n_params), 0.0);
                let init_params = self.init_params.as_ref().unwrap();
                v_params
                    .outer_iter_mut()
                    .for_each(|mut row| row.assign(init_params));
                Some(v_params)
            }
        } else {
            None
        };

        let mut best_val_loss = f64::INFINITY;
        let mut best_iter = 0;
        let mut val_loss_list: Vec<f64> = Vec::new();

        for itr in 0..self.n_estimators {
            // Sample data for this iteration
            let (col_idxs, x_sampled, y_sampled, params_sampled, weight_sampled) =
                self.sample(&x_train, &y_train, &params, sample_weight);
            self.col_idxs.push(col_idxs);

            // Dereference Cow to get &Array references (zero-cost for Borrowed)
            let x_sampled_ref = &*x_sampled;
            let y_sampled_ref = &*y_sampled;
            let params_sampled_ref = &*params_sampled;
            let weight_ref: Option<&Array1<f64>> = weight_sampled.as_ref().map(|w| w.as_ref());

            // Build distribution from minibatch params (matches Python: D = Manifold(P_batch.T))
            let batch_dist = D::from_params(params_sampled_ref);

            // Compute training loss on minibatch BEFORE update (matches Python)
            let train_loss = if let Some(monitor) = &self.train_loss_monitor {
                monitor(&batch_dist, y_sampled_ref, weight_ref)
            } else {
                Scorable::total_score(&batch_dist, y_sampled_ref, weight_ref)
            };
            self.evals_result.train.push(train_loss);

            // Compute gradients on minibatch (matches Python: D.grad(Y_batch, ...))
            let grads = if self.natural_gradient && self.tikhonov_reg > 0.0 {
                // Fast path: diagonal metric uses fused d_score + diagonal_metric
                if Scorable::<S>::is_diagonal_metric(&batch_dist) {
                    Scorable::<S>::diagonal_natural_grad(
                        &batch_dist,
                        y_sampled_ref,
                        self.tikhonov_reg,
                    )
                } else {
                    let standard_grad = Scorable::d_score(&batch_dist, y_sampled_ref);
                    let metric = Scorable::metric(&batch_dist);
                    crate::scores::natural_gradient_regularized(
                        &standard_grad,
                        &metric,
                        self.tikhonov_reg,
                    )
                }
            } else {
                Scorable::grad(&batch_dist, y_sampled_ref, self.natural_gradient)
            };

            // Fit base learners for each parameter in parallel using rayon
            let grad_cols: Vec<Array1<f64>> =
                (0..n_params).map(|j| grads.column(j).to_owned()).collect();

            let fit_results: Vec<Result<(Box<dyn TrainedBaseLearner>, Array1<f64>), &'static str>> = {
                let learners: Vec<B> = (0..n_params).map(|_| self.base_learner.clone()).collect();
                learners
                    .into_par_iter()
                    .zip(grad_cols.into_par_iter())
                    .map(|(learner, grad_j)| {
                        let fitted =
                            learner.fit_with_weights(x_sampled_ref, &grad_j, weight_ref)?;
                        let preds = fitted.predict(x_sampled_ref);
                        Ok((fitted, preds))
                    })
                    .collect()
            };

            // Unpack results, propagating any errors
            let mut fitted_learners: Vec<Box<dyn TrainedBaseLearner>> =
                Vec::with_capacity(n_params);
            let mut predictions_cols: Vec<Array1<f64>> = Vec::with_capacity(n_params);
            for result in fit_results {
                let (fitted, preds) = result?;
                fitted_learners.push(fitted);
                predictions_cols.push(preds);
            }

            let proj_grad = to_2d_array(predictions_cols);

            // Pass precomputed train_loss to avoid redundant from_params + total_score
            // in line search. Only valid when no custom train_loss_monitor changes the score.
            let precomputed_score = if self.train_loss_monitor.is_none() {
                Some(train_loss)
            } else {
                None
            };
            let scale = self.line_search(
                &proj_grad,
                params_sampled_ref,
                y_sampled_ref,
                weight_ref,
                precomputed_score,
            );
            self.scalings.push(scale);
            self.base_models.push(fitted_learners);

            // Apply learning rate schedule and store it for prediction-time replay
            let progress = itr as f64 / self.n_estimators as f64;
            let effective_learning_rate = self.compute_learning_rate(itr, progress);
            self.effective_learning_rates.push(effective_learning_rate);

            // Update parameters for ALL training samples (matches Python)
            // Fused predict + scale + subtract avoids intermediate Array2 allocations
            let col_idxs_ref = self.col_idxs.last().unwrap();
            let fitted_learners = self.base_models.last().unwrap();
            let factor = effective_learning_rate * scale;
            Self::predict_and_update_params(
                fitted_learners,
                &x_train,
                col_idxs_ref,
                &mut params,
                factor,
            );

            // Update validation parameters if validation data is provided
            let mut val_loss = 0.0;
            if let (Some(xv), Some(yv), Some(vp)) = (x_val, y_val, val_params.as_mut()) {
                Self::predict_and_update_params(fitted_learners, xv, col_idxs_ref, vp, factor);

                let val_dist = D::from_params(vp);
                val_loss = if let Some(monitor) = &self.val_loss_monitor {
                    monitor(&val_dist, yv, val_sample_weight)
                } else {
                    Scorable::total_score(&val_dist, yv, val_sample_weight)
                };

                self.evals_result.val.push(val_loss);
                val_loss_list.push(val_loss);

                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    best_iter = itr;
                    self.best_val_loss_itr = Some(itr as usize);
                }

                // Early stopping: match Python's sliding-window logic
                // Stop if len(val_loss_list) > early_stopping_rounds AND
                // best_val_loss < min(val_loss_list[-early_stopping_rounds:])
                if let Some(rounds) = self.early_stopping_rounds {
                    let rounds = rounds as usize;
                    if val_loss_list.len() > rounds {
                        let recent_min = val_loss_list[val_loss_list.len() - rounds..]
                            .iter()
                            .cloned()
                            .fold(f64::INFINITY, f64::min);
                        if best_val_loss < recent_min {
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
                }
            }

            // Verbose logging
            if self.should_print_verbose(itr) {
                let grad_norm: f64 = grads
                    .rows()
                    .into_iter()
                    .map(|row| {
                        let norm_sq: f64 = row.iter().map(|&x| x * x).sum();
                        norm_sq.sqrt()
                    })
                    .sum::<f64>()
                    / grads.nrows() as f64
                    * scale;
                if x_val.is_some() {
                    println!(
                        "[iter {}] loss={:.4} val_loss={:.4} scale={:.4} norm={:.4}",
                        itr, train_loss, val_loss, scale, grad_norm
                    );
                } else {
                    println!(
                        "[iter {}] loss={:.4} scale={:.4} norm={:.4}",
                        itr, train_loss, scale, grad_norm
                    );
                }
            }

            // Gradient-norm convergence check (matches Python)
            // Uses proj_grad (base learner predictions on minibatch), not raw grads
            let proj_grad_norm: f64 = proj_grad
                .rows()
                .into_iter()
                .map(|row| {
                    let norm_sq: f64 = row.iter().map(|&x| x * x).sum();
                    norm_sq.sqrt()
                })
                .sum::<f64>()
                / proj_grad.nrows() as f64;
            if proj_grad_norm < self.tol {
                if self.verbose {
                    println!("== Quitting at iteration / GRAD {}", itr);
                }
                break;
            }
        }

        Ok(())
    }

    fn sample<'a>(
        &mut self,
        x: &'a Array2<f64>,
        y: &'a Array1<f64>,
        params: &'a Array2<f64>,
        sample_weight: Option<&'a Array1<f64>>,
    ) -> (
        Vec<usize>,
        Cow<'a, Array2<f64>>,
        Cow<'a, Array1<f64>>,
        Cow<'a, Array2<f64>>,
        Option<Cow<'a, Array1<f64>>>,
    ) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let no_row_sampling = self.minibatch_frac >= 1.0;
        let no_col_sampling = self.col_sample >= 1.0;

        // Fast path: no sampling at all — borrow original data, zero copies
        if no_row_sampling && no_col_sampling {
            let col_idxs: Vec<usize> = (0..n_features).collect();
            return (
                col_idxs,
                Cow::Borrowed(x),
                Cow::Borrowed(y),
                Cow::Borrowed(params),
                sample_weight.map(Cow::Borrowed),
            );
        }

        // Sample rows (minibatch)
        let sample_size = if no_row_sampling {
            n_samples
        } else {
            ((n_samples as f64) * self.minibatch_frac) as usize
        };

        // Uniform random sampling without replacement (matches Python's np.random.choice behavior)
        let row_idxs: Vec<usize> = if sample_size == n_samples {
            (0..n_samples).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..indices.len()).rev() {
                let j = self.rng.random_range(0..=i);
                indices.swap(i, j);
            }
            indices.into_iter().take(sample_size).collect()
        };

        // Sample columns (matches Python: max(1, int(col_sample * n_features)))
        let col_size = if no_col_sampling {
            n_features
        } else if self.col_sample > 0.0 {
            (((n_features as f64) * self.col_sample) as usize).max(1)
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
        let x_sampled = if col_size == n_features {
            x.select(ndarray::Axis(0), &row_idxs)
        } else {
            let mut result = Array2::zeros((row_idxs.len(), col_idxs.len()));
            for (new_row, &old_row) in row_idxs.iter().enumerate() {
                for (new_col, &old_col) in col_idxs.iter().enumerate() {
                    result[[new_row, new_col]] = x[[old_row, old_col]];
                }
            }
            result
        };
        let y_sampled = y.select(ndarray::Axis(0), &row_idxs);
        let params_sampled = params.select(ndarray::Axis(0), &row_idxs);

        let sample_weights_sampled =
            sample_weight.map(|weights| Cow::Owned(weights.select(ndarray::Axis(0), &row_idxs)));

        (
            col_idxs,
            Cow::Owned(x_sampled),
            Cow::Owned(y_sampled),
            Cow::Owned(params_sampled),
            sample_weights_sampled,
        )
    }

    /// Compute predictions from a set of learners on input data.
    /// Handles column subsampling if col_idxs doesn't match all columns.
    /// Returns an Array2 with shape (n_samples, n_learners).
    #[inline]
    fn predict_from_learners(
        learners: &[Box<dyn TrainedBaseLearner>],
        x: &Array2<f64>,
        col_idxs: &[usize],
    ) -> Array2<f64> {
        if learners.is_empty() {
            return Array2::zeros((x.nrows(), 0));
        }

        let n_samples = x.nrows();
        let n_params = learners.len();

        // Pre-allocate the result array
        let mut result = Array2::zeros((n_samples, n_params));

        if col_idxs.len() == x.ncols() {
            // No column subsampling - predict directly
            for (j, learner) in learners.iter().enumerate() {
                let preds = learner.predict(x);
                result.column_mut(j).assign(&preds);
            }
        } else {
            // Column subsampling - create subset once and reuse
            let x_subset = x.select(ndarray::Axis(1), col_idxs);
            for (j, learner) in learners.iter().enumerate() {
                let preds = learner.predict(&x_subset);
                result.column_mut(j).assign(&preds);
            }
        }

        result
    }

    /// Fused predict + scale + subtract: params -= factor * predictions.
    /// Avoids allocating the full predictions Array2 and intermediate scaled copies.
    /// Instead, predicts one parameter at a time and updates params in-place.
    #[inline]
    fn predict_and_update_params(
        learners: &[Box<dyn TrainedBaseLearner>],
        x: &Array2<f64>,
        col_idxs: &[usize],
        params: &mut Array2<f64>,
        factor: f64,
    ) {
        if learners.is_empty() {
            return;
        }

        if col_idxs.len() == x.ncols() {
            for (j, learner) in learners.iter().enumerate() {
                let preds = learner.predict(x);
                let mut col = params.column_mut(j);
                ndarray::Zip::from(&mut col)
                    .and(&preds)
                    .for_each(|p, &pred| {
                        *p -= factor * pred;
                    });
            }
        } else {
            let x_subset = x.select(ndarray::Axis(1), col_idxs);
            for (j, learner) in learners.iter().enumerate() {
                let preds = learner.predict(&x_subset);
                let mut col = params.column_mut(j);
                ndarray::Zip::from(&mut col)
                    .and(&preds)
                    .for_each(|p, &pred| {
                        *p -= factor * pred;
                    });
            }
        }
    }

    fn get_params(&self, x: &Array2<f64>) -> Array2<f64> {
        self.get_params_at(x, None)
    }

    fn get_params_at(&self, x: &Array2<f64>, max_iter: Option<usize>) -> Array2<f64> {
        if x.nrows() == 0 {
            return Array2::zeros((0, 0));
        }

        let init_params = self
            .init_params
            .as_ref()
            .expect("Model has not been fitted. Call fit() before predict().");
        let n_params = init_params.len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        // Match Python: `if max_iter and i == max_iter` — in Python, max_iter=0
        // is falsy, so it iterates ALL models. Only a positive max_iter limits.
        let n_iters = match max_iter {
            Some(m) if m > 0 => m.min(self.base_models.len()),
            _ => self.base_models.len(),
        };

        for (i, (learners, col_idx)) in self
            .base_models
            .iter()
            .zip(self.col_idxs.iter())
            .enumerate()
            .take(n_iters)
        {
            let scale = self.scalings[i];
            let lr = if i < self.effective_learning_rates.len() {
                self.effective_learning_rates[i]
            } else {
                self.learning_rate
            };
            let factor = lr * scale;
            Self::predict_and_update_params(learners, x, col_idx, &mut params, factor);
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

    /// Returns staged predictions (predictions at each boosting iteration).
    /// Uses incremental parameter accumulation — O(n_estimators) total work
    /// instead of O(n_estimators²).
    pub fn staged_predict(&self, x: &Array2<f64>) -> Vec<Array1<f64>> {
        self.staged_pred_dist(x)
            .into_iter()
            .map(|d| d.predict())
            .collect()
    }

    /// Returns staged distribution predictions.
    /// Uses incremental parameter accumulation — O(n_estimators) total work
    /// instead of O(n_estimators²).
    pub fn staged_pred_dist(&self, x: &Array2<f64>) -> Vec<D> {
        let init_params = match self.init_params.as_ref() {
            Some(p) => p,
            None => return Vec::new(),
        };
        let n_params = init_params.len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(init_params));

        let mut results = Vec::with_capacity(self.base_models.len());

        for (i, (learners, col_idx)) in self
            .base_models
            .iter()
            .zip(self.col_idxs.iter())
            .enumerate()
        {
            let scale = self.scalings[i];
            let lr = if i < self.effective_learning_rates.len() {
                self.effective_learning_rates[i]
            } else {
                self.learning_rate
            };
            let factor = lr * scale;
            Self::predict_and_update_params(learners, x, col_idx, &mut params, factor);
            results.push(D::from_params(&params));
        }

        results
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

    /// Compute the effective learning rate for the given iteration using the configured schedule.
    fn compute_learning_rate(&self, iteration: u32, progress: f64) -> f64 {
        // Legacy adaptive_learning_rate takes precedence for backward compatibility
        if self.adaptive_learning_rate {
            return self.learning_rate * (1.0 - 0.7 * progress).max(0.1);
        }

        match self.lr_schedule {
            LearningRateSchedule::Constant => self.learning_rate,
            LearningRateSchedule::Linear {
                decay_rate,
                min_lr_fraction,
            } => self.learning_rate * (1.0 - decay_rate * progress).max(min_lr_fraction),
            LearningRateSchedule::Exponential { decay_rate } => {
                self.learning_rate * (-decay_rate * progress).exp()
            }
            LearningRateSchedule::Cosine => {
                self.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            LearningRateSchedule::CosineWarmRestarts { restart_period } => {
                let period_progress = (iteration % restart_period) as f64 / restart_period as f64;
                self.learning_rate * 0.5 * (1.0 + (std::f64::consts::PI * period_progress).cos())
            }
        }
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
        let n_iters = self.base_models.len();

        // Collect per-param, per-iteration importance vectors mapped to full feature space
        // all_params_importances[param][iter] = Array1<f64> of length n_features
        let mut all_params_importances: Vec<Vec<Array1<f64>>> =
            (0..n_params).map(|_| Vec::with_capacity(n_iters)).collect();

        for (iter_idx, learners) in self.base_models.iter().enumerate() {
            let col_idxs = if iter_idx < self.col_idxs.len() {
                &self.col_idxs[iter_idx]
            } else {
                // No column subsampling info — assume all columns
                &vec![]
            };

            for (param_idx, learner) in learners.iter().enumerate() {
                let mut full_importance = Array1::zeros(n_features);

                if let Some(tree_importance) = learner.feature_importances() {
                    // Map subsampled feature importances back to full feature space
                    if col_idxs.is_empty() || col_idxs.len() == n_features {
                        // No column subsampling — use directly
                        let len = tree_importance.len().min(n_features);
                        for j in 0..len {
                            full_importance[j] = tree_importance[j];
                        }
                    } else {
                        // Column subsampling — map back using col_idxs
                        for (local_idx, &global_idx) in col_idxs.iter().enumerate() {
                            if local_idx < tree_importance.len() && global_idx < n_features {
                                full_importance[global_idx] = tree_importance[local_idx];
                            }
                        }
                    }
                }

                if param_idx < n_params {
                    all_params_importances[param_idx].push(full_importance);
                }
            }
        }

        // Weighted average of importance by tree scaling factors (matches Python's np.average)
        let mut importances = Array2::zeros((n_params, n_features));
        let weight_sum: f64 = self.scalings.iter().sum();

        for param_idx in 0..n_params {
            if weight_sum > 0.0 {
                for (iter_idx, imp) in all_params_importances[param_idx].iter().enumerate() {
                    let w = if iter_idx < self.scalings.len() {
                        self.scalings[iter_idx]
                    } else {
                        0.0
                    };
                    for j in 0..n_features {
                        importances[[param_idx, j]] += imp[j] * w;
                    }
                }
                for j in 0..n_features {
                    importances[[param_idx, j]] /= weight_sum;
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

    /// Compute mean of per-row L2 norms of (scale * resids), matching Python's
    /// `np.mean(np.linalg.norm(scaled_resids, axis=1))`.
    fn mean_row_norms(&self, resids: &Array2<f64>, scale: f64) -> f64 {
        let n_rows = resids.nrows();
        if n_rows == 0 {
            return 0.0;
        }
        let abs_scale = scale.abs();
        let sum_norms: f64 = resids
            .rows()
            .into_iter()
            .map(|row| {
                let row_norm_sq: f64 = row.iter().map(|&x| x * x).sum();
                row_norm_sq.sqrt()
            })
            .sum();
        abs_scale * sum_norms / n_rows as f64
    }

    fn line_search(
        &self,
        resids: &Array2<f64>,
        start: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        initial_score: Option<f64>,
    ) -> f64 {
        match self.line_search_method {
            LineSearchMethod::Binary => {
                self.line_search_binary(resids, start, y, sample_weight, initial_score)
            }
            LineSearchMethod::GoldenSection { max_iters } => self.line_search_golden_section(
                resids,
                start,
                y,
                sample_weight,
                max_iters,
                initial_score,
            ),
        }
    }

    /// Binary line search (original NGBoost method).
    /// Optimized to reuse a single buffer for next_params computation and cache distributions.
    fn line_search_binary(
        &self,
        resids: &Array2<f64>,
        start: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        precomputed_initial_score: Option<f64>,
    ) -> f64 {
        let mut scale = 1.0;
        let initial_score = precomputed_initial_score
            .unwrap_or_else(|| Scorable::total_score(&D::from_params(start), y, sample_weight));

        // Pre-allocate buffer for next_params - reused across all iterations
        let mut next_params = Array2::zeros(start.raw_dim());

        // Helper to compute next_params in-place: next_params = start - scale * resids
        // Uses efficient Zip operation to avoid intermediate allocations
        let compute_next_params = |next_params: &mut Array2<f64>, scale: f64| {
            ndarray::Zip::from(next_params)
                .and(start)
                .and(resids)
                .for_each(|np, &s, &r| {
                    *np = s - scale * r;
                });
        };

        // Helper to compute score - distribution is constructed only once per scale value
        let compute_score_at_scale = |next_params: &mut Array2<f64>, scale: f64| -> f64 {
            compute_next_params(next_params, scale);
            Scorable::total_score(&D::from_params(next_params), y, sample_weight)
        };

        // Scale up phase: test at current scale, then double if good.
        // Matches Python: exits with scale = first value that failed.
        loop {
            let score = compute_score_at_scale(&mut next_params, scale);
            if !score.is_finite() || score > initial_score || scale > 256.0 {
                break;
            }
            scale *= 2.0;
        }

        // Scale down phase: find a step that actually reduces loss
        loop {
            let norm = self.mean_row_norms(resids, scale);
            if norm < self.tol {
                break;
            }

            let score = compute_score_at_scale(&mut next_params, scale);
            if score.is_finite() && score < initial_score {
                break;
            }
            scale *= 0.5;
        }

        scale
    }

    /// Golden section line search for more accurate step size.
    /// Uses the golden ratio to efficiently narrow down the optimal step size.
    /// Optimized to reuse a single buffer for next_params computation and minimize
    /// distribution constructions by caching scores at interior points.
    fn line_search_golden_section(
        &self,
        resids: &Array2<f64>,
        start: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        max_iters: usize,
        precomputed_initial_score: Option<f64>,
    ) -> f64 {
        // Pre-allocate buffer for next_params - reused across all iterations
        let mut next_params = Array2::zeros(start.raw_dim());

        // Helper to compute score at a given scale, reusing the buffer
        let mut compute_score = |scale: f64| -> f64 {
            ndarray::Zip::from(&mut next_params)
                .and(start)
                .and(resids)
                .for_each(|np, &s, &r| {
                    *np = s - scale * r;
                });
            Scorable::total_score(&D::from_params(&next_params), y, sample_weight)
        };

        let initial_score = precomputed_initial_score.unwrap_or_else(|| compute_score(0.0));

        // First, find a reasonable upper bound by scaling up
        let mut upper = 1.0;
        while upper < 256.0 {
            let score = compute_score(upper * 2.0);
            if score >= initial_score || !score.is_finite() {
                break;
            }
            upper *= 2.0;
        }

        // Golden section search between 0 and upper
        let mut a = 0.0;
        let mut b = upper;
        let inv_phi = 1.0 / GOLDEN_RATIO;

        // Initial interior points
        let mut c = b - (b - a) * inv_phi;
        let mut d = a + (b - a) * inv_phi;
        let mut fc = compute_score(c);
        let mut fd = compute_score(d);

        for _ in 0..max_iters {
            if (b - a).abs() < self.tol {
                break;
            }

            if fc < fd {
                // Minimum is in [a, d]
                b = d;
                d = c;
                fd = fc;
                c = b - (b - a) * inv_phi;
                fc = compute_score(c);
            } else {
                // Minimum is in [c, b]
                a = c;
                c = d;
                fc = fd;
                d = a + (b - a) * inv_phi;
                fd = compute_score(d);
            }
        }

        // Return the midpoint of the final interval
        let scale = (a + b) / 2.0;

        // Verify the scale actually reduces loss, otherwise fall back
        let final_score = compute_score(scale);
        if final_score < initial_score && final_score.is_finite() {
            scale
        } else {
            // Fall back to a small step
            1.0
        }
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
            effective_learning_rates: self.effective_learning_rates.clone(),
            col_idxs: self.col_idxs.clone(),
            best_val_loss_itr: self.best_val_loss_itr,
            base_models: serialized_base_models,
            lr_schedule: self.lr_schedule,
            tikhonov_reg: self.tikhonov_reg,
            line_search_method: self.line_search_method,
            n_features: self.n_features,
            random_state: self.random_state,
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
        let mut model = Self::with_options_seeded(
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
            serialized.random_state,
        );

        // Restore trained state
        if let Some(init_params) = serialized.init_params {
            model.init_params = Some(Array1::from(init_params));
        }
        model.scalings = serialized.scalings;
        model.effective_learning_rates = serialized.effective_learning_rates;
        model.col_idxs = serialized.col_idxs;
        model.best_val_loss_itr = serialized.best_val_loss_itr;

        // Restore advanced options (added in v0.3)
        model.lr_schedule = serialized.lr_schedule;
        model.tikhonov_reg = serialized.tikhonov_reg;
        model.line_search_method = serialized.line_search_method;
        model.n_features = serialized.n_features;

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
    /// Verbose evaluation interval. Can be >= 1.0 for iteration count or 0 < x < 1.0 for percentage.
    pub verbose_eval: f64,
    pub tol: f64,
    pub early_stopping_rounds: Option<u32>,
    pub validation_fraction: f64,
    pub init_params: Option<Vec<f64>>,
    pub scalings: Vec<f64>,
    /// Effective learning rates per iteration (for LR schedule replay)
    #[serde(default)]
    pub effective_learning_rates: Vec<f64>,
    pub col_idxs: Vec<Vec<usize>>,
    pub best_val_loss_itr: Option<usize>,
    /// Serialized base models - each inner Vec contains learners for each parameter
    pub base_models: Vec<Vec<crate::learners::SerializableTrainedLearner>>,
    /// Learning rate schedule (added in v0.3)
    #[serde(default)]
    pub lr_schedule: LearningRateSchedule,
    /// Tikhonov regularization parameter (added in v0.3)
    #[serde(default)]
    pub tikhonov_reg: f64,
    /// Line search method (added in v0.3)
    #[serde(default)]
    pub line_search_method: LineSearchMethod,
    /// Number of features the model was trained on (added in v0.3)
    #[serde(default)]
    pub n_features: Option<usize>,
    /// Random state for reproducibility (added in v0.3)
    #[serde(default)]
    pub random_state: Option<u64>,
}

fn to_2d_array(cols: Vec<Array1<f64>>) -> Array2<f64> {
    if cols.is_empty() {
        return Array2::zeros((0, 0));
    }
    let nrows = cols[0].len();
    let ncols = cols.len();
    // Build directly in row-major order — avoids the column-major → row-major copy
    let mut result = Array2::zeros((nrows, ncols));
    for (j, col) in cols.iter().enumerate() {
        result.column_mut(j).assign(col);
    }
    result
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
    pub fn staged_predict(&self, x: &Array2<f64>) -> Vec<Array1<f64>> {
        self.model.staged_predict(x)
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> Normal {
        self.model.pred_dist(x)
    }

    /// Get the predicted distribution up to a specific iteration
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> Normal {
        self.model.pred_dist_at(x, max_iter)
    }

    /// Returns staged distribution predictions
    pub fn staged_pred_dist(&self, x: &Array2<f64>) -> Vec<Normal> {
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
        verbose_eval: f64,
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
        verbose_eval: f64,
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

    /// Get the training history (losses at each iteration).
    pub fn evals_result(&self) -> &EvalsResult {
        self.model.evals_result()
    }

    /// Set the random seed for reproducibility.
    pub fn set_random_state(&mut self, seed: u64) {
        self.model.set_random_state(seed);
    }

    /// Get the current random state (seed), if set.
    pub fn random_state(&self) -> Option<u64> {
        self.model.random_state()
    }

    /// Fit with sample weights.
    pub fn fit_with_weights(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, None, None, sample_weight, None)
    }

    /// Fit with sample weights and validation data.
    pub fn fit_with_weights_and_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, x_val, y_val, sample_weight, val_sample_weight)
    }

    /// Get all hyperparameters as a struct.
    pub fn get_params(&self) -> NGBoostParams {
        NGBoostParams {
            n_estimators: self.model.n_estimators,
            learning_rate: self.model.learning_rate,
            natural_gradient: self.model.natural_gradient,
            minibatch_frac: self.model.minibatch_frac,
            col_sample: self.model.col_sample,
            verbose: self.model.verbose,
            verbose_eval: self.model.verbose_eval,
            tol: self.model.tol,
            early_stopping_rounds: self.model.early_stopping_rounds,
            validation_fraction: self.model.validation_fraction,
            random_state: self.model.random_state(),
            lr_schedule: self.model.lr_schedule,
            tikhonov_reg: self.model.tikhonov_reg,
            line_search_method: self.model.line_search_method,
        }
    }

    /// Set hyperparameters from a struct.
    /// Note: This only sets hyperparameters, not trained state.
    pub fn set_params(&mut self, params: NGBoostParams) {
        self.model.n_estimators = params.n_estimators;
        self.model.learning_rate = params.learning_rate;
        self.model.natural_gradient = params.natural_gradient;
        self.model.minibatch_frac = params.minibatch_frac;
        self.model.col_sample = params.col_sample;
        self.model.verbose = params.verbose;
        self.model.verbose_eval = params.verbose_eval;
        self.model.tol = params.tol;
        self.model.early_stopping_rounds = params.early_stopping_rounds;
        self.model.validation_fraction = params.validation_fraction;
        self.model.lr_schedule = params.lr_schedule;
        self.model.tikhonov_reg = params.tikhonov_reg;
        self.model.line_search_method = params.line_search_method;
        if let Some(seed) = params.random_state {
            self.model.set_random_state(seed);
        }
    }
}

/// Multi-class classifier using NGBoost with Categorical distribution.
///
/// This is a generic struct parameterized by the number of classes K.
/// For binary classification, use `NGBClassifier` instead.
///
/// # Example
/// ```ignore
/// use ngboost_rs::ngboost::NGBMultiClassifier;
///
/// // Create a 5-class classifier
/// let mut model: NGBMultiClassifier<5> = NGBMultiClassifier::new(100, 0.1);
/// model.fit(&x_train, &y_train).unwrap();
/// let probs = model.predict_proba(&x_test);
/// ```
pub struct NGBMultiClassifier<const K: usize> {
    model: NGBoost<Categorical<K>, LogScore, DecisionTreeLearner>,
}

impl<const K: usize> NGBMultiClassifier<K> {
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
        verbose_eval: f64,
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
        F: Fn(&Categorical<K>, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
    {
        self.model.set_train_loss_monitor(Box::new(monitor));
    }

    /// Set a custom validation loss monitor function
    pub fn set_val_loss_monitor<F>(&mut self, monitor: F)
    where
        F: Fn(&Categorical<K>, &Array1<f64>, Option<&Array1<f64>>) -> f64 + Send + Sync + 'static,
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

    /// Predict class labels.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.model.predict(x)
    }

    /// Get predictions up to a specific iteration.
    pub fn predict_at(&self, x: &Array2<f64>, max_iter: usize) -> Array1<f64> {
        self.model.predict_at(x, max_iter)
    }

    /// Returns an iterator over staged predictions.
    pub fn staged_predict(&self, x: &Array2<f64>) -> Vec<Array1<f64>> {
        self.model.staged_predict(x)
    }

    /// Predict class probabilities.
    /// Returns a (N, K) array where N is the number of samples and K is the number of classes.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let dist = self.model.pred_dist(x);
        dist.class_probs()
    }

    /// Get class probabilities up to a specific iteration.
    pub fn predict_proba_at(&self, x: &Array2<f64>, max_iter: usize) -> Array2<f64> {
        let dist = self.model.pred_dist_at(x, max_iter);
        dist.class_probs()
    }

    /// Returns staged probability predictions.
    pub fn staged_predict_proba(&self, x: &Array2<f64>) -> Vec<Array2<f64>> {
        self.staged_pred_dist(x)
            .into_iter()
            .map(|d| d.class_probs())
            .collect()
    }

    /// Get the predicted distribution.
    pub fn pred_dist(&self, x: &Array2<f64>) -> Categorical<K> {
        self.model.pred_dist(x)
    }

    /// Get the predicted distribution up to a specific iteration.
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> Categorical<K> {
        self.model.pred_dist_at(x, max_iter)
    }

    /// Returns staged distribution predictions.
    pub fn staged_pred_dist(&self, x: &Array2<f64>) -> Vec<Categorical<K>> {
        self.model.staged_pred_dist(x)
    }

    /// Get the predicted distribution parameters.
    pub fn pred_param(&self, x: &Array2<f64>) -> Array2<f64> {
        self.model.pred_param(x)
    }

    /// Compute the average score (loss) on the given data.
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        self.model.score(x, y)
    }

    /// Get the number of estimators (boosting iterations).
    pub fn n_estimators(&self) -> u32 {
        self.model.n_estimators
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.model.learning_rate
    }

    /// Get whether natural gradient is used.
    pub fn natural_gradient(&self) -> bool {
        self.model.natural_gradient
    }

    /// Get the minibatch fraction.
    pub fn minibatch_frac(&self) -> f64 {
        self.model.minibatch_frac
    }

    /// Get the column sampling fraction.
    pub fn col_sample(&self) -> f64 {
        self.model.col_sample
    }

    /// Get the best validation iteration.
    pub fn best_val_loss_itr(&self) -> Option<usize> {
        self.model.best_val_loss_itr
    }

    /// Get early stopping rounds.
    pub fn early_stopping_rounds(&self) -> Option<u32> {
        self.model.early_stopping_rounds
    }

    /// Get validation fraction.
    pub fn validation_fraction(&self) -> f64 {
        self.model.validation_fraction
    }

    /// Get number of features the model was trained on.
    pub fn n_features(&self) -> Option<usize> {
        self.model.n_features()
    }

    /// Compute feature importances per distribution parameter.
    pub fn feature_importances(&self) -> Option<Array2<f64>> {
        self.model.feature_importances()
    }

    /// Compute aggregated feature importances across all distribution parameters.
    pub fn feature_importances_aggregated(&self) -> Option<Array1<f64>> {
        self.model.feature_importances_aggregated()
    }

    /// Get the training history (losses at each iteration).
    pub fn evals_result(&self) -> &EvalsResult {
        self.model.evals_result()
    }

    /// Set the random seed for reproducibility.
    pub fn set_random_state(&mut self, seed: u64) {
        self.model.set_random_state(seed);
    }

    /// Get the current random state (seed), if set.
    pub fn random_state(&self) -> Option<u64> {
        self.model.random_state()
    }

    /// Fit with sample weights.
    pub fn fit_with_weights(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, None, None, sample_weight, None)
    }

    /// Fit with sample weights and validation data.
    pub fn fit_with_weights_and_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, x_val, y_val, sample_weight, val_sample_weight)
    }

    /// Get all hyperparameters as a struct.
    pub fn get_params(&self) -> NGBoostParams {
        NGBoostParams {
            n_estimators: self.model.n_estimators,
            learning_rate: self.model.learning_rate,
            natural_gradient: self.model.natural_gradient,
            minibatch_frac: self.model.minibatch_frac,
            col_sample: self.model.col_sample,
            verbose: self.model.verbose,
            verbose_eval: self.model.verbose_eval,
            tol: self.model.tol,
            early_stopping_rounds: self.model.early_stopping_rounds,
            validation_fraction: self.model.validation_fraction,
            random_state: self.model.random_state(),
            lr_schedule: self.model.lr_schedule,
            tikhonov_reg: self.model.tikhonov_reg,
            line_search_method: self.model.line_search_method,
        }
    }

    /// Set hyperparameters from a struct.
    pub fn set_params(&mut self, params: NGBoostParams) {
        self.model.n_estimators = params.n_estimators;
        self.model.learning_rate = params.learning_rate;
        self.model.natural_gradient = params.natural_gradient;
        self.model.minibatch_frac = params.minibatch_frac;
        self.model.col_sample = params.col_sample;
        self.model.verbose = params.verbose;
        self.model.verbose_eval = params.verbose_eval;
        self.model.tol = params.tol;
        self.model.early_stopping_rounds = params.early_stopping_rounds;
        self.model.validation_fraction = params.validation_fraction;
        self.model.lr_schedule = params.lr_schedule;
        self.model.tikhonov_reg = params.tikhonov_reg;
        self.model.line_search_method = params.line_search_method;
        if let Some(seed) = params.random_state {
            self.model.set_random_state(seed);
        }
    }

    /// Get the number of classes.
    pub fn n_classes(&self) -> usize {
        K
    }
}

/// Type alias for 3-class classification.
pub type NGBMultiClassifier3 = NGBMultiClassifier<3>;

/// Type alias for 4-class classification.
pub type NGBMultiClassifier4 = NGBMultiClassifier<4>;

/// Type alias for 5-class classification.
pub type NGBMultiClassifier5 = NGBMultiClassifier<5>;

/// Type alias for 10-class classification (e.g., digit recognition).
pub type NGBMultiClassifier10 = NGBMultiClassifier<10>;

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
        verbose_eval: f64,
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

    /// Returns staged predictions
    pub fn staged_predict(&self, x: &Array2<f64>) -> Vec<Array1<f64>> {
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

    /// Returns staged probability predictions
    pub fn staged_predict_proba(&self, x: &Array2<f64>) -> Vec<Array2<f64>> {
        self.staged_pred_dist(x)
            .into_iter()
            .map(|d| d.class_probs())
            .collect()
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> Bernoulli {
        self.model.pred_dist(x)
    }

    /// Get the predicted distribution up to a specific iteration
    pub fn pred_dist_at(&self, x: &Array2<f64>, max_iter: usize) -> Bernoulli {
        self.model.pred_dist_at(x, max_iter)
    }

    /// Returns staged distribution predictions
    pub fn staged_pred_dist(&self, x: &Array2<f64>) -> Vec<Bernoulli> {
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

    /// Calibrate uncertainty estimates using validation data.
    /// This improves the quality of probabilistic predictions for classification.
    ///
    /// Note: For classification, this adjusts the temperature scaling of the logits.
    pub fn calibrate_uncertainty(
        &mut self,
        _x_val: &Array2<f64>,
        _y_val: &Array1<f64>,
    ) -> Result<(), &'static str> {
        // For classification, temperature scaling could be applied here
        // Currently a no-op placeholder - classification calibration is more complex
        // than regression and typically uses Platt scaling or temperature scaling
        Ok(())
    }

    /// Get the training history (losses at each iteration).
    pub fn evals_result(&self) -> &EvalsResult {
        self.model.evals_result()
    }

    /// Set the random seed for reproducibility.
    pub fn set_random_state(&mut self, seed: u64) {
        self.model.set_random_state(seed);
    }

    /// Get the current random state (seed), if set.
    pub fn random_state(&self) -> Option<u64> {
        self.model.random_state()
    }

    /// Fit with sample weights.
    pub fn fit_with_weights(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, None, None, sample_weight, None)
    }

    /// Fit with sample weights and validation data.
    pub fn fit_with_weights_and_validation(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        sample_weight: Option<&Array1<f64>>,
        val_sample_weight: Option<&Array1<f64>>,
    ) -> Result<(), &'static str> {
        self.model
            .fit_with_validation(x, y, x_val, y_val, sample_weight, val_sample_weight)
    }

    /// Get all hyperparameters as a struct.
    pub fn get_params(&self) -> NGBoostParams {
        NGBoostParams {
            n_estimators: self.model.n_estimators,
            learning_rate: self.model.learning_rate,
            natural_gradient: self.model.natural_gradient,
            minibatch_frac: self.model.minibatch_frac,
            col_sample: self.model.col_sample,
            verbose: self.model.verbose,
            verbose_eval: self.model.verbose_eval,
            tol: self.model.tol,
            early_stopping_rounds: self.model.early_stopping_rounds,
            validation_fraction: self.model.validation_fraction,
            random_state: self.model.random_state(),
            lr_schedule: self.model.lr_schedule,
            tikhonov_reg: self.model.tikhonov_reg,
            line_search_method: self.model.line_search_method,
        }
    }

    /// Set hyperparameters from a struct.
    /// Note: This only sets hyperparameters, not trained state.
    pub fn set_params(&mut self, params: NGBoostParams) {
        self.model.n_estimators = params.n_estimators;
        self.model.learning_rate = params.learning_rate;
        self.model.natural_gradient = params.natural_gradient;
        self.model.minibatch_frac = params.minibatch_frac;
        self.model.col_sample = params.col_sample;
        self.model.verbose = params.verbose;
        self.model.verbose_eval = params.verbose_eval;
        self.model.tol = params.tol;
        self.model.early_stopping_rounds = params.early_stopping_rounds;
        self.model.validation_fraction = params.validation_fraction;
        self.model.lr_schedule = params.lr_schedule;
        self.model.tikhonov_reg = params.tikhonov_reg;
        self.model.line_search_method = params.line_search_method;
        if let Some(seed) = params.random_state {
            self.model.set_random_state(seed);
        }
    }
}
