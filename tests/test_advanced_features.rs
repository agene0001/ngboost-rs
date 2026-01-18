#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Tests for advanced NGBoost features:
// - Learning rate schedules
// - Line search methods
// - Tikhonov regularization

use ndarray::{Array1, Array2};
use ngboost_rs::dist::normal::Normal;
use ngboost_rs::learners::default_tree_learner;
use ngboost_rs::ngboost::{LearningRateSchedule, LineSearchMethod, NGBoost};
use ngboost_rs::scores::LogScore;

/// Generate simple synthetic data for testing
fn generate_test_data() -> (Array2<f64>, Array1<f64>) {
    let n = 50;
    let n_features = 3;

    let mut x_data = Vec::with_capacity(n * n_features);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) / (n as f64);
        let x2 = ((i * 7) % n) as f64 / (n as f64);
        let x3 = ((i * 13) % n) as f64 / (n as f64);
        x_data.push(x1);
        x_data.push(x2);
        x_data.push(x3);
        y_data.push(x1 * 2.0 + x2 * 1.5 + x3 * 0.5 + 0.1);
    }

    (
        Array2::from_shape_vec((n, n_features), x_data).unwrap(),
        Array1::from_vec(y_data),
    )
}

// ============================================================================
// Learning Rate Schedule Tests
// ============================================================================

#[test]
fn test_constant_learning_rate() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
}

#[test]
fn test_cosine_learning_rate() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Cosine,
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
}

#[test]
fn test_exponential_learning_rate() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Exponential { decay_rate: 2.0 },
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
}

#[test]
fn test_linear_learning_rate() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Linear {
            decay_rate: 0.7,
            min_lr_fraction: 0.1,
        },
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
}

#[test]
fn test_cosine_warm_restarts() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        30,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::CosineWarmRestarts { restart_period: 10 },
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
}

// ============================================================================
// Line Search Method Tests
// ============================================================================

#[test]
fn test_binary_line_search() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let score = model.score(&x, &y);
    assert!(score.is_finite());
}

#[test]
fn test_golden_section_line_search() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::GoldenSection { max_iters: 20 },
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let score = model.score(&x, &y);
    assert!(score.is_finite());
}

#[test]
fn test_golden_section_few_iterations() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::GoldenSection { max_iters: 5 },
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert!(predictions.iter().all(|&p| p.is_finite()));
}

// ============================================================================
// Tikhonov Regularization Tests
// ============================================================================

#[test]
fn test_tikhonov_regularization() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        1e-6, // Tikhonov regularization
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert!(predictions.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_strong_tikhonov_regularization() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        1e-3, // Stronger regularization
        LineSearchMethod::Binary,
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert!(predictions.iter().all(|&p| p.is_finite()));
}

// ============================================================================
// Combined Features Tests
// ============================================================================

#[test]
fn test_all_advanced_features_combined() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Cosine,
        1e-6,
        LineSearchMethod::GoldenSection { max_iters: 15 },
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert_eq!(predictions.len(), x.nrows());
    assert!(predictions.iter().all(|&p| p.is_finite()));

    let score = model.score(&x, &y);
    assert!(score.is_finite());
    // Note: Log score (negative log probability) can be negative when PDF > 1
    // which happens for Normal distributions with small variance
}

#[test]
fn test_exponential_with_golden_section() {
    let (x, y) = generate_test_data();
    let base_learner = default_tree_learner();
    let mut model: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Exponential { decay_rate: 1.5 },
        1e-5,
        LineSearchMethod::GoldenSection { max_iters: 10 },
    );

    let result = model.fit(&x, &y);
    assert!(result.is_ok());

    let predictions = model.predict(&x);
    assert!(predictions.iter().all(|&p| p.is_finite()));
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_zero_tikhonov_same_as_none() {
    let (x, y) = generate_test_data();

    // Model without Tikhonov regularization
    let base_learner = default_tree_learner();
    let mut model1: NGBoost<Normal, LogScore, _> = NGBoost::new(20, 0.1, base_learner);
    model1.fit(&x, &y).unwrap();

    // Model with zero Tikhonov regularization (should behave the same)
    let base_learner = default_tree_learner();
    let mut model2: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        20,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0, // Zero regularization
        LineSearchMethod::Binary,
    );
    model2.fit(&x, &y).unwrap();

    // Scores should be identical (or very close)
    let score1 = model1.score(&x, &y);
    let score2 = model2.score(&x, &y);
    assert!((score1 - score2).abs() < 1e-10);
}
