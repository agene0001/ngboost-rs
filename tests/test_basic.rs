//! Basic tests matching Python's test_basic.py
//!
//! These tests verify basic regression and classification functionality
//! using synthetic data that mimics real-world datasets.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::{ClassificationDistn, Distribution};
use ngboost_rs::ngboost::{NGBClassifier, NGBRegressor};

// ============================================================================
// Test data generation (mimicking sklearn datasets)
// ============================================================================

/// Generate synthetic regression data similar to California Housing.
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    // Generate features
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());

    // Generate target with a nonlinear relationship
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x0: f64 = x[[i, 0]];
        let x1: f64 = x[[i, 1]];
        let x2: f64 = x[[i, 2 % n_features]];
        let x3: f64 = x[[i, 3 % n_features]];
        y[i] = 2.0 * x0 + 3.0 * x1.powi(2) - 1.5 * x2 + (x3 * std::f64::consts::PI).sin() + 0.1;
    }

    // Scale y to be in a reasonable range
    let y_mean = y.mean().unwrap();
    let y_std = y.std(0.0).max(0.1);
    let y = y.mapv(|v| (v - y_mean) / y_std * 1.5 + 2.0);

    (x, y)
}

/// Generate synthetic classification data similar to Breast Cancer dataset.
fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    // Generate features
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());

    // Generate binary labels based on a decision boundary
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x0: f64 = x[[i, 0]];
        let x1: f64 = x[[i, 1]];
        let x2: f64 = x[[i, 2 % n_features]];
        let score = x0 * 2.0 + x1.powi(2) - x2 * 1.5;
        y[i] = if score > 0.5 { 1.0 } else { 0.0 };
    }

    (x, y)
}

/// Split data into train and test sets.
fn train_test_split(
    x: Array2<f64>,
    y: Array1<f64>,
    test_size: f64,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let n_samples = x.nrows();
    let n_test = (n_samples as f64 * test_size) as usize;
    let n_train = n_samples - n_test;

    let train_indices: Vec<usize> = (0..n_train).collect();
    let test_indices: Vec<usize> = (n_train..n_samples).collect();

    let x_train = x.select(Axis(0), &train_indices);
    let x_test = x.select(Axis(0), &test_indices);
    let y_train = y.select(Axis(0), &train_indices);
    let y_test = y.select(Axis(0), &test_indices);

    (x_train, x_test, y_train, y_test)
}

// ============================================================================
// Metric functions
// ============================================================================

/// Mean Squared Error
fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    (y_true - y_pred).mapv(|a| a.powi(2)).mean().unwrap()
}

/// ROC-AUC Score (simplified implementation)
fn roc_auc_score(y_true: &Array1<f64>, y_scores: &Array1<f64>) -> f64 {
    let n = y_true.len();
    let mut pairs: Vec<(f64, f64)> = y_true
        .iter()
        .zip(y_scores.iter())
        .map(|(&t, &s)| (t, s))
        .collect();

    // Sort by score descending
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_pos = y_true.iter().filter(|&&v| v > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;

    for (true_label, _) in pairs.iter() {
        if *true_label > 0.5 {
            tp += 1.0;
        } else {
            auc += tp;
        }
    }

    auc / (n_pos * n_neg)
}

/// Log Loss (Binary Cross-Entropy)
fn log_loss(y_true: &Array1<f64>, y_prob: &Array2<f64>) -> f64 {
    let eps = 1e-15;
    let n = y_true.len();

    let mut loss = 0.0;
    for i in 0..n {
        let p = y_prob[[i, 1]].max(eps).min(1.0 - eps);
        let y = y_true[i];
        loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }

    loss / n as f64
}

// ============================================================================
// Regression Tests (matching Python's test_regression)
// ============================================================================

#[test]
fn test_regression() {
    let (x, y) = generate_regression_data(1000, 8);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBRegressor::with_options(
        500, 0.01, true, 1.0, 1.0, false, 100, 1e-4, None, 0.1, false,
    );

    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let preds = ngb.predict(&x_test);
    let score = mean_squared_error(&y_test, &preds);

    println!("Regression MSE: {:.4}", score);
    assert!(score <= 2.0, "MSE {:.4} should be <= 2.0", score);

    let model_score = ngb.score(&x_test, &y_test);
    assert!(
        model_score <= 5.0,
        "Model score {:.4} should be reasonable",
        model_score
    );

    let dist = ngb.pred_dist(&x_test);
    assert_eq!(dist.predict().len(), x_test.nrows());
}

#[test]
fn test_regression_with_early_stopping() {
    let (x, y) = generate_regression_data(800, 6);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBRegressor::with_options(
        500,
        0.01,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        Some(10),
        0.1,
        false,
    );

    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let preds = ngb.predict(&x_test);
    let score = mean_squared_error(&y_test, &preds);

    println!("Regression with early stopping MSE: {:.4}", score);
    assert!(score <= 3.0, "MSE should be reasonable");
}

// ============================================================================
// Classification Tests (matching Python's test_classification)
// ============================================================================

#[test]
fn test_classification() {
    let (x, y) = generate_classification_data(500, 10);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBClassifier::with_options(
        500, 0.01, true, 1.0, 1.0, false, 100, 1e-4, None, 0.1, false,
    );

    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let _preds = ngb.predict(&x_test);
    let proba = ngb.predict_proba(&x_test);
    let roc_auc = roc_auc_score(&y_test, &proba.column(1).to_owned());

    println!("Classification ROC-AUC: {:.4}", roc_auc);
    assert!(roc_auc >= 0.70, "ROC-AUC {:.4} should be >= 0.70", roc_auc);

    let ll = log_loss(&y_test, &proba);
    println!("Classification Log Loss: {:.4}", ll);
    assert!(ll <= 0.70, "Log loss {:.4} should be <= 0.70", ll);

    let dist = ngb.pred_dist(&x_test);
    let class_probs = dist.class_probs();
    assert_eq!(class_probs.nrows(), x_test.nrows());
    assert_eq!(class_probs.ncols(), 2);

    for i in 0..class_probs.nrows() {
        let sum = class_probs[[i, 0]] + class_probs[[i, 1]];
        assert!((sum - 1.0).abs() < 1e-6, "Probabilities should sum to 1");
    }
}

#[test]
fn test_classification_with_validation() {
    let (x, y) = generate_classification_data(400, 8);
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.2);

    let n_train = (x_train.nrows() as f64 * 0.8) as usize;
    let x_val = x_train.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_val = y_train.slice(ndarray::s![n_train..]).to_owned();
    let x_train_split = x_train.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train_split = y_train.slice(ndarray::s![..n_train]).to_owned();

    let mut ngb = NGBClassifier::with_options(
        300,
        0.01,
        true,
        1.0,
        1.0,
        false,
        100,
        1e-4,
        Some(10),
        0.1,
        false,
    );

    ngb.fit_with_validation(&x_train_split, &y_train_split, Some(&x_val), Some(&y_val))
        .expect("Fit should succeed");

    let proba = ngb.predict_proba(&x_test);
    let roc_auc = roc_auc_score(&y_test, &proba.column(1).to_owned());

    println!("Classification with validation ROC-AUC: {:.4}", roc_auc);
    assert!(roc_auc >= 0.60, "ROC-AUC should be reasonable");
}

// ============================================================================
// Additional tests for API parity
// ============================================================================

#[test]
fn test_staged_predict_regression() {
    let (x, y) = generate_regression_data(200, 5);
    let (x_train, x_test, y_train, _y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBRegressor::new(50, 0.1);
    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let staged_preds: Vec<_> = ngb.staged_predict(&x_test).collect();
    assert_eq!(staged_preds.len(), 50);

    for pred in staged_preds.iter() {
        assert_eq!(pred.len(), x_test.nrows());
    }
}

#[test]
fn test_staged_predict_classification() {
    let (x, y) = generate_classification_data(200, 5);
    let (x_train, x_test, y_train, _y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBClassifier::new(30, 0.1);
    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let staged_proba: Vec<_> = ngb.staged_predict_proba(&x_test).collect();
    assert_eq!(staged_proba.len(), 30);

    for proba in staged_proba.iter() {
        assert_eq!(proba.nrows(), x_test.nrows());
        assert_eq!(proba.ncols(), 2);
    }
}

#[test]
fn test_predict_at_iteration() {
    let (x, y) = generate_regression_data(200, 5);
    let (x_train, x_test, y_train, _y_test) = train_test_split(x, y, 0.2);

    let mut ngb = NGBRegressor::new(100, 0.1);
    ngb.fit(&x_train, &y_train).expect("Fit should succeed");

    let pred_10 = ngb.predict_at(&x_test, 10);
    let pred_50 = ngb.predict_at(&x_test, 50);
    let pred_100 = ngb.predict(&x_test);

    assert_eq!(pred_10.len(), x_test.nrows());
    assert_eq!(pred_50.len(), x_test.nrows());
    assert_eq!(pred_100.len(), x_test.nrows());

    let diff_10_50 = (&pred_10 - &pred_50).mapv(|v| v.abs()).sum();
    assert!(
        diff_10_50 > 0.0,
        "Predictions at different iterations should differ"
    );
}

#[test]
fn test_feature_importances() {
    let (x, y) = generate_regression_data(200, 8);

    let mut ngb = NGBRegressor::new(50, 0.1);
    ngb.fit(&x, &y).expect("Fit should succeed");

    let importances = ngb.feature_importances();
    assert!(importances.is_some());

    let imp = importances.unwrap();
    assert_eq!(imp.ncols(), 8);

    for val in imp.iter() {
        assert!(*val >= 0.0);
    }

    let agg_imp = ngb.feature_importances_aggregated();
    assert!(agg_imp.is_some());

    let agg = agg_imp.unwrap();
    assert_eq!(agg.len(), 8);

    let sum: f64 = agg.sum();
    assert!((sum - 1.0).abs() < 1e-6 || sum == 0.0);
}

#[test]
fn test_partial_fit() {
    let (x1, y1) = generate_regression_data(100, 5);
    let (x2, y2) = generate_regression_data(100, 5);

    let mut ngb = NGBRegressor::new(30, 0.1);

    // First fit
    ngb.fit(&x1, &y1).expect("First fit should succeed");
    let preds_after_first = ngb.predict(&x1);

    // Partial fit with new data
    ngb.partial_fit(&x2, &y2)
        .expect("Partial fit should succeed");

    // Model should have more estimators now
    let preds_after_partial = ngb.predict(&x1);

    // Predictions should be different after partial_fit
    let diff: f64 = (&preds_after_first - &preds_after_partial)
        .mapv(|v| v.abs())
        .sum();
    assert!(diff > 0.0, "Predictions should change after partial_fit");
}
