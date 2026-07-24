#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Basic tests matching Python's test_basic.py
//
// These tests verify basic regression and classification functionality
// using synthetic data that mimics real-world datasets.

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
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
        500, 0.01, true, 1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
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
        100.0,
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
        500, 0.01, true, 1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
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
        100.0,
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

    let staged_preds = ngb.staged_predict(&x_test);
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

    let staged_proba = ngb.staged_predict_proba(&x_test);
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

#[test]
fn test_parallel_prediction_path_bit_exact() {
    // get_params_at takes a rayon row-chunk path when nrows >= 512 and a
    // sequential path below. Predicting 1200 rows at once (parallel) must be
    // bit-identical to predicting the same rows in sub-512 slices (sequential).
    let (x, y) = generate_regression_data(1200, 6);

    let mut ngb = NGBRegressor::new(50, 0.1);
    ngb.fit(&x, &y).expect("fit should succeed");

    let full = ngb.pred_param(&x);

    let mut pieces = Vec::new();
    for chunk_start in (0..1200).step_by(300) {
        let xs = x.slice(ndarray::s![chunk_start..chunk_start + 300, ..]).to_owned();
        pieces.push(ngb.pred_param(&xs));
    }

    for (i, piece) in pieces.iter().enumerate() {
        for r in 0..300 {
            for c in 0..full.ncols() {
                assert_eq!(
                    full[[i * 300 + r, c]].to_bits(),
                    piece[[r, c]].to_bits(),
                    "params differ at row {} col {}",
                    i * 300 + r,
                    c
                );
            }
        }
    }
}

#[test]
fn test_parallel_prediction_path_bit_exact_col_subsample() {
    // Same as above but with col_sample < 1 so trees were fit on column
    // subsets, exercising the remapped predict_rows walk.
    let (x, y) = generate_regression_data(1200, 6);

    use ngboost_rs::learners::HistogramLearner;
    let mut ngb = ngboost_rs::NGBHistRegressor::with_options_seeded(
        40,
        0.1,
        HistogramLearner::default_histogram(),
        true,
        1.0,
        0.5, // col_sample
        false,
        0.0,
        1e-4,
        None,
        0.1,
        false,
        Some(7),
    );
    ngb.fit(&x, &y).expect("fit should succeed");

    let full = ngb.pred_param(&x);

    let mut pieces = Vec::new();
    for chunk_start in (0..1200).step_by(300) {
        let xs = x.slice(ndarray::s![chunk_start..chunk_start + 300, ..]).to_owned();
        pieces.push(ngb.pred_param(&xs));
    }

    for (i, piece) in pieces.iter().enumerate() {
        for r in 0..300 {
            for c in 0..full.ncols() {
                assert_eq!(
                    full[[i * 300 + r, c]].to_bits(),
                    piece[[r, c]].to_bits(),
                    "params differ at row {} col {}",
                    i * 300 + r,
                    c
                );
            }
        }
    }
}

// ============================================================================
// Early stopping truncation / partial_fit interaction (June 2026 audit)
// ============================================================================

/// A partial_fit WITHOUT validation, on a model whose earlier fit early-stopped,
/// must keep everything it trained: the truncation step may only act on a best
/// iteration observed during the same run, never on the stale one.
#[test]
fn test_partial_fit_after_early_stopped_fit_is_not_truncated() {
    let (x, y) = generate_regression_data(400, 4);
    let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.25);

    // validation_fraction = 0.0 so the later partial_fit (no explicit val
    // data) does not auto-split: early stopping is configured but cannot run.
    let mut ngb = NGBRegressor::with_options(
        40, 0.05, true, 1.0, 1.0, false, 100.0, 1e-12, Some(5), 0.0, false,
    );
    ngb.fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val))
        .expect("fit should succeed");

    let n_after_fit = ngb.staged_predict(&x_val).len();
    let best_after_fit = ngb.best_val_loss_itr().expect("validation ran");
    // With early stopping + validation, fit truncates to the best iteration.
    assert_eq!(
        n_after_fit,
        best_after_fit + 1,
        "fit should keep exactly best_val_loss_itr + 1 models"
    );

    ngb.partial_fit(&x_train, &y_train)
        .expect("partial_fit should succeed");
    let n_after_partial = ngb.staged_predict(&x_val).len();

    assert_eq!(
        n_after_partial,
        n_after_fit + 40,
        "partial_fit must append all its models; a stale best_val_loss_itr must not truncate them"
    );
    // No validation ran in the partial run, so the recorded best is unchanged.
    assert_eq!(ngb.best_val_loss_itr(), Some(best_after_fit));
}

/// `predict_best`/`pred_dist_best` must reproduce the state INCLUDING the
/// best iteration's model — i.e. `pred_param_at(best + 1)`, not the
/// off-by-one `pred_param_at(best)` that mirrors Python's foot-gun.
#[test]
fn test_predict_best_includes_best_iteration() {
    let (x, y) = generate_regression_data(400, 4);
    let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.25);

    // No early stopping: nothing is truncated, so the best iteration can sit
    // strictly inside the model and the foot-gun is observable.
    let mut ngb = NGBRegressor::with_options(
        30, 0.3, true, 1.0, 1.0, false, 100.0, 1e-12, None, 0.0, false,
    );
    ngb.fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val))
        .expect("fit should succeed");
    let best = ngb.best_val_loss_itr().expect("validation ran");

    let best_params = ngb.pred_dist_best(&x_val).params();
    let inclusive = ngb.pred_dist_at(&x_val, best + 1).params();
    let footgun = ngb.pred_dist_at(&x_val, best).params();

    // pred_dist_best == the state AFTER applying the best iteration's model
    for (a, b) in best_params.iter().zip(inclusive.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "best != at(best+1)");
    }
    // ... which differs from the off-by-one call whenever best >= 1
    if best >= 1 {
        let identical = best_params
            .iter()
            .zip(footgun.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(!identical, "at(best) should exclude the best model");
    }

    // predict_best consistency with the distribution path
    let preds = ngb.predict_best(&x_val);
    let dist_preds = ngb.pred_dist_best(&x_val).predict();
    for (a, b) in preds.iter().zip(dist_preds.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// `pred_dist_at(x, 0)` must be the INIT-ONLY distribution (0 stages means
/// 0 stages), not the full model (Python's falsy-zero foot-gun, deliberately
/// deviated from).
#[test]
fn test_pred_dist_at_zero_is_init_only() {
    let (x, y) = generate_regression_data(200, 4);
    let mut ngb = NGBRegressor::new(20, 0.1);
    ngb.fit(&x, &y).expect("fit should succeed");

    let at0 = ngb.pred_dist_at(&x, 0).params();
    // every row must equal the init params (marginal fit), i.e. all rows equal
    for r in 1..at0.nrows() {
        for c in 0..at0.ncols() {
            assert_eq!(
                at0[[r, c]].to_bits(),
                at0[[0, c]].to_bits(),
                "init-only params must be identical across rows"
            );
        }
    }
    // and differ from the fully boosted model
    let full = ngb.pred_dist(&x).params();
    assert!(
        at0.iter().zip(full.iter()).any(|(a, b)| a != b),
        "0-stage params should differ from the full model"
    );
    // one stage differs from zero stages
    let at1 = ngb.pred_dist_at(&x, 1).params();
    assert!(at0.iter().zip(at1.iter()).any(|(a, b)| a != b));
}

/// best_val_loss_itr is a GLOBAL model index (Python parity: its loop counter
/// starts at len(col_idxs)), so after a partial_fit with validation it must
/// point past the models from the first fit.
#[test]
fn test_best_val_loss_itr_is_global_across_partial_fits() {
    let (x, y) = generate_regression_data(400, 4);
    let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.25);

    // No early stopping: nothing is truncated; indices must still be global.
    let mut ngb = NGBRegressor::with_options(
        15, 0.05, true, 1.0, 1.0, false, 100.0, 1e-12, None, 0.0, false,
    );
    ngb.fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val))
        .expect("fit should succeed");
    assert_eq!(ngb.staged_predict(&x_val).len(), 15);

    ngb.partial_fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val))
        .expect("partial_fit should succeed");
    assert_eq!(ngb.staged_predict(&x_val).len(), 30);

    // The second run resets its best-loss tracker, so its best iteration is
    // one of its own models: global index 15..30.
    let best = ngb.best_val_loss_itr().expect("validation ran");
    assert!(
        (15..30).contains(&best),
        "best_val_loss_itr={best} should be a global index in 15..30"
    );
}

/// Tiny datasets where n * validation_fraction < 1 must still get a real
/// (1-row, sklearn-ceil) validation split. The old floor produced an EMPTY
/// validation set whose loss was 0.0 at every iteration, which "early
/// stopped" immediately and truncated the model to a single estimator.
#[test]
fn test_early_stopping_auto_split_tiny_dataset() {
    let (x, y) = generate_regression_data(8, 2);
    let mut ngb = NGBRegressor::with_options(
        30, 0.1, true, 1.0, 1.0, false, 100.0, 1e-12, Some(10), 0.1, false,
    );
    ngb.fit(&x, &y).expect("fit should succeed");

    let val_losses = &ngb.evals_result().val;
    assert!(
        !val_losses.is_empty(),
        "auto-split should have produced validation losses"
    );
    assert!(
        val_losses.iter().any(|&v| v != 0.0),
        "validation losses must come from a real (non-empty) validation set"
    );
}

/// Python parity: early stopping on explicit validation data requires
/// consistent weighting — train weights without val weights (or vice versa)
/// is an error, since it would silently skew early stopping.
#[test]
fn test_early_stopping_weight_mismatch_is_an_error() {
    use ngboost_rs::learners::default_base_learner;
    use ngboost_rs::ngboost::NGBoost;
    use ngboost_rs::dist::Normal;
    use ngboost_rs::scores::LogScore;

    let (x, y) = generate_regression_data(200, 3);
    let (x_train, x_val, y_train, y_val) = train_test_split(x, y, 0.25);
    let w = Array1::from_elem(y_train.len(), 1.0);

    let mut ngb = NGBoost::<Normal, LogScore, _>::with_options(
        20,
        0.05,
        default_base_learner(),
        true,
        1.0,
        1.0,
        false,
        100.0,
        1e-12,
        Some(5),
        0.1,
        false,
    );

    let result = ngb.fit_with_validation(
        &x_train,
        &y_train,
        Some(&x_val),
        Some(&y_val),
        Some(&w),
        None, // val_sample_weight missing while sample_weight is set
    );
    assert!(
        result.is_err(),
        "train weights without val weights under early stopping must error (Python parity)"
    );

    // Without early stopping the combination stays permitted (Python only
    // checks under early_stopping_rounds).
    let mut ngb2 = NGBoost::<Normal, LogScore, _>::with_options(
        20,
        0.05,
        default_base_learner(),
        true,
        1.0,
        1.0,
        false,
        100.0,
        1e-12,
        None,
        0.1,
        false,
    );
    ngb2.fit_with_validation(&x_train, &y_train, Some(&x_val), Some(&y_val), Some(&w), None)
        .expect("permitted without early stopping");
}

/// The row-subsampled histogram path (full-X bin edges reused across
/// iterations) must produce a model that learns the signal, for both the
/// row-only case (subset fast path) and row+column subsampling (per-iteration
/// cache path).
#[test]
fn test_minibatch_histogram_subset_path_learns() {
    let (x, y) = generate_regression_data(600, 5);
    let (x_test, y_test) = generate_regression_data(200, 5);

    for (mb, cs) in [(0.7_f64, 1.0_f64), (0.7, 0.6)] {
        let mut model = NGBRegressor::with_options(
            150, 0.05, true, mb, cs, false, 100.0, 1e-4, None, 0.1, false,
        );
        model.fit(&x, &y).unwrap();

        let pred = model.predict(&x_test);
        let mean_y = y_test.mean().unwrap();
        let ss_tot: f64 = y_test.iter().map(|v| (v - mean_y).powi(2)).sum();
        let ss_res: f64 = pred
            .iter()
            .zip(y_test.iter())
            .map(|(p, v)| (p - v).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(
            r2 > 0.5,
            "subsampled fit (mb={mb}, cs={cs}) should learn: R² = {r2}"
        );
    }
}

// ============================================================================
// 2026-07-23 audit regressions
// ============================================================================

/// Constant targets used to hang forever: Normal::fit floors sigma to 1.0,
/// the NLL is then unbounded below on zero-residual rows, boosting drives
/// log-scale to ~-383 where sigma^2 underflows to 0.0, gradients become NaN,
/// and the line-search down phase had no exit for non-finite residuals
/// (100% CPU, never returns). Training must now fail loudly instead.
#[test]
fn test_constant_target_errors_instead_of_hanging() {
    let x = Array2::random((200, 4), Uniform::new(0.0, 1.0).unwrap());
    let y = Array1::from_elem(200, 3.0);

    let mut model = NGBRegressor::new(500, 0.1);
    let result = model.fit(&x, &y);
    let err = result.expect_err("constant-y fit must diverge with an error, not hang");
    assert!(
        err.contains("diverged"),
        "expected a divergence error, got: {err}"
    );
}

/// Degenerate hyperparameters / weights must be rejected up front instead of
/// silently corrupting training (audit findings F4/F5/F6).
#[test]
fn test_degenerate_config_rejected() {
    let x = Array2::random((50, 4), Uniform::new(0.0, 1.0).unwrap());
    let (_, y) = generate_regression_data(50, 4);

    // early_stopping_rounds = 0 silently trained exactly 1 estimator
    let mut m = NGBRegressor::with_options(
        20, 0.1, true, 1.0, 1.0, false, 100.0, 1e-4, Some(0), 0.1, false,
    );
    assert!(m.fit(&x, &y).is_err());

    // early stopping with no possible validation data silently disabled it
    let mut m = NGBRegressor::with_options(
        20, 0.1, true, 1.0, 1.0, false, 100.0, 1e-4, Some(5), 0.0, false,
    );
    assert!(m.fit(&x, &y).is_err());

    // all-zero weights produced NaN losses reported as Ok
    let mut m = NGBRegressor::new(20, 0.1);
    let w = Array1::zeros(50);
    assert!(m.fit_with_weights(&x, &y, Some(&w)).is_err());

    // negative weights
    let mut m = NGBRegressor::new(20, 0.1);
    let mut w = Array1::ones(50);
    w[0] = -1.0;
    assert!(m.fit_with_weights(&x, &y, Some(&w)).is_err());
}

/// Predicting on an empty batch panicked (get_params_at returned shape
/// (0, 0) and from_params indexed column 0 of a 0-column array).
#[test]
fn test_empty_batch_predict() {
    let (x, y) = generate_regression_data(80, 4);
    let mut model = NGBRegressor::new(10, 0.1);
    model.fit(&x, &y).unwrap();

    let empty = Array2::<f64>::zeros((0, 4));
    let preds = model.predict(&empty);
    assert_eq!(preds.len(), 0);
    let dist = model.pred_dist(&empty);
    assert_eq!(dist.params().nrows(), 0);
}

/// 2026-07-23 audit regression: classifier label validation. A -1.0 label
/// silently trained as class 0 ({-1,+1}-encoded data corrupted with no
/// error); a label >= n_classes PANICKED with an index error mid-fit. Both
/// must be clean errors now.
#[test]
fn test_classifier_label_validation() {
    let x = Array2::random((40, 3), Uniform::new(0.0, 1.0).unwrap());

    // negative labels (SVM-style {-1, +1})
    let y_neg = Array1::from_vec((0..40).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect());
    let mut m = NGBClassifier::new(10, 0.1);
    assert!(m.fit(&x, &y_neg).is_err());

    // label >= K (binary classifier fed a 2.0) — used to panic
    let y_big = Array1::from_vec((0..40).map(|i| f64::from(u8::from(i % 3 == 0)) + 1.0).collect());
    let mut m = NGBClassifier::new(10, 0.1);
    assert!(m.fit(&x, &y_big).is_err());

    // non-integer labels
    let y_frac = Array1::from_vec((0..40).map(|i| if i % 2 == 0 { 0.5 } else { 1.0 }).collect());
    let mut m = NGBClassifier::new(10, 0.1);
    assert!(m.fit(&x, &y_frac).is_err());

    // valid labels still fit
    let y_ok = Array1::from_vec((0..40).map(|i| f64::from(u8::from(i % 2 == 0))).collect());
    let mut m = NGBClassifier::new(10, 0.1);
    assert!(m.fit(&x, &y_ok).is_ok());
}
