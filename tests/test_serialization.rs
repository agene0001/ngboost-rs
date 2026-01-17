#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Serialization tests matching Python's test_pickling.py
//
// Tests that models can be saved and loaded correctly with predictions matching.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::ngboost::{NGBClassifier, NGBRegressor};
use tempfile::tempdir;

// ============================================================================
// Helper functions
// ============================================================================

fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());
    let y = x.column(0).mapv(|v| v * 2.0 + 1.0)
        + x.column(1).mapv(|v| v * 0.5)
        + Array1::random(n_samples, Uniform::new(-0.1, 0.1).unwrap());
    (x, y.mapv(|v| v.abs() + 0.1))
}

fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());
    let linear = x.column(0).mapv(|v| v * 2.0) - x.column(1).mapv(|v| v * 1.5);
    let y = linear.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
    (x, y)
}

fn arrays_approx_equal(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

// ============================================================================
// Regression serialization tests
// ============================================================================

#[test]
fn test_regressor_save_load() {
    let (x, y) = generate_regression_data(100, 5);

    // Train model
    let mut model = NGBRegressor::new(50, 0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    // Get predictions before saving
    let preds_before = model.predict(&x);

    // Create temp directory and save
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("model.bin");
    let path_str = path.to_str().unwrap();

    model.save_model(path_str).expect("Save should succeed");

    // Verify file exists
    assert!(path.exists(), "Model file should exist");

    // Load model
    let loaded_model = NGBRegressor::load_model(path_str).expect("Load should succeed");

    // Get predictions after loading
    let preds_after = loaded_model.predict(&x);

    // Predictions should be identical
    assert!(
        arrays_approx_equal(&preds_before, &preds_after, 1e-10),
        "Predictions should match after load"
    );

    // Cleanup
    dir.close().expect("Failed to close temp dir");
}

#[test]
fn test_classifier_save_load() {
    let (x, y) = generate_classification_data(100, 5);

    // Train model
    let mut model = NGBClassifier::new(30, 0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    // Get predictions before saving
    let preds_before = model.predict(&x);
    let proba_before = model.predict_proba(&x);

    // Create temp directory and save
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("classifier.bin");
    let path_str = path.to_str().unwrap();

    model.save_model(path_str).expect("Save should succeed");

    // Load model
    let loaded_model = NGBClassifier::load_model(path_str).expect("Load should succeed");

    // Get predictions after loading
    let preds_after = loaded_model.predict(&x);
    let proba_after = loaded_model.predict_proba(&x);

    // Predictions should be identical
    assert!(
        arrays_approx_equal(&preds_before, &preds_after, 1e-10),
        "Class predictions should match after load"
    );

    // Probabilities should be identical
    for i in 0..proba_before.nrows() {
        for j in 0..proba_before.ncols() {
            let diff = (proba_before[[i, j]] - proba_after[[i, j]]).abs();
            assert!(diff < 1e-10, "Probabilities should match after load");
        }
    }

    // Cleanup
    dir.close().expect("Failed to close temp dir");
}

#[test]
fn test_regressor_save_load_with_options() {
    let (x, y) = generate_regression_data(100, 5);

    // Train model with various options
    let mut model = NGBRegressor::with_options(
        100, 0.05, true, 0.8, // minibatch_frac
        0.9, // col_sample
        false, 50, 1e-5, None, 0.15, false,
    );
    model.fit(&x, &y).expect("Fit should succeed");

    let preds_before = model.predict(&x);

    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("model_options.bin");
    let path_str = path.to_str().unwrap();

    model.save_model(path_str).expect("Save should succeed");
    let loaded_model = NGBRegressor::load_model(path_str).expect("Load should succeed");

    let preds_after = loaded_model.predict(&x);

    assert!(
        arrays_approx_equal(&preds_before, &preds_after, 1e-10),
        "Predictions should match for model with options"
    );

    dir.close().expect("Failed to close temp dir");
}

#[test]
fn test_save_load_new_data() {
    let (x_train, y_train) = generate_regression_data(100, 5);
    let (x_test, _) = generate_regression_data(20, 5);

    // Train model
    let mut model = NGBRegressor::new(50, 0.1);
    model.fit(&x_train, &y_train).expect("Fit should succeed");

    // Get predictions on test data
    let preds_before = model.predict(&x_test);

    // Save and load
    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("model.bin");
    let path_str = path.to_str().unwrap();

    model.save_model(path_str).expect("Save should succeed");
    let loaded_model = NGBRegressor::load_model(path_str).expect("Load should succeed");

    // Predictions on test data should match
    let preds_after = loaded_model.predict(&x_test);

    assert!(
        arrays_approx_equal(&preds_before, &preds_after, 1e-10),
        "Test predictions should match after load"
    );

    dir.close().expect("Failed to close temp dir");
}

#[test]
fn test_load_nonexistent_file() {
    let result = NGBRegressor::load_model("/nonexistent/path/model.bin");
    assert!(result.is_err(), "Loading nonexistent file should fail");
}

#[test]
fn test_multiple_save_load_cycles() {
    let (x, y) = generate_regression_data(80, 4);

    let mut model = NGBRegressor::new(30, 0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    let original_preds = model.predict(&x);

    let dir = tempdir().expect("Failed to create temp dir");

    // Save/load multiple times
    for i in 0..3 {
        let path = dir.path().join(format!("model_{}.bin", i));
        let path_str = path.to_str().unwrap();

        model.save_model(path_str).expect("Save should succeed");
        let loaded = NGBRegressor::load_model(path_str).expect("Load should succeed");

        let preds = loaded.predict(&x);
        assert!(
            arrays_approx_equal(&original_preds, &preds, 1e-10),
            "Predictions should remain consistent through save/load cycles"
        );
    }

    dir.close().expect("Failed to close temp dir");
}

#[test]
fn test_save_creates_file() {
    let (x, y) = generate_regression_data(50, 3);

    let mut model = NGBRegressor::new(20, 0.1);
    model.fit(&x, &y).expect("Fit should succeed");

    let dir = tempdir().expect("Failed to create temp dir");
    let path = dir.path().join("test_model.bin");
    let path_str = path.to_str().unwrap();

    // File should not exist before save
    assert!(!path.exists());

    model.save_model(path_str).expect("Save should succeed");

    // File should exist after save
    assert!(path.exists());

    // File should have content
    let metadata = std::fs::metadata(&path).expect("Should get metadata");
    assert!(metadata.len() > 0, "File should not be empty");

    dir.close().expect("Failed to close temp dir");
}
