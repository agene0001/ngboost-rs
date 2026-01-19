//! Tests for the evaluation module.

use approx::assert_relative_eq;
use ndarray::Array1;
use ngboost_rs::evaluation::{
    brier_score, calculate_calib_error, calibration_curve_data, calibration_regression,
    concordance_index, concordance_index_uncensored_only, log_loss, mean_absolute_error,
    mean_squared_error, pit_histogram, root_mean_squared_error, CalibrationResult,
};

// ============================================================================
// Calibration Error Tests
// ============================================================================

#[test]
fn test_calib_error_perfect() {
    let predicted = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    let observed = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

    let error = calculate_calib_error(&predicted, &observed);
    assert_relative_eq!(error, 0.0, epsilon = 1e-10);
}

#[test]
fn test_calib_error_constant_offset() {
    let predicted = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let observed = Array1::from_vec(vec![0.2, 0.3, 0.4, 0.5, 0.6]);

    // Each difference is 0.1, so squared is 0.01, mean is 0.01
    let error = calculate_calib_error(&predicted, &observed);
    assert_relative_eq!(error, 0.01, epsilon = 1e-10);
}

#[test]
fn test_calib_error_empty() {
    let predicted = Array1::from_vec(vec![]);
    let observed = Array1::from_vec(vec![]);

    let error = calculate_calib_error(&predicted, &observed);
    assert_relative_eq!(error, 0.0, epsilon = 1e-10);
}

// ============================================================================
// Calibration Regression Tests
// ============================================================================

#[test]
fn test_calibration_regression_well_calibrated() {
    // For a well-calibrated model, ppf(q) should give quantiles such that
    // proportion of y < ppf(q) ≈ q

    // Create simple "perfect calibration" scenario
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // PPF function that returns uniform quantiles
    let ppf_fn = |q: f64| -> Array1<f64> {
        // For uniform distribution on [0, 10), ppf(q) = 10*q
        Array1::from_elem(y.len(), 10.0 * q)
    };

    let result = calibration_regression(ppf_fn, &y, 11, 1e-3);

    // For well-calibrated model, slope should be close to 1
    assert!(result.slope > 0.8 && result.slope < 1.2);
    // Intercept should be close to 0
    assert!(result.intercept.abs() < 0.2);
}

#[test]
fn test_calibration_result_methods() {
    let result = CalibrationResult {
        predicted: Array1::from_vec(vec![0.1, 0.5, 0.9]),
        observed: Array1::from_vec(vec![0.1, 0.5, 0.9]),
        slope: 1.0,
        intercept: 0.0,
    };

    assert!(result.is_well_calibrated(0.1, 0.1));
    assert_relative_eq!(result.calibration_error(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_calibration_result_poorly_calibrated() {
    let result = CalibrationResult {
        predicted: Array1::from_vec(vec![0.1, 0.5, 0.9]),
        observed: Array1::from_vec(vec![0.3, 0.5, 0.7]),
        slope: 0.5,
        intercept: 0.25,
    };

    assert!(!result.is_well_calibrated(0.1, 0.1));
}

// ============================================================================
// Concordance Index Tests
// ============================================================================

#[test]
fn test_concordance_perfect() {
    // Perfect concordance: predictions perfectly rank the true times
    let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from_vec(vec![true, true, true, true, true]);

    let c_index = concordance_index(&predictions, &times, &events);
    assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);
}

#[test]
fn test_concordance_inverse() {
    // Inverse concordance: predictions are opposite of true times
    let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from_vec(vec![true, true, true, true, true]);

    let c_index = concordance_index(&predictions, &times, &events);
    assert_relative_eq!(c_index, 0.0, epsilon = 1e-10);
}

#[test]
fn test_concordance_random() {
    // Random/tied predictions should give ~0.5
    let predictions = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from_vec(vec![true, true, true, true, true]);

    let c_index = concordance_index(&predictions, &times, &events);
    assert_relative_eq!(c_index, 0.5, epsilon = 1e-10);
}

#[test]
fn test_concordance_with_censoring() {
    // Test with censored observations
    // (time=1, event) vs (time=3, censored) - comparable: event time < censoring time
    // (time=2, censored) vs (time=1, event) - comparable: censoring time > event time
    let predictions = Array1::from_vec(vec![3.0, 2.0, 1.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let events = Array1::from_vec(vec![true, false, true]);

    let c_index = concordance_index(&predictions, &times, &events);
    // Should be between 0 and 1, and finite
    assert!(c_index >= 0.0 && c_index <= 1.0);
    assert!(c_index.is_finite());
}

#[test]
fn test_concordance_all_censored() {
    // All censored - no comparable pairs
    let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let events = Array1::from_vec(vec![false, false, false]);

    let c_index = concordance_index(&predictions, &times, &events);
    // No comparable pairs, should return 0.5
    assert_relative_eq!(c_index, 0.5, epsilon = 1e-10);
}

#[test]
fn test_concordance_uncensored_only() {
    let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from_vec(vec![true, false, true, false, true]);

    let c_index = concordance_index_uncensored_only(&predictions, &times, &events);
    // Only uncensored: times [1, 3, 5] with predictions [5, 3, 1]
    // Perfect concordance for uncensored only
    assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);
}

#[test]
fn test_concordance_single_observation() {
    let predictions = Array1::from_vec(vec![1.0]);
    let times = Array1::from_vec(vec![1.0]);
    let events = Array1::from_vec(vec![true]);

    // With only one observation, no comparable pairs
    let c_index = concordance_index(&predictions, &times, &events);
    assert_relative_eq!(c_index, 0.5, epsilon = 1e-10);
}

// ============================================================================
// Brier Score Tests
// ============================================================================

#[test]
fn test_brier_score_perfect() {
    let predicted = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

    let score = brier_score(&predicted, &outcomes);
    assert_relative_eq!(score, 0.0, epsilon = 1e-10);
}

#[test]
fn test_brier_score_worst() {
    let predicted = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

    let score = brier_score(&predicted, &outcomes);
    assert_relative_eq!(score, 1.0, epsilon = 1e-10);
}

#[test]
fn test_brier_score_uncertain() {
    // 50% probability for all outcomes
    let predicted = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

    let score = brier_score(&predicted, &outcomes);
    // (0.5-1)^2 + (0.5-0)^2 + ... = 4 * 0.25 / 4 = 0.25
    assert_relative_eq!(score, 0.25, epsilon = 1e-10);
}

#[test]
fn test_brier_score_empty() {
    let predicted = Array1::from_vec(vec![]);
    let outcomes = Array1::from_vec(vec![]);

    let score = brier_score(&predicted, &outcomes);
    assert_relative_eq!(score, 0.0, epsilon = 1e-10);
}

// ============================================================================
// Log Loss Tests
// ============================================================================

#[test]
fn test_log_loss_confident_correct() {
    let predicted = Array1::from_vec(vec![0.99, 0.01]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0]);

    let loss = log_loss(&predicted, &outcomes, 1e-15);
    // Should be close to 0 for confident correct predictions
    assert!(loss < 0.05);
}

#[test]
fn test_log_loss_confident_wrong() {
    let predicted = Array1::from_vec(vec![0.01, 0.99]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0]);

    let loss = log_loss(&predicted, &outcomes, 1e-15);
    // Should be large for confident wrong predictions
    assert!(loss > 2.0);
}

#[test]
fn test_log_loss_uncertain() {
    let predicted = Array1::from_vec(vec![0.5, 0.5]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0]);

    let loss = log_loss(&predicted, &outcomes, 1e-15);
    // -ln(0.5) ≈ 0.693
    assert_relative_eq!(loss, std::f64::consts::LN_2, epsilon = 1e-6);
}

#[test]
fn test_log_loss_clamping() {
    // Test that extreme probabilities are clamped
    let predicted = Array1::from_vec(vec![0.0, 1.0]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0]);

    let loss = log_loss(&predicted, &outcomes, 1e-15);
    // Should be finite due to clamping
    assert!(loss.is_finite());
}

// ============================================================================
// MSE, MAE, RMSE Tests
// ============================================================================

#[test]
fn test_mse_perfect() {
    let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let actual = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    let mse = mean_squared_error(&predicted, &actual);
    assert_relative_eq!(mse, 0.0, epsilon = 1e-10);
}

#[test]
fn test_mse_constant_error() {
    let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let actual = Array1::from_vec(vec![2.0, 3.0, 4.0]);

    // Each error is 1, squared is 1, mean is 1
    let mse = mean_squared_error(&predicted, &actual);
    assert_relative_eq!(mse, 1.0, epsilon = 1e-10);
}

#[test]
fn test_mae_perfect() {
    let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let actual = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let mae = mean_absolute_error(&predicted, &actual);
    assert_relative_eq!(mae, 0.0, epsilon = 1e-10);
}

#[test]
fn test_mae_constant_error() {
    let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let actual = Array1::from_vec(vec![2.0, 3.0, 4.0]);

    let mae = mean_absolute_error(&predicted, &actual);
    assert_relative_eq!(mae, 1.0, epsilon = 1e-10);
}

#[test]
fn test_rmse_is_sqrt_mse() {
    let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let actual = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);

    let mse = mean_squared_error(&predicted, &actual);
    let rmse = root_mean_squared_error(&predicted, &actual);

    assert_relative_eq!(rmse, mse.sqrt(), epsilon = 1e-10);
}

#[test]
fn test_errors_empty() {
    let predicted = Array1::from_vec(vec![]);
    let actual = Array1::from_vec(vec![]);

    assert_relative_eq!(
        mean_squared_error(&predicted, &actual),
        0.0,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        mean_absolute_error(&predicted, &actual),
        0.0,
        epsilon = 1e-10
    );
    assert_relative_eq!(
        root_mean_squared_error(&predicted, &actual),
        0.0,
        epsilon = 1e-10
    );
}

// ============================================================================
// PIT Histogram Tests
// ============================================================================

#[test]
fn test_pit_histogram_uniform() {
    // For well-calibrated predictions, PIT values should be uniform
    let cdf_values = Array1::from_vec(vec![
        0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    ]);

    let result = pit_histogram(&cdf_values, 10);

    // Each bin should have 1 observation, so density = 1
    assert_eq!(result.densities.len(), 10);
    for density in result.densities.iter() {
        assert_relative_eq!(*density, 1.0, epsilon = 1e-10);
    }
    assert_relative_eq!(result.expected_density, 1.0, epsilon = 1e-10);
}

#[test]
fn test_pit_histogram_concentrated() {
    // All predictions in first bin
    let cdf_values = Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04, 0.05]);

    let result = pit_histogram(&cdf_values, 10);

    // First bin should have all 5 observations
    assert_relative_eq!(result.densities[0], 10.0, epsilon = 1e-10);
    // Other bins should be empty
    for i in 1..result.densities.len() {
        assert_relative_eq!(result.densities[i], 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_pit_histogram_bin_edges() {
    let cdf_values = Array1::from_vec(vec![0.5]);
    let result = pit_histogram(&cdf_values, 10);

    // Should have n_bins + 1 edges
    assert_eq!(result.bin_edges.len(), 11);

    // First edge should be 0, last should be 1
    assert_relative_eq!(result.bin_edges[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.bin_edges[10], 1.0, epsilon = 1e-10);
}

// ============================================================================
// Calibration Curve Data Tests
// ============================================================================

#[test]
fn test_calibration_curve_data() {
    let predicted = Array1::from_vec(vec![0.1, 0.3, 0.5, 0.7, 0.9]);
    let observed = Array1::from_vec(vec![0.15, 0.35, 0.45, 0.75, 0.85]);

    let data = calibration_curve_data(&predicted, &observed);

    // Check that fit line is computed
    assert!(data.slope.is_finite());
    assert!(data.intercept.is_finite());

    // Check that fit_x and fit_y have same length
    assert_eq!(data.fit_x.len(), data.fit_y.len());

    // Check that original data is preserved
    assert_eq!(data.predicted.len(), 5);
    assert_eq!(data.observed.len(), 5);
}

#[test]
fn test_calibration_curve_perfect_calibration() {
    let predicted = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let observed = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);

    let data = calibration_curve_data(&predicted, &observed);

    // Perfect calibration: slope ≈ 1, intercept ≈ 0
    assert_relative_eq!(data.slope, 1.0, epsilon = 1e-10);
    assert_relative_eq!(data.intercept, 0.0, epsilon = 1e-10);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_metrics_ordering() {
    // Better predictions should have better scores

    // Good predictions
    let good_pred = Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2]);
    let outcomes = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

    // Bad predictions
    let bad_pred = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);

    let good_brier = brier_score(&good_pred, &outcomes);
    let bad_brier = brier_score(&bad_pred, &outcomes);

    let good_log = log_loss(&good_pred, &outcomes, 1e-15);
    let bad_log = log_loss(&bad_pred, &outcomes, 1e-15);

    // Good predictions should have lower scores
    assert!(good_brier < bad_brier);
    assert!(good_log < bad_log);
}

#[test]
fn test_concordance_consistency() {
    // concordance_index should be >= concordance_index_uncensored_only
    // when there are censored observations (not always true, but often)
    let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let events = Array1::from_vec(vec![true, true, true, true, true]);

    let c_full = concordance_index(&predictions, &times, &events);
    let c_uncensored = concordance_index_uncensored_only(&predictions, &times, &events);

    // Both should be 1.0 for this perfect case
    assert_relative_eq!(c_full, c_uncensored, epsilon = 1e-10);
}

#[test]
fn test_real_world_scenario() {
    // Simulate a realistic survival analysis scenario
    let predictions = Array1::from_vec(vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.35]);
    let times = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 5.5]);
    let events = Array1::from_vec(vec![
        true, true, false, true, true, false, true, true, true, false,
    ]);

    let c_index = concordance_index(&predictions, &times, &events);

    // Should be between 0 and 1
    assert!(c_index >= 0.0 && c_index <= 1.0);
    // Should be reasonably good (>0.5) since predictions roughly follow time ordering
    assert!(c_index > 0.5);
}
