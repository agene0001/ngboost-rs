//! Evaluation metrics for NGBoost models.
//!
//! This module provides functions for evaluating probabilistic predictions,
//! including calibration metrics and concordance indices for survival analysis.

use ndarray::Array1;

/// Result of calibration analysis.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// The predicted quantiles/percentiles.
    pub predicted: Array1<f64>,
    /// The observed proportions.
    pub observed: Array1<f64>,
    /// The slope of the calibration line.
    pub slope: f64,
    /// The intercept of the calibration line.
    pub intercept: f64,
}

impl CalibrationResult {
    /// Calculate the calibration error (sum of squared differences).
    pub fn calibration_error(&self) -> f64 {
        calculate_calib_error(&self.predicted, &self.observed)
    }

    /// Check if the model is well-calibrated (slope close to 1, intercept close to 0).
    pub fn is_well_calibrated(&self, slope_tol: f64, intercept_tol: f64) -> bool {
        (self.slope - 1.0).abs() <= slope_tol && self.intercept.abs() <= intercept_tol
    }
}

/// Calculate calibration in the regression setting.
///
/// Computes how well-calibrated the predicted distributions are by comparing
/// predicted quantiles to observed proportions.
///
/// # Arguments
/// * `ppf_fn` - Function that computes the percent point function (inverse CDF)
///              given a percentile value. Should return an Array1<f64> of quantiles.
/// * `y` - Observed values.
/// * `bins` - Number of bins/percentiles to evaluate (default: 11).
/// * `eps` - Small value to avoid edge effects (default: 1e-3).
///
/// # Returns
/// A `CalibrationResult` containing predicted percentiles, observed proportions,
/// and the fitted calibration line parameters.
///
/// # Example
/// ```ignore
/// use ngboost_rs::evaluation::calibration_regression;
/// use ndarray::Array1;
///
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let result = calibration_regression(
///     |p| ppf_values_at_percentile_p,
///     &y,
///     11,
///     1e-3
/// );
/// println!("Slope: {}, Intercept: {}", result.slope, result.intercept);
/// ```
pub fn calibration_regression<F>(
    ppf_fn: F,
    y: &Array1<f64>,
    bins: usize,
    eps: f64,
) -> CalibrationResult
where
    F: Fn(f64) -> Array1<f64>,
{
    let pctles: Vec<f64> = (0..bins)
        .map(|i| eps + (1.0 - 2.0 * eps) * (i as f64) / ((bins - 1) as f64))
        .collect();

    let mut observed = Vec::with_capacity(bins);

    for &pctle in &pctles {
        let icdfs = ppf_fn(pctle);
        let count_below: usize = y
            .iter()
            .zip(icdfs.iter())
            .filter(|&(yi, qi)| yi < qi)
            .count();
        observed.push(count_below as f64 / y.len() as f64);
    }

    let pctles_arr = Array1::from_vec(pctles);
    let observed_arr = Array1::from_vec(observed);

    let (slope, intercept) = polyfit_1(&pctles_arr, &observed_arr);

    CalibrationResult {
        predicted: pctles_arr,
        observed: observed_arr,
        slope,
        intercept,
    }
}

/// Calculate calibration in the time-to-event (survival) setting.
///
/// Uses the probability integral transform and Kaplan-Meier estimation
/// to assess calibration of survival predictions.
///
/// # Arguments
/// * `cdf_at_t` - CDF values at the observed times (F(T) for each observation).
/// * `event` - Event indicators (true = event occurred, false = censored).
///
/// # Returns
/// A `CalibrationResult` containing the calibration analysis.
pub fn calibration_time_to_event(
    cdf_at_t: &Array1<f64>,
    event: &Array1<bool>,
) -> CalibrationResult {
    // Compute Kaplan-Meier estimate on the CDF values
    // The idea: if well-calibrated, CDF(T) should be uniform on [0,1] for uncensored
    let km_result = kaplan_meier(cdf_at_t, event);

    // Sample at 11 evenly spaced points
    let n_points = 11;
    let predicted: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points - 1) as f64)
        .collect();

    let mut observed = Vec::with_capacity(n_points);
    for &p in &predicted {
        // Find the survival probability at this CDF value
        let survival = interpolate_km(&km_result, p);
        observed.push(1.0 - survival);
    }

    let predicted_arr = Array1::from_vec(predicted);
    let observed_arr = Array1::from_vec(observed);

    let (slope, intercept) = polyfit_1(&predicted_arr, &observed_arr);

    CalibrationResult {
        predicted: predicted_arr,
        observed: observed_arr,
        slope,
        intercept,
    }
}

/// Calculate calibration error as sum of squared differences.
///
/// # Arguments
/// * `predicted` - Predicted values/quantiles.
/// * `observed` - Observed proportions.
///
/// # Returns
/// The mean squared calibration error.
pub fn calculate_calib_error(predicted: &Array1<f64>, observed: &Array1<f64>) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = predicted
        .iter()
        .zip(observed.iter())
        .map(|(p, o)| (p - o).powi(2))
        .sum();
    sum_sq / n as f64
}

/// Data for a PIT (Probability Integral Transform) histogram.
#[derive(Debug, Clone)]
pub struct PITHistogramData {
    /// Bin edges.
    pub bin_edges: Array1<f64>,
    /// Density values for each bin.
    pub densities: Array1<f64>,
    /// Expected uniform density (1 / (n_bins)).
    pub expected_density: f64,
}

/// Compute PIT histogram data.
///
/// The PIT histogram shows how well-calibrated a probabilistic forecast is.
/// For a well-calibrated model, the histogram should be approximately uniform.
///
/// # Arguments
/// * `cdf_values` - CDF evaluated at the observed values (F(y) for each y).
/// * `n_bins` - Number of bins for the histogram (default: 10).
///
/// # Returns
/// PIT histogram data including bin edges and densities.
pub fn pit_histogram(cdf_values: &Array1<f64>, n_bins: usize) -> PITHistogramData {
    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();

    let mut counts = vec![0usize; n_bins];
    let n = cdf_values.len();

    for &cdf in cdf_values.iter() {
        let bin_idx = ((cdf * n_bins as f64).floor() as usize).min(n_bins - 1);
        counts[bin_idx] += 1;
    }

    let densities: Vec<f64> = counts
        .iter()
        .map(|&c| c as f64 / n as f64 * n_bins as f64)
        .collect();

    PITHistogramData {
        bin_edges: Array1::from_vec(bin_edges),
        densities: Array1::from_vec(densities),
        expected_density: 1.0,
    }
}

/// Data for a calibration curve plot.
#[derive(Debug, Clone)]
pub struct CalibrationCurveData {
    /// Predicted probabilities/quantiles.
    pub predicted: Array1<f64>,
    /// Observed proportions.
    pub observed: Array1<f64>,
    /// Fitted line x-values.
    pub fit_x: Array1<f64>,
    /// Fitted line y-values.
    pub fit_y: Array1<f64>,
    /// Slope of the calibration line.
    pub slope: f64,
    /// Intercept of the calibration line.
    pub intercept: f64,
}

/// Compute calibration curve data for plotting.
///
/// # Arguments
/// * `predicted` - Predicted probabilities/quantiles.
/// * `observed` - Observed proportions.
///
/// # Returns
/// Data for plotting a calibration curve.
pub fn calibration_curve_data(
    predicted: &Array1<f64>,
    observed: &Array1<f64>,
) -> CalibrationCurveData {
    let (slope, intercept) = polyfit_1(predicted, observed);

    let fit_x = Array1::linspace(0.0, 1.0, 50);
    let fit_y = fit_x.mapv(|x| slope * x + intercept);

    CalibrationCurveData {
        predicted: predicted.clone(),
        observed: observed.clone(),
        fit_x,
        fit_y,
        slope,
        intercept,
    }
}

/// Calculate Harrell's C-statistic (concordance index) with censoring support.
///
/// The concordance index measures the ability of a model to correctly rank
/// pairs of observations by their predicted risk/time.
///
/// # Comparable Pairs
/// - Both uncensored: can compare
/// - One censored, one not: can compare if censored time > uncensored time
/// - Both censored: cannot compare
///
/// # Arguments
/// * `predictions` - Predicted risk scores or times (higher = higher risk).
/// * `times` - Observed times to event or censoring.
/// * `events` - Event indicators (true = event occurred, false = censored).
///
/// # Returns
/// The concordance index in [0, 1]. A value of 0.5 indicates random predictions,
/// while 1.0 indicates perfect concordance.
pub fn concordance_index(
    predictions: &Array1<f64>,
    times: &Array1<f64>,
    events: &Array1<bool>,
) -> f64 {
    let n = times.len();
    let mut concordant = 0.0;
    let mut total_comparable = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let e_i = events[i];
            let e_j = events[j];
            let t_i = times[i];
            let t_j = times[j];
            let p_i = predictions[i];
            let p_j = predictions[j];

            // Determine if this pair is comparable
            let comparable = if e_i && e_j {
                // Both uncensored: always comparable
                true
            } else if e_i && !e_j && t_i < t_j {
                // i uncensored, j censored, and i's event time < j's censoring time
                true
            } else if !e_i && e_j && t_i > t_j {
                // i censored, j uncensored, and i's censoring time > j's event time
                true
            } else {
                false
            };

            if comparable {
                total_comparable += 1.0;

                // Compare predictions based on true ordering
                // For survival: lower predicted time (or higher risk) = earlier event
                if (t_i < t_j && p_i > p_j) || (t_i > t_j && p_i < p_j) {
                    concordant += 1.0;
                } else if (p_i - p_j).abs() < 1e-10 {
                    // Tie in predictions
                    concordant += 0.5;
                }
            }
        }
    }

    if total_comparable == 0.0 {
        return 0.5; // No comparable pairs
    }

    concordant / total_comparable
}

/// Calculate concordance index considering only uncensored observations.
///
/// This is a simplified version that ignores censored observations entirely.
///
/// # Arguments
/// * `predictions` - Predicted risk scores or times.
/// * `times` - Observed times to event.
/// * `events` - Event indicators (true = event occurred, false = censored).
///
/// # Returns
/// The concordance index computed only on uncensored pairs.
pub fn concordance_index_uncensored_only(
    predictions: &Array1<f64>,
    times: &Array1<f64>,
    events: &Array1<bool>,
) -> f64 {
    // Filter to only uncensored observations
    let uncensored_indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|&(_, e)| *e)
        .map(|(i, _)| i)
        .collect();

    let n = uncensored_indices.len();
    if n < 2 {
        return 0.5;
    }

    let mut concordant = 0.0;
    let mut total = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let idx_i = uncensored_indices[i];
            let idx_j = uncensored_indices[j];

            let t_i = times[idx_i];
            let t_j = times[idx_j];
            let p_i = predictions[idx_i];
            let p_j = predictions[idx_j];

            total += 1.0;

            if (t_i < t_j && p_i > p_j) || (t_i > t_j && p_i < p_j) {
                concordant += 1.0;
            } else if (p_i - p_j).abs() < 1e-10 {
                concordant += 0.5;
            }
        }
    }

    if total == 0.0 {
        return 0.5;
    }

    concordant / total
}

/// Compute the Brier score for probabilistic predictions.
///
/// The Brier score measures the accuracy of probabilistic predictions.
/// Lower is better, with 0 being perfect predictions.
///
/// # Arguments
/// * `predicted_probs` - Predicted probabilities.
/// * `outcomes` - Binary outcomes (0 or 1).
///
/// # Returns
/// The Brier score.
pub fn brier_score(predicted_probs: &Array1<f64>, outcomes: &Array1<f64>) -> f64 {
    let n = predicted_probs.len();
    if n == 0 {
        return 0.0;
    }

    let sum_sq: f64 = predicted_probs
        .iter()
        .zip(outcomes.iter())
        .map(|(p, o)| (p - o).powi(2))
        .sum();

    sum_sq / n as f64
}

/// Compute the log loss (cross-entropy) for probabilistic predictions.
///
/// # Arguments
/// * `predicted_probs` - Predicted probabilities (should be in (0, 1)).
/// * `outcomes` - Binary outcomes (0 or 1).
/// * `eps` - Small value to avoid log(0) (default: 1e-15).
///
/// # Returns
/// The log loss.
pub fn log_loss(predicted_probs: &Array1<f64>, outcomes: &Array1<f64>, eps: f64) -> f64 {
    let n = predicted_probs.len();
    if n == 0 {
        return 0.0;
    }

    let sum: f64 = predicted_probs
        .iter()
        .zip(outcomes.iter())
        .map(|(&p, &o)| {
            let p_clamped = p.clamp(eps, 1.0 - eps);
            -o * p_clamped.ln() - (1.0 - o) * (1.0 - p_clamped).ln()
        })
        .sum();

    sum / n as f64
}

/// Compute the mean absolute error.
pub fn mean_absolute_error(predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).abs())
        .sum();
    sum / n as f64
}

/// Compute the mean squared error.
pub fn mean_squared_error(predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum();
    sum / n as f64
}

/// Compute the root mean squared error.
pub fn root_mean_squared_error(predicted: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    mean_squared_error(predicted, actual).sqrt()
}

// ============================================================================
// Helper functions
// ============================================================================

/// Simple linear regression to fit a line y = slope * x + intercept.
fn polyfit_1(x: &Array1<f64>, y: &Array1<f64>) -> (f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (1.0, 0.0);
    }

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let denom = sum_x2 - n * mean_x * mean_x;
    if denom.abs() < 1e-15 {
        return (1.0, mean_y - mean_x);
    }

    let slope = (sum_xy - n * mean_x * mean_y) / denom;
    let intercept = mean_y - slope * mean_x;

    (slope, intercept)
}

/// Kaplan-Meier estimate result.
struct KaplanMeierResult {
    /// Unique event times.
    times: Vec<f64>,
    /// Survival probabilities at each time.
    survival: Vec<f64>,
}

/// Compute Kaplan-Meier survival estimate.
fn kaplan_meier(times: &Array1<f64>, events: &Array1<bool>) -> KaplanMeierResult {
    // Sort by time
    let mut indices: Vec<usize> = (0..times.len()).collect();
    indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

    let mut unique_times = Vec::new();
    let mut survival_probs = Vec::new();

    let mut at_risk = times.len();
    let mut survival = 1.0;

    let mut i = 0;
    while i < indices.len() {
        let idx = indices[i];
        let t = times[idx];

        // Count events and censored at this time
        let mut n_events = 0;
        let mut n_at_time = 0;

        while i < indices.len() && (times[indices[i]] - t).abs() < 1e-10 {
            if events[indices[i]] {
                n_events += 1;
            }
            n_at_time += 1;
            i += 1;
        }

        if n_events > 0 && at_risk > 0 {
            survival *= 1.0 - (n_events as f64 / at_risk as f64);
        }

        unique_times.push(t);
        survival_probs.push(survival);

        at_risk -= n_at_time;
    }

    KaplanMeierResult {
        times: unique_times,
        survival: survival_probs,
    }
}

/// Interpolate Kaplan-Meier survival function at a given time.
fn interpolate_km(km: &KaplanMeierResult, t: f64) -> f64 {
    if km.times.is_empty() {
        return 1.0;
    }

    if t <= km.times[0] {
        return 1.0;
    }

    for i in 0..km.times.len() {
        if t <= km.times[i] {
            return km.survival[i.saturating_sub(1)];
        }
    }

    *km.survival.last().unwrap_or(&0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_calculate_calib_error() {
        let predicted = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let observed = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        assert_relative_eq!(
            calculate_calib_error(&predicted, &observed),
            0.0,
            epsilon = 1e-10
        );

        let observed_off = Array1::from_vec(vec![0.2, 0.3, 0.4, 0.5, 0.6]);
        let error = calculate_calib_error(&predicted, &observed_off);
        assert_relative_eq!(error, 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_polyfit_1() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);
        let (slope, intercept) = polyfit_1(&x, &y);
        assert_relative_eq!(slope, 2.0, epsilon = 1e-10);
        assert_relative_eq!(intercept, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pit_histogram() {
        // Well-calibrated predictions should give uniform PIT
        let cdf_values = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]);
        let result = pit_histogram(&cdf_values, 10);
        assert_eq!(result.densities.len(), 10);
        assert_eq!(result.bin_edges.len(), 11);
        assert_relative_eq!(result.expected_density, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_concordance_index_perfect() {
        // Perfect concordance: predictions match true ordering
        let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let events = Array1::from_vec(vec![true, true, true, true, true]);

        let c_index = concordance_index(&predictions, &times, &events);
        assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_concordance_index_random() {
        // Random/independent predictions should give ~0.5
        let predictions = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let events = Array1::from_vec(vec![true, true, true, true, true]);

        let c_index = concordance_index(&predictions, &times, &events);
        assert_relative_eq!(c_index, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_concordance_index_with_censoring() {
        // Test with some censored observations
        let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let events = Array1::from_vec(vec![true, false, true, false, true]);

        let c_index = concordance_index(&predictions, &times, &events);
        assert!(c_index >= 0.0 && c_index <= 1.0);
    }

    #[test]
    fn test_brier_score() {
        // Perfect predictions
        let predicted = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        let outcomes = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        assert_relative_eq!(brier_score(&predicted, &outcomes), 0.0, epsilon = 1e-10);

        // Worst predictions
        let predicted = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let outcomes = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        assert_relative_eq!(brier_score(&predicted, &outcomes), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_loss() {
        // Perfect confident predictions
        let predicted = Array1::from_vec(vec![0.99, 0.01]);
        let outcomes = Array1::from_vec(vec![1.0, 0.0]);
        let loss = log_loss(&predicted, &outcomes, 1e-15);
        assert!(loss < 0.1);
    }

    #[test]
    fn test_mean_squared_error() {
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_relative_eq!(
            mean_squared_error(&predicted, &actual),
            0.0,
            epsilon = 1e-10
        );

        let actual = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert_relative_eq!(
            mean_squared_error(&predicted, &actual),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_mean_absolute_error() {
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let actual = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        assert_relative_eq!(
            mean_absolute_error(&predicted, &actual),
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_calibration_result() {
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
    fn test_kaplan_meier() {
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let events = Array1::from_vec(vec![true, false, true, false, true]);

        let km = kaplan_meier(&times, &events);
        assert_eq!(km.times.len(), 5);
        assert!(km.survival[0] < 1.0);
        assert!(km.survival.last().unwrap() < &km.survival[0]);
    }

    #[test]
    fn test_concordance_uncensored_only() {
        let predictions = Array1::from_vec(vec![5.0, 4.0, 3.0]);
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let events = Array1::from_vec(vec![true, true, true]);

        let c_index = concordance_index_uncensored_only(&predictions, &times, &events);
        assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);
    }
}
