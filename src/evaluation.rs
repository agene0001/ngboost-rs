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
/// If `y` contains NaN, the observed proportions, slope, and intercept are NaN.
/// If `bins == 0`, an empty result with NaN slope/intercept is returned.
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
    if bins == 0 {
        return CalibrationResult {
            predicted: Array1::zeros(0),
            observed: Array1::zeros(0),
            slope: f64::NAN,
            intercept: f64::NAN,
        };
    }
    if y.iter().any(|v| v.is_nan()) {
        let pctles: Vec<f64> = (0..bins)
            .map(|i| eps + (1.0 - 2.0 * eps) * (i as f64) / ((bins.max(2) - 1) as f64))
            .collect();
        return CalibrationResult {
            predicted: Array1::from_vec(pctles),
            observed: Array1::from_elem(bins, f64::NAN),
            slope: f64::NAN,
            intercept: f64::NAN,
        };
    }
    // bins.max(2) keeps bins == 1 from dividing by zero (yields [eps], like
    // np.linspace(eps, 1-eps, 1)).
    let pctles: Vec<f64> = (0..bins)
        .map(|i| eps + (1.0 - 2.0 * eps) * (i as f64) / ((bins.max(2) - 1) as f64))
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
///
/// If `cdf_at_t` contains NaN, the observed proportions, slope, and intercept
/// are NaN.
pub fn calibration_time_to_event(
    cdf_at_t: &Array1<f64>,
    event: &Array1<bool>,
) -> CalibrationResult {
    if cdf_at_t.iter().any(|v| v.is_nan()) {
        let n_points = 11;
        let predicted: Vec<f64> = (0..n_points)
            .map(|i| i as f64 / (n_points - 1) as f64)
            .collect();
        return CalibrationResult {
            predicted: Array1::from_vec(predicted),
            observed: Array1::from_elem(n_points, f64::NAN),
            slope: f64::NAN,
            intercept: f64::NAN,
        };
    }

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
///
/// NaN entries in `cdf_values` are skipped; densities are normalized over the
/// non-NaN count (all-NaN or empty input yields NaN densities). `n_bins == 0`
/// returns an empty histogram.
pub fn pit_histogram(cdf_values: &Array1<f64>, n_bins: usize) -> PITHistogramData {
    if n_bins == 0 {
        return PITHistogramData {
            bin_edges: Array1::zeros(0),
            densities: Array1::zeros(0),
            expected_density: 1.0,
        };
    }

    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();

    let mut counts = vec![0usize; n_bins];
    let mut n = 0usize;

    for &cdf in cdf_values.iter() {
        if cdf.is_nan() {
            continue;
        }
        let bin_idx = ((cdf * n_bins as f64).floor() as usize).min(n_bins - 1);
        counts[bin_idx] += 1;
        n += 1;
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
/// while 1.0 indicates perfect concordance. Returns NaN if `predictions` or
/// `times` contain NaN.
///
/// # Algorithm
/// This implementation uses a sorting-based approach that is O(n log n) for uncensored
/// data. The general (censored) path is also O(n log n), using a Fenwick tree over
/// prediction ranks while sweeping observations in descending time order; it produces
/// counts identical to the naive pairwise definition.
pub fn concordance_index(
    predictions: &Array1<f64>,
    times: &Array1<f64>,
    events: &Array1<bool>,
) -> f64 {
    let n = times.len();
    if n < 2 {
        return 0.5;
    }

    // NaN times/predictions make pair comparability and concordance undefined;
    // surface that instead of silently counting NaN pairs as discordant.
    if times.iter().chain(predictions.iter()).any(|v| v.is_nan()) {
        return f64::NAN;
    }

    // Check if all observations are uncensored - allows for optimized algorithm
    let all_uncensored = events.iter().all(|&e| e);

    if all_uncensored {
        // Use optimized O(n log n) algorithm for uncensored data.
        // Returns None when times contain ties, which the inversion-counting
        // algorithm cannot exclude from the comparable-pair total; fall through
        // to the general path in that case so both paths agree.
        if let Some(c) = concordance_index_uncensored_fast(predictions, times) {
            return c;
        }
    }

    let (concordant, ties, comparable) =
        concordance_counts_censored_fenwick(predictions, times, events);

    if comparable == 0 {
        return 0.5;
    }

    // Both counts are well below 2^53, and 0.5 * ties is a multiple of 0.5,
    // so this sum (and hence the division) is exact and bit-identical to the
    // pairwise accumulation of 1.0 / 0.5 increments.
    (concordant as f64 + 0.5 * ties as f64) / comparable as f64
}

/// General-path concordance counts via a Fenwick (binary indexed) tree: O(n log n).
///
/// Semantics are exactly those of the naive pairwise loop over time-sorted pairs
/// (i earlier, j later):
/// - comparable iff `events[i] && times[i] < times[j]` (exact f64 `<`)
/// - concordant (+1) iff `p_i > p_j`
/// - tie credit (+0.5) iff `!(p_i > p_j) && (p_i - p_j).abs() < 1e-10`
///
/// Observations are swept in descending time order; each equal-time group (f64 `==`,
/// so -0.0 and +0.0 group together, matching `<`) is fully queried before being
/// inserted, so tied-time pairs never count. Predicates on prediction values are
/// evaluated verbatim on the sorted unique prediction values via `partition_point`
/// (they are monotone in v), so the epsilon window `(p_i - v).abs() < 1e-10` is
/// reproduced with the exact same floating-point subtraction as the pairwise loop.
///
/// Returns `(concordant_pairs, tie_credit_pairs, comparable_pairs)` as raw counts.
/// Callers must reject NaN inputs first.
fn concordance_counts_censored_fenwick(
    predictions: &Array1<f64>,
    times: &Array1<f64>,
    events: &Array1<bool>,
) -> (u64, u64, u64) {
    let n = times.len();

    // Sort indices by time (ascending), as the pairwise loop does.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    // Sorted unique prediction values. total_cmp ordering is consistent with `<`
    // once `==`-equal values (only -0.0 vs +0.0) are collapsed by dedup.
    let mut uniq: Vec<f64> = predictions.iter().copied().collect();
    uniq.sort_unstable_by(f64::total_cmp);
    uniq.dedup_by(|a, b| a == b);

    // Fenwick tree over prediction ranks (1-based internally).
    let mut tree = vec![0u64; uniq.len() + 1];
    let bit_prefix = |tree: &[u64], mut i: usize| -> u64 {
        // Count of inserted values with rank < i.
        let mut s = 0u64;
        while i > 0 {
            s += tree[i];
            i &= i - 1;
        }
        s
    };

    let mut inserted: u64 = 0;
    let mut concordant: u64 = 0;
    let mut ties: u64 = 0;
    let mut comparable: u64 = 0;

    // Sweep groups of `==`-equal times in descending time order. The BIT holds
    // exactly the observations with strictly later time (`times[i] < times[j]`).
    let mut pos = n;
    while pos > 0 {
        let t = times[indices[pos - 1]];
        let mut start = pos - 1;
        while start > 0 && times[indices[start - 1]] == t {
            start -= 1;
        }

        // Query phase: every already-inserted j is comparable with each event i here.
        for &i in &indices[start..pos] {
            if events[i] {
                let p_i = predictions[i];
                comparable += inserted;

                // Concordant: p_j < p_i (identical to p_i > p_j).
                let lo = uniq.partition_point(|&v| v < p_i);
                // Tie credit: p_j >= p_i within the epsilon window. The predicate is
                // the verbatim pairwise test and is monotone in v, so partition_point
                // finds the exact boundary.
                let hi = uniq.partition_point(|&v| v < p_i || (p_i - v).abs() < 1e-10);

                let below = bit_prefix(&tree, lo);
                concordant += below;
                ties += bit_prefix(&tree, hi) - below;
            }
        }

        // Insert phase: the whole group enters the BIT only after all its queries,
        // so tied-time pairs are never counted.
        for &j in &indices[start..pos] {
            let mut rank = uniq.partition_point(|&v| v < predictions[j]) + 1;
            while rank < tree.len() {
                tree[rank] += 1;
                rank += rank & rank.wrapping_neg();
            }
            inserted += 1;
        }

        pos = start;
    }

    (concordant, ties, comparable)
}

/// The original O(n²) pairwise loop, kept verbatim as the differential-test
/// reference for `concordance_counts_censored_fenwick`.
///
/// Returns `(concordant, total_comparable)` accumulated exactly as the old
/// `concordance_index` general path did.
#[cfg(test)]
fn concordance_counts_censored_bruteforce(
    predictions: &Array1<f64>,
    times: &Array1<f64>,
    events: &Array1<bool>,
) -> (f64, f64) {
    let n = times.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let mut concordant = 0.0;
    let mut total_comparable = 0.0;

    for (idx_i, &i) in indices.iter().enumerate() {
        let e_i = events[i];
        let t_i = times[i];
        let p_i = predictions[i];

        for &j in indices.iter().skip(idx_i + 1) {
            let e_j = events[j];
            let t_j = times[j];
            let p_j = predictions[j];

            let comparable = if e_i && e_j {
                t_i < t_j
            } else if e_i && !e_j {
                t_i < t_j
            } else {
                false
            };

            if comparable {
                total_comparable += 1.0;

                if p_i > p_j {
                    concordant += 1.0;
                } else if (p_i - p_j).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }
    }

    (concordant, total_comparable)
}

/// Fast concordance index for fully uncensored data using O(n log n) algorithm.
/// Uses merge sort to count inversions.
///
/// Returns `None` if `times` contains ties: tied-time pairs are not comparable
/// under Harrell's definition, and the inversion count cannot exclude them.
fn concordance_index_uncensored_fast(predictions: &Array1<f64>, times: &Array1<f64>) -> Option<f64> {
    let n = times.len();
    if n < 2 {
        return Some(0.5);
    }

    // Create pairs of (time, prediction, original_index) and sort by time
    let mut pairs: Vec<(f64, f64, usize)> = times
        .iter()
        .zip(predictions.iter())
        .enumerate()
        .map(|(i, (&t, &p))| (t, p, i))
        .collect();

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    if pairs.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }

    // Count concordant pairs using merge sort inversion counting
    // After sorting by time, we need to count how many pairs have
    // predictions in the correct order (higher risk = lower time)
    let preds_sorted_by_time: Vec<f64> = pairs.iter().map(|p| p.1).collect();

    // Count inversions in predictions (where earlier time has higher prediction = concordant)
    let (concordant, ties, total) = count_concordant_pairs(&preds_sorted_by_time);

    if total == 0.0 {
        return Some(0.5);
    }

    Some((concordant + 0.5 * ties) / total)
}

/// Count concordant pairs, ties, and total comparable pairs using O(n log n) merge sort.
fn count_concordant_pairs(predictions: &[f64]) -> (f64, f64, f64) {
    let n = predictions.len();
    if n <= 1 {
        return (0.0, 0.0, 0.0);
    }

    // For survival: earlier observation (smaller index after time sort) should have
    // higher risk score (higher prediction) for concordance
    // So concordant = prediction[i] > prediction[j] for i < j

    // Use a simple O(n log n) approach: for each element, count how many
    // elements to its right are smaller (concordant) or equal (ties)

    let mut concordant = 0.0;
    let mut ties = 0.0;
    let total = (n * (n - 1) / 2) as f64;

    // For smaller arrays, use direct counting (cache-friendly)
    if n < 100 {
        for i in 0..n {
            for j in (i + 1)..n {
                if predictions[i] > predictions[j] {
                    concordant += 1.0;
                } else if (predictions[i] - predictions[j]).abs() < 1e-10 {
                    ties += 1.0;
                }
            }
        }
    } else {
        // For larger arrays, use merge sort based counting
        let mut sorted: Vec<(f64, usize)> = predictions
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i))
            .collect();
        let mut temp = vec![(0.0, 0); n];
        concordant = merge_sort_count(&mut sorted, &mut temp, 0, n);

        // Count ties separately (merge sort doesn't handle ties well)
        // This is still O(n log n) with a sorted array
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut i = 0;
        while i < n {
            let mut j = i + 1;
            while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-10 {
                j += 1;
            }
            let group_size = j - i;
            if group_size > 1 {
                // Ties within this group that come from different original positions
                // Each pair of ties where original_i < original_j counts
                for k in i..j {
                    for l in (k + 1)..j {
                        if sorted[k].1 < sorted[l].1 {
                            ties += 1.0;
                        }
                    }
                }
            }
            i = j;
        }
    }

    (concordant, ties, total)
}

/// Merge sort that counts inversions (concordant pairs for our use case).
fn merge_sort_count(
    arr: &mut [(f64, usize)],
    temp: &mut [(f64, usize)],
    left: usize,
    right: usize,
) -> f64 {
    let mut count = 0.0;
    if right - left > 1 {
        let mid = left + (right - left) / 2;
        count += merge_sort_count(arr, temp, left, mid);
        count += merge_sort_count(arr, temp, mid, right);
        count += merge_count(arr, temp, left, mid, right);
    }
    count
}

/// Merge two sorted halves and count inversions.
fn merge_count(
    arr: &mut [(f64, usize)],
    temp: &mut [(f64, usize)],
    left: usize,
    mid: usize,
    right: usize,
) -> f64 {
    let mut i = left;
    let mut j = mid;
    let mut k = left;
    let mut count = 0.0;

    while i < mid && j < right {
        // arr[i].1 is original index, arr[i].0 is prediction
        // We want to count pairs where original_index[i] < original_index[j] and pred[i] > pred[j]
        // After sorting by original index... actually we need to be more careful here

        // For concordance: if arr[i].1 < arr[j].1 (i comes before j in time order)
        // and arr[i].0 > arr[j].0 (i has higher prediction), that's concordant
        if arr[i].0 > arr[j].0 {
            // All remaining elements in left half are concordant with arr[j]
            // because they all have higher predictions and earlier original indices
            count += (mid - i) as f64;
            temp[k] = arr[j];
            j += 1;
        } else {
            temp[k] = arr[i];
            i += 1;
        }
        k += 1;
    }

    while i < mid {
        temp[k] = arr[i];
        i += 1;
        k += 1;
    }

    while j < right {
        temp[k] = arr[j];
        j += 1;
        k += 1;
    }

    arr[left..right].copy_from_slice(&temp[left..right]);
    count
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
/// The concordance index computed only on uncensored pairs. Returns NaN if any
/// uncensored observation has a NaN prediction or time.
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

    // NaN comparisons would silently count as discordant; surface them instead.
    if uncensored_indices
        .iter()
        .any(|&i| times[i].is_nan() || predictions[i].is_nan())
    {
        return f64::NAN;
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
///
/// Callers must reject NaN times first (see `calibration_time_to_event`);
/// `total_cmp` merely keeps the sort panic-free.
fn kaplan_meier(times: &Array1<f64>, events: &Array1<bool>) -> KaplanMeierResult {
    // Sort by time
    let mut indices: Vec<usize> = (0..times.len()).collect();
    indices.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

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
///
/// KM is a right-continuous step function: S(t) = P(T > t), so at an exact
/// event time the post-drop value applies.
fn interpolate_km(km: &KaplanMeierResult, t: f64) -> f64 {
    if km.times.is_empty() || t < km.times[0] {
        return 1.0;
    }

    for i in 0..km.times.len() {
        if t < km.times[i] {
            return km.survival[i - 1];
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
    fn test_concordance_index_tied_times_excluded() {
        // Tied event times are not comparable pairs (Harrell). With the only
        // distinct-time pairs being (1,2) and (1,2)', both concordant:
        // pairs: (0,1) tied -> excluded; (0,2) t 1<2, p 3>1 concordant;
        // (1,2) t 1<2, p 2>1 concordant => C = 1.0
        let predictions = Array1::from_vec(vec![3.0, 2.0, 1.0]);
        let times = Array1::from_vec(vec![1.0, 1.0, 2.0]);
        let events = Array1::from_vec(vec![true, true, true]);

        let c_index = concordance_index(&predictions, &times, &events);
        assert_relative_eq!(c_index, 1.0, epsilon = 1e-10);

        // Only-tied-times data has zero comparable pairs -> 0.5 by convention
        let predictions = Array1::from_vec(vec![1.0, 2.0]);
        let times = Array1::from_vec(vec![5.0, 5.0]);
        let events = Array1::from_vec(vec![true, true]);
        let c_index = concordance_index(&predictions, &times, &events);
        assert_relative_eq!(c_index, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_concordance_index_fast_matches_general_loop() {
        // Untied times take the fast inversion-count path; it must equal a
        // brute-force pairwise computation of Harrell's C.
        let predictions: Array1<f64> = Array1::from_vec(vec![2.0, 7.0, 1.0, 5.0, 3.0, 6.0, 4.0]);
        let times: Array1<f64> = Array1::from_vec(vec![3.0, 1.0, 7.0, 2.0, 5.0, 4.0, 6.0]);
        let events = Array1::from_vec(vec![true; 7]);

        let mut concordant = 0.0;
        let mut total = 0.0;
        for i in 0..7 {
            for j in 0..7 {
                if times[i] < times[j] {
                    total += 1.0;
                    if predictions[i] > predictions[j] {
                        concordant += 1.0;
                    } else if (predictions[i] - predictions[j]).abs() < 1e-10 {
                        concordant += 0.5;
                    }
                }
            }
        }
        let expected = concordant / total;

        let c_index = concordance_index(&predictions, &times, &events);
        assert_relative_eq!(c_index, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_km_right_continuous() {
        // Events at t=1,2 with no censoring: S(1)=0.5 (post-drop), S(0.5)=1.0,
        // S(1.5)=0.5, S(2)=0.0
        let times = Array1::from_vec(vec![1.0, 2.0]);
        let events = Array1::from_vec(vec![true, true]);
        let km = kaplan_meier(&times, &events);

        assert_relative_eq!(interpolate_km(&km, 0.5), 1.0, epsilon = 1e-10);
        assert_relative_eq!(interpolate_km(&km, 1.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(interpolate_km(&km, 1.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(interpolate_km(&km, 2.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(interpolate_km(&km, 3.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nan_inputs_yield_nan_not_panic() {
        // concordance_index: NaN time (uncensored fast path) and NaN prediction
        let preds = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let times_nan = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let events = Array1::from_vec(vec![true, true, true]);
        assert!(concordance_index(&preds, &times_nan, &events).is_nan());

        let preds_nan = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(concordance_index(&preds_nan, &times, &events).is_nan());

        // censored path (sorts by time) must not panic either
        let events_cens = Array1::from_vec(vec![true, false, true]);
        assert!(concordance_index(&preds, &times_nan, &events_cens).is_nan());

        // uncensored_only: NaN on an uncensored row -> NaN; NaN only on a
        // censored row is ignored (that row never enters a pair)
        assert!(concordance_index_uncensored_only(&preds_nan, &times, &events).is_nan());
        let preds_nan_censored_row = Array1::from_vec(vec![3.0, f64::NAN, 1.0]);
        let c = concordance_index_uncensored_only(&preds_nan_censored_row, &times, &events_cens);
        assert_relative_eq!(c, 1.0, epsilon = 1e-10);

        // calibration_time_to_event: NaN CDF values -> NaN slope, no panic
        let cdf_nan = Array1::from_vec(vec![0.2, f64::NAN, 0.8]);
        let result = calibration_time_to_event(&cdf_nan, &events);
        assert!(result.slope.is_nan());
        assert!(result.intercept.is_nan());
        assert!(result.observed.iter().all(|v| v.is_nan()));
        assert_eq!(result.predicted.len(), 11);

        // calibration_regression: NaN y -> NaN observed/slope, percentile grid intact
        let y_nan = Array1::from_vec(vec![1.0, f64::NAN]);
        let result = calibration_regression(|_p| Array1::zeros(2), &y_nan, 11, 1e-3);
        assert!(result.slope.is_nan());
        assert!(result.observed.iter().all(|v| v.is_nan()));
        assert_eq!(result.predicted.len(), 11);
        assert_relative_eq!(result.predicted[0], 1e-3, epsilon = 1e-12);
    }

    #[test]
    fn test_degenerate_bin_counts() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // bins == 0: empty result, no usize underflow
        let result = calibration_regression(|_p| Array1::zeros(3), &y, 0, 1e-3);
        assert_eq!(result.predicted.len(), 0);
        assert!(result.slope.is_nan());

        // bins == 1: [eps] like np.linspace(eps, 1-eps, 1), not NaN
        let result = calibration_regression(|_p| Array1::zeros(3), &y, 1, 1e-3);
        assert_eq!(result.predicted.len(), 1);
        assert_relative_eq!(result.predicted[0], 1e-3, epsilon = 1e-12);

        // pit_histogram n_bins == 0: empty, no underflow
        let cdf = Array1::from_vec(vec![0.1, 0.5, 0.9]);
        let hist = pit_histogram(&cdf, 0);
        assert_eq!(hist.densities.len(), 0);
        assert_eq!(hist.bin_edges.len(), 0);
    }

    #[test]
    fn test_pit_histogram_skips_nan() {
        // NaN used to be cast to bin 0 (`NaN as usize` == 0), inflating it.
        let cdf = Array1::from_vec(vec![0.95, f64::NAN, 0.95, f64::NAN]);
        let hist = pit_histogram(&cdf, 10);
        // both valid values in the last bin: density = 2/2 * 10
        assert_relative_eq!(hist.densities[9], 10.0, epsilon = 1e-10);
        assert_relative_eq!(hist.densities[0], 0.0, epsilon = 1e-10);

        // all-NaN input: densities NaN, no panic
        let cdf = Array1::from_vec(vec![f64::NAN, f64::NAN]);
        let hist = pit_histogram(&cdf, 4);
        assert!(hist.densities.iter().all(|v| v.is_nan()));
    }

    /// Minimal deterministic RNG (LCG, MMIX constants) so the differential test
    /// needs no external dependencies and is reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // xorshift the high bits down to decorrelate low bits of an LCG
            let x = self.0;
            (x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd) >> 11
        }

        fn next_f64(&mut self) -> f64 {
            // 53 random bits in [0, 1)
            (self.next_u64() & ((1u64 << 53) - 1)) as f64 / (1u64 << 53) as f64
        }

        fn gen_range(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// C-index exactly as the Fenwick counts define it (mirrors concordance_index's
    /// final arithmetic).
    fn c_from_fenwick_counts(concordant: u64, ties: u64, comparable: u64) -> f64 {
        if comparable == 0 {
            0.5
        } else {
            (concordant as f64 + 0.5 * ties as f64) / comparable as f64
        }
    }

    #[test]
    fn test_concordance_censored_fenwick_differential() {
        // 240 randomized trials engineered to stress every exactness hazard:
        // - epsilon-window prediction deltas straddling the 1e-10 boundary
        // - exact prediction ties (delta = 0) and grid-valued predictions
        // - exact time ties (grid times and copied times)
        // - censoring rates 0%, 30%, 70%, 100%
        // The Fenwick path must reproduce the O(n^2) loop BIT-FOR-BIT.
        let deltas = [0.0, 1e-11, 9.9e-11, 1.00001e-10, 2e-10, -5e-11];

        for trial in 0u64..240 {
            let mut rng = Lcg(0x9E3779B97F4A7C15u64.wrapping_mul(trial + 1) ^ 0xD1B54A32D192ED03);
            let n = 5 + rng.gen_range(296);
            let censor_rate = [0.0, 0.3, 0.7, 1.0][(trial % 4) as usize];

            // Times: every third trial draws from a tiny grid (many exact ties);
            // otherwise continuous, with ~n/5 exact ties injected by copying.
            let mut times: Vec<f64> = if trial % 3 == 0 {
                let grid = 2 + rng.gen_range(8);
                (0..n).map(|_| rng.gen_range(grid) as f64).collect()
            } else {
                (0..n).map(|_| rng.next_f64() * 10.0).collect()
            };
            for _ in 0..n / 5 {
                let a = rng.gen_range(n);
                let b = rng.gen_range(n);
                if a != b {
                    times[b] = times[a];
                }
            }

            // Predictions: base draw (continuous, or coarse grid every fourth trial
            // to force exact equality ties), then engineer epsilon-window pairs.
            let mut preds: Vec<f64> = if trial % 4 == 1 {
                (0..n).map(|_| (rng.gen_range(7) as f64) * 0.1).collect()
            } else {
                (0..n).map(|_| rng.next_f64()).collect()
            };
            for _ in 0..n / 3 {
                let i = rng.gen_range(n);
                let j = rng.gen_range(n);
                if i != j {
                    preds[j] = preds[i] + deltas[rng.gen_range(deltas.len())];
                }
            }

            let events: Vec<bool> = (0..n).map(|_| rng.next_f64() >= censor_rate).collect();

            let predictions = Array1::from_vec(preds);
            let times = Array1::from_vec(times);
            let events = Array1::from_vec(events);

            let (old_conc, old_comp) =
                concordance_counts_censored_bruteforce(&predictions, &times, &events);
            let (fc, ft, fp) =
                concordance_counts_censored_fenwick(&predictions, &times, &events);

            // Raw counts must match exactly (float accumulation of multiples of
            // 0.5 below 2^53 is exact, so f64 == is a legitimate exact check).
            let new_conc = fc as f64 + 0.5 * ft as f64;
            assert_eq!(
                new_conc.to_bits(),
                old_conc.to_bits(),
                "trial {trial}: concordant mismatch (fenwick {new_conc} vs brute {old_conc})"
            );
            assert_eq!(
                (fp as f64).to_bits(),
                old_comp.to_bits(),
                "trial {trial}: comparable mismatch (fenwick {fp} vs brute {old_comp})"
            );

            // Final C-index must be bit-identical.
            let old_c = if old_comp == 0.0 { 0.5 } else { old_conc / old_comp };
            let new_c = c_from_fenwick_counts(fc, ft, fp);
            assert_eq!(
                new_c.to_bits(),
                old_c.to_bits(),
                "trial {trial}: C-index mismatch (fenwick {new_c} vs brute {old_c})"
            );

            // The public function must agree too whenever it takes the general
            // path (any censoring, or any exact time tie).
            let has_censoring = events.iter().any(|&e| !e);
            let mut ts: Vec<f64> = times.to_vec();
            ts.sort_by(f64::total_cmp);
            let has_time_ties = ts.windows(2).any(|w| w[0] == w[1]);
            if has_censoring || has_time_ties {
                let public_c = concordance_index(&predictions, &times, &events);
                assert_eq!(
                    public_c.to_bits(),
                    old_c.to_bits(),
                    "trial {trial}: public concordance_index mismatch"
                );
            }
        }
    }

    #[test]
    fn test_concordance_censored_fenwick_edge_cases() {
        // -0.0 and +0.0 times are equal under `<`, so they form a tied group:
        // zero comparable pairs -> 0.5. The brute-force loop must agree.
        let predictions = Array1::from_vec(vec![2.0, 1.0]);
        let times = Array1::from_vec(vec![-0.0, 0.0]);
        let events = Array1::from_vec(vec![true, false]);
        let (fc, ft, fp) = concordance_counts_censored_fenwick(&predictions, &times, &events);
        assert_eq!((fc, ft, fp), (0, 0, 0));
        let (bc, bp) = concordance_counts_censored_bruteforce(&predictions, &times, &events);
        assert_eq!((bc, bp), (0.0, 0.0));
        assert_eq!(concordance_index(&predictions, &times, &events), 0.5);

        // Epsilon-boundary trap: p_j = p_i + 1e-10 exactly (as rounded). The tie
        // predicate is (p_i - p_j).abs() < 1e-10 via subtraction, which is NOT
        // p_j < p_i + 1e-10. Check both sides of the window, plus exact tie.
        let p_i = 0.1;
        for (p_j, _label) in [
            (p_i, "exact tie"),
            (p_i + 1e-11, "inside window"),
            (p_i + 9.9e-11, "near upper edge"),
            (p_i + 1e-10, "at nominal boundary"),
            (p_i + 1.00001e-10, "just outside"),
            (p_i - 5e-11, "concordant within window"),
        ] {
            let predictions = Array1::from_vec(vec![p_i, p_j, 0.5]);
            let times = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let events = Array1::from_vec(vec![true, false, true]);
            let (fc, ft, fp) =
                concordance_counts_censored_fenwick(&predictions, &times, &events);
            let (bc, bp) =
                concordance_counts_censored_bruteforce(&predictions, &times, &events);
            let new_conc = fc as f64 + 0.5 * ft as f64;
            assert_eq!(new_conc.to_bits(), bc.to_bits(), "p_j offset {:e}", p_j - p_i);
            assert_eq!((fp as f64).to_bits(), bp.to_bits());
        }
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
