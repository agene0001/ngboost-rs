//! Tests for DistributionMethods trait implementations across all distributions.

use approx::assert_relative_eq;
use ndarray::{Array1, Array2};
use ngboost_rs::dist::{
   Cauchy, CauchyFixedVar, Distribution, DistributionMethods, Exponential,
    Gamma, HalfNormal, Laplace, LogNormal, Normal, NormalFixedMean, NormalFixedVar, Poisson,
    StudentT, TFixedDf, TFixedDfFixedVar, Weibull,
};

// ============================================================================
// Normal Distribution Tests
// ============================================================================

#[test]
fn test_normal_methods_basic() {
    // params: [loc, log(scale)]
    // First obs: loc=0, scale=1 (log(scale)=0)
    // Second obs: loc=1, scale=1 (log(scale)=0)
    // Third obs: loc=-1, scale=2 (log(scale)=ln(2))
    let params =
        Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, -1.0, 2.0_f64.ln()]).unwrap();
    let dist = Normal::from_params(&params);

    // Mean should equal loc
    let mean = dist.mean();
    assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(mean[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(mean[2], -1.0, epsilon = 1e-10);

    // Variance should equal scale^2
    let var = dist.variance();
    assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(var[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(var[2], 4.0, epsilon = 1e-10); // scale=2, var=4

    // Mode equals mean for Normal
    let mode = dist.mode();
    assert_relative_eq!(mode[0], mean[0], epsilon = 1e-10);
}

#[test]
fn test_normal_cdf_ppf_roundtrip() {
    let params = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 5.0, 1.0_f64.ln()]).unwrap();
    let dist = Normal::from_params(&params);

    // Test that ppf(cdf(x)) ≈ x
    let y = Array1::from_vec(vec![0.5, 6.0]);
    let cdf_y = dist.cdf(&y);
    let ppf_cdf_y = dist.ppf(&cdf_y);
    assert_relative_eq!(ppf_cdf_y[0], y[0], epsilon = 1e-6);
    assert_relative_eq!(ppf_cdf_y[1], y[1], epsilon = 1e-6);

    // Test that cdf(ppf(q)) ≈ q
    let q = Array1::from_vec(vec![0.25, 0.75]);
    let ppf_q = dist.ppf(&q);
    let cdf_ppf_q = dist.cdf(&ppf_q);
    assert_relative_eq!(cdf_ppf_q[0], q[0], epsilon = 1e-6);
    assert_relative_eq!(cdf_ppf_q[1], q[1], epsilon = 1e-6);
}

#[test]
fn test_normal_sample_statistics() {
    let params = Array2::from_shape_vec((1, 2), vec![10.0, 0.5_f64.ln()]).unwrap();
    let dist = Normal::from_params(&params);

    let samples = dist.sample(10000);
    assert_eq!(samples.shape(), &[10000, 1]);

    let sample_mean: f64 = samples.column(0).mean().unwrap();
    let sample_var: f64 = samples.column(0).var(0.0);

    // Sample mean should be close to theoretical mean
    assert!((sample_mean - 10.0).abs() < 0.1);
    // Sample variance should be close to theoretical variance (0.5^2 = 0.25... wait, exp(0.5*ln)=sqrt(e)
    // Actually exp(ln(0.5)) = 0.5, so scale = 0.5, var = 0.25
    // Hmm, params[1] = ln(0.5), so scale = exp(ln(0.5)) = 0.5
    // Actually ln(0.5) ≈ -0.693, so I wrote 0.5_f64.ln() which is ln(0.5)
    // Let me reconsider: if params[1] = ln(scale), and I set it to ln(0.5),
    // then scale = 0.5, and variance = 0.25
    assert!((sample_var - 0.25).abs() < 0.05);
}

#[test]
fn test_normal_pdf_integrates() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Normal::from_params(&params);

    // PDF at mean should be 1/sqrt(2*pi*var) = 1/sqrt(2*pi) ≈ 0.399
    let y = Array1::from_vec(vec![0.0]);
    let pdf = dist.pdf(&y);
    assert_relative_eq!(
        pdf[0],
        1.0 / (2.0 * std::f64::consts::PI).sqrt(),
        epsilon = 1e-10
    );
}

#[test]
fn test_normal_interval() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Normal::from_params(&params);

    // 95% interval for standard normal should be approximately (-1.96, 1.96)
    let (lower, upper) = dist.interval(0.05);
    assert_relative_eq!(lower[0], -1.96, epsilon = 0.01);
    assert_relative_eq!(upper[0], 1.96, epsilon = 0.01);
}

// ============================================================================
// LogNormal Distribution Tests
// ============================================================================

#[test]
fn test_lognormal_methods_basic() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = LogNormal::from_params(&params);

    // Mean of lognormal is exp(mu + sigma^2/2) = exp(0 + 0.5) = exp(0.5)
    let mean = dist.mean();
    assert_relative_eq!(mean[0], (0.5_f64).exp(), epsilon = 1e-10);

    // Median is exp(mu) = exp(0) = 1
    let median = dist.median();
    assert_relative_eq!(median[0], 1.0, epsilon = 1e-10);

    // Mode is exp(mu - sigma^2) = exp(0 - 1) = exp(-1)
    let mode = dist.mode();
    assert_relative_eq!(mode[0], (-1.0_f64).exp(), epsilon = 1e-10);
}

#[test]
fn test_lognormal_samples_positive() {
    let params = Array2::from_shape_vec((1, 2), vec![1.0, 0.5_f64.ln()]).unwrap();
    let dist = LogNormal::from_params(&params);

    let samples = dist.sample(1000);
    assert!(samples.iter().all(|&x| x > 0.0));
}

// ============================================================================
// Exponential Distribution Tests
// ============================================================================

#[test]
fn test_exponential_methods_basic() {
    let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
    let dist = Exponential::from_params(&params);

    // Mean = scale = 2
    let mean = dist.mean();
    assert_relative_eq!(mean[0], 2.0, epsilon = 1e-10);

    // Variance = scale^2 = 4
    let var = dist.variance();
    assert_relative_eq!(var[0], 4.0, epsilon = 1e-10);

    // Mode = 0
    let mode = dist.mode();
    assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);

    // Median = ln(2) * scale = ln(2) * 2
    let median = dist.median();
    assert_relative_eq!(median[0], std::f64::consts::LN_2 * 2.0, epsilon = 1e-10);
}

#[test]
fn test_exponential_memoryless_property() {
    // Test the memoryless property: P(X > s + t | X > s) = P(X > t)
    let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
    let dist = Exponential::from_params(&params);

    let s = 1.0;
    let t = 0.5;

    // P(X > s) = 1 - CDF(s) = SF(s)
    let sf_s = dist.sf(&Array1::from_vec(vec![s]))[0];
    // P(X > s + t) = SF(s + t)
    let sf_s_plus_t = dist.sf(&Array1::from_vec(vec![s + t]))[0];
    // P(X > t) = SF(t)
    let sf_t = dist.sf(&Array1::from_vec(vec![t]))[0];

    // P(X > s + t | X > s) = P(X > s + t) / P(X > s) should equal P(X > t)
    assert_relative_eq!(sf_s_plus_t / sf_s, sf_t, epsilon = 1e-10);
}

// ============================================================================
// Gamma Distribution Tests
// ============================================================================

#[test]
fn test_gamma_methods_basic() {
    // shape=2, rate=1 -> mean=2, var=2
    let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
    let dist = Gamma::from_params(&params);

    let mean = dist.mean();
    assert_relative_eq!(mean[0], 2.0, epsilon = 1e-10);

    let var = dist.variance();
    assert_relative_eq!(var[0], 2.0, epsilon = 1e-10);

    // Mode = (shape - 1) / rate = 1
    let mode = dist.mode();
    assert_relative_eq!(mode[0], 1.0, epsilon = 1e-10);
}

#[test]
fn test_gamma_samples_positive() {
    let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.5_f64.ln()]).unwrap();
    let dist = Gamma::from_params(&params);

    let samples = dist.sample(1000);
    assert!(samples.iter().all(|&x| x >= 0.0));
}

// ============================================================================
// Laplace Distribution Tests
// ============================================================================

#[test]
fn test_laplace_methods_basic() {
    let params = Array2::from_shape_vec((1, 2), vec![5.0, 0.0]).unwrap();
    let dist = Laplace::from_params(&params);

    // Mean = loc = 5
    let mean = dist.mean();
    assert_relative_eq!(mean[0], 5.0, epsilon = 1e-10);

    // Variance = 2 * scale^2 = 2 * 1 = 2
    let var = dist.variance();
    assert_relative_eq!(var[0], 2.0, epsilon = 1e-10);

    // Median = Mode = loc = 5
    assert_relative_eq!(dist.median()[0], 5.0, epsilon = 1e-10);
    assert_relative_eq!(dist.mode()[0], 5.0, epsilon = 1e-10);
}

#[test]
fn test_laplace_symmetry() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Laplace::from_params(&params);

    // PDF should be symmetric around loc
    let y_pos = Array1::from_vec(vec![1.0]);
    let y_neg = Array1::from_vec(vec![-1.0]);

    assert_relative_eq!(dist.pdf(&y_pos)[0], dist.pdf(&y_neg)[0], epsilon = 1e-10);

    // CDF at -x and SF at x should be equal
    assert_relative_eq!(dist.cdf(&y_neg)[0], dist.sf(&y_pos)[0], epsilon = 1e-10);
}

// ============================================================================
// Cauchy Distribution Tests
// ============================================================================

#[test]
fn test_cauchy_undefined_moments() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Cauchy::from_params(&params);

    // Mean and variance should be NaN for Cauchy
    assert!(dist.mean()[0].is_nan());
    assert!(dist.variance()[0].is_nan());

    // But median and mode are well-defined and equal to loc
    assert_relative_eq!(dist.median()[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dist.mode()[0], 0.0, epsilon = 1e-10);
}

#[test]
fn test_cauchy_heavy_tails() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Cauchy::from_params(&params);

    // Cauchy has very heavy tails
    // P(|X| > 10) should be substantial
    let sf_10 = dist.sf(&Array1::from_vec(vec![10.0]))[0];
    assert!(sf_10 > 0.01); // Should be noticeable probability in tail
}

// ============================================================================
// Weibull Distribution Tests
// ============================================================================

#[test]
fn test_weibull_methods_basic() {
    // shape=1, scale=1 -> Exponential(1)
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = Weibull::from_params(&params);

    // For shape=1 (exponential), mean = scale = 1
    let mean = dist.mean();
    assert_relative_eq!(mean[0], 1.0, epsilon = 1e-6);

    // Mode = 0 for shape <= 1
    let mode = dist.mode();
    assert_relative_eq!(mode[0], 0.0, epsilon = 1e-10);
}

#[test]
fn test_weibull_rayleigh_case() {
    // shape=2, scale=1 -> Rayleigh distribution
    let params = Array2::from_shape_vec((1, 2), vec![2.0_f64.ln(), 0.0]).unwrap();
    let dist = Weibull::from_params(&params);

    // Mode = scale * ((k-1)/k)^(1/k) = 1 * (0.5)^0.5 ≈ 0.707
    let mode = dist.mode();
    assert_relative_eq!(mode[0], 0.5_f64.sqrt(), epsilon = 1e-6);
}

// ============================================================================
// HalfNormal Distribution Tests
// ============================================================================

#[test]
fn test_halfnormal_methods_basic() {
    let params = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
    let dist = HalfNormal::from_params(&params);

    // Mean = scale * sqrt(2/pi)
    let expected_mean = (2.0 / std::f64::consts::PI).sqrt();
    assert_relative_eq!(dist.mean()[0], expected_mean, epsilon = 1e-10);

    // Mode = 0
    assert_relative_eq!(dist.mode()[0], 0.0, epsilon = 1e-10);
}

#[test]
fn test_halfnormal_samples_positive() {
    let params = Array2::from_shape_vec((1, 1), vec![1.0_f64.ln()]).unwrap();
    let dist = HalfNormal::from_params(&params);

    let samples = dist.sample(1000);
    assert!(samples.iter().all(|&x| x >= 0.0));
}

// ============================================================================
// Poisson Distribution Tests
// ============================================================================

#[test]
fn test_poisson_methods_basic() {
    let params = Array2::from_shape_vec((2, 1), vec![1.0_f64.ln(), 5.0_f64.ln()]).unwrap();
    let dist = Poisson::from_params(&params);

    // Mean = Variance = rate
    assert_relative_eq!(dist.mean()[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(dist.variance()[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(dist.mean()[1], 5.0, epsilon = 1e-10);
    assert_relative_eq!(dist.variance()[1], 5.0, epsilon = 1e-10);

    // Mode = floor(rate) - for integer rate, mode can be rate or rate-1
    assert_relative_eq!(dist.mode()[0], 1.0, epsilon = 1e-10);
    // Due to floating point, exp(ln(5)) ≈ 5 but floor might give 4
    assert!(dist.mode()[1] == 4.0 || dist.mode()[1] == 5.0);
}

#[test]
fn test_poisson_samples_integers() {
    let params = Array2::from_shape_vec((1, 1), vec![3.0_f64.ln()]).unwrap();
    let dist = Poisson::from_params(&params);

    let samples = dist.sample(1000);
    // All samples should be non-negative integers
    assert!(samples.iter().all(|&x| x >= 0.0 && x.fract() == 0.0));
}

// ============================================================================
// Student's T Distribution Tests
// ============================================================================

#[test]
fn test_studentt_methods_basic() {
    // loc=0, scale=1, df=5
    let params = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 5.0_f64.ln()]).unwrap();
    let dist = StudentT::from_params(&params);

    // Mean = loc for df > 1
    assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);

    // Variance = scale^2 * df / (df - 2) for df > 2
    // = 1 * 5 / 3 = 5/3
    let expected_var = 5.0 / 3.0;
    assert_relative_eq!(dist.variance()[0], expected_var, epsilon = 1e-10);

    // Mode = Median = loc
    assert_relative_eq!(dist.mode()[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dist.median()[0], 0.0, epsilon = 1e-10);
}

#[test]
fn test_studentt_approaches_normal() {
    // As df -> infinity, T distribution approaches Normal
    let params_t = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 100.0_f64.ln()]).unwrap();
    let dist_t = StudentT::from_params(&params_t);

    let params_n = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist_n = Normal::from_params(&params_n);

    // PDF at 0 should be very close
    let y = Array1::from_vec(vec![0.0]);
    let pdf_t = dist_t.pdf(&y)[0];
    let pdf_n = dist_n.pdf(&y)[0];
    assert!((pdf_t - pdf_n).abs() < 0.01);

    // CDF at 1 should be very close
    let y = Array1::from_vec(vec![1.0]);
    let cdf_t = dist_t.cdf(&y)[0];
    let cdf_n = dist_n.cdf(&y)[0];
    assert!((cdf_t - cdf_n).abs() < 0.01);
}

// ============================================================================
// Fixed Variance/Mean Distribution Tests
// ============================================================================

#[test]
fn test_normal_fixed_var() {
    let params = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let dist = NormalFixedVar::from_params(&params);

    assert_relative_eq!(dist.mean()[0], 5.0, epsilon = 1e-10);
    assert_relative_eq!(dist.variance()[0], 1.0, epsilon = 1e-10);
}

#[test]
fn test_normal_fixed_mean() {
    // params = [log(scale)] for NormalFixedMean
    // log(scale) = 1.0, so scale = e
    let params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let dist = NormalFixedMean::from_params(&params);

    assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(dist.std()[0], std::f64::consts::E, epsilon = 1e-10);
}

#[test]
fn test_tfixeddf() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let dist = TFixedDf::from_params(&params);

    // Default df = 3
    assert_relative_eq!(dist.mean()[0], 0.0, epsilon = 1e-10);
    // Variance = scale^2 * df / (df - 2) = 1 * 3 / 1 = 3
    assert_relative_eq!(dist.variance()[0], 3.0, epsilon = 1e-10);
}

#[test]
fn test_tfixeddfvar() {
    let params = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap();
    let dist = TFixedDfFixedVar::from_params(&params);

    assert_relative_eq!(dist.mean()[0], 2.0, epsilon = 1e-10);
    assert_relative_eq!(dist.median()[0], 2.0, epsilon = 1e-10);
}

#[test]
fn test_cauchy_fixed_var() {
    let params = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
    let dist = CauchyFixedVar::from_params(&params);

    // Mean undefined but median = loc
    assert!(dist.mean()[0].is_nan());
    assert_relative_eq!(dist.median()[0], 3.0, epsilon = 1e-10);
}

// ============================================================================
// Cross-Distribution Consistency Tests
// ============================================================================

#[test]
fn test_cdf_sf_sum_to_one() {
    // For all distributions, CDF(x) + SF(x) should equal 1

    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let normal = Normal::from_params(&params);
    let laplace = Laplace::from_params(&params);
    let cauchy = Cauchy::from_params(&params);

    let y = Array1::from_vec(vec![0.5]);

    assert_relative_eq!(normal.cdf(&y)[0] + normal.sf(&y)[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(laplace.cdf(&y)[0] + laplace.sf(&y)[0], 1.0, epsilon = 1e-10);
    assert_relative_eq!(cauchy.cdf(&y)[0] + cauchy.sf(&y)[0], 1.0, epsilon = 1e-10);
}

#[test]
fn test_interval_contains_median() {
    // The confidence interval should contain the median for any distribution

    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let normal = Normal::from_params(&params);
    let laplace = Laplace::from_params(&params);

    for alpha in [0.1, 0.2, 0.3, 0.4] {
        let (lower, upper) = normal.interval(alpha);
        let median = normal.median()[0];
        assert!(lower[0] <= median && median <= upper[0]);

        let (lower, upper) = laplace.interval(alpha);
        let median = laplace.median()[0];
        assert!(lower[0] <= median && median <= upper[0]);
    }
}

#[test]
fn test_pdf_logpdf_consistency() {
    let params = Array2::from_shape_vec((1, 2), vec![1.0, 0.5_f64.ln()]).unwrap();
    let normal = Normal::from_params(&params);

    let y = Array1::from_vec(vec![0.5]);
    let pdf = normal.pdf(&y)[0];
    let logpdf = normal.logpdf(&y)[0];

    assert_relative_eq!(pdf.ln(), logpdf, epsilon = 1e-10);
}

// ============================================================================
// Sample Statistics Tests
// ============================================================================

#[test]
fn test_samples_have_correct_mean_variance() {
    let n_samples = 10000;
    let tolerance = 0.1; // 10% relative tolerance for sample statistics

    // Normal
    let params = Array2::from_shape_vec((1, 2), vec![5.0, 1.0_f64.ln()]).unwrap();
    let normal = Normal::from_params(&params);
    let samples = normal.sample(n_samples);
    let sample_mean = samples.column(0).mean().unwrap();
    let sample_var = samples.column(0).var(0.0);

    let expected_mean = normal.mean()[0];
    let expected_var = normal.variance()[0];

    assert!(
        (sample_mean - expected_mean).abs() / expected_mean.abs().max(1.0) < tolerance,
        "Normal sample mean {} differs from expected {}",
        sample_mean,
        expected_mean
    );
    assert!(
        (sample_var - expected_var).abs() / expected_var < tolerance,
        "Normal sample variance {} differs from expected {}",
        sample_var,
        expected_var
    );

    // Exponential
    let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
    let exp = Exponential::from_params(&params);
    let samples = exp.sample(n_samples);
    let sample_mean = samples.column(0).mean().unwrap();

    let expected_mean = exp.mean()[0];
    assert!(
        (sample_mean - expected_mean).abs() / expected_mean < tolerance,
        "Exponential sample mean {} differs from expected {}",
        sample_mean,
        expected_mean
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_cdf_at_extremes() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let normal = Normal::from_params(&params);

    // CDF at very negative should be close to 0
    let y = Array1::from_vec(vec![-10.0]);
    assert!(normal.cdf(&y)[0] < 0.001);

    // CDF at very positive should be close to 1
    let y = Array1::from_vec(vec![10.0]);
    assert!(normal.cdf(&y)[0] > 0.999);
}

#[test]
fn test_ppf_at_boundaries() {
    let params = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
    let normal = Normal::from_params(&params);

    // PPF at 0.5 should be the median
    let q = Array1::from_vec(vec![0.5]);
    assert_relative_eq!(normal.ppf(&q)[0], normal.median()[0], epsilon = 1e-10);

    // PPF should handle values close to 0 and 1
    let q_low = Array1::from_vec(vec![0.001]);
    let q_high = Array1::from_vec(vec![0.999]);
    assert!(normal.ppf(&q_low)[0].is_finite());
    assert!(normal.ppf(&q_high)[0].is_finite());
}

#[test]
fn test_multiple_observations() {
    // Test that methods work correctly with multiple observations
    let params = Array2::from_shape_vec(
        (5, 2),
        vec![
            0.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            -1.0,
            0.5_f64.ln(),
            0.0,
            1.0_f64.ln(),
        ],
    )
    .unwrap();
    let normal = Normal::from_params(&params);

    let mean = normal.mean();
    assert_eq!(mean.len(), 5);
    assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
    assert_relative_eq!(mean[1], 1.0, epsilon = 1e-10);
    assert_relative_eq!(mean[2], 2.0, epsilon = 1e-10);
    assert_relative_eq!(mean[3], -1.0, epsilon = 1e-10);
    assert_relative_eq!(mean[4], 0.0, epsilon = 1e-10);

    let samples = normal.sample(100);
    assert_eq!(samples.shape(), &[100, 5]);
}
