#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Tests for scoring rules including numerical gradient verification.
// This mirrors Python's test_score.py to ensure gradient implementations are correct.

use ndarray::Array1;
use ngboost_rs::dist::{
    exponential::Exponential, gamma::Gamma, halfnormal::HalfNormal, laplace::Laplace,
    normal::Normal, poisson::Poisson, weibull::Weibull, Distribution,
};
use ngboost_rs::scores::{CRPScore, LogScore, Scorable};

// ============================================================================
// Numerical gradient verification using finite differences
// This matches Python's scipy.optimize.approx_fprime approach
// ============================================================================

/// Compute numerical gradient using central differences
/// Similar to scipy.optimize.approx_fprime but with central differences for better accuracy
fn approx_gradient<F>(params: &Array1<f64>, f: F, epsilon: f64) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = params.len();
    let mut grad = Array1::zeros(n);

    for i in 0..n {
        let mut params_plus = params.clone();
        let mut params_minus = params.clone();
        params_plus[i] += epsilon;
        params_minus[i] -= epsilon;

        // Central difference: (f(x+h) - f(x-h)) / (2h)
        grad[i] = (f(&params_plus) - f(&params_minus)) / (2.0 * epsilon);
    }

    grad
}

/// Test gradient accuracy for a distribution with LogScore
fn test_gradient_accuracy<D>(dist_name: &str, params: Array1<f64>, y_sample: f64, tol: f64)
where
    D: Distribution + Scorable<LogScore>,
{
    let n_params = params.len();

    // Create params as 2D array with single observation
    let params_2d = params.clone().into_shape_with_order((1, n_params)).unwrap();
    let dist = D::from_params(&params_2d);
    let y = Array1::from(vec![y_sample]);

    // Get analytical gradient
    let analytical_grad = Scorable::<LogScore>::d_score(&dist, &y);
    let analytical_grad_vec = analytical_grad.row(0).to_owned();

    // Compute numerical gradient
    let score_fn = |p: &Array1<f64>| -> f64 {
        let p_2d = p.clone().into_shape_with_order((1, n_params)).unwrap();
        let d = D::from_params(&p_2d);
        Scorable::<LogScore>::score(&d, &y)[0]
    };

    let numerical_grad = approx_gradient(&params, score_fn, 1e-6);

    // Compute relative error
    let diff = &analytical_grad_vec - &numerical_grad;
    let error = diff.mapv(|x| x.abs()).sum() / n_params as f64;

    assert!(
        error < tol,
        "{} gradient error {:.2e} exceeds tolerance {:.2e}\n\
         Analytical: {:?}\n\
         Numerical:  {:?}",
        dist_name,
        error,
        tol,
        analytical_grad_vec,
        numerical_grad
    );
}

// ============================================================================
// Gradient verification tests for each distribution
// ============================================================================

#[test]
fn test_normal_gradient() {
    // params: [loc, log_scale]
    let params = Array1::from(vec![1.5, 0.3]);
    let y_sample = 2.0;
    test_gradient_accuracy::<Normal>("Normal", params, y_sample, 1e-4);
}

#[test]
fn test_normal_gradient_negative_y() {
    let params = Array1::from(vec![-0.5, 0.1]);
    let y_sample = -1.0;
    test_gradient_accuracy::<Normal>("Normal (negative y)", params, y_sample, 1e-4);
}

#[test]
fn test_halfnormal_gradient() {
    // params: [log_scale]
    let params = Array1::from(vec![0.5]);
    let y_sample = 1.5; // must be positive
    test_gradient_accuracy::<HalfNormal>("HalfNormal", params, y_sample, 1e-4);
}

#[test]
fn test_laplace_gradient() {
    // params: [loc, log_scale]
    let params = Array1::from(vec![1.0, 0.2]);
    let y_sample = 1.5;
    test_gradient_accuracy::<Laplace>("Laplace", params, y_sample, 1e-4);
}

#[test]
fn test_exponential_gradient() {
    // params: [log_rate]
    let params = Array1::from(vec![0.5]);
    let y_sample = 2.0; // must be positive
    test_gradient_accuracy::<Exponential>("Exponential", params, y_sample, 1e-4);
}

#[test]
fn test_gamma_gradient() {
    // params: [log_shape, log_rate]
    let params = Array1::from(vec![0.5, 0.3]);
    let y_sample = 2.0; // must be positive
    test_gradient_accuracy::<Gamma>("Gamma", params, y_sample, 1e-3); // slightly higher tolerance
}

#[test]
fn test_poisson_gradient() {
    // params: [log_rate]
    let params = Array1::from(vec![1.0]);
    let y_sample = 3.0; // count data
    test_gradient_accuracy::<Poisson>("Poisson", params, y_sample, 1e-4);
}

#[test]
fn test_weibull_gradient() {
    // params: [log_shape, log_scale]
    let params = Array1::from(vec![0.5, 0.5]);
    let y_sample = 1.5; // must be positive
    test_gradient_accuracy::<Weibull>("Weibull", params, y_sample, 1e-3);
}

// ============================================================================
// CRPS gradient tests (for distributions that support it)
// ============================================================================

fn test_crps_gradient_accuracy<D>(dist_name: &str, params: Array1<f64>, y_sample: f64, tol: f64)
where
    D: Distribution + Scorable<CRPScore>,
{
    let n_params = params.len();

    let params_2d = params.clone().into_shape_with_order((1, n_params)).unwrap();
    let dist = D::from_params(&params_2d);
    let y = Array1::from(vec![y_sample]);

    let analytical_grad = Scorable::<CRPScore>::d_score(&dist, &y);
    let analytical_grad_vec = analytical_grad.row(0).to_owned();

    let score_fn = |p: &Array1<f64>| -> f64 {
        let p_2d = p.clone().into_shape_with_order((1, n_params)).unwrap();
        let d = D::from_params(&p_2d);
        Scorable::<CRPScore>::score(&d, &y)[0]
    };

    let numerical_grad = approx_gradient(&params, score_fn, 1e-6);

    let diff = &analytical_grad_vec - &numerical_grad;
    let error = diff.mapv(|x| x.abs()).sum() / n_params as f64;

    assert!(
        error < tol,
        "{} CRPS gradient error {:.2e} exceeds tolerance {:.2e}\n\
         Analytical: {:?}\n\
         Numerical:  {:?}",
        dist_name,
        error,
        tol,
        analytical_grad_vec,
        numerical_grad
    );
}

#[test]
fn test_normal_crps_gradient() {
    let params = Array1::from(vec![1.5, 0.3]);
    let y_sample = 2.0;
    test_crps_gradient_accuracy::<Normal>("Normal CRPS", params, y_sample, 1e-4);
}

// ============================================================================
// Metric (Fisher Information) tests
// ============================================================================

/// Test that the metric is positive semi-definite and symmetric
fn test_metric_properties<D>(dist_name: &str, params: Array1<f64>)
where
    D: Distribution + Scorable<LogScore>,
{
    let n_params = params.len();
    let params_2d = params.into_shape_with_order((1, n_params)).unwrap();
    let dist = D::from_params(&params_2d);

    let metric = Scorable::<LogScore>::metric(&dist);
    let metric_0 = metric.index_axis(ndarray::Axis(0), 0);

    // Check symmetry
    for i in 0..n_params {
        for j in 0..n_params {
            let diff = (metric_0[[i, j]] - metric_0[[j, i]]).abs();
            assert!(
                diff < 1e-10,
                "{} metric not symmetric at ({}, {}): {} vs {}",
                dist_name,
                i,
                j,
                metric_0[[i, j]],
                metric_0[[j, i]]
            );
        }
    }

    // Check positive semi-definiteness (all eigenvalues >= 0)
    // For small matrices, we can check the determinant and trace
    if n_params == 1 {
        assert!(
            metric_0[[0, 0]] >= 0.0,
            "{} 1x1 metric should be non-negative",
            dist_name
        );
    } else if n_params == 2 {
        // For 2x2: positive semi-definite if det >= 0 and trace >= 0
        let trace = metric_0[[0, 0]] + metric_0[[1, 1]];
        let det = metric_0[[0, 0]] * metric_0[[1, 1]] - metric_0[[0, 1]] * metric_0[[1, 0]];
        assert!(
            trace >= -1e-10 && det >= -1e-10,
            "{} 2x2 metric not positive semi-definite: trace={}, det={}",
            dist_name,
            trace,
            det
        );
    }
}

#[test]
fn test_normal_metric_properties() {
    let params = Array1::from(vec![1.0, 0.5]);
    test_metric_properties::<Normal>("Normal", params);
}

#[test]
fn test_halfnormal_metric_properties() {
    let params = Array1::from(vec![0.5]);
    test_metric_properties::<HalfNormal>("HalfNormal", params);
}

#[test]
fn test_laplace_metric_properties() {
    let params = Array1::from(vec![1.0, 0.3]);
    test_metric_properties::<Laplace>("Laplace", params);
}

#[test]
fn test_gamma_metric_properties() {
    let params = Array1::from(vec![0.5, 0.3]);
    test_metric_properties::<Gamma>("Gamma", params);
}

#[test]
fn test_exponential_metric_properties() {
    let params = Array1::from(vec![0.5]);
    test_metric_properties::<Exponential>("Exponential", params);
}

#[test]
fn test_poisson_metric_properties() {
    let params = Array1::from(vec![1.0]);
    test_metric_properties::<Poisson>("Poisson", params);
}

#[test]
fn test_weibull_metric_properties() {
    let params = Array1::from(vec![0.5, 0.5]);
    test_metric_properties::<Weibull>("Weibull", params);
}

// ============================================================================
// Natural gradient tests
// ============================================================================

#[test]
fn test_natural_gradient_computation() {
    // Test that natural gradient = metric^{-1} @ gradient
    let params = Array1::from(vec![1.0, 0.5]);
    let params_2d = params.into_shape_with_order((1, 2)).unwrap();
    let dist = Normal::from_params(&params_2d);
    let y = Array1::from(vec![1.5]);

    let grad = Scorable::<LogScore>::d_score(&dist, &y);
    let natural_grad = Scorable::<LogScore>::grad(&dist, &y, true);
    let regular_grad = Scorable::<LogScore>::grad(&dist, &y, false);

    // Regular gradient should equal d_score
    for i in 0..2 {
        assert!(
            (regular_grad[[0, i]] - grad[[0, i]]).abs() < 1e-10,
            "Regular gradient mismatch"
        );
    }

    // Natural gradient should be different from regular gradient (in general)
    // and should satisfy: metric @ natural_grad = grad
    let metric = Scorable::<LogScore>::metric(&dist);
    let metric_0 = metric.index_axis(ndarray::Axis(0), 0);
    let ng_0 = natural_grad.row(0);
    let reconstructed = metric_0.dot(&ng_0);

    for i in 0..2 {
        assert!(
            (reconstructed[i] - grad[[0, i]]).abs() < 1e-6,
            "Natural gradient verification failed: metric @ ng != grad\n\
             Expected: {:?}\n\
             Got: {:?}",
            grad.row(0),
            reconstructed
        );
    }
}
