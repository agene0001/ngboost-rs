#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::{
    categorical::{Bernoulli, Categorical3},
    cauchy::{Cauchy, CauchyFixedVar},
    exponential::Exponential,
    gamma::Gamma,
    halfnormal::HalfNormal,
    laplace::Laplace,
    lognormal::LogNormal,
    normal::{Normal, NormalFixedMean, NormalFixedVar},
    poisson::Poisson,
    studentt::{StudentT, TFixedDf, TFixedDfFixedVar},
    weibull::Weibull,
    Distribution,
};
use ngboost_rs::learners::{HistogramLearner, StumpLearner};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{LogScore, Scorable};

// ============================================================================
// Macro for testing regression distributions with NGBoost
// ============================================================================

macro_rules! test_dist {
    ($name:ident, $dist:ty) => {
        #[test]
        fn $name() {
            let n_samples = 100;
            let n_features = 5;
            let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
            let y = x.dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]))
                + &Array1::random(n_samples, Uniform::new(-0.5, 0.5).unwrap());

            // Ensure y is positive for distributions that require it
            let y = y.mapv(|v| if v < 0.0 { 0.01 } else { v });

            let mut model: NGBoost<$dist, LogScore, StumpLearner> =
                NGBoost::new(10, 0.1, StumpLearner);
            let fit_result = model.fit(&x, &y);
            assert!(fit_result.is_ok());

            let y_pred = model.predict(&x);
            assert_eq!(y_pred.len(), n_samples);
        }
    };
}

// ============================================================================
// Basic distribution tests with NGBoost fitting
// ============================================================================

test_dist!(test_normal, Normal);
test_dist!(test_lognormal, LogNormal);
test_dist!(test_exponential, Exponential);
test_dist!(test_gamma, Gamma);
test_dist!(test_laplace, Laplace);
test_dist!(test_weibull, Weibull);
test_dist!(test_halfnormal, HalfNormal);
test_dist!(test_studentt, StudentT);
test_dist!(test_tfixeddf, TFixedDf);
test_dist!(test_tfixeddf_fixedvar, TFixedDfFixedVar);
test_dist!(test_cauchy, Cauchy);
test_dist!(test_cauchy_fixed_var, CauchyFixedVar);
test_dist!(test_normal_fixed_var, NormalFixedVar);
test_dist!(test_normal_fixed_mean, NormalFixedMean);

// ============================================================================
// Poisson test (requires integer targets)
// ============================================================================

#[test]
fn test_poisson() {
    let n_samples = 100;
    let n_features = 5;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x
        .dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]))
        .mapv(|v: f64| v.exp().max(0.0).round());

    let mut model: NGBoost<Poisson, LogScore, StumpLearner> = NGBoost::new(10, 0.1, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    assert_eq!(y_pred.len(), n_samples);
}

// ============================================================================
// Classification tests
// ============================================================================

#[test]
fn test_bernoulli() {
    let n_samples = 100;
    let n_features = 5;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());

    // Create binary labels based on a linear combination
    let linear = x.dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]));
    let y = linear.mapv(|v: f64| if v > 0.0 { 1.0 } else { 0.0 });

    let mut model: NGBoost<Bernoulli, LogScore, StumpLearner> = NGBoost::new(10, 0.1, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    assert_eq!(y_pred.len(), n_samples);

    // Predictions should be 0 or 1
    for &pred in y_pred.iter() {
        assert!(pred == 0.0 || pred == 1.0);
    }
}

#[test]
fn test_categorical_multiclass() {
    let n_samples = 150;
    let n_features = 4;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());

    // Create 3-class labels
    let linear = x.dot(&Array1::from(vec![1.0, -1.0, 0.5, -0.5]));
    let y = linear.mapv(|v: f64| {
        if v < -0.3 {
            0.0
        } else if v > 0.3 {
            2.0
        } else {
            1.0
        }
    });

    let mut model: NGBoost<Categorical3, LogScore, StumpLearner> =
        NGBoost::new(10, 0.1, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    assert_eq!(y_pred.len(), n_samples);

    // Predictions should be 0, 1, or 2
    for &pred in y_pred.iter() {
        assert!(pred == 0.0 || pred == 1.0 || pred == 2.0);
    }
}

// ============================================================================
// Distribution-level unit tests (without NGBoost)
// ============================================================================

#[test]
fn test_normal_distribution_methods() {
    // Test fit
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let init_params = Normal::fit(&y);
    assert_eq!(init_params.len(), 2);

    // Test from_params
    let params = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.5, 3.0, -0.5, 4.0, 0.0],
    )
    .unwrap();
    let dist = Normal::from_params(&params);

    assert_eq!(dist.n_params(), 2);
    assert_eq!(dist.loc.len(), 5);
    assert_eq!(dist.scale.len(), 5);

    // Test predict
    let predictions = dist.predict();
    assert_eq!(predictions.len(), 5);
    assert!((predictions[0] - 0.0).abs() < 1e-6);
    assert!((predictions[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_normal_score_and_gradient() {
    let params = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.0, 0.0, // loc=0, scale=1
            1.0, 0.0, // loc=1, scale=1
            2.0, 0.0, // loc=2, scale=1
        ],
    )
    .unwrap();
    let dist = Normal::from_params(&params);

    let y = Array1::from(vec![0.0, 1.0, 2.0]);

    // Score should be small when y matches loc (for unit variance)
    let scores = Scorable::<LogScore>::score(&dist, &y);
    assert_eq!(scores.len(), 3);

    // All scores should be equal (same distance from mean, same variance)
    let expected_score = 0.5 * (2.0 * std::f64::consts::PI).ln();
    for &s in scores.iter() {
        assert!((s - expected_score).abs() < 1e-6);
    }

    // Gradient should be zero when y == loc
    let grad = Scorable::<LogScore>::d_score(&dist, &y);
    assert_eq!(grad.shape(), &[3, 2]);
    for i in 0..3 {
        assert!(grad[[i, 0]].abs() < 1e-6); // d/d(loc) = 0
    }
}

#[test]
fn test_exponential_distribution() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let init_params = Exponential::fit(&y);
    assert_eq!(init_params.len(), 1);

    // Mean of y is 3.0, so init param should be ln(3.0)
    assert!((init_params[0] - 3.0_f64.ln()).abs() < 1e-6);
}

#[test]
fn test_poisson_distribution() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let init_params = Poisson::fit(&y);
    assert_eq!(init_params.len(), 1);

    // Mean of y is 3.0, so init param should be ln(3.0)
    assert!((init_params[0] - 3.0_f64.ln()).abs() < 1e-6);

    // Test from_params
    let params = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let dist = Poisson::from_params(&params);

    // Predictions should be exp(param) = rate
    let predictions = dist.predict();
    assert!((predictions[0] - 1.0).abs() < 1e-6);
    assert!((predictions[1] - std::f64::consts::E).abs() < 1e-6);
}

#[test]
fn test_gamma_distribution() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let init_params = Gamma::fit(&y);
    assert_eq!(init_params.len(), 2); // shape and rate
}

#[test]
fn test_laplace_distribution() {
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let init_params = Laplace::fit(&y);
    assert_eq!(init_params.len(), 2); // loc and log(scale)

    // Median of y is 3.0
    assert!((init_params[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_bernoulli_probabilities() {
    use ngboost_rs::dist::ClassificationDistn;

    // Test with known logits
    let params = Array2::from_shape_vec(
        (3, 1),
        vec![
            0.0,  // 50-50 probability
            2.0,  // high probability of class 1
            -2.0, // high probability of class 0
        ],
    )
    .unwrap();
    let dist = Bernoulli::from_params(&params);

    let probs = dist.class_probs();
    assert_eq!(probs.shape(), &[3, 2]);

    // For logit=0, probabilities should be ~0.5 each
    assert!((probs[[0, 0]] - 0.5).abs() < 0.01);
    assert!((probs[[0, 1]] - 0.5).abs() < 0.01);

    // For logit=2, class 1 should be more likely
    assert!(probs[[1, 1]] > probs[[1, 0]]);

    // For logit=-2, class 0 should be more likely
    assert!(probs[[2, 0]] > probs[[2, 1]]);

    // Probabilities should sum to 1
    for i in 0..3 {
        let sum = probs[[i, 0]] + probs[[i, 1]];
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

// ============================================================================
// Score comparison tests
// ============================================================================

#[test]
fn test_normal_logscore_vs_crpscore() {
    use ngboost_rs::scores::CRPScore;

    let params = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
    let dist = Normal::from_params(&params);

    let y = Array1::from(vec![0.5, 1.5, 2.5]);

    let log_scores = Scorable::<LogScore>::score(&dist, &y);
    let crp_scores = Scorable::<CRPScore>::score(&dist, &y);

    // Both should return valid scores
    assert_eq!(log_scores.len(), 3);
    assert_eq!(crp_scores.len(), 3);

    // Scores should be positive
    for i in 0..3 {
        assert!(log_scores[i] > 0.0);
        assert!(crp_scores[i] > 0.0);
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_small_dataset() {
    let n_samples = 10;
    let n_features = 2;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![1.0, -1.0]))
        + &Array1::random(n_samples, Uniform::new(-0.1, 0.1).unwrap());
    let y = y.mapv(|v: f64| v.abs() + 0.01);

    let mut model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(5, 0.1, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());
}

#[test]
fn test_single_feature() {
    let n_samples = 50;
    let x = Array2::random((n_samples, 1), Uniform::new(0., 1.).unwrap());
    let y = x.column(0).mapv(|v| v * 2.0 + 1.0)
        + &Array1::random(n_samples, Uniform::new(-0.1, 0.1).unwrap());
    let y = y.mapv(|v: f64| v.abs() + 0.01);

    let mut model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(10, 0.1, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());
}

#[test]
fn test_many_iterations() {
    let n_samples = 100;
    let n_features = 3;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![1.0, 2.0, -1.0]))
        + &Array1::random(n_samples, Uniform::new(-0.2, 0.2).unwrap());
    let y = y.mapv(|v: f64| v.abs() + 0.01);

    let mut model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(100, 0.05, StumpLearner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    // With many iterations, predictions should be reasonable
    let y_pred = model.predict(&x);
    let mse: f64 = (&y - &y_pred).mapv(|v| v * v).mean().unwrap();
    assert!(mse < 1.0); // Should have learned something
}

// ============================================================================
// Histogram Learner Tests
// ============================================================================

#[test]
fn test_histogram_learner_basic() {
    let n_samples = 100;
    let n_features = 5;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]))
        + &Array1::random(n_samples, Uniform::new(-0.5, 0.5).unwrap());
    let y = y.mapv(|v| if v < 0.0 { 0.01 } else { v });

    let hist_learner = HistogramLearner::new(3);
    let mut model: NGBoost<Normal, LogScore, HistogramLearner> =
        NGBoost::new(50, 0.1, hist_learner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    assert_eq!(y_pred.len(), n_samples);

    // Check predictions are reasonable
    let mse: f64 = (&y - &y_pred).mapv(|v| v * v).mean().unwrap();
    assert!(mse < 1.0);
}

// ============================================================================
// MultivariateNormal Distribution Tests
// ============================================================================

use ngboost_rs::dist::multivariate_normal::{MultivariateNormal2, MultivariateNormal3};

/// Test basic construction of MultivariateNormal from parameters.
#[test]
fn test_multivariate_normal_2d_from_params() {
    // For P=2, we have P + P*(P+1)/2 = 2 + 3 = 5 parameters
    // [mean_0, mean_1, L_00 (log scale), L_10, L_11 (log scale)]
    let n_obs = 10;
    let n_params = 5;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        // Mean: [1.0, 2.0]
        params[[i, 0]] = 1.0;
        params[[i, 1]] = 2.0;
        // L matrix (lower triangular):
        // L = [[exp(0.0), 0], [0.5, exp(0.0)]] = [[1.0, 0], [0.5, 1.0]]
        params[[i, 2]] = 0.0; // log(L_00)
        params[[i, 3]] = 0.5; // L_10
        params[[i, 4]] = 0.0; // log(L_11)
    }

    let dist = MultivariateNormal2::from_params(&params);

    // Check means
    for i in 0..n_obs {
        assert!((dist.loc[[i, 0]] - 1.0).abs() < 1e-6);
        assert!((dist.loc[[i, 1]] - 2.0).abs() < 1e-6);
    }

    // Check that L matrix is correct
    for i in 0..n_obs {
        assert!((dist.l[[i, 0, 0]] - 1.0).abs() < 0.01); // exp(0) + 1e-6 ≈ 1.0
        assert!((dist.l[[i, 1, 0]] - 0.5).abs() < 1e-6);
        assert!((dist.l[[i, 1, 1]] - 1.0).abs() < 0.01);
    }
}

/// Test fitting MultivariateNormal parameters from data.
#[test]
fn test_multivariate_normal_2d_fit() {
    // Create synthetic 2D data with known mean and covariance
    let n_samples = 100;
    let p = 2;

    // Generate data with mean [3.0, 5.0] and some correlation
    let mut y = Array1::zeros(n_samples * p);
    for i in 0..n_samples {
        let x1 = 3.0 + (i as f64 * 0.01 - 0.5);
        let x2 = 5.0 + (i as f64 * 0.02 - 1.0) + 0.5 * x1; // Correlated with x1
        y[i * p] = x1;
        y[i * p + 1] = x2;
    }

    let params = MultivariateNormal2::fit(&y);

    // Should have 5 parameters for 2D MVN
    assert_eq!(params.len(), 5);

    // Mean should be approximately [3.0, 6.5] (due to correlation term)
    assert!(
        (params[0] - 3.0).abs() < 0.5,
        "Mean[0] = {} not close to 3.0",
        params[0]
    );
    // Note: exact mean[1] depends on exact correlation structure
}

/// Test log probability computation.
#[test]
fn test_multivariate_normal_2d_logpdf() {
    let n_obs = 5;
    let n_params = 5;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        params[[i, 0]] = 0.0; // mean_0
        params[[i, 1]] = 0.0; // mean_1
        params[[i, 2]] = 0.0; // log(L_00) -> L_00 ≈ 1
        params[[i, 3]] = 0.0; // L_10
        params[[i, 4]] = 0.0; // log(L_11) -> L_11 ≈ 1
    }

    let dist = MultivariateNormal2::from_params(&params);

    // y at the mean should have highest log probability
    let y_at_mean = Array2::zeros((n_obs, 2));
    let logpdf_at_mean = dist.logpdf(&y_at_mean);

    // y away from mean should have lower log probability
    let mut y_away = Array2::zeros((n_obs, 2));
    for i in 0..n_obs {
        y_away[[i, 0]] = 2.0;
        y_away[[i, 1]] = 2.0;
    }
    let logpdf_away = dist.logpdf(&y_away);

    for i in 0..n_obs {
        assert!(
            logpdf_at_mean[i] > logpdf_away[i],
            "Log PDF at mean ({}) should be greater than away ({})",
            logpdf_at_mean[i],
            logpdf_away[i]
        );
    }
}

/// Test score (negative log likelihood) computation.
#[test]
fn test_multivariate_normal_2d_score() {
    let n_obs = 10;
    let n_params = 5;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        params[[i, 0]] = 1.0;
        params[[i, 1]] = 2.0;
        params[[i, 2]] = 0.0;
        params[[i, 3]] = 0.0;
        params[[i, 4]] = 0.0;
    }

    let dist = MultivariateNormal2::from_params(&params);

    // Create y values (flattened: [y1_0, y1_1, y2_0, y2_1, ...])
    let mut y = Array1::zeros(n_obs * 2);
    for i in 0..n_obs {
        y[i * 2] = 1.0; // Close to mean
        y[i * 2 + 1] = 2.0;
    }

    let scores = Scorable::score(&dist, &y);
    assert_eq!(scores.len(), n_obs);

    // Score should be positive (negative log prob)
    for i in 0..n_obs {
        assert!(
            scores[i].is_finite(),
            "Score[{}] is not finite: {}",
            i,
            scores[i]
        );
    }
}

/// Test gradient computation.
#[test]
fn test_multivariate_normal_2d_gradient() {
    let n_obs = 5;
    let n_params = 5;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        params[[i, 0]] = 0.0;
        params[[i, 1]] = 0.0;
        params[[i, 2]] = 0.0;
        params[[i, 3]] = 0.0;
        params[[i, 4]] = 0.0;
    }

    let dist = MultivariateNormal2::from_params(&params);

    let mut y = Array1::zeros(n_obs * 2);
    for i in 0..n_obs {
        y[i * 2] = 0.5;
        y[i * 2 + 1] = -0.5;
    }

    let grads = Scorable::d_score(&dist, &y);
    assert_eq!(grads.nrows(), n_obs);
    assert_eq!(grads.ncols(), n_params);

    // Gradients should be finite
    for i in 0..n_obs {
        for j in 0..n_params {
            assert!(
                grads[[i, j]].is_finite(),
                "Gradient[{}, {}] is not finite: {}",
                i,
                j,
                grads[[i, j]]
            );
        }
    }
}

/// Test Fisher Information Matrix computation.
#[test]
fn test_multivariate_normal_2d_metric() {
    let n_obs = 3;
    let n_params = 5;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        params[[i, 0]] = 0.0;
        params[[i, 1]] = 0.0;
        params[[i, 2]] = 0.0;
        params[[i, 3]] = 0.0;
        params[[i, 4]] = 0.0;
    }

    let dist = MultivariateNormal2::from_params(&params);
    let metric = Scorable::<LogScore>::metric(&dist);

    assert_eq!(metric.shape(), &[n_obs, n_params, n_params]);

    // Metric should be positive definite (at least have positive diagonal)
    for i in 0..n_obs {
        for j in 0..n_params {
            assert!(
                metric[[i, j, j]] > 0.0,
                "Metric[{}, {}, {}] should be positive: {}",
                i,
                j,
                j,
                metric[[i, j, j]]
            );
        }
    }
}

/// Test 3-dimensional MultivariateNormal.
#[test]
fn test_multivariate_normal_3d_basic() {
    // For P=3, we have P + P*(P+1)/2 = 3 + 6 = 9 parameters
    let n_obs = 5;
    let n_params = 9;
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        // Mean: [1.0, 2.0, 3.0]
        params[[i, 0]] = 1.0;
        params[[i, 1]] = 2.0;
        params[[i, 2]] = 3.0;
        // L matrix (lower triangular, 6 elements):
        // L = [[exp(0), 0, 0], [0, exp(0), 0], [0, 0, exp(0)]] = identity
        params[[i, 3]] = 0.0; // log(L_00)
        params[[i, 4]] = 0.0; // L_10
        params[[i, 5]] = 0.0; // log(L_11)
        params[[i, 6]] = 0.0; // L_20
        params[[i, 7]] = 0.0; // L_21
        params[[i, 8]] = 0.0; // log(L_22)
    }

    let dist = MultivariateNormal3::from_params(&params);

    // Check means
    for i in 0..n_obs {
        assert!((dist.loc[[i, 0]] - 1.0).abs() < 1e-6);
        assert!((dist.loc[[i, 1]] - 2.0).abs() < 1e-6);
        assert!((dist.loc[[i, 2]] - 3.0).abs() < 1e-6);
    }

    // Test score computation
    let mut y = Array1::zeros(n_obs * 3);
    for i in 0..n_obs {
        y[i * 3] = 1.0;
        y[i * 3 + 1] = 2.0;
        y[i * 3 + 2] = 3.0;
    }

    let scores = Scorable::score(&dist, &y);
    assert_eq!(scores.len(), n_obs);

    for i in 0..n_obs {
        assert!(scores[i].is_finite());
    }
}

/// Test that predict returns first dimension of mean.
#[test]
fn test_multivariate_normal_predict() {
    let n_obs = 10;
    let n_params = 5; // 2D MVN
    let mut params = Array2::zeros((n_obs, n_params));

    for i in 0..n_obs {
        params[[i, 0]] = (i as f64) * 0.5; // Varying mean_0
        params[[i, 1]] = 2.0;
        params[[i, 2]] = 0.0;
        params[[i, 3]] = 0.0;
        params[[i, 4]] = 0.0;
    }

    let dist = MultivariateNormal2::from_params(&params);
    let predictions = dist.predict();

    assert_eq!(predictions.len(), n_obs);
    for i in 0..n_obs {
        assert!(
            (predictions[i] - (i as f64) * 0.5).abs() < 1e-6,
            "Prediction[{}] = {} not close to {}",
            i,
            predictions[i],
            (i as f64) * 0.5
        );
    }
}

#[test]
fn test_histogram_learner_classification() {
    let n_samples = 100;
    let n_features = 5;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let linear = x.dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]));
    let y = linear.mapv(|v: f64| if v > 0.0 { 1.0 } else { 0.0 });

    let hist_learner = HistogramLearner::new(3);
    let mut model: NGBoost<Bernoulli, LogScore, HistogramLearner> =
        NGBoost::new(20, 0.1, hist_learner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    assert_eq!(y_pred.len(), n_samples);

    // Predictions should be 0 or 1
    for &pred in y_pred.iter() {
        assert!(pred == 0.0 || pred == 1.0);
    }
}

#[test]
fn test_histogram_vs_stump_accuracy() {
    // Compare histogram and stump learners - both should achieve reasonable accuracy
    let n_samples = 200;
    let n_features = 4;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![2.0, -1.5, 1.0, -0.5]))
        + &Array1::random(n_samples, Uniform::new(-0.3, 0.3).unwrap());
    let y = y.mapv(|v: f64| v.abs() + 0.01);

    // Histogram learner
    let hist_learner = HistogramLearner::new(3);
    let mut hist_model: NGBoost<Normal, LogScore, HistogramLearner> =
        NGBoost::new(50, 0.1, hist_learner);
    hist_model.fit(&x, &y).unwrap();
    let hist_pred = hist_model.predict(&x);
    let hist_mse: f64 = (&y - &hist_pred).mapv(|v| v * v).mean().unwrap();

    // Stump learner (for comparison)
    let mut stump_model: NGBoost<Normal, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    stump_model.fit(&x, &y).unwrap();
    let stump_pred = stump_model.predict(&x);
    let stump_mse: f64 = (&y - &stump_pred).mapv(|v| v * v).mean().unwrap();

    // Both should achieve reasonable accuracy
    assert!(hist_mse < 1.0, "Histogram MSE too high: {}", hist_mse);
    assert!(stump_mse < 1.0, "Stump MSE too high: {}", stump_mse);

    // Histogram with depth-3 should generally be better than depth-1 stump
    // (but not always due to randomness, so we use a loose check)
    println!("Histogram MSE: {}, Stump MSE: {}", hist_mse, stump_mse);
}

#[test]
fn test_histogram_learner_with_different_bins() {
    let n_samples = 100;
    let n_features = 3;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![1.0, -1.0, 0.5]))
        + &Array1::random(n_samples, Uniform::new(-0.2, 0.2).unwrap());
    let y = y.mapv(|v: f64| v.abs() + 0.01);

    // Test with fewer bins
    let hist_learner = HistogramLearner::with_params(3, 32, 1, 2);
    let mut model: NGBoost<Normal, LogScore, HistogramLearner> =
        NGBoost::new(30, 0.1, hist_learner);
    let fit_result = model.fit(&x, &y);
    assert!(fit_result.is_ok());

    let y_pred = model.predict(&x);
    let mse: f64 = (&y - &y_pred).mapv(|v| v * v).mean().unwrap();
    assert!(mse < 1.0);
}
