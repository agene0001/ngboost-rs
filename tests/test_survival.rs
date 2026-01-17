#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Survival analysis tests matching Python's survival-related tests.
//
// Tests the NGBSurvival implementation with censored data.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::exponential::Exponential;
use ngboost_rs::dist::lognormal::LogNormal;
use ngboost_rs::dist::Distribution;
use ngboost_rs::learners::default_tree_learner;
use ngboost_rs::scores::{CRPScoreCensored, CensoredScorable, LogScoreCensored, SurvivalData};
use ngboost_rs::survival::{NGBSurvival, NGBSurvivalExponential, NGBSurvivalLogNormal};

// ============================================================================
// Helper functions for generating survival data
// ============================================================================

/// Generate synthetic survival data with censoring.
/// Uses deterministic pattern based on seed for reproducibility without external RNG.
fn generate_survival_data(
    n_samples: usize,
    n_features: usize,
    censoring_rate: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    // Generate features using ndarray_rand
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());

    // Generate true event times based on features (exponential-like)
    // Use a simple deterministic pattern for reproducibility
    let mut true_times = Array1::zeros(n_samples);
    for i in 0..n_samples {
        // Base rate depends on features
        let rate = 0.5 + x[[i, 0]] * 0.3 + x[[i, 1 % n_features]] * 0.2;
        // Use a pseudo-random value based on seed and index
        let u = ((seed.wrapping_mul(1103515245).wrapping_add(12345 + i as u64)) % 1000000) as f64
            / 1000000.0;
        let u = u.max(0.001); // Avoid log(0)
        true_times[i] = -u.ln() / rate;
    }

    // Generate censoring times
    let censor_threshold = true_times.mean().unwrap() * (1.0 / (1.0 - censoring_rate + 0.1));
    let mut observed_times = Array1::zeros(n_samples);
    let mut events = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Use another pseudo-random value
        let u = ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407 + i as u64))
            % 1000000) as f64
            / 1000000.0;
        let censor_time = u * censor_threshold * 2.0;
        if true_times[i] <= censor_time {
            observed_times[i] = true_times[i];
            events[i] = 1.0; // Event observed
        } else {
            observed_times[i] = censor_time;
            events[i] = 0.0; // Censored
        }
    }

    // Ensure all times are positive
    observed_times = observed_times.mapv(|v| v.max(0.01));

    (x, observed_times, events)
}

// ============================================================================
// Censored score tests
// ============================================================================

#[test]
fn test_lognormal_censored_score() {
    // Create a LogNormal distribution
    let params = ndarray::array![[1.0, 0.5], [1.5, 0.3], [0.8, 0.4]];
    let dist = LogNormal::from_params(&params);

    // Create survival data
    let time = ndarray::array![2.0, 3.0, 1.5];
    let event = ndarray::array![true, false, true];
    let y = SurvivalData::new(event, time);

    // Compute censored scores
    let scores = CensoredScorable::<LogScoreCensored>::censored_score(&dist, &y);

    assert_eq!(scores.len(), 3);

    // Scores should be positive
    for &s in scores.iter() {
        assert!(s.is_finite(), "Score should be finite");
        assert!(s > 0.0, "Score should be positive");
    }
}

#[test]
fn test_lognormal_censored_gradient() {
    let params = ndarray::array![[1.0, 0.5], [1.5, 0.3]];
    let dist = LogNormal::from_params(&params);

    let time = ndarray::array![2.0, 3.0];
    let event = ndarray::array![true, false];
    let y = SurvivalData::new(event, time);

    let grad = CensoredScorable::<LogScoreCensored>::censored_d_score(&dist, &y);

    assert_eq!(grad.shape(), &[2, 2]);

    // Gradients should be finite
    for &g in grad.iter() {
        assert!(g.is_finite(), "Gradient should be finite");
    }
}

#[test]
fn test_exponential_censored_score() {
    let params = ndarray::array![[0.5], [1.0], [0.2]];
    let dist = Exponential::from_params(&params);

    let time = ndarray::array![1.0, 2.0, 0.5];
    let event = ndarray::array![true, false, true];
    let y = SurvivalData::new(event, time);

    let scores = CensoredScorable::<LogScoreCensored>::censored_score(&dist, &y);

    assert_eq!(scores.len(), 3);

    for &s in scores.iter() {
        assert!(s.is_finite(), "Score should be finite");
    }
}

#[test]
fn test_exponential_crps_censored() {
    let params = ndarray::array![[0.5], [1.0]];
    let dist = Exponential::from_params(&params);

    let time = ndarray::array![1.0, 2.0];
    let event = ndarray::array![true, false];
    let y = SurvivalData::new(event, time);

    let scores = CensoredScorable::<CRPScoreCensored>::censored_score(&dist, &y);

    assert_eq!(scores.len(), 2);

    for &s in scores.iter() {
        assert!(s.is_finite(), "CRPS score should be finite");
    }
}

// ============================================================================
// NGBSurvival model tests
// ============================================================================

#[test]
fn test_ngbsurvival_lognormal_fit() {
    let (x, time, event) = generate_survival_data(100, 5, 0.3, 42);

    let mut model =
        NGBSurvival::<LogNormal, LogScoreCensored, _>::new(50, 0.1, default_tree_learner());

    let result = model.fit(&x, &time, &event);
    assert!(result.is_ok(), "Fit should succeed");

    // Make predictions
    let preds = model.predict(&x);

    assert_eq!(preds.len(), x.nrows());

    // Predictions should be positive (survival times)
    for &p in preds.iter() {
        assert!(p > 0.0, "Survival time predictions should be positive");
        assert!(p.is_finite(), "Predictions should be finite");
    }
}

#[test]
fn test_ngbsurvival_exponential_fit() {
    let (x, time, event) = generate_survival_data(100, 5, 0.3, 42);

    let mut model =
        NGBSurvival::<Exponential, LogScoreCensored, _>::new(50, 0.1, default_tree_learner());

    let result = model.fit(&x, &time, &event);
    assert!(result.is_ok(), "Fit should succeed");

    let preds = model.predict(&x);

    assert_eq!(preds.len(), x.nrows());

    for &p in preds.iter() {
        assert!(p > 0.0, "Predictions should be positive");
        assert!(p.is_finite(), "Predictions should be finite");
    }
}

#[test]
fn test_ngbsurvival_convenience_constructors() {
    let (x, time, event) = generate_survival_data(80, 4, 0.25, 42);

    // Test LogNormal convenience constructor
    let mut model_ln = NGBSurvivalLogNormal::lognormal(30, 0.1);
    model_ln.fit(&x, &time, &event).expect("Fit should succeed");

    let preds_ln = model_ln.predict(&x);
    assert_eq!(preds_ln.len(), x.nrows());

    // Test Exponential convenience constructor
    let mut model_exp = NGBSurvivalExponential::exponential(30, 0.1);
    model_exp
        .fit(&x, &time, &event)
        .expect("Fit should succeed");

    let preds_exp = model_exp.predict(&x);
    assert_eq!(preds_exp.len(), x.nrows());
}

#[test]
fn test_ngbsurvival_with_options() {
    let (x, time, event) = generate_survival_data(100, 5, 0.3, 42);

    let mut model = NGBSurvival::<LogNormal, LogScoreCensored, _>::with_options(
        100,
        0.05,
        default_tree_learner(),
        true,  // natural_gradient
        0.8,   // minibatch_frac
        1.0,   // col_sample
        false, // verbose
        50,    // verbose_eval
        1e-4,  // tol
        None,  // early_stopping_rounds
        0.1,   // validation_fraction
    );

    model.fit(&x, &time, &event).expect("Fit should succeed");

    let preds = model.predict(&x);
    assert_eq!(preds.len(), x.nrows());
}

#[test]
fn test_ngbsurvival_pred_dist() {
    let (x, time, event) = generate_survival_data(50, 4, 0.3, 42);

    let mut model = NGBSurvivalLogNormal::lognormal(30, 0.1);
    model.fit(&x, &time, &event).expect("Fit should succeed");

    let dist = model.pred_dist(&x);

    // Distribution should have proper parameters
    assert_eq!(dist.loc.len(), x.nrows());
    assert_eq!(dist.scale.len(), x.nrows());

    // Scales should be positive
    for &s in dist.scale.iter() {
        assert!(s > 0.0, "Scale should be positive");
    }
}

#[test]
fn test_ngbsurvival_all_censored() {
    let (x, time, _) = generate_survival_data(50, 4, 0.0, 42);
    let event = Array1::zeros(50); // All censored

    let mut model = NGBSurvivalExponential::exponential(20, 0.1);
    let result = model.fit(&x, &time, &event);

    // Should still fit, even with all censored data
    assert!(result.is_ok());

    let preds = model.predict(&x);
    assert_eq!(preds.len(), x.nrows());
}

#[test]
fn test_ngbsurvival_all_observed() {
    let (x, time, _) = generate_survival_data(50, 4, 0.0, 42);
    let event = Array1::ones(50); // All observed

    let mut model = NGBSurvivalLogNormal::lognormal(20, 0.1);
    model.fit(&x, &time, &event).expect("Fit should succeed");

    let preds = model.predict(&x);
    assert_eq!(preds.len(), x.nrows());

    // With all observed data, predictions should correlate with observed times
    // Just check they're in a reasonable range
    let time_mean = time.mean().unwrap();
    for &p in preds.iter() {
        assert!(p > time_mean * 0.1 && p < time_mean * 10.0);
    }
}

#[test]
fn test_ngbsurvival_validation_error() {
    let (x, _time, event) = generate_survival_data(50, 4, 0.3, 42);

    // Mismatched dimensions
    let wrong_time = Array1::zeros(30);

    let mut model = NGBSurvivalLogNormal::lognormal(20, 0.1);
    let result = model.fit(&x, &wrong_time, &event);

    assert!(result.is_err(), "Should fail with mismatched dimensions");
}

#[test]
fn test_survival_data_struct() {
    // Test SurvivalData constructors
    let time = ndarray::array![1.0, 2.0, 3.0];
    let event_f64 = ndarray::array![1.0, 0.0, 1.0];

    let y = SurvivalData::from_arrays(&time, &event_f64);

    assert_eq!(y.len(), 3);
    assert!(!y.is_empty());
    assert!(y.event[0]);
    assert!(!y.event[1]);
    assert!(y.event[2]);

    // Test uncensored constructor
    let y_uncensored = SurvivalData::uncensored(time.clone());

    assert!(y_uncensored.event.iter().all(|&e| e));
}
