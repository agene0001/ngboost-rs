//! Basic regression example using NGBoost with Normal distribution.
//!
//! This example demonstrates:
//! - Creating synthetic regression data
//! - Training an NGBoost model with Normal distribution
//! - Making predictions and evaluating performance

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::Normal;
use ngboost_rs::learners::StumpLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;

fn main() {
    println!("NGBoost Regression Example");
    println!("==========================\n");

    // Generate synthetic data
    let n_train = 500;
    let n_test = 100;
    let n_features = 5;

    // True coefficients
    let true_coef = Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]);

    // Training data
    let x_train = Array2::random((n_train, n_features), Uniform::new(0., 1.).unwrap());
    let noise_train = Array1::random(n_train, Uniform::new(-0.5, 0.5).unwrap());
    let y_train = x_train.dot(&true_coef) + &noise_train;

    // Test data
    let x_test = Array2::random((n_test, n_features), Uniform::new(0., 1.).unwrap());
    let noise_test = Array1::random(n_test, Uniform::new(-0.5, 0.5).unwrap());
    let y_test = x_test.dot(&true_coef) + &noise_test;

    println!("Training samples: {}", n_train);
    println!("Test samples: {}", n_test);
    println!("Features: {}", n_features);
    println!();

    // Create and train the model
    let n_estimators = 100;
    let learning_rate = 0.1;

    println!("Training NGBoost with {} estimators...", n_estimators);

    let mut model: NGBoost<Normal, LogScore, StumpLearner> =
        NGBoost::new(n_estimators, learning_rate, StumpLearner);

    model.fit(&x_train, &y_train).expect("Failed to fit model");

    println!("Training complete!\n");

    // Make predictions
    let y_pred_train = model.predict(&x_train);
    let y_pred_test = model.predict(&x_test);

    // Calculate metrics
    let mse_train = mean_squared_error(&y_train, &y_pred_train);
    let mse_test = mean_squared_error(&y_test, &y_pred_test);
    let rmse_train = mse_train.sqrt();
    let rmse_test = mse_test.sqrt();

    println!("Results:");
    println!("--------");
    println!("Training RMSE: {:.4}", rmse_train);
    println!("Test RMSE:     {:.4}", rmse_test);
    println!();

    // Get the predicted distribution for uncertainty estimation
    let pred_dist = model.pred_dist(&x_test);

    println!("Predicted distribution parameters (first 5 test samples):");
    println!(
        "{:<10} {:<10} {:<10} {:<10}",
        "y_true", "y_pred", "loc", "scale"
    );
    for i in 0..5 {
        println!(
            "{:<10.4} {:<10.4} {:<10.4} {:<10.4}",
            y_test[i], y_pred_test[i], pred_dist.loc[i], pred_dist.scale[i]
        );
    }
}

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    (y_true - y_pred).mapv(|a| a.powi(2)).mean().unwrap()
}
