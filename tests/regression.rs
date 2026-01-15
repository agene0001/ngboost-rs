#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::ngboost::NGBRegressor;

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    (y_true - y_pred).mapv(|a| a.powi(2)).mean().unwrap()
}

#[test]
fn test_regression_synthetic() {
    // 1. Generate synthetic data
    let n_samples = 100;
    let n_features = 5;
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let y = x.dot(&Array1::from(vec![1.5, -2.3, 0.4, 3.1, -1.1]))
        + &Array1::random(n_samples, Uniform::new(-0.5, 0.5).unwrap());

    // 2. Instantiate and fit NGBRegressor
    let mut regressor = NGBRegressor::new(100, 0.1);
    let fit_result = regressor.fit(&x, &y);
    assert!(fit_result.is_ok());

    // 3. Make predictions
    let y_pred = regressor.predict(&x);

    // 4. Assert MSE
    let mse = mean_squared_error(&y, &y_pred);
    println!("MSE: {}", mse);
    assert!(mse < 0.5);
}
