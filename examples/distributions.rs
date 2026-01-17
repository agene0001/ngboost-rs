#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Example demonstrating different probability distributions available in NGBoost.
//
// This example shows how to use various distributions for different types of data:
// - Normal: General continuous data
// - Poisson: Count data
// - Gamma: Positive continuous data (e.g., wait times)
// - Laplace: Data with heavy tails


use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::{Exponential, Gamma, Laplace, Normal, Poisson};
use ngboost_rs::learners::StumpLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;

fn main() {
    println!("NGBoost Distribution Comparison Example");
    println!("=======================================\n");

    let n_samples = 200;
    let n_features = 3;

    // Generate features
    let x = Array2::random((n_samples, n_features), Uniform::new(0., 1.).unwrap());
    let coef = Array1::from(vec![2.0, -1.0, 1.5]);

    // =========================================================================
    // Normal Distribution - for general continuous data
    // =========================================================================
    println!("1. Normal Distribution");
    println!("   Use case: General continuous data, symmetric errors");

    let y_normal = x.dot(&coef) + &Array1::random(n_samples, Uniform::new(-1., 1.).unwrap());

    let mut model_normal: NGBoost<Normal, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    model_normal.fit(&x, &y_normal).unwrap();

    let pred_normal = model_normal.predict(&x);
    let rmse_normal = rmse(&y_normal, &pred_normal);
    println!("   RMSE: {:.4}\n", rmse_normal);

    // =========================================================================
    // Laplace Distribution - for heavy-tailed data
    // =========================================================================
    println!("2. Laplace Distribution");
    println!("   Use case: Data with outliers, heavy tails");

    // Add some outliers
    let mut y_laplace = x.dot(&coef) + &Array1::random(n_samples, Uniform::new(-0.5, 0.5).unwrap());
    for i in 0..10 {
        y_laplace[i * 20] += 5.0; // Add outliers
    }

    let mut model_laplace: NGBoost<Laplace, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    model_laplace.fit(&x, &y_laplace).unwrap();

    let pred_laplace = model_laplace.predict(&x);
    let rmse_laplace = rmse(&y_laplace, &pred_laplace);
    println!("   RMSE: {:.4}\n", rmse_laplace);

    // =========================================================================
    // Poisson Distribution - for count data
    // =========================================================================
    println!("3. Poisson Distribution");
    println!("   Use case: Count data (integers >= 0)");

    // Generate count data
    let linear = x.dot(&coef);
    let y_poisson = linear.mapv(|v: f64| (v.exp() * 2.0).round().max(0.0));

    let mut model_poisson: NGBoost<Poisson, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    model_poisson.fit(&x, &y_poisson).unwrap();

    let pred_poisson = model_poisson.predict(&x);
    let rmse_poisson = rmse(&y_poisson, &pred_poisson);
    println!("   RMSE: {:.4}", rmse_poisson);
    println!(
        "   Sample predictions: {:?}\n",
        &pred_poisson.slice(ndarray::s![0..5])
    );

    // =========================================================================
    // Gamma Distribution - for positive continuous data
    // =========================================================================
    println!("4. Gamma Distribution");
    println!("   Use case: Positive continuous data (e.g., durations, costs)");

    // Generate positive data
    let y_gamma = (x.dot(&coef) + 3.0).mapv(|v: f64| v.max(0.1));

    let mut model_gamma: NGBoost<Gamma, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    model_gamma.fit(&x, &y_gamma).unwrap();

    let pred_gamma = model_gamma.predict(&x);
    let rmse_gamma = rmse(&y_gamma, &pred_gamma);
    println!("   RMSE: {:.4}\n", rmse_gamma);

    // =========================================================================
    // Exponential Distribution - for waiting times
    // =========================================================================
    println!("5. Exponential Distribution");
    println!("   Use case: Waiting times, time-to-event data");

    // Generate exponential-like data
    let y_exp = (x.dot(&coef) + 2.0).mapv(|v: f64| v.max(0.01));

    let mut model_exp: NGBoost<Exponential, LogScore, StumpLearner> =
        NGBoost::new(50, 0.1, StumpLearner);
    model_exp.fit(&x, &y_exp).unwrap();

    let pred_exp = model_exp.predict(&x);
    let rmse_exp = rmse(&y_exp, &pred_exp);
    println!("   RMSE: {:.4}\n", rmse_exp);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("Summary");
    println!("-------");
    println!("Choose the distribution that best matches your data characteristics:");
    println!("- Normal: Symmetric, continuous data");
    println!("- Laplace: Heavy-tailed, outlier-prone data");
    println!("- Poisson: Non-negative integer counts");
    println!("- Gamma: Positive continuous (skewed right)");
    println!("- Exponential: Positive continuous (memoryless)");
}

fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mse = (y_true - y_pred).mapv(|a| a.powi(2)).mean().unwrap();
    mse.sqrt()
}
