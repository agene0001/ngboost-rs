//! Example demonstrating uncertainty quantification with NGBoost.
//!
//! NGBoost provides full probability distributions, not just point predictions.
//! This example shows how to use the predicted distributions to:
//! - Estimate prediction uncertainty
//! - Compute confidence intervals
//! - Identify uncertain predictions

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
    println!("NGBoost Uncertainty Quantification Example");
    println!("==========================================\n");

    // Generate data with heteroscedastic noise (variance depends on x)
    let n_train = 500;
    let n_test = 50;

    // Training data
    let x_train = Array2::random((n_train, 1), Uniform::new(0., 10.).unwrap());
    let y_train: Array1<f64> = x_train
        .column(0)
        .iter()
        .enumerate()
        .map(|(_, &x)| {
            // True function: y = sin(x) + noise
            // Noise increases with x (heteroscedastic)
            let noise_scale = 0.1 + 0.1 * x;
            let noise = (rand::random::<f64>() - 0.5) * 2.0 * noise_scale;
            x.sin() + noise
        })
        .collect();

    // Test data - evenly spaced for visualization
    let x_test: Array2<f64> =
        Array2::from_shape_fn((n_test, 1), |(i, _)| i as f64 * 10.0 / (n_test - 1) as f64);

    println!("Training NGBoost model...");

    let mut model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(200, 0.05, StumpLearner);
    model.fit(&x_train, &y_train).unwrap();

    println!("Model trained!\n");

    // Get predicted distributions
    let pred_dist = model.pred_dist(&x_test);

    // Compute confidence intervals (approximately 95% for Normal: mean ± 1.96*std)
    let z_95 = 1.96;

    println!("Predictions with 95% Confidence Intervals:");
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12}",
        "x", "y_pred", "std_dev", "CI_lower", "CI_upper"
    );
    println!("{}", "-".repeat(60));

    for i in 0..n_test {
        let x = x_test[[i, 0]];
        let mean = pred_dist.loc[i];
        let std = pred_dist.scale[i];
        let ci_lower = mean - z_95 * std;
        let ci_upper = mean + z_95 * std;

        println!(
            "{:<8.2} {:<12.4} {:<12.4} {:<12.4} {:<12.4}",
            x, mean, std, ci_lower, ci_upper
        );
    }

    println!("\n");

    // Analyze uncertainty patterns
    let stds: Vec<f64> = pred_dist.scale.iter().cloned().collect();
    let min_std = stds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_std = stds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_std: f64 = stds.iter().sum::<f64>() / stds.len() as f64;

    println!("Uncertainty Statistics:");
    println!("-----------------------");
    println!("Minimum std dev: {:.4}", min_std);
    println!("Maximum std dev: {:.4}", max_std);
    println!("Mean std dev:    {:.4}", mean_std);
    println!();

    // Find most and least certain predictions
    let (min_idx, _) = stds
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let (max_idx, _) = stds
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("Most certain prediction:");
    println!(
        "  x = {:.2}, predicted = {:.4} ± {:.4}",
        x_test[[min_idx, 0]],
        pred_dist.loc[min_idx],
        pred_dist.scale[min_idx]
    );

    println!("Least certain prediction:");
    println!(
        "  x = {:.2}, predicted = {:.4} ± {:.4}",
        x_test[[max_idx, 0]],
        pred_dist.loc[max_idx],
        pred_dist.scale[max_idx]
    );

    println!("\nNote: The model should learn that uncertainty increases with x,");
    println!("reflecting the heteroscedastic nature of the training data.");
}
