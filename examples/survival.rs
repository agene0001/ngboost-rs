#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Survival Analysis Example
//
// This example demonstrates how to use NGBoost for survival analysis with
// right-censored data. Survival analysis is used when we want to model the
// time until an event occurs, but some observations may be censored (i.e.,
// the event hasn't occurred by the end of the study).
//
// Run with: cargo run --example survival --features accelerate

use ndarray::{Array1, Array2};
use ngboost_rs::survival::NGBSurvivalLogNormal;

fn main() {
    println!("=== NGBoost Survival Analysis Example ===\n");

    // Generate synthetic survival data
    // In real applications, this would be patient survival times, customer churn times, etc.
    let n_samples = 100;
    let n_features = 5;

    // Create feature matrix (e.g., patient characteristics, risk factors)
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            // Create some structured features
            let val = ((i * 17 + j * 31) % 100) as f64 / 100.0;
            x_data.push(val);
        }
    }
    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

    // Generate survival times (log-normal distributed based on features)
    let mut time = Vec::with_capacity(n_samples);
    let mut event = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Base survival time influenced by features
        let base_time = 1.0 + x[[i, 0]] * 2.0 + x[[i, 1]] * 1.5;
        // Add some noise
        let noise = ((i * 7) % 10) as f64 / 10.0;
        let survival_time = (base_time + noise).max(0.1);
        time.push(survival_time);

        // Simulate censoring: ~30% of observations are censored
        let is_observed = (i * 13) % 10 >= 3;
        event.push(if is_observed { 1.0 } else { 0.0 });
    }

    let time = Array1::from_vec(time);
    let event = Array1::from_vec(event);

    let n_censored = event.iter().filter(|&&e| e == 0.0).count();
    println!("Dataset: {} samples, {} features", n_samples, n_features);
    println!(
        "Events: {} observed, {} censored ({:.1}% censoring rate)\n",
        n_samples - n_censored,
        n_censored,
        100.0 * n_censored as f64 / n_samples as f64
    );

    // Split into train and test sets
    let train_size = 80;
    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();
    let time_train = time.slice(ndarray::s![..train_size]).to_owned();
    let time_test = time.slice(ndarray::s![train_size..]).to_owned();
    let event_train = event.slice(ndarray::s![..train_size]).to_owned();
    let event_test = event.slice(ndarray::s![train_size..]).to_owned();

    // Create and train the survival model
    println!("Training NGBSurvival with LogNormal distribution...");
    let mut model = NGBSurvivalLogNormal::lognormal(100, 0.05);
    model.fit(&x_train, &time_train, &event_train).unwrap();
    println!("Training complete!\n");

    // Make predictions
    let predictions = model.predict(&x_test);
    let _dist = model.pred_dist(&x_test);

    println!("=== Predictions on Test Set ===");
    println!(
        "{:<10} {:>12} {:>12} {:>10}",
        "Sample", "Actual Time", "Predicted", "Event"
    );
    println!("{}", "-".repeat(50));

    for i in 0..x_test.nrows().min(10) {
        let event_str = if event_test[i] > 0.5 {
            "observed"
        } else {
            "censored"
        };
        println!(
            "{:<10} {:>12.3} {:>12.3} {:>10}",
            i, time_test[i], predictions[i], event_str
        );
    }

    // Show distribution parameters for uncertainty quantification
    println!("\n=== Uncertainty Quantification ===");
    println!("The LogNormal distribution provides full uncertainty estimates.");
    println!("For each prediction, we have access to the entire distribution.\n");

    // Calculate some summary statistics
    let mean_actual: f64 = time_test.iter().sum::<f64>() / time_test.len() as f64;
    let mean_predicted: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;

    println!("Mean actual survival time: {:.3}", mean_actual);
    println!("Mean predicted survival time: {:.3}", mean_predicted);

    // Calculate MAE
    let mae: f64 = time_test
        .iter()
        .zip(predictions.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / time_test.len() as f64;
    println!("Mean Absolute Error: {:.3}", mae);

    println!("\n=== Model Information ===");
    println!("Number of boosting iterations: {}", 100);
    println!("Learning rate: 0.05");
    println!("Distribution: LogNormal (2 parameters: loc, scale)");
}
