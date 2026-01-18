#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Multiclass Classification Example
//
// This example demonstrates how to use NGBoost for multiclass classification
// using the Categorical distribution. Unlike binary classification (Bernoulli),
// multiclass classification predicts probabilities for K > 2 classes.
//
// Run with: cargo run --example multiclass --features accelerate

use ndarray::{Array1, Array2};
use ngboost_rs::dist::categorical::{Categorical3, Categorical5};
use ngboost_rs::dist::Distribution;
use ngboost_rs::learners::default_tree_learner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{LogScore, Scorable};

fn main() {
    println!("=== NGBoost Multiclass Classification Example ===\n");

    // Example 1: 3-class classification
    println!("--- 3-Class Classification ---\n");
    run_classification::<Categorical3>(3);

    println!("\n");

    // Example 2: 5-class classification
    println!("--- 5-Class Classification ---\n");
    run_classification::<Categorical5>(5);
}

fn run_classification<D>(n_classes: usize)
where
    D: Distribution + Scorable<LogScore> + Clone + ngboost_rs::dist::ClassificationDistn,
{
    let n_samples = 150;
    let n_features = 4;

    // Generate synthetic data with n_classes clusters
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Assign class based on sample index
        let class = i % n_classes;
        y_data.push(class as f64);

        // Generate features that cluster by class
        for j in 0..n_features {
            let class_offset = class as f64 * 2.0;
            let noise = ((i * 7 + j * 13) % 100) as f64 / 100.0 - 0.5;
            let feature_val = class_offset + noise + (j as f64 * 0.1);
            x_data.push(feature_val);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    // Split into train and test
    let train_size = 120;
    let x_train = x.slice(ndarray::s![..train_size, ..]).to_owned();
    let x_test = x.slice(ndarray::s![train_size.., ..]).to_owned();
    let y_train = y.slice(ndarray::s![..train_size]).to_owned();
    let y_test = y.slice(ndarray::s![train_size..]).to_owned();

    println!(
        "Dataset: {} samples, {} features, {} classes",
        n_samples, n_features, n_classes
    );
    println!(
        "Train: {} samples, Test: {} samples\n",
        train_size,
        n_samples - train_size
    );

    // Create and train the model
    let base_learner = default_tree_learner();
    let mut model: NGBoost<D, LogScore, _> = NGBoost::new(50, 0.1, base_learner);

    println!(
        "Training NGBoost with Categorical{} distribution...",
        n_classes
    );
    model.fit(&x_train, &y_train).unwrap();
    println!("Training complete!\n");

    // Make predictions
    let predictions = model.predict(&x_test);
    let dist = model.pred_dist(&x_test);
    let class_probs = dist.class_probs();

    // Calculate accuracy
    let mut correct = 0;
    for i in 0..y_test.len() {
        if (predictions[i] - y_test[i]).abs() < 0.5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / y_test.len() as f64;

    println!("=== Results ===");
    println!("Test Accuracy: {:.1}%\n", accuracy * 100.0);

    // Show predictions with probabilities for first few samples
    println!("Sample predictions with class probabilities:");
    println!(
        "{:<8} {:>8} {:>10} {}",
        "Sample", "Actual", "Predicted", "Class Probabilities"
    );
    println!("{}", "-".repeat(60));

    for i in 0..y_test.len().min(10) {
        let probs: Vec<String> = (0..n_classes)
            .map(|c| format!("{:.2}", class_probs[[i, c]]))
            .collect();
        println!(
            "{:<8} {:>8} {:>10} [{}]",
            i,
            y_test[i] as usize,
            predictions[i] as usize,
            probs.join(", ")
        );
    }

    // Show average confidence
    let avg_max_prob: f64 = (0..y_test.len())
        .map(|i| {
            (0..n_classes)
                .map(|c| class_probs[[i, c]])
                .fold(0.0_f64, |a, b| a.max(b))
        })
        .sum::<f64>()
        / y_test.len() as f64;

    println!(
        "\nAverage prediction confidence (max probability): {:.1}%",
        avg_max_prob * 100.0
    );
}
