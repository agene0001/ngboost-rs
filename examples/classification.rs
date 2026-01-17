#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Binary classification example using NGBoost with Bernoulli distribution.
//
// This example demonstrates:
// - Creating synthetic classification data
// - Training an NGBoost classifier
// - Making predictions and getting class probabilities


use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ngboost_rs::dist::{categorical::Bernoulli, ClassificationDistn};
use ngboost_rs::learners::StumpLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;

fn main() {
    println!("NGBoost Binary Classification Example");
    println!("=====================================\n");

    // Generate synthetic binary classification data
    let n_train = 400;
    let n_test = 100;
    let n_features = 4;

    // True coefficients for decision boundary
    let true_coef = Array1::from(vec![2.0, -1.5, 1.0, -0.5]);

    // Training data
    let x_train = Array2::random((n_train, n_features), Uniform::new(-1., 1.).unwrap());
    let linear_train = x_train.dot(&true_coef);
    let y_train = linear_train.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

    // Test data
    let x_test = Array2::random((n_test, n_features), Uniform::new(-1., 1.).unwrap());
    let linear_test = x_test.dot(&true_coef);
    let y_test = linear_test.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

    println!("Training samples: {}", n_train);
    println!("Test samples: {}", n_test);
    println!("Features: {}", n_features);
    println!(
        "Class distribution (train): {:.1}% class 0, {:.1}% class 1",
        100.0 * y_train.iter().filter(|&&y| y == 0.0).count() as f64 / n_train as f64,
        100.0 * y_train.iter().filter(|&&y| y == 1.0).count() as f64 / n_train as f64
    );
    println!();

    // Create and train the model
    let n_estimators = 50;
    let learning_rate = 0.1;

    println!(
        "Training NGBoost classifier with {} estimators...",
        n_estimators
    );

    let mut model: NGBoost<Bernoulli, LogScore, StumpLearner> =
        NGBoost::new(n_estimators, learning_rate, StumpLearner);

    model.fit(&x_train, &y_train).expect("Failed to fit model");

    println!("Training complete!\n");

    // Make predictions
    let y_pred_train = model.predict(&x_train);
    let y_pred_test = model.predict(&x_test);

    // Calculate accuracy
    let acc_train = accuracy(&y_train, &y_pred_train);
    let acc_test = accuracy(&y_test, &y_pred_test);

    println!("Results:");
    println!("--------");
    println!("Training Accuracy: {:.2}%", acc_train * 100.0);
    println!("Test Accuracy:     {:.2}%", acc_test * 100.0);
    println!();

    // Get predicted probabilities
    let pred_dist = model.pred_dist(&x_test);
    let probs = pred_dist.class_probs();

    println!("Predicted probabilities (first 10 test samples):");
    println!(
        "{:<8} {:<8} {:<12} {:<12}",
        "y_true", "y_pred", "P(class=0)", "P(class=1)"
    );
    for i in 0..10 {
        println!(
            "{:<8.0} {:<8.0} {:<12.4} {:<12.4}",
            y_test[i],
            y_pred_test[i],
            probs[[i, 0]],
            probs[[i, 1]]
        );
    }
}

fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    correct as f64 / y_true.len() as f64
}
