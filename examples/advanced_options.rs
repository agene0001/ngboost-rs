#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Advanced Options Example
//
// This example demonstrates the advanced features of NGBoost-rs:
// - Learning rate schedules (Cosine, Exponential, Linear)
// - Line search methods (Binary vs Golden Section)
// - Tikhonov regularization for numerical stability
//
// Run with: cargo run --example advanced_options --features accelerate

use ndarray::{Array1, Array2};
use ngboost_rs::dist::normal::Normal;
use ngboost_rs::learners::default_tree_learner;
use ngboost_rs::ngboost::{LearningRateSchedule, LineSearchMethod, NGBoost};
use ngboost_rs::scores::{LogScore, Scorable};

fn main() {
    println!("=== NGBoost Advanced Options Example ===\n");

    // Generate synthetic regression data
    let (x_train, y_train, x_test, y_test) = generate_data();

    println!("Dataset: 200 train samples, 50 test samples, 5 features\n");

    // 1. Default settings (baseline)
    println!("--- Configuration 1: Default Settings ---");
    let base_learner = default_tree_learner();
    let mut model1: NGBoost<Normal, LogScore, _> = NGBoost::new(100, 0.1, base_learner);
    model1.fit(&x_train, &y_train).unwrap();
    let loss1 = evaluate(&model1, &x_test, &y_test);
    println!("  Learning Rate: Constant");
    println!("  Line Search: Binary");
    println!("  Test Loss: {:.4}\n", loss1);

    // 2. Cosine annealing learning rate
    println!("--- Configuration 2: Cosine Annealing ---");
    let base_learner = default_tree_learner();
    let mut model2: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        100, // n_estimators
        0.1, // learning_rate
        base_learner,
        true,                         // natural_gradient
        1.0,                          // minibatch_frac
        1.0,                          // col_sample
        false,                        // verbose
        100.0,                        // verbose_eval
        1e-4,                         // tol
        None,                         // early_stopping_rounds
        0.1,                          // validation_fraction
        LearningRateSchedule::Cosine, // lr_schedule
        0.0,                          // tikhonov_reg
        LineSearchMethod::Binary,     // line_search_method
    );
    model2.fit(&x_train, &y_train).unwrap();
    let loss2 = evaluate(&model2, &x_test, &y_test);
    println!("  Learning Rate: Cosine Annealing");
    println!("  Line Search: Binary");
    println!("  Test Loss: {:.4}\n", loss2);

    // 3. Exponential decay learning rate
    println!("--- Configuration 3: Exponential Decay ---");
    let base_learner = default_tree_learner();
    let mut model3: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        100,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Exponential { decay_rate: 2.0 },
        0.0,
        LineSearchMethod::Binary,
    );
    model3.fit(&x_train, &y_train).unwrap();
    let loss3 = evaluate(&model3, &x_test, &y_test);
    println!("  Learning Rate: Exponential (decay_rate=2.0)");
    println!("  Line Search: Binary");
    println!("  Test Loss: {:.4}\n", loss3);

    // 4. Golden Section line search
    println!("--- Configuration 4: Golden Section Search ---");
    let base_learner = default_tree_learner();
    let mut model4: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        100,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::GoldenSection { max_iters: 20 },
    );
    model4.fit(&x_train, &y_train).unwrap();
    let loss4 = evaluate(&model4, &x_test, &y_test);
    println!("  Learning Rate: Constant");
    println!("  Line Search: Golden Section (max_iters=20)");
    println!("  Test Loss: {:.4}\n", loss4);

    // 5. Combined: Cosine + Golden Section + Tikhonov
    println!("--- Configuration 5: All Advanced Features ---");
    let base_learner = default_tree_learner();
    let mut model5: NGBoost<Normal, LogScore, _> = NGBoost::with_advanced_options(
        100,
        0.1,
        base_learner,
        true,
        1.0,
        1.0,
        false,
        100.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Cosine,
        1e-6, // Tikhonov regularization
        LineSearchMethod::GoldenSection { max_iters: 20 },
    );
    model5.fit(&x_train, &y_train).unwrap();
    let loss5 = evaluate(&model5, &x_test, &y_test);
    println!("  Learning Rate: Cosine Annealing");
    println!("  Line Search: Golden Section");
    println!("  Tikhonov Regularization: 1e-6");
    println!("  Test Loss: {:.4}\n", loss5);

    // Summary
    println!("=== Summary ===");
    println!("{:<30} {:>12}", "Configuration", "Test Loss");
    println!("{}", "-".repeat(45));
    println!("{:<30} {:>12.4}", "1. Default", loss1);
    println!("{:<30} {:>12.4}", "2. Cosine LR", loss2);
    println!("{:<30} {:>12.4}", "3. Exponential LR", loss3);
    println!("{:<30} {:>12.4}", "4. Golden Section", loss4);
    println!("{:<30} {:>12.4}", "5. All Advanced", loss5);

    // Find best
    let losses = [loss1, loss2, loss3, loss4, loss5];
    let best_idx = losses
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let configs = [
        "Default",
        "Cosine LR",
        "Exponential LR",
        "Golden Section",
        "All Advanced",
    ];
    println!(
        "\nBest configuration: {} (loss={:.4})",
        configs[best_idx], losses[best_idx]
    );
}

fn generate_data() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let n_train = 200;
    let n_test = 50;
    let n_features = 5;

    let mut x_train_data = Vec::with_capacity(n_train * n_features);
    let mut y_train_data = Vec::with_capacity(n_train);

    for i in 0..n_train {
        let mut y = 0.0;
        for j in 0..n_features {
            let val = ((i * 17 + j * 31) % 100) as f64 / 50.0 - 1.0;
            x_train_data.push(val);
            y += val * (j as f64 + 1.0) * 0.5;
        }
        let noise = ((i * 7) % 20) as f64 / 20.0 - 0.5;
        y_train_data.push(y + noise);
    }

    let mut x_test_data = Vec::with_capacity(n_test * n_features);
    let mut y_test_data = Vec::with_capacity(n_test);

    for i in 0..n_test {
        let mut y = 0.0;
        for j in 0..n_features {
            let val = ((i * 23 + j * 41) % 100) as f64 / 50.0 - 1.0;
            x_test_data.push(val);
            y += val * (j as f64 + 1.0) * 0.5;
        }
        let noise = ((i * 11) % 20) as f64 / 20.0 - 0.5;
        y_test_data.push(y + noise);
    }

    (
        Array2::from_shape_vec((n_train, n_features), x_train_data).unwrap(),
        Array1::from_vec(y_train_data),
        Array2::from_shape_vec((n_test, n_features), x_test_data).unwrap(),
        Array1::from_vec(y_test_data),
    )
}

fn evaluate<D, S, B>(model: &NGBoost<D, S, B>, x: &Array2<f64>, y: &Array1<f64>) -> f64
where
    D: ngboost_rs::dist::Distribution + Scorable<S> + Clone,
    S: ngboost_rs::scores::Score,
    B: ngboost_rs::learners::BaseLearner + Clone,
{
    model.score(x, y)
}
