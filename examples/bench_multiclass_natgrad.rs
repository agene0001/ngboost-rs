//! Quick timing harness for the Categorical Sherman-Morrison natural gradient
//! vs the generic dense-metric + LAPACK path, plus an end-to-end multiclass fit.
//!
//! Run: cargo run --release --features accelerate --example bench_multiclass_natgrad

use ndarray::{Array1, Array2};
use ngboost_rs::dist::categorical::Categorical;
use ngboost_rs::Distribution;
use ngboost_rs::scores::{natural_gradient_regularized, LogScore, Scorable};
use ngboost_rs::NGBMultiClassifier;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    const K: usize = 10;
    let n = 5000;

    // Random logits / labels
    let params = Array2::from_shape_fn((n, K - 1), |_| rng.random_range(-2.0..2.0));
    let y = Array1::from_shape_fn(n, |_| rng.random_range(0..K) as f64);
    let dist = Categorical::<K>::from_params(&params);

    // Warm up + correctness spot check
    let fast = Scorable::<LogScore>::dense_natural_grad(&dist, &y, 0.0);
    let grad = Scorable::<LogScore>::d_score(&dist, &y);
    let metric = Scorable::<LogScore>::metric(&dist);
    let slow = natural_gradient_regularized(&grad, &metric, 0.0);
    let max_diff = fast
        .iter()
        .zip(slow.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("max |fast - lapack| = {max_diff:.3e}");

    let reps = 20;

    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(Scorable::<LogScore>::dense_natural_grad(&dist, &y, 0.0));
    }
    let fast_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    let t = Instant::now();
    for _ in 0..reps {
        let grad = Scorable::<LogScore>::d_score(&dist, &y);
        let metric = Scorable::<LogScore>::metric(&dist);
        std::hint::black_box(natural_gradient_regularized(&grad, &metric, 0.0));
    }
    let slow_ms = t.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    println!("natural gradient, n={n}, K={K}:");
    println!("  closed form (Sherman-Morrison): {fast_ms:.3} ms");
    println!("  dense metric + LAPACK:          {slow_ms:.3} ms");
    println!("  speedup: {:.1}x", slow_ms / fast_ms);

    // End-to-end multiclass fit (the closed form is now the default path)
    let n_train = 2000;
    let p = 10;
    let x = Array2::from_shape_fn((n_train, p), |_| rng.random_range(-1.0..1.0));
    let y_cls = Array1::from_shape_fn(n_train, |i| {
        let s: f64 = x.row(i).sum();
        (((s + 5.0) / 10.0 * K as f64) as usize).min(K - 1) as f64
    });

    for _ in 0..3 {
        let t = Instant::now();
        let mut model: NGBMultiClassifier<K> = NGBMultiClassifier::new(100, 0.1);
        model.fit(&x, &y_cls).unwrap();
        println!(
            "end-to-end NGBMultiClassifier<{K}> fit, n={n_train}, p={p}, 100 est: {:.0} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
}
