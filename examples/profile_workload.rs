//! Profiling workload: repeated fits/predicts on the default regression path.
//!
//! Build with debug symbols and record:
//!   cargo build --profile bench --features accelerate --example profile_workload
//!   samply record --save-only -o /tmp/ngboost_profile.json \
//!     ./target/bench/examples/profile_workload

use ndarray::{Array1, Array2};
use ngboost_rs::NGBRegressor;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 5000;
    let p = 10;

    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let signal = r[0] * 2.0 + r[1] * r[1] - (r[2] * 1.5).sin() * 3.0 + r[3] * r[4];
        let noise: f64 = rng.random_range(-0.5..0.5) * (1.0 + r[5].abs());
        signal + noise
    });

    let t = Instant::now();
    let mut last_model = None;
    for _ in 0..30 {
        let mut model = NGBRegressor::new(200, 0.05);
        model.fit(&x, &y).unwrap();
        last_model = Some(model);
    }
    println!("fit phase: {:.1} s", t.elapsed().as_secs_f64());

    let model = last_model.unwrap();
    let t = Instant::now();
    for _ in 0..150 {
        std::hint::black_box(model.pred_dist(&x));
    }
    println!("predict phase: {:.1} s", t.elapsed().as_secs_f64());
}
