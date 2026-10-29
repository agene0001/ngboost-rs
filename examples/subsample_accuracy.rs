//! Accuracy + speed A/B for the row-subsampled fit path (behavioral change:
//! full-X quantile bin edges instead of per-minibatch edges). Run this
//! binary from two builds; seeds drive identical row draws in both, so the
//! only difference is the binning. Compare per-seed held-out NLL/RMSE and
//! total fit time.
use ndarray::{Array1, Array2};
use ngboost_rs::NGBHistRegressor;
use ngboost_rs::learners::default_base_learner;
use ngboost_rs::scores::{LogScore, Scorable};
use std::time::Instant;

fn make_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let s = seed as f64;
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        ((i as f64 * 0.7315 + j as f64 * 1.618 + s * 12.9898).sin() * 43758.5453).fract() * 4.0
            - 2.0
    });
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let signal = r[0] * 2.0 + r[1] * r[1] - (r[2] * 1.5).sin() * 3.0 + r[3] * r[p - 1];
        let noise = ((i as f64 * 2.399 + s * 78.233).sin() * 43758.5453).fract() - 0.5;
        signal + noise * (1.0 + r[4].abs())
    });
    (x, y)
}

fn main() {
    let mut total_fit = 0.0f64;
    let mut sum_nll = 0.0f64;
    let mut sum_rmse = 0.0f64;
    let n_seeds = 8u64;

    for seed in 0..n_seeds {
        let (x, y) = make_data(2000, 8, seed);
        let (xt, yt) = make_data(800, 8, seed + 1000);

        let mut m = NGBHistRegressor::with_seed(300, 0.05, default_base_learner(), seed);
        m.minibatch_frac = 0.7;
        let t = Instant::now();
        m.fit(&x, &y).unwrap();
        let dt = t.elapsed().as_secs_f64();
        total_fit += dt;

        let dist = m.pred_dist(&xt);
        let nll = Scorable::<LogScore>::score(&dist, &yt).mean().unwrap();
        let pred = m.predict(&xt);
        let rmse = (&pred - &yt).mapv(|v| v * v).mean().unwrap().sqrt();
        sum_nll += nll;
        sum_rmse += rmse;
        println!("seed {seed}: nll {nll:.15} rmse {rmse:.15} fit {dt:.3}s");
    }
    println!(
        "MEAN nll {:.15} rmse {:.15} total_fit {:.2}s",
        sum_nll / n_seeds as f64,
        sum_rmse / n_seeds as f64,
        total_fit
    );
}
