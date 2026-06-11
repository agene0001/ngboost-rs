//! Prints held-out NLL/RMSE at full precision for the default regression path.
//! Used to A/B numeric changes (e.g. vForce exp/ln): run before and after,
//! diff the output.
//!
//!   cargo run --release --features accelerate --example accuracy_snapshot

use ndarray::{Array1, Array2, s};
use ngboost_rs::NGBRegressor;
use ngboost_rs::dist::DistributionMethods;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn main() {
    for seed in [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 2200;
        let p = 10;
        let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
        let y = Array1::from_shape_fn(n, |i| {
            let r = x.row(i);
            let signal = r[0] * 2.0 + r[1] * r[1] - (r[2] * 1.5).sin() * 3.0 + r[3] * r[4];
            let noise: f64 = rng.random_range(-0.5..0.5) * (1.0 + r[5].abs());
            signal + noise
        });

        let (x_tr, x_te) = (
            x.slice(s![..1500, ..]).to_owned(),
            x.slice(s![1500.., ..]).to_owned(),
        );
        let (y_tr, y_te) = (
            y.slice(s![..1500]).to_owned(),
            y.slice(s![1500..]).to_owned(),
        );

        let mut model = NGBRegressor::new(200, 0.05);
        model.fit(&x_tr, &y_tr).unwrap();

        let dist = model.pred_dist(&x_te);
        let nll = -dist.logpdf(&y_te).mean().unwrap();
        let mu = dist.mean();
        let rmse = ((&mu - &y_te).mapv(|d| d * d).mean().unwrap()).sqrt();

        println!("seed {seed}: NLL {nll:.15} RMSE {rmse:.15}");
    }
}
