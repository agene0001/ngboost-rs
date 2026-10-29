//! Quick A/B: StudentT LogScore fit quality (held-out NLL) across seeds.
//! Run once on old metric (stash) and once on new; compare.
use ndarray::{Array1, Array2};
use ngboost_rs::dist::StudentT;
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::Scorable;
use ngboost_rs::{Distribution, LogScore};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn gen_data(seed: u64, n: usize, p: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, p), |_| rng.random_range(-1.0..1.0));
    // heavy-tailed noise: standard t(df=4) via ratio of normal / sqrt(chi2/df)
    let y = Array1::from_shape_fn(n, |i| {
        let mu = 2.0 * x[[i, 0]] - x[[i, 1]] + 0.5 * x[[i, 2]] * x[[i, 0]];
        let z: f64 = {
            let u1: f64 = rng.random_range(1e-12..1.0);
            let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
            (-2.0 * u1.ln()).sqrt() * u2.cos()
        };
        let chi: f64 = (0..4)
            .map(|_| {
                let u1: f64 = rng.random_range(1e-12..1.0);
                let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                let g = (-2.0 * u1.ln()).sqrt() * u2.cos();
                g * g
            })
            .sum();
        mu + 0.8 * z / (chi / 4.0).sqrt()
    });
    (x, y)
}

fn main() {
    let mut nlls = Vec::new();
    for seed in [11u64, 23, 47] {
        let (x, y) = gen_data(seed, 600, 5);
        let (xtr, xte) = (
            x.slice(ndarray::s![..400, ..]).to_owned(),
            x.slice(ndarray::s![400.., ..]).to_owned(),
        );
        let (ytr, yte) = (
            y.slice(ndarray::s![..400]).to_owned(),
            y.slice(ndarray::s![400..]).to_owned(),
        );
        let mut model = NGBoost::<StudentT, LogScore, HistogramLearner>::new(
            100,
            0.03,
            HistogramLearner::new(3),
        );
        model.fit(&xtr, &ytr).unwrap();
        let dist = model.pred_dist(&xte);
        let nll = Scorable::<LogScore>::score(&dist, &yte).mean().unwrap();
        println!("seed {seed}: held-out NLL = {nll:.6}");
        nlls.push(nll);
    }
    println!(
        "mean NLL = {:.6}",
        nlls.iter().sum::<f64>() / nlls.len() as f64
    );
}
