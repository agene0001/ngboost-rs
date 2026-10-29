//! Bit-exactness A/B harness: prints FNV hashes of fitted model outputs for
//! deterministic (seeded) configs. Run against two builds (e.g. HEAD vs
//! working tree) and diff the output — any semantic training change shows up
//! as a hash mismatch.
use ndarray::{Array1, Array2};
use ngboost_rs::learners::{HistogramLearner, default_base_learner, default_tree_learner};
use ngboost_rs::survival::NGBSurvivalLogNormal;
use ngboost_rs::{Distribution, NGBExactRegressor, NGBHistRegressor};

fn fnv(bits: impl Iterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn hash_params(p: &Array2<f64>) -> u64 {
    fnv(p.iter().map(|v| v.to_bits()))
}

// Deterministic synthetic data (no RNG dependencies).
fn make_data(n: usize, p: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = ((i * 31 + j * 17) as f64 * 0.7315).sin() * 2.0;
        }
    }
    let y = (0..n)
        .map(|i| {
            let r: f64 = (0..p).map(|j| x[[i, j]] * (j as f64 + 1.0) * 0.3).sum();
            r + (i as f64 * 2.399).cos() * 0.5
        })
        .collect();
    (x, Array1::from_vec(y))
}

fn main() {
    let (x, y) = make_data(400, 5);
    let (xt, _) = make_data(97, 5);

    // 1. Histogram learner, default config (stable-X fast path + line search)
    let mut m = NGBHistRegressor::with_seed(60, 0.1, default_base_learner(), 42);
    m.fit(&x, &y).unwrap();
    println!(
        "hist_default {:016x}",
        hash_params(&m.pred_dist(&xt).params())
    );
    println!(
        "hist_default_scalings {:016x}",
        fnv(m.scalings.iter().map(|s| s.to_bits()))
    );

    // 2. Exact tree learner, default config
    let mut m = NGBExactRegressor::with_seed(60, 0.1, default_tree_learner(), 42);
    m.fit(&x, &y).unwrap();
    println!(
        "exact_default {:016x}",
        hash_params(&m.pred_dist(&xt).params())
    );

    // 3. Row + column subsampling (seeded RNG; exercises the non-stable path)
    let mut m = NGBHistRegressor::with_seed(60, 0.1, default_base_learner(), 7);
    m.minibatch_frac = 0.7;
    m.col_sample = 0.6;
    m.fit(&x, &y).unwrap();
    println!(
        "hist_subsampled {:016x}",
        hash_params(&m.pred_dist(&xt).params())
    );

    // 4. Validation + early stopping
    let (xv, yv) = make_data(120, 5);
    let mut m = NGBHistRegressor::with_seed(200, 0.3, default_base_learner(), 42);
    m.early_stopping_rounds = Some(10);
    m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    println!(
        "hist_earlystop {:016x}",
        hash_params(&m.pred_dist(&xt).params())
    );

    // 5. Big learning rate to force line-search scale-up/down branches
    let mut m = NGBHistRegressor::with_seed(40, 0.9, default_base_learner(), 42);
    m.fit(&x, &y).unwrap();
    println!("hist_lr09 {:016x}", hash_params(&m.pred_dist(&xt).params()));
    println!(
        "hist_lr09_scalings {:016x}",
        fnv(m.scalings.iter().map(|s| s.to_bits()))
    );

    // 6. Survival, deterministic config only (ThreadRng is unseedable, but no
    //    subsampling means the RNG is never consulted)
    let time = y.mapv(|v| v.abs() + 0.1);
    let event: Array1<f64> = (0..time.len())
        .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 })
        .collect();
    let mut m = NGBSurvivalLogNormal::lognormal(60, 0.1);
    m.fit(&x, &time, &event).unwrap();
    println!(
        "survival_lognormal {:016x}",
        hash_params(&m.pred_dist(&xt).params())
    );
    println!(
        "survival_scalings {:016x}",
        fnv(m.scalings.iter().map(|s| s.to_bits()))
    );
}
