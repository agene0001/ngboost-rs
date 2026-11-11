//! Research probe (research/ngboost): WHERE does β = 1 (Fisher-weighted μ
//! trees) lose on UCI protein? Fits Normal/LogScore with β = 0 and β = 1 on
//! identical 70/10/20 splits (5 seeds) and reports paired test ΔNLL / RMSE
//! by target band and by decile of the β = 0 model's predicted σ.
//!
//!   cargo run --release --features accelerate --example protein_probe
use ndarray::{Array1, Array2};
use ngboost_rs::dist::Normal;
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{LogScore, Scorable};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn load() -> (Array2<f64>, Array1<f64>) {
    let text = std::fs::read_to_string("../ngboost/data/uci/protein.csv").unwrap();
    let rows: Vec<Vec<f64>> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let n = rows.len();
    let x = Array2::from_shape_fn((n, 9), |(i, j)| rows[i][j + 1]);
    let y = Array1::from_shape_fn(n, |i| rows[i][0]);
    (x, y)
}

fn fit(
    beta: f64,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
) -> NGBoost<Normal, LogScore, HistogramLearner> {
    let mut m = NGBoost::<Normal, LogScore, HistogramLearner>::with_full_options(
        2000,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        Some(50),
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    );
    m.auto_calibrate_scale = false;
    m.fisher_weight_power = Some(beta);
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .unwrap();
    m
}

fn main() {
    let (x, y) = load();
    let bands: [(f64, f64, &str); 4] = [
        (0.0, 1.0, "y<1"),
        (1.0, 4.0, "1-4"),
        (4.0, 10.0, "4-10"),
        (10.0, 1e9, ">=10"),
    ];
    // accumulators: [band] -> (n, Σnll0, Σnll1, Σse0, Σse1)
    let mut by_band = vec![(0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64); bands.len()];
    let mut by_dec = vec![(0usize, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); 10];
    let mut weight_share = vec![0.0f64; bands.len()];
    for seed in 0..5u64 {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5e11_7000);
        let mut idx: Vec<usize> = (0..x.nrows()).collect();
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }
        idx.truncate(12000);
        let (a, b) = (8400, 9600);
        let take = |r: &[usize]| (x.select(ndarray::Axis(0), r), y.select(ndarray::Axis(0), r));
        let (xtr, ytr) = take(&idx[..a]);
        let (xva, yva) = take(&idx[a..b]);
        let (xte, yte) = take(&idx[b..]);
        let m0 = fit(0.0, seed, &xtr, &ytr, &xva, &yva);
        let m1 = fit(1.0, seed, &xtr, &ytr, &xva, &yva);
        let (d0, d1) = (m0.pred_dist(&xte), m1.pred_dist(&xte));
        let (s0, s1) = (
            Scorable::<LogScore>::score(&d0, &yte),
            Scorable::<LogScore>::score(&d1, &yte),
        );
        // σ̂ deciles of the β = 0 model
        let mut order: Vec<usize> = (0..yte.len()).collect();
        order.sort_by(|&i, &j| d0.scale[i].total_cmp(&d0.scale[j]));
        for (rank, &i) in order.iter().enumerate() {
            let dcl = rank * 10 / yte.len();
            let e = &mut by_dec[dcl];
            e.0 += 1;
            e.1 += s0[i];
            e.2 += s1[i];
            e.3 += (d0.loc[i] - yte[i]).powi(2);
            e.4 += (d1.loc[i] - yte[i]).powi(2);
            e.5 += d0.scale[i];
        }
        for i in 0..yte.len() {
            let k = bands
                .iter()
                .position(|&(lo, hi, _)| yte[i] >= lo && yte[i] < hi)
                .unwrap();
            let e = &mut by_band[k];
            e.0 += 1;
            e.1 += s0[i];
            e.2 += s1[i];
            e.3 += (d0.loc[i] - yte[i]).powi(2);
            e.4 += (d1.loc[i] - yte[i]).powi(2);
        }
        // share of the β = 1 μ-tree weight (1/σ̂², at the final β=0 σ̂) by band, on train
        let dtr = m0.pred_dist(&xtr);
        let tot: f64 = dtr.scale.iter().map(|s| 1.0 / (s * s)).sum();
        for i in 0..ytr.len() {
            let k = bands
                .iter()
                .position(|&(lo, hi, _)| ytr[i] >= lo && ytr[i] < hi)
                .unwrap();
            weight_share[k] += 1.0 / (dtr.scale[i] * dtr.scale[i]) / tot / 5.0;
        }
        println!(
            "seed {seed}: iters β0 {} β1 {} | test NLL β0 {:.4} β1 {:.4}",
            m0.base_models.len(),
            m1.base_models.len(),
            s0.mean().unwrap(),
            s1.mean().unwrap()
        );
    }
    let n_all: usize = by_band.iter().map(|e| e.0).sum();
    println!(
        "\nby target band (5 seeds pooled): share of rows | share of β=1 weight | ΔNLL contribution | mean ΔNLL | RMSE β0 → β1"
    );
    for (k, &(_, _, name)) in bands.iter().enumerate() {
        let e = by_band[k];
        let n = e.0 as f64;
        println!(
            "  {name:>5}: rows {:5.1}% | weight {:5.1}% | contrib {:+.4} | mean {:+.4} | RMSE {:.3} → {:.3}",
            100.0 * n / n_all as f64,
            100.0 * weight_share[k],
            (e.2 - e.1) / n_all as f64,
            (e.2 - e.1) / n,
            (e.3 / n).sqrt(),
            (e.4 / n).sqrt()
        );
    }
    println!("\nby decile of β=0 predicted σ: mean σ̂ | mean ΔNLL (β1−β0) | RMSE β0 → β1");
    for (d, e) in by_dec.iter().enumerate() {
        let n = e.0 as f64;
        println!(
            "  d{d}: σ̂ {:6.3} | ΔNLL {:+.4} | RMSE {:.3} → {:.3}",
            e.5 / n,
            (e.2 - e.1) / n,
            (e.3 / n).sqrt(),
            (e.4 / n).sqrt()
        );
    }
}
