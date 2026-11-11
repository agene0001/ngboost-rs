//! Research probe (research/ngboost Cycle 11): does a nearly unused mixture
//! component's σ drift during boosting, or does the joint scale
//! recalibration amplify it? Full protein (45 730 rows), NormalMixture3,
//! seed 0 split as in research_bench (70/10/20), with and without the
//! default recalibration; per-component weight / σ quantiles at the end and
//! along the boosting path.
//!
//!   cargo run --release --features accelerate --example mixture_sigma_probe
use ndarray::{Array1, Array2};
use ngboost_rs::dist::NormalMixture3;
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
    (
        Array2::from_shape_fn((n, 9), |(i, j)| rows[i][j + 1]),
        Array1::from_shape_fn(n, |i| rows[i][0]),
    )
}

fn q(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p) as usize]
}

fn report(tag: &str, d: &NormalMixture3) {
    for k in 0..3 {
        let mut w: Vec<f64> = d.weights.column(k).to_vec();
        let mut s: Vec<f64> = d.sigma.column(k).to_vec();
        println!(
            "  {tag} comp {k}: weight median {:.4} (p99 {:.4}) | σ median {:.3} p99 {:.3} max {:.3}",
            q(&mut w, 0.5),
            q(&mut w, 0.99),
            q(&mut s, 0.5),
            q(&mut s, 0.99),
            q(&mut s, 1.0)
        );
    }
}

fn main() {
    // seeds 0..3, default recalibration on; guard on/off via NGBOOST_MIX_GUARD
    let (x, y) = load();
    let guard = std::env::var("NGBOOST_MIX_GUARD").map_or(true, |v| v.trim() != "0");
    for seed in 0..3u64 {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5e11_7000);
        let mut idx: Vec<usize> = (0..x.nrows()).collect();
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }
        let n = idx.len();
        let (a, b) = ((0.7 * n as f64) as usize, (0.8 * n as f64) as usize);
        let take = |r: &[usize]| (x.select(ndarray::Axis(0), r), y.select(ndarray::Axis(0), r));
        let (xtr, ytr) = take(&idx[..a]);
        let (xva, yva) = take(&idx[a..b]);
        let (xte, yte) = take(&idx[b..]);
        let mut m = NGBoost::<NormalMixture3, LogScore, HistogramLearner>::with_full_options(
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
        m.fit_with_validation(&xtr, &ytr, Some(&xva), Some(&yva), None, None)
            .unwrap();
        let big = |d: &NormalMixture3| {
            (0..d.sigma.nrows())
                .filter(|&i| (0..3).any(|k| d.sigma[[i, k]] > 100.0))
                .count()
        };
        let (dtr, dte) = (m.pred_dist(&xtr), m.pred_dist(&xte));
        let mut smax: Vec<f64> = dte.sigma.iter().cloned().collect();
        println!(
            "guard={guard} seed {seed}: {} rounds | test NLL {:.4} | rows with a component σ > 100: train {} / {}, test {} / {} | test σ max {:.3e}",
            m.base_models.len(),
            Scorable::<LogScore>::score(&dte, &yte).mean().unwrap(),
            big(&dtr),
            dtr.sigma.nrows(),
            big(&dte),
            dte.sigma.nrows(),
            q(&mut smax, 1.0)
        );
    }
}
