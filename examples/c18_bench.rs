//! Cycle 18 research harness (see `research/automl-hyperparameter/` and
//! `research/optimization-algorithm/`). NOT part of the public API.
//!
//! Head: `NGBoost<Normal, LogScore, HistogramLearner>` with the 1.2.0
//! defaults unless a `--fit` mode says otherwise. Data, shuffling and the
//! 70/10/20 split are copied from `examples/research_bench.rs` so results
//! pair with the earlier reports.
//!
//! Modes (`--mode`):
//! * `pool` — N random configurations per (dataset, seed); each is scored
//!   the way `hyper_opt` scores a fold (early stopping, recalibration and
//!   score on the same block: `val_reuse`) and on an untouched block
//!   (`val_nested`), plus held-out test.
//! * `search` — sampler comparison (random / TPE as in `hyper_opt` / TPE
//!   with random startup) on the real objective or on a k-NN surrogate
//!   built from a pool file.
//! * `lrcap` — learning rate × (round cap, patience) policies.
//! * `select` — choose among fit variants by one holdout vs 5-fold CV.
//! * `oof` — staged location→scale fit with the scale fitted to in-sample
//!   vs cross-fitted (out-of-fold) location residuals.
//! * `leaf` — `min_samples_leaf` sweep at the default shape.
//! * `leafsplit` — separate `min_samples_leaf` for location and scale trees.
//! * `hpo` — the library tuner end to end (`hyper_opt_with_config`, 5-fold
//!   CV): the 1.2.0 module example's space against `default_search_space`
//!   at an equal trial budget, refit on the training pool, scored on the
//!   untouched test block (Cycle 19; `--head lognormal|poisson|negbin`
//!   since Cycle 20).
//! * `rows` — per-test-row predictions and scores of chosen fit modes
//!   (Cycle 20, O11: where does cross-fitting lose on protein?).
//! * `noise` — the same fits repeated with the training targets perturbed
//!   in the 12th digit (Cycle 20, O12: how much of a small effect is
//!   last-bit noise?).
//! * `binning` — quantile bin edges against 1-D k-means bin edges for the
//!   histogram trees, by pre-coding the features (Cycle 22, O14).
//! * `mixinit` — the mixture head's EM started from equal-count quantile
//!   groups against a 1-D k-means start and the best of many random starts
//!   (Cycle 22, O15).
//! * `xdef` — the shipped default model with the staged fit's
//!   cross-fitting off against on, across heads and pool sizes (Cycle 25,
//!   O16: should cross-fitting be the default?).
//! * `counts` — Poisson and NegativeBinomial on real low-count data
//!   (football goals and three public count sets) and goals-like synthetic
//!   rates: leaf default against leaf 1, and the staged NegativeBinomial
//!   fit with cross-fitted residuals (Cycle 26, O17).
//! * `survmean` — does the survival wrapper's LogNormal head predict
//!   absurd means on a target that is not log-normal in its tails?
//!   (Cycle 20, O9)
//! * `hposel` — choosing the head by the tuner's CV score: one search per
//!   head, the winner at 15 and at 30 trials refitted and scored (Cycle 24,
//!   A10).
//! * `treeleaf` — `min_samples_leaf` sweep for the exact tree learners
//!   (`--learner exact|arena`, Cycle 19).

use ndarray::{Array1, Array2, Axis};
use ngboost_rs::dist::{
    Distribution, DistributionMethods, LogNormal, NegativeBinomial, Normal, NormalFixedMean,
    NormalFixedVar, Poisson,
};
use ngboost_rs::learners::{
    ArenaDecisionTreeLearner, BaseLearner, DecisionTreeLearner, HistogramLearner,
};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{CRPScore, LogScore, Scorable};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;
use tpe::{TpeOptimizer, TpeOptimizerBuilder, parzen_estimator, range};

const DATA_DIR: &str = "../ngboost/data/uci";

// ------------------------------------------------------------------- args

struct Args(HashMap<String, String>);
impl Args {
    fn parse() -> Self {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        assert!(
            argv.len().is_multiple_of(2),
            "arguments must be --key value pairs"
        );
        let mut m = HashMap::new();
        for kv in argv.chunks(2) {
            let k = kv[0].strip_prefix("--").expect("--key expected");
            m.insert(k.to_string(), kv[1].clone());
        }
        Args(m)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn list(&self, k: &str, d: &str) -> Vec<String> {
        self.s(k, d).split(',').map(String::from).collect()
    }
    fn u(&self, k: &str, d: usize) -> usize {
        self.0.get(k).map(|v| v.parse().unwrap()).unwrap_or(d)
    }
}

// ------------------------------------------------- data (= research_bench)

fn load_csv(path: &str, target: &str) -> (Array2<f64>, Array1<f64>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();
    let t = header.iter().position(|h| h == target).expect("target");
    let rows: Vec<Vec<f64>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(|s| s.trim().parse().unwrap()).collect())
        .collect();
    let (n, p) = (rows.len(), header.len() - 1);
    let mut x = Array2::zeros((n, p));
    let mut y = Array1::zeros(n);
    for (i, r) in rows.iter().enumerate() {
        let mut k = 0;
        for (j, &v) in r.iter().enumerate() {
            if j == t {
                y[i] = v;
            } else {
                x[[i, k]] = v;
                k += 1;
            }
        }
    }
    (x, y)
}

struct Gauss<'a>(&'a mut StdRng);
impl Gauss<'_> {
    fn z(&mut self) -> f64 {
        let u1: f64 = self.0.random_range(1e-12..1.0);
        let u2: f64 = self.0.random_range(0.0..std::f64::consts::TAU);
        (-2.0 * u1.ln()).sqrt() * u2.cos()
    }
    fn t(&mut self, df: usize) -> f64 {
        let z = self.z();
        let chi: f64 = (0..df).map(|_| self.z().powi(2)).sum();
        z / (chi / df as f64).sqrt()
    }
}

/// The four Gaussian-family DGPs of `research_bench::synth`, draw for draw.
fn synth(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_0000);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let mut g = Gauss(&mut rng);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        let mu_a = 2.0 * r[0].sin() + r[1] * r[1] - r[2] * r[3];
        let (mu, sigma, noise) = match name {
            "het_a" => {
                let s = (-0.5 + 0.6 * r[4] + 0.4 * (2.0 * r[5]).sin()).exp();
                (mu_a, s, g.z())
            }
            "het_b" => {
                let mu = 3.0 * (2.0 * r[0]).sin() + r[1];
                let s = (-1.0 + 1.0 * r[0]).exp();
                (mu, s, g.z())
            }
            "homo" => (mu_a, 1.0, g.z()),
            "tnoise" => {
                let s = (-0.5 + 0.6 * r[4] + 0.4 * (2.0 * r[5]).sin()).exp();
                (mu_a, s, g.t(4) / 2f64.sqrt())
            }
            // truly log-normal with log-space heteroscedasticity
            "lognorm_het" => {
                let s = (-1.0 + 0.5 * r[4]).exp();
                y[i] = (1.0 + 0.3 * mu_a + s * g.z()).exp();
                continue;
            }
            // truly Gamma, log-linear mean, shape 1.4–7.7 driven by a
            // feature the mean ignores (the DGP of `research_bench`)
            "gamma_het" => {
                let m = (0.5 + 0.3 * mu_a).exp();
                let al = 1.0 + (0.5 + 0.7 * r[4]).exp();
                y[i] = m * g.gamma(al) / al;
                continue;
            }
            _ => panic!("unknown dataset {name}"),
        };
        y[i] = mu + sigma * noise;
    }
    (x, y)
}

impl Gauss<'_> {
    /// Poisson(λ) by Knuth's multiplication method (fine for λ ≲ 500).
    fn poisson(&mut self, lambda: f64) -> f64 {
        let l = (-lambda).exp();
        let (mut k, mut p) = (0u32, 1.0f64);
        loop {
            k += 1;
            p *= self.0.random_range(0.0..1.0f64);
            if p <= l {
                return (k - 1) as f64;
            }
        }
    }
    /// Gamma(shape r ≥ 1, scale 1) by Marsaglia–Tsang.
    fn gamma(&mut self, r: f64) -> f64 {
        let d = r - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.z();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u: f64 = self.0.random_range(1e-300..1.0);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

/// The count DGPs of `research_bench::synth_count`, draw for draw.
fn synth_count(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xc0_0a7);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let mut g = Gauss(&mut rng);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        let nl = (2.0 + 0.8 * (2.0 * r[0]).sin() + 0.25 * r[1] * r[1] - 0.3 * r[2].abs()
            + 0.2 * r[3] * r[4])
            .exp();
        let low = (1.75 + 0.25 * r[0] + 0.15 * (2.0 * r[1]).sin()).exp();
        // goals-like rates (Cycle 26): mean about 0.3–4.5, median about 1.2
        let goals = (0.2 + 0.25 * r[0] - 0.2 * r[1] + 0.1 * r[2] * r[3]).exp();
        let (mu, disp): (f64, Option<f64>) = match name {
            "pois_goals" => (goals, None),
            "nb_goals" => (goals, Some(6.0)),
            "pois_lin" => ((0.8 + 0.6 * r[0] - 0.4 * r[1] + 0.3 * r[2]).exp(), None),
            "pois_nl" => (nl, None),
            "pois_low" => (low, None),
            "nb_over" => (nl, Some(4.0)),
            "nb_low" => (low, Some(10.0)),
            _ => panic!("unknown count dataset {name}"),
        };
        let lam_i = match disp {
            None => mu,
            Some(rr) => mu * g.gamma(rr) / rr,
        };
        y[i] = g.poisson(lam_i);
    }
    (x, y)
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5e11_7000u64);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        idx.swap(i, j);
    }
    idx
}

fn take(x: &Array2<f64>, y: &Array1<f64>, idx: &[usize]) -> (Array2<f64>, Array1<f64>) {
    (x.select(Axis(0), idx), y.select(Axis(0), idx))
}

/// 70/10/20 train (A) / validation (V) / test, as in `research_bench`.
struct Split {
    xa: Array2<f64>,
    ya: Array1<f64>,
    xv: Array2<f64>,
    yv: Array1<f64>,
    xte: Array2<f64>,
    yte: Array1<f64>,
}

fn load_split(ds: &str, seed: u64, a: &Args) -> Split {
    let (x, y) = match ds {
        "kin8nm" => load_csv(&format!("{DATA_DIR}/kin8nm.csv"), "y"),
        "protein" => load_csv(&format!("{DATA_DIR}/protein.csv"), "RMSD"),
        d if d.starts_with("pois_") || d.starts_with("nb_") => {
            synth_count(d, a.u("n-synth", 3000), seed)
        }
        // prepared real count files (Cycle 26); football files carry the
        // season in column 0, which is a split key and not a feature
        d if d.starts_with("fb") || ["doctor", "biochem", "nmes"].contains(&d) => {
            let dir = a.s("data-dir", "../data_counts/prepared");
            let (x, y) = load_csv(&format!("{dir}/{d}.csv"), "y");
            if d.starts_with("fb") {
                let feats: Vec<usize> = (1..x.ncols()).collect();
                (x.select(Axis(1), &feats), y)
            } else {
                (x, y)
            }
        }
        _ => synth(ds, a.u("n-synth", 3000), seed),
    };
    let mut idx = shuffled_indices(x.nrows(), seed);
    idx.truncate(a.u("max-rows", 12000).min(idx.len()));
    let n = idx.len();
    let n_tr = (0.7 * n as f64) as usize;
    let n_va = (0.1 * n as f64) as usize;
    let n_va_used = n_va.min(a.u("val-rows", usize::MAX));
    let (xa, ya) = take(&x, &y, &idx[..n_tr]);
    let (xv, yv) = take(&x, &y, &idx[n_tr..n_tr + n_va_used]);
    let (xte, yte) = take(&x, &y, &idx[n_tr + n_va..]);
    Split {
        xa,
        ya,
        xv,
        yv,
        xte,
        yte,
    }
}

// ---------------------------------------------------------------- metrics

#[derive(Clone, Copy)]
struct Ev {
    nll: f64,
    crps: f64,
    rmse: f64,
    cov90: f64,
    calib: f64,
}
const EV_NAN: Ev = Ev {
    nll: f64::NAN,
    crps: f64::NAN,
    rmse: f64::NAN,
    cov90: f64::NAN,
    calib: f64::NAN,
};

fn evaluate(d: &Normal, y: &Array1<f64>) -> Ev {
    let (lo, hi) = d.interval(0.1);
    let c = (0..y.len())
        .filter(|&i| y[i] >= lo[i] && y[i] <= hi[i])
        .count();
    let mse = (&d.loc - y).mapv(|e| e * e).mean().unwrap();
    let var = d.scale.mapv(|s| s * s).mean().unwrap();
    Ev {
        nll: Scorable::<LogScore>::score(d, y).mean().unwrap(),
        crps: Scorable::<CRPScore>::score(d, y).mean().unwrap(),
        rmse: mse.sqrt(),
        cov90: c as f64 / y.len() as f64,
        calib: (mse / var).sqrt(),
    }
}

// ---------------------------------------------------------- search space

#[derive(Clone, Copy, Debug)]
struct Cfg {
    lr: f64,
    depth: usize,
    msl: usize,
    mb: f64,
    cs: f64,
}
const DEFAULT_CFG: Cfg = Cfg {
    lr: 0.05,
    depth: 3,
    msl: 1,
    mb: 1.0,
    cs: 1.0,
};

/// Raw (sampling-space) ranges in `hyper_opt`'s sorted-name order:
/// col_sample, learning_rate (log), max_depth (int), min_samples_leaf
/// (log int), minibatch_frac.
fn dims() -> [(f64, f64); 5] {
    [
        (0.4, 1.0),
        (0.02f64.ln(), 0.2f64.ln()),
        (2.0, 8.0),
        (1f64.ln(), 200f64.ln()),
        (0.5, 1.0),
    ]
}
fn cfg_from_raw(r: &[f64; 5]) -> Cfg {
    Cfg {
        cs: r[0],
        lr: r[1].exp(),
        depth: r[2].round() as usize,
        msl: (r[3].exp().round() as usize).max(1),
        mb: r[4],
    }
}
fn cfg_unit(c: &Cfg) -> [f64; 5] {
    let raw = [c.cs, c.lr.ln(), c.depth as f64, (c.msl as f64).ln(), c.mb];
    let d = dims();
    std::array::from_fn(|i| (raw[i] - d[i].0) / (d[i].1 - d[i].0))
}
fn random_raw(rng: &mut StdRng) -> [f64; 5] {
    let d = dims();
    std::array::from_fn(|i| rng.random_range(d[i].0..d[i].1))
}
/// Pool configuration `i`: 0 = the research default, 1 = the same shape at
/// lr 0.02, the rest uniform in raw space (i.e. plain random search).
fn pool_cfg(i: usize) -> Cfg {
    match i {
        0 => DEFAULT_CFG,
        1 => Cfg {
            lr: 0.02,
            ..DEFAULT_CFG
        },
        _ => cfg_from_raw(&random_raw(&mut StdRng::seed_from_u64(
            0xC18_0000 + i as u64,
        ))),
    }
}
fn cap_for(lr: f64) -> u32 {
    ((100.0 / lr).ceil() as u32).min(6000)
}
fn patience_for(lr: f64) -> u32 {
    ((2.5 / lr).round() as u32).clamp(50, 200)
}

// ------------------------------------------------------------------ models

type Model<D> = NGBoost<D, LogScore, HistogramLearner>;

fn build<D>(c: &Cfg, n_est: u32, patience: Option<u32>, seed: u64) -> Model<D>
where
    D: Distribution + Scorable<LogScore> + Clone,
{
    NGBoost::with_full_options(
        n_est,
        c.lr,
        HistogramLearner::with_params(c.depth, 255, c.msl, 2),
        true,
        c.mb,
        c.cs,
        false,
        1.0,
        1e-4,
        patience,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    )
}

fn apply_fit_mode(m: &mut Model<Normal>, mode: &str) {
    match mode {
        "default" => {}
        "joint" => m.staged_fit = Some(false),
        "joint_b05" => {
            m.staged_fit = Some(false);
            m.fisher_weight_power = Some(0.5);
        }
        "joint_b1" => {
            m.staged_fit = Some(false);
            m.fisher_weight_power = Some(1.0);
        }
        "staged" => m.staged_fit = Some(true),
        "staged_nocal" => {
            m.staged_fit = Some(true);
            m.auto_calibrate_scale = false;
        }
        // library cross-fitted staged fit (`staged_crossfit_folds`)
        "xfit" => m.staged_crossfit_folds = 5,
        "xfit_nocal" => {
            m.staged_crossfit_folds = 5;
            m.staged_fit = Some(true);
            m.auto_calibrate_scale = false;
        }
        other => panic!("unknown fit mode {other}"),
    }
}

/// One fit scored the way a `hyper_opt` fold is: early stopping,
/// recalibration and the score all on `(xv, yv)`. Returns the model and
/// that score (inf on failure).
#[allow(clippy::too_many_arguments)]
fn fit_scored(
    c: &Cfg,
    n_est: u32,
    patience: u32,
    seed: u64,
    mode: &str,
    xt: &Array2<f64>,
    yt: &Array1<f64>,
    xv: &Array2<f64>,
    yv: &Array1<f64>,
) -> Option<(Model<Normal>, f64)> {
    let mut m = build::<Normal>(c, n_est, Some(patience), seed);
    apply_fit_mode(&mut m, mode);
    m.fit_with_validation(xt, yt, Some(xv), Some(yv), None, None)
        .ok()?;
    let v = m.score(xv, yv);
    v.is_finite().then_some((m, v))
}

fn emit(line: String) {
    let out = std::io::stdout();
    let mut l = out.lock();
    writeln!(l, "{line}").unwrap();
    l.flush().unwrap();
}

// ------------------------------------------------------------- mode: pool

fn mode_pool(a: &Args) {
    let mode = a.s("fit", "default");
    let (cfg0, n_cfg) = (a.u("cfg0", 0), a.u("n-cfg", 100));
    println!(
        "dataset,seed,cfg,lr,depth,msl,mb,cs,n_iter,secs,val_reuse,val_nested,test_nll,test_crps,test_rmse,cov90,calib,test2_nll"
    );
    for ds in a.list("datasets", "het_a") {
        for seed in a.u("seed0", 0)..a.u("seed0", 0) + a.u("seeds", 1) {
            let seed = seed as u64;
            let s = load_split(&ds, seed, a);
            // nested split: the last |V| rows of A become the inner
            // early-stopping/calibration block, V stays untouched
            let (na, nv) = (s.ya.len(), s.yv.len());
            let inner: Vec<usize> = (0..na - nv).collect();
            let hold: Vec<usize> = (na - nv..na).collect();
            let (xa2, ya2) = take(&s.xa, &s.ya, &inner);
            let (xv2, yv2) = take(&s.xa, &s.ya, &hold);
            (cfg0..cfg0 + n_cfg).into_par_iter().for_each(|i| {
                let c = pool_cfg(i);
                let (cap, pat) = (cap_for(c.lr), patience_for(c.lr));
                let t = Instant::now();
                let f1 = fit_scored(&c, cap, pat, seed, &mode, &s.xa, &s.ya, &s.xv, &s.yv);
                let secs = t.elapsed().as_secs_f64();
                let (n_iter, reuse, ev) = match &f1 {
                    Some((m, v)) => (m.base_models.len(), *v, evaluate(&m.pred_dist(&s.xte), &s.yte)),
                    None => (0, f64::NAN, EV_NAN),
                };
                let f2 = fit_scored(&c, cap, pat, seed, &mode, &xa2, &ya2, &xv2, &yv2);
                let (nested, test2) = match &f2 {
                    Some((m, _)) => (
                        m.score(&s.xv, &s.yv),
                        evaluate(&m.pred_dist(&s.xte), &s.yte).nll,
                    ),
                    None => (f64::NAN, f64::NAN),
                };
                emit(format!(
                    "{ds},{seed},{i},{:.5},{},{},{:.4},{:.4},{n_iter},{secs:.2},{reuse:.5},{nested:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{test2:.5}",
                    c.lr, c.depth, c.msl, c.mb, c.cs, ev.nll, ev.crps, ev.rmse, ev.cov90, ev.calib
                ));
            });
        }
    }
}

// ----------------------------------------------------------- mode: search

/// k-NN (k = 4, inverse squared distance) surrogate of (val score, test
/// NLL) over the unit-scaled search space, from one dataset's pool rows.
struct Surrogate(Vec<([f64; 5], f64, f64)>);
impl Surrogate {
    fn load(path: &str, ds: &str, seed: u64) -> Self {
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines();
        let h: Vec<&str> = lines.next().unwrap().split(',').collect();
        let col = |n: &str| h.iter().position(|c| *c == n).unwrap();
        let (cd, cseed) = (col("dataset"), col("seed"));
        let cols = ["lr", "depth", "msl", "mb", "cs", "val_reuse", "test_nll"].map(col);
        let mut pts = Vec::new();
        for l in lines {
            let f: Vec<&str> = l.split(',').collect();
            if f[cd] != ds || f[cseed].parse::<u64>().unwrap() != seed {
                continue;
            }
            let v: Vec<f64> = cols.iter().map(|&c| f[c].parse().unwrap()).collect();
            if !v[5].is_finite() {
                continue;
            }
            let c = Cfg {
                lr: v[0],
                depth: v[1] as usize,
                msl: v[2] as usize,
                mb: v[3],
                cs: v[4],
            };
            pts.push((cfg_unit(&c), v[5], v[6]));
        }
        assert!(pts.len() >= 20, "too few pool rows for {ds} seed {seed}");
        Surrogate(pts)
    }
    fn eval(&self, c: &Cfg) -> (f64, f64) {
        let u = cfg_unit(c);
        let mut d: Vec<(f64, f64, f64)> = self
            .0
            .iter()
            .map(|(p, v, t)| {
                let d2: f64 = (0..5).map(|i| (p[i] - u[i]).powi(2)).sum();
                (d2, *v, *t)
            })
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut w, mut sv, mut st) = (0.0, 0.0, 0.0);
        for (d2, v, t) in d.iter().take(4) {
            let wi = 1.0 / (d2 + 1e-6);
            w += wi;
            sv += wi * v;
            st += wi * t;
        }
        (sv / w, st / w)
    }
}

/// (startup random trials, gamma); `None` = pure random search.
fn parse_sampler(s: &str) -> Option<(usize, f64)> {
    match s {
        "random" => None,
        "tpe" => Some((0, 0.1)),
        "tpe10" => Some((10, 0.1)),
        "tpe10g25" => Some((10, 0.25)),
        "tpe20" => Some((20, 0.1)),
        other => panic!("unknown sampler {other}"),
    }
}

fn mode_search(a: &Args) {
    let mode = a.s("fit", "default");
    let trials = a.u("trials", 40);
    let objective = a.s("objective", "real");
    println!(
        "dataset,objective,sampler,hp_seed,trial,lr,depth,msl,mb,cs,score,test_nll,best_score,test_at_best"
    );
    for ds in a.list("datasets", "het_a") {
        let seed = a.u("seed0", 0) as u64;
        let surr = objective
            .strip_prefix("surr:")
            .map(|p| Surrogate::load(p, &ds, seed));
        let split = surr.is_none().then(|| load_split(&ds, seed, a));
        let eval = |c: &Cfg| -> (f64, f64) {
            if let Some(s) = &surr {
                return s.eval(c);
            }
            let s = split.as_ref().unwrap();
            match fit_scored(
                c,
                cap_for(c.lr),
                patience_for(c.lr),
                seed,
                &mode,
                &s.xa,
                &s.ya,
                &s.xv,
                &s.yv,
            ) {
                Some((m, v)) => (v, evaluate(&m.pred_dist(&s.xte), &s.yte).nll),
                None => (f64::INFINITY, f64::NAN),
            }
        };
        let jobs: Vec<(String, usize)> = a
            .list("samplers", "random,tpe,tpe10")
            .into_iter()
            .flat_map(|s| (0..a.u("hp-seeds", 8)).map(move |h| (s.clone(), h)))
            .collect();
        let obj_name = if surr.is_some() { "surrogate" } else { "real" };
        jobs.into_par_iter().for_each(|(sampler, h)| {
            let spec = parse_sampler(&sampler);
            let mut rng = StdRng::seed_from_u64(7919 * h as u64 + 17);
            let d = dims();
            let mut opts: Vec<TpeOptimizer> = d
                .iter()
                .map(|&(lo, hi)| {
                    TpeOptimizerBuilder::new()
                        .gamma(spec.map(|s| s.1).unwrap_or(0.1))
                        .build(parzen_estimator(), range(lo, hi).unwrap())
                        .unwrap()
                })
                .collect();
            let (mut best, mut test_at_best) = (f64::INFINITY, f64::NAN);
            let mut rows = Vec::with_capacity(trials);
            for t in 0..trials {
                let raw: [f64; 5] = match spec {
                    Some((startup, _)) if t >= startup => {
                        std::array::from_fn(|i| opts[i].ask(&mut rng).unwrap())
                    }
                    _ => random_raw(&mut rng),
                };
                let c = cfg_from_raw(&raw);
                let (score, test) = eval(&c);
                if spec.is_some() {
                    for (o, r) in opts.iter_mut().zip(raw) {
                        o.tell(r, score).unwrap();
                    }
                }
                if score < best {
                    best = score;
                    test_at_best = test;
                }
                rows.push(format!(
                    "{ds},{obj_name},{sampler},{h},{t},{:.5},{},{},{:.4},{:.4},{score:.5},{test:.5},{best:.5},{test_at_best:.5}",
                    c.lr, c.depth, c.msl, c.mb, c.cs
                ));
            }
            emit(rows.join("\n"));
        });
    }
}

// ------------------------------------------------------------ mode: lrcap

fn mode_lrcap(a: &Args) {
    println!(
        "dataset,seed,fit,lr,policy,n_est,patience,n_iter,secs,val,test_nll,test_crps,test_rmse,cov90"
    );
    let lrs: Vec<f64> = a
        .list("lrs", "0.2,0.1,0.05,0.02,0.01,0.005")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    for ds in a.list("datasets", "het_a") {
        for seed in 0..a.u("seeds", 3) as u64 {
            let s = load_split(&ds, seed, a);
            let mut jobs = Vec::new();
            for fit in a.list("fit", "default,joint") {
                for &lr in &lrs {
                    let scaled = ((60.0 / lr).ceil() as u32).min(12000);
                    let pscaled = ((2.5 / lr).round() as u32).clamp(50, 500);
                    for (name, n_est, pat) in [
                        ("cap500_p50", 500, 50),
                        ("cap1500_p50", 1500, 50),
                        ("capS_p50", scaled, 50),
                        ("capS_pS", scaled, pscaled),
                    ] {
                        jobs.push((fit.clone(), lr, name, n_est, pat));
                    }
                }
            }
            jobs.into_par_iter().for_each(|(fit, lr, name, n_est, pat)| {
                let c = Cfg { lr, ..DEFAULT_CFG };
                let t = Instant::now();
                let r = fit_scored(&c, n_est, pat, seed, &fit, &s.xa, &s.ya, &s.xv, &s.yv);
                let secs = t.elapsed().as_secs_f64();
                let (n_iter, v, ev) = match &r {
                    Some((m, v)) => (m.base_models.len(), *v, evaluate(&m.pred_dist(&s.xte), &s.yte)),
                    None => (0, f64::NAN, EV_NAN),
                };
                emit(format!(
                    "{ds},{seed},{fit},{lr},{name},{n_est},{pat},{n_iter},{secs:.2},{v:.5},{:.5},{:.5},{:.5},{:.4}",
                    ev.nll, ev.crps, ev.rmse, ev.cov90
                ));
            });
        }
    }
}

// ----------------------------------------------------------- mode: select

fn mode_select(a: &Args) {
    println!("dataset,seed,cand,holdout,cv5,test_nll,test_rmse,cov90");
    let cands = a.list("cands", "joint,joint_b05,joint_b1,staged");
    let c = DEFAULT_CFG;
    let (cap, pat) = (cap_for(c.lr), patience_for(c.lr));
    for ds in a.list("datasets", "het_a") {
        (0..a.u("seeds", 10) as u64)
            .into_par_iter()
            .for_each(|seed| {
                let s = load_split(&ds, seed, a);
                // train-all = A ∪ V (already shuffled), 5 contiguous folds as in
                // hyper_opt's KFold
                let xall = ndarray::concatenate(Axis(0), &[s.xa.view(), s.xv.view()]).unwrap();
                let yall = ndarray::concatenate(Axis(0), &[s.ya.view(), s.yv.view()]).unwrap();
                let n = yall.len();
                for cand in &cands {
                    let hold = fit_scored(&c, cap, pat, seed, cand, &s.xa, &s.ya, &s.xv, &s.yv);
                    let (hv, ev) = match &hold {
                        Some((m, v)) => (*v, evaluate(&m.pred_dist(&s.xte), &s.yte)),
                        None => (f64::NAN, EV_NAN),
                    };
                    let cv: f64 = (0..5)
                        .into_par_iter()
                        .map(|f| {
                            let (lo, hi) = (f * n / 5, (f + 1) * n / 5);
                            let tr: Vec<usize> = (0..lo).chain(hi..n).collect();
                            let te: Vec<usize> = (lo..hi).collect();
                            let (xt, yt) = take(&xall, &yall, &tr);
                            let (xf, yf) = take(&xall, &yall, &te);
                            fit_scored(&c, cap, pat, seed, cand, &xt, &yt, &xf, &yf)
                                .map(|r| r.1)
                                .unwrap_or(f64::NAN)
                        })
                        .sum::<f64>()
                        / 5.0;
                    emit(format!(
                        "{ds},{seed},{cand},{hv:.5},{cv:.5},{:.5},{:.5},{:.4}",
                        ev.nll, ev.rmse, ev.cov90
                    ));
                }
            });
    }
}

// -------------------------------------------------------------- mode: oof

/// Out-of-fold location predictions: K contiguous folds, each fitted for a
/// fixed `rounds` (the round count the full-data fit kept), no early stop.
fn oof_mean(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: Option<&Array1<f64>>,
    c: &Cfg,
    rounds: usize,
    seed: u64,
    k: usize,
) -> Array1<f64> {
    let n = y.len();
    let parts: Vec<(usize, Array1<f64>)> = (0..k)
        .into_par_iter()
        .map(|f| {
            let (lo, hi) = (f * n / k, (f + 1) * n / k);
            let tr: Vec<usize> = (0..lo).chain(hi..n).collect();
            let te: Vec<usize> = (lo..hi).collect();
            let (xt, yt) = take(x, y, &tr);
            let wt = w.map(|w| w.select(Axis(0), &tr));
            let mut m = build::<NormalFixedVar>(c, rounds.max(1) as u32, None, seed);
            m.fit_with_validation(&xt, &yt, None, None, wt.as_ref(), None)
                .expect("fold fit failed");
            (lo, m.predict(&x.select(Axis(0), &te)))
        })
        .collect();
    let mut out = Array1::zeros(n);
    for (lo, p) in parts {
        for (j, v) in p.iter().enumerate() {
            out[lo + j] = *v;
        }
    }
    out
}

/// Location → scale → Fisher-weighted location → scale, by separate
/// `NormalFixedVar` / `NormalFixedMean` fits (the recipe `fit_staged`
/// folds into one model). `oof`: the scale stages see cross-fitted
/// location residuals instead of in-sample ones. `cal`: the default
/// validation recalibration on the final scale fit.
/// `c` configures the location fits, `c_scale` the scale fits (same
/// learning rate, so the round cap and patience are shared).
fn manual_staged(
    s: &Split,
    c: &Cfg,
    c_scale: &Cfg,
    seed: u64,
    oof: bool,
    cal: bool,
    k: usize,
) -> (Normal, Normal, usize, f64) {
    let (cap, pat) = (cap_for(c.lr), Some(patience_for(c.lr)));
    let fit_mu = |w: Option<(&Array1<f64>, &Array1<f64>)>| {
        let mut m = build::<NormalFixedVar>(c, cap, pat, seed);
        m.fit_with_validation(
            &s.xa,
            &s.ya,
            Some(&s.xv),
            Some(&s.yv),
            w.map(|w| w.0),
            w.map(|w| w.1),
        )
        .expect("location fit failed");
        m
    };
    let resid = |m: &Model<NormalFixedVar>, w: Option<&Array1<f64>>| -> Array1<f64> {
        if oof {
            &s.ya - &oof_mean(&s.xa, &s.ya, w, c, m.base_models.len(), seed, k)
        } else {
            &s.ya - &m.predict(&s.xa)
        }
    };
    let fit_sg = |m: &Model<NormalFixedVar>, r: &Array1<f64>, cal: bool| {
        let mut sg = build::<NormalFixedMean>(c_scale, cap, pat, seed);
        sg.auto_calibrate_scale = cal;
        sg.fit_with_validation(
            &s.xa,
            r,
            Some(&s.xv),
            Some(&(&s.yv - &m.predict(&s.xv))),
            None,
            None,
        )
        .expect("scale fit failed");
        sg
    };
    let mu1 = fit_mu(None);
    let sg1 = fit_sg(&mu1, &resid(&mu1, None), false);
    let wfn = |x: &Array2<f64>, norm: f64| sg1.pred_dist(x).scale.mapv(|v| 1.0 / (v * v)) / norm;
    let norm = wfn(&s.xa, 1.0).mean().unwrap();
    let (wa, wv) = (wfn(&s.xa, norm), wfn(&s.xv, norm));
    let mu2 = fit_mu(Some((&wa, &wv)));
    let r2 = resid(&mu2, Some(&wa));
    // fraction of residual variance the in-sample fit hides (diagnostic)
    let shrink = (&s.ya - &mu2.predict(&s.xa))
        .mapv(|e| e * e)
        .mean()
        .unwrap()
        / r2.mapv(|e| e * e).mean().unwrap();
    let sg2 = fit_sg(&mu2, &r2, cal);
    let mk = |x: &Array2<f64>| {
        let mut p = Array2::zeros((x.nrows(), 2));
        p.column_mut(0).assign(&mu2.predict(x));
        p.column_mut(1)
            .assign(&sg2.pred_dist(x).scale.mapv(f64::ln));
        Normal::from_params(&p)
    };
    (
        mk(&s.xte),
        mk(&s.xv),
        mu2.base_models.len() + sg2.base_models.len(),
        shrink,
    )
}

fn mode_oof(a: &Args) {
    println!("dataset,seed,variant,n_iter,secs,nll,crps,rmse,cov90,calib,val_nll,extra");
    let variants = a.list(
        "variants",
        "lib_staged,lib_staged_nocal,ts_in,ts_in_cal,ts_oof,ts_oof_cal",
    );
    // `--leaf`: min_samples_leaf of every tree (interaction with O2)
    let c = Cfg {
        msl: a.u("leaf", 1),
        ..DEFAULT_CFG
    };
    let k = a.u("k", 5);
    for ds in a.list("datasets", "het_a") {
        let jobs: Vec<(u64, String)> = (0..a.u("seeds", 10) as u64)
            .flat_map(|s| variants.iter().map(move |v| (s, v.clone())))
            .collect();
        jobs.into_par_iter().for_each(|(seed, v)| {
            let s = load_split(&ds, seed, a);
            let t = Instant::now();
            let (d, dv, n_iter, extra) = match v.as_str() {
                "lib_staged" | "lib_staged_nocal" | "lib_joint" | "lib_default" | "lib_xfit"
                | "lib_xfit_nocal" => {
                    // lib_default: whatever 1.2.0 does by itself (joint and
                    // uncalibrated below 40 validation rows)
                    let mode = match v.as_str() {
                        "lib_staged" => "staged",
                        "lib_staged_nocal" => "staged_nocal",
                        "lib_default" => "default",
                        "lib_xfit" => "xfit",
                        "lib_xfit_nocal" => "xfit_nocal",
                        _ => "joint",
                    };
                    let (m, _) = fit_scored(
                        &c,
                        cap_for(c.lr),
                        patience_for(c.lr),
                        seed,
                        mode,
                        &s.xa,
                        &s.ya,
                        &s.xv,
                        &s.yv,
                    )
                    .expect("fit failed");
                    (
                        m.pred_dist(&s.xte),
                        m.pred_dist(&s.xv),
                        m.base_models.len(),
                        f64::NAN,
                    )
                }
                "ts_in" => manual_staged(&s, &c, &c, seed, false, false, k),
                "ts_in_cal" => manual_staged(&s, &c, &c, seed, false, true, k),
                "ts_oof" => manual_staged(&s, &c, &c, seed, true, false, k),
                "ts_oof_cal" => manual_staged(&s, &c, &c, seed, true, true, k),
                other => panic!("unknown variant {other}"),
            };
            let secs = t.elapsed().as_secs_f64();
            let ev = evaluate(&d, &s.yte);
            let vn = Scorable::<LogScore>::score(&dv, &s.yv).mean().unwrap();
            emit(format!(
                "{ds},{seed},{v},{n_iter},{secs:.2},{:.5},{:.5},{:.5},{:.4},{:.4},{vn:.5},{extra:.4}",
                ev.nll, ev.crps, ev.rmse, ev.cov90, ev.calib
            ));
        });
    }
}

// ------------------------------------------------------------- mode: leaf

/// `min_samples_leaf` sweep at the default shape, per fit mode.
fn mode_leaf(a: &Args) {
    println!("dataset,seed,fit,msl,depth,n_iter,secs,val,test_nll,test_crps,test_rmse,cov90,calib");
    let leaves: Vec<usize> = a
        .list("leaves", "1,5,10,20,50,100,200")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let depth = a.u("depth", 3);
    let weighted = match a.s("weights", "none").as_str() {
        "none" => false,
        "tri" => true,
        other => panic!("unknown weights {other}"),
    };
    for ds in a.list("datasets", "het_a") {
        let mut jobs = Vec::new();
        for seed in 0..a.u("seeds", 10) as u64 {
            for fit in a.list("fit", "default,joint") {
                for &msl in &leaves {
                    jobs.push((seed, fit.clone(), msl));
                }
            }
        }
        jobs.into_par_iter().for_each(|(seed, fit, msl)| {
            let s = load_split(&ds, seed, a);
            let c = Cfg {
                msl,
                depth,
                ..DEFAULT_CFG
            };
            let t = Instant::now();
            // `--weights tri`: every training and validation row gets a
            // sample weight of 0.25, 1 or 4 (equal odds, independent of the
            // data); the test block is scored unweighted
            let r = if weighted {
                let mut g = StdRng::seed_from_u64(seed ^ 0x77_e167);
                let mut draw = |n: usize| {
                    Array1::from_shape_fn(n, |_| [0.25, 1.0, 4.0][g.random_range(0..3)])
                };
                let (w, wv) = (draw(s.ya.len()), draw(s.yv.len()));
                let mut m = build::<Normal>(&c, cap_for(c.lr), Some(patience_for(c.lr)), seed);
                apply_fit_mode(&mut m, &fit);
                m.fit_with_validation(&s.xa, &s.ya, Some(&s.xv), Some(&s.yv), Some(&w), Some(&wv))
                    .ok()
                    .map(|_| {
                        let v = m.score(&s.xv, &s.yv);
                        (m, v)
                    })
            } else {
                fit_scored(
                    &c,
                    cap_for(c.lr),
                    patience_for(c.lr),
                    seed,
                    &fit,
                    &s.xa,
                    &s.ya,
                    &s.xv,
                    &s.yv,
                )
            };
            let secs = t.elapsed().as_secs_f64();
            let (n_iter, v, ev) = match &r {
                Some((m, v)) => (m.base_models.len(), *v, evaluate(&m.pred_dist(&s.xte), &s.yte)),
                None => (0, f64::NAN, EV_NAN),
            };
            let fit = if weighted { format!("{fit}_w") } else { fit };
            emit(format!(
                "{ds},{seed},{fit},{msl},{depth},{n_iter},{secs:.2},{v:.5},{:.5},{:.5},{:.5},{:.4},{:.4}",
                ev.nll, ev.crps, ev.rmse, ev.cov90, ev.calib
            ));
        });
    }
}

// -------------------------------------------------------- mode: leafsplit

/// Which trees want the larger leaves? The manual staged recipe with
/// separate `min_samples_leaf` for the location and the scale fits
/// (in-sample residuals, default recalibration): a 2 × 2 factorial.
fn mode_leafsplit(a: &Args) {
    println!("dataset,seed,variant,n_iter,secs,nll,crps,rmse,cov90,calib,val_nll,extra");
    let big = a.u("big", 50);
    for ds in a.list("datasets", "het_a") {
        let jobs: Vec<(u64, usize, usize)> = (0..a.u("seeds", 10) as u64)
            .flat_map(|s| [(1, 1), (big, 1), (1, big), (big, big)].map(move |(m, g)| (s, m, g)))
            .collect();
        jobs.into_par_iter().for_each(|(seed, msl_mu, msl_sg)| {
            let s = load_split(&ds, seed, a);
            let (cm, cg) = (
                Cfg {
                    msl: msl_mu,
                    ..DEFAULT_CFG
                },
                Cfg {
                    msl: msl_sg,
                    ..DEFAULT_CFG
                },
            );
            let t = Instant::now();
            let (d, dv, n_iter, extra) = manual_staged(&s, &cm, &cg, seed, false, true, 5);
            let secs = t.elapsed().as_secs_f64();
            let ev = evaluate(&d, &s.yte);
            let vn = Scorable::<LogScore>::score(&dv, &s.yv).mean().unwrap();
            emit(format!(
                "{ds},{seed},mu{msl_mu}_sg{msl_sg},{n_iter},{secs:.2},{:.5},{:.5},{:.5},{:.4},{:.4},{vn:.5},{extra:.4}",
                ev.nll, ev.crps, ev.rmse, ev.cov90, ev.calib
            ));
        });
    }
}

// -------------------------------------------------------------- mode: hpo

/// End-to-end tuner comparison. Arms:
/// * `untuned` — `default_histogram_model` with no parameters;
/// * `old` — the 1.2.0 module example: learning rate 0.005–0.1 (log), depth
///   2–8, `n_estimators` 200–1500; leaf 1 and a fold patience of 50, as the
///   1.2.0 library behaved;
/// * `oldspace` — the same space on the 1.3.0 defaults (leaf 20, patience
///   from the learning rate): separates the space from the default changes;
/// * `new` — `default_search_space` with `default_histogram_model`.
///
/// Every arm gets the same trials, folds and seeds, and the winner is refit
/// the same way: the arm's builder on the whole training pool (train +
/// validation blocks of the split), early stopping on the library's own
/// 10 % holdout. The test block is only scored.
fn mode_hpo(a: &Args) {
    /// Scores every head can report: mean log score and RMSE of the mean.
    fn plain<D>(m: &Model<D>, s: &Split) -> Ev
    where
        D: Distribution + Scorable<LogScore> + Clone,
    {
        let p = m.predict(&s.xte);
        Ev {
            nll: m.score(&s.xte, &s.yte),
            rmse: (&p - &s.yte).mapv(|e| e * e).mean().unwrap().sqrt(),
            ..EV_NAN
        }
    }
    match a.s("head", "normal").as_str() {
        "normal" => hpo_head::<Normal>(a, |m, s| evaluate(&m.pred_dist(&s.xte), &s.yte)),
        "lognormal" => hpo_head::<LogNormal>(a, plain),
        "poisson" => hpo_head::<Poisson>(a, plain),
        "negbin" => hpo_head::<NegativeBinomial>(a, plain),
        other => panic!("unknown head {other}"),
    }
}

#[allow(clippy::type_complexity)]
fn hpo_head<D>(a: &Args, ev_fn: fn(&Model<D>, &Split) -> Ev)
where
    D: Distribution + DistributionMethods + Scorable<LogScore> + Clone + 'static,
{
    use ngboost_rs::hyper_opt::{
        HyperOptConfig, cv_early_stopping_rounds, default_histogram_model, default_search_space,
        hyper_opt_with_config,
    };
    use serde_json::{Value, json};
    use std::collections::HashMap;
    type P = HashMap<String, Value>;

    println!(
        "dataset,seed,arm,trials,cv_score,opt_rounds,lr,depth,msl,cs,n_est,search_secs,n_iter,test_nll,test_crps,test_rmse,cov90,calib,max_bins,xfit"
    );
    let trials = a.u("trials", 30) as u32;
    let folds = a.u("folds", 5);
    fn space4() -> P {
        let mut hp = HashMap::new();
        hp.insert(
            "learning_rate".to_string(),
            json!({"low": 0.02, "high": 0.2, "log": true}),
        );
        hp.insert(
            "max_depth".to_string(),
            json!({"low": 2, "high": 8, "type": "int"}),
        );
        hp.insert(
            "min_samples_leaf".to_string(),
            json!({"low": 5, "high": 200, "type": "int", "log": true}),
        );
        hp.insert("col_sample".to_string(), json!({"low": 0.6, "high": 1.0}));
        hp
    }
    fn old_space() -> P {
        let mut hp = HashMap::new();
        hp.insert(
            "learning_rate".to_string(),
            json!({"low": 0.005, "high": 0.1, "log": true}),
        );
        hp.insert(
            "max_depth".to_string(),
            json!({"low": 2, "high": 8, "type": "int"}),
        );
        hp.insert(
            "n_estimators".to_string(),
            json!({"low": 200, "high": 1500, "type": "int"}),
        );
        hp
    }
    fn old_builder<D>(p: &P, leaf: usize, fixed_patience: bool) -> Model<D>
    where
        D: Distribution + Scorable<LogScore> + Clone,
    {
        let lr = p
            .get("learning_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.01);
        let depth = p.get("max_depth").and_then(|v| v.as_i64()).unwrap_or(3) as usize;
        let n_est = p
            .get("n_estimators")
            .and_then(|v| v.as_i64())
            .unwrap_or(500) as u32;
        let mut m = NGBoost::<D, LogScore, HistogramLearner>::new(
            n_est,
            lr,
            HistogramLearner::with_params(depth, 255, leaf, 2),
        );
        m.early_stopping_rounds = Some(if fixed_patience {
            50
        } else {
            cv_early_stopping_rounds(lr)
        });
        m
    }
    for ds in a.list("datasets", "het_a") {
        for seed in a.u("seed0", 0)..a.u("seed0", 0) + a.u("seeds", 3) {
            let seed = seed as u64;
            let s = load_split(&ds, seed, a);
            // training pool = train + validation blocks (already shuffled)
            let xp = ndarray::concatenate(Axis(0), &[s.xa.view(), s.xv.view()]).unwrap();
            let yp = ndarray::concatenate(Axis(0), &[s.ya.view(), s.yv.view()]).unwrap();
            for arm in a.list("arms", "untuned,old,oldspace,new") {
                let cfg = HyperOptConfig {
                    n_trials: trials,
                    n_folds: folds,
                    seed,
                    verbose: false,
                    ..Default::default()
                };
                let t = Instant::now();
                let build: Box<dyn Fn(&P) -> Model<D> + Sync> = match arm.as_str() {
                    "untuned" | "new" | "newbins" | "shipped" | "xfit" => {
                        Box::new(default_histogram_model::<D, LogScore>)
                    }
                    "old" => Box::new(|p: &P| old_builder(p, 1, true)),
                    "oldspace" => Box::new(|p: &P| old_builder(p, 20, false)),
                    other => panic!("unknown arm {other}"),
                };
                let (best, cv, opt_rounds, n_tr): (P, f64, usize, usize) = match arm.as_str() {
                    "untuned" => (HashMap::new(), f64::NAN, 0, 0),
                    _ => {
                        // `new` is the four-key space 1.3.0 was measured with
                        // in A6/A7, spelled out so that those files stay
                        // reproducible whatever `default_search_space`
                        // returns; `newbins` adds `max_bins` (Cycle 23, A8)
                        let space = match arm.as_str() {
                            // whatever the library ships today
                            "shipped" => default_search_space(),
                            // the shipped space plus cross-fitting as a
                            // two-way choice (Cycle 24, A9)
                            "xfit" => ngboost_rs::hyper_opt::default_search_space_with_crossfit(),
                            "new" => space4(),
                            "newbins" => {
                                let mut hp = space4();
                                hp.insert(
                                    "max_bins".to_string(),
                                    json!({"low": 8, "high": 255, "type": "int", "log": true}),
                                );
                                hp
                            }
                            _ => old_space(),
                        };
                        match hyper_opt_with_config::<D, LogScore, HistogramLearner, _>(
                            &xp,
                            &yp,
                            &space,
                            cfg,
                            |p| build(p),
                        ) {
                            Ok(r) => (r.best_params, r.best_score, r.opt_rounds, r.trials.len()),
                            Err(e) => {
                                eprintln!("{ds} {seed} {arm}: search failed: {e}");
                                (HashMap::new(), f64::NAN, 0, 0)
                            }
                        }
                    }
                };
                let search_secs = t.elapsed().as_secs_f64();
                let mut m = build(&best);
                m.set_random_state(seed);
                let g = |k: &str| best.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
                let (n_iter, ev) = match m.fit(&xp, &yp) {
                    Ok(()) => (m.base_models.len(), ev_fn(&m, &s)),
                    Err(e) => {
                        eprintln!("{ds} {seed} {arm}: refit failed: {e}");
                        (0, EV_NAN)
                    }
                };
                emit(format!(
                    "{ds},{seed},{arm},{n_tr},{cv:.5},{opt_rounds},{:.5},{},{},{:.3},{},{search_secs:.1},{n_iter},{:.5},{:.5},{:.5},{:.4},{:.4},{},{}",
                    g("learning_rate"),
                    g("max_depth"),
                    g("min_samples_leaf"),
                    g("col_sample"),
                    m.n_estimators,
                    ev.nll,
                    ev.crps,
                    ev.rmse,
                    ev.cov90,
                    ev.calib,
                    g("max_bins"),
                    g("staged_crossfit_folds")
                ));
            }
        }
    }
}

// ------------------------------------------------------------- mode: rows

/// Per-test-row predictions of each `--fit` mode (leaf `--leaf`, default
/// 20): one line per (dataset, seed, fit, test row).
fn mode_rows(a: &Args) {
    println!("dataset,seed,fit,row,y,mu,sigma,nll,n_iter");
    let c = Cfg {
        msl: a.u("leaf", 20),
        ..DEFAULT_CFG
    };
    for ds in a.list("datasets", "protein") {
        for seed in 0..a.u("seeds", 10) as u64 {
            let s = load_split(&ds, seed, a);
            for fit in a.list("fit", "default,xfit") {
                let Some((m, _)) = fit_scored(
                    &c,
                    cap_for(c.lr),
                    patience_for(c.lr),
                    seed,
                    &fit,
                    &s.xa,
                    &s.ya,
                    &s.xv,
                    &s.yv,
                ) else {
                    eprintln!("{ds} {seed} {fit}: fit failed");
                    continue;
                };
                let d = m.pred_dist(&s.xte);
                let nll = Scorable::<LogScore>::score(&d, &s.yte);
                let k = m.base_models.len();
                for i in 0..s.yte.len() {
                    emit(format!(
                        "{ds},{seed},{fit},{i},{:.6},{:.6},{:.6},{:.6},{k}",
                        s.yte[i], d.loc[i], d.scale[i], nll[i]
                    ));
                }
            }
        }
    }
}

// ------------------------------------------------------------ mode: noise

/// Last-bit sensitivity. Replicate 0 is the plain fit; replicate r > 0
/// multiplies every *training* target by 1 ± `--jitter` (default 1e-12, a
/// seeded random sign per row). Validation and test targets are untouched,
/// so any difference in the test score comes from the fit.
fn mode_noise(a: &Args) {
    println!("dataset,seed,rep,fit,n_iter,val,test_nll,test_rmse");
    let c = Cfg {
        msl: a.u("leaf", 20),
        ..DEFAULT_CFG
    };
    let jitter: f64 = a.s("jitter", "1e-12").parse().unwrap();
    let reps = a.u("reps", 4) as u64;
    for ds in a.list("datasets", "het_a") {
        let mut jobs = Vec::new();
        for seed in 0..a.u("seeds", 10) as u64 {
            for rep in 0..reps {
                for fit in a.list("fit", "default,xfit") {
                    jobs.push((seed, rep, fit.clone()));
                }
            }
        }
        jobs.into_par_iter().for_each(|(seed, rep, fit)| {
            let s = load_split(&ds, seed, a);
            let mut ya = s.ya.clone();
            if rep > 0 {
                let mut g = StdRng::seed_from_u64(seed * 1000 + rep);
                for v in ya.iter_mut() {
                    let sign = if g.random_range(0.0..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    };
                    *v *= 1.0 + sign * jitter;
                }
            }
            let r = fit_scored(
                &c,
                cap_for(c.lr),
                patience_for(c.lr),
                seed,
                &fit,
                &s.xa,
                &ya,
                &s.xv,
                &s.yv,
            );
            let (n_iter, v, ev) = match &r {
                Some((m, v)) => (
                    m.base_models.len(),
                    *v,
                    evaluate(&m.pred_dist(&s.xte), &s.yte),
                ),
                None => (0, f64::NAN, EV_NAN),
            };
            emit(format!(
                "{ds},{seed},{rep},{fit},{n_iter},{v:.5},{:.5},{:.5}",
                ev.nll, ev.rmse
            ));
        });
    }
}

// ---------------------------------------------- 1-D k-means (Cycle 22)

/// Lloyd's algorithm on sorted values `v`, started from `k` equal-count
/// groups. Returns the start index of every (non-empty) cluster. In one
/// dimension clusters are contiguous in sorted order, so a cluster is a
/// range and an update is a prefix-sum lookup.
fn kmeans_1d(v: &[f64], k: usize) -> Vec<usize> {
    let n = v.len();
    let k = k.min(n).max(1);
    let mut pre = vec![0.0; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + v[i];
    }
    let mut starts: Vec<usize> = (0..k).map(|j| j * n / k).collect();
    starts.dedup();
    for _ in 0..500 {
        let m = starts.len();
        let cent: Vec<f64> = (0..m)
            .map(|j| {
                let (lo, hi) = (starts[j], if j + 1 < m { starts[j + 1] } else { n });
                (pre[hi] - pre[lo]) / (hi - lo) as f64
            })
            .collect();
        let mut next = vec![0usize];
        for j in 1..m {
            // rows below the midpoint of two neighbouring centroids go left
            let mid = 0.5 * (cent[j - 1] + cent[j]);
            let cut = v.partition_point(|&x| x < mid);
            if cut > *next.last().unwrap() && cut < n {
                next.push(cut);
            }
        }
        if next == starts {
            break;
        }
        starts = next;
    }
    starts
}

// ---------------------------------------------------------- mode: binning

/// Replace every feature by the index of its bin, with edges learned on the
/// training rows: `q` = the library's rule (equally spaced order statistics
/// of the distinct values), `km` = 1-D k-means. With at most 255 codes the
/// library then gives every code its own bin, so the two rules are compared
/// through the same tree builder.
fn code_features(s: &Split, k: usize, kmeans: bool) -> Split {
    let p = s.xa.ncols();
    let mut out = Split {
        xa: s.xa.clone(),
        ya: s.ya.clone(),
        xv: s.xv.clone(),
        yv: s.yv.clone(),
        xte: s.xte.clone(),
        yte: s.yte.clone(),
    };
    for f in 0..p {
        let mut v: Vec<f64> = s.xa.column(f).to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let edges: Vec<f64> = if kmeans {
            let mut e: Vec<f64> = kmeans_1d(&v, k).into_iter().map(|i| v[i]).collect();
            e.dedup();
            e
        } else {
            let mut u = v.clone();
            u.dedup();
            let nb = k.min(u.len());
            let mut e: Vec<f64> = if nb <= 1 {
                vec![u[0]]
            } else {
                (0..nb).map(|i| u[(i * (u.len() - 1)) / (nb - 1)]).collect()
            };
            e.dedup();
            e
        };
        // the library's `find_bin`: the last edge that is ≤ the value
        let code = |x: f64| edges.partition_point(|&e| e <= x).saturating_sub(1) as f64;
        for a in [&mut out.xa, &mut out.xv, &mut out.xte] {
            a.column_mut(f).mapv_inplace(code);
        }
    }
    out
}

fn mode_binning(a: &Args) {
    println!("dataset,seed,variant,bins,n_iter,val,test_nll,test_crps,test_rmse,cov90");
    let bins: Vec<usize> = a
        .list("bins", "255,32")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    // `--xtransform exp`: every feature becomes exp(1.5·x), a strictly
    // increasing map. Tree splits and quantile edges do not see it.
    let xt = a.s("xtransform", "none");
    for ds in a.list("datasets", "het_a") {
        let mut jobs = Vec::new();
        for seed in 0..a.u("seeds", 10) as u64 {
            for &k in &bins {
                for variant in a.list("variants", "lib,q,km") {
                    jobs.push((seed, k, variant.clone()));
                }
            }
        }
        jobs.into_par_iter().for_each(|(seed, k, variant)| {
            let mut s = load_split(&ds, seed, a);
            match xt.as_str() {
                "none" => {}
                "exp" => {
                    for x in [&mut s.xa, &mut s.xv, &mut s.xte] {
                        x.mapv_inplace(|v| (1.5 * v).exp());
                    }
                }
                other => panic!("unknown xtransform {other}"),
            }
            let (s, max_bins) = match variant.as_str() {
                "lib" => (s, k),
                "q" => (code_features(&s, k, false), 255),
                "km" => (code_features(&s, k, true), 255),
                other => panic!("unknown variant {other}"),
            };
            let c = DEFAULT_CFG;
            let mut m: Model<Normal> = NGBoost::with_full_options(
                cap_for(c.lr),
                c.lr,
                HistogramLearner::with_params(c.depth, max_bins, 20, 2),
                true,
                1.0,
                1.0,
                false,
                1.0,
                1e-4,
                Some(patience_for(c.lr)),
                0.1,
                LearningRateSchedule::Constant,
                0.0,
                LineSearchMethod::Binary,
                Some(seed),
            );
            let (n_iter, v, ev) =
                match m.fit_with_validation(&s.xa, &s.ya, Some(&s.xv), Some(&s.yv), None, None) {
                    Ok(()) => (
                        m.base_models.len(),
                        m.score(&s.xv, &s.yv),
                        evaluate(&m.pred_dist(&s.xte), &s.yte),
                    ),
                    Err(_) => (0, f64::NAN, EV_NAN),
                };
            let name = if xt == "none" {
                ds.clone()
            } else {
                format!("{ds}_{xt}")
            };
            emit(format!(
                "{name},{seed},{variant},{k},{n_iter},{v:.5},{:.5},{:.5},{:.5},{:.4}",
                ev.nll, ev.crps, ev.rmse, ev.cov90
            ));
        });
    }
}

// ---------------------------------------------------------- mode: mixinit

/// (weights, means, standard deviations) of a normal mixture.
type MixFit = (Vec<f64>, Vec<f64>, Vec<f64>);

/// EM for a K-component normal mixture, the loop of `NormalMixture::fit`
/// (variance floor, weights clamped to [0.01, 0.99], 300 rounds) from a
/// given start. Returns (weights, means, sds).
fn em_mixture(
    y: &[f64],
    mut pi: Vec<f64>,
    mut m: Vec<f64>,
    mut s: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, k) = (y.len(), m.len());
    let mean = y.iter().sum::<f64>() / n as f64;
    let sd = (y.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let floor = (1e-3 * sd).max(1e-9);
    s.iter_mut().for_each(|v| *v = v.max(floor));
    for _ in 0..300 {
        let (mut w, mut a, mut b) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
        for &yi in y {
            let t: Vec<f64> = (0..k)
                .map(|j| {
                    let z = (yi - m[j]) / s[j];
                    pi[j].ln() - s[j].ln() - 0.5 * z * z
                })
                .collect();
            let mx = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let tot: f64 = t.iter().map(|x| (x - mx).exp()).sum();
            for j in 0..k {
                let r = (t[j] - mx).exp() / tot;
                w[j] += r;
                a[j] += r * yi;
                b[j] += r * yi * yi;
            }
        }
        if w.iter().any(|&x| x < 1e-9) {
            break;
        }
        let mut delta = 0.0;
        for j in 0..k {
            let nm = a[j] / w[j];
            let ns = (b[j] / w[j] - nm * nm).max(floor * floor).sqrt();
            let np = (w[j] / n as f64).clamp(0.01, 0.99);
            delta += (nm - m[j]).abs() + (np - pi[j]).abs();
            (m[j], s[j], pi[j]) = (nm, ns, np);
        }
        let tot: f64 = pi.iter().sum();
        pi.iter_mut().for_each(|p| *p /= tot);
        if delta < 1e-10 * (1.0 + sd) {
            break;
        }
    }
    (pi, m, s)
}

/// Mean log-likelihood of `y` under a normal mixture.
fn mixture_ll(y: &[f64], pi: &[f64], m: &[f64], s: &[f64]) -> f64 {
    let c = 0.5 * (2.0 * std::f64::consts::PI).ln();
    y.iter()
        .map(|&yi| {
            let t: Vec<f64> = (0..m.len())
                .map(|j| {
                    let z = (yi - m[j]) / s[j];
                    pi[j].ln() - s[j].ln() - 0.5 * z * z - c
                })
                .collect();
            let mx = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            mx + t.iter().map(|x| (x - mx).exp()).sum::<f64>().ln()
        })
        .sum::<f64>()
        / y.len() as f64
}

/// Start values from a partition of the sorted sample into ranges.
fn start_from_groups(v: &[f64], starts: &[usize]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = v.len();
    let (mut pi, mut m, mut s) = (Vec::new(), Vec::new(), Vec::new());
    for (j, &lo) in starts.iter().enumerate() {
        let hi = if j + 1 < starts.len() {
            starts[j + 1]
        } else {
            n
        };
        let g = &v[lo..hi.max(lo + 1)];
        let gm = g.iter().sum::<f64>() / g.len() as f64;
        pi.push(g.len() as f64 / n as f64);
        m.push(gm);
        s.push((g.iter().map(|x| (x - gm).powi(2)).sum::<f64>() / g.len() as f64).sqrt());
    }
    (pi, m, s)
}

fn mode_mixinit(a: &Args) {
    println!("target,seed,k,init,train_ll,test_ll,weights,means,sds");
    let restarts = a.u("restarts", 30);
    for ds in a.list(
        "targets",
        "mix_sep,mix_rare,mix_overlap,mix_three,protein,kin8nm,het_a,tnoise",
    ) {
        for seed in 0..a.u("seeds", 10) as u64 {
            // marginal targets: a synthetic mixture or a dataset's target
            let (ytr, yte): (Vec<f64>, Vec<f64>) = if ds.starts_with("mix_") {
                let mut rng = StdRng::seed_from_u64(seed ^ 0x3117);
                let mut g = Gauss(&mut rng);
                let mut draw = |n: usize| -> Vec<f64> {
                    (0..n)
                        .map(|_| {
                            let u: f64 = g.0.random_range(0.0..1.0);
                            match ds.as_str() {
                                // two well separated modes
                                "mix_sep" if u < 0.3 => -3.0 + 0.5 * g.z(),
                                "mix_sep" => 2.0 + g.z(),
                                // a 5 % component far from the bulk
                                "mix_rare" if u < 0.05 => 8.0 + 0.3 * g.z(),
                                "mix_rare" => g.z(),
                                // a narrow mode inside a wide one
                                "mix_overlap" if u < 0.5 => g.z(),
                                "mix_overlap" => 1.5 + 0.3 * g.z(),
                                // three modes of unequal size
                                "mix_three" if u < 0.1 => -6.0 + 0.5 * g.z(),
                                "mix_three" if u < 0.7 => g.z(),
                                "mix_three" => 7.0 + 1.5 * g.z(),
                                other => panic!("unknown target {other}"),
                            }
                        })
                        .collect()
                };
                (draw(2100), draw(2000))
            } else {
                let s = load_split(&ds, seed, a);
                (s.ya.to_vec(), s.yte.to_vec())
            };
            let mut v = ytr.clone();
            v.sort_by(|x, y| x.total_cmp(y));
            let n = v.len();
            for k in [2usize, 3] {
                let report = |init: &str, (pi, m, s): MixFit| {
                    let mut ord: Vec<usize> = (0..m.len()).collect();
                    ord.sort_by(|&p, &q| m[p].total_cmp(&m[q]));
                    let j = |x: &[f64]| {
                        ord.iter()
                            .map(|&i| format!("{:.3}", x[i]))
                            .collect::<Vec<_>>()
                            .join(";")
                    };
                    emit(format!(
                        "{ds},{seed},{k},{init},{:.5},{:.5},{},{},{}",
                        mixture_ll(&ytr, &pi, &m, &s),
                        mixture_ll(&yte, &pi, &m, &s),
                        j(&pi),
                        j(&m),
                        j(&s)
                    ));
                };
                // the library's own fit
                let lib = if k == 2 {
                    ngboost_rs::dist::NormalMixture2::fit(&Array1::from(ytr.clone())).to_vec()
                } else {
                    ngboost_rs::dist::NormalMixture3::fit(&Array1::from(ytr.clone())).to_vec()
                };
                {
                    // columns: K−1 logits against the last component, then
                    // (mean, log sd) per component
                    let mut w: Vec<f64> = lib[..k - 1].iter().map(|l| l.exp()).collect();
                    w.push(1.0);
                    let tot: f64 = w.iter().sum();
                    let pi: Vec<f64> = w.iter().map(|x| x / tot).collect();
                    let m: Vec<f64> = (0..k).map(|j| lib[(k - 1) + 2 * j]).collect();
                    let s: Vec<f64> = (0..k).map(|j| lib[(k - 1) + 2 * j + 1].exp()).collect();
                    report("lib", (pi, m, s));
                }
                // the same start through this file's copy of the loop
                let q: Vec<usize> = (0..k).map(|j| j * n / k).collect();
                let (_, m0, s0) = start_from_groups(&v, &q);
                report(
                    "quantile",
                    em_mixture(&ytr, vec![1.0 / k as f64; k], m0, s0),
                );
                // 1-D k-means groups, with their own sizes as weights
                let km = kmeans_1d(&v, k);
                let (p1, m1, s1) = start_from_groups(&v, &km);
                if m1.len() == k {
                    report("kmeans", em_mixture(&ytr, p1, m1, s1));
                }
                // reference: the best of many random starts by training
                // likelihood (means at random data points, pooled sd)
                let mut rng = StdRng::seed_from_u64(seed * 31 + k as u64);
                let mean = v.iter().sum::<f64>() / n as f64;
                let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
                let mut best: Option<(f64, MixFit)> = None;
                for _ in 0..restarts {
                    let m0: Vec<f64> = (0..k).map(|_| v[rng.random_range(0..n)]).collect();
                    let scale: f64 = rng.random_range(0.2..1.0);
                    let fit = em_mixture(&ytr, vec![1.0 / k as f64; k], m0, vec![sd * scale; k]);
                    let ll = mixture_ll(&ytr, &fit.0, &fit.1, &fit.2);
                    if ll.is_finite() && best.as_ref().is_none_or(|b| ll > b.0) {
                        best = Some((ll, fit));
                    }
                }
                if let Some((_, fit)) = best {
                    report("best_random", fit);
                }
            }
        }
    }
}

// --------------------------------------------------------- mode: survmean

/// `NGBSurvival<LogNormal>` on a regression target used as an event time
/// (every time observed, or a share censored at a uniform fraction of the
/// time). Reports the largest fitted log σ, the largest mean/median ratio
/// and the RMSE of the predicted mean and median against the true time.
fn mode_survmean(a: &Args) {
    use ngboost_rs::dist::LogNormal;
    use ngboost_rs::scores::LogScoreCensored;
    use ngboost_rs::survival::NGBSurvival;
    println!(
        "dataset,seed,censored,n_iter,max_log_sigma,med_log_sigma,max_mean_over_median,rmse_mean,rmse_median"
    );
    let leaf = a.u("leaf", 20);
    for ds in a.list("datasets", "kin8nm") {
        for seed in 0..a.u("seeds", 10) as u64 {
            for cens in a.list("censored", "0,0.3") {
                let frac: f64 = cens.parse().unwrap();
                let s = load_split(&ds, seed, a);
                let mut g = StdRng::seed_from_u64(seed ^ 0xce45);
                let mut censor = |y: &Array1<f64>| {
                    let mut t = y.clone();
                    let mut e = Array1::ones(y.len());
                    for i in 0..y.len() {
                        if g.random_range(0.0..1.0) < frac {
                            t[i] = y[i] * g.random_range(0.3..1.0);
                            e[i] = 0.0;
                        }
                    }
                    (t, e)
                };
                let (ta, ea) = censor(&s.ya);
                let (tv, ev) = censor(&s.yv);
                let mut m = NGBSurvival::<LogNormal, LogScoreCensored, HistogramLearner>::new(
                    cap_for(0.05),
                    0.05,
                    HistogramLearner::with_params(3, 255, leaf, 2),
                );
                m.early_stopping_rounds = Some(patience_for(0.05));
                if let Err(e) = m.fit_with_validation(
                    &s.xa,
                    &ta,
                    &ea,
                    Some(&s.xv),
                    Some(&tv),
                    Some(&ev),
                    None,
                    None,
                ) {
                    eprintln!("{ds} {seed} {cens}: fit failed: {e}");
                    continue;
                }
                let p = m.pred_param(&s.xte);
                let mut ls: Vec<f64> = p.column(1).to_vec();
                ls.sort_by(|x, y| x.total_cmp(y));
                let (mut se_mean, mut se_med, mut ratio) = (0.0, 0.0, 0.0_f64);
                for i in 0..s.yte.len() {
                    let (mu, sg) = (p[[i, 0]], p[[i, 1]].exp());
                    let (mean, med) = ((mu + 0.5 * sg * sg).exp(), mu.exp());
                    se_mean += (mean - s.yte[i]).powi(2);
                    se_med += (med - s.yte[i]).powi(2);
                    ratio = ratio.max(mean / med);
                }
                let n = s.yte.len() as f64;
                emit(format!(
                    "{ds},{seed},{cens},{},{:.4},{:.4},{:.4e},{:.5},{:.5}",
                    m.base_models.len(),
                    ls[ls.len() - 1],
                    ls[ls.len() / 2],
                    ratio,
                    (se_mean / n).sqrt(),
                    (se_med / n).sqrt()
                ));
            }
        }
    }
}

// ----------------------------------------------------------- mode: hposel

/// One search of the shipped space per head (`--heads`), and for each the
/// winner after the first 15 trials and after all 30, refitted on the pool
/// and scored on the test block. Choosing the head is then a comparison of
/// `cv_score` across the rows of a (dataset, seed) — done in the summary
/// script, not here. TPE is sequential, so the best of the first 15 trials
/// of a 30-trial search is what a 15-trial search returns.
fn mode_hposel(a: &Args) {
    use ngboost_rs::dist::{GammaMean, StudentTFiniteVar};
    use ngboost_rs::hyper_opt::{
        HyperOptConfig, default_histogram_model, default_search_space, hyper_opt_with_config,
    };
    println!(
        "dataset,seed,head,trials,cv_score,lr,depth,msl,cs,max_bins,search_secs,n_iter,test_nll,test_rmse"
    );
    fn one<D>(
        a: &Args,
        ds: &str,
        seed: u64,
        head: &str,
        xp: &Array2<f64>,
        yp: &Array1<f64>,
        s: &Split,
    ) where
        D: Distribution + DistributionMethods + Scorable<LogScore> + Clone + 'static,
    {
        let trials = a.u("trials", 30) as u32;
        let cfg = HyperOptConfig {
            n_trials: trials,
            n_folds: a.u("folds", 5),
            seed,
            verbose: false,
            ..Default::default()
        };
        let t = Instant::now();
        let r = match hyper_opt_with_config::<D, LogScore, HistogramLearner, _>(
            xp,
            yp,
            &default_search_space(),
            cfg,
            default_histogram_model::<D, LogScore>,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{ds} {seed} {head}: search failed: {e}");
                return;
            }
        };
        let secs = t.elapsed().as_secs_f64();
        for budget in [15usize, trials as usize] {
            let Some(best) = r.trials[..budget.min(r.trials.len())]
                .iter()
                .filter(|t| t.score.is_finite())
                .min_by(|p, q| p.score.total_cmp(&q.score))
            else {
                continue;
            };
            let mut m = default_histogram_model::<D, LogScore>(&best.params);
            m.set_random_state(seed);
            let g = |k: &str| {
                best.params
                    .get(k)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(f64::NAN)
            };
            let (n_iter, nll, rmse) = match m.fit(xp, yp) {
                Ok(()) => {
                    let p = m.predict(&s.xte);
                    (
                        m.base_models.len(),
                        m.score(&s.xte, &s.yte),
                        (&p - &s.yte).mapv(|e| e * e).mean().unwrap().sqrt(),
                    )
                }
                Err(e) => {
                    eprintln!("{ds} {seed} {head}: refit failed: {e}");
                    (0, f64::NAN, f64::NAN)
                }
            };
            emit(format!(
                "{ds},{seed},{head},{budget},{:.5},{:.5},{},{},{:.3},{},{secs:.1},{n_iter},{nll:.5},{rmse:.5}",
                best.score,
                g("learning_rate"),
                g("max_depth"),
                g("min_samples_leaf"),
                g("col_sample"),
                g("max_bins")
            ));
        }
    }
    for ds in a.list("datasets", "het_a") {
        for seed in a.u("seed0", 0)..a.u("seed0", 0) + a.u("seeds", 3) {
            let seed = seed as u64;
            let s = load_split(&ds, seed, a);
            let xp = ndarray::concatenate(Axis(0), &[s.xa.view(), s.xv.view()]).unwrap();
            let yp = ndarray::concatenate(Axis(0), &[s.ya.view(), s.yv.view()]).unwrap();
            for head in a.list("heads", "normal,studenttfv") {
                match head.as_str() {
                    "normal" => one::<Normal>(a, &ds, seed, &head, &xp, &yp, &s),
                    "studenttfv" => one::<StudentTFiniteVar>(a, &ds, seed, &head, &xp, &yp, &s),
                    "lognormal" => one::<LogNormal>(a, &ds, seed, &head, &xp, &yp, &s),
                    "gammamean" => one::<GammaMean>(a, &ds, seed, &head, &xp, &yp, &s),
                    other => panic!("unknown head {other}"),
                }
            }
        }
    }
}

// --------------------------------------------------------- mode: treeleaf

/// `min_samples_leaf` sweep for the exact (non-histogram) tree learners, same
/// model settings, data and scoring as `leaf` with `--fit default`.
fn mode_treeleaf(a: &Args) {
    println!("dataset,seed,fit,msl,depth,n_iter,secs,val,test_nll,test_crps,test_rmse,cov90,calib");
    let leaves: Vec<usize> = a
        .list("leaves", "1,20")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let depth = a.u("depth", 3);
    let learner = a.s("learner", "exact");
    fn run<B: BaseLearner + Clone + Send + Sync + 'static>(
        b: B,
        seed: u64,
        s: &Split,
    ) -> Option<(usize, f64, Ev)> {
        let c = DEFAULT_CFG;
        let mut m: NGBoost<Normal, LogScore, B> = NGBoost::with_full_options(
            cap_for(c.lr),
            c.lr,
            b,
            true,
            c.mb,
            c.cs,
            false,
            1.0,
            1e-4,
            Some(patience_for(c.lr)),
            0.1,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            Some(seed),
        );
        m.fit_with_validation(&s.xa, &s.ya, Some(&s.xv), Some(&s.yv), None, None)
            .ok()?;
        let v = m.score(&s.xv, &s.yv);
        Some((
            m.base_models.len(),
            v,
            evaluate(&m.pred_dist(&s.xte), &s.yte),
        ))
    }
    for ds in a.list("datasets", "het_a") {
        let mut jobs = Vec::new();
        for seed in 0..a.u("seeds", 10) as u64 {
            for &msl in &leaves {
                jobs.push((seed, msl));
            }
        }
        jobs.into_par_iter().for_each(|(seed, msl)| {
            let s = load_split(&ds, seed, a);
            let t = Instant::now();
            let r = match learner.as_str() {
                "exact" => run(DecisionTreeLearner::with_params(depth, 2, msl), seed, &s),
                "arena" => run(ArenaDecisionTreeLearner::with_params(depth, 2, msl), seed, &s),
                other => panic!("unknown learner {other}"),
            };
            let secs = t.elapsed().as_secs_f64();
            let (n_iter, v, ev) = r.unwrap_or((0, f64::NAN, EV_NAN));
            emit(format!(
                "{ds},{seed},{learner},{msl},{depth},{n_iter},{secs:.2},{v:.5},{:.5},{:.5},{:.5},{:.4},{:.4}",
                ev.nll, ev.crps, ev.rmse, ev.cov90, ev.calib
            ));
        });
    }
}

// ----------------------------------------------------------- mode: counts

/// Cycle 26 (O17): the count heads on real low-count data, and the staged
/// NegativeBinomial fit with cross-fitted residuals. Datasets: prepared
/// CSVs in `--data-dir` (`research/optimization-algorithm/prep_counts.py`)
/// or the synthetic count DGPs. Arms: `pois`, `nb` (library defaults on
/// the shipped default-shape model), suffix `_l1` (leaf 1, the 1.2.0
/// trees), `nb_st` (staged), `nb_x5` / `nb_x3` (staged, cross-fitted),
/// `auto` (`CountModel::fit_auto`; its `staged` column is 1 when it chose
/// NegativeBinomial).
/// `--design iid`: random shuffle per seed, test = 20 % of the rows (at
/// most `--test-rows`), pool = the rest capped at each of `--sizes`.
/// `--design chrono` (football files, whose first column is the season):
/// test = one season, training = all earlier seasons, per `--leagues`
/// entry (`all` or a league code), chronological validation split.
fn mode_counts(a: &Args) {
    use ngboost_rs::hyper_opt::default_histogram_model;
    use ngboost_rs::ngboost::ValidationSplit;
    println!(
        "dataset,design,cell,seed,n_train,n_test,arm,staged,n_iter,secs,nll,rmse,bias,cov90,finite"
    );
    /// Features, target and (football) season and league per row.
    type Loaded = (Array2<f64>, Array1<f64>, Option<(Vec<usize>, Vec<usize>)>);
    fn load(a: &Args, ds: &str, n_synth: usize, seed: u64) -> Loaded {
        if ds.starts_with("pois_") || ds.starts_with("nb_") {
            let (x, y) = synth_count(ds, n_synth, seed);
            return (x, y, None);
        }
        let dir = a.s("data-dir", "../data_counts/prepared");
        let (x, y) = load_csv(&format!("{dir}/{ds}.csv"), "y");
        if ds.starts_with("fb") {
            // column 0 = season (split key only), column 1 = league
            let season = x.column(0).iter().map(|&v| v as usize).collect();
            let league = x.column(1).iter().map(|&v| v as usize).collect();
            let feats: Vec<usize> = (1..x.ncols()).collect();
            (x.select(Axis(1), &feats), y, Some((season, league)))
        } else {
            (x, y, None)
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn fit_eval<D>(
        tag: &str,
        arm: &str,
        chrono: bool,
        seed: u64,
        xp: &Array2<f64>,
        yp: &Array1<f64>,
        xte: &Array2<f64>,
        yte: &Array1<f64>,
    ) where
        D: Distribution + DistributionMethods + Scorable<LogScore> + Clone + 'static,
    {
        // `_l1`: an explicit leaf of 1 (the 1.2.0 trees) instead of the
        // adaptive default of 20
        let mut hp = HashMap::new();
        if arm.ends_with("_l1") {
            hp.insert("min_samples_leaf".to_string(), serde_json::json!(1));
        }
        let mut m = default_histogram_model::<D, LogScore>(&hp);
        m.set_random_state(seed);
        if arm.ends_with("_st") {
            m.staged_fit = Some(true);
        }
        if let Some(k) = arm
            .rsplit_once("_x")
            .and_then(|(_, k)| k.parse::<usize>().ok())
        {
            m.staged_fit = Some(true);
            m.staged_crossfit_folds = k;
        }
        if chrono {
            m.validation_split = ValidationSplit::Chronological;
        }
        let staged = m.uses_staged_fit(yp.len(), None);
        let t = Instant::now();
        let r = m.fit(xp, yp);
        let secs = t.elapsed().as_secs_f64();
        let head = format!("{tag},{},{},{arm},{}", yp.len(), yte.len(), staged as u8);
        if let Err(e) = r {
            eprintln!("{tag} {arm}: fit failed: {e}");
            emit(format!("{head},0,{secs:.3},NaN,NaN,NaN,NaN,0"));
            return;
        }
        let d = m.pred_dist(xte);
        let p = m.predict(xte);
        let nll = Scorable::<LogScore>::score(&d, yte).mean().unwrap();
        let rmse = (&p - yte).mapv(|e| e * e).mean().unwrap().sqrt();
        let (lo, hi) = d.interval(0.1);
        let cov = (0..yte.len())
            .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
            .count() as f64
            / yte.len() as f64;
        let finite = nll.is_finite() && p.iter().all(|v| v.is_finite());
        emit(format!(
            "{head},{},{secs:.3},{nll:.5},{rmse:.5},{:.4},{cov:.4},{}",
            m.base_models.len(),
            yte.sum() / p.sum(),
            finite as u8
        ));
    }
    #[allow(clippy::too_many_arguments)]
    fn run_arm(
        tag: &str,
        arm: &str,
        chrono: bool,
        seed: u64,
        xp: &Array2<f64>,
        yp: &Array1<f64>,
        xte: &Array2<f64>,
        yte: &Array1<f64>,
    ) {
        if arm == "auto" {
            // `CountModel::fit_auto`: Poisson, switched to NegativeBinomial
            // when the validation dispersion test fires. It needs explicit
            // validation rows: the last 10 % of the pool (the pool is in
            // date order under `chrono` and already shuffled otherwise).
            let mut tpl = default_histogram_model::<Poisson, LogScore>(&HashMap::new());
            tpl.set_random_state(seed);
            let n_tr = yp.len() - yp.len().div_ceil(10);
            let (tr, va): (Vec<usize>, Vec<usize>) =
                ((0..n_tr).collect(), (n_tr..yp.len()).collect());
            let ((xt, yt), (xv, yv)) = (take(xp, yp, &tr), take(xp, yp, &va));
            let t = Instant::now();
            let r = ngboost_rs::CountModel::fit_auto(&tpl, &xt, &yt, &xv, &yv);
            let secs = t.elapsed().as_secs_f64();
            let head = format!("{tag},{},{},{arm}", yp.len(), yte.len());
            match r {
                Err(e) => {
                    eprintln!("{tag} {arm}: fit failed: {e}");
                    emit(format!("{head},0,0,{secs:.3},NaN,NaN,NaN,NaN,0"));
                }
                Ok((m, _z)) => {
                    let p = m.predict(xte);
                    let nll = m.score(xte, yte);
                    let rmse = (&p - yte).mapv(|e| e * e).mean().unwrap().sqrt();
                    let (lo, hi) = m.interval(xte, 0.1);
                    let cov = (0..yte.len())
                        .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
                        .count() as f64
                        / yte.len() as f64;
                    // the `staged` column reports the head: 1 = NegativeBinomial
                    emit(format!(
                        "{head},{},0,{secs:.3},{nll:.5},{rmse:.5},{:.4},{cov:.4},{}",
                        m.is_negative_binomial() as u8,
                        yte.sum() / p.sum(),
                        nll.is_finite() as u8
                    ));
                }
            }
        } else if arm.starts_with("pois") {
            fit_eval::<Poisson>(tag, arm, chrono, seed, xp, yp, xte, yte)
        } else if arm.starts_with("nb") {
            fit_eval::<NegativeBinomial>(tag, arm, chrono, seed, xp, yp, xte, yte)
        } else {
            panic!("unknown arm {arm}")
        }
    }
    let arms = a.list("arms", "pois,nb");
    let chrono = a.s("design", "iid") == "chrono";
    let n_test_max = a.u("test-rows", 2000);
    let sizes: Vec<usize> = a
        .list("sizes", "800,3200")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    // (dataset, cell, seed, size) jobs; every arm of a job sees the same rows
    let mut jobs: Vec<(String, String, u64, usize)> = Vec::new();
    for ds in a.list("datasets", "pois_goals") {
        if chrono {
            for lg in a.list("leagues", "all") {
                for season in 2..=5u64 {
                    jobs.push((ds.clone(), lg.clone(), season, 0));
                }
            }
        } else {
            for seed in a.u("seed0", 300)..a.u("seed0", 300) + a.u("seeds", 10) {
                for &size in &sizes {
                    jobs.push((ds.clone(), "iid".to_string(), seed as u64, size));
                }
            }
        }
    }
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    jobs.into_par_iter().for_each(|(ds, cell, seed, size)| {
        let (x, y, meta) = load(a, &ds, (max_size + n_test_max).max(5 * n_test_max), seed);
        let (tr, te): (Vec<usize>, Vec<usize>) = if chrono {
            let (season, league) = meta.as_ref().expect("chrono needs a football file");
            let keep = |i: usize| cell == "all" || league[i] == cell.parse::<usize>().unwrap();
            let s = seed as usize;
            (
                (0..y.len()).filter(|&i| keep(i) && season[i] < s).collect(),
                (0..y.len())
                    .filter(|&i| keep(i) && season[i] == s)
                    .collect(),
            )
        } else {
            let idx = shuffled_indices(y.len(), seed);
            let n_test = n_test_max.min(y.len() / 5);
            let avail = y.len() - n_test;
            // a size beyond the data would repeat the full-pool job
            if size > avail && sizes.iter().any(|&s| s < size && s >= avail) {
                return;
            }
            (
                idx[n_test..n_test + size.min(avail)].to_vec(),
                idx[..n_test].to_vec(),
            )
        };
        let (xp, yp) = take(&x, &y, &tr);
        let (xte, yte) = take(&x, &y, &te);
        let design = if chrono { "chrono" } else { "iid" };
        let tag = format!("{ds},{design},{cell},{seed}");
        arms.par_iter()
            .for_each(|arm| run_arm(&tag, arm, chrono, seed, &xp, &yp, &xte, &yte));
    });
}

// ------------------------------------------------------------- mode: xdef

/// Cycle 25 (O16): cross-fitting as the default of the staged fit. The
/// shipped default-shape model (`default_histogram_model` with no keys:
/// learning rate 0.05, depth 3, adaptive leaf, early stopping) fitted the
/// way a user fits it — `fit(x, y)`, which holds out `validation_fraction`
/// itself — with `staged_crossfit_folds` 0 (`off`) against K (`k5`, `k3`),
/// on a pool of `--sizes` rows, scored on `--test-rows` rows outside it.
/// `--weights tri`: random sample weights 0.25 / 1 / 4; `--split chrono`:
/// the chronological validation split (contiguous folds); `--serial 1`:
/// one fit at a time, for clean wall times.
fn mode_xdef(a: &Args) {
    use ngboost_rs::dist::{GammaMean, StudentT, StudentTFiniteVar};
    use ngboost_rs::hyper_opt::default_histogram_model;
    use ngboost_rs::ngboost::ValidationSplit;
    println!("head,dataset,size,seed,arm,staged,n_iter,secs,nll,rmse,cov90,finite");
    fn one<D>(a: &Args, head: &str, ds: &str, size: usize, seed: u64, arm: &str)
    where
        D: Distribution + DistributionMethods + Scorable<LogScore> + Clone + 'static,
    {
        let n_test = a.u("test-rows", 2000);
        let (x, y) = match ds {
            "kin8nm" => load_csv(&format!("{DATA_DIR}/kin8nm.csv"), "y"),
            "protein" => load_csv(&format!("{DATA_DIR}/protein.csv"), "RMSD"),
            _ => synth(ds, size + n_test, seed),
        };
        let idx = shuffled_indices(x.nrows(), seed);
        if idx.len() < size + n_test {
            eprintln!("{head} {ds} {size}: not enough rows, skipped");
            return;
        }
        let (xp, yp) = take(&x, &y, &idx[..size]);
        let (xte, yte) = take(&x, &y, &idx[size..size + n_test]);
        let mut m = default_histogram_model::<D, LogScore>(&HashMap::new());
        m.set_random_state(seed);
        m.staged_crossfit_folds = match arm {
            "off" => 0,
            k => k
                .strip_prefix('k')
                .and_then(|v| v.parse().ok())
                .expect("arm"),
        };
        if a.s("split", "random") == "chrono" {
            m.validation_split = ValidationSplit::Chronological;
        }
        let w = (a.s("weights", "none") == "tri").then(|| {
            let mut g = StdRng::seed_from_u64(seed ^ 0x77_e167);
            Array1::from_shape_fn(size, |_| [0.25, 1.0, 4.0][g.random_range(0..3)])
        });
        let staged = m.uses_staged_fit(size, None);
        let t = Instant::now();
        let r = m.fit_with_validation(&xp, &yp, None, None, w.as_ref(), None);
        let secs = t.elapsed().as_secs_f64();
        if let Err(e) = r {
            eprintln!("{head} {ds} {size} {seed} {arm}: fit failed: {e}");
            emit(format!(
                "{head},{ds},{size},{seed},{arm},{},0,{secs:.3},NaN,NaN,NaN,0",
                staged as u8
            ));
            return;
        }
        let d = m.pred_dist(&xte);
        let p = m.predict(&xte);
        let nll = Scorable::<LogScore>::score(&d, &yte).mean().unwrap();
        let rmse = (&p - &yte).mapv(|e| e * e).mean().unwrap().sqrt();
        let (lo, hi) = d.interval(0.1);
        let cov = (0..yte.len())
            .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
            .count() as f64
            / yte.len() as f64;
        let finite = nll.is_finite() && p.iter().all(|v| v.is_finite());
        if std::env::var("NGB_XDEF_DIAG").is_ok() {
            // the test rows that carry the squared error, with the model's
            // parameters (bounded as predicted, and as fitted)
            let (pb, pu) = (m.pred_param(&xte), m.pred_param_unbounded(&xte));
            let med = d.median();
            let mut ord: Vec<usize> = (0..yte.len()).collect();
            ord.sort_by(|&i, &j| (p[j] - yte[j]).abs().total_cmp(&(p[i] - yte[i]).abs()));
            let sse: f64 = ord.iter().map(|&i| (p[i] - yte[i]).powi(2)).sum();
            eprintln!(
                "XDEFDIAG {head},{ds},{size},{seed},{arm}: init={:?} rmse(median)={:.4}",
                m.init_params.as_ref().map(|v| v.to_vec()),
                (&med - &yte).mapv(|e| e * e).mean().unwrap().sqrt()
            );
            for &i in ord.iter().take(6) {
                eprintln!(
                    "XDEFDIAG   row {i}: y={:.3} pred={:.3} median={:.3} share_sse={:.3} params={:?} unbounded={:?}",
                    yte[i],
                    p[i],
                    med[i],
                    (p[i] - yte[i]).powi(2) / sse,
                    pb.row(i).to_vec(),
                    pu.row(i).to_vec()
                );
            }
        }
        emit(format!(
            "{head},{ds},{size},{seed},{arm},{},{},{secs:.3},{nll:.5},{rmse:.5},{cov:.4},{}",
            staged as u8,
            m.base_models.len(),
            finite as u8
        ));
    }
    let sizes: Vec<usize> = a
        .list("sizes", "400,1600")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let mut jobs = Vec::new();
    for head in a.list("heads", "normal") {
        for ds in a.list("datasets", "het_a") {
            for &size in &sizes {
                for seed in a.u("seed0", 100)..a.u("seed0", 100) + a.u("seeds", 10) {
                    for arm in a.list("arms", "off,k5") {
                        jobs.push((head.clone(), ds.clone(), size, seed as u64, arm));
                    }
                }
            }
        }
    }
    let run =
        |(head, ds, size, seed, arm): (String, String, usize, u64, String)| match head.as_str() {
            "normal" => one::<Normal>(a, &head, &ds, size, seed, &arm),
            "lognormal" => one::<LogNormal>(a, &head, &ds, size, seed, &arm),
            "studentt" => one::<StudentT>(a, &head, &ds, size, seed, &arm),
            "studenttfv" => one::<StudentTFiniteVar>(a, &head, &ds, size, seed, &arm),
            "gammamean" => one::<GammaMean>(a, &head, &ds, size, seed, &arm),
            other => panic!("unknown head {other}"),
        };
    if a.u("serial", 0) == 1 {
        jobs.into_iter().for_each(run);
    } else {
        jobs.into_par_iter().for_each(run);
    }
}

fn main() {
    let a = Args::parse();
    match a.s("mode", "pool").as_str() {
        "pool" => mode_pool(&a),
        "search" => mode_search(&a),
        "lrcap" => mode_lrcap(&a),
        "select" => mode_select(&a),
        "oof" => mode_oof(&a),
        "leaf" => mode_leaf(&a),
        "leafsplit" => mode_leafsplit(&a),
        "treeleaf" => mode_treeleaf(&a),
        "hpo" => mode_hpo(&a),
        "hposel" => mode_hposel(&a),
        "survmean" => mode_survmean(&a),
        "rows" => mode_rows(&a),
        "noise" => mode_noise(&a),
        "binning" => mode_binning(&a),
        "mixinit" => mode_mixinit(&a),
        "xdef" => mode_xdef(&a),
        "counts" => mode_counts(&a),
        other => panic!("unknown mode {other}"),
    }
}
