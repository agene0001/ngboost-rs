//! Temporal research benchmark (research/ngboost Cycle 16). NOT public API.
//!
//! Normal, Poisson and NegativeBinomial heads on time series with lag
//! features and a chronological test block, using the SHIPPED defaults
//! (staged fit and scale recalibration where they apply; β = 1 for counts).
//!
//! Synthetic series (fresh series per seed, n rows after burn-in):
//!   train = [0, 0.7n), val = [0.7n, 0.8n), test = [0.8n, n).
//! Real series (one series; "seed" s = rolling origin): test = the s-th
//! 10 % block from the end, val = the 10 % block before it, train = the rest
//! before that (expanding window).
//!
//! Variant tokens (joined by '+'):
//!   chrono (default) | random — validation rows: the block just before the
//!       test block, or the same number of rows drawn at random from the
//!       whole pre-test period (the rest is train);
//!   joint | staged — override `staged_fit`;   nocal — no scale recalibration;
//!   refit — `NGBoost::refit_full` on train ∪ val (time order) for the kept
//!       number of rounds, carrying the first fit's recalibration (joint
//!       fits only; Cycle 16's R60/R61 `refit`/`carry` were the manual
//!       versions of the same, without / with the carried calibration);
//!   refits — as refit, rounds scaled by (train + val) / train;
//!   loccal — post-hoc location shift fitted on val (Normal: precision-
//!       weighted mean residual; counts: log Σy/Σμ̂);
//!   rcal — NB: affine map on log r fitted on val (≥ 300 rows);
//!   oracle — the data-generating distribution (synthetic only).
//!
//! Output: research_bench's CSV columns. calib = sqrt(mean err² / mean
//! predicted variance) (> 1 overconfident); mean_sigma = mean σ̂ (Normal),
//! geometric-mean r̂ (NB), mean √λ̂ (Poisson); extra = Σy/Σμ̂ on test
//! (counts) or mean(y − μ̂) (Normal); extra2 = test NLL of the second half
//! of the test block minus the first half (drift sensitivity).
//!
//!   ts_bench --dist normal --datasets tsn_stat,tsn_drift --variants chrono,random --seeds 10

use ndarray::{Array1, Array2, Axis, s};
use ngboost_rs::dist::{Distribution, DistributionMethods, NegativeBinomial, Normal, Poisson};
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{CRPScore, LogScore, Scorable, Score};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::f64::consts::TAU;
use std::time::Instant;

const TS_DIR: &str = "research/ngboost/data_ts";

struct Args {
    dist: String,
    datasets: Vec<String>,
    variants: Vec<String>,
    seeds: u64,
    seed0: u64,
    lr: f64,
    n_est: u32,
    patience: u32,
    depth: usize,
    n: usize,
    te_frac: f64,
    va_frac: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        dist: "normal".into(),
        datasets: vec!["tsn_stat".into()],
        variants: vec!["chrono".into()],
        seeds: 3,
        seed0: 0,
        lr: 0.05,
        n_est: 2000,
        patience: 50,
        depth: 3,
        n: 4000,
        te_frac: 0.1,
        va_frac: 0.1,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    assert!(argv.len() % 2 == 0, "arguments come in --flag value pairs");
    for p in argv.chunks(2) {
        let v = &p[1];
        match p[0].as_str() {
            "--dist" => a.dist = v.clone(),
            "--datasets" => a.datasets = v.split(',').map(String::from).collect(),
            "--variants" => a.variants = v.split(',').map(String::from).collect(),
            "--seeds" => a.seeds = v.parse().unwrap(),
            "--seed0" => a.seed0 = v.parse().unwrap(),
            "--lr" => a.lr = v.parse().unwrap(),
            "--n-est" => a.n_est = v.parse().unwrap(),
            "--patience" => a.patience = v.parse().unwrap(),
            "--depth" => a.depth = v.parse().unwrap(),
            "--n" => a.n = v.parse().unwrap(),
            "--te-frac" => a.te_frac = v.parse().unwrap(),
            "--va-frac" => a.va_frac = v.parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    a
}

// ------------------------------------------------------------------ data

/// Ground truth per row (synthetic only): mean and (Normal) σ or (counts) r.
#[derive(Clone)]
struct Truth {
    mu: Array1<f64>,
    /// Normal: σ; NB: r (f64::INFINITY = Poisson)
    shape: Array1<f64>,
}

struct Series {
    x: Array2<f64>,
    y: Array1<f64>,
    truth: Option<Truth>,
}

fn z(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

fn gamma(rng: &mut StdRng, shape: f64) -> f64 {
    // Marsaglia–Tsang (shape ≥ 1; boost for < 1)
    if shape < 1.0 {
        let u: f64 = rng.random_range(1e-12..1.0);
        return gamma(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = z(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random_range(1e-12..1.0);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn poisson(rng: &mut StdRng, lam: f64) -> f64 {
    if lam < 30.0 {
        let l = (-lam).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= rng.random_range(0.0..1.0f64);
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
    // normal approximation is too crude for a benchmark; split the rate
    let h = lam / 2.0;
    poisson(rng, h) + poisson(rng, lam - h)
}

/// Synthetic series. Features at t: 3 lags of y (log1p for counts), weekly
/// and yearly sin/cos, two observed AR(1) covariates, two irrelevant AR(1)
/// covariates, and the time index.
fn synth(name: &str, n: usize, seed: u64) -> Series {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17));
    let burn = 400;
    let total = n + burn;
    let count = !name.starts_with("tsn_");
    let drift = name.ends_with("_drift");
    let nb_r = if name.starts_with("tsnb_") {
        Some(5.0)
    } else {
        None
    };
    let ar1 = |rng: &mut StdRng, phi: f64| -> Vec<f64> {
        let mut v = vec![0.0; total];
        let sd = (1.0 - phi * phi).sqrt();
        for t in 1..total {
            v[t] = phi * v[t - 1] + sd * z(rng);
        }
        v
    };
    let x1 = ar1(&mut rng, 0.9);
    let x2 = ar1(&mut rng, 0.9);
    let n1 = ar1(&mut rng, 0.95);
    let n2 = ar1(&mut rng, 0.5);
    let mut y = vec![0.0; total];
    let mut mu = vec![0.0; total];
    let mut shape = vec![0.0; total];
    let mut level = 0.0;
    let lag = |y: &Vec<f64>, t: usize, k: usize| -> f64 {
        if t >= k {
            if count {
                (1.0 + y[t - k]).ln()
            } else {
                y[t - k]
            }
        } else {
            0.0
        }
    };
    for t in 0..total {
        let tf = t as f64;
        let wk = (TAU * tf / 7.0).sin();
        let yr = (TAU * tf / 365.25).sin();
        if drift && t >= burn {
            level += if count { 0.006 } else { 0.03 } * z(&mut rng);
        }
        if count {
            let eta = 1.0
                + 0.35 * lag(&y, t, 1)
                + 0.15 * lag(&y, t, 7)
                + 0.4 * wk
                + 0.4 * x1[t]
                + 0.3 * (2.0 * x2[t]).tanh()
                + 0.3 * yr
                + level;
            let m = eta.min(6.0).exp();
            mu[t] = m;
            y[t] = match nb_r {
                Some(r) => {
                    shape[t] = r;
                    {
                        let g = gamma(&mut rng, r);
                        poisson(&mut rng, g * m / r)
                    }
                }
                None => {
                    shape[t] = f64::INFINITY;
                    poisson(&mut rng, m)
                }
            };
        } else {
            let m = 0.5 * lag(&y, t, 1) - 0.2 * lag(&y, t, 2)
                + 0.15 * lag(&y, t, 7)
                + 1.0 * wk
                + 0.8 * x1[t]
                + 0.5 * x2[t] * x2[t]
                + 0.6 * yr
                + level;
            let dy = if t >= 2 {
                (y[t - 1] - y[t - 2]).abs()
            } else {
                0.0
            };
            let mut sg = 0.3 + 0.5 / (1.0 + (-2.0 * x2[t]).exp()) + 0.25 * dy.min(10.0);
            if drift && t >= burn {
                sg *= (0.5 * (t - burn) as f64 / n as f64).exp();
            }
            mu[t] = m;
            shape[t] = sg;
            y[t] = m + sg * z(&mut rng);
        }
    }
    let nf = 12;
    let mut x = Array2::zeros((n, nf));
    for i in 0..n {
        let t = i + burn;
        let tf = t as f64;
        let row = [
            lag(&y, t, 1),
            lag(&y, t, 2),
            lag(&y, t, 7),
            (TAU * tf / 7.0).sin(),
            (TAU * tf / 7.0).cos(),
            (TAU * tf / 365.25).sin(),
            (TAU * tf / 365.25).cos(),
            x1[t],
            x2[t],
            n1[t],
            n2[t],
            i as f64,
        ];
        for (j, v) in row.iter().enumerate() {
            x[[i, j]] = *v;
        }
    }
    Series {
        x,
        y: Array1::from(y[burn..].to_vec()),
        truth: Some(Truth {
            mu: Array1::from(mu[burn..].to_vec()),
            shape: Array1::from(shape[burn..].to_vec()),
        }),
    }
}

/// iid count DGPs of research_bench (`synth_count`), for controls: 10
/// uniform(−2, 2) features, no time structure.
fn synth_iid(name: &str, n: usize, seed: u64) -> Series {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xc0_0a7);
    let x: Array2<f64> = Array2::from_shape_fn((n, 10), |_| rng.random_range(-2.0..2.0));
    let mut y = Array1::zeros(n);
    let mut mu = Array1::zeros(n);
    let mut shape = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        let nl = (2.0 + 0.8 * (2.0 * r[0]).sin() + 0.25 * r[1] * r[1] - 0.3 * r[2].abs()
            + 0.2 * r[3] * r[4])
            .exp();
        let low = (1.75 + 0.25 * r[0] + 0.15 * (2.0 * r[1]).sin()).exp();
        let (m, disp): (f64, Option<f64>) = match name {
            "iid_pois_lin" => ((0.8 + 0.6 * r[0] - 0.4 * r[1] + 0.3 * r[2]).exp(), None),
            "iid_pois_nl" => (nl, None),
            "iid_pois_low" => (low, None),
            "iid_nb_over" => (nl, Some(4.0)),
            "iid_nb_low" => (low, Some(10.0)),
            other => panic!("unknown iid dataset {other}"),
        };
        let lam = match disp {
            None => m,
            Some(rr) => {
                let g = gamma(&mut rng, rr);
                m * g / rr
            }
        };
        y[i] = poisson(&mut rng, lam);
        mu[i] = m;
        shape[i] = disp.unwrap_or(f64::INFINITY);
    }
    Series {
        x,
        y,
        truth: Some(Truth { mu, shape }),
    }
}

fn read_rows(file: &str) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(format!("{TS_DIR}/{file}"))
        .unwrap_or_else(|e| panic!("{file}: {e}"));
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|c| c.trim().trim_matches('"').to_string())
                .collect()
        })
        .collect()
}

/// Lagged design from a 1-D series: lags `lags` (optionally log1p), a
/// seasonal phase with period `period` (sin/cos), and the time index.
fn lag_design(v: &[f64], lags: &[usize], period: f64, log1p: bool) -> (Array2<f64>, Array1<f64>) {
    let start = *lags.iter().max().unwrap();
    let n = v.len() - start;
    let nf = lags.len() + 3;
    let mut x = Array2::zeros((n, nf));
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let t = i + start;
        for (j, &k) in lags.iter().enumerate() {
            let l = v[t - k];
            x[[i, j]] = if log1p { (1.0 + l).ln() } else { l };
        }
        let ph = TAU * t as f64 / period;
        x[[i, lags.len()]] = ph.sin();
        x[[i, lags.len() + 1]] = ph.cos();
        x[[i, lags.len() + 2]] = t as f64;
        y[i] = v[t];
    }
    (x, y)
}

fn real(name: &str) -> Series {
    let (x, y) = match name {
        // weekly CO2 (Mauna Loa), 1958–2001; gaps linearly interpolated;
        // target = weekly change (ppm), Normal head
        "co2" => {
            let rows = read_rows("co2.csv");
            let mut v: Vec<Option<f64>> = rows.iter().map(|r| r[1].parse::<f64>().ok()).collect();
            let mut last: Option<usize> = None;
            for i in 0..v.len() {
                if v[i].is_some() {
                    if let Some(j) = last
                        && i > j + 1
                    {
                        let (a, b) = (v[j].unwrap(), v[i].unwrap());
                        for k in j + 1..i {
                            let w = (k - j) as f64 / (i - j) as f64;
                            v[k] = Some(a + w * (b - a));
                        }
                    }
                    last = Some(i);
                }
            }
            let first = v.iter().position(|o| o.is_some()).unwrap();
            let lvl: Vec<f64> = v[first..].iter().map(|o| o.unwrap()).collect();
            let d: Vec<f64> = lvl.windows(2).map(|w| w[1] - w[0]).collect();
            lag_design(&d, &[1, 2, 3, 52], 52.18, false)
        }
        // monthly Niño sea-surface temperature, 1950–2010, Normal head
        "elnino" => {
            let rows = read_rows("elnino.csv");
            let v: Vec<f64> = rows
                .iter()
                .flat_map(|r| {
                    r[1..13]
                        .iter()
                        .map(|c| c.parse::<f64>().unwrap())
                        .collect::<Vec<_>>()
                })
                .collect();
            lag_design(&v, &[1, 2, 3, 12], 12.0, false)
        }
        // yearly sunspot numbers 1700–2008, rounded to counts
        "sunspots" => {
            let rows = read_rows("sunspots.csv");
            let v: Vec<f64> = rows
                .iter()
                .map(|r| r[1].parse::<f64>().unwrap().round())
                .collect();
            lag_design(&v, &[1, 2, 3, 9, 10, 11], 11.0, true)
        }
        other => panic!("unknown dataset {other}"),
    };
    Series { x, y, truth: None }
}

struct Split {
    tr: Vec<usize>,
    va: Vec<usize>,
    te: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn split(
    n: usize,
    synthetic: bool,
    origin: u64,
    random_val: bool,
    seed: u64,
    te_frac: f64,
    va_frac: f64,
) -> Split {
    let (te_lo, te_hi, n_va) = if synthetic {
        ((0.8 * n as f64) as usize, n, (0.1 * n as f64) as usize)
    } else {
        let h = (te_frac * n as f64).round() as usize;
        let hv = (va_frac * n as f64).round() as usize;
        let hi = n - origin as usize * h;
        (hi - h, hi, hv)
    };
    let te: Vec<usize> = (te_lo..te_hi).collect();
    let (tr, va) = if random_val {
        let mut idx: Vec<usize> = (0..te_lo).collect();
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }
        let mut va = idx[..n_va].to_vec();
        let mut tr = idx[n_va..].to_vec();
        va.sort_unstable();
        tr.sort_unstable();
        (tr, va)
    } else {
        ((0..te_lo - n_va).collect(), (te_lo - n_va..te_lo).collect())
    };
    Split { tr, va, te }
}

fn take(s: &Series, idx: &[usize]) -> (Array2<f64>, Array1<f64>) {
    (s.x.select(Axis(0), idx), s.y.select(Axis(0), idx))
}

// ------------------------------------------------------------------ models

struct Out {
    n_iter: usize,
    secs: f64,
    train_nll: f64,
    nll: Array1<f64>,
    crps: f64,
    rmse: f64,
    cov50: f64,
    cov90: f64,
    calib: f64,
    mean_sigma: f64,
    extra: f64,
    /// mean validation NLL of the final model (Normal only; else NaN)
    val_nll: f64,
}

struct Opts {
    random_val: bool,
    staged: Option<bool>,
    nocal: bool,
    refit: bool,
    scale_rounds: bool,
    loccal: bool,
    rcal: bool,
    oracle: bool,
}

fn parse_variant(v: &str) -> Opts {
    let mut o = Opts {
        random_val: false,
        staged: None,
        nocal: false,
        refit: false,
        scale_rounds: false,
        loccal: false,
        rcal: false,
        oracle: false,
    };
    for tok in v.split('+') {
        match tok {
            "chrono" => {}
            "random" => o.random_val = true,
            "joint" => o.staged = Some(false),
            "staged" => o.staged = Some(true),
            "nocal" => o.nocal = true,
            "refit" => o.refit = true,
            // refit with rounds scaled by (train + val) / train
            "refits" => {
                o.refit = true;
                o.scale_rounds = true;
            }
            "loccal" => o.loccal = true,
            "rcal" => o.rcal = true,
            "oracle" => o.oracle = true,
            other => panic!("unknown variant token {other}"),
        }
    }
    o
}

fn model<D, S>(a: &Args, seed: u64, o: &Opts) -> NGBoost<D, S, HistogramLearner>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
{
    let mut m = NGBoost::<D, S, HistogramLearner>::with_full_options(
        a.n_est,
        a.lr,
        HistogramLearner::new(a.depth),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        Some(a.patience),
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    );
    if let Some(st) = o.staged {
        m.staged_fit = Some(st);
    }
    if o.nocal {
        m.auto_calibrate_scale = false;
    }
    m
}

/// Fit on (train, val); with `refit`, refit a fresh model on train ∪ val in
/// time order for the selected number of rounds. Returns (model, secs).
#[allow(clippy::too_many_arguments)]
fn fit<D, S>(
    a: &Args,
    seed: u64,
    o: &Opts,
    s: &Series,
    sp: &Split,
) -> (NGBoost<D, S, HistogramLearner>, f64)
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
{
    let (xtr, ytr) = take(s, &sp.tr);
    let (xva, yva) = take(s, &sp.va);
    let t = Instant::now();
    let mut m = model::<D, S>(a, seed, o);
    m.fit_with_validation(&xtr, &ytr, Some(&xva), Some(&yva), None, None)
        .expect("fit failed");
    if o.refit {
        // library helper (Cycle 17): same hyperparameters, the kept round
        // count (or scaled by the extra rows with `refits`), the first fit's
        // recalibration carried over
        let mut all: Vec<usize> = sp.tr.iter().chain(&sp.va).copied().collect();
        all.sort_unstable();
        let (xa, ya) = take(s, &all);
        let k = m.base_models.len().max(1);
        let rounds = if o.scale_rounds {
            (k as f64 * all.len() as f64 / sp.tr.len() as f64).round() as u32
        } else {
            k as u32
        };
        m = m
            .refit_full(&xa, &ya, None, Some(rounds))
            .expect("refit failed");
    }
    (m, t.elapsed().as_secs_f64())
}

fn half_diff(nll: &Array1<f64>) -> f64 {
    let h = nll.len() / 2;
    nll.slice(s![h..]).mean().unwrap() - nll.slice(s![..h]).mean().unwrap()
}

fn run_normal(a: &Args, seed: u64, o: &Opts, s: &Series, sp: &Split) -> Out {
    let (xtr, ytr) = take(s, &sp.tr);
    let (xva, yva) = take(s, &sp.va);
    let (xte, yte) = take(s, &sp.te);
    let mut val_nll = f64::NAN;
    let (mut p, n_iter, secs, train_nll) = if o.oracle {
        let t = s.truth.as_ref().expect("oracle needs a synthetic series");
        let mut p = Array2::zeros((sp.te.len(), 2));
        for (k, &i) in sp.te.iter().enumerate() {
            p[[k, 0]] = t.mu[i];
            p[[k, 1]] = t.shape[i].ln();
        }
        (p, 0, 0.0, f64::NAN)
    } else {
        let (m, secs) = fit::<Normal, LogScore>(a, seed, o, s, sp);
        val_nll = m.score(&xva, &yva);
        let tn = Scorable::<LogScore>::score(&m.pred_dist(&xtr), &ytr)
            .mean()
            .unwrap();
        let mut p = m.pred_param(&xte);
        if o.loccal {
            let pv = m.pred_param(&xva);
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..yva.len() {
                let w = (-2.0 * pv[[i, 1]]).exp();
                num += w * (yva[i] - pv[[i, 0]]);
                den += w;
            }
            let c = num / den;
            p.column_mut(0).mapv_inplace(|v| v + c);
        }
        (p, m.base_models.len(), secs, tn)
    };
    if !p.iter().all(|v| v.is_finite()) {
        p.mapv_inplace(|v| if v.is_finite() { v } else { 0.0 });
    }
    let d = Normal::from_params(&p);
    let nll = Scorable::<LogScore>::score(&d, &yte);
    let crps = Scorable::<CRPScore>::score(&d, &yte).mean().unwrap();
    let err = &yte - &d.loc;
    let mse = err.mapv(|e| e * e).mean().unwrap();
    let var = d.scale.mapv(|v| v * v).mean().unwrap();
    let cov = |alpha: f64| {
        let (lo, hi) = d.interval(alpha);
        (0..yte.len())
            .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
            .count() as f64
            / yte.len() as f64
    };
    Out {
        n_iter,
        secs,
        train_nll,
        crps,
        rmse: mse.sqrt(),
        cov50: cov(0.5),
        cov90: cov(0.1),
        calib: (mse / var).sqrt(),
        mean_sigma: d.scale.mean().unwrap(),
        extra: err.mean().unwrap(),
        val_nll,
        nll,
    }
}

/// Count metrics shared by Poisson and NB (randomised PIT coverage).
fn count_out<D: DistributionMethods + Scorable<LogScore> + Scorable<CRPScore>>(
    d: &D,
    yte: &Array1<f64>,
    seed: u64,
    mean_sigma: f64,
) -> (Array1<f64>, f64, f64, f64, f64, f64, f64) {
    let nll = Scorable::<LogScore>::score(d, yte);
    let crps = Scorable::<CRPScore>::score(d, yte).mean().unwrap();
    let mu = d.mean();
    let var = d.variance();
    let err = yte - &mu;
    let mse = err.mapv(|e| e * e).mean().unwrap();
    let disp = (0..yte.len())
        .map(|i| err[i] * err[i] / var[i])
        .sum::<f64>()
        / yte.len() as f64;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x917);
    let cdf = d.cdf(yte);
    let pmf = d.pdf(yte);
    let (mut c50, mut c90) = (0usize, 0usize);
    for i in 0..yte.len() {
        let v: f64 = rng.random_range(0.0..1.0);
        let u = (cdf[i] - pmf[i]) + v * pmf[i];
        c50 += (0.25..=0.75).contains(&u) as usize;
        c90 += (0.05..=0.95).contains(&u) as usize;
    }
    let _ = mean_sigma;
    (
        nll,
        crps,
        mse.sqrt(),
        c50 as f64 / yte.len() as f64,
        c90 as f64 / yte.len() as f64,
        disp.sqrt(),
        yte.sum() / mu.sum(),
    )
}

fn run_count(a: &Args, seed: u64, o: &Opts, s: &Series, sp: &Split, nb: bool) -> Out {
    let (xtr, ytr) = take(s, &sp.tr);
    let (xva, yva) = take(s, &sp.va);
    let (xte, yte) = take(s, &sp.te);
    let ncol = if nb { 2 } else { 1 };
    let (mut p, n_iter, secs, train_nll) = if o.oracle {
        let t = s.truth.as_ref().expect("oracle needs a synthetic series");
        let mut p = Array2::zeros((sp.te.len(), ncol));
        for (k, &i) in sp.te.iter().enumerate() {
            p[[k, 0]] = t.mu[i].ln();
            if nb {
                p[[k, 1]] = t.shape[i].min(1e6).ln();
            }
        }
        (p, 0, 0.0, f64::NAN)
    } else if nb {
        let (mut m, secs) = fit::<NegativeBinomial, LogScore>(a, seed, o, s, sp);
        if o.rcal && yva.len() >= 300 {
            let _ = m.calibrate_param_affine(&xva, &yva, 1, None);
        }
        let tn = Scorable::<LogScore>::score(&m.pred_dist(&xtr), &ytr)
            .mean()
            .unwrap();
        let mut p = m.pred_param(&xte);
        if o.loccal {
            // shift of log μ minimising the val NLL (ternary search)
            let pv = m.pred_param(&xva);
            let f = |c: f64| {
                let mut q = pv.clone();
                q.column_mut(0).mapv_inplace(|v| v + c);
                Scorable::<LogScore>::score(&NegativeBinomial::from_params(&q), &yva).sum()
            };
            let (mut lo, mut hi) = (-1.0f64, 1.0f64);
            for _ in 0..60 {
                let m1 = lo + (hi - lo) / 3.0;
                let m2 = hi - (hi - lo) / 3.0;
                if f(m1) < f(m2) { hi = m2 } else { lo = m1 }
            }
            let c = 0.5 * (lo + hi);
            p.column_mut(0).mapv_inplace(|v| v + c);
        }
        (p, m.base_models.len(), secs, tn)
    } else {
        let (m, secs) = fit::<Poisson, LogScore>(a, seed, o, s, sp);
        let tn = Scorable::<LogScore>::score(&m.pred_dist(&xtr), &ytr)
            .mean()
            .unwrap();
        let mut p = m.pred_param(&xte);
        if o.loccal {
            let c = (yva.sum() / m.predict(&xva).sum()).ln();
            p.column_mut(0).mapv_inplace(|v| v + c);
        }
        (p, m.base_models.len(), secs, tn)
    };
    let (nll, crps, rmse, cov50, cov90, calib, extra, ms) = if nb {
        let d = NegativeBinomial::from_params(&p);
        let ms = d.r.mapv(|r| r.min(1e6).ln()).mean().unwrap().exp();
        let r = count_out(&d, &yte, seed, ms);
        (r.0, r.1, r.2, r.3, r.4, r.5, r.6, ms)
    } else {
        let d = Poisson::from_params(&p);
        let ms = d.mean().mapv(f64::sqrt).mean().unwrap();
        let r = count_out(&d, &yte, seed, ms);
        (r.0, r.1, r.2, r.3, r.4, r.5, r.6, ms)
    };
    Out {
        n_iter,
        secs,
        train_nll,
        nll,
        crps,
        rmse,
        cov50,
        cov90,
        calib,
        mean_sigma: ms,
        extra,
        val_nll: f64::NAN,
    }
}

/// Dean–Lawless score statistic for NB2 over-dispersion of a Poisson fit,
/// T = Σ[(y−λ̂)² − y] / √(2 Σ λ̂²) (≈ N(0, 1) under Poisson with known λ).
fn dispersion_z(y: &Array1<f64>, lam: &Array1<f64>) -> f64 {
    let num: f64 = (0..y.len()).map(|i| (y[i] - lam[i]).powi(2) - y[i]).sum();
    let den = (2.0 * lam.mapv(|l| l * l).sum()).sqrt();
    num / den
}

/// `--dist countsel`: fit Poisson and NB once (shipped defaults, chrono
/// val), then report both and three selection rules. For the sel_* rows,
/// n_iter = 1 if NB was chosen (else 0) and train_nll = the Dean–Lawless
/// z of the Poisson fit on the validation rows.
fn run_countsel(a: &Args, seed: u64, s: &Series, sp: &Split) -> Vec<(String, Out)> {
    let o = parse_variant("chrono");
    let (xva, yva) = take(s, &sp.va);
    let (xte, yte) = take(s, &sp.te);
    let (pm, ps) = fit::<Poisson, LogScore>(a, seed, &o, s, sp);
    let (nm, ns) = fit::<NegativeBinomial, LogScore>(a, seed, &o, s, sp);
    let zval = dispersion_z(&yva, &pm.predict(&xva));
    let (vp, vn) = (pm.score(&xva, &yva), nm.score(&xva, &yva));
    let pois = {
        let d = Poisson::from_params(&pm.pred_param(&xte));
        let ms = d.mean().mapv(f64::sqrt).mean().unwrap();
        let r = count_out(&d, &yte, seed, ms);
        Out {
            n_iter: pm.base_models.len(),
            secs: ps,
            train_nll: zval,
            nll: r.0,
            crps: r.1,
            rmse: r.2,
            cov50: r.3,
            cov90: r.4,
            calib: r.5,
            mean_sigma: ms,
            extra: r.6,
            val_nll: f64::NAN,
        }
    };
    let nb = {
        let d = NegativeBinomial::from_params(&nm.pred_param(&xte));
        let ms = d.r.mapv(|r| r.min(1e6).ln()).mean().unwrap().exp();
        let r = count_out(&d, &yte, seed, ms);
        Out {
            n_iter: nm.base_models.len(),
            secs: ns,
            train_nll: zval,
            nll: r.0,
            crps: r.1,
            rmse: r.2,
            cov50: r.3,
            cov90: r.4,
            calib: r.5,
            mean_sigma: ms,
            extra: r.6,
            val_nll: f64::NAN,
        }
    };
    let pick = |use_nb: bool| -> Out {
        let src = if use_nb { &nb } else { &pois };
        Out {
            n_iter: use_nb as usize,
            secs: ps + ns,
            train_nll: zval,
            nll: src.nll.clone(),
            crps: src.crps,
            rmse: src.rmse,
            cov50: src.cov50,
            cov90: src.cov90,
            calib: src.calib,
            mean_sigma: src.mean_sigma,
            extra: src.extra,
            val_nll: f64::NAN,
        }
    };
    // the library API (`CountModel::fit_auto`) must reproduce sel_z95
    let (xtr, ytr) = take(s, &sp.tr);
    let template = model::<Poisson, LogScore>(a, seed, &o);
    let (cm, zlib) = ngboost_rs::CountModel::fit_auto(&template, &xtr, &ytr, &xva, &yva)
        .expect("fit_auto failed");
    assert!(
        (zlib - zval).abs() <= 1e-12 * zval.abs().max(1.0),
        "fit_auto z differs"
    );
    assert_eq!(cm.is_negative_binomial(), zval > 1.645);
    let lib_nll = cm.score(&xte, &yte);
    let chosen = if zval > 1.645 { &nb } else { &pois };
    assert!(
        (lib_nll - chosen.nll.mean().unwrap()).abs() < 1e-12,
        "fit_auto NLL {lib_nll} vs {}",
        chosen.nll.mean().unwrap()
    );
    vec![
        ("sel_nll".into(), pick(vn < vp)),
        ("sel_z95".into(), pick(zval > 1.645)),
        ("sel_z99".into(), pick(zval > 2.326)),
        ("pois".into(), pois),
        ("nb".into(), nb),
    ]
}

fn main() {
    let a = parse_args();
    println!(
        "dataset,seed,variant,n_iter,secs,train_nll,nll,crps,rmse,cov50,cov90,calib,mean_sigma,extra,extra2,val_nll"
    );
    for ds in &a.datasets {
        let synthetic = ds.starts_with("ts") || ds.starts_with("iid_");
        for k in 0..a.seeds {
            let seed = a.seed0 + k;
            let s = if ds.starts_with("iid_") {
                synth_iid(ds, a.n, seed)
            } else if synthetic {
                synth(ds, a.n, seed)
            } else {
                real(ds)
            };
            if a.dist == "countsel" {
                let sp = split(
                    s.y.len(),
                    synthetic,
                    seed,
                    false,
                    seed,
                    a.te_frac,
                    a.va_frac,
                );
                for (v, out) in run_countsel(&a, seed, &s, &sp) {
                    print_row(ds, seed, &v, &out);
                }
                continue;
            }
            for v in &a.variants {
                let o = parse_variant(v);
                if o.oracle && s.truth.is_none() {
                    continue;
                }
                let sp = split(
                    s.y.len(),
                    synthetic,
                    seed,
                    o.random_val,
                    seed,
                    a.te_frac,
                    a.va_frac,
                );
                let out = match a.dist.as_str() {
                    "normal" => run_normal(&a, seed, &o, &s, &sp),
                    "poisson" => run_count(&a, seed, &o, &s, &sp, false),
                    "negbin" => run_count(&a, seed, &o, &s, &sp, true),
                    other => panic!("unknown dist {other}"),
                };
                print_row(ds, seed, v, &out);
            }
        }
    }
}

fn print_row(ds: &str, seed: u64, v: &str, out: &Out) {
    println!(
        "{ds},{seed},{v},{},{:.3},{:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.4},{:.5},{:.5},{:.5}",
        out.n_iter,
        out.secs,
        out.train_nll,
        out.nll.mean().unwrap(),
        out.crps,
        out.rmse,
        out.cov50,
        out.cov90,
        out.calib,
        out.mean_sigma,
        out.extra,
        half_diff(&out.nll),
        out.val_nll,
    );
}
