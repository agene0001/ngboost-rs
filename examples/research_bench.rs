//! Research benchmark harness (see `research/ngboost/`). NOT part of the
//! public API. Trains `NGBoost<Normal, LogScore, HistogramLearner>` variants
//! under an identical protocol and prints out-of-sample probabilistic
//! metrics as CSV rows plus a paired summary against the first variant.
//!
//! Protocol per (dataset, seed): shuffle rows with StdRng(seed), split
//! 70/10/20 train/val/test, early-stop on val (patience `--patience`), score
//! the best iteration on test. Identical hyperparameters for every variant.
//!
//!   cargo run --release --features accelerate --example research_bench -- \
//!     --variants base,fw --datasets het_a,kin8nm --seeds 5 \
//!     --lr 0.05 --n-est 2000 --patience 50 --depth 3 --max-rows 12000

use ndarray::{Array1, Array2};
use ngboost_rs::dist::NormalMixture;
use ngboost_rs::dist::{
    Bernoulli, Categorical, ClassificationDistn, Distribution, DistributionMethods, Laplace,
    LogNormal, NegativeBinomial, Normal, NormalMixture2, NormalMixture3, Poisson, StudentT,
};
use ngboost_rs::dist::{Gamma, HalfNormal, Weibull};
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::learners::{HistTreeNode, SerializableTrainedLearner};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::Score;
use ngboost_rs::scores::{CRPScore, LogScore, Scorable};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::BTreeMap;
use std::time::Instant;

/// Dataset currently being run (for side outputs such as NGB_POINT_CSV).
static CURRENT_DS: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

const DATA_DIR: &str = "../ngboost/data/uci";

struct Args {
    variants: Vec<String>,
    datasets: Vec<String>,
    seeds: u64,
    seed0: u64,
    lr: f64,
    n_est: u32,
    patience: u32,
    depth: usize,
    max_rows: usize,
    n_synth: usize,
    val_rows: usize,
    dist: String,
    y_shift: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        variants: vec!["base".into()],
        datasets: vec!["het_a".into()],
        seeds: 3,
        seed0: 0,
        lr: 0.05,
        n_est: 2000,
        patience: 50,
        depth: 3,
        max_rows: usize::MAX,
        n_synth: 3000,
        val_rows: usize::MAX,
        dist: "normal".into(),
        y_shift: 0.0,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < argv.len() {
        let v = &argv[i + 1];
        match argv[i].as_str() {
            "--variants" => a.variants = v.split(',').map(String::from).collect(),
            "--datasets" => a.datasets = v.split(',').map(String::from).collect(),
            "--seeds" => a.seeds = v.parse().unwrap(),
            "--seed0" => a.seed0 = v.parse().unwrap(),
            "--lr" => a.lr = v.parse().unwrap(),
            "--n-est" => a.n_est = v.parse().unwrap(),
            "--patience" => a.patience = v.parse().unwrap(),
            "--depth" => a.depth = v.parse().unwrap(),
            "--max-rows" => a.max_rows = v.parse().unwrap(),
            "--n-synth" => a.n_synth = v.parse().unwrap(),
            "--val-rows" => a.val_rows = v.parse().unwrap(),
            "--dist" => a.dist = v.clone(),
            "--y-shift" => a.y_shift = v.parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
        i += 2;
    }
    // an unpaired trailing argument used to be silently ignored (e.g. a zsh
    // `$VAR` holding several flags is passed as ONE argument)
    assert!(
        i == argv.len(),
        "unparsed trailing argument(s): {:?}",
        &argv[i..]
    );
    a
}

// ---------------------------------------------------------------- datasets

fn load_csv(path: &str, target: &str) -> (Array2<f64>, Array1<f64>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();
    let t = header
        .iter()
        .position(|h| h == target)
        .expect("target column");
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let v: Vec<f64> = l.split(',').map(|s| s.trim().parse().unwrap()).collect();
        rows.push(v);
    }
    let n = rows.len();
    let p = header.len() - 1;
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
    /// Standard Student-t with `df` degrees of freedom.
    fn t(&mut self, df: usize) -> f64 {
        let z = self.z();
        let chi: f64 = (0..df).map(|_| self.z().powi(2)).sum();
        z / (chi / df as f64).sqrt()
    }
}

/// Synthetic DGPs with known noise structure. Returns (X, y, true_sigma).
fn synth(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5eed_0000);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let mut g = Gauss(&mut rng);
    let mut y = Array1::zeros(n);
    let mut sig = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        let mu_a = 2.0 * r[0].sin() + r[1] * r[1] - r[2] * r[3];
        let (mu, sigma, noise) = match name {
            // heteroscedastic, variance driven by features the mean ignores
            "het_a" => {
                let s = (-0.5 + 0.6 * r[4] + 0.4 * (2.0 * r[5]).sin()).exp();
                (mu_a, s, g.z())
            }
            // heteroscedastic, variance shares the mean's dominant feature
            "het_b" => {
                let mu = 3.0 * (2.0 * r[0]).sin() + r[1];
                let s = (-1.0 + 1.0 * r[0]).exp();
                (mu, s, g.z())
            }
            // homoscedastic control
            "homo" => (mu_a, 1.0, g.z()),
            // truly log-normal, skewed, positive target with log-space
            // heteroscedasticity (misspecification test for the Normal head)
            "lognorm_het" => {
                let s = (-1.0 + 0.5 * r[4]).exp();
                let ly = 1.0 + 0.3 * mu_a + s * g.z();
                y[i] = ly.exp();
                sig[i] = s;
                continue;
            }
            // conditionally bimodal (protein-like): x-dependent mixing between
            // a tight low mode and a wide high mode
            "bimodal" => {
                let pz = 1.0 / (1.0 + (-(1.5 * r[0] - 0.5 * r[1] * r[1] + 0.8)).exp());
                let u: f64 = g.0.random_range(0.0..1.0);
                y[i] = if u < pz {
                    2.0 + 0.5 * r[2] + 0.6 * g.z()
                } else {
                    12.0 + 2.0 * (r[3]).sin() + 2.5 * g.z()
                };
                sig[i] = f64::NAN;
                continue;
            }
            // three modes with x-dependent mixing (multinomial logit)
            "trimodal" => {
                let l = [1.2 * r[0], -0.8 * r[0] + 0.8 * r[1], 0.0];
                let mx = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = l.iter().map(|v| (v - mx).exp()).collect();
                let tot: f64 = w.iter().sum();
                let u: f64 = g.0.random_range(0.0..1.0) * tot;
                y[i] = if u < w[0] {
                    0.0 + 0.5 * r[2] + 0.5 * g.z()
                } else if u < w[0] + w[1] {
                    8.0 + 1.5 * g.z()
                } else {
                    16.0 + 1.5 * r[3].sin() + 1.0 * g.z()
                };
                sig[i] = f64::NAN;
                continue;
            }
            // exact zeros with x-dependent probability plus a log-normal
            // positive part (intermittent demand / rainfall / claims)
            "zi_lognorm" => {
                let pz = 1.0 / (1.0 + (-(1.2 * r[0] - 0.8 * r[1] * r[1] + 0.3)).exp());
                let u: f64 = g.0.random_range(0.0..1.0);
                let s = (-0.7 + 0.4 * r[4]).exp();
                y[i] = if u < pz {
                    0.0
                } else {
                    (1.0 + 0.3 * mu_a + s * g.z()).exp()
                };
                sig[i] = f64::NAN;
                continue;
            }
            // exact zeros inside the support of a heteroscedastic Normal
            "zi_normal" => {
                let pz = 1.0 / (1.0 + (-(1.2 * r[0] - 0.8 * r[1] * r[1] + 0.3)).exp());
                let u: f64 = g.0.random_range(0.0..1.0);
                let s = (-0.5 + 0.6 * r[4] + 0.4 * (2.0 * r[5]).sin()).exp();
                let z = g.z();
                y[i] = if u < pz { 0.0 } else { 1.0 + mu_a + s * z };
                sig[i] = f64::NAN;
                continue;
            }
            // misspecified heavy tails: t(4) noise scaled like het_a
            "tnoise" => {
                let s = (-0.5 + 0.6 * r[4] + 0.4 * (2.0 * r[5]).sin()).exp();
                (mu_a, s, g.t(4) / 2f64.sqrt()) // t4 var = 2 -> unit variance
            }
            _ => panic!("unknown synthetic dataset {name}"),
        };
        y[i] = mu + sigma * noise;
        sig[i] = sigma;
    }
    (x, y, sig)
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

/// Count DGPs. `pois_*`: y ~ Poisson(λ(x)); `nb_*`: Gamma–Poisson mixture
/// (negative binomial, Var = λ + λ²/r) fitted with a Poisson head =
/// overdispersion / misspecification. Returns (X, y, true λ).
fn synth_count(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xc0_0a7);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let mut g = Gauss(&mut rng);
    let mut y = Array1::zeros(n);
    let mut lam = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        // "points-like" nonlinear rate, roughly 1–40
        let nl = (2.0 + 0.8 * (2.0 * r[0]).sin() + 0.25 * r[1] * r[1] - 0.3 * r[2].abs()
            + 0.2 * r[3] * r[4])
            .exp();
        // "strikeouts-like" low rate, roughly 3–10
        let low = (1.75 + 0.25 * r[0] + 0.15 * (2.0 * r[1]).sin()).exp();
        let (mu, disp): (f64, Option<f64>) = match name {
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
        lam[i] = mu;
    }
    (x, y, lam)
}

/// Binary DGPs: y ~ Bernoulli(σ(logit(x))). Returns (X, y, p_true).
fn synth_bern(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xbe57);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let mut y = Array1::zeros(n);
    let mut pt = Array1::zeros(n);
    for i in 0..n {
        let r = x.row(i);
        let logit: f64 = match name {
            "bern_lin" => 0.5 + 1.2 * r[0] - 0.8 * r[1] + 0.5 * r[2],
            "bern_nl" => -1.0 + 1.5 * (2.0 * r[0]).sin() + 0.6 * r[1] * r[1] + 0.4 * r[2] * r[3],
            // imbalanced (~12 % positives): p(1−p) varies most across rows
            "bern_imb" => -2.5 + 1.0 * r[0] + 0.8 * (2.0 * r[1]).sin(),
            // ~6 % positives, nonlinear
            "bern_imb6" => -3.3 + 1.0 * r[0] + 0.8 * (2.0 * r[1]).sin() + 0.3 * r[2] * r[3],
            // ~2–3 % positives, nonlinear
            "bern_rare" => -4.5 + 1.2 * r[0] + 0.8 * (2.0 * r[1]).sin() + 0.4 * r[2] * r[2],
            _ => panic!("unknown binary dataset {name}"),
        };
        let pr = 1.0 / (1.0 + (-logit).exp());
        pt[i] = pr;
        y[i] = if rng.random_range(0.0..1.0f64) < pr {
            1.0
        } else {
            0.0
        };
    }
    (x, y, pt)
}

const SURV_DIR: &str = "../ngboost/data/surv";

/// Parse one CSV cell: numbers, booleans and sex codes; anything else → NaN
/// (missing). Handles `"`-quoting and padding.
fn cell(v: &str) -> f64 {
    let v = v.trim().trim_matches('"').trim();
    match v {
        "True" | "true" | "M" | "male" => 1.0,
        "False" | "false" | "F" | "female" => 0.0,
        _ => v.parse::<f64>().unwrap_or(f64::NAN),
    }
}

/// Real binary tasks from the survival CSVs (event indicator as the label;
/// time and post-outcome columns are dropped to avoid leakage).
fn load_real_binary(name: &str) -> (Array2<f64>, Array1<f64>) {
    let (file, target, feats): (&str, &str, &[&str]) = match name {
        "sprint_cvd" => (
            "sprint-cut.csv",
            "cvd",
            &[
                "INTENSIVE",
                "AGE",
                "FEMALE",
                "RACE_BLACK",
                "hisp",
                "SBP.y",
                "DBP.y",
                "N_AGENTS",
                "currentsmoker",
                "formersmoker",
                "ASPIRIN",
                "STATIN",
                "SCREAT",
                "CHR",
                "HDL",
                "TRR",
                "BMI",
            ],
        ),
        "flchain_death" => (
            "flchain.csv",
            "death",
            &[
                "age",
                "sex",
                "sample.yr",
                "kappa",
                "lambda",
                "flc.grp",
                "creatinine",
                "mgus",
            ],
        ),
        "support_hospdead" => (
            "support2.csv",
            "hospdead",
            &[
                "age", "sex", "num.co", "edu", "scoma", "avtisst", "sps", "aps", "diabetes",
                "dementia", "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea",
                "sod", "ph", "glucose", "bun", "urine", "adlsc",
            ],
        ),
        _ => panic!("unknown real binary dataset {name}"),
    };
    let text = std::fs::read_to_string(format!("{SURV_DIR}/{file}")).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|h| h.trim().trim_matches('"').to_string())
        .collect();
    let col = |c: &str| {
        header
            .iter()
            .position(|h| h == c)
            .unwrap_or_else(|| panic!("{file}: no column {c}"))
    };
    let t = col(target);
    let fi: Vec<usize> = feats.iter().map(|f| col(f)).collect();
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let mut parts: Vec<&str> = l.split(',').collect();
        // support2 rows carry an unnamed leading index column the header lacks
        if parts.len() == header.len() + 1 {
            parts.remove(0);
        }
        if parts.len() != header.len() {
            continue; // quoted commas (support2 income) — skip malformed rows
        }
        let yv = cell(parts[t]);
        if !yv.is_finite() {
            continue;
        }
        ys.push(if yv > 0.5 { 1.0 } else { 0.0 });
        xs.extend(fi.iter().map(|&j| cell(parts[j])));
    }
    let n = ys.len();
    (
        Array2::from_shape_vec((n, fi.len()), xs).unwrap(),
        Array1::from(ys),
    )
}

/// Real 4-class tasks from the survival CSVs. support2 `dzclass` (disease
/// class from physiology; the finer `dzgroup` is excluded as leakage) and
/// flchain cause-of-death `chapter` among deaths (Circulatory / Neoplasms /
/// Respiratory / other; time columns excluded).
fn load_real_multiclass(name: &str) -> (Array2<f64>, Array1<f64>) {
    let (file, target, feats, classes): (&str, &str, &[&str], &[&str]) = match name {
        "support_dzclass" => (
            "support2.csv",
            "dzclass",
            &[
                "age", "sex", "num.co", "edu", "scoma", "avtisst", "sps", "aps", "diabetes",
                "dementia", "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea",
                "sod", "ph", "glucose", "bun", "urine", "adlsc",
            ],
            &["ARF/MOSF", "COPD/CHF/Cirrhosis", "Cancer", "Coma"],
        ),
        "flchain_cause" => (
            "flchain.csv",
            "chapter",
            &[
                "age",
                "sex",
                "sample.yr",
                "kappa",
                "lambda",
                "flc.grp",
                "creatinine",
                "mgus",
            ],
            &["Circulatory", "Neoplasms", "Respiratory"],
        ),
        "support_dzgroup" => (
            "support2.csv",
            "dzgroup",
            &[
                "age", "sex", "num.co", "edu", "scoma", "avtisst", "sps", "aps", "diabetes",
                "dementia", "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea",
                "sod", "ph", "glucose", "bun", "urine", "adlsc",
            ],
            &[
                "ARF/MOSF w/Sepsis",
                "CHF",
                "COPD",
                "Lung Cancer",
                "MOSF w/Malig",
                "Coma",
                "Colon Cancer",
                "Cirrhosis",
            ],
        ),
        _ => panic!("unknown real multiclass dataset {name}"),
    };
    let text = std::fs::read_to_string(format!("{SURV_DIR}/{file}")).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|h| h.trim().trim_matches('"').to_string())
        .collect();
    let col = |c: &str| header.iter().position(|h| h == c).unwrap();
    let t = col(target);
    let death = header.iter().position(|h| h == "death");
    let fi: Vec<usize> = feats.iter().map(|f| col(f)).collect();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for l in lines {
        let mut parts: Vec<&str> = l.split(',').collect();
        if parts.len() == header.len() + 1 {
            parts.remove(0);
        }
        if parts.len() != header.len() {
            continue;
        }
        if name == "flchain_cause" && death.map(|d| parts[d].trim() != "1").unwrap_or(true) {
            continue; // cause of death is only defined for deaths
        }
        let lab = parts[t].trim().trim_matches('"');
        if lab.is_empty() || lab == "NA" {
            continue;
        }
        // listed classes map to 0..; anything else is the last ("other") class
        let k = classes
            .iter()
            .position(|c| *c == lab)
            .unwrap_or(classes.len());
        if name.starts_with("support_dz") && k == classes.len() {
            continue;
        }
        ys.push(k as f64);
        xs.extend(fi.iter().map(|&j| cell(parts[j])));
    }
    let n = ys.len();
    (
        Array2::from_shape_vec((n, fi.len()), xs).unwrap(),
        Array1::from(ys),
    )
}

/// Real regression targets from the survival CSVs, chosen for their shape
/// (research/ngboost Cycle 12). Censored follow-up times are used as plain
/// targets: a test of whether a head can represent multimodality, not
/// survival analysis. Outcome-revealing columns (death flags, length of stay,
/// the dataset's own survival estimates) are excluded.
///  - `flchain_futime`: follow-up days (early deaths + administrative censoring)
///  - `support_logdtime`: ln(days to death or censoring)
///  - `support_logcost`: ln(total cost), costs > 0 — skewed but unimodal control
fn load_real_regression(name: &str) -> (Array2<f64>, Array1<f64>) {
    let physio: &[&str] = &[
        "age", "sex", "num.co", "edu", "scoma", "avtisst", "sps", "aps", "diabetes", "dementia",
        "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili", "crea", "sod", "ph",
        "glucose", "bun", "urine", "adlsc",
    ];
    let (file, target, feats, tf): (&str, &str, &[&str], fn(f64) -> f64) = match name {
        "flchain_futime" => (
            "flchain.csv",
            "futime",
            &[
                "age",
                "sex",
                "sample.yr",
                "kappa",
                "lambda",
                "flc.grp",
                "creatinine",
                "mgus",
            ],
            |v| v,
        ),
        "support_logdtime" => ("support2.csv", "d.time", physio, |v| v.ln()),
        // raw total cost > 0 (LogNormal / Laplace / StudentT on the y scale)
        "support_cost" => ("support2.csv", "totcst", physio, |v| {
            if v > 0.0 { v } else { f64::NAN }
        }),
        // imputed activities-of-daily-living score: 34 % exact zeros plus a
        // continuous part (and integer atoms where it was observed)
        "support_adlsc" => ("support2.csv", "adlsc", &physio[..physio.len() - 1], |v| v),
        // raw total micro-cost: 1.8 % exact zeros plus a skewed positive part
        // (the 2 negative accounting entries are dropped)
        "support_totmcst" => ("support2.csv", "totmcst", physio, |v| {
            if v >= 0.0 { v } else { f64::NAN }
        }),
        // 17 exact-zero costs form a point mass a continuous mixture component
        // collapses onto (NaN NLL); non-positive costs are excluded
        "support_logcost" => ("support2.csv", "totcst", physio, |v| {
            if v > 0.0 { v.ln() } else { f64::NAN }
        }),
        _ => panic!("unknown real regression dataset {name}"),
    };
    let text = std::fs::read_to_string(format!("{SURV_DIR}/{file}")).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|h| h.trim().trim_matches('"').to_string())
        .collect();
    let col = |c: &str| header.iter().position(|h| h == c).unwrap();
    let t = col(target);
    let fi: Vec<usize> = feats.iter().map(|f| col(f)).collect();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for l in lines {
        let mut parts: Vec<&str> = l.split(',').collect();
        if parts.len() == header.len() + 1 {
            parts.remove(0);
        }
        if parts.len() != header.len() {
            continue;
        }
        let yv = tf(cell(parts[t]));
        if !yv.is_finite() {
            continue;
        }
        ys.push(yv);
        xs.extend(fi.iter().map(|&j| cell(parts[j])));
    }
    let n = ys.len();
    (
        Array2::from_shape_vec((n, fi.len()), xs).unwrap(),
        Array1::from(ys),
    )
}

/// K-class DGPs: y = argmax_k (f_k(x) + Gumbel). Returns (X, y as 0..K-1).
fn synth_cat(name: &str, n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let p = 10;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xca7);
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.random_range(-2.0..2.0));
    let k: usize = if name.starts_with("cat5") { 5 } else { 3 };
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let mut best = (0usize, f64::NEG_INFINITY);
        for c in 0..k {
            let cf = c as f64;
            let mut f = match name {
                "cat3_nl" | "cat5_nl" => {
                    (1.5 * r[0] + 0.7 * cf).sin() * 1.5
                        + 0.5 * r[1] * (cf - 1.0)
                        + 0.4 * r[2] * r[3] * if c % 2 == 0 { 1.0 } else { -1.0 }
                }
                // imbalanced: class 0 dominant
                "cat3_imb" => (if c == 0 { 2.0 } else { -0.5 }) + 1.2 * (r[0] + 0.5 * cf).sin(),
                _ => panic!("unknown categorical dataset {name}"),
            };
            let u: f64 = rng.random_range(1e-12..1.0);
            f += -(-u.ln()).ln(); // Gumbel
            if f > best.1 {
                best = (c, f);
            }
        }
        best.0 as f64
    });
    (x, y)
}

fn load_dataset(name: &str, seed: u64, a: &Args) -> (Array2<f64>, Array1<f64>) {
    if matches!(
        name,
        "support_dzclass" | "support_dzgroup" | "flchain_cause"
    ) {
        return load_real_multiclass(name);
    }
    if matches!(
        name,
        "flchain_futime"
            | "support_logdtime"
            | "support_logcost"
            | "support_adlsc"
            | "support_totmcst"
            | "support_cost"
    ) {
        return load_real_regression(name);
    }
    if matches!(name, "digits" | "wine") {
        return load_csv(&format!("research/ngboost/data/{name}.csv"), "target");
    }
    if name.starts_with("cat3") || name.starts_with("cat5") {
        return synth_cat(name, a.n_synth, seed);
    }
    if matches!(name, "sprint_cvd" | "flchain_death" | "support_hospdead") {
        return load_real_binary(name);
    }
    if name.starts_with("bern_") {
        let (x, y, _) = synth_bern(name, a.n_synth, seed);
        return (x, y);
    }
    if name.starts_with("pois_") || name.starts_with("nb_") {
        let (x, y, _) = synth_count(name, a.n_synth, seed);
        return (x, y);
    }
    match name {
        "kin8nm" => load_csv(&format!("{DATA_DIR}/kin8nm.csv"), "y"),
        "protein" => load_csv(&format!("{DATA_DIR}/protein.csv"), "RMSD"),
        _ => {
            let (x, y, _) = synth(name, a.n_synth, seed);
            (x, y)
        }
    }
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
    (
        x.select(ndarray::Axis(0), idx),
        y.select(ndarray::Axis(0), idx),
    )
}

// ----------------------------------------------------------------- metrics

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    n_iter: usize,
    secs: f64,
    train_nll: f64,
    nll: f64,
    crps: f64,
    rmse: f64,
    cov50: f64,
    cov90: f64,
    /// sqrt(mean(err^2) / mean(sigma^2)); > 1 = overconfident. Poisson: the
    /// dispersion index sqrt(mean((y-λ)²/λ)); > 1 = overdispersed.
    calib: f64,
    mean_sigma: f64,
    /// Distribution-specific: Poisson = bias ratio Σy/Σλ̂ on test; Normal =
    /// in-sample z-rms sqrt(mean z²) on train; Bernoulli = mean p̂ / mean y.
    extra: f64,
    /// Normal: h̄ = 1 − Π_t(1 − lr·s_t·L_t/n), the approximate mean hat-matrix
    /// diagonal of the μ trees (L_t = leaves); else NaN.
    extra2: f64,
}

fn coverage(dist: &Normal, y: &Array1<f64>, alpha: f64) -> f64 {
    let (lo, hi) = dist.interval(alpha);
    let mut c = 0usize;
    for i in 0..y.len() {
        if y[i] >= lo[i] && y[i] <= hi[i] {
            c += 1;
        }
    }
    c as f64 / y.len() as f64
}

fn evaluate(dist: &Normal, y: &Array1<f64>) -> (f64, f64, f64, f64, f64, f64, f64) {
    let nll = Scorable::<LogScore>::score(dist, y).mean().unwrap();
    let crps = Scorable::<CRPScore>::score(dist, y).mean().unwrap();
    let err = &dist.loc - y;
    let mse = err.mapv(|e| e * e).mean().unwrap();
    let var = dist.scale.mapv(|s| s * s).mean().unwrap();
    (
        nll,
        crps,
        mse.sqrt(),
        coverage(dist, y, 0.5),
        coverage(dist, y, 0.1),
        (mse / var).sqrt(),
        dist.scale.mean().unwrap(),
    )
}

type Model = NGBoost<Normal, LogScore, HistogramLearner>;

/// Post-hoc recalibration of the predicted log-scale, fitted on the
/// validation split, applied to the test distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PostHoc {
    None,
    /// NLL-optimal multiplicative factor: c² = mean(z²) on val (MLE of a
    /// common variance inflation), vs. the library's moment ratio
    /// mean(err²)/mean(σ²) in `calibrate_uncertainty`.
    Mle,
    /// Affine map log σ' = a + b·log σ, (a, b) by Newton on the val NLL
    /// (convex in (a, b)). b < 1 shrinks over-dispersed σ̂.
    Affine,
}

fn fit_affine(dist: &Normal, y: &Array1<f64>) -> (f64, f64) {
    let r: Vec<f64> = (0..y.len()).map(|i| y[i] - dist.loc[i]).collect();
    let l: Vec<f64> = dist.log_scale.to_vec();
    let (mut a, mut b) = (0.0f64, 1.0f64);
    let obj = |a: f64, b: f64| -> f64 {
        r.iter()
            .zip(&l)
            .map(|(&ri, &li)| {
                let u = a + b * li;
                u + 0.5 * ri * ri * (-2.0 * u).exp()
            })
            .sum::<f64>()
    };
    let mut f = obj(a, b);
    for _ in 0..50 {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&ri, &li) in r.iter().zip(&l) {
            let u = a + b * li;
            let w = ri * ri * (-2.0 * u).exp();
            ga += 1.0 - w;
            gb += li * (1.0 - w);
            haa += 2.0 * w;
            hab += 2.0 * w * li;
            hbb += 2.0 * w * li * li;
        }
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-300 {
            break;
        }
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        let mut t = 1.0;
        loop {
            let (na, nb) = (a - t * da, b - t * db);
            let nf = obj(na, nb);
            if nf <= f {
                a = na;
                b = nb;
                let done = (f - nf).abs() < 1e-12 * f.abs().max(1.0);
                f = nf;
                if done {
                    return (a, b);
                }
                break;
            }
            t *= 0.5;
            if t < 1e-8 {
                return (a, b);
            }
        }
    }
    (a, b)
}

fn apply_posthoc(kind: PostHoc, val: &Normal, yva: &Array1<f64>, test: &Normal) -> Normal {
    match kind {
        PostHoc::None => test.clone(),
        PostHoc::Mle => {
            let z2 = (0..yva.len())
                .map(|i| ((yva[i] - val.loc[i]) / val.scale[i]).powi(2))
                .sum::<f64>()
                / yva.len() as f64;
            let c = z2.sqrt();
            Normal::new(test.loc.clone(), &test.scale * c)
        }
        PostHoc::Affine => {
            let (a, b) = fit_affine(val, yva);
            let scale = test.log_scale.mapv(|l| (a + b * l).exp());
            Normal::new(test.loc.clone(), scale)
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum LibCalib {
    None,
    Moment,
    Affine,
}

fn build_model(variant: &str, a: &Args, seed: u64) -> (Model, LibCalib, PostHoc) {
    let mut m = Model::with_full_options(
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
        0.1, // unused: explicit validation set is passed
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    );
    // research baseline = pre-1.2 behaviour: no automatic recalibration
    m.auto_calibrate_scale = false;
    let mut calibrate = LibCalib::None;
    let mut posthoc = PostHoc::None;
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            "mle" => posthoc = PostHoc::Mle,
            "affine" => posthoc = PostHoc::Affine,
            "fw" => m.fisher_weighted_learner = true,
            "calib" => calibrate = LibCalib::Moment,
            "libaffine" => calibrate = LibCalib::Affine,
            "auto" => m.auto_calibrate_scale = true,
            // select β ∈ {0, 0.5, 1} by validation score (realistic user procedure)
            "fwsel" => m.fisher_weight_power = Some(f64::NAN),
            t if t.starts_with("fwp") => {
                // Fisher-weight power beta: weights = F_jj^beta (0 = base, 1 = fw)
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            t if t.starts_with("gs") => {
                m.line_search_method = LineSearchMethod::GoldenSection {
                    max_iters: t[2..].parse().unwrap_or(20),
                };
            }
            t if t.starts_with("tik") => m.tikhonov_reg = t[3..].parse().expect("tik<reg>"),
            t if t.starts_with("cap") => {
                m.fisher_weight_cap = Some(t[3..].parse().expect("cap<kappa>"))
            }
            other => panic!("unknown variant token {other}"),
        }
    }
    (m, calibrate, posthoc)
}

fn run_one(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    let (mut m, calibrate, posthoc) = build_model(variant, a, seed);
    let t = Instant::now();
    if m.fisher_weight_power.is_some_and(f64::is_nan) {
        // "fwsel": fit β ∈ {0, 0.5, 1}, keep the best validation score
        let mut best: Option<(f64, Model)> = None;
        for beta in [0.0, 0.5, 1.0] {
            let (mut cand, _, _) = build_model(variant, a, seed);
            cand.fisher_weight_power = Some(beta);
            cand.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .expect("fit failed");
            let v = cand.score(xva, yva);
            if best.as_ref().is_none_or(|(bv, _)| v < *bv) {
                best = Some((v, cand));
            }
        }
        m = best.unwrap().1;
    } else {
        m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit failed");
    }
    match calibrate {
        LibCalib::None => {}
        LibCalib::Moment => {
            m.calibrate_uncertainty(xva, yva).unwrap();
        }
        LibCalib::Affine => {
            m.calibrate_scale_affine(xva, yva).unwrap();
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_dist = m.pred_dist(xtr);
    let train_nll = Scorable::<LogScore>::score(&train_dist, ytr)
        .mean()
        .unwrap();
    // leverage diagnostics: in-sample z-rms and the approximate mean hat
    // diagonal of the μ trees, h̄ = 1 − Π_t (1 − lr·s_t·L_t/n)
    let z_rms = (0..ytr.len())
        .map(|i| ((ytr[i] - train_dist.loc[i]) / train_dist.scale[i]).powi(2))
        .sum::<f64>()
        / ytr.len() as f64;
    let n_tr = xtr.nrows() as f64;
    let mut keep = 1.0f64;
    for (t, round) in m.base_models.iter().enumerate() {
        let leaves = match round[0].to_serializable() {
            Some(SerializableTrainedLearner::HistogramTree(h)) => count_leaves(&h.root),
            _ => 0,
        } as f64;
        let lr = m
            .effective_learning_rates
            .get(t)
            .copied()
            .unwrap_or(m.learning_rate);
        keep *= 1.0 - lr * m.scalings[t] * leaves / n_tr;
    }
    let h_bar = 1.0 - keep;
    let dist = apply_posthoc(posthoc, &m.pred_dist(xva), yva, &m.pred_dist(xte));
    let (nll, crps, rmse, cov50, cov90, calib, mean_sigma) = evaluate(&dist, yte);
    if std::env::var("NGB_CHECK_CRPS").is_ok() {
        eprintln!(
            "CRPS closed form {crps:.6} vs numeric {:.6}",
            numeric_crps(&dist, yte)
        );
    }
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps,
        rmse,
        cov50,
        cov90,
        calib,
        mean_sigma,
        extra: z_rms.sqrt(),
        extra2: h_bar,
    }
}

fn count_leaves(n: &HistTreeNode) -> usize {
    match n {
        HistTreeNode::Leaf { .. } => 1,
        HistTreeNode::Split { left, right, .. } => count_leaves(left) + count_leaves(right),
    }
}

/// CRPS by numerical integration, ∫ (F(t) − 1{t ≥ y})² dt, per row on a
/// 600-point midpoint grid spanning [min(y, q₁e-5), max(y, q₁₋₁e-5)].
/// Distribution-generic (uses only `cdf` and `ppf`).
fn numeric_crps<D: DistributionMethods>(dist: &D, y: &Array1<f64>) -> f64 {
    let n = y.len();
    let lo = dist.ppf(&Array1::from_elem(n, 1e-5));
    let hi = dist.ppf(&Array1::from_elem(n, 1.0 - 1e-5));
    let a: Array1<f64> = (0..n).map(|i| lo[i].min(y[i])).collect();
    let b: Array1<f64> = (0..n).map(|i| hi[i].max(y[i])).collect();
    const K: usize = 600;
    let h: Array1<f64> = (&b - &a) / K as f64;
    let mut acc = Array1::<f64>::zeros(n);
    for k in 0..K {
        let t = &a + &(&h * (k as f64 + 0.5));
        let f = dist.cdf(&t);
        for i in 0..n {
            let ind = if t[i] >= y[i] { 1.0 } else { 0.0 };
            acc[i] += (f[i] - ind).powi(2) * h[i];
        }
    }
    acc.mean().unwrap()
}

/// Distribution-generic runner (NLL, RMSE via mean(), coverage via interval(),
/// calib via variance(); CRPS reported as NaN). Tokens: base, auto, libaffine,
/// fw, fwp<β>.
/// Exact CRPS for heads that implement `Scorable<CRPScore>` (mixtures): the
/// numerical integrator degrades when a component's σ explodes.
fn exact_crps<D: Scorable<CRPScore>>(d: &D, y: &Array1<f64>) -> f64 {
    Scorable::<CRPScore>::score(d, y).mean().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_one_generic<D, S>(
    crps_fn: Option<fn(&D, &Array1<f64>) -> f64>,
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics
where
    D: Distribution + Scorable<S> + Scorable<LogScore> + DistributionMethods + Clone + 'static,
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
    m.auto_calibrate_scale = false;
    let mut libaffine = false;
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            "auto" => m.auto_calibrate_scale = true,
            "libaffine" => libaffine = true,
            "joint" => m.staged_fit = Some(false),
            "staged" => m.staged_fit = Some(true),
            "fw" => m.fisher_weighted_learner = true,
            t if t.starts_with("fwp") => {
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            t if t.starts_with("cap") => {
                m.fisher_weight_cap = Some(t[3..].parse().expect("cap<kappa>"))
            }
            other => panic!("unknown generic variant token {other}"),
        }
    }
    let t = Instant::now();
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    if libaffine {
        let col = D::scale_param_index().expect("distribution has no scale column");
        m.calibrate_param_affine(xva, yva, col, None).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_nll = Scorable::<LogScore>::score(&m.pred_dist(xtr), ytr)
        .mean()
        .unwrap();
    let dist = m.pred_dist(xte);
    let nll = Scorable::<LogScore>::score(&dist, yte).mean().unwrap();
    let mean = dist.mean();
    if let Ok(path) = std::env::var("NGB_POINT_CSV") {
        // point forecasts: mean / median / mode, RMSE and MAE
        let med = dist.median();
        let mode = dist.mode();
        let err = |p: &Array1<f64>, sq: bool| {
            (0..yte.len())
                .map(|i| {
                    if sq {
                        (p[i] - yte[i]).powi(2)
                    } else {
                        (p[i] - yte[i]).abs()
                    }
                })
                .sum::<f64>()
                / yte.len() as f64
        };
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(
            f,
            "{},{seed},{variant},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            CURRENT_DS.lock().unwrap(),
            err(&mean, true).sqrt(),
            err(&med, true).sqrt(),
            err(&mode, true).sqrt(),
            err(&mean, false),
            err(&med, false),
            err(&mode, false)
        )
        .unwrap();
    }
    let mse = (&mean - yte).mapv(|e| e * e).mean().unwrap();
    let var = dist.variance().mean().unwrap();
    let cov = |alpha: f64| {
        let (lo, hi) = dist.interval(alpha);
        (0..yte.len())
            .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
            .count() as f64
            / yte.len() as f64
    };
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps: match crps_fn {
            Some(f) => f(&dist, yte),
            None => numeric_crps(&dist, yte),
        },
        rmse: mse.sqrt(),
        cov50: cov(0.5),
        cov90: cov(0.1),
        calib: (mse / var).sqrt(),
        mean_sigma: dist.std().mean().unwrap(),
        // validation NLL of the final (truncated + calibrated) model — the
        // signal for choosing between heads / numbers of components
        extra: Scorable::<LogScore>::score(&m.pred_dist(xva), yva)
            .mean()
            .unwrap(),
        extra2: f64::NAN,
    }
}

/// Poisson head. Tokens: base, fw, fwp<β>, `ratio` (post-hoc: λ̂ × Σy_val/Σλ̂_val,
/// the closed-form score-optimal common shift of log λ), `rate` (library
/// `calibrate_param_affine` on column 0: log λ ← a + b·log λ on val).
/// Coverage via randomized PIT u = F(y−1) + v·p(y); calib = dispersion
/// index sqrt(mean((y−λ̂)²/λ̂)); extra = Σy/Σλ̂ on test.
fn run_one_poisson(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
    lam_true: Option<&Array1<f64>>,
) -> Metrics {
    let mut m = NGBoost::<Poisson, LogScore, HistogramLearner>::with_full_options(
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
    m.auto_calibrate_scale = false;
    let (mut ratio, mut rate, mut oracle) = (false, false, false);
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            "oracle" => oracle = true,
            "fw" => m.fisher_weighted_learner = true,
            t if t.starts_with("fwp") => {
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            "ratio" => ratio = true,
            "rate" => rate = true,
            other => panic!("unknown poisson variant token {other}"),
        }
    }
    let t = Instant::now();
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    if rate {
        m.calibrate_param_affine(xva, yva, 0, None).unwrap();
    }
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_nll = m.score(xtr, ytr);
    let mut log_rate = m.pred_param(xte);
    if oracle {
        // reference: the data-generating rate (attainable optimum for a Poisson head)
        let lt = lam_true.expect("oracle needs a synthetic count dataset");
        log_rate.column_mut(0).assign(&lt.mapv(f64::ln));
    }
    if ratio {
        let c = yva.sum() / m.predict(xva).sum();
        log_rate.column_mut(0).mapv_inplace(|v| v + c.ln());
    }
    let dist = Poisson::from_params(&log_rate);
    let nll = Scorable::<LogScore>::score(&dist, yte).mean().unwrap();
    let crps = Scorable::<CRPScore>::score(&dist, yte).mean().unwrap();
    let lam = dist.mean();
    let err = &lam - yte;
    let mse = err.mapv(|e| e * e).mean().unwrap();
    let disp = (0..yte.len())
        .map(|i| err[i] * err[i] / lam[i])
        .sum::<f64>()
        / yte.len() as f64;
    // randomized PIT
    let mut rng = StdRng::seed_from_u64(seed ^ 0x917);
    let cdf = dist.cdf(yte);
    let pmf = dist.pdf(yte);
    let (mut c50, mut c90) = (0usize, 0usize);
    for i in 0..yte.len() {
        let v: f64 = rng.random_range(0.0..1.0);
        let u = (cdf[i] - pmf[i]) + v * pmf[i];
        if (0.25..=0.75).contains(&u) {
            c50 += 1;
        }
        if (0.05..=0.95).contains(&u) {
            c90 += 1;
        }
    }
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps,
        rmse: mse.sqrt(),
        cov50: c50 as f64 / yte.len() as f64,
        cov90: c90 as f64 / yte.len() as f64,
        calib: disp.sqrt(),
        mean_sigma: lam.mapv(f64::sqrt).mean().unwrap(),
        extra: yte.sum() / lam.sum(),
        extra2: ytr.sum() / m.predict(xtr).sum(),
    }
}

/// Negative Binomial head (log μ, log r). Tokens: base, fw, fwp<β>, oracle
/// (true μ and the DGP's r; r = 1e6 for `pois_*`). calib = dispersion index
/// relative to the predicted variance, sqrt(mean((y−μ̂)²/v̂)); extra = Σy/Σμ̂.
fn run_one_negbin<S: Score>(
    variant: &str,
    a: &Args,
    seed: u64,
    ds: &str,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
    lam_true: Option<&Array1<f64>>,
) -> Metrics
where
    NegativeBinomial: Scorable<S>,
{
    let mut m = NGBoost::<NegativeBinomial, S, HistogramLearner>::with_full_options(
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
    m.auto_calibrate_scale = false;
    let (mut oracle, mut rcal) = (false, false);
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            "oracle" => oracle = true,
            "fw" => m.fisher_weighted_learner = true,
            t if t.starts_with("fwp") => {
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            t if t.starts_with("tik") => m.tikhonov_reg = t[3..].parse().expect("tik<reg>"),
            "staged" => m.staged_fit = Some(true),
            "joint" => m.staged_fit = Some(false),
            "rcal" => rcal = true,
            "auto" => m.auto_calibrate_scale = true,
            other => panic!("unknown negbin variant token {other}"),
        }
    }
    let t = Instant::now();
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    if rcal {
        // validation-fitted affine map on log r (the count analogue of P2)
        let _ = m.calibrate_param_affine(xva, yva, 1, None);
    }
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_nll = Scorable::<LogScore>::score(&m.pred_dist(xtr), ytr)
        .mean()
        .unwrap();
    let mut params = m.pred_param(xte);
    if oracle {
        let lt = lam_true.expect("oracle needs a synthetic count dataset");
        let r_true: f64 = match ds {
            "nb_over" => 4.0,
            "nb_low" => 10.0,
            _ => 1e6,
        };
        params.column_mut(0).assign(&lt.mapv(f64::ln));
        params.column_mut(1).fill(r_true.ln());
    }
    let dist = NegativeBinomial::from_params(&params);
    let nll = Scorable::<LogScore>::score(&dist, yte).mean().unwrap();
    let crps = Scorable::<CRPScore>::score(&dist, yte).mean().unwrap();
    let mu = dist.mean();
    let var = dist.variance();
    let err = &mu - yte;
    let mse = err.mapv(|e| e * e).mean().unwrap();
    let disp = (0..yte.len())
        .map(|i| err[i] * err[i] / var[i])
        .sum::<f64>()
        / yte.len() as f64;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x917);
    let cdf = dist.cdf(yte);
    let pmf = dist.pdf(yte);
    let (mut c50, mut c90) = (0usize, 0usize);
    for i in 0..yte.len() {
        let v: f64 = rng.random_range(0.0..1.0);
        let u = (cdf[i] - pmf[i]) + v * pmf[i];
        if (0.25..=0.75).contains(&u) {
            c50 += 1;
        }
        if (0.05..=0.95).contains(&u) {
            c90 += 1;
        }
    }
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps,
        rmse: mse.sqrt(),
        cov50: c50 as f64 / yte.len() as f64,
        cov90: c90 as f64 / yte.len() as f64,
        calib: disp.sqrt(),
        mean_sigma: dist.r.mapv(f64::ln).mean().unwrap().exp(), // geometric-mean r̂
        extra: yte.sum() / mu.sum(),
        extra2: ytr.sum() / m.predict(xtr).sum(),
    }
}

/// Bernoulli (= Categorical<2>) head. nll = log loss, crps = Brier, rmse = NaN,
/// cov50 = accuracy, cov90 = NaN, calib = ECE (10 equal-width bins),
/// extra = mean p̂ / mean y. Tokens: base, fw, fwp<β>, cap<κ>.
fn run_one_bernoulli(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    let mut m = NGBoost::<Bernoulli, LogScore, HistogramLearner>::with_full_options(
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
    m.auto_calibrate_scale = false;
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            "fw" => m.fisher_weighted_learner = true,
            t if t.starts_with("fwp") => {
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            t if t.starts_with("cap") => {
                m.fisher_weight_cap = Some(t[3..].parse().expect("cap<kappa>"))
            }
            other => panic!("unknown bernoulli variant token {other}"),
        }
    }
    let t = Instant::now();
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_nll = m.score(xtr, ytr);
    let nll = m.score(xte, yte);
    let probs = m.pred_dist(xte).class_probs();
    let p: Vec<f64> = (0..yte.len()).map(|i| probs[[i, 1]]).collect();
    let brier = (0..yte.len()).map(|i| (p[i] - yte[i]).powi(2)).sum::<f64>() / yte.len() as f64;
    let acc = (0..yte.len())
        .filter(|&i| (p[i] >= 0.5) == (yte[i] >= 0.5))
        .count() as f64
        / yte.len() as f64;
    let mut bins = vec![(0.0f64, 0.0f64, 0usize); 10];
    for i in 0..yte.len() {
        let b = ((p[i] * 10.0) as usize).min(9);
        bins[b].0 += p[i];
        bins[b].1 += yte[i];
        bins[b].2 += 1;
    }
    let ece = bins
        .iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.2 as f64 / yte.len() as f64) * ((b.0 - b.1) / b.2 as f64).abs())
        .sum::<f64>();
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps: brier,
        rmse: f64::NAN,
        cov50: acc,
        cov90: f64::NAN,
        calib: ece,
        mean_sigma: f64::NAN,
        extra: p.iter().sum::<f64>() / yte.sum(),
        extra2: {
            let pt = m.pred_dist(xtr).class_probs();
            pt.column(1).sum() / ytr.sum()
        },
    }
}

/// Categorical<K> head. nll = log loss, crps = multiclass Brier (Σ_k), cov50 =
/// accuracy, calib = top-label ECE (10 bins), extra = max_k |mean p̂_k − freq_k|.
/// Tokens: base, fwp<β> (explicit β uses the diagonal Fisher), icpt.
fn run_one_categorical<const K: usize>(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    let mut m = NGBoost::<Categorical<K>, LogScore, HistogramLearner>::with_full_options(
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
    for tok in variant.split('+') {
        match tok {
            "base" => {}
            "icpt" => m.refit_intercept = Some(true),
            "noicpt" => m.refit_intercept = Some(false),
            t if t.starts_with("fwp") => {
                m.fisher_weight_power = Some(t[3..].parse().expect("fwp<beta>"));
            }
            other => panic!("unknown categorical variant token {other}"),
        }
    }
    let t = Instant::now();
    m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    let secs = t.elapsed().as_secs_f64();
    let n_iter = m.base_models.len();
    let train_nll = m.score(xtr, ytr);
    let nll = m.score(xte, yte);
    let probs = m.pred_dist(xte).class_probs();
    let n = yte.len();
    let mut brier = 0.0;
    let mut correct = 0usize;
    let mut bins = vec![(0.0f64, 0.0f64, 0usize); 10];
    let mut mean_p = vec![0.0f64; K];
    let mut freq = vec![0.0f64; K];
    for i in 0..n {
        let yi = yte[i] as usize;
        let (mut arg, mut pmax) = (0usize, -1.0f64);
        for k in 0..K {
            let pk = probs[[i, k]];
            brier += (pk - if k == yi { 1.0 } else { 0.0 }).powi(2);
            mean_p[k] += pk / n as f64;
            if pk > pmax {
                pmax = pk;
                arg = k;
            }
        }
        freq[yi] += 1.0 / n as f64;
        if arg == yi {
            correct += 1;
        }
        let b = ((pmax * 10.0) as usize).min(9);
        bins[b].0 += pmax;
        bins[b].1 += if arg == yi { 1.0 } else { 0.0 };
        bins[b].2 += 1;
    }
    let ece = bins
        .iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.2 as f64 / n as f64) * ((b.0 - b.1) / b.2 as f64).abs())
        .sum::<f64>();
    let prior_gap = (0..K)
        .map(|k| (mean_p[k] - freq[k]).abs())
        .fold(0.0, f64::max);
    Metrics {
        n_iter,
        secs,
        train_nll,
        nll,
        crps: brier / n as f64,
        rmse: f64::NAN,
        cov50: correct as f64 / n as f64,
        cov90: f64::NAN,
        calib: ece,
        mean_sigma: f64::NAN,
        extra: prior_gap,
        extra2: f64::NAN,
    }
}

/// `NGBAutoMixtureRegressor` (K ∈ 1..=4). Tokens: `cold`, `warm` (split
/// growth), `misfit` (misfit growth), `greedy`. nll/crps(NaN)/rmse/coverage
/// as usual; extra = chosen K; mean_sigma = 1 if the chosen head is warm;
/// extra2 = total boosting rounds trained over
/// all candidates (the search cost); secs = wall time of the whole search.
fn run_one_automix(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    use ngboost_rs::{GrowStrategy, KSearch, NGBAutoMixtureRegressor};
    // `crps` token: CRPScore-trained candidates selected by validation CRPS
    macro_rules! run {
        ($m:expr) => {{
            let mut m = $m;
            m.base_learner = HistogramLearner::new(a.depth);
            m.early_stopping_rounds = Some(a.patience);
            m.random_state = Some(seed);
            for tok in variant.split('+') {
                match tok {
                    "crps" => {}
                    "cold" => m.search = KSearch::Cold,
                    "warm" => {
                        m.search = KSearch::Warm;
                        m.grow_strategy = GrowStrategy::Split;
                    }
                    "misfit" => {
                        m.search = KSearch::Warm;
                        m.grow_strategy = GrowStrategy::Misfit;
                    }
                    "both" => m.search = KSearch::Both,
                    "greedy" => m.greedy = true,
                    other => panic!("unknown automix token {other}"),
                }
            }
            let t = Instant::now();
            m.fit(xtr, ytr, xva, yva).expect("fit failed");
            let secs = t.elapsed().as_secs_f64();
            let nll = m.score(xte, yte);
            let pred = m.predict(xte);
            let rmse = (&pred - yte).mapv(|e| e * e).mean().unwrap().sqrt();
            let cov = |alpha: f64| {
                let (lo, hi) = m.interval(xte, alpha);
                (0..yte.len())
                    .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
                    .count() as f64
                    / yte.len() as f64
            };
            Metrics {
                n_iter: m.tried.iter().map(|r| r.3).sum(),
                secs,
                train_nll: m.score(xtr, ytr),
                nll,
                crps: m.crps(xte, yte),
                rmse,
                cov50: cov(0.5),
                cov90: cov(0.1),
                calib: f64::NAN,
                mean_sigma: if m.selected_is_warm() { 1.0 } else { 0.0 },
                extra: m.n_components().unwrap() as f64,
                extra2: m.tried.iter().map(|r| r.3).sum::<usize>() as f64,
            }
        }};
    }
    if variant.split('+').any(|t| t == "crps") {
        run!(NGBAutoMixtureRegressor::new_crps(a.n_est, a.lr))
    } else {
        run!(NGBAutoMixtureRegressor::new(a.n_est, a.lr))
    }
}

/// Cycle 13 probe: why do warm-grown mixtures beat the Normal head on
/// unimodal data? Every variant refits the same Normal head (automix K = 1:
/// early stopping on val, default scale recalibration) and then:
///   base    — nothing (extra = rounds, extra2 = val NLL)
///   cont    — continue the Normal head with early stopping
///   matched — continue the Normal head for exactly the rounds `warm2` adds,
///             no early stopping
///   sym2    — grow K = 2 by a symmetric split (δ = 0, σ × 1): the Normal
///             reparametrised, then continue boosting (control)
///   warm2   — grow K = 2 by the default split (δ = 0.8, σ × 0.6)
///   warm2d / warm2s — split with only the shift / only the shrink
///   l2      — μ-only boosting (NormalFixedVar, σ ≡ 1): the RMSE reachable
///             by the mean trees alone (NLL meaningless)
///   trace   — Normal without early stopping (2000 rounds, no recalibration);
///             n_iter = val-NLL-best stage, extra = val-RMSE-best stage,
///             rmse/nll = test at the val-NLL stage, extra2 = test RMSE and
///             mean_sigma = test NLL at the val-RMSE stage
/// For mixtures extra = NLL of the moment-matched Normal (same mean and
/// variance per row), extra2 = rounds added on top of the Normal head.
#[allow(clippy::too_many_arguments)]
fn run_one_normprobe(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    use ngboost_rs::{AutoMixtureModel, NGBAutoMixtureRegressor};
    fn build<D: Distribution + Scorable<LogScore> + Clone>(
        a: &Args,
        seed: u64,
        es: Option<u32>,
    ) -> NGBoost<D, LogScore, HistogramLearner> {
        let mut m = NGBoost::with_full_options(
            a.n_est,
            a.lr,
            HistogramLearner::new(a.depth),
            true,
            1.0,
            1.0,
            false,
            1.0,
            1e-4,
            es,
            0.1,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            Some(seed),
        );
        // Cycle 13 probes compare against the joint fit explicitly
        m.staged_fit = Some(false);
        m
    }
    let rmse_of = |p: &Array1<f64>, y: &Array1<f64>| (p - y).mapv(|e| e * e).mean().unwrap().sqrt();
    let mm_nll = |mean: &Array1<f64>, var: &Array1<f64>, y: &Array1<f64>| {
        (0..y.len())
            .map(|i| {
                0.5 * (std::f64::consts::TAU * var[i]).ln()
                    + (y[i] - mean[i]).powi(2) / (2.0 * var[i])
            })
            .sum::<f64>()
            / y.len() as f64
    };
    fn finish<D>(
        d: &D,
        dv: &D,
        yva: &Array1<f64>,
        yte: &Array1<f64>,
        n_iter: usize,
        secs: f64,
        extra: f64,
        extra2: f64,
    ) -> Metrics
    where
        D: Scorable<LogScore> + Scorable<CRPScore> + DistributionMethods,
    {
        let mean = d.mean();
        let mse = (&mean - yte).mapv(|e| e * e).mean().unwrap();
        let cov = |alpha: f64| {
            let (lo, hi) = d.interval(alpha);
            (0..yte.len())
                .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
                .count() as f64
                / yte.len() as f64
        };
        let _ = dv;
        let _ = yva;
        Metrics {
            n_iter,
            secs,
            train_nll: f64::NAN,
            nll: Scorable::<LogScore>::score(d, yte).mean().unwrap(),
            crps: exact_crps(d, yte),
            rmse: mse.sqrt(),
            cov50: cov(0.5),
            cov90: cov(0.1),
            calib: (mse / d.variance().mean().unwrap()).sqrt(),
            mean_sigma: d.std().mean().unwrap(),
            extra,
            extra2,
        }
    }
    let t = Instant::now();
    if variant == "scal" || variant == "l2scal" {
        // mean / median line-search scaling over the kept rounds
        let sc: Vec<f64> = if variant == "scal" {
            let mut m = build::<Normal>(a, seed, Some(a.patience));
            m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .unwrap();
            m.scalings.clone()
        } else {
            let mut m = build::<ngboost_rs::dist::NormalFixedVar>(a, seed, Some(a.patience));
            m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .unwrap();
            m.scalings.clone()
        };
        let mut v = sc.clone();
        v.sort_by(|x, y| x.total_cmp(y));
        let mut out = finish(
            &Normal::from_params(&Array2::zeros((1, 2))),
            &Normal::from_params(&Array2::zeros((1, 2))),
            yva,
            &Array1::zeros(1),
            sc.len(),
            0.0,
            sc.iter().sum::<f64>() / sc.len() as f64,
            v[v.len() / 2],
        );
        out.nll = f64::NAN;
        return out;
    }
    if variant == "l2" {
        let mut m = build::<ngboost_rs::dist::NormalFixedVar>(a, seed, Some(a.patience));
        m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit failed");
        let d = m.pred_dist(xte);
        let mut out = finish(
            &d,
            &d,
            yva,
            yte,
            m.base_models.len(),
            t.elapsed().as_secs_f64(),
            f64::NAN,
            f64::NAN,
        );
        out.rmse = rmse_of(&d.mean(), yte);
        return out;
    }
    if variant == "staged" {
        // the generic NGBoost::fit_staged via fit_with_validation
        let mut m = build::<Normal>(a, seed, Some(a.patience));
        m.staged_fit = Some(true);
        m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("staged fit failed");
        let (d, dv) = (m.pred_dist(xte), m.pred_dist(xva));
        let v = Scorable::<LogScore>::score(&dv, yva).mean().unwrap();
        return finish(
            &d,
            &dv,
            yva,
            yte,
            m.base_models.len(),
            t.elapsed().as_secs_f64(),
            v,
            f64::NAN,
        );
    }
    if variant == "lib2stage" {
        // the library's NGBoost::fit_two_stage (must reproduce twostage+fgls)
        let mut m = build::<Normal>(a, seed, Some(a.patience));
        m.fit_two_stage(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit_two_stage failed");
        let (d, dv) = (m.pred_dist(xte), m.pred_dist(xva));
        let v = Scorable::<LogScore>::score(&dv, yva).mean().unwrap();
        return finish(
            &d,
            &dv,
            yva,
            yte,
            m.base_models.len(),
            t.elapsed().as_secs_f64(),
            v,
            f64::NAN,
        );
    }
    if variant == "twostage"
        || variant == "twostage+nocal"
        || variant == "twostage+polish"
        || variant == "twostage+fgls"
    {
        // mean by μ-only boosting (early stop on val), then log σ boosted on
        // the training residuals (NormalFixedMean, early stop on val
        // residuals, default affine recalibration unless +nocal)
        let mut mu = build::<ngboost_rs::dist::NormalFixedVar>(a, seed, Some(a.patience));
        mu.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit failed");
        let (mtr, mva, mte) = (mu.predict(xtr), mu.predict(xva), mu.predict(xte));
        let mut sg = build::<ngboost_rs::dist::NormalFixedMean>(a, seed, Some(a.patience));
        sg.auto_calibrate_scale = variant == "twostage";
        sg.fit_with_validation(
            xtr,
            &(ytr - &mtr),
            Some(xva),
            Some(&(yva - &mva)),
            None,
            None,
        )
        .expect("fit failed");
        let (mu, mtr, mva, mte, sg) = if variant == "twostage+fgls" {
            // feasible GLS: refit the mean with weights 1/σ̂² (normalised to
            // mean 1, validation weighted alike), then σ on its residuals
            let w = |x: &Array2<f64>| {
                let v = sg.pred_dist(x).scale.mapv(|s| 1.0 / (s * s));
                let m = v.mean().unwrap();
                v / m
            };
            let (wtr, wva) = (w(xtr), w(xva));
            let mut mu = build::<ngboost_rs::dist::NormalFixedVar>(a, seed, Some(a.patience));
            mu.fit_with_validation(xtr, ytr, Some(xva), Some(yva), Some(&wtr), Some(&wva))
                .expect("fit failed");
            let (mtr, mva, mte) = (mu.predict(xtr), mu.predict(xva), mu.predict(xte));
            let mut sg = build::<ngboost_rs::dist::NormalFixedMean>(a, seed, Some(a.patience));
            sg.fit_with_validation(
                xtr,
                &(ytr - &mtr),
                Some(xva),
                Some(&(yva - &mva)),
                None,
                None,
            )
            .expect("fit failed");
            (mu, mtr, mva, mte, sg)
        } else {
            (mu, mtr, mva, mte, sg)
        };
        let _ = &mtr;
        let mk = |m: &Array1<f64>, x: &Array2<f64>| {
            let ls = sg.pred_dist(x).scale.mapv(f64::ln);
            let mut p = Array2::zeros((m.len(), 2));
            p.column_mut(0).assign(m);
            p.column_mut(1).assign(&ls);
            Normal::from_params(&p)
        };
        if variant == "twostage+polish" {
            // the two stages as one Normal model (μ rounds [tree, 0], σ
            // rounds [0, tree]), then joint boosting with early stopping
            use ngboost_rs::learners::{TrainedBaseLearner, TrainedHistogramTree};
            let nf = xtr.ncols();
            let zero = || -> Box<dyn TrainedBaseLearner> {
                Box::new(TrainedHistogramTree::new(
                    HistTreeNode::Leaf { value: 0.0 },
                    nf,
                ))
            };
            let cl =
                |l: &Box<dyn TrainedBaseLearner>| l.to_serializable().unwrap().to_trait_object();
            let mut m = build::<Normal>(a, seed, Some(a.patience));
            for r in &mu.base_models {
                m.base_models.push(vec![cl(&r[0]), zero()]);
            }
            for r in &sg.base_models {
                m.base_models.push(vec![zero(), cl(&r[0])]);
            }
            let (i0, i1) = (
                mu.init_params.as_ref().unwrap()[0],
                sg.init_params.as_ref().unwrap()[0],
            );
            m.init_params = Some(Array1::from(vec![i0, i1]));
            m.scalings = mu.scalings.iter().chain(&sg.scalings).cloned().collect();
            m.effective_learning_rates = mu
                .effective_learning_rates
                .iter()
                .chain(&sg.effective_learning_rates)
                .cloned()
                .collect();
            m.col_idxs = mu.col_idxs.iter().chain(&sg.col_idxs).cloned().collect();
            let r0 = m.base_models.len();
            m.partial_fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .expect("polish failed");
            let d = m.pred_dist(xte);
            let v = Scorable::<LogScore>::score(&m.pred_dist(xva), yva)
                .mean()
                .unwrap();
            return finish(
                &d,
                &d,
                yva,
                yte,
                m.base_models.len(),
                t.elapsed().as_secs_f64(),
                v,
                (m.base_models.len() - r0) as f64,
            );
        }
        if std::env::var("NGB_SIGMA_DIAG").is_ok() {
            let q = |x: &Array2<f64>| {
                let mut v: Vec<f64> = sg.pred_dist(x).scale.mapv(f64::ln).to_vec();
                v.sort_by(|a, b| a.total_cmp(b));
                (v[0], v[v.len() / 2], v[v.len() - 1])
            };
            let mx = |m: &[f64]| m.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sc: Vec<f64> = sg.scalings.clone();
            eprintln!(
                "seed {seed} {variant}: ln σ train {:?} test {:?} | σ rounds {} max scaling {}",
                q(xtr),
                q(xte),
                sg.base_models.len(),
                mx(&sc)
            );
        }
        let (d, dv) = (mk(&mte, xte), mk(&mva, xva));
        let v = Scorable::<LogScore>::score(&dv, yva).mean().unwrap();
        return finish(
            &d,
            &dv,
            yva,
            yte,
            mu.base_models.len() + sg.base_models.len(),
            t.elapsed().as_secs_f64(),
            v,
            sg.base_models.len() as f64,
        );
    }
    if variant == "trace" {
        let mut m = build::<Normal>(a, seed, None);
        m.auto_calibrate_scale = false;
        m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit failed");
        let sv = m.staged_pred_dist(xva);
        let st = m.staged_pred_dist(xte);
        let vn: Vec<f64> = sv
            .iter()
            .map(|d| Scorable::<LogScore>::score(d, yva).mean().unwrap())
            .collect();
        let vr: Vec<f64> = sv.iter().map(|d| rmse_of(&d.loc, yva)).collect();
        let argmin = |v: &[f64]| (0..v.len()).min_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
        let (bn, br) = (argmin(&vn), argmin(&vr));
        let d = &st[bn];
        let mut out = finish(
            d,
            d,
            yva,
            yte,
            bn + 1,
            t.elapsed().as_secs_f64(),
            (br + 1) as f64,
            rmse_of(&st[br].loc, yte),
        );
        out.mean_sigma = Scorable::<LogScore>::score(&st[br], yte).mean().unwrap();
        return out;
    }
    // Normal-head fixes: pls = per-parameter line search, cales = early
    // stopping on the validation NLL after the optimal log-σ shift (closed
    // form: mean ln σ + ½ ln mean z² + ½ ln 2π + ½)
    let toks: Vec<&str> = variant.split('+').collect();
    if toks.iter().all(|t| matches!(*t, "pls" | "cales" | "nocal")) {
        let mut m = build::<Normal>(a, seed, Some(a.patience));
        for t in &toks {
            match *t {
                "pls" => m.per_param_line_search = true,
                "nocal" => m.auto_calibrate_scale = false,
                _ => m.set_val_loss_monitor(Box::new(
                    |d: &Normal, y: &Array1<f64>, _w: Option<&Array1<f64>>| {
                        let n = y.len() as f64;
                        let z2 = (0..y.len())
                            .map(|i| ((y[i] - d.loc[i]) / d.scale[i]).powi(2))
                            .sum::<f64>()
                            / n;
                        d.log_scale.sum() / n
                            + 0.5 * z2.ln()
                            + 0.5 * std::f64::consts::TAU.ln()
                            + 0.5
                    },
                )),
            }
        }
        m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
            .expect("fit failed");
        let d = m.pred_dist(xte);
        let v = Scorable::<LogScore>::score(&m.pred_dist(xva), yva)
            .mean()
            .unwrap();
        let sc = m.scalings.iter().sum::<f64>() / m.scalings.len() as f64;
        let mut out = finish(
            &d,
            &d,
            yva,
            yte,
            m.base_models.len(),
            t.elapsed().as_secs_f64(),
            v,
            sc,
        );
        out.train_nll = Scorable::<LogScore>::score(&m.pred_dist(xtr), ytr)
            .mean()
            .unwrap();
        return out;
    }
    let mut base = build::<Normal>(a, seed, Some(a.patience));
    base.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
        .expect("fit failed");
    let r0 = base.base_models.len();
    let mut auto = NGBAutoMixtureRegressor::new(a.n_est, a.lr);
    auto.base_learner = HistogramLearner::new(a.depth);
    auto.early_stopping_rounds = Some(a.patience);
    auto.random_state = Some(seed);
    let normal_out = |m: &NGBoost<Normal, LogScore, HistogramLearner>, secs: f64| {
        let d = m.pred_dist(xte);
        let v = Scorable::<LogScore>::score(&m.pred_dist(xva), yva)
            .mean()
            .unwrap();
        finish(
            &d,
            &d,
            yva,
            yte,
            m.base_models.len(),
            secs,
            m.base_models.len() as f64,
            v,
        )
    };
    let mixture_out = |g: AutoMixtureModel, secs: f64| match g {
        AutoMixtureModel::K2(m) => {
            let d = m.pred_dist(xte);
            let mm = mm_nll(&d.mean(), &d.variance(), yte);
            finish(
                &d,
                &d,
                yva,
                yte,
                m.base_models.len(),
                secs,
                mm,
                (m.base_models.len() - r0) as f64,
            )
        }
        _ => unreachable!(),
    };
    match variant {
        "base" => normal_out(&base, t.elapsed().as_secs_f64()),
        "cont" => {
            base.partial_fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .expect("continue failed");
            normal_out(&base, t.elapsed().as_secs_f64())
        }
        "matched" => {
            let src = AutoMixtureModel::Normal(base);
            let g = auto
                .grow_from(2, &src, xtr, ytr, xva, yva)
                .expect("grow failed");
            let r = match &g {
                AutoMixtureModel::K2(m) => m.base_models.len() - r0,
                _ => unreachable!(),
            };
            let AutoMixtureModel::Normal(mut base) = src else {
                unreachable!()
            };
            base.n_estimators = r as u32;
            base.early_stopping_rounds = None;
            base.partial_fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None)
                .expect("continue failed");
            normal_out(&base, t.elapsed().as_secs_f64())
        }
        "sym2" | "warm2" | "warm2d" | "warm2s" => {
            let (dl, sh) = match variant {
                "sym2" => (0.0, 1.0),
                "warm2d" => (0.8, 1.0),
                "warm2s" => (0.0, 0.6),
                _ => (0.8, 0.6),
            };
            auto.split_delta = dl;
            auto.split_shrink = sh;
            let src = AutoMixtureModel::Normal(base);
            let g = auto
                .grow_from(2, &src, xtr, ytr, xva, yva)
                .expect("grow failed");
            mixture_out(g, t.elapsed().as_secs_f64())
        }
        other => panic!("unknown normprobe token {other}"),
    }
}

/// CRPS ∫(F(t) − 1{t ≥ y})² dt by the midpoint rule on segments split at
/// the atom (0), at y and at the 1 % / 99 % quantiles, so the integrand is
/// smooth on every piece (exact for mixed discrete/continuous laws up to
/// quadrature error). Rows are handled one by one through 1-row closures.
fn crps_split(
    cdf: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    ppf: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    y: &Array1<f64>,
) -> Array1<f64> {
    let n = y.len();
    let q = |p: f64| ppf(&Array1::from_elem(n, p));
    let (lo, q01, q99, hi) = (q(1e-6), q(0.01), q(0.99), q(1.0 - 1e-6));
    const M: usize = 160;
    let mut pts: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut wts: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut b = vec![lo[i], q01[i], q99[i], hi[i], 0.0, y[i]];
        let a0 = b.iter().cloned().fold(f64::INFINITY, f64::min);
        let a1 = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        b.retain(|v| v.is_finite());
        b.push(a0);
        b.push(a1);
        b.sort_by(|u, v| u.total_cmp(v));
        b.dedup();
        let (mut p, mut w) = (Vec::new(), Vec::new());
        for s in b.windows(2) {
            let h = (s[1] - s[0]) / M as f64;
            if !(h > 0.0) || !h.is_finite() {
                continue;
            }
            for k in 0..M {
                p.push(s[0] + (k as f64 + 0.5) * h);
                w.push(h);
            }
        }
        pts.push(p);
        wts.push(w);
    }
    let m = pts.iter().map(|p| p.len()).max().unwrap_or(0);
    let mut acc = Array1::<f64>::zeros(n);
    for k in 0..m {
        let t = Array1::from_shape_fn(n, |i| *pts[i].get(k).unwrap_or(&y[i]));
        let f = cdf(&t);
        for i in 0..n {
            if let Some(&w) = wts[i].get(k) {
                let ind = if t[i] >= y[i] { 1.0 } else { 0.0 };
                acc[i] += w * (f[i] - ind).powi(2);
            }
        }
    }
    acc
}

/// Metrics for targets with exact zeros (Cycle 13). nll = mean −log score
/// over all test rows (mixed measure for hurdle heads — comparable only
/// between hurdle heads); extra = NLL on the non-zero test rows (a Lebesgue
/// density for every head, so comparable across heads); calib = Brier score
/// of P(Y = 0); mean_sigma = mean predicted P(Y = 0); extra2 = test share
/// of zeros. CRPS by `crps_split` for every head.
#[allow(clippy::too_many_arguments)]
fn zero_metrics(
    cdf: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    ppf: &dyn Fn(&Array1<f64>) -> Array1<f64>,
    logpdf: &Array1<f64>,
    mean: &Array1<f64>,
    yte: &Array1<f64>,
    n_iter: usize,
    secs: f64,
) -> Metrics {
    let n = yte.len();
    let p0 = &cdf(&Array1::zeros(n)) - &cdf(&Array1::from_elem(n, -1e-12));
    let zero: Vec<bool> = yte.iter().map(|&v| v == 0.0).collect();
    let brier = (0..n)
        .map(|i| (p0[i] - if zero[i] { 1.0 } else { 0.0 }).powi(2))
        .sum::<f64>()
        / n as f64;
    let nz: Vec<usize> = (0..n).filter(|&i| !zero[i]).collect();
    let nll_nz = nz.iter().map(|&i| -logpdf[i]).sum::<f64>() / nz.len().max(1) as f64;
    let cov = |alpha: f64| {
        let lo = ppf(&Array1::from_elem(n, alpha / 2.0));
        let hi = ppf(&Array1::from_elem(n, 1.0 - alpha / 2.0));
        (0..n)
            .filter(|&i| yte[i] >= lo[i] && yte[i] <= hi[i])
            .count() as f64
            / n as f64
    };
    Metrics {
        n_iter,
        secs,
        train_nll: f64::NAN,
        nll: -logpdf.mean().unwrap(),
        crps: crps_split(cdf, ppf, yte).mean().unwrap(),
        rmse: (mean - yte).mapv(|e| e * e).mean().unwrap().sqrt(),
        cov50: cov(0.5),
        cov90: cov(0.1),
        calib: brier,
        mean_sigma: p0.mean().unwrap(),
        extra: nll_nz,
        extra2: (n - nz.len()) as f64 / n as f64,
    }
}

/// Heads for targets with exact zeros. Variants (one head each): normal,
/// lognormal (only if y > 0), hnormal, hlognormal, hmix2 (Hurdle over a
/// 2-component mixture), automix (NGBAutoMixtureRegressor, default search).
/// Library defaults otherwise (scale recalibration on). A failed fit
/// (diverged / invalid targets) is reported as NaN.
#[allow(clippy::too_many_arguments)]
fn run_one_zero(
    variant: &str,
    a: &Args,
    seed: u64,
    xtr: &Array2<f64>,
    ytr: &Array1<f64>,
    xva: &Array2<f64>,
    yva: &Array1<f64>,
    xte: &Array2<f64>,
    yte: &Array1<f64>,
) -> Metrics {
    use ngboost_rs::dist::{Hurdle, HurdleLogNormal, HurdleNormal};
    fn head<D, S>(
        a: &Args,
        seed: u64,
        xtr: &Array2<f64>,
        ytr: &Array1<f64>,
        xva: &Array2<f64>,
        yva: &Array1<f64>,
        xte: &Array2<f64>,
        yte: &Array1<f64>,
    ) -> Metrics
    where
        D: Distribution + Scorable<S> + Scorable<LogScore> + DistributionMethods + Clone + 'static,
        S: Score,
    {
        let t = Instant::now();
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
        if let Err(e) = m.fit_with_validation(xtr, ytr, Some(xva), Some(yva), None, None) {
            eprintln!("fit failed ({e}); reporting NaN");
            return nan_metrics();
        }
        let secs = t.elapsed().as_secs_f64();
        let d = m.pred_dist(xte);
        let row =
            |i: usize| D::from_params(&d.params().slice(ndarray::s![i..i + 1, ..]).to_owned());
        let _ = row;
        let cdf = |v: &Array1<f64>| d.cdf(v);
        let ppf = |v: &Array1<f64>| d.ppf(v);
        zero_metrics(
            &cdf,
            &ppf,
            &d.logpdf(yte),
            &d.mean(),
            yte,
            m.base_models.len(),
            secs,
        )
    }
    if variant == "crpscheck" {
        // integrator check: crps_split vs the closed-form Normal CRPS
        let mut l = Array2::zeros((yte.len(), 2));
        for i in 0..yte.len() {
            l[[i, 0]] = (i % 7) as f64 - 3.0;
            l[[i, 1]] = ((i % 5) as f64 - 2.0) * 0.7;
        }
        let d = Normal::from_params(&l);
        let cdf = |v: &Array1<f64>| d.cdf(v);
        let ppf = |v: &Array1<f64>| d.ppf(v);
        let num = crps_split(&cdf, &ppf, yte);
        let ex = Scorable::<CRPScore>::score(&d, yte);
        let err = (&num - &ex).mapv(f64::abs);
        eprintln!(
            "crps_split vs exact: max abs err {:.2e}, max rel err {:.2e}, mean {:.6} vs {:.6}",
            err.iter().cloned().fold(0.0, f64::max),
            (0..yte.len()).map(|i| err[i] / ex[i]).fold(0.0, f64::max),
            num.mean().unwrap(),
            ex.mean().unwrap()
        );
        return nan_metrics();
    }
    match variant {
        "normal" => head::<Normal, LogScore>(a, seed, xtr, ytr, xva, yva, xte, yte),
        "lognormal" => {
            if ytr.iter().chain(yva).chain(yte).any(|&v| !(v > 0.0)) {
                return nan_metrics();
            }
            head::<LogNormal, LogScore>(a, seed, xtr, ytr, xva, yva, xte, yte)
        }
        "hnormal" => head::<HurdleNormal, LogScore>(a, seed, xtr, ytr, xva, yva, xte, yte),
        "hlognormal" => {
            if ytr.iter().chain(yva).chain(yte).any(|&v| v < 0.0) {
                return nan_metrics();
            }
            head::<HurdleLogNormal, LogScore>(a, seed, xtr, ytr, xva, yva, xte, yte)
        }
        "hnormalcrps" => head::<HurdleNormal, CRPScore>(a, seed, xtr, ytr, xva, yva, xte, yte),
        "hlognormalcrps" => {
            if ytr.iter().chain(yva).chain(yte).any(|&v| v < 0.0) {
                return nan_metrics();
            }
            head::<HurdleLogNormal, CRPScore>(a, seed, xtr, ytr, xva, yva, xte, yte)
        }
        "hmix2" => head::<Hurdle<NormalMixture2>, LogScore>(a, seed, xtr, ytr, xva, yva, xte, yte),
        "automix" => {
            use ngboost_rs::NGBAutoMixtureRegressor;
            let t = Instant::now();
            let mut m = NGBAutoMixtureRegressor::new(a.n_est, a.lr);
            m.base_learner = HistogramLearner::new(a.depth);
            m.early_stopping_rounds = Some(a.patience);
            m.random_state = Some(seed);
            if let Err(e) = m.fit(xtr, ytr, xva, yva) {
                eprintln!("automix fit failed ({e}); reporting NaN");
                return nan_metrics();
            }
            let secs = t.elapsed().as_secs_f64();
            let cdf = |v: &Array1<f64>| m.cdf(xte, v);
            let ppf = |v: &Array1<f64>| m.ppf(xte, v);
            let mut out = zero_metrics(
                &cdf,
                &ppf,
                &m.logpdf(xte, yte),
                &m.predict(xte),
                yte,
                m.tried.iter().map(|r| r.3).sum(),
                secs,
            );
            out.train_nll = m.n_components().unwrap() as f64;
            out
        }
        other => panic!("unknown zero-head variant {other}"),
    }
}

fn nan_metrics() -> Metrics {
    Metrics {
        n_iter: 0,
        secs: f64::NAN,
        train_nll: f64::NAN,
        nll: f64::NAN,
        crps: f64::NAN,
        rmse: f64::NAN,
        cov50: f64::NAN,
        cov90: f64::NAN,
        calib: f64::NAN,
        mean_sigma: f64::NAN,
        extra: f64::NAN,
        extra2: f64::NAN,
    }
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let sd = if v.len() > 1 {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    (m, sd)
}

fn main() {
    let a = parse_args();
    println!(
        "dataset,seed,variant,n_iter,secs,train_nll,nll,crps,rmse,cov50,cov90,calib,mean_sigma,extra,extra2"
    );
    // results[(dataset, variant)] = Vec<Metrics> indexed by seed order
    let mut results: BTreeMap<(String, String), Vec<Metrics>> = BTreeMap::new();

    for ds in &a.datasets {
        for s in 0..a.seeds {
            let seed = a.seed0 + s;
            *CURRENT_DS.lock().unwrap() = ds.clone();
            let (x, y) = load_dataset(ds, seed, &a);
            let y = if a.y_shift != 0.0 { y + a.y_shift } else { y };
            let mut idx = shuffled_indices(x.nrows(), seed);
            idx.truncate(a.max_rows.min(idx.len()));
            let n = idx.len();
            let n_tr = (0.7 * n as f64) as usize;
            let n_va = (0.1 * n as f64) as usize;
            let (xtr, ytr) = take(&x, &y, &idx[..n_tr]);
            let n_va_used = n_va.min(a.val_rows);
            let (xva, yva) = take(&x, &y, &idx[n_tr..n_tr + n_va_used]);
            let (xte, yte) = take(&x, &y, &idx[n_tr + n_va..]);
            let lam_true: Option<Array1<f64>> = (ds.starts_with("pois_") || ds.starts_with("nb_"))
                .then(|| {
                    synth_count(ds, a.n_synth, seed)
                        .2
                        .select(ndarray::Axis(0), &idx[n_tr + n_va..])
                });
            for v in &a.variants {
                let m = match a.dist.as_str() {
                    "normal" => run_one(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte),
                    "laplace" => run_one_generic::<Laplace, LogScore>(
                        None, v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte,
                    ),
                    "studentt" => run_one_generic::<StudentT, LogScore>(
                        None, v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte,
                    ),
                    "poisson" => run_one_poisson(
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                        lam_true.as_ref(),
                    ),
                    "lognormal" => run_one_generic::<LogNormal, LogScore>(
                        None, v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte,
                    ),
                    "bernoulli" => {
                        run_one_bernoulli(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "studenttcrps" => run_one_generic::<StudentT, CRPScore>(
                        Some(exact_crps::<StudentT>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "poissoncrps" => run_one_generic::<Poisson, CRPScore>(
                        Some(exact_crps::<Poisson>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "gamma" => run_one_generic::<Gamma, LogScore>(
                        Some(exact_crps::<Gamma>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "weibull" => run_one_generic::<Weibull, LogScore>(
                        Some(exact_crps::<Weibull>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "gammacrps" => run_one_generic::<Gamma, CRPScore>(
                        Some(exact_crps::<Gamma>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "weibullcrps" => run_one_generic::<Weibull, CRPScore>(
                        Some(exact_crps::<Weibull>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "halfnormalcrps" => run_one_generic::<HalfNormal, CRPScore>(
                        Some(exact_crps::<HalfNormal>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "normalcrps" => run_one_generic::<Normal, CRPScore>(
                        Some(exact_crps::<Normal>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "normalg" => run_one_generic::<Normal, LogScore>(
                        None, v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte,
                    ),
                    "mix2crps" => run_one_generic::<NormalMixture2, CRPScore>(
                        Some(exact_crps::<NormalMixture2>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "mix3" => run_one_generic::<NormalMixture3, LogScore>(
                        Some(exact_crps::<NormalMixture3>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "mix4" => run_one_generic::<NormalMixture<4>, LogScore>(
                        Some(exact_crps::<NormalMixture<4>>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "mix3crps" => run_one_generic::<NormalMixture3, CRPScore>(
                        Some(exact_crps::<NormalMixture3>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "automix" => run_one_automix(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte),
                    "zero" => run_one_zero(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte),
                    "normprobe" => {
                        run_one_normprobe(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "mix2" => run_one_generic::<NormalMixture2, LogScore>(
                        Some(exact_crps::<NormalMixture2>),
                        v,
                        &a,
                        seed,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                    ),
                    "cat3" => {
                        run_one_categorical::<3>(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "cat4" => {
                        run_one_categorical::<4>(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "cat8" => {
                        run_one_categorical::<8>(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "cat10" => {
                        run_one_categorical::<10>(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "cat5" => {
                        run_one_categorical::<5>(v, &a, seed, &xtr, &ytr, &xva, &yva, &xte, &yte)
                    }
                    "negbincrps" => run_one_negbin::<CRPScore>(
                        v,
                        &a,
                        seed,
                        ds,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                        lam_true.as_ref(),
                    ),
                    "negbin" => run_one_negbin::<LogScore>(
                        v,
                        &a,
                        seed,
                        ds,
                        &xtr,
                        &ytr,
                        &xva,
                        &yva,
                        &xte,
                        &yte,
                        lam_true.as_ref(),
                    ),
                    other => panic!("unknown --dist {other}"),
                };
                println!(
                    "{ds},{seed},{v},{},{:.2},{:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    m.n_iter,
                    m.secs,
                    m.train_nll,
                    m.nll,
                    m.crps,
                    m.rmse,
                    m.cov50,
                    m.cov90,
                    m.calib,
                    m.mean_sigma,
                    m.extra,
                    m.extra2
                );
                results.entry((ds.clone(), v.clone())).or_default().push(m);
            }
        }
    }

    // ---- summary: mean ± sd per (dataset, variant); paired diff vs variants[0]
    println!("\n# SUMMARY (mean ± sd over {} seeds)", a.seeds);
    println!(
        "# {:<8} {:<10} {:>7} {:>16} {:>16} {:>16} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "dataset",
        "variant",
        "iters",
        "NLL",
        "CRPS",
        "RMSE",
        "cov50",
        "cov90",
        "calib",
        "extra",
        "extra2"
    );
    for ds in &a.datasets {
        for v in &a.variants {
            let r = &results[&(ds.clone(), v.clone())];
            let f = |g: &dyn Fn(&Metrics) -> f64| mean_sd(&r.iter().map(g).collect::<Vec<_>>());
            let (nll, nll_sd) = f(&|m| m.nll);
            let (crps, crps_sd) = f(&|m| m.crps);
            let (rmse, rmse_sd) = f(&|m| m.rmse);
            let (c50, _) = f(&|m| m.cov50);
            let (c90, _) = f(&|m| m.cov90);
            let (cal, _) = f(&|m| m.calib);
            let (ex, _) = f(&|m| m.extra);
            let (ex2, _) = f(&|m| m.extra2);
            let (it, _) = f(&|m| m.n_iter as f64);
            println!(
                "# {:<8} {:<10} {:>7.0} {:>8.4}±{:<7.4} {:>8.4}±{:<7.4} {:>8.4}±{:<7.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
                ds, v, it, nll, nll_sd, crps, crps_sd, rmse, rmse_sd, c50, c90, cal, ex, ex2
            );
        }
        if a.variants.len() > 1 {
            let base = &results[&(ds.clone(), a.variants[0].clone())];
            for v in &a.variants[1..] {
                let r = &results[&(ds.clone(), v.clone())];
                let d_nll: Vec<f64> = r.iter().zip(base).map(|(m, b)| m.nll - b.nll).collect();
                let d_crps: Vec<f64> = r.iter().zip(base).map(|(m, b)| m.crps - b.crps).collect();
                let d_rmse: Vec<f64> = r.iter().zip(base).map(|(m, b)| m.rmse - b.rmse).collect();
                let wins = d_nll.iter().filter(|d| **d < 0.0).count();
                let (mn, sn) = mean_sd(&d_nll);
                let (mc, sc) = mean_sd(&d_crps);
                let (mr, sr) = mean_sd(&d_rmse);
                let se = |sd: f64| sd / (d_nll.len() as f64).sqrt();
                println!(
                    "#   Δ({v} − {}): NLL {:+.4} (se {:.4}, wins {wins}/{}) | CRPS {:+.4} (se {:.4}) | RMSE {:+.4} (se {:.4})",
                    a.variants[0],
                    mn,
                    se(sn),
                    d_nll.len(),
                    mc,
                    se(sc),
                    mr,
                    se(sr)
                );
            }
        }
    }
}
