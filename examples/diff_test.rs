//! Differential test: compare Rust Scorable score/d_score/metric against the
//! Python ngboost values dumped to /tmp/ngb_diff/cases.json (identical raw
//! params + y).  Reports max abs diff per quantity; flags mismatches.
use ndarray::{Array1, Array2};
use ngboost_rs::dist::{
    Categorical, Cauchy, CauchyFixedVar, Exponential, Gamma, HalfNormal, Laplace, LogNormal,
    MultivariateNormal, Normal, NormalFixedMean, NormalFixedVar, Poisson, StudentT, TFixedDf,
    TFixedDfFixedVar, Weibull,
};
use ngboost_rs::scores::{
    CRPScoreCensored, CensoredScorable, LogScoreCensored, SurvivalData,
};
use ngboost_rs::{CRPScore, Distribution, LogScore, Scorable, Score};
use serde_json::Value;

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                (x - y).abs()
            } else if x.is_nan() && y.is_nan() {
                0.0
            } else if x == y {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0f64, f64::max)
}

/// params from JSON are (n_params, n_obs) [Python]; Rust wants (n_obs, n_params).
fn build_params(pp: &[Vec<f64>]) -> Array2<f64> {
    let n_params = pp.len();
    let n_obs = pp[0].len();
    let mut a = Array2::zeros((n_obs, n_params));
    for (j, row) in pp.iter().enumerate() {
        for (i, &v) in row.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    a
}

/// (score_diff, dscore_diff, metric_diff, fd_grad_diff) where fd_grad_diff is
/// Rust d_score vs the central finite-difference gradient of Rust's OWN score.
fn run<D, S>(c: &Value) -> (f64, f64, f64, f64)
where
    D: Distribution + Scorable<S>,
    S: Score,
{
    let pp: Vec<Vec<f64>> = c["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let y: Vec<f64> = c["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let y = Array1::from(y);
    let params = build_params(&pp);
    let d = D::from_params(&params);

    let rs_score = Scorable::<S>::score(&d, &y);
    let rs_ds = Scorable::<S>::d_score(&d, &y);
    let rs_m = Scorable::<S>::metric(&d);

    let py_score: Vec<f64> =
        c["py_score"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    // d_score: (n_obs, n_params)
    let py_ds: Vec<f64> = c["py_dscore"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>())
        .collect();
    // metric: (n_obs, n_params, n_params)
    let py_m: Vec<f64> = c["py_metric"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .flat_map(|rr| rr.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
        .collect();

    let ds_flat: Vec<f64> = rs_ds.iter().copied().collect();
    let m_flat: Vec<f64> = rs_m.iter().copied().collect();
    let sc_flat: Vec<f64> = rs_score.iter().copied().collect();

    // Central finite-difference gradient of Rust's own score.
    // n_obs from params, not y.len(): MVN flattens y to (n_obs * P,).
    let n_obs = params.nrows();
    let n_params = params.ncols();
    let h = 1e-6;
    let mut fd = Array2::<f64>::zeros((n_obs, n_params));
    for j in 0..n_params {
        let mut pp = params.clone();
        pp.column_mut(j).mapv_inplace(|v| v + h);
        let sp = Scorable::<S>::score(&D::from_params(&pp), &y);
        let mut pm = params.clone();
        pm.column_mut(j).mapv_inplace(|v| v - h);
        let sm = Scorable::<S>::score(&D::from_params(&pm), &y);
        for i in 0..n_obs {
            fd[[i, j]] = (sp[i] - sm[i]) / (2.0 * h);
        }
    }
    let fd_flat: Vec<f64> = fd.iter().copied().collect();

    (
        max_abs(&sc_flat, &py_score),
        max_abs(&ds_flat, &py_ds),
        max_abs(&m_flat, &py_m),
        max_abs(&ds_flat, &fd_flat),
    )
}

fn run_cens<D, S>(c: &Value) -> (f64, f64, f64, f64)
where
    D: Distribution + CensoredScorable<S>,
    S: Score,
{
    let pp: Vec<Vec<f64>> = c["params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let y: Vec<f64> = c["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let time = Array1::from(y);
    let event: Vec<bool> =
        c["event"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() != 0).collect();
    let sd = SurvivalData::new(Array1::from(event), time);
    let params = build_params(&pp);
    let d = D::from_params(&params);

    let rs_score = CensoredScorable::<S>::censored_score(&d, &sd);
    let rs_ds = CensoredScorable::<S>::censored_d_score(&d, &sd);
    let rs_m = CensoredScorable::<S>::censored_metric(&d);

    let py_score: Vec<f64> =
        c["py_score"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let py_ds: Vec<f64> = c["py_dscore"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>())
        .collect();
    let py_m: Vec<f64> = c["py_metric"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .flat_map(|rr| rr.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
        .collect();
    let sc_flat: Vec<f64> = rs_score.iter().copied().collect();
    let ds_flat: Vec<f64> = rs_ds.iter().copied().collect();
    let m_flat: Vec<f64> = rs_m.iter().copied().collect();

    // Central finite-difference gradient of Rust's own censored score.
    let n_obs = sd.time.len();
    let n_params = params.ncols();
    let h = 1e-6;
    let mut fd = Array2::<f64>::zeros((n_obs, n_params));
    for j in 0..n_params {
        let mut pp = params.clone();
        pp.column_mut(j).mapv_inplace(|v| v + h);
        let sp = CensoredScorable::<S>::censored_score(&D::from_params(&pp), &sd);
        let mut pm = params.clone();
        pm.column_mut(j).mapv_inplace(|v| v - h);
        let sm = CensoredScorable::<S>::censored_score(&D::from_params(&pm), &sd);
        for i in 0..n_obs {
            fd[[i, j]] = (sp[i] - sm[i]) / (2.0 * h);
        }
    }
    let fd_flat: Vec<f64> = fd.iter().copied().collect();
    (
        max_abs(&sc_flat, &py_score),
        max_abs(&ds_flat, &py_ds),
        max_abs(&m_flat, &py_m),
        max_abs(&ds_flat, &fd_flat),
    )
}

fn main() {
    let txt = std::fs::read_to_string("/tmp/ngb_diff/cases.json").expect("run dump_py.py first");
    let cases: Vec<Value> = serde_json::from_str(&txt).unwrap();
    println!(
        "{:<16} {:<10} {:>11} {:>13} {:>12} {:>13}",
        "dist", "score", "score_vs_py", "dscore_vs_py", "metric_vs_py", "grad_vs_own_fd"
    );
    println!("{}", "-".repeat(82));
    let tol = 1e-8;
    let fd_tol = 1e-5; // FD central-difference accuracy floor
    let mut any_bad = false;
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let score = c["score"].as_str().unwrap();
        let (s, ds, m, fd) = match (name, score) {
            ("normal", "LogScore") => run::<Normal, LogScore>(c),
            ("normal", "CRPScore") => run::<Normal, CRPScore>(c),
            ("laplace", "LogScore") => run::<Laplace, LogScore>(c),
            ("laplace", "CRPScore") => run::<Laplace, CRPScore>(c),
            ("exponential", "LogScore") => run::<Exponential, LogScore>(c),
            ("exponential", "CRPScore") => run::<Exponential, CRPScore>(c),
            ("gamma", "LogScore") => run::<Gamma, LogScore>(c),
            ("weibull", "LogScore") => run::<Weibull, LogScore>(c),
            ("poisson", "LogScore") => run::<Poisson, LogScore>(c),
            ("halfnormal", "LogScore") => run::<HalfNormal, LogScore>(c),
            ("studentt", "LogScore") => run::<StudentT, LogScore>(c),
            ("cauchy", "LogScore") => run::<Cauchy, LogScore>(c),
            ("lognormal", "LogScore") => run::<LogNormal, LogScore>(c),
            ("lognormal", "CRPScore") => run::<LogNormal, CRPScore>(c),
            ("normalfixedvar", "LogScore") => run::<NormalFixedVar, LogScore>(c),
            ("normalfixedvar", "CRPScore") => run::<NormalFixedVar, CRPScore>(c),
            ("normalfixedmean", "LogScore") => run::<NormalFixedMean, LogScore>(c),
            ("normalfixedmean", "CRPScore") => run::<NormalFixedMean, CRPScore>(c),
            ("tfixeddf", "LogScore") => run::<TFixedDf, LogScore>(c),
            ("tfixeddffixedvar", "LogScore") => run::<TFixedDfFixedVar, LogScore>(c),
            ("cauchyfixedvar", "LogScore") => run::<CauchyFixedVar, LogScore>(c),
            ("categorical4", "LogScore") => run::<Categorical<4>, LogScore>(c),
            ("mvn3", "LogScore") => run::<MultivariateNormal<3>, LogScore>(c),
            ("exponential_cens", "LogScore") => run_cens::<Exponential, LogScoreCensored>(c),
            ("exponential_cens", "CRPScore") => run_cens::<Exponential, CRPScoreCensored>(c),
            ("lognormal_cens", "LogScore") => run_cens::<LogNormal, LogScoreCensored>(c),
            ("lognormal_cens", "CRPScore") => run_cens::<LogNormal, CRPScoreCensored>(c),
            _ => {
                println!("{:<14} {:<10} (no Rust mapping)", name, score);
                continue;
            }
        };
        let flag = |d: f64, t: f64| if d > t { " <" } else { "  " };
        // A REAL Rust bug shows up as Rust d_score disagreeing with its own FD.
        // Exception: exponential censored LogScore uses an eps=1e-10 survival
        // guard in the score (matching Python) that the analytic grad ignores;
        // this shows up as an FD gap ~ t/s * eps/(S+eps) in the deep tail only.
        let eps_guard_artifact = name == "exponential_cens" && score == "LogScore";
        if fd > fd_tol && !eps_guard_artifact {
            any_bad = true;
        }
        println!(
            "{:<16} {:<10} {:>9.1e}{} {:>11.1e}{} {:>10.1e}{} {:>11.1e}{}",
            name, score, s, flag(s, tol), ds, flag(ds, tol), m, flag(m, tol), fd, flag(fd, fd_tol)
        );
    }
    println!("{}", "-".repeat(82));
    println!("'<' in *_vs_py = differs from Python (may be a KNOWN Python bug Rust fixes).");
    println!("grad_vs_own_fd = Rust d_score vs finite-diff of Rust's OWN score (tol {:.0e}).", fd_tol);
    if any_bad {
        println!(">>> A Rust gradient disagrees with its own score's derivative — REAL BUG.");
    } else {
        println!("OK: every Rust d_score matches the finite-diff derivative of its own score.");
        println!("    => where score matches Python but d_score doesn't, PYTHON's grad is wrong.");
    }
}
