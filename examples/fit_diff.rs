//! Full-fit differential vs Python: replay the deterministic fit from
//! /tmp/ngb_diff/fit_case.json (Normal/LogScore, exact tree, no subsampling)
//! and compare per-iteration staged params against Python's.
use ndarray::{Array1, Array2};
use ngboost_rs::dist::Normal;
use ngboost_rs::learners::default_tree_learner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::{Distribution, LogScore};
use serde_json::Value;

fn main() {
    let txt = std::fs::read_to_string("/tmp/ngb_diff/fit_case.json").unwrap();
    let c: Value = serde_json::from_str(&txt).unwrap();

    let xv: Vec<Vec<f64>> = c["X"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let n = xv.len();
    let p = xv[0].len();
    let mut x = Array2::zeros((n, p));
    for (i, row) in xv.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x[[i, j]] = v;
        }
    }
    let y: Array1<f64> = c["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let m = c["staged_params"].as_array().unwrap().len();
    let mut model: NGBoost<Normal, LogScore, _> =
        NGBoost::new(m as u32, 0.9, default_tree_learner());
    model.fit(&x, &y).unwrap();

    // init params
    let init_py: Vec<f64> = c["init_params"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let init_rs = Normal::fit(&y);
    println!(
        "init diff: {:.2e}",
        init_py
            .iter()
            .zip(init_rs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    );

    // scalings
    let sc_py: Vec<f64> = c["scalings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let sc_diff = sc_py
        .iter()
        .zip(model.scalings.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("scalings diff: {:.2e} (py first 3: {:?}, rs first 3: {:?})",
        sc_diff, &sc_py[..3.min(sc_py.len())], &model.scalings[..3.min(model.scalings.len())]);

    // staged params: python (n_params, N); rust .params() -> (N, n_params)
    let staged_rs = model.staged_pred_dist(&x);
    let staged_py = c["staged_params"].as_array().unwrap();
    for (it, (d_rs, sp)) in staged_rs.iter().zip(staged_py.iter()).enumerate() {
        let prs = d_rs.params();
        let ppy: Vec<Vec<f64>> = sp
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        let mut max_d = 0.0f64;
        for i in 0..n {
            for j in 0..ppy.len() {
                max_d = max_d.max((prs[[i, j]] - ppy[j][i]).abs());
            }
        }
        if it < 3 || it == staged_py.len() - 1 || max_d > 1e-6 {
            println!("iter {:2}: max param diff = {:.3e}", it + 1, max_d);
        }
    }
}
