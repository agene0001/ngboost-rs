//! NormalMixture2 head: training behaviour on bimodal vs unimodal data.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::NGBMixtureRegressor;
use ngboost_rs::dist::{DistributionMethods, Normal, NormalMixture2};
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (u32::MAX as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next().max(1e-12);
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// x-dependent mixture of a tight low mode and a wide high mode.
fn bimodal(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let p = 1.0 / (1.0 + (-(1.5 * r[0] + 0.5)).exp());
        if rng.next() < p {
            2.0 + 0.5 * r[1] + 0.6 * rng.gauss()
        } else {
            12.0 + 2.0 * r[2].sin() + 2.5 * rng.gauss()
        }
    });
    (x, y)
}

fn mk<D: ngboost_rs::Distribution + ngboost_rs::Scorable<LogScore> + Clone>(
    seed: u64,
) -> NGBoost<D, LogScore, HistogramLearner> {
    NGBoost::with_full_options(
        400,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        Some(30),
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    )
}

#[test]
fn mixture_beats_normal_on_bimodal_targets() {
    let (x, y) = bimodal(2500, 1);
    let (xv, yv) = bimodal(600, 2);
    let (xt, yt) = bimodal(1500, 3);
    let mut mix: NGBoost<NormalMixture2, _, _> = mk(7);
    mix.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let mut nor: NGBoost<Normal, _, _> = mk(7);
    nor.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (a, b) = (mix.score(&xt, &yt), nor.score(&xt, &yt));
    assert!(a < b - 0.3, "mixture {a} should clearly beat Normal {b}");
    let d = mix.pred_dist(&xt);
    assert!(d.mean().iter().all(|v| v.is_finite()));
    assert!(
        mix.scale_calibration.is_some(),
        "default calibration runs on both scales"
    );
}

#[test]
fn mixture_regressor_alias_and_serialization_round_trip() {
    let (x, y) = bimodal(600, 11);
    let mut m: NGBMixtureRegressor = mk(7);
    m.n_estimators = 40;
    m.early_stopping_rounds = None;
    m.fit(&x, &y).unwrap();
    let ser = m.serialize().unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let back: ngboost_rs::ngboost::SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 = NGBMixtureRegressor::deserialize(back, HistogramLearner::new(3)).unwrap();
    let (p1, p2) = (m.pred_param(&x), m2.pred_param(&x));
    for (u, v) in p1.iter().zip(p2.iter()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
}

/// Unimodal Gaussian data (heteroscedastic) for the selection test.
fn unimodal(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        2.0 * r[0].sin() + r[1] + (-0.5 + 0.4 * r[2]).exp() * rng.gauss()
    });
    (x, y)
}

#[test]
fn auto_mixture_selects_by_validation_nll() {
    use ngboost_rs::{KSearch, NGBAutoMixtureRegressor};
    // bimodal target → more than one component
    let (x, y) = bimodal(2000, 41);
    let (xv, yv) = bimodal(600, 42);
    let (xt, yt) = bimodal(1000, 43);
    let mut m = NGBAutoMixtureRegressor::new(300, 0.05);
    m.random_state = Some(7);
    m.candidates = vec![1, 2, 3];
    m.search = KSearch::Cold;
    m.fit(&x, &y, &xv, &yv).unwrap();
    assert_eq!(m.tried.len(), 3);
    let best = m.tried.iter().min_by(|a, b| a.2.total_cmp(&b.2)).unwrap().0;
    assert_eq!(
        m.n_components(),
        Some(best),
        "selects the lowest validation NLL"
    );
    assert!(
        best >= 2,
        "bimodal target should not pick the single Normal: {:?}",
        m.tried
    );
    let nll = m.score(&xt, &yt);
    assert!(nll.is_finite());
    let (lo, hi) = m.interval(&xt, 0.1);
    let cov = (0..yt.len())
        .filter(|&i| yt[i] >= lo[i] && yt[i] <= hi[i])
        .count() as f64
        / yt.len() as f64;
    assert!(cov > 0.8 && cov < 0.97, "90 % interval coverage {cov}");
    assert!(m.predict_median(&xt).iter().all(|v| v.is_finite()));

    // unimodal Gaussian target: no worse on test than the single Normal by
    // more than noise
    let (x, y) = unimodal(2000, 51);
    let (xv, yv) = unimodal(600, 52);
    let (xt, yt) = unimodal(1000, 53);
    let mut a = NGBAutoMixtureRegressor::new(300, 0.05);
    a.random_state = Some(7);
    a.candidates = vec![1, 2, 3];
    a.fit(&x, &y, &xv, &yv).unwrap();
    let mut n1 = NGBAutoMixtureRegressor::new(300, 0.05);
    n1.random_state = Some(7);
    n1.candidates = vec![1];
    n1.fit(&x, &y, &xv, &yv).unwrap();
    assert!(
        a.score(&xt, &yt) < n1.score(&xt, &yt) + 0.05,
        "{:?}",
        a.tried
    );
}

#[test]
fn warm_and_both_searches() {
    use ngboost_rs::{KSearch, NGBAutoMixtureRegressor};
    let (x, y) = bimodal(1500, 61);
    let (xv, yv) = bimodal(500, 62);
    let mut w = NGBAutoMixtureRegressor::new(200, 0.05);
    w.random_state = Some(7);
    w.search = KSearch::Warm;
    w.candidates = vec![1, 2, 3];
    w.fit(&x, &y, &xv, &yv).unwrap();
    assert_eq!(w.tried.len(), 3);
    assert!(w.tried[1].1 && w.tried[2].1, "K = 2, 3 are warm-grown");
    // split growth escapes the single-Normal solution on bimodal data
    assert!(w.tried[1].2 < w.tried[0].2 - 0.1, "{:?}", w.tried);
    // Both: cold K = 1..3 plus warm K = 2..3, the selection is the overall min
    let mut b = NGBAutoMixtureRegressor::new(200, 0.05);
    b.random_state = Some(7);
    b.search = KSearch::Both;
    b.candidates = vec![1, 2, 3];
    b.fit(&x, &y, &xv, &yv).unwrap();
    assert_eq!(b.tried.len(), 5);
    let best = b.tried.iter().min_by(|p, q| p.2.total_cmp(&q.2)).unwrap();
    assert_eq!(b.n_components(), Some(best.0));
    assert_eq!(b.selected_is_warm(), best.1);
    // warm search needs contiguous candidates
    let mut bad = NGBAutoMixtureRegressor::new(50, 0.05);
    bad.search = KSearch::Warm;
    bad.candidates = vec![1, 3];
    assert!(bad.fit(&x, &y, &xv, &yv).is_err());
}
