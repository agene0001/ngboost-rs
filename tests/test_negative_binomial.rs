//! NegativeBinomial head: training behaviour vs the Poisson head.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::dist::{DistributionMethods, NegativeBinomial, Poisson};
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
    fn gamma(&mut self, r: f64) -> f64 {
        let d = r - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.gauss();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next().max(1e-300);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
    fn poisson(&mut self, lam: f64) -> f64 {
        let l = (-lam).exp();
        let (mut k, mut p) = (0u32, 1.0);
        loop {
            k += 1;
            p *= self.next();
            if p <= l {
                return (k - 1) as f64;
            }
        }
    }
}

/// Over-dispersed counts: μ(x) nonlinear in 1–40, Gamma–Poisson with r.
fn nb_data(n: usize, seed: u64, r: Option<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let x = Array2::from_shape_fn((n, 5), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let row = x.row(i);
        let mu = (2.0 + 0.8 * (2.0 * row[0]).sin() + 0.25 * row[1] * row[1]).exp();
        let lam = match r {
            Some(rr) => mu * rng.gamma(rr) / rr,
            None => mu,
        };
        rng.poisson(lam)
    });
    (x, y)
}

fn nb_model(seed: u64) -> NGBoost<NegativeBinomial, LogScore, HistogramLearner> {
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
fn pois_model(seed: u64) -> NGBoost<Poisson, LogScore, HistogramLearner> {
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
fn negbin_beats_poisson_on_overdispersed_counts() {
    let (x, y) = nb_data(2500, 1, Some(4.0));
    let (xv, yv) = nb_data(500, 2, Some(4.0));
    let (xt, yt) = nb_data(1500, 3, Some(4.0));
    let mut nb = nb_model(7);
    nb.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let mut po = pois_model(7);
    po.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (nll_nb, nll_po) = (nb.score(&xt, &yt), po.score(&xt, &yt));
    assert!(
        nll_nb < nll_po - 0.1,
        "NB {nll_nb} should clearly beat Poisson {nll_po}"
    );
    // learned dispersion in the right ballpark (true r = 4), and finite params
    let d = nb.pred_dist(&xt);
    let r_geo = d.r.mapv(f64::ln).mean().unwrap().exp();
    assert!(r_geo > 1.5 && r_geo < 12.0, "geometric mean r̂ = {r_geo}");
    assert!(d.mu.iter().all(|v| v.is_finite()) && d.r.iter().all(|v| v.is_finite()));
    // dispersion index relative to the predicted variance ≈ 1
    let var = d.variance();
    let disp = (0..yt.len())
        .map(|i| (yt[i] - d.mu[i]).powi(2) / var[i])
        .sum::<f64>()
        / yt.len() as f64;
    assert!(disp > 0.7 && disp < 1.4, "dispersion index {disp}");
}

#[test]
fn negbin_is_not_much_worse_than_poisson_on_true_poisson_counts() {
    // the weakly identified regime (r → ∞): the head must still learn μ
    let (x, y) = nb_data(2500, 11, None);
    let (xv, yv) = nb_data(500, 12, None);
    let (xt, yt) = nb_data(1500, 13, None);
    let mut nb = nb_model(7);
    nb.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let mut po = pois_model(7);
    po.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (nll_nb, nll_po) = (nb.score(&xt, &yt), po.score(&xt, &yt));
    assert!(nll_nb < nll_po + 0.03, "NB {nll_nb} vs Poisson {nll_po}");
    let rmse = |p: &Array1<f64>| (p - &yt).mapv(|e| e * e).mean().unwrap().sqrt();
    let (r_nb, r_po) = (rmse(&nb.predict(&xt)), rmse(&po.predict(&xt)));
    assert!(r_nb < r_po * 1.05, "RMSE NB {r_nb} vs Poisson {r_po}");
}

#[test]
fn negbin_serialization_round_trip() {
    let (x, y) = nb_data(600, 21, Some(3.0));
    let mut nb = nb_model(7);
    nb.n_estimators = 40;
    nb.early_stopping_rounds = None;
    nb.fit(&x, &y).unwrap();
    let ser = nb.serialize().unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let back: ngboost_rs::ngboost::SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 = NGBoost::<NegativeBinomial, LogScore, HistogramLearner>::deserialize(
        back,
        HistogramLearner::new(3),
    )
    .unwrap();
    let (a, b) = (nb.pred_param(&x), m2.pred_param(&x));
    for (u, v) in a.iter().zip(b.iter()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
}

#[test]
fn negbin_auto_calibrates_log_r_on_validation_only() {
    // Cycle 16: the default end-of-fit recalibration maps log r ← a + b·log r
    // on the validation rows; μ is untouched and the val NLL cannot rise.
    let (x, y) = nb_data(2000, 31, Some(4.0));
    let (xv, yv) = nb_data(400, 32, Some(4.0));
    let mut cal = nb_model(7);
    cal.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (col, _a, b) = cal.scale_calibration.expect("log r should be recalibrated");
    assert_eq!(col, 1);
    assert!(b > 0.0);
    let mut raw = nb_model(7);
    raw.auto_calibrate_scale = false;
    raw.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(raw.scale_calibration.is_none());
    let (pc, pr) = (cal.pred_param(&xv), raw.pred_param(&xv));
    for (u, v) in pc.column(0).iter().zip(pr.column(0).iter()) {
        assert_eq!(u.to_bits(), v.to_bits(), "log μ must not move");
    }
    assert!(cal.score(&xv, &yv) <= raw.score(&xv, &yv) + 1e-12);
    // fewer than 300 validation rows: the dispersion map is not fitted
    let (xs, ys) = nb_data(150, 33, Some(4.0));
    let mut small = nb_model(7);
    small
        .fit_with_validation(&x, &y, Some(&xs), Some(&ys), None, None)
        .unwrap();
    assert!(small.scale_calibration.is_none());
    // without validation data nothing is calibrated
    let mut nov = nb_model(7);
    nov.early_stopping_rounds = None;
    nov.n_estimators = 50;
    nov.fit(&x, &y).unwrap();
    assert!(nov.scale_calibration.is_none());
}

#[test]
fn negbin_staged_fit_is_opt_in_and_sound() {
    let (x, y) = nb_data(2000, 41, Some(4.0));
    let (xv, yv) = nb_data(400, 42, Some(4.0));
    let (xt, yt) = nb_data(1500, 43, Some(4.0));
    let joint = nb_model(7);
    assert!(
        !joint.uses_staged_fit(2000, Some(400)),
        "staged is not the NB default"
    );
    let mut st = nb_model(7);
    st.staged_fit = Some(true);
    assert!(st.uses_staged_fit(2000, Some(400)));
    st.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let p = st.pred_param(&xt);
    assert!(p.iter().all(|v| v.is_finite()));
    let mut jm = nb_model(7);
    jm.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (s_st, s_j) = (st.score(&xt, &yt), jm.score(&xt, &yt));
    assert!(s_st < s_j + 0.03, "staged {s_st} vs joint {s_j}");
}
