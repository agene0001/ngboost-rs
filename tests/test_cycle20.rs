//! research Cycle 20 (1.3.0): predicted means stay sane on the heads whose
//! mean is steeply nonlinear in a scale-like parameter — `Gamma` with the
//! default learner, the survival wrapper's `LogNormal`, and `LogNormal`
//! itself with and without cross-fitted residuals — and `HurdleLogNormal`,
//! whose `predict` is bounded by the largest training target (Cycle 21:
//! `NGBoost::target_max`, a stored number; pre-1.3.0 files still load).
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::dist::{Distribution, DistributionMethods, Gamma, HurdleLogNormal, LogNormal};
use ngboost_rs::learners::{BaseLearner, HistogramLearner};
use ngboost_rs::ngboost::{NGBoost, SerializedNGBoost};
use ngboost_rs::scores::{LogScore, LogScoreCensored, Scorable};
use ngboost_rs::survival::NGBSurvival;
use ngboost_rs::{LearningRateSchedule, LineSearchMethod, NGBRegressor};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next().max(1e-12);
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// A bounded positive target that is roughly Gaussian on its own scale and
/// therefore heavy in its *lower* tail on the log scale (the shape of
/// kin8nm, where the LogNormal mean exploded): values in [0.04, ~1.5].
fn bounded(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 5), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let f = v[0].sin() + 0.5 * v[1] * v[2] - 0.3 * v[3].abs();
        (0.7 + 0.25 * f + 0.12 * r.gauss()).max(0.04)
    });
    (x, y)
}

/// Gamma target whose shape follows the feature that drives the mean (the
/// benchmark's `gamma_shared`, where the `Gamma` head once predicted a
/// mean of 2800 for a target below 40).
fn gamma_shared(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 5), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let mean = (0.5 + 0.6 * v[0].sin() + 0.3 * v[1] * v[1] - 0.3 * v[2] * v[3]).exp();
        // integer shape 1…5 rising with the mean's dominant feature;
        // Gamma(k) as a sum of k exponentials
        let k = (3.0 + v[0]).round().clamp(1.0, 5.0) as usize;
        let g: f64 = (0..k).map(|_| -r.next().max(1e-12).ln()).sum();
        mean * g / k as f64
    });
    (x, y)
}

fn model<D, B>(learner: B, seed: u64) -> NGBoost<D, LogScore, B>
where
    D: Distribution + Scorable<LogScore> + Clone,
    B: BaseLearner + Clone,
{
    NGBoost::with_full_options(
        2000,
        0.05,
        learner,
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
    )
}

fn max(v: &Array1<f64>) -> f64 {
    v.iter().cloned().fold(f64::MIN, f64::max)
}

/// The property: no predicted mean beyond three times the largest target
/// ever seen. Every blow-up recorded in research Cycles 18–20 violated it
/// by a factor of 20 to 10²⁴.
fn assert_sane(what: &str, mean: &Array1<f64>, y_train: &Array1<f64>) {
    assert!(
        mean.iter().all(|m| m.is_finite()),
        "{what}: non-finite mean"
    );
    let (top, cap) = (max(mean), 3.0 * max(y_train));
    assert!(
        top <= cap,
        "{what}: predicted mean {top} against a cap of {cap}"
    );
}

#[test]
fn gamma_mean_stays_sane_with_the_default_learner() {
    for seed in 0..6 {
        let (x, y) = gamma_shared(3000, 100 + seed);
        let (xt, _) = gamma_shared(1500, 200 + seed);
        let mut m = model::<Gamma, _>(HistogramLearner::new(3), seed);
        m.fit(&x, &y).unwrap();
        assert_sane(&format!("Gamma, seed {seed}"), &m.predict(&xt), &y);
    }
}

#[test]
fn lognormal_family_means_stay_sane_on_a_bounded_target() {
    let mut worst_plain_mean = 0.0_f64;
    for seed in 0..3 {
        let (x, y) = bounded(3000, 300 + seed);
        let (xt, _) = bounded(2000, 400 + seed);

        // LogNormal, default staged fit and cross-fitted: bounded by the
        // scale ceiling
        for folds in [0usize, 5] {
            let mut m = model::<LogNormal, _>(HistogramLearner::new(3), seed);
            m.staged_crossfit_folds = folds;
            m.fit(&x, &y).unwrap();
            assert_sane(
                &format!("LogNormal, folds {folds}, seed {seed}"),
                &m.predict(&xt),
                &y,
            );
        }

        // HurdleLogNormal (a joint fit): a fifth of the targets set to 0
        let mut r = Lcg(500 + seed);
        let yz = y.mapv(|v| if r.next() < 0.2 { 0.0 } else { v });
        let mut h = model::<HurdleLogNormal, _>(HistogramLearner::new(3), seed);
        h.fit(&x, &yz).unwrap();
        // The hurdle's joint fit has no scale ceiling, and with this
        // target's point mass at its floor one fit in four stops early with
        // σ ≈ 3.4 on a few rows: the plain mean is then about twenty times
        // the largest target. `predict` is bounded by that target instead.
        let cap = h.target_max.unwrap();
        assert_eq!(cap, max(&yz));
        let (pred, mean) = (h.predict(&xt), h.pred_dist(&xt).mean());
        assert_sane(&format!("HurdleLogNormal, seed {seed}"), &pred, &yz);
        for i in 0..pred.len() {
            assert!(pred[i] >= 0.0 && pred[i] <= cap, "row {i}: {}", pred[i]);
            // never above the plain mean, and equal to it where the cap is
            // far in the tail
            assert!(pred[i] <= mean[i] * (1.0 + 1e-12), "row {i}");
        }
        worst_plain_mean = worst_plain_mean.max(max(&mean) / cap);
        let staged = h.staged_predict(&xt);
        assert_eq!(staged.last().unwrap().to_vec(), pred.to_vec());

        // the survival wrapper (its own joint fit), 30 % of the times
        // censored at a uniform fraction of the event time
        let mut r = Lcg(600 + seed);
        let mut time = y.clone();
        let mut event = Array1::ones(y.len());
        for i in 0..y.len() {
            if r.next() < 0.3 {
                time[i] = y[i] * (0.3 + 0.7 * r.next());
                event[i] = 0.0;
            }
        }
        let mut s = NGBSurvival::<LogNormal, LogScoreCensored, HistogramLearner>::new(
            2000,
            0.05,
            HistogramLearner::new(3),
        );
        s.early_stopping_rounds = Some(50);
        s.fit(&x, &time, &event).unwrap();
        assert_sane(
            &format!("survival LogNormal, seed {seed}"),
            &s.predict(&xt),
            &y,
        );
    }
    // the data does reproduce the failure the bound is there for
    assert!(
        worst_plain_mean > 5.0,
        "plain hurdle mean never exceeded 5× the largest target ({worst_plain_mean})"
    );
}

/// E[min(Y, c)] by a midpoint rule over the quantile function.
fn winsorized_numeric(mu: f64, sigma: f64, cap: f64) -> f64 {
    let d = LogNormal::from_params(&ndarray::arr2(&[[mu, sigma.ln()]]));
    let n = 200_000;
    let u = Array1::from_shape_fn(n, |i| (i as f64 + 0.5) / n as f64);
    // one row per call keeps the distribution scalar
    let mut total = 0.0;
    for &q in u.iter() {
        total += d.ppf(&Array1::from_elem(1, q))[0].min(cap);
    }
    total / n as f64
}

#[test]
fn lognormal_winsorized_mean_is_exact_and_stable() {
    for (mu, sigma, cap) in [
        (0.0, 0.5, 3.0),
        (-1.0, 1.0, 1.5),
        (0.3, 0.2, 1.0),
        (-0.5, 2.0, 2.0),
        (1.0, 0.8, 0.5),
    ] {
        let d = LogNormal::from_params(&ndarray::arr2(&[[mu, f64::ln(sigma)]]));
        let got = d.winsorized_mean(cap).unwrap()[0];
        let want = winsorized_numeric(mu, sigma, cap);
        assert!(
            (got - want).abs() < 2e-3 * want.max(1e-3),
            "μ {mu} σ {sigma} cap {cap}: {got} vs numeric {want}"
        );
        // bounded by both the cap and the plain mean
        assert!(got <= cap && got <= d.predict()[0] * (1.0 + 1e-12));
    }
    // the plain mean as the cap grows; monotone in the cap
    let d = LogNormal::from_params(&ndarray::arr2(&[[0.2, f64::ln(0.7)]]));
    let mean = d.predict()[0];
    assert!((d.winsorized_mean(1e9).unwrap()[0] - mean).abs() < 1e-9 * mean);
    assert_eq!(d.winsorized_mean(f64::INFINITY).unwrap()[0], mean);
    let caps = [0.1, 0.5, 1.0, 2.0, 5.0, 50.0];
    let vals: Vec<f64> = caps
        .iter()
        .map(|&c| d.winsorized_mean(c).unwrap()[0])
        .collect();
    assert!(vals.windows(2).all(|w| w[0] < w[1]));
    // a scale for which the plain mean overflows still gives a finite,
    // bounded value
    let wild = LogNormal::from_params(&ndarray::arr2(&[[-1.0, f64::ln(40.0)]]));
    assert!(wild.predict()[0].is_infinite());
    let w = wild.winsorized_mean(2.0).unwrap()[0];
    assert!(w.is_finite() && w > 0.0 && w <= 2.0, "{w}");
    // no cap, no value; and the heads without one decline
    assert!(d.winsorized_mean(0.0).is_none());
    assert!(
        d.predict_capped(1.0).is_none(),
        "plain LogNormal keeps its mean"
    );
    assert!(
        Gamma::from_params(&ndarray::arr2(&[[0.0, 0.0]]))
            .winsorized_mean(1.0)
            .is_none()
    );
}

#[test]
fn target_max_is_stored_and_old_model_files_still_load() {
    let (x, y) = bounded(1500, 900);
    let (xt, _) = bounded(300, 901);
    let mut r = Lcg(902);
    let yz = y.mapv(|v| if r.next() < 0.2 { 0.0 } else { v });
    let mut h = model::<HurdleLogNormal, _>(HistogramLearner::new(3), 3);
    h.fit(&x, &yz).unwrap();
    // the largest target of the whole fit call, validation rows included
    assert_eq!(h.target_max, Some(max(&yz)));
    let pred = h.predict(&xt);

    // current format: the number travels with the model
    let ser = h.serialize().unwrap();
    assert_eq!(ser.target_max, h.target_max);
    let bytes = bincode::serialize(&ser).unwrap();
    let back = SerializedNGBoost::from_bincode(&bytes).unwrap();
    let h2 = NGBoost::<HurdleLogNormal, LogScore, HistogramLearner>::deserialize(
        back,
        HistogramLearner::new(3),
    )
    .unwrap();
    assert_eq!(h2.target_max, h.target_max);
    assert!(
        h2.predict(&xt)
            .iter()
            .zip(pred.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // a file written before 1.3.0 ends where the new field begins
    // (`Some(f64)` is the last 9 bytes): plain bincode refuses it, the
    // helper reads it, and the model falls back to the plain mean
    let old = &bytes[..bytes.len() - 9];
    assert!(bincode::deserialize::<SerializedNGBoost>(old).is_err());
    let legacy = SerializedNGBoost::from_bincode(old).unwrap();
    assert_eq!(legacy.target_max, None);
    let h3 = NGBoost::<HurdleLogNormal, LogScore, HistogramLearner>::deserialize(
        legacy,
        HistogramLearner::new(3),
    )
    .unwrap();
    assert_eq!(
        h3.predict(&xt).to_vec(),
        h.pred_dist(&xt).predict().to_vec()
    );
    assert!(
        h3.pred_param(&xt)
            .iter()
            .zip(h.pred_param(&xt).iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    // garbage is still an error, not a silently padded model
    assert!(SerializedNGBoost::from_bincode(&bytes[..bytes.len() / 2]).is_err());
    // an older reader ignores the trailing field: plain bincode accepts
    // bytes after the last field it knows
    let mut longer = bytes.clone();
    longer.extend_from_slice(&[1, 2, 3]);
    assert!(bincode::deserialize::<SerializedNGBoost>(&longer).is_ok());

    // self-describing formats: a missing field is the default
    let mut json = serde_json::to_value(&ser).unwrap();
    json.as_object_mut().unwrap().remove("target_max").unwrap();
    let from_json: SerializedNGBoost = serde_json::from_value(json).unwrap();
    assert_eq!(from_json.target_max, None);

    // the file API: a truncated (pre-1.3.0) file loads and predicts as before
    let mut reg = NGBRegressor::new(60, 0.1);
    reg.fit(&x, &y).unwrap();
    let before = reg.predict(&xt);
    let path = std::env::temp_dir().join(format!("ngboost_rs_c20_{}.bin", std::process::id()));
    let path = path.to_str().unwrap().to_string();
    reg.save_model(&path).unwrap();
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len() - 9]).unwrap();
    let loaded = NGBRegressor::load_model(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(
        loaded
            .predict(&xt)
            .iter()
            .zip(before.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );

    // a continued fit keeps the larger of the old and the new maximum
    let mut m = model::<Gamma, _>(HistogramLearner::new(3), 1);
    let (xg, yg) = gamma_shared(600, 77);
    m.fit(&xg, &yg).unwrap();
    let first = m.target_max.unwrap();
    m.partial_fit(&xg, &yg.mapv(|v| 0.5 * v)).unwrap();
    assert_eq!(m.target_max, Some(first));
    m.partial_fit(&xg, &yg.mapv(|v| 2.0 * v)).unwrap();
    assert_eq!(m.target_max, Some(2.0 * first));
}
