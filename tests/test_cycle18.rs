//! research Cycle 18 (1.3.0): `HistogramLearner` leaf default, cross-fitted
//! staged fit, `GammaMean`, `StudentTFiniteVar`.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::dist::{
    Distribution, DistributionMethods, Gamma, GammaMean, Normal, StudentT, StudentTFiniteVar,
};
use ngboost_rs::learners::{BaseLearner, HistogramLearner};
use ngboost_rs::ngboost::{NGBoost, SerializedNGBoost};
use ngboost_rs::scores::{CRPScore, LogScore, Scorable, Score};
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

/// Nonlinear mean, σ depends on x (the benchmark's `het_a` shape).
fn het(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 6), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let sd = (0.5 * v[4]).exp() * 0.7;
        2.0 * v[0].sin() + v[1] * v[1] - v[2] * v[3] + sd * r.gauss()
    });
    (x, y)
}

/// Heavy-tailed noise (t with 2.5 degrees of freedom) around a smooth mean.
fn heavy(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        // t_ν = z / sqrt(χ²_ν / ν), χ² by a sum of squares plus a half step
        let z = r.gauss();
        let chi = r.gauss().powi(2) + r.gauss().powi(2) + 0.5 * r.gauss().powi(2);
        1.5 * v[0] + v[1].sin() + 0.6 * z / (chi / 2.5).sqrt()
    });
    (x, y)
}

/// Gamma-distributed target with mean exp(…) and shape 4.
fn gamma_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| r.next() * 2.0 - 1.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let mu = (0.8 * v[0] + 0.5 * v[1] * v[2]).exp();
        // Gamma(4, ·) as a sum of four exponentials
        let g: f64 = (0..4).map(|_| -r.next().max(1e-12).ln()).sum();
        mu * g / 4.0
    });
    (x, y)
}

fn model<D, S, B>(learner: B, patience: Option<u32>, seed: u64) -> NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    NGBoost::with_full_options(
        1500,
        0.05,
        learner,
        true,
        1.0,
        1.0,
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

fn nll<D, B>(m: &NGBoost<D, LogScore, B>, x: &Array2<f64>, y: &Array1<f64>) -> f64
where
    D: Distribution + Scorable<LogScore> + Clone,
    B: BaseLearner + Clone,
{
    Scorable::<LogScore>::total_score(&m.pred_dist(x), y, None)
}

fn bits_equal(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    a.dim() == b.dim()
        && a.iter()
            .zip(b.iter())
            .all(|(p, q)| p.to_bits() == q.to_bits())
}

#[test]
fn histogram_learner_default_leaf_is_twenty() {
    assert_eq!(HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF, 20);
    assert_eq!(HistogramLearner::new(3).min_samples_leaf, 20);
    assert_eq!(HistogramLearner::default_histogram().min_samples_leaf, 20);
    // explicit values are untouched (1 reproduces pre-1.3 fits)
    assert_eq!(
        HistogramLearner::with_params(3, 255, 1, 2).min_samples_leaf,
        1
    );
    // the default is a real constraint: no leaf of a fitted tree is
    // smaller, seen as at most n / 20 distinct predictions
    let (x, y) = het(400, 11);
    let tree = HistogramLearner::new(6).fit(&x, &y).unwrap();
    let mut preds: Vec<u64> = tree.predict(&x).iter().map(|v| v.to_bits()).collect();
    preds.sort_unstable();
    preds.dedup();
    assert!(
        preds.len() <= 400 / 20,
        "{} leaves on 400 rows",
        preds.len()
    );
    for p in &preds {
        let rows = tree
            .predict(&x)
            .iter()
            .filter(|v| v.to_bits() == *p)
            .count();
        assert!(rows >= 20, "leaf with {rows} rows");
    }
}

#[test]
fn crossfit_off_is_the_previous_staged_fit() {
    let (x, y) = het(1500, 1);
    let (xv, yv) = het(300, 2);
    let mut a = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    a.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    // 1 fold is "off" as well
    let mut b = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    b.staged_crossfit_folds = 1;
    b.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(bits_equal(&a.pred_param(&xv), &b.pred_param(&xv)));
}

#[test]
fn crossfit_scale_is_honest_without_recalibration() {
    let (x, y) = het(2000, 1);
    let (xv, yv) = het(400, 2);
    let (xt, yt) = het(4000, 3);
    let fit = |folds: usize| {
        let mut m = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
        m.staged_fit = Some(true);
        m.auto_calibrate_scale = false;
        m.staged_crossfit_folds = folds;
        m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
            .unwrap();
        m
    };
    let (plain, cross) = (fit(0), fit(5));
    // 90 % central interval coverage on fresh data
    let cover = |m: &NGBoost<Normal, LogScore, HistogramLearner>| {
        let d = m.pred_dist(&xt);
        let (lo, hi) = d.interval(0.1);
        (0..yt.len())
            .filter(|&i| yt[i] >= lo[i] && yt[i] <= hi[i])
            .count() as f64
            / yt.len() as f64
    };
    let (cp, cc) = (cover(&plain), cover(&cross));
    let (np, nc) = (nll(&plain, &xt, &yt), nll(&cross, &xt, &yt));
    assert!(
        (cc - 0.9).abs() < (cp - 0.9).abs(),
        "coverage: cross-fit {cc} vs in-sample {cp}"
    );
    assert!(cc > 0.85, "cross-fit coverage {cc}");
    assert!(nc < np, "NLL: cross-fit {nc} vs in-sample {np}");
    assert!(cross.scale_calibration.is_none());

    // same seed, same model; and an ordinary model: the format round-trips
    let again = fit(5);
    assert!(bits_equal(&cross.pred_param(&xt), &again.pred_param(&xt)));
    assert!(cross.base_models.iter().all(|r| r.len() == 2));
    let bytes = bincode::serialize(&cross.serialize().unwrap()).unwrap();
    let back: SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 =
        NGBoost::<Normal, LogScore, HistogramLearner>::deserialize(back, HistogramLearner::new(3))
            .unwrap();
    assert!(bits_equal(&cross.pred_param(&xt), &m2.pred_param(&xt)));
}

#[test]
fn crossfit_lifts_the_recalibration_condition_and_handles_edge_cases() {
    let mut m = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    // 30 validation rows: too few to recalibrate, so the default is joint …
    assert!(!m.uses_staged_fit(1000, Some(30)));
    // … unless the residuals are cross-fitted
    m.staged_crossfit_folds = 5;
    assert!(m.uses_staged_fit(1000, Some(30)));
    // still needs early stopping and validation data
    let mut no_es = model::<Normal, LogScore, _>(HistogramLearner::new(3), None, 5);
    no_es.staged_crossfit_folds = 5;
    assert!(!no_es.uses_staged_fit(1000, Some(30)));
    // an explicit joint fit wins
    m.staged_fit = Some(false);
    assert!(!m.uses_staged_fit(1000, Some(300)));

    // a small validation block fits and predicts finite values
    let (x, y) = het(600, 4);
    let (xv, yv) = het(30, 5);
    let mut m = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    m.staged_crossfit_folds = 5;
    m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(m.pred_param(&xv).iter().all(|v| v.is_finite()));
    assert!(m.scale_calibration.is_none());

    // sample weights and the automatic validation split
    let w = Array1::from_shape_fn(600, |i| 0.5 + (i % 3) as f64 * 0.5);
    let mut mw = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    mw.staged_crossfit_folds = 4;
    mw.fit_with_validation(&x, &y, None, None, Some(&w), None)
        .unwrap();
    assert!(mw.pred_param(&xv).iter().all(|v| v.is_finite()));

    // a later partial_fit is an ordinary joint continuation (no stale
    // cross-fit state): it must run and keep predictions finite
    mw.n_estimators = 5;
    mw.early_stopping_rounds = None;
    mw.partial_fit(&x, &y).unwrap();
    assert!(mw.pred_param(&xv).iter().all(|v| v.is_finite()));

    // a decaying learning-rate schedule: the fold fits replay the first
    // rounds of the same schedule (same n_estimators), and the fit is
    // reproducible
    let sched = || {
        let mut m = model::<Normal, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
        m.lr_schedule = LearningRateSchedule::Linear {
            decay_rate: 0.7,
            min_lr_fraction: 0.1,
        };
        m.staged_crossfit_folds = 3;
        m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
            .unwrap();
        m.pred_param(&xv)
    };
    let (p1, p2) = (sched(), sched());
    assert!(p1.iter().all(|v| v.is_finite()));
    assert!(bits_equal(&p1, &p2));

    // fewer than two rows per fold is an error, not a silent in-sample fit
    let (xs, ys) = het(12, 6);
    let (xsv, ysv) = het(8, 7);
    let mut tiny =
        model::<Normal, LogScore, _>(HistogramLearner::with_params(2, 255, 1, 2), Some(5), 5);
    tiny.staged_crossfit_folds = 10;
    assert!(
        tiny.fit_with_validation(&xs, &ys, Some(&xsv), Some(&ysv), None, None)
            .is_err()
    );

    // the sklearn-style parameter struct carries the option
    let mut reg: NGBRegressor = NGBRegressor::new(50, 0.1);
    let mut p = reg.get_params();
    assert_eq!(p.staged_crossfit_folds, 0);
    p.staged_crossfit_folds = 5;
    reg.set_params(p);
    assert_eq!(reg.get_params().staged_crossfit_folds, 5);
}

#[test]
fn student_t_finite_var_never_predicts_an_undefined_moment() {
    let (x, y) = heavy(2500, 1);
    let (xv, yv) = heavy(400, 2);
    let (xt, yt) = heavy(3000, 3);
    let mut fv = model::<StudentTFiniteVar, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    fv.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let d = fv.pred_dist(&xt);
    assert!(
        d.df().iter().all(|&v| v > 2.0),
        "min ν {:?}",
        d.df().iter().cloned().fold(f64::INFINITY, f64::min)
    );
    assert!(d.variance().iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(d.mean().iter().all(|v| v.is_finite()));
    let crps = Scorable::<CRPScore>::score(&d, &yt);
    assert!(crps.iter().all(|v| v.is_finite()));

    // no worse than the unconstrained head where that one is well defined
    let mut t = model::<StudentT, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    t.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (a, b) = (nll(&fv, &xt, &yt), nll(&t, &xt, &yt));
    assert!(a < b + 0.03, "bounded {a} vs unconstrained {b}");

    // the bound is part of the type: the saved model reloads bit-identically
    let bytes = bincode::serialize(&fv.serialize().unwrap()).unwrap();
    let back: SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 = NGBoost::<StudentTFiniteVar, LogScore, HistogramLearner>::deserialize(
        back,
        HistogramLearner::new(3),
    )
    .unwrap();
    assert!(bits_equal(&fv.pred_param(&xt), &m2.pred_param(&xt)));

    // CRPS training works through the chain rule as well
    let mut c = model::<StudentTFiniteVar, CRPScore, _>(HistogramLearner::new(3), Some(20), 5);
    c.n_estimators = 60;
    c.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(c.pred_dist(&xt).df().iter().all(|&v| v > 2.0));
}

#[test]
fn gamma_mean_fits_staged_by_default_and_matches_or_beats_gamma() {
    let (x, y) = gamma_data(3000, 1);
    let (xv, yv) = gamma_data(400, 2);
    let (xt, yt) = gamma_data(3000, 3);
    let mut gm = model::<GammaMean, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    // staged by default once the log α recalibration can run (≥ 300 rows)
    assert!(gm.uses_staged_fit(3000, Some(400)));
    assert!(!gm.uses_staged_fit(3000, Some(100)));
    gm.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(gm.scale_calibration.is_some_and(|(c, _, _)| c == 1));
    let mut g = model::<Gamma, LogScore, _>(HistogramLearner::new(3), Some(40), 5);
    g.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (a, b) = (nll(&gm, &xt, &yt), nll(&g, &xt, &yt));
    assert!(a < b + 0.01, "GammaMean {a} vs Gamma {b}");
    let d = gm.pred_dist(&xt);
    assert!(d.mu.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(d.shape.iter().all(|v| v.is_finite() && *v > 0.0));

    // CRPS: trains, and scores exactly as the equivalent (shape, rate) Gamma
    let mut c = model::<GammaMean, CRPScore, _>(HistogramLearner::new(3), Some(20), 5);
    c.n_estimators = 60;
    c.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let dc = c.pred_dist(&xt);
    let (s1, s2) = (
        Scorable::<CRPScore>::score(&dc, &yt),
        Scorable::<CRPScore>::score(dc.as_gamma(), &yt),
    );
    assert!(
        s1.iter()
            .zip(s2.iter())
            .all(|(p, q)| p == q && p.is_finite())
    );
    let init = Scorable::<CRPScore>::total_score(
        &GammaMean::from_params(
            &GammaMean::fit(&y)
                .insert_axis(ndarray::Axis(0))
                .broadcast((yt.len(), 2))
                .unwrap()
                .to_owned(),
        ),
        &yt,
        None,
    );
    let fitted = Scorable::<CRPScore>::total_score(&dc, &yt, None);
    assert!(fitted < init, "CRPS fit {fitted} vs marginal {init}");
}
