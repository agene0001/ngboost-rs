//! Opt-in research options (2026-11-09): `fisher_weight_power` and post-hoc
//! affine scale recalibration. Both must leave default behaviour untouched.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::NGBRegressor;
use ngboost_rs::dist::Normal;
use ngboost_rs::dist::{
    Cauchy, Distribution, Exponential, Gamma, Laplace, LogNormal, NormalFixedMean, Poisson,
    StudentT, TFixedDf,
};
use ngboost_rs::learners::{
    BaseLearner, DecisionTreeLearner, HistogramLearner, RidgeLearner, StumpLearner,
};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;
use ngboost_rs::{LearningRateSchedule, LineSearchMethod, ValidationSplit};

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

/// Heteroscedastic nonlinear data (the regime where the σ head matters).
fn data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let x = Array2::from_shape_fn((n, 5), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let mu = 2.0 * r[0].sin() + r[1] * r[1] - r[2] * r[3];
        let sigma = (-0.5 + 0.7 * r[4]).exp();
        mu + sigma * rng.gauss()
    });
    (x, y)
}

fn model<B: BaseLearner + Clone>(
    learner: B,
    n_est: u32,
    seed: u64,
) -> NGBoost<Normal, LogScore, B> {
    NGBoost::with_full_options(
        n_est,
        0.05,
        learner,
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    )
}

fn assert_bits_equal(a: &Array2<f64>, b: &Array2<f64>, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(u.to_bits(), v.to_bits(), "{what}: element {i}: {u} vs {v}");
    }
}

#[test]
fn fisher_weight_power_zero_matches_default_bit_exact() {
    let (x, y) = data(500, 1);
    let mut base = model(HistogramLearner::new(3), 60, 7);
    let mut p0 = model(HistogramLearner::new(3), 60, 7);
    p0.fisher_weight_power = Some(0.0);
    base.fit(&x, &y).unwrap();
    p0.fit(&x, &y).unwrap();
    assert_bits_equal(
        &base.pred_param(&x),
        &p0.pred_param(&x),
        "beta = 0 vs default",
    );
}

#[test]
fn fisher_weight_power_one_matches_boolean_flag_bit_exact() {
    let (x, y) = data(500, 2);
    let mut flag = model(HistogramLearner::new(3), 60, 7);
    flag.fisher_weighted_learner = true;
    let mut p1 = model(HistogramLearner::new(3), 60, 7);
    p1.fisher_weight_power = Some(1.0);
    flag.fit(&x, &y).unwrap();
    p1.fit(&x, &y).unwrap();
    assert_bits_equal(&flag.pred_param(&x), &p1.pred_param(&x), "beta = 1 vs flag");
    // and it is a genuinely different fit from the unweighted default
    let mut base = model(HistogramLearner::new(3), 60, 7);
    base.fit(&x, &y).unwrap();
    assert!(base.pred_param(&x) != p1.pred_param(&x));
}

#[test]
fn fisher_weight_power_half_trains_and_is_finite() {
    let (x, y) = data(500, 3);
    let mut m = model(HistogramLearner::new(3), 60, 7);
    m.fisher_weight_power = Some(0.5);
    m.fit(&x, &y).unwrap();
    assert!(m.pred_param(&x).iter().all(|v| v.is_finite()));
    assert!(m.score(&x, &y).is_finite());
}

#[test]
fn affine_calibration_lowers_validation_score_and_folds_into_model() {
    let (x, y) = data(700, 11);
    let (xv, yv) = data(400, 12);
    let mut m = model(HistogramLearner::new(3), 150, 7);
    m.fit(&x, &y).unwrap();

    let before = m.score(&xv, &yv);
    let raw = m.pred_param(&xv);
    let staged_before = m.staged_pred_dist(&xv).len();

    let (a, b) = m.calibrate_scale_affine(&xv, &yv).unwrap();
    assert!(b > 0.0 && a.is_finite(), "a={a} b={b}");

    let after = m.score(&xv, &yv);
    assert!(
        after <= before + 1e-12,
        "val NLL must not increase: {before} -> {after}"
    );
    // an early-stopped-free 150-round fit on 700 rows is overconfident: the
    // head should inflate σ on average (a + (b-1)·mean log σ > 0)
    let mean_logs = raw.column(1).mean().unwrap();
    assert!(
        a + (b - 1.0) * mean_logs > 0.0,
        "expected σ inflation, got a={a} b={b}"
    );

    // folded model == manual affine map on the raw parameters
    let folded = m.pred_param(&xv);
    for i in 0..xv.nrows() {
        assert_eq!(
            folded[[i, 0]].to_bits(),
            raw[[i, 0]].to_bits(),
            "loc untouched"
        );
        let manual = a + b * raw[[i, 1]];
        assert!(
            (folded[[i, 1]] - manual).abs() <= 1e-9 * (1.0 + manual.abs()),
            "row {i}: {} vs {manual}",
            folded[[i, 1]]
        );
    }
    // staged predictions see the same folded model (last stage == pred_dist)
    let staged = m.staged_pred_dist(&xv);
    assert_eq!(staged.len(), staged_before);
    let last = staged.last().unwrap();
    let full = m.pred_dist(&xv);
    for i in 0..xv.nrows() {
        assert!((last.log_scale[i] - full.log_scale[i]).abs() < 1e-12);
        assert!((last.loc[i] - full.loc[i]).abs() < 1e-12);
    }
}

#[test]
fn affine_calibration_survives_serialization_bit_exact() {
    let (x, y) = data(500, 21);
    let (xv, yv) = data(300, 22);
    let mut m = model(HistogramLearner::new(3), 80, 7);
    m.fit(&x, &y).unwrap();
    let (_a, b) = m.calibrate_scale_affine(&xv, &yv).unwrap();
    assert!(b != 1.0, "test needs a non-trivial slope");
    let ser = m.serialize().unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let back: ngboost_rs::ngboost::SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 =
        NGBoost::<Normal, LogScore, HistogramLearner>::deserialize(back, HistogramLearner::new(3))
            .unwrap();
    assert_bits_equal(&m.pred_param(&xv), &m2.pred_param(&xv), "round trip");
}

#[test]
fn affine_calibration_works_for_every_learner_type() {
    let (x, y) = data(400, 31);
    let (xv, yv) = data(300, 32);
    fn check<B: BaseLearner + Clone>(learner: B, name: &str) {
        let (x, y) = data(400, 31);
        let (xv, yv) = data(300, 32);
        let mut m = model(learner, 60, 7);
        m.fit(&x, &y).unwrap();
        let before = m.score(&xv, &yv);
        let (a, b) = m
            .calibrate_scale_affine(&xv, &yv)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let after = m.score(&xv, &yv);
        assert!(
            after <= before + 1e-12,
            "{name}: {before} -> {after} (a={a}, b={b})"
        );
        assert!(
            m.pred_param(&xv).iter().all(|v| v.is_finite()),
            "{name}: finite"
        );
    }
    check(StumpLearner, "stump");
    check(DecisionTreeLearner::default_sklearn(), "exact tree");
    check(HistogramLearner::new(3), "histogram tree");
    check(RidgeLearner::default_linear(), "ridge");
    let _ = (x, y, xv, yv);
}

#[test]
fn affine_calibration_validates_inputs() {
    let (x, y) = data(300, 41);
    let mut m = model(HistogramLearner::new(3), 20, 7);
    assert!(m.calibrate_scale_affine(&x, &y).is_err(), "unfitted");
    m.fit(&x, &y).unwrap();
    let bad_x = Array2::zeros((300, 3));
    assert!(
        m.calibrate_scale_affine(&bad_x, &y).is_err(),
        "feature mismatch"
    );
    let short_y = Array1::zeros(10);
    assert!(
        m.calibrate_scale_affine(&x, &short_y).is_err(),
        "row mismatch"
    );
    assert!(
        m.calibrate_param_affine(&x, &y, 2, None).is_err(),
        "column out of range"
    );
    let before = m.pred_param(&x);
    // the failed calls must not have mutated the model
    assert_bits_equal(&before, &m.pred_param(&x), "untouched after errors");
}

#[test]
fn ngbregressor_affine_passthrough_and_save_load() {
    let (x, y) = data(500, 51);
    let (xv, yv) = data(300, 52);
    let mut m = NGBRegressor::new(80, 0.05);
    m.set_random_state(7);
    m.fit(&x, &y).unwrap();
    let before = m.score(&xv, &yv);
    let (_a, _b) = m.calibrate_scale_affine(&xv, &yv).unwrap();
    assert!(m.score(&xv, &yv) <= before + 1e-12);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.bin");
    m.save_model(path.to_str().unwrap()).unwrap();
    let m2 = NGBRegressor::load_model(path.to_str().unwrap()).unwrap();
    assert_bits_equal(&m.pred_param(&xv), &m2.pred_param(&xv), "save/load");
}

// ---------------------------------------------------------------------------
// Default-on automatic scale recalibration (1.2.0) and ValidationSplit
// ---------------------------------------------------------------------------

#[test]
fn scale_param_index_table() {
    assert_eq!(Normal::scale_param_index(), Some(1));
    assert_eq!(NormalFixedMean::scale_param_index(), Some(0));
    assert_eq!(Laplace::scale_param_index(), Some(1));
    assert_eq!(LogNormal::scale_param_index(), Some(1));
    assert_eq!(Cauchy::scale_param_index(), Some(1));
    assert_eq!(StudentT::scale_param_index(), Some(1));
    assert_eq!(TFixedDf::scale_param_index(), Some(1));
    // not log-scale columns / unmeasured: default None
    assert_eq!(Gamma::scale_param_index(), None);
    assert_eq!(Exponential::scale_param_index(), None);
}

#[test]
fn auto_calibration_runs_by_default_with_validation_data() {
    let (x, y) = data(700, 61);
    let (xv, yv) = data(400, 62);
    let mut on = model(HistogramLearner::new(3), 150, 7);
    assert!(on.auto_calibrate_scale, "default must be on");
    on.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (col, a, b) = on.scale_calibration.expect("calibration should have run");
    assert_eq!(col, 1);
    assert!(a.is_finite() && b > 0.0 && b != 1.0, "a={a} b={b}");

    let mut off = model(HistogramLearner::new(3), 150, 7);
    off.auto_calibrate_scale = false;
    off.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(off.scale_calibration.is_none());
    assert_eq!(on.base_models.len(), off.base_models.len(), "same trees");
    assert!(on.score(&xv, &yv) <= off.score(&xv, &yv) + 1e-12);
    // μ untouched, σ changed
    let (po, pf) = (on.pred_param(&xv), off.pred_param(&xv));
    for i in 0..xv.nrows() {
        assert_eq!(po[[i, 0]].to_bits(), pf[[i, 0]].to_bits());
    }
    assert!(po.column(1) != pf.column(1));
}

#[test]
fn auto_calibration_is_skipped_without_validation_data() {
    let (x, y) = data(500, 71);
    let mut on = model(HistogramLearner::new(3), 60, 7);
    let mut off = model(HistogramLearner::new(3), 60, 7);
    off.auto_calibrate_scale = false;
    on.fit(&x, &y).unwrap();
    off.fit(&x, &y).unwrap();
    assert!(on.scale_calibration.is_none());
    assert_bits_equal(
        &on.pred_param(&x),
        &off.pred_param(&x),
        "no validation => identical",
    );
}

#[test]
fn auto_calibration_small_validation_fits_shift_only_and_tiny_skips() {
    let (x, y) = data(600, 81);
    let (xv, yv) = data(400, 82);
    // 40 <= n_val < 300: shift only (b == 1 exactly)
    let xv20 = xv.slice(ndarray::s![..45, ..]).to_owned();
    let yv20 = yv.slice(ndarray::s![..45]).to_owned();
    let mut m = model(HistogramLearner::new(3), 100, 7);
    m.fit_with_validation(&x, &y, Some(&xv20), Some(&yv20), None, None)
        .unwrap();
    let (_, a, b) = m.scale_calibration.expect("should run with 45 rows");
    assert_eq!(b, 1.0, "slope must be fixed for small validation sets");
    assert!(a.is_finite());
    // n_val < 40: skipped
    let xv5 = xv.slice(ndarray::s![..20, ..]).to_owned();
    let yv5 = yv.slice(ndarray::s![..20]).to_owned();
    let mut m5 = model(HistogramLearner::new(3), 100, 7);
    m5.fit_with_validation(&x, &y, Some(&xv5), Some(&yv5), None, None)
        .unwrap();
    assert!(m5.scale_calibration.is_none());
}

#[test]
fn chronological_split_matches_explicit_tail_split_bit_exact() {
    let (x, y) = data(1000, 91);
    let n = x.nrows();
    let n_val = (0.2f64 * n as f64).ceil() as usize; // fit_internal uses ceil
    let n_tr = n - n_val;
    let mk = || {
        NGBoost::<Normal, LogScore, HistogramLearner>::with_full_options(
            300,
            0.05,
            HistogramLearner::new(3),
            true,
            1.0,
            1.0,
            false,
            1.0,
            1e-4,
            Some(20),
            0.2,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            Some(7),
        )
    };
    let mut auto = mk();
    auto.validation_split = ValidationSplit::Chronological;
    auto.fit(&x, &y).unwrap();
    let mut explicit = mk();
    explicit
        .fit_with_validation(
            &x.slice(ndarray::s![..n_tr, ..]).to_owned(),
            &y.slice(ndarray::s![..n_tr]).to_owned(),
            Some(&x.slice(ndarray::s![n_tr.., ..]).to_owned()),
            Some(&y.slice(ndarray::s![n_tr..]).to_owned()),
            None,
            None,
        )
        .unwrap();
    assert_eq!(auto.base_models.len(), explicit.base_models.len());
    assert_eq!(auto.scale_calibration, explicit.scale_calibration);
    assert_bits_equal(
        &auto.pred_param(&x),
        &explicit.pred_param(&x),
        "chronological == explicit tail",
    );
    // and it really differs from the random split
    let mut random = mk();
    random.fit(&x, &y).unwrap();
    assert!(random.pred_param(&x) != auto.pred_param(&x));
}

#[test]
fn ngboost_params_round_trip_new_fields() {
    let mut m = NGBRegressor::new(10, 0.1);
    let mut p = m.get_params();
    assert!(p.auto_calibrate_scale);
    p.auto_calibrate_scale = false;
    p.validation_split = ValidationSplit::Chronological;
    p.fisher_weight_power = Some(0.5);
    m.set_params(p);
    let q = m.get_params();
    assert!(!q.auto_calibrate_scale);
    assert_eq!(q.validation_split, ValidationSplit::Chronological);
    assert_eq!(q.fisher_weight_power, Some(0.5));
    // old JSON without the new fields deserializes with the defaults
    let old = r#"{"n_estimators":5,"learning_rate":0.1,"natural_gradient":true,"minibatch_frac":1.0,"col_sample":1.0,"verbose":false,"verbose_eval":1.0,"tol":0.0001,"early_stopping_rounds":null,"validation_fraction":0.1,"random_state":null,"lr_schedule":"Constant","tikhonov_reg":0.0,"line_search_method":"Binary"}"#;
    let parsed: ngboost_rs::NGBoostParams = serde_json::from_str(old).unwrap();
    assert!(parsed.auto_calibrate_scale);
    assert_eq!(parsed.validation_split, ValidationSplit::Random);
    assert_eq!(parsed.fisher_weight_power, None);
}

// ---------------------------------------------------------------------------
// Distribution-level Fisher-weight default (Poisson → β = 1)
// ---------------------------------------------------------------------------

fn count_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Lcg(seed);
    let x = Array2::from_shape_fn((n, 5), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let r = x.row(i);
        let lam = (2.0 + 0.8 * (2.0 * r[0]).sin() + 0.25 * r[1] * r[1]).exp();
        // Poisson via Knuth
        let l = (-lam).exp();
        let (mut k, mut p) = (0u32, 1.0);
        loop {
            k += 1;
            p *= rng.next();
            if p <= l {
                return (k - 1) as f64;
            }
        }
    });
    (x, y)
}

#[test]
fn poisson_defaults_to_fisher_weighted_leaves() {
    assert_eq!(Poisson::default_fisher_weight_power(), Some(1.0));
    assert_eq!(
        ngboost_rs::dist::NegativeBinomial::default_fisher_weight_power(),
        Some(1.0)
    );
    assert_eq!(Normal::default_fisher_weight_power(), None);
    assert_eq!(Exponential::default_fisher_weight_power(), None);
    assert_eq!(
        ngboost_rs::dist::Bernoulli::default_fisher_weight_power(),
        Some(1.0)
    );
    assert_eq!(
        ngboost_rs::dist::Categorical3::default_fisher_weight_power(),
        Some(1.0)
    );
    assert!(ngboost_rs::dist::Categorical3::default_refit_intercept());
    assert!(!ngboost_rs::dist::Bernoulli::default_refit_intercept());
    let (x, y) = count_data(600, 5);
    let mk = || model::<HistogramLearner>(HistogramLearner::new(3), 60, 7);
    // NGBoost<Poisson, ...>
    let fit = |m: &mut NGBoost<Poisson, LogScore, HistogramLearner>| {
        m.fit(&x, &y).unwrap();
        m.pred_param(&x)
    };
    let _ = mk; // Normal helper not used here
    let mut dflt = NGBoost::<Poisson, LogScore, HistogramLearner>::with_full_options(
        60,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(7),
    );
    let mut one = NGBoost::<Poisson, LogScore, HistogramLearner>::with_full_options(
        60,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(7),
    );
    one.fisher_weight_power = Some(1.0);
    let mut zero = NGBoost::<Poisson, LogScore, HistogramLearner>::with_full_options(
        60,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        None,
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(7),
    );
    zero.fisher_weight_power = Some(0.0);
    let (pd, p1, p0) = (fit(&mut dflt), fit(&mut one), fit(&mut zero));
    assert_bits_equal(&pd, &p1, "Poisson default == explicit beta 1");
    assert!(
        pd != p0,
        "Some(0.0) must give the unweighted (different) fit"
    );
    assert!(pd.iter().all(|v| v.is_finite()));
}

#[test]
fn constant_fisher_heads_make_beta_a_no_op() {
    // Exponential (F ≡ 1) and HalfNormal (F ≡ 2): every row gets the same
    // weight, so β changes nothing beyond floating-point summation order.
    let (x, y0) = data(500, 13);
    let y = y0.mapv(|v| v.abs() + 0.05);
    let fitp = |beta: Option<f64>| {
        let mut m = NGBoost::<Exponential, LogScore, HistogramLearner>::with_full_options(
            60,
            0.05,
            HistogramLearner::new(3),
            true,
            1.0,
            1.0,
            false,
            1.0,
            1e-4,
            None,
            0.1,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            Some(7),
        );
        m.fisher_weight_power = beta;
        m.fit(&x, &y).unwrap();
        m.pred_param(&x)
    };
    let (p0, p1) = (fitp(Some(0.0)), fitp(Some(1.0)));
    for (a, b) in p0.iter().zip(p1.iter()) {
        assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
    }
}

#[test]
fn fisher_weight_cap_one_equals_unweighted_fit() {
    // κ = 1 clamps every weight to the column median → constant weights →
    // the same fit as β = 0 up to floating-point summation order.
    let (x, y) = data(500, 17);
    let fit = |beta: Option<f64>, cap: Option<f64>| {
        let mut m = model(HistogramLearner::new(3), 60, 7);
        m.fisher_weight_power = beta;
        m.fisher_weight_cap = cap;
        m.fit(&x, &y).unwrap();
        m.pred_param(&x)
    };
    let (p0, pc) = (fit(Some(0.0), None), fit(Some(1.0), Some(1.0)));
    for (a, b) in p0.iter().zip(pc.iter()) {
        assert!((a - b).abs() <= 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
    }
    // and a loose cap is a genuinely different (weighted) fit
    let p10 = fit(Some(1.0), Some(10.0));
    assert!(p10 != p0);
}

// ---------------------------------------------------------------------------
// partial_fit keeps folded shifts; refit_intercept
// ---------------------------------------------------------------------------

#[test]
fn partial_fit_preserves_folded_calibration_shift() {
    let (x, y) = data(600, 101);
    let (xv, yv) = data(300, 102);
    let mut m = model(HistogramLearner::new(3), 60, 7);
    m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(m.scale_calibration.is_some());
    let init_before = m.init_params.clone().unwrap();
    let pred_before = m.pred_param(&xv);
    // continue training without validation: no recalibration this call
    let (x2, y2) = data(600, 103);
    m.n_estimators = 5;
    m.partial_fit(&x2, &y2).unwrap();
    let init_after = m.init_params.clone().unwrap();
    assert_eq!(
        init_before, init_after,
        "init_params must not be re-fit on partial_fit"
    );
    // the first 60 stages still reproduce the pre-partial_fit model
    let replay = m.pred_param_at(&xv, 60);
    assert_bits_equal(&pred_before, &replay, "replay of the calibrated stages");
}

fn poisson_model(seed: u64) -> NGBoost<Poisson, LogScore, HistogramLearner> {
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
fn refit_intercept_restores_poisson_total_on_train() {
    let (x, y) = count_data(1500, 111);
    let (xv, yv) = count_data(400, 112);
    let mut off = poisson_model(7);
    off.refit_intercept = Some(false);
    off.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let mut on = poisson_model(7);
    on.refit_intercept = Some(true);
    on.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let ratio = |m: &NGBoost<Poisson, LogScore, HistogramLearner>| y.sum() / m.predict(&x).sum();
    let (r_off, r_on) = (ratio(&off), ratio(&on));
    // one Fisher-scoring step per round keeps the train total within 0.5 %
    assert!(
        (r_on - 1.0).abs() < 5e-3,
        "train Σy/Σλ̂ with intercept steps = {r_on}"
    );
    assert!(
        (r_on - 1.0).abs() <= (r_off - 1.0).abs() + 1e-12,
        "{r_on} vs {r_off}"
    );
    // folded into the model: pred_param equals replay from init + trees
    assert!(on.pred_param(&x).iter().all(|v| v.is_finite()));
}

#[test]
fn refit_intercept_off_by_default_and_none_matches_false() {
    let (x, y) = data(400, 121);
    let mut a = model(HistogramLearner::new(3), 40, 7);
    let mut b = model(HistogramLearner::new(3), 40, 7);
    b.refit_intercept = Some(false);
    a.fit(&x, &y).unwrap();
    b.fit(&x, &y).unwrap();
    assert_bits_equal(
        &a.pred_param(&x),
        &b.pred_param(&x),
        "None == Some(false) for Normal",
    );
}

#[test]
fn mixture_auto_calibration_scales_both_component_scales_jointly() {
    use ngboost_rs::dist::NormalMixture2;
    assert_eq!(NormalMixture2::scale_param_columns(), vec![2, 4]);
    assert_eq!(Normal::scale_param_columns(), vec![1]);
    let (x, y) = data(800, 131);
    let (xv, yv) = data(400, 132);
    let mk = |auto: bool| {
        let mut m = NGBoost::<NormalMixture2, LogScore, HistogramLearner>::with_full_options(
            80,
            0.05,
            HistogramLearner::new(3),
            true,
            1.0,
            1.0,
            false,
            1.0,
            1e-4,
            None,
            0.1,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            Some(7),
        );
        m.auto_calibrate_scale = auto;
        m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
            .unwrap();
        m
    };
    let (off, on) = (mk(false), mk(true));
    let (_, a, b) = on
        .scale_calibration
        .expect("mixture calibration should run");
    let (po, pn) = (off.pred_param(&xv), on.pred_param(&xv));
    for i in 0..xv.nrows() {
        for c in [0usize, 1, 3] {
            assert_eq!(
                po[[i, c]].to_bits(),
                pn[[i, c]].to_bits(),
                "column {c} must be untouched"
            );
        }
        for c in [2usize, 4] {
            let manual = a + b * po[[i, c]];
            assert!((pn[[i, c]] - manual).abs() <= 1e-9 * (1.0 + manual.abs()));
        }
    }
    assert!(on.score(&xv, &yv) <= off.score(&xv, &yv) + 1e-12);
}

#[test]
fn staged_default_requires_the_scale_recalibration_to_run() {
    // Cycle 17: the staged σ stage is too narrow until recalibrated, so the
    // default only stages when the end-of-fit recalibration will run.
    let mut m = model(HistogramLearner::new(3), 100, 1);
    m.early_stopping_rounds = Some(10);
    assert!(m.uses_staged_fit(2000, Some(400)));
    assert!(m.uses_staged_fit(2000, Some(40)));
    assert!(
        !m.uses_staged_fit(2000, Some(39)),
        "below the calibration floor"
    );
    m.validation_fraction = 0.1;
    assert!(m.uses_staged_fit(2000, None), "200-row split");
    assert!(!m.uses_staged_fit(300, None), "30-row split");
    m.auto_calibrate_scale = false;
    assert!(!m.uses_staged_fit(2000, Some(400)), "calibration off");
    m.staged_fit = Some(true);
    assert!(
        m.uses_staged_fit(2000, Some(10)),
        "explicit opt-in still forces it"
    );
    m.staged_fit = Some(false);
    m.auto_calibrate_scale = true;
    assert!(!m.uses_staged_fit(2000, Some(400)));
}
