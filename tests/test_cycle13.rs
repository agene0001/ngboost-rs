//! research/ngboost Cycle 13: staged Normal fit, Normal log-σ step guard,
//! hurdle (point mass at zero) heads, CRPS-trained auto mixture.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2, array};
use ngboost_rs::dist::{
    Distribution, DistributionMethods, Hurdle, HurdleLogNormal, HurdleNormal, Normal,
    NormalMixture2,
};
use ngboost_rs::learners::{BaseLearner, DecisionTreeLearner, HistogramLearner};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{CRPScore, LogScore, Scorable};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod, NGBAutoMixtureRegressor, NGBRegressor};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // 53 random bits → uniform on [0, 1)
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next().max(1e-12);
        let u2 = self.next();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Smooth nonlinear mean, σ = 1 (the benchmark's `homo` shape).
fn homo(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 6), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        2.0 * v[0].sin() + v[1] * v[1] - v[2] * v[3] + r.gauss()
    });
    (x, y)
}

/// Zeros with x-dependent probability, else a log-normal positive part.
fn zero_inflated(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let p0 = 1.0 / (1.0 + (-(1.5 * v[0])).exp());
        if r.next() < p0 {
            0.0
        } else {
            (1.0 + 0.5 * v[1] + 0.4 * r.gauss()).exp()
        }
    });
    (x, y)
}

fn model<D, B>(learner: B, seed: u64) -> NGBoost<D, LogScore, B>
where
    D: Distribution + Scorable<LogScore> + Clone,
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
        Some(40),
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(seed),
    )
}

fn nll(m: &NGBoost<Normal, LogScore, HistogramLearner>, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
    Scorable::<LogScore>::total_score(&m.pred_dist(x), y, None)
}

#[test]
fn two_stage_beats_joint_fit_and_is_an_ordinary_normal_model() {
    let (x, y) = homo(3000, 1);
    let (xv, yv) = homo(600, 2);
    let (xt, yt) = homo(2000, 3);
    let mut joint = model::<Normal, _>(HistogramLearner::new(3), 5);
    joint.staged_fit = Some(false);
    joint
        .fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let mut staged = model::<Normal, _>(HistogramLearner::new(3), 5);
    staged
        .fit_two_stage(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let rmse = |m: &NGBoost<Normal, LogScore, HistogramLearner>| {
        (&m.predict(&xt) - &yt)
            .mapv(|e| e * e)
            .mean()
            .unwrap()
            .sqrt()
    };
    let (nj, ns) = (nll(&joint, &xt, &yt), nll(&staged, &xt, &yt));
    assert!(ns < nj - 0.02, "staged NLL {ns} vs joint {nj}");
    assert!(
        rmse(&staged) < rmse(&joint),
        "staged RMSE {} vs joint {}",
        rmse(&staged),
        rmse(&joint)
    );
    assert!(staged.scale_calibration.is_some_and(|(c, _, _)| c == 1));
    // every round is [μ tree, 0] or [0, σ tree]; the format round-trips
    assert!(staged.base_models.iter().all(|r| r.len() == 2));
    let ser = staged.serialize().unwrap();
    let bytes = bincode::serialize(&ser).unwrap();
    let back: ngboost_rs::ngboost::SerializedNGBoost = bincode::deserialize(&bytes).unwrap();
    let m2 =
        NGBoost::<Normal, LogScore, HistogramLearner>::deserialize(back, HistogramLearner::new(3))
            .unwrap();
    let (a, b) = (staged.pred_param(&xt), m2.pred_param(&xt));
    assert!(
        a.iter()
            .zip(b.iter())
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

#[test]
#[ignore]
fn two_stage_seed_sweep() {
    for (n, seed) in [
        (3000, 1),
        (3000, 31),
        (3000, 61),
        (3000, 91),
        (8000, 1),
        (8000, 31),
    ] {
        let (x, y) = homo(n, seed);
        let (xv, yv) = homo(n / 5, seed + 1);
        let (xt, yt) = homo(2000, seed + 2);
        let mut j = model::<Normal, _>(HistogramLearner::new(3), 5);
        j.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
            .unwrap();
        let mut s = model::<Normal, _>(HistogramLearner::new(3), 5);
        s.fit_two_stage(&x, &y, Some(&xv), Some(&yv), None, None)
            .unwrap();
        let r = |m: &NGBoost<Normal, LogScore, HistogramLearner>| {
            (&m.predict(&xt) - &yt)
                .mapv(|e| e * e)
                .mean()
                .unwrap()
                .sqrt()
        };
        println!(
            "n {n} seed {seed}: joint NLL {:.4} RMSE {:.4} ({} rounds) | staged NLL {:.4} RMSE {:.4} ({} rounds)",
            nll(&j, &xt, &yt),
            r(&j),
            j.base_models.len(),
            nll(&s, &xt, &yt),
            r(&s),
            s.base_models.len()
        );
    }
}

#[test]
fn two_stage_works_with_internal_validation_split_and_other_learners() {
    let (x, y) = homo(1500, 7);
    let mut m = model::<Normal, _>(DecisionTreeLearner::new(3), 3);
    m.fit_two_stage(&x, &y, None, None, None, None).unwrap();
    assert!(m.pred_param(&x).iter().all(|v| v.is_finite()));
    let mut r = NGBRegressor::new(300, 0.05);
    let (xv, yv) = homo(300, 8);
    r.fit_two_stage(&x, &y, Some(&xv), Some(&yv)).unwrap();
    assert!(r.predict(&xv).iter().all(|v| v.is_finite()));
    assert!(
        m.fit_two_stage(&x, &y, Some(&xv), None, None, None)
            .is_err()
    );
}

#[test]
fn normal_log_sigma_step_is_capped_for_outliers_only() {
    let d = Normal::from_params(&array![[0.0, 0.0], [0.0, 0.0]]);
    let y = array![1.5, 100.0];
    let ng = Scorable::<LogScore>::grad(&d, &y, true);
    // ordinary row: the Fisher step (1 − z²)/2
    assert!((ng[[0, 1]] - (1.0 - 1.5f64 * 1.5) / 2.0).abs() < 1e-12);
    // |z| = 100: (1 − 10⁴)/2 ≈ −5000 uncapped, capped at −25
    assert!((ng[[1, 1]] + 25.0).abs() < 1e-9, "{}", ng[[1, 1]]);
    // the μ step is untouched
    assert!((ng[[1, 0]] + 100.0).abs() < 1e-9);
}

#[test]
fn hurdle_heads_learn_the_atom_probability() {
    let (x, y) = zero_inflated(3000, 11);
    let (xv, yv) = zero_inflated(600, 12);
    let (xt, yt) = zero_inflated(2000, 13);
    let share = yt.iter().filter(|&&v| v == 0.0).count() as f64 / yt.len() as f64;
    let mut h = model::<HurdleLogNormal, _>(HistogramLearner::new(3), 2);
    h.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let d = h.pred_dist(&xt);
    assert!((d.pi.mean().unwrap() - share).abs() < 0.03);
    // P(Y = 0) tracks x: the Brier score beats the constant share
    let brier = |p: &Array1<f64>| {
        (0..yt.len())
            .map(|i| (p[i] - if yt[i] == 0.0 { 1.0 } else { 0.0 }).powi(2))
            .sum::<f64>()
            / yt.len() as f64
    };
    assert!(brier(&d.pi) < 0.85 * brier(&Array1::from_elem(yt.len(), share)));
    // the cdf jumps by π at 0 and the score is finite everywhere
    let c = d.cdf(&Array1::zeros(yt.len()));
    assert!((0..yt.len()).all(|i| (c[i] - d.pi[i]).abs() < 1e-12));
    assert!(
        Scorable::<LogScore>::score(&d, &yt)
            .iter()
            .all(|v| v.is_finite())
    );
    // a mixture continuous part trains too, and the HurdleNormal variant
    let mut hm = model::<Hurdle<NormalMixture2>, _>(HistogramLearner::new(3), 2);
    hm.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(hm.pred_param(&xt).iter().all(|v| v.is_finite()));
    let mut hn = model::<HurdleNormal, _>(HistogramLearner::new(3), 2);
    hn.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(hn.pred_param(&xt).iter().all(|v| v.is_finite()));
    // the log-normal part rejects negative non-zero targets
    let mut bad = y.clone();
    bad[5] = -1.0;
    let mut hb = model::<HurdleLogNormal, _>(HistogramLearner::new(3), 2);
    assert!(
        hb.fit_with_validation(&x, &bad, Some(&xv), Some(&yv), None, None)
            .is_err()
    );
}

#[test]
fn crps_trained_auto_mixture_selects_by_validation_crps() {
    let (x, y) = homo(1500, 21);
    let (xv, yv) = homo(400, 22);
    let mut m = NGBAutoMixtureRegressor::new_crps(400, 0.05);
    m.candidates = vec![1, 2];
    m.random_state = Some(1);
    m.fit(&x, &y, &xv, &yv).unwrap();
    let best = m.tried.iter().map(|t| t.2).fold(f64::INFINITY, f64::min);
    assert!(
        (m.crps(&xv, &yv) - best).abs() < 1e-9,
        "selected = min validation CRPS"
    );
    assert!(m.score(&xv, &yv).is_finite());
    let _ = Scorable::<CRPScore>::score(&Normal::from_params(&array![[0.0, 0.0]]), &array![0.3]);
}
