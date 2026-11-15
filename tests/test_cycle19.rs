//! research Cycle 19 (1.3.0): adaptive leaf size of the default tree
//! learners, the exact tree learners' leaf default, the LogNormal scale
//! ceiling.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::dist::{Distribution, LogNormal, Normal};
use ngboost_rs::hyper_opt::default_histogram_model;
use ngboost_rs::learners::{
    ArenaDecisionTreeLearner, BaseLearner, DecisionTreeLearner, HistogramLearner,
    TrainedBaseLearner, adaptive_min_samples_leaf, arena_tree_learner, default_tree_learner,
};
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::{LogScore, Scorable};
use ngboost_rs::{LearningRateSchedule, LineSearchMethod};
use serde_json::json;
use std::collections::HashMap;

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

/// Strong nonlinear signal, mild noise.
fn signal(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 4), |_| r.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        3.0 * v[0] + 2.0 * (2.0 * v[1]).sin() + v[2] * v[3] + 0.2 * r.gauss()
    });
    (x, y)
}

/// Rows per leaf of a fitted tree, seen through its predictions on the
/// rows it was fitted on (distinct leaf values ⇒ distinct leaves here).
fn leaf_sizes(tree: &dyn TrainedBaseLearner, x: &Array2<f64>) -> Vec<usize> {
    let mut bits: Vec<u64> = tree.predict(x).iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    let mut sizes = Vec::new();
    let mut i = 0;
    while i < bits.len() {
        let j = bits[i..].iter().take_while(|&&b| b == bits[i]).count();
        sizes.push(j);
        i += j;
    }
    sizes
}

type Tree = Box<dyn TrainedBaseLearner>;

#[test]
fn adaptive_leaf_is_a_tenth_of_the_rows_capped() {
    for (n, want) in [
        (0, 1),
        (9, 1),
        (19, 1),
        (20, 2),
        (70, 7),
        (105, 10),
        (199, 19),
        (200, 20),
        (210, 20),
        (100_000, 20),
    ] {
        assert_eq!(adaptive_min_samples_leaf(20, n), want, "n = {n}");
    }
    // the cap is the learner's own leaf size
    assert_eq!(adaptive_min_samples_leaf(5, 1000), 5);
    assert_eq!(adaptive_min_samples_leaf(50, 300), 30);

    let h = HistogramLearner::new(3);
    assert!(h.adaptive_leaf);
    assert_eq!(h.effective_min_samples_leaf(80), 8);
    assert_eq!(h.effective_min_samples_leaf(5000), 20);
    // an explicit leaf size is used as given
    let e = HistogramLearner::with_params(3, 255, 20, 2);
    assert!(!e.adaptive_leaf);
    assert_eq!(e.effective_min_samples_leaf(80), 20);
}

#[test]
fn default_tree_learners_share_the_leaf_default() {
    // constructors
    let d = DecisionTreeLearner::new(3);
    let a = ArenaDecisionTreeLearner::new(3);
    assert_eq!(
        (d.min_samples_leaf, d.adaptive_leaf),
        (HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF, true)
    );
    assert_eq!(
        (a.min_samples_leaf, a.adaptive_leaf),
        (HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF, true)
    );
    // the scikit-learn-compatible entry points keep leaf 1
    for l in [
        DecisionTreeLearner::default_sklearn(),
        default_tree_learner(),
    ] {
        assert_eq!(
            (l.min_samples_leaf, l.adaptive_leaf, l.max_depth),
            (1, false, 3)
        );
    }
    for l in [
        ArenaDecisionTreeLearner::default_sklearn(),
        arena_tree_learner(),
    ] {
        assert_eq!(
            (l.min_samples_leaf, l.adaptive_leaf, l.max_depth),
            (1, false, 3)
        );
    }
    let w = DecisionTreeLearner::with_params(3, 2, 7);
    assert_eq!((w.min_samples_leaf, w.adaptive_leaf), (7, false));

    // 400 rows: the default is exactly leaf 20
    let (x, y) = signal(400, 3);
    let pairs: [(Tree, Tree, &str); 3] = [
        (
            HistogramLearner::new(6).fit(&x, &y).unwrap(),
            HistogramLearner::with_params(6, 255, 20, 2)
                .fit(&x, &y)
                .unwrap(),
            "histogram",
        ),
        (
            DecisionTreeLearner::new(6).fit(&x, &y).unwrap(),
            DecisionTreeLearner::with_params(6, 2, 20)
                .fit(&x, &y)
                .unwrap(),
            "exact",
        ),
        (
            ArenaDecisionTreeLearner::new(6).fit(&x, &y).unwrap(),
            ArenaDecisionTreeLearner::with_params(6, 2, 20)
                .fit(&x, &y)
                .unwrap(),
            "arena",
        ),
    ];
    for (default, explicit, name) in &pairs {
        let (p, q) = (default.predict(&x), explicit.predict(&x));
        assert!(
            p.iter()
                .zip(q.iter())
                .all(|(u, v)| u.to_bits() == v.to_bits()),
            "{name}: default differs from leaf 20 on 400 rows"
        );
        let sizes = leaf_sizes(default.as_ref(), &x);
        assert!(*sizes.iter().min().unwrap() >= 20, "{name}: {sizes:?}");
    }

    // 80 rows: leaves of at least 8 rows, and finer than leaf 20 allows
    let (x, y) = signal(80, 4);
    let small: [(Tree, &str); 3] = [
        (HistogramLearner::new(6).fit(&x, &y).unwrap(), "histogram"),
        (DecisionTreeLearner::new(6).fit(&x, &y).unwrap(), "exact"),
        (
            ArenaDecisionTreeLearner::new(6).fit(&x, &y).unwrap(),
            "arena",
        ),
    ];
    for (tree, name) in &small {
        let sizes = leaf_sizes(tree.as_ref(), &x);
        assert!(*sizes.iter().min().unwrap() >= 8, "{name}: {sizes:?}");
        assert!(sizes.len() > 80 / 20, "{name}: only {} leaves", sizes.len());
    }
    // explicit 20 on the same rows stays at 20
    let t = HistogramLearner::with_params(6, 255, 20, 2)
        .fit(&x, &y)
        .unwrap();
    assert!(*leaf_sizes(t.as_ref(), &x).iter().min().unwrap() >= 20);
    // scikit-learn defaults: single-row leaves are allowed
    let t = DecisionTreeLearner::with_params(6, 2, 1)
        .fit(&x, &y)
        .unwrap();
    assert!(*leaf_sizes(t.as_ref(), &x).iter().min().unwrap() < 8);
}

#[test]
fn default_histogram_model_adapts_only_without_a_leaf_key() {
    let (x, y) = signal(90, 5);
    let first_tree_leaves = |params: &HashMap<String, serde_json::Value>| {
        let mut m = default_histogram_model::<Normal, LogScore>(params);
        m.set_random_state(1);
        m.fit(&x, &y).unwrap();
        leaf_sizes(m.base_models[0][0].as_ref(), &x).len()
    };
    // 81 fitted rows (10 % held out): leaf 8, so more than 81 / 20 leaves
    assert!(first_tree_leaves(&HashMap::new()) > 4);
    // a sampled leaf size is used as given
    let mut p = HashMap::new();
    p.insert("min_samples_leaf".to_string(), json!(20));
    assert!(first_tree_leaves(&p) <= 4);
}

/// Log-normal target whose log-scale is 0.05 except on 3 % of the rows
/// (x0 > 0.94), where it is 3: the rms log-residual scale is ≈ 0.5, so the
/// honest σ of that region is about six times the baseline.
fn spiky_lognormal(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(seed);
    let x = Array2::from_shape_fn((n, 3), |_| r.next());
    let y = Array1::from_shape_fn(n, |i| {
        let v = x.row(i);
        let sd = if v[0] > 0.94 { 3.0 } else { 0.05 };
        (0.5 * v[1] + sd * r.gauss()).exp()
    });
    (x, y)
}

fn regressor<D>(seed: u64) -> NGBoost<D, LogScore, HistogramLearner>
where
    D: Distribution + Scorable<LogScore> + Clone,
{
    NGBoost::with_full_options(
        600,
        0.05,
        HistogramLearner::new(3),
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

/// `LogNormal` with every hook and score delegated but no scale ceiling:
/// what the head was before the ceiling existed.
#[derive(Debug, Clone)]
struct LogNormalNoCeiling(LogNormal);

impl Distribution for LogNormalNoCeiling {
    fn from_params(params: &Array2<f64>) -> Self {
        Self(LogNormal::from_params(params))
    }
    fn fit(y: &Array1<f64>) -> Array1<f64> {
        LogNormal::fit(y)
    }
    fn validate_targets(y: &Array1<f64>) -> Result<(), &'static str> {
        LogNormal::validate_targets(y)
    }
    fn n_params(&self) -> usize {
        self.0.n_params()
    }
    fn scale_param_index() -> Option<usize> {
        LogNormal::scale_param_index()
    }
    fn auto_calibration_min_rows() -> usize {
        LogNormal::auto_calibration_min_rows()
    }
    fn default_fisher_weight_power() -> Option<f64> {
        LogNormal::default_fisher_weight_power()
    }
    fn location_column() -> Option<usize> {
        LogNormal::location_column()
    }
    fn default_staged_fit() -> bool {
        LogNormal::default_staged_fit()
    }
    fn default_refit_intercept() -> bool {
        LogNormal::default_refit_intercept()
    }
    fn predict(&self) -> Array1<f64> {
        self.0.predict()
    }
    fn params(&self) -> Array2<f64> {
        self.0.params()
    }
}

impl Scorable<LogScore> for LogNormalNoCeiling {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Scorable::<LogScore>::score(&self.0, y)
    }
    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        Scorable::<LogScore>::d_score(&self.0, y)
    }
    fn metric(&self) -> ndarray::Array3<f64> {
        Scorable::<LogScore>::metric(&self.0)
    }
    fn is_diagonal_metric(&self) -> bool {
        Scorable::<LogScore>::is_diagonal_metric(&self.0)
    }
    fn diagonal_metric(&self) -> Array2<f64> {
        Scorable::<LogScore>::diagonal_metric(&self.0)
    }
}

#[test]
fn lognormal_scale_ceiling_bounds_every_prediction() {
    assert_eq!(LogNormal::scale_ceiling(), Some(4.0));
    assert_eq!(Normal::scale_ceiling(), None);

    let (x, y) = spiky_lognormal(3000, 21);
    let (xt, yt) = spiky_lognormal(1500, 22);
    for folds in [0usize, 5] {
        let mut m = regressor::<LogNormal>(7);
        m.staged_crossfit_folds = folds;
        m.fit(&x, &y).unwrap();
        let ceil = m.init_params.as_ref().unwrap()[1] + 4.0_f64.ln();

        // the model's own sum goes above the bound on this data
        let raw = m.pred_param_unbounded(&xt);
        let raw_max = raw.column(1).iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            raw_max > ceil + 0.2,
            "folds {folds}: the test needs an unbounded sum above the ceiling ({raw_max} vs {ceil})"
        );

        // The ceiling does not reach into the fit: the same head without a
        // ceiling trains the same model, bit for bit.
        let mut twin = regressor::<LogNormalNoCeiling>(7);
        twin.staged_crossfit_folds = folds;
        twin.fit(&x, &y).unwrap();
        assert_eq!(twin.base_models.len(), m.base_models.len(), "folds {folds}");
        let pt = twin.pred_param(&xt);
        assert!(
            raw.iter()
                .zip(pt.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "folds {folds}: the ceiling changed the fitted model"
        );

        // every public prediction is clamped, and the clamp binds
        let p = m.pred_param(&xt);
        let hit = p.column(1).iter().filter(|&&v| v == ceil).count();
        assert!(hit > 0, "folds {folds}: ceiling never reached");
        assert!(p.column(1).iter().all(|&v| v <= ceil));
        let bound = ceil.exp() * (1.0 + 1e-12);
        assert!(m.pred_dist(&xt).scale.iter().all(|&s| s <= bound));
        assert!(m.pred_dist_best(&xt).scale.iter().all(|&s| s <= bound));
        let k = m.base_models.len();
        assert!(m.pred_dist_at(&xt, k / 2).scale.iter().all(|&s| s <= bound));
        let staged = m.staged_pred_dist(&xt);
        assert!(staged.iter().all(|d| d.scale.iter().all(|&s| s <= bound)));
        assert_eq!(
            staged.last().unwrap().scale.to_vec(),
            m.pred_dist(&xt).scale.to_vec()
        );
        // rows below the ceiling are the model's own sum: the location is
        // never touched
        assert!(p.column(0).iter().all(|v| v.is_finite()));
        // the predictive mean stays within exp(μ + ceiling²/2)
        let mean = m.predict(&xt);
        for i in 0..xt.nrows() {
            assert!(mean[i] <= (p[[i, 0]] + 0.5 * bound * bound).exp() * (1.0 + 1e-9));
        }

        // nothing new is stored: a saved model predicts the same
        let back = m.serialize().unwrap();
        let m2 = NGBoost::<LogNormal, LogScore, HistogramLearner>::deserialize(
            back,
            HistogramLearner::new(3),
        )
        .unwrap();
        let p2 = m2.pred_param(&xt);
        assert!(
            p.iter()
                .zip(p2.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        // rows below the ceiling are untouched, bit for bit
        for i in 0..xt.nrows() {
            assert_eq!(p[[i, 0]].to_bits(), raw[[i, 0]].to_bits());
            if raw[[i, 1]] <= ceil {
                assert_eq!(p[[i, 1]].to_bits(), raw[[i, 1]].to_bits());
            }
        }
        let nll = m.score(&xt, &yt);
        let nll_raw = Scorable::<LogScore>::total_score(&LogNormal::from_params(&raw), &yt, None);
        assert!(nll.is_finite() && nll_raw.is_finite());
        eprintln!(
            "folds {folds}: bounded {nll:.4} unbounded {nll_raw:.4}, {hit} of {} rows at the ceiling, raw max {raw_max:.3} ceiling {ceil:.3}",
            xt.nrows()
        );
    }
}
