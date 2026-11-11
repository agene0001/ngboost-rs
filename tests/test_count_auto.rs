//! Cycle 17: the Poisson → NegativeBinomial switch (`CountModel::fit_auto`),
//! `NGBoost::unfitted_like` and `NGBoost::refit_full`.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2, s};
use ngboost_rs::dist::{NegativeBinomial, Normal, Poisson};
use ngboost_rs::learners::HistogramLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;
use ngboost_rs::{CountModel, LearningRateSchedule, LineSearchMethod, dispersion_z};

/// Uniform [0, 1) generator (53-bit).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
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

/// Counts with a simple log-linear rate (≈ 3–8); Gamma–Poisson with `r`.
fn counts(n: usize, seed: u64, r: Option<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Rng(seed);
    let x = Array2::from_shape_fn((n, 3), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        let mu = (1.6 + 0.25 * x[[i, 0]]).exp();
        let lam = match r {
            Some(rr) => mu * rng.gamma(rr) / rr,
            None => mu,
        };
        rng.poisson(lam)
    });
    (x, y)
}

fn template() -> NGBoost<Poisson, LogScore, HistogramLearner> {
    NGBoost::with_full_options(
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
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(3),
    )
}

#[test]
fn fit_auto_switches_to_negbin_only_when_overdispersed() {
    let (x, y) = counts(2000, 1, Some(3.0));
    let (xv, yv) = counts(400, 2, Some(3.0));
    let (xt, yt) = counts(1500, 3, Some(3.0));
    let (m, z) = CountModel::fit_auto(&template(), &x, &y, &xv, &yv).unwrap();
    assert!(z > 5.0, "over-dispersed data: z = {z}");
    assert!(m.is_negative_binomial());
    let mut pois = template();
    pois.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(m.score(&xt, &yt) < pois.score(&xt, &yt) - 0.05);
    let (lo, hi) = m.interval(&xt, 0.1);
    let cov = (0..yt.len())
        .filter(|&i| yt[i] >= lo[i] && yt[i] <= hi[i])
        .count() as f64
        / yt.len() as f64;
    assert!(cov > 0.85, "NB 90 % interval coverage {cov}");

    let (x, y) = counts(2000, 11, None);
    let (xv, yv) = counts(400, 12, None);
    let (m, z) = CountModel::fit_auto(&template(), &x, &y, &xv, &yv).unwrap();
    assert!(z < 1.645, "true Poisson data: z = {z}");
    assert!(!m.is_negative_binomial());
    // the statistic is the library's
    if let CountModel::Poisson(p) = &m {
        assert_eq!(z, dispersion_z(&yv, &p.predict(&xv)));
    }
}

#[test]
fn unfitted_like_copies_settings_across_heads() {
    let mut t = template();
    t.fisher_weight_power = Some(0.5);
    t.auto_calibrate_scale = false;
    t.staged_fit = Some(true);
    t.tikhonov_reg = 0.01;
    let nb = t.unfitted_like::<NegativeBinomial>();
    assert_eq!(nb.n_estimators, 300);
    assert_eq!(nb.learning_rate, 0.05);
    assert_eq!(nb.early_stopping_rounds, Some(20));
    assert_eq!(nb.fisher_weight_power, Some(0.5));
    assert!(!nb.auto_calibrate_scale);
    assert_eq!(nb.staged_fit, Some(true));
    assert_eq!(nb.tikhonov_reg, 0.01);
    assert!(nb.base_models.is_empty());
}

#[test]
fn refit_full_replays_the_kept_rounds_on_new_rows() {
    let (x, y) = counts(1500, 21, None);
    let (xv, yv) = counts(300, 22, None);
    let mut m = template();
    assert!(m.refit_full(&x, &y, None, None).is_err(), "unfitted");
    m.fit_with_validation(&x, &y, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let k = m.base_models.len();
    let all_x = ndarray::concatenate![ndarray::Axis(0), x, xv];
    let all_y = ndarray::concatenate![ndarray::Axis(0), y, yv];
    let r = m.refit_full(&all_x, &all_y, None, None).unwrap();
    assert_eq!(r.base_models.len(), k);
    assert_eq!(r.n_estimators, m.n_estimators, "settings restored");
    assert_eq!(r.early_stopping_rounds, m.early_stopping_rounds);
    // identical to a plain fit for k rounds without early stopping
    let mut plain = template();
    plain.n_estimators = k as u32;
    plain.early_stopping_rounds = None;
    plain.fit(&all_x, &all_y).unwrap();
    let (a, b) = (r.pred_param(&all_x), plain.pred_param(&all_x));
    for (u, v) in a.iter().zip(b.iter()) {
        assert_eq!(u.to_bits(), v.to_bits());
    }
    let r2 = m.refit_full(&all_x, &all_y, None, Some(7)).unwrap();
    assert_eq!(r2.base_models.len(), 7);
}

#[test]
fn refit_full_carries_the_scale_recalibration_and_refuses_staged_fits() {
    let mut rng = Rng(5);
    let n = 1200;
    let x = Array2::from_shape_fn((n, 3), |_| rng.next() * 4.0 - 2.0);
    let y = Array1::from_shape_fn(n, |i| {
        x[[i, 0]].sin() + (0.3 + 0.2 * x[[i, 1]].abs()) * rng.gauss()
    });
    let (xt, yt) = (
        x.slice(s![..900, ..]).to_owned(),
        y.slice(s![..900]).to_owned(),
    );
    let (xv, yv) = (
        x.slice(s![900.., ..]).to_owned(),
        y.slice(s![900..]).to_owned(),
    );
    let mut staged: NGBoost<Normal, LogScore, HistogramLearner> = NGBoost::with_full_options(
        200,
        0.05,
        HistogramLearner::new(3),
        true,
        1.0,
        1.0,
        false,
        1.0,
        1e-4,
        Some(20),
        0.1,
        LearningRateSchedule::Constant,
        0.0,
        LineSearchMethod::Binary,
        Some(1),
    );
    let mut joint = staged.unfitted_like::<Normal>();
    joint.staged_fit = Some(false);
    staged
        .fit_with_validation(&xt, &yt, Some(&xv), Some(&yv), None, None)
        .unwrap();
    assert!(staged.refit_full(&x, &y, None, None).is_err());
    joint
        .fit_with_validation(&xt, &yt, Some(&xv), Some(&yv), None, None)
        .unwrap();
    let (col, a, b) = joint.scale_calibration.expect("calibrated");
    let r = joint.refit_full(&x, &y, None, None).unwrap();
    assert_eq!(r.scale_calibration, Some((col, a, b)));
    // the refit's log σ is the uncalibrated refit mapped by (a, b)
    let mut raw = joint.unfitted_like::<Normal>();
    raw.n_estimators = joint.base_models.len() as u32;
    raw.early_stopping_rounds = None;
    raw.auto_calibrate_scale = false;
    raw.fit(&x, &y).unwrap();
    let (pr, pc) = (raw.pred_param(&x), r.pred_param(&x));
    for i in 0..x.nrows() {
        assert_eq!(pr[[i, 0]].to_bits(), pc[[i, 0]].to_bits(), "μ untouched");
        assert!((a + b * pr[[i, 1]] - pc[[i, 1]]).abs() < 1e-9);
    }
}
