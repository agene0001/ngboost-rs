//! CRPS audit of every head with a CRPScore (research/ngboost Cycle 14):
//! score = ∫(F(t) − 1{t ≥ y})² dt, d_score = ∂score/∂θ, and the metric =
//! 2∫∇F(t)∇F(t)ᵀ dt (sums for discrete heads), all checked numerically
//! against the head's own cdf. `audit_report` (ignored) prints the table;
//! the asserting test covers the heads that pass.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use ndarray::{Array1, Array2};
use ngboost_rs::dist::{
    Distribution, DistributionMethods, Exponential, Gamma, HalfNormal, HurdleLogNormal,
    HurdleNormal, Laplace, NegativeBinomial, Normal, NormalFixedMean, NormalFixedVar,
    NormalMixture, Poisson, StudentT, TFixedDf, TFixedDfFixedVar, Weibull,
};
use ngboost_rs::scores::{CRPScore, Scorable};

#[derive(Debug)]
struct Audit {
    score: f64,
    grad: f64,
    metric: f64,
    /// implemented / reference, diagonal entries
    ratio: Vec<f64>,
}

/// Max relative errors (vs the scale of the reference) of score, d_score
/// and metric for one parameter row and a few y values.
fn audit<D>(p: &[f64], ys: &[f64], discrete: bool) -> Audit
where
    D: Distribution + DistributionMethods + Scorable<CRPScore>,
{
    let k = p.len();
    let row = Array2::from_shape_vec((1, k), p.to_vec()).unwrap();
    let d = D::from_params(&row);
    let q = |u: f64| d.ppf(&Array1::from_elem(1, u))[0];
    let (mut lo, mut hi) = (q(1e-10), q(1.0 - 1e-10));
    for &y in ys {
        lo = lo.min(y - 1.0);
        hi = hi.max(y + 1.0);
    }
    if discrete {
        lo = 0.0;
    }
    // integration grid (midpoints) or support points
    let (t, h): (Array1<f64>, f64) = if discrete {
        let kmax = hi.ceil() as usize + 5;
        ((0..=kmax).map(|v| v as f64).collect(), 1.0)
    } else {
        let h = (hi - lo) / 200_000.0;
        // 0 on a cell boundary: the hurdle heads' cdf jumps there, and a
        // midpoint cell straddling a jump costs O(h) accuracy
        let lo = if lo < 0.0 { -(-lo / h).ceil() * h } else { lo };
        let n = ((hi - lo) / h).ceil() as usize;
        ((0..n).map(|i| lo + (i as f64 + 0.5) * h).collect(), h)
    };
    let nt = t.len();
    let wide = |r: &Array2<f64>| Array2::from_shape_fn((nt, k), |(_, j)| r[[0, j]]);
    let f0 = D::from_params(&wide(&row)).cdf(&t);
    // ∇F by central differences in the raw parameters
    let eps = 1e-6;
    let grads: Vec<Array1<f64>> = (0..k)
        .map(|j| {
            let (mut a, mut b) = (row.clone(), row.clone());
            a[[0, j]] += eps;
            b[[0, j]] -= eps;
            let (da, db) = (D::from_params(&wide(&a)), D::from_params(&wide(&b)));
            (da.cdf(&t) - db.cdf(&t)) / (2.0 * eps)
        })
        .collect();
    let m = Scorable::<CRPScore>::metric(&d);
    let mut err = Audit {
        score: 0.0,
        grad: 0.0,
        metric: 0.0,
        ratio: Vec::new(),
    };
    for a in 0..k {
        for b in 0..k {
            let refv = 2.0 * (&grads[a] * &grads[b]).sum() * h;
            let scale =
                (2.0 * (&grads[a] * &grads[a]).sum() * h * 2.0 * (&grads[b] * &grads[b]).sum() * h)
                    .sqrt()
                    .max(1e-300);
            err.metric = err.metric.max((m[[0, a, b]] - refv).abs() / scale);
            if a == b {
                err.ratio.push(m[[0, a, a]] / refv);
            }
        }
    }
    for &y in ys {
        let yv = Array1::from_elem(1, y);
        let s = Scorable::<CRPScore>::score(&d, &yv)[0];
        // y on a cell boundary too (the indicator jumps there)
        let (ty, fy) = if discrete {
            (t.clone(), f0.clone())
        } else {
            let lo = t[0] - 0.5 * h;
            let shift = (y - lo) - h * ((y - lo) / h).round();
            let ty = t.mapv(|v| v + shift);
            let fy = D::from_params(&wide(&row)).cdf(&ty);
            (ty, fy)
        };
        let ind = ty.mapv(|v| if v >= y { 1.0 } else { 0.0 });
        let refs = ((&fy - &ind).mapv(|v| v * v)).sum() * h;
        err.score = err.score.max((s - refs).abs() / refs.abs().max(1e-12));
        let g = Scorable::<CRPScore>::d_score(&d, &yv);
        for j in 0..k {
            let (mut a, mut b) = (row.clone(), row.clone());
            a[[0, j]] += 1e-6;
            b[[0, j]] -= 1e-6;
            let fd = (Scorable::<CRPScore>::score(&D::from_params(&a), &yv)[0]
                - Scorable::<CRPScore>::score(&D::from_params(&b), &yv)[0])
                / 2e-6;
            err.grad = err
                .grad
                .max((g[[0, j]] - fd).abs() / fd.abs().max(1e-3 * refs.abs()).max(1e-9));
        }
    }
    err
}

type AuditFn = fn() -> Audit;

fn cases() -> Vec<(&'static str, AuditFn)> {
    vec![
        ("Normal", || {
            audit::<Normal>(&[0.3, 0.7f64.ln()], &[-0.5, 0.3, 2.0], false)
        }),
        ("NormalFixedVar", || {
            audit::<NormalFixedVar>(&[0.3], &[-0.5, 0.3, 2.0], false)
        }),
        ("NormalFixedMean", || {
            audit::<NormalFixedMean>(&[0.7f64.ln()], &[-0.5, 0.3, 2.0], false)
        }),
        ("Laplace", || {
            audit::<Laplace>(&[0.3, 0.7f64.ln()], &[-0.5, 0.3, 2.0], false)
        }),
        ("Exponential", || {
            audit::<Exponential>(&[1.5f64.ln()], &[0.2, 1.0, 4.0], false)
        }),
        ("Gamma", || {
            audit::<Gamma>(&[2.5f64.ln(), 1.3f64.ln()], &[0.2, 1.5, 4.0], false)
        }),
        ("HalfNormal", || {
            audit::<HalfNormal>(&[1.3f64.ln()], &[0.2, 1.0, 3.0], false)
        }),
        ("Weibull", || {
            audit::<Weibull>(&[1.7f64.ln(), 2.0f64.ln()], &[0.3, 1.5, 4.0], false)
        }),
        ("StudentT", || {
            audit::<StudentT>(&[0.3, 0.7f64.ln(), 6.0f64.ln()], &[-0.5, 0.3, 2.0], false)
        }),
        ("TFixedDf", || {
            audit::<TFixedDf>(&[0.3, 0.7f64.ln()], &[-0.5, 0.3, 2.0], false)
        }),
        ("TFixedDfFixedVar", || {
            audit::<TFixedDfFixedVar>(&[0.3], &[-0.5, 0.3, 2.0], false)
        }),
        ("Poisson", || {
            audit::<Poisson>(&[3.2f64.ln()], &[0.0, 2.0, 7.0], true)
        }),
        ("NegativeBinomial", || {
            audit::<NegativeBinomial>(&[3.2f64.ln(), 1.5f64.ln()], &[0.0, 2.0, 7.0], true)
        }),
        ("HurdleNormal", || {
            audit::<HurdleNormal>(&[-0.4, 1.2, 0.8f64.ln()], &[0.0, 0.7, 2.5], false)
        }),
        ("HurdleLogNormal", || {
            audit::<HurdleLogNormal>(&[-0.4, 0.3, 0.6f64.ln()], &[0.0, 0.7, 2.5], false)
        }),
    ]
}

/// NormalMixture uses only the diagonal of 2∫∇F∇Fᵀ by design (the full
/// matrix was worse in R39), so only its diagonal ratios are checked.
fn mixture_audit(c: f64) -> Audit {
    let l = c.ln();
    audit::<NormalMixture<2>>(
        &[0.4, -1.0 * c, 0.6f64.ln() + l, 1.0 * c, 0.9f64.ln() + l],
        &[-1.2 * c, 0.1 * c, 1.5 * c],
        false,
    )
}

#[test]
#[ignore]
fn audit_report() {
    println!(
        "{:<18}{:>12}{:>12}{:>12}",
        "head", "score", "d_score", "metric"
    );
    for (name, f) in cases() {
        let a = f();
        println!(
            "{name:<18}{:>12.2e}{:>12.2e}{:>12.2e}",
            a.score, a.grad, a.metric
        );
    }
    let a = mixture_audit(1.0);
    println!(
        "{:<18}{:>12.2e}{:>12.2e}  diagonal ratios {:?}",
        "NormalMixture<2>", a.score, a.grad, a.ratio
    );
}

#[test]
#[ignore]
fn audit_scale_dependence() {
    // ratio implemented/reference metric across the scale parameter: a
    // constant ratio is a benign rescaling, a varying one a units bug
    for c in [0.1f64, 1.0, 10.0] {
        let l = c.ln();
        println!("scale ×{c}:");
        println!(
            "  HalfNormal {:?}",
            audit::<HalfNormal>(&[l], &[0.2 * c, c, 3.0 * c], false).ratio
        );
        println!(
            "  Gamma(rate/c) {:?}",
            audit::<Gamma>(
                &[2.5f64.ln(), 1.3f64.ln() - l],
                &[0.2 * c, 1.5 * c, 4.0 * c],
                false
            )
            .ratio
        );
        println!(
            "  Weibull {:?}",
            audit::<Weibull>(
                &[1.7f64.ln(), 2.0f64.ln() + l],
                &[0.3 * c, 1.5 * c, 4.0 * c],
                false
            )
            .ratio
        );
        println!(
            "  StudentT {:?}",
            audit::<StudentT>(
                &[0.3 * c, 0.7f64.ln() + l, 6.0f64.ln()],
                &[-0.5 * c, 0.3 * c, 2.0 * c],
                false
            )
            .ratio
        );
        println!(
            "  TFixedDf {:?}",
            audit::<TFixedDf>(
                &[0.3 * c, 0.7f64.ln() + l],
                &[-0.5 * c, 0.3 * c, 2.0 * c],
                false
            )
            .ratio
        );
        println!(
            "  Exponential {:?}",
            audit::<Exponential>(&[1.5f64.ln() + l], &[0.2 * c, c, 4.0 * c], false).ratio
        );
    }
    for lam in [0.5f64, 3.2, 30.0] {
        println!(
            "  Poisson λ={lam} {:?}",
            audit::<Poisson>(&[lam.ln()], &[0.0, lam.floor(), 2.0 * lam], true).ratio
        );
    }
}

/// The CRPS metric of every head equals 2∫∇F∇Fᵀ dt (sums for Poisson), the
/// score equals ∫(F − 1{t ≥ y})² dt and d_score its derivative. Upstream's
/// metrics were off by a scale (or λ) dependent factor (Cycles 13–14).
#[test]
fn crps_metrics_match_definition() {
    for (name, f) in cases() {
        let a = f();
        assert!(a.metric < 1e-4, "{name}: metric rel err {}", a.metric);
        assert!(a.score < 1e-3, "{name}: score rel err {}", a.score);
        assert!(a.grad < 1e-4, "{name}: d_score rel err {}", a.grad);
    }
    // and stays right across scales (the units bug)
    for c in [0.1f64, 10.0] {
        let l = c.ln();
        for r in [
            audit::<HalfNormal>(&[l], &[0.2 * c, c], false).ratio,
            audit::<Gamma>(&[2.5f64.ln(), 1.3f64.ln() - l], &[0.2 * c, 1.5 * c], false).ratio,
            audit::<Weibull>(&[1.7f64.ln(), 2.0f64.ln() + l], &[0.3 * c, 1.5 * c], false).ratio,
            audit::<StudentT>(&[0.3 * c, 0.7f64.ln() + l, 6.0f64.ln()], &[0.3 * c], false).ratio,
            audit::<TFixedDf>(&[0.3 * c, 0.7f64.ln() + l], &[0.3 * c], false).ratio,
        ] {
            assert!(
                r.iter().all(|v| (v - 1.0).abs() < 1e-4),
                "scale ×{c}: {r:?}"
            );
        }
    }
    for lam in [0.5f64, 30.0] {
        let r = audit::<Poisson>(&[lam.ln()], &[lam.floor()], true).ratio;
        assert!((r[0] - 1.0).abs() < 1e-6, "Poisson λ={lam}: {r:?}");
        let r = audit::<NegativeBinomial>(&[lam.ln(), 1.5f64.ln()], &[lam.floor()], true).ratio;
        assert!(
            r.iter().all(|v| (v - 1.0).abs() < 1e-4),
            "NegativeBinomial μ={lam}: {r:?}"
        );
    }
    for c in [0.1f64, 1.0, 10.0] {
        let l = c.ln();
        for r in [
            audit::<HurdleNormal>(&[-0.4, 1.2 * c, 0.8f64.ln() + l], &[0.0, 0.7 * c], false).ratio,
            audit::<HurdleLogNormal>(&[-0.4, 0.3 + l, 0.6f64.ln()], &[0.0, 0.7 * c], false).ratio,
        ] {
            assert!(
                r.iter().all(|v| (v - 1.0).abs() < 1e-4),
                "hurdle scale ×{c}: {r:?}"
            );
        }
        let a = mixture_audit(c);
        assert!(a.score < 1e-3 && a.grad < 1e-4, "mixture ×{c}: {a:?}");
        assert!(
            a.ratio.iter().all(|v| (v - 1.0).abs() < 1e-4),
            "NormalMixture<2> diagonal ×{c}: {:?}",
            a.ratio
        );
    }
}
