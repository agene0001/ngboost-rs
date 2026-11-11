//! CRPS metric of the Normal family (research/ngboost Cycle 13): the metric
//! must equal 2∫∇F(t)∇F(t)ᵀ dt, and the natural gradient must be
//! equivariant under a change of units (y → c·y). Upstream's metric was σ
//! times too large, which broke both.
use ndarray::{Array1, Array2, array};
use ngboost_rs::dist::{Distribution, DistributionMethods, LogNormal, Normal, NormalFixedMean};
use ngboost_rs::scores::{CRPScore, Scorable};

/// 2∫ ∂F/∂θa ∂F/∂θb dt by finite differences in the parameters and a
/// midpoint rule over t (in the variable the CRPS is taken on).
fn quad_metric<D: Distribution + DistributionMethods>(
    p: &Array2<f64>,
    lo: f64,
    hi: f64,
    to_y: fn(f64) -> f64,
) -> Array2<f64> {
    let k = p.ncols();
    let n = 60_000;
    let h = (hi - lo) / n as f64;
    let t: Array1<f64> = (0..n).map(|i| to_y(lo + (i as f64 + 0.5) * h)).collect();
    let rows = Array2::from_shape_fn((n, k), |(_, j)| p[[0, j]]);
    let mut grads = Vec::new();
    for j in 0..k {
        let e = 1e-6;
        let (mut a, mut b) = (rows.clone(), rows.clone());
        a.column_mut(j).mapv_inplace(|v| v + e);
        b.column_mut(j).mapv_inplace(|v| v - e);
        grads.push((D::from_params(&a).cdf(&t) - D::from_params(&b).cdf(&t)) / (2.0 * e));
    }
    Array2::from_shape_fn((k, k), |(a, b)| 2.0 * (&grads[a] * &grads[b]).sum() * h)
}

#[test]
fn normal_crps_metric_is_integrated_cdf_gradient_product() {
    for (mu, s) in [(0.0, 1.0), (3.0, 0.05), (-2.0, 40.0)] {
        let p = array![[mu, f64::ln(s)]];
        let m = Scorable::<CRPScore>::metric(&Normal::from_params(&p));
        let q = quad_metric::<Normal>(&p, mu - 12.0 * s, mu + 12.0 * s, |v| v);
        for a in 0..2 {
            for b in 0..2 {
                let tol = 1e-5 * q[[a, a]].abs().max(1e-12).max(m[[0, a, a]]);
                assert!(
                    (m[[0, a, b]] - q[[a, b]]).abs() < tol.max(1e-9),
                    "σ={s} ({a},{b}): {} vs {}",
                    m[[0, a, b]],
                    q[[a, b]]
                );
            }
        }
        // NormalFixedMean (loc = 0): the log σ entry
        let pf = array![[f64::ln(s)]];
        let mf = Scorable::<CRPScore>::metric(&NormalFixedMean::from_params(&pf));
        let qf = quad_metric::<NormalFixedMean>(&pf, -12.0 * s, 12.0 * s, |v| v);
        assert!(
            (mf[[0, 0, 0]] - qf[[0, 0]]).abs() < 1e-5 * qf[[0, 0]],
            "fixed mean σ={s}"
        );
        // LogNormal: CRPS of ln y, integrated over u = ln y
        let ml = Scorable::<CRPScore>::metric(&LogNormal::from_params(&p));
        let ql = quad_metric::<LogNormal>(&p, mu - 12.0 * s, mu + 12.0 * s, f64::exp);
        // dt = du on the log scale: the quadrature above integrates over u
        for a in 0..2 {
            assert!(
                (ml[[0, a, a]] - ql[[a, a]]).abs() < 1e-5 * ql[[a, a]],
                "lognormal σ={s} ({a},{a})"
            );
        }
    }
}

#[test]
fn normal_crps_natural_gradient_is_unit_equivariant() {
    // y → c·y with μ → c·μ, σ → c·σ: the μ step scales by c, the log σ step
    // is unchanged (a unit change must not change the fit)
    let p = array![[0.7, f64::ln(0.3)], [-1.0, f64::ln(2.0)]];
    let y = array![0.2, 1.5];
    let ng = Scorable::<CRPScore>::grad(&Normal::from_params(&p), &y, true);
    for c in [1e-3, 37.0, 1e4] {
        let pc = array![[0.7 * c, f64::ln(0.3 * c)], [-1.0 * c, f64::ln(2.0 * c)]];
        let ngc = Scorable::<CRPScore>::grad(&Normal::from_params(&pc), &(&y * c), true);
        for i in 0..2 {
            assert!(
                (ngc[[i, 0]] - c * ng[[i, 0]]).abs() < 1e-9 * c.max(1.0),
                "μ step, c={c}"
            );
            assert!((ngc[[i, 1]] - ng[[i, 1]]).abs() < 1e-9, "log σ step, c={c}");
        }
    }
}
