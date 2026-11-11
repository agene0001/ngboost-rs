//! Upstream-parity guard (research Cycle 15).
//!
//! Every default changed since 1.1.2 has an opt-out. With all of them set —
//! `auto_calibrate_scale = false`, `fisher_weight_power = Some(0.0)`,
//! `refit_intercept = Some(false)`, `staged_fit = Some(false)`, and the
//! process-wide `NGBOOST_NORMAL_GUARD=0`, `NGBOOST_CRPS_METRIC_LEGACY=1` —
//! the fit must be bit-identical to 1.1.2 (the last release that matched
//! upstream NGBoost's algorithm). The expected hashes of the test-set
//! parameters were recorded by building this same probe against the 1.1.2
//! source (`git archive 7c91877`), see `research/ngboost/reports/c15/`.
//!
//! Exact float hashes depend on the platform and BLAS, so the test runs only
//! where they were recorded (macOS aarch64 with `accelerate`). This file is
//! its own test binary on purpose: the env switches are read once per
//! process.
use ndarray::{Array1, Array2};
use ngboost_rs::dist::*;
use ngboost_rs::learners::histogram_learner;
use ngboost_rs::{CRPScore, LogScore, NGBoost};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.u()).cos()
    }
}

fn hash(p: &Array2<f64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in p.iter() {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

macro_rules! knobs {
    ($m:ident) => {
        $m.auto_calibrate_scale = false;
        $m.fisher_weight_power = Some(0.0);
        $m.refit_intercept = Some(false);
        $m.staged_fit = Some(false);
    };
}

macro_rules! case {
    ($out:ident; $name:expr, $d:ty, $s:ty, $x:expr, $y:expr, $xv:expr, $yv:expr, $xt:expr, $val:expr, $n:expr) => {{
        let mut m = NGBoost::<$d, $s, _>::with_seed($n, 0.05, histogram_learner(), 7);
        knobs!(m);
        let r = if $val == 1 {
            m.early_stopping_rounds = Some(20);
            m.fit_with_validation($x, $y, Some($xv), Some($yv), None, None)
        } else if $val == 2 {
            m.early_stopping_rounds = Some(20);
            m.validation_fraction = 0.2;
            m.fit($x, $y)
        } else {
            m.fit($x, $y)
        };
        let p = r.map(|()| m.pred_param($xt));
        $out.push((
            $name,
            p.map(|p| (m.base_models.len(), format!("{:016x}", hash(&p)))),
        ));
    }};
}

#[test]
#[cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64", feature = "accelerate")),
    ignore = "expected hashes recorded on macOS aarch64 with Accelerate"
)]
fn opt_out_switches_reproduce_v1_1_2_bit_for_bit() {
    // SAFETY: set before any fit, and this test binary has a single test.
    unsafe {
        std::env::set_var("NGBOOST_NORMAL_GUARD", "0");
        std::env::set_var("NGBOOST_CRPS_METRIC_LEGACY", "1");
    }
    let mut out: Vec<(&str, Result<(usize, String), &str>)> = Vec::new();
    let (n, nv, nt, p) = (600usize, 200usize, 200usize, 4usize);
    let mut r = Lcg(42);
    let tot = n + nv + nt;
    let x = Array2::from_shape_fn((tot, p), |_| r.u() * 2.0 - 1.0);
    let mut yn = Array1::zeros(tot);
    let mut ypos = Array1::zeros(tot);
    let mut ycnt = Array1::zeros(tot);
    let mut ybin = Array1::zeros(tot);
    let mut ycat = Array1::zeros(tot);
    for i in 0..tot {
        let (a, b, c) = (x[[i, 0]], x[[i, 1]], x[[i, 2]]);
        let mut e = r.n();
        if r.u() < 0.03 {
            e *= 8.0; // outliers: exercise the log-sigma guard
        }
        yn[i] = 2.0 * a + (3.0 * b).sin() + (0.3 + 0.5 * (c + 1.0)) * e;
        ypos[i] = (0.5 * a + 0.4 * b + 0.3 * (0.6 + 0.3 * c) * r.n()).exp() * 3.0;
        let lam = (0.8 + a + 0.5 * b).exp();
        let (mut k, mut pk, u) = (0.0f64, (-lam).exp(), r.u());
        let mut cdf = pk;
        while u > cdf && k < 200.0 {
            k += 1.0;
            pk *= lam / k;
            cdf += pk;
        }
        ycnt[i] = k;
        let z = 1.5 * a - b + 0.5 * r.n();
        ybin[i] = if z > 0.0 { 1.0 } else { 0.0 };
        ycat[i] = if z < -0.7 {
            0.0
        } else if z < 0.7 {
            1.0
        } else {
            2.0
        };
    }
    let sl = |v: &Array1<f64>, lo: usize, hi: usize| v.slice(ndarray::s![lo..hi]).to_owned();
    let xs = |lo: usize, hi: usize| x.slice(ndarray::s![lo..hi, ..]).to_owned();
    let (xtr, xva, xte) = (xs(0, n), xs(n, n + nv), xs(n + nv, tot));
    let (ntr, nva) = (sl(&yn, 0, n), sl(&yn, n, n + nv));
    let (ptr, pva) = (sl(&ypos, 0, n), sl(&ypos, n, n + nv));
    let (ctr, cva) = (sl(&ycnt, 0, n), sl(&ycnt, n, n + nv));
    let (btr, bva) = (sl(&ybin, 0, n), sl(&ybin, n, n + nv));
    let (ktr, kva) = (sl(&ycat, 0, n), sl(&ycat, n, n + nv));

    case!(out; "normal log val", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "normal log valfrac", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 2, 300);
    case!(out; "normal log noval", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!(out; "normal crps", Normal, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!(out; "normal crps val", Normal, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "normalfixedmean log", NormalFixedMean, LogScore, &xtr, &ntr, &xva, &nva, &xte, 0, 100);
    case!(out; "normalfixedvar log val", NormalFixedVar, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "laplace log val", Laplace, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "laplace crps", Laplace, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!(out; "studentt log val", StudentT, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 200);
    case!(out; "studentt crps", StudentT, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 60);
    case!(out; "tfixeddf log val", TFixedDf, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "tfixeddf crps", TFixedDf, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!(out; "cauchy log val", Cauchy, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!(out; "lognormal log val", LogNormal, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!(out; "lognormal crps", LogNormal, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!(out; "halfnormal crps", HalfNormal, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!(out; "exponential crps", Exponential, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!(out; "gamma log val", Gamma, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!(out; "gamma crps", Gamma, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 60);
    case!(out; "weibull log val", Weibull, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!(out; "weibull crps", Weibull, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 60);
    case!(out; "poisson log val", Poisson, LogScore, &xtr, &ctr, &xva, &cva, &xte, 1, 300);
    case!(out; "poisson crps", Poisson, CRPScore, &xtr, &ctr, &xva, &cva, &xte, 0, 150);
    case!(out; "bernoulli log val", Bernoulli, LogScore, &xtr, &btr, &xva, &bva, &xte, 1, 300);
    case!(out; "categorical3 log val", Categorical3, LogScore, &xtr, &ktr, &xva, &kva, &xte, 1, 300);

    // Recorded from the 1.1.2 build of this probe: (case, iterations, hash).
    let expected: [(&str, usize, &str); 26] = [
        ("normal log val", 36, "a219861a8c120d8d"),
        ("normal log valfrac", 23, "766d893f976aca6f"),
        ("normal log noval", 150, "9f012ebd05a40a8d"),
        ("normal crps", 150, "3f60eb0345107c32"),
        ("normal crps val", 64, "82074cb81c6f47a3"),
        ("normalfixedmean log", 100, "e1cbbcca3e122c85"),
        ("normalfixedvar log val", 29, "89ea22d23b852552"),
        ("laplace log val", 56, "adf4eb891675bb23"),
        ("laplace crps", 150, "5bc38c30dd7b4a10"),
        ("studentt log val", 23, "e5937526cb8099db"),
        ("studentt crps", 60, "81d9debf97d3323e"),
        ("tfixeddf log val", 47, "e3e540b2f66224ac"),
        ("tfixeddf crps", 150, "ab185ea4246641f2"),
        ("cauchy log val", 52, "5c56bd9291155ff9"),
        ("lognormal log val", 40, "c5f2017b4dd61d0f"),
        ("lognormal crps", 150, "75ed39cc40e0850f"),
        ("halfnormal crps", 150, "2b571adbc1b3c3fd"),
        ("exponential crps", 150, "77108639f6e7af93"),
        ("gamma log val", 73, "e22ac1924ef2dbe2"),
        ("gamma crps", 60, "091db8e308ddf9fa"),
        ("weibull log val", 48, "5bff37e9a3582580"),
        ("weibull crps", 60, "20131f3a0ecebd64"),
        ("poisson log val", 44, "87d5bca421fad7ad"),
        ("poisson crps", 150, "639e85d46f77aa66"),
        ("bernoulli log val", 29, "746b6c6fd0140e9d"),
        ("categorical3 log val", 37, "a177b82c97899ca5"),
    ];
    assert_eq!(out.len(), expected.len());
    let mut bad = Vec::new();
    for ((name, got), (en, ei, eh)) in out.iter().zip(expected.iter()) {
        assert_eq!(name, en);
        match got {
            Ok((i, h)) if i == ei && h == eh => {}
            other => bad.push(format!("{name}: got {other:?}, 1.1.2 gave ({ei}, {eh})")),
        }
    }
    assert!(bad.is_empty(), "differs from 1.1.2:\n{}", bad.join("\n"));
}
