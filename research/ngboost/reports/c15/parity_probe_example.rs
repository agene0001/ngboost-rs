//! Parity probe (research Cycle 15): fits a fixed set of heads/scores on
//! deterministic synthetic data and prints an exact hash of the test-set
//! parameters. Built against 1.1.2 (no knobs) and the current tree (every
//! opt-out switch set), the hashes must agree for "the switches restore
//! upstream behaviour" to hold.
use ndarray::{Array1, Array2};
use ngboost_rs::dist::*;
use ngboost_rs::learners::histogram_learner;
use ngboost_rs::{CRPScore, LogScore, NGBoost};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

macro_rules! knobs { ($m:ident) => {}; }

macro_rules! case {
    ($name:expr, $d:ty, $s:ty, $x:expr, $y:expr, $xv:expr, $yv:expr, $xt:expr, $val:expr, $n:expr) => {{
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
        match r {
            Ok(()) => {
                let p = m.pred_param($xt);
                println!(
                    "{:<26} iters {:>4}  hash {:016x}  sum {:+.15e}",
                    $name,
                    m.base_models.len(),
                    hash(&p),
                    p.sum()
                );
            }
            Err(e) => println!("{:<26} ERROR {}", $name, e),
        }
    }};
}

fn main() {
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
        ycat[i] = if z < -0.7 { 0.0 } else if z < 0.7 { 1.0 } else { 2.0 };
    }
    let sl = |v: &Array1<f64>, lo: usize, hi: usize| v.slice(ndarray::s![lo..hi]).to_owned();
    let xs = |lo: usize, hi: usize| x.slice(ndarray::s![lo..hi, ..]).to_owned();
    let (xtr, xva, xte) = (xs(0, n), xs(n, n + nv), xs(n + nv, tot));
    let (ntr, nva) = (sl(&yn, 0, n), sl(&yn, n, n + nv));
    let (ptr, pva) = (sl(&ypos, 0, n), sl(&ypos, n, n + nv));
    let (ctr, cva) = (sl(&ycnt, 0, n), sl(&ycnt, n, n + nv));
    let (btr, bva) = (sl(&ybin, 0, n), sl(&ybin, n, n + nv));
    let (ktr, kva) = (sl(&ycat, 0, n), sl(&ycat, n, n + nv));

    case!("normal log val", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("normal log valfrac", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 2, 300);
    case!("normal log noval", Normal, LogScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!("normal crps", Normal, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!("normal crps val", Normal, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("normalfixedmean log", NormalFixedMean, LogScore, &xtr, &ntr, &xva, &nva, &xte, 0, 100);
    case!("normalfixedvar log val", NormalFixedVar, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("laplace log val", Laplace, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("laplace crps", Laplace, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!("studentt log val", StudentT, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 200);
    case!("studentt crps", StudentT, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 60);
    case!("tfixeddf log val", TFixedDf, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("tfixeddf crps", TFixedDf, CRPScore, &xtr, &ntr, &xva, &nva, &xte, 0, 150);
    case!("cauchy log val", Cauchy, LogScore, &xtr, &ntr, &xva, &nva, &xte, 1, 300);
    case!("lognormal log val", LogNormal, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!("lognormal crps", LogNormal, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!("halfnormal crps", HalfNormal, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!("exponential crps", Exponential, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 150);
    case!("gamma log val", Gamma, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!("gamma crps", Gamma, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 60);
    case!("weibull log val", Weibull, LogScore, &xtr, &ptr, &xva, &pva, &xte, 1, 300);
    case!("weibull crps", Weibull, CRPScore, &xtr, &ptr, &xva, &pva, &xte, 0, 60);
    case!("poisson log val", Poisson, LogScore, &xtr, &ctr, &xva, &cva, &xte, 1, 300);
    case!("poisson crps", Poisson, CRPScore, &xtr, &ctr, &xva, &cva, &xte, 0, 150);
    case!("bernoulli log val", Bernoulli, LogScore, &xtr, &btr, &xva, &bva, &xte, 1, 300);
    case!("categorical3 log val", Categorical3, LogScore, &xtr, &ktr, &xva, &kva, &xte, 1, 300);
}
