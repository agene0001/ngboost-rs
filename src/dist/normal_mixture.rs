//! K-component Normal mixture head for conditionally multimodal targets
//! (added 1.2.0, research — see `docs/normal_mixture.md`).
//!
//! Internal columns (3K − 1): logits `l₁..l_{K−1}` of the component weights
//! (softmax with the last component as reference, l_K = 0), then `(μ_k,
//! log σ_k)` for k = 1..K. For K = 2 this is `[logit π, μ₁, log σ₁, μ₂,
//! log σ₂]`. f(y) = Σ_k π_k φ(y; μ_k, σ_k).
//!
//! **LogScore metric.** The observed Fisher information of a mixture has no
//! closed form; the head uses the *complete-data* Fisher (component labels
//! treated as observed). It dominates the observed Fisher in the Loewner
//! order (verified by quadrature in the tests), so natural-gradient steps are
//! conservative and EM-like (cf. Xu & Jordan, Neural Computation 1996): with
//! responsibilities r_k the μ_k step is the responsibility-weighted residual
//! (r_k/π_k)(μ_k − y). Blocks: weights logits diag(p) − ppᵀ (inverted in
//! closed form, D⁻¹ + 11ᵀ/p_K), μ_k: π_k/σ_k², log σ_k: 2π_k. Weights in the
//! metric are floored at `PI_FLOOR`.
//!
//! **CRPScore.** Closed form (Grimit, Gneiting, Berrocal & Johnson, QJRMS
//! 2006): CRPS = Σ_k π_k A(y−μ_k, σ_k) − ½ Σ_k Σ_m π_k π_m A(μ_k−μ_m, s_km),
//! A(m, s) = m(2Φ(m/s) − 1) + 2sφ(m/s), s_km = √(σ_k² + σ_m²). The metric is
//! the diagonal of the CRPS Riemannian metric 2∫∇F∇Fᵀ dy. The full matrix
//! is also available in closed form (`NGBOOST_MIX_CRPS_FULL=1`): μ/log σ
//! pairs are Gaussian-product moments, weight × μ pairs Φ-cross
//! probabilities P(X_a ≤ X_b), weight × log σ pairs Stein's lemma, weight ×
//! weight pairs energy distances. Neither makes CRPS training competitive
//! with LogScore for this head (research/ngboost R35, R39).
//!
//! **Safeguard.** The LogScore natural gradient uses, for the μ and log σ
//! columns, the larger of the expected complete-data Fisher and the per-row
//! observed curvature (`guarded_entries`); this bounds the step for rows far
//! from every component. `NGBOOST_MIX_GUARD=0` disables it (research A/B).
use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3};

const HALF_LN_2PI: f64 = 0.9189385332046727;
const INV_SQRT_2PI: f64 = 0.3989422804014327;
const INV_SQRT_PI: f64 = 0.5641895835477563;
const PI_FLOOR: f64 = 1e-3;
const LOGIT_MAX: f64 = 15.0;
/// Cap on a row's log σ natural-gradient step (see `guarded_entries`).
const GUARD_LOG_SIGMA_STEP: f64 = 5.0;

/// Observed-curvature safeguard for the LogScore natural gradient (default
/// on; `NGBOOST_MIX_GUARD=0` disables it for A/B research). See
/// `guarded_entries`.
fn guard_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NGBOOST_MIX_GUARD").map_or(true, |v| v.trim() != "0"))
}

/// CRPScore metric: the diagonal of the closed-form 2∫∇F∇Fᵀ (default) or the
/// full matrix (`NGBOOST_MIX_CRPS_FULL=1`, research). The full metric was
/// worse on 4 of 6 datasets and still lost to LogScore training everywhere
/// (research/ngboost R39).
fn crps_diag_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("NGBOOST_MIX_CRPS_FULL").is_ok_and(|v| v.trim() == "1"))
}

/// N(d; 0, s²) density.
#[inline]
fn gauss(d: f64, s2: f64) -> f64 {
    (-0.5 * d * d / s2).exp() / (2.0 * std::f64::consts::PI * s2).sqrt()
}

/// K-component Normal mixture (see module docs).
#[derive(Debug, Clone)]
pub struct NormalMixture<const K: usize> {
    /// Component weights, shape (n, K).
    pub weights: Array2<f64>,
    /// Component means, shape (n, K).
    pub mu: Array2<f64>,
    /// Component standard deviations, shape (n, K).
    pub sigma: Array2<f64>,
    raw: Array2<f64>,
}

/// Two-component mixture (`NGBMixtureRegressor`).
pub type NormalMixture2 = NormalMixture<2>;
/// Three-component mixture.
pub type NormalMixture3 = NormalMixture<3>;

#[inline]
fn std_cdf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

#[inline]
fn std_pdf(z: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// A(m, s) = E|m + sZ| = m(2Φ(m/s) − 1) + 2sφ(m/s); returns (A, ∂A/∂m, ∂A/∂s).
#[inline]
fn a_fn(m: f64, s: f64) -> (f64, f64, f64) {
    let z = m / s;
    let (cdf, pdf) = (std_cdf(z), std_pdf(z));
    (
        m * (2.0 * cdf - 1.0) + 2.0 * s * pdf,
        2.0 * cdf - 1.0,
        2.0 * pdf,
    )
}

impl<const K: usize> NormalMixture<K> {
    #[inline]
    fn mu_col(k: usize) -> usize {
        (K - 1) + 2 * k
    }
    #[inline]
    fn ls_col(k: usize) -> usize {
        (K - 1) + 2 * k + 1
    }

    /// (log f(y), responsibilities) for row i.
    fn row_logpdf_resp(&self, i: usize, y: f64) -> (f64, [f64; K]) {
        let mut t = [0.0f64; K];
        let mut mx = f64::NEG_INFINITY;
        for k in 0..K {
            let s = self.sigma[[i, k]];
            let z = (y - self.mu[[i, k]]) / s;
            t[k] = self.weights[[i, k]].max(1e-300).ln() - HALF_LN_2PI - s.ln() - 0.5 * z * z;
            mx = mx.max(t[k]);
        }
        let sum: f64 = t.iter().map(|v| (v - mx).exp()).sum();
        let lf = mx + sum.ln();
        let mut r = [0.0f64; K];
        for k in 0..K {
            r[k] = (t[k] - lf).exp();
        }
        (lf, r)
    }

    fn row_cdf(&self, i: usize, y: f64) -> f64 {
        (0..K)
            .map(|k| self.weights[[i, k]] * std_cdf((y - self.mu[[i, k]]) / self.sigma[[i, k]]))
            .sum()
    }

    fn row_pdf(&self, i: usize, y: f64) -> f64 {
        self.row_logpdf_resp(i, y).0.exp()
    }

    /// LogScore metric entries for row i with the step safeguard: μ_k →
    /// π_k/σ_k² (unchanged), log σ_k → max(2π_k, |r_k(1 − z_k²)|/C), C =
    /// `GUARD_LOG_SIGMA_STEP` — i.e. a row's log σ natural-gradient step is
    /// capped at C. Without it, one training row far in a component's tail,
    /// alone in a leaf, drove that component's σ to 2·10⁸ in a single round
    /// on full protein (research/ngboost R37): the step (1 − z²)/2 linearizes
    /// a log-scale move whose exact per-point optimum is only ln|z|. The
    /// entry never drops below the expected complete-data Fisher, so the
    /// metric stays positive definite and steps are only damped. A broader
    /// version (max(π, r·max(z², 1)), damping every row with r_k > π_k)
    /// removed the drift but cost +0.03…+0.12 NLL (R38) and was rejected.
    fn guarded_entries(&self, i: usize, y: f64, r: &[f64; K], k: usize) -> (f64, f64) {
        let p = self.weights[[i, k]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
        let s = self.sigma[[i, k]];
        if !guard_enabled() {
            return (p / (s * s), 2.0 * p);
        }
        // log σ entry raised just enough that |ñ| = |g|/m ≤ GUARD_LOG_SIGMA_STEP;
        // g = r(1 − z²). Engages only for rows far into a component's tail
        // (≈ |z| > 3.3 when r ≈ π); μ is left unguarded.
        let z = (y - self.mu[[i, k]]) / s;
        let g = r[k] * (1.0 - z * z);
        (p / (s * s), (2.0 * p).max(g.abs() / GUARD_LOG_SIGMA_STEP))
    }

    /// Full closed-form CRPS metric 2∫∇F∇Fᵀ dy for row i (see module docs):
    /// Gaussian-product moments for μ/log σ pairs, Φ-cross probabilities and
    /// Stein's lemma for weight × (μ, log σ) pairs, energy distances for
    /// weight × weight pairs.
    fn crps_full_metric(&self, i: usize) -> Array2<f64> {
        let np = 3 * K - 1;
        let p: Vec<f64> = (0..K).map(|k| self.weights[[i, k]]).collect();
        let mu: Vec<f64> = (0..K).map(|k| self.mu[[i, k]]).collect();
        let sg: Vec<f64> = (0..K).map(|k| self.sigma[[i, k]]).collect();
        let mut c = [[0.0f64; K]; K]; // ∫ f_a f_b
        let mut pc = [[0.0f64; K]; K]; // ∫ Φ_a f_b = P(X_a ≤ X_b)
        let mut e = [[0.0f64; K]; K]; // energy distances
        for a in 0..K {
            for b in 0..K {
                let s2 = sg[a] * sg[a] + sg[b] * sg[b];
                c[a][b] = gauss(mu[a] - mu[b], s2);
                pc[a][b] = std_cdf((mu[b] - mu[a]) / s2.sqrt());
                e[a][b] = self.energy(i, a, b);
            }
        }
        let mut m = Array2::zeros((np, np));
        // μ / log σ blocks
        for a in 0..K {
            for b in 0..K {
                let s2 = sg[a] * sg[a] + sg[b] * sg[b];
                let v = sg[a] * sg[a] * sg[b] * sg[b] / s2;
                let ma_minus_mub = sg[b] * sg[b] * (mu[a] - mu[b]) / s2; // m − μ_b
                let m_minus_mua = sg[a] * sg[a] * (mu[b] - mu[a]) / s2; // m − μ_a
                let w = p[a] * p[b] * c[a][b];
                m[[Self::mu_col(a), Self::mu_col(b)]] = 2.0 * w;
                m[[Self::mu_col(a), Self::ls_col(b)]] = 2.0 * w * ma_minus_mub;
                m[[Self::ls_col(a), Self::ls_col(b)]] = 2.0 * w * (v + m_minus_mua * ma_minus_mub);
            }
        }
        for a in 0..K {
            for b in 0..K {
                m[[Self::ls_col(b), Self::mu_col(a)]] = m[[Self::mu_col(a), Self::ls_col(b)]];
            }
        }
        // weight logits × (μ, log σ)
        for j in 0..K - 1 {
            for k in 0..K {
                let mixp: f64 = (0..K).map(|q| p[q] * pc[q][k]).sum();
                let mixc: f64 = (0..K).map(|q| p[q] * c[q][k]).sum();
                let wm = -2.0 * p[j] * p[k] * (pc[j][k] - mixp);
                let ws = -2.0 * p[j] * p[k] * sg[k] * sg[k] * (c[j][k] - mixc);
                m[[j, Self::mu_col(k)]] = wm;
                m[[Self::mu_col(k), j]] = wm;
                m[[j, Self::ls_col(k)]] = ws;
                m[[Self::ls_col(k), j]] = ws;
            }
        }
        // weight logits × weight logits
        for a in 0..K - 1 {
            for b in 0..K - 1 {
                let mut acc = 0.0;
                for k in 0..K {
                    for q in 0..K {
                        acc += p[k] * p[q] * 0.5 * (e[a][q] + e[k][b] - e[a][b] - e[k][q]);
                    }
                }
                m[[a, b]] = 2.0 * p[a] * p[b] * acc;
            }
        }
        // tiny relative jitter: the metric is a Gram matrix (PSD) but can be
        // singular when components coincide
        let tr: f64 = (0..np).map(|d| m[[d, d]]).sum::<f64>() / np as f64;
        for d in 0..np {
            m[[d, d]] += 1e-10 * tr.max(1e-300);
        }
        m
    }

    /// Energy distance ∫(Φ_a − Φ_b)² dy between components a and b of row i.
    fn energy(&self, i: usize, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }
        let (sa, sb) = (self.sigma[[i, a]], self.sigma[[i, b]]);
        let cross = a_fn(
            self.mu[[i, a]] - self.mu[[i, b]],
            (sa * sa + sb * sb).sqrt(),
        )
        .0;
        // A(0, √2 σ) = 2σ/√π
        cross - sa * INV_SQRT_PI - sb * INV_SQRT_PI
    }
}

impl<const K: usize> Distribution for NormalMixture<K> {
    /// All component log-scales share one recalibration map (relative
    /// component widths are preserved).
    fn scale_param_columns() -> Vec<usize> {
        (0..K).map(Self::ls_col).collect()
    }

    fn from_params(params: &Array2<f64>) -> Self {
        assert!(K >= 2, "NormalMixture needs K >= 2");
        let n = params.nrows();
        let mut raw = params.to_owned();
        for j in 0..K - 1 {
            raw.column_mut(j)
                .mapv_inplace(|a| a.clamp(-LOGIT_MAX, LOGIT_MAX));
        }
        let mut weights = Array2::zeros((n, K));
        let mut mu = Array2::zeros((n, K));
        let mut sigma = Array2::zeros((n, K));
        for i in 0..n {
            let mut mx = 0.0f64; // reference logit l_K = 0
            for j in 0..K - 1 {
                mx = mx.max(raw[[i, j]]);
            }
            let mut tot = 0.0;
            for k in 0..K {
                let l = if k < K - 1 { raw[[i, k]] } else { 0.0 };
                weights[[i, k]] = (l - mx).exp();
                tot += weights[[i, k]];
            }
            for k in 0..K {
                weights[[i, k]] /= tot;
                mu[[i, k]] = raw[[i, Self::mu_col(k)]];
                sigma[[i, k]] = raw[[i, Self::ls_col(k)]].exp();
            }
        }
        NormalMixture {
            weights,
            mu,
            sigma,
            raw,
        }
    }

    /// EM on the marginal target, initialised from K equal-count quantile
    /// groups; components returned ordered by mean.
    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let n = y.len();
        let mut out = Array1::zeros(3 * K - 1);
        let mean = y.mean().unwrap_or(0.0);
        if n < 2 * K {
            for k in 0..K {
                out[Self::mu_col(k)] = mean + k as f64 - (K - 1) as f64 / 2.0;
            }
            return out;
        }
        let mut v: Vec<f64> = y.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let floor = (1e-3 * sd).max(1e-9);
        let (mut pi, mut m, mut s) = ([1.0 / K as f64; K], [0.0f64; K], [0.0f64; K]);
        for k in 0..K {
            let g = &v[k * n / K..((k + 1) * n / K).max(k * n / K + 1)];
            let gm = g.iter().sum::<f64>() / g.len() as f64;
            m[k] = gm;
            s[k] = (g.iter().map(|x| (x - gm).powi(2)).sum::<f64>() / g.len() as f64)
                .sqrt()
                .max(floor);
        }
        for _ in 0..300 {
            let (mut w, mut a, mut b) = ([0.0f64; K], [0.0f64; K], [0.0f64; K]);
            for &yi in y.iter() {
                let mut t = [0.0f64; K];
                let mut mx = f64::NEG_INFINITY;
                for k in 0..K {
                    let z = (yi - m[k]) / s[k];
                    t[k] = pi[k].ln() - s[k].ln() - 0.5 * z * z;
                    mx = mx.max(t[k]);
                }
                let tot: f64 = t.iter().map(|x| (x - mx).exp()).sum();
                for k in 0..K {
                    let r = (t[k] - mx).exp() / tot;
                    w[k] += r;
                    a[k] += r * yi;
                    b[k] += r * yi * yi;
                }
            }
            if w.iter().any(|&x| x < 1e-9) {
                break;
            }
            let mut delta = 0.0;
            for k in 0..K {
                let nm = a[k] / w[k];
                let ns = (b[k] / w[k] - nm * nm).max(floor * floor).sqrt();
                let np = (w[k] / n as f64).clamp(0.01, 0.99);
                delta += (nm - m[k]).abs() + (np - pi[k]).abs();
                (m[k], s[k], pi[k]) = (nm, ns, np);
            }
            let tot: f64 = pi.iter().sum();
            pi.iter_mut().for_each(|p| *p /= tot);
            if delta < 1e-10 * (1.0 + sd) {
                break;
            }
        }
        let mut order: Vec<usize> = (0..K).collect();
        order.sort_by(|&p, &q| m[p].total_cmp(&m[q]));
        let last = order[K - 1];
        for (slot, &k) in order.iter().enumerate() {
            if slot < K - 1 {
                out[slot] = (pi[k] / pi[last]).ln();
            }
            out[Self::mu_col(slot)] = m[k];
            out[Self::ls_col(slot)] = s[k].ln();
        }
        out
    }

    fn n_params(&self) -> usize {
        3 * K - 1
    }

    fn predict(&self) -> Array1<f64> {
        self.mean()
    }

    fn params(&self) -> Array2<f64> {
        self.raw.clone()
    }
}

impl<const K: usize> RegressionDistn for NormalMixture<K> {}

impl<const K: usize> DistributionMethods for NormalMixture<K> {
    fn mean(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.weights.nrows(), |i| {
            (0..K).map(|k| self.weights[[i, k]] * self.mu[[i, k]]).sum()
        })
    }

    fn variance(&self) -> Array1<f64> {
        let m = self.mean();
        Array1::from_shape_fn(self.weights.nrows(), |i| {
            (0..K)
                .map(|k| {
                    let (mu, s) = (self.mu[[i, k]], self.sigma[[i, k]]);
                    self.weights[[i, k]] * (s * s + mu * mu)
                })
                .sum::<f64>()
                - m[i] * m[i]
        })
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| self.row_pdf(i, y[i]))
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| self.row_logpdf_resp(i, y[i]).0)
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| self.row_cdf(i, y[i]))
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(q.len(), |i| {
            let qi = q[i];
            if !(0.0..=1.0).contains(&qi) {
                return f64::NAN;
            }
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for k in 0..K {
                lo = lo.min(self.mu[[i, k]] - 40.0 * self.sigma[[i, k]]);
                hi = hi.max(self.mu[[i, k]] + 40.0 * self.sigma[[i, k]]);
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if self.row_cdf(i, mid) < qi {
                    lo = mid
                } else {
                    hi = mid
                }
                if hi - lo <= 1e-12 * (1.0 + mid.abs()) {
                    break;
                }
            }
            0.5 * (lo + hi)
        })
    }

    /// Highest mode, by mean-shift (the Gaussian-mixture fixed point
    /// x ← Σ w_k(x) μ_k/σ_k² / Σ w_k(x)/σ_k², w_k(x) = π_k φ_k(x)) started
    /// from every component mean; the candidate with the highest density wins.
    fn mode(&self) -> Array1<f64> {
        Array1::from_shape_fn(self.weights.nrows(), |i| {
            let mut best = (f64::NAN, f64::NEG_INFINITY);
            for start in 0..K {
                let mut x = self.mu[[i, start]];
                for _ in 0..200 {
                    let (mut num, mut den) = (0.0, 0.0);
                    for k in 0..K {
                        let s = self.sigma[[i, k]];
                        let w = self.weights[[i, k]] * std_pdf((x - self.mu[[i, k]]) / s) / s;
                        num += w * self.mu[[i, k]] / (s * s);
                        den += w / (s * s);
                    }
                    if !(den > 0.0) {
                        break;
                    }
                    let nx = num / den;
                    let done = (nx - x).abs() <= 1e-12 * (1.0 + x.abs());
                    x = nx;
                    if done {
                        break;
                    }
                }
                let f = self.row_pdf(i, x);
                if f > best.1 {
                    best = (x, f);
                }
            }
            best.0
        })
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        use rand::RngExt;
        let n = self.weights.nrows();
        let mut out = Array2::zeros((n_samples, n));
        let mut rng = rand::rng();
        let mut z = crate::vmath::StdNormalSampler::new();
        for i in 0..n {
            for s in 0..n_samples {
                let u: f64 = rng.random::<f64>();
                let mut acc = 0.0;
                let mut k = K - 1;
                for c in 0..K {
                    acc += self.weights[[i, c]];
                    if u < acc {
                        k = c;
                        break;
                    }
                }
                out[[s, i]] = self.mu[[i, k]] + self.sigma[[i, k]] * z.next(&mut rng);
            }
        }
        out
    }
}

impl<const K: usize> Scorable<LogScore> for NormalMixture<K> {
    /// The weight block diag(p) − ppᵀ is diagonal only for K = 2.
    fn is_diagonal_metric(&self) -> bool {
        K == 2
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let n = self.weights.nrows();
        let mut d = Array2::zeros((n, 3 * K - 1));
        for i in 0..n {
            for k in 0..K {
                let p = self.weights[[i, k]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
                if k < K - 1 {
                    d[[i, k]] = p * (1.0 - p);
                }
                let s = self.sigma[[i, k]];
                d[[i, Self::mu_col(k)]] = p / (s * s);
                d[[i, Self::ls_col(k)]] = 2.0 * p;
            }
        }
        d
    }

    /// Training path for K = 2 (diagonal metric): gradient plus the
    /// safeguarded per-row metric (see `guarded_entries`).
    fn d_score_and_diagonal_metric(&self, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let g = Scorable::<LogScore>::d_score(self, y);
        let mut d = Scorable::<LogScore>::diagonal_metric(self);
        for i in 0..y.len() {
            let (_, r) = self.row_logpdf_resp(i, y[i]);
            for k in 0..K {
                let (em, es) = self.guarded_entries(i, y[i], &r, k);
                d[[i, Self::mu_col(k)]] = em;
                d[[i, Self::ls_col(k)]] = es;
            }
        }
        (g, d)
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| -self.row_logpdf_resp(i, y[i]).0)
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n = y.len();
        let mut g = Array2::zeros((n, 3 * K - 1));
        for i in 0..n {
            let (_, r) = self.row_logpdf_resp(i, y[i]);
            for k in 0..K {
                if k < K - 1 {
                    g[[i, k]] = self.weights[[i, k]] - r[k];
                }
                let s = self.sigma[[i, k]];
                let z = (y[i] - self.mu[[i, k]]) / s;
                g[[i, Self::mu_col(k)]] = -r[k] * z / s;
                g[[i, Self::ls_col(k)]] = r[k] * (1.0 - z * z);
            }
        }
        g
    }

    fn metric(&self) -> Array3<f64> {
        let n = self.weights.nrows();
        let d = Scorable::<LogScore>::diagonal_metric(self);
        let mut m = Array3::zeros((n, 3 * K - 1, 3 * K - 1));
        for i in 0..n {
            for j in 0..3 * K - 1 {
                m[[i, j, j]] = d[[i, j]];
            }
            for a in 0..K - 1 {
                for b in 0..K - 1 {
                    if a != b {
                        let pa = self.weights[[i, a]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
                        let pb = self.weights[[i, b]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
                        m[[i, a, b]] = -pa * pb;
                    }
                }
            }
        }
        m
    }

    /// Closed-form solve: the weight block (diag(p) − ppᵀ)⁻¹ = D⁻¹ + 11ᵀ/p_K
    /// (Sherman–Morrison), every other block diagonal. Falls back to the
    /// generic solver when Tikhonov regularization is requested.
    fn dense_natural_grad(&self, y: &Array1<f64>, reg: f64) -> Array2<f64> {
        let g = Scorable::<LogScore>::d_score(self, y);
        if reg > 0.0 {
            return crate::scores::natural_gradient_regularized(
                &g,
                &Scorable::<LogScore>::metric(self),
                reg,
            );
        }
        let mut out = Array2::zeros(g.raw_dim());
        for i in 0..g.nrows() {
            let pk = self.weights[[i, K - 1]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
            let gsum: f64 = (0..K - 1).map(|j| g[[i, j]]).sum();
            for j in 0..K - 1 {
                let pj = self.weights[[i, j]].clamp(PI_FLOOR, 1.0 - PI_FLOOR);
                out[[i, j]] = g[[i, j]] / pj + gsum / pk;
            }
            let (_, r) = self.row_logpdf_resp(i, y[i]);
            for k in 0..K {
                let (em, es) = self.guarded_entries(i, y[i], &r, k);
                out[[i, Self::mu_col(k)]] = g[[i, Self::mu_col(k)]] / em;
                out[[i, Self::ls_col(k)]] = g[[i, Self::ls_col(k)]] / es;
            }
        }
        out
    }
}

impl<const K: usize> Scorable<CRPScore> for NormalMixture<K> {
    /// Diagonal of the closed-form metric by default; `NGBOOST_MIX_CRPS_FULL=1`
    /// uses the full matrix.
    fn is_diagonal_metric(&self) -> bool {
        crps_diag_only()
    }

    /// Training path (diagonal CRPS metric) with a per-row step cap: the
    /// metric diagonal scales like π_k²σ_k, so as a component shrinks or loses
    /// weight its natural-gradient step grows like 1/(π_k²σ_k) and can
    /// collapse it (test NLL 6.8·10⁶ on one het_a seed, R35). Entries are
    /// raised just enough that |ñ| ≤ GUARD_LOG_SIGMA_STEP for log σ and the
    /// weight logits and |ñ| ≤ GUARD_LOG_SIGMA_STEP·σ_k for μ_k. Never below the
    /// unguarded entry; `NGBOOST_MIX_GUARD=0` disables (research).
    fn d_score_and_diagonal_metric(&self, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let g = Scorable::<CRPScore>::d_score(self, y);
        let mut d = Scorable::<CRPScore>::diagonal_metric(self);
        if guard_enabled() {
            for i in 0..y.len() {
                for c in 0..3 * K - 1 {
                    let cap = if c >= K - 1 && (c - (K - 1)) % 2 == 0 {
                        GUARD_LOG_SIGMA_STEP * self.sigma[[i, (c - (K - 1)) / 2]]
                    } else {
                        GUARD_LOG_SIGMA_STEP
                    };
                    d[[i, c]] = d[[i, c]].max(g[[i, c]].abs() / cap);
                }
            }
        }
        (g, d)
    }

    /// Diagonal of 2∫∇F∇Fᵀ dy: μ_k → π_k²/(√π σ_k); log σ_k → π_k² σ_k/(2√π);
    /// logit l_j → 2π_j² Σ_{k,m} π_k π_m ½[E(j,k) + E(j,m) − E(k,m)] with E
    /// the energy distance between components.
    fn diagonal_metric(&self) -> Array2<f64> {
        let n = self.weights.nrows();
        let mut d = Array2::zeros((n, 3 * K - 1));
        for i in 0..n {
            let p: Vec<f64> = (0..K).map(|k| self.weights[[i, k]]).collect();
            let pf: Vec<f64> = p.iter().map(|v| v.max(PI_FLOOR)).collect();
            let mut e = [[0.0f64; K]; K];
            for a in 0..K {
                for b in 0..K {
                    e[a][b] = self.energy(i, a, b);
                }
            }
            for k in 0..K {
                let s = self.sigma[[i, k]];
                d[[i, Self::mu_col(k)]] = (pf[k] * pf[k] * INV_SQRT_PI / s).max(1e-12);
                d[[i, Self::ls_col(k)]] = (pf[k] * pf[k] * s * 0.5 * INV_SQRT_PI).max(1e-12);
            }
            for j in 0..K - 1 {
                let mut acc = 0.0;
                for k in 0..K {
                    for m in 0..K {
                        acc += p[k] * p[m] * 0.5 * (e[j][k] + e[j][m] - e[k][m]);
                    }
                }
                d[[i, j]] = (2.0 * pf[j] * pf[j] * acc).max(1e-12);
            }
        }
        d
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| {
            let mut s = 0.0;
            for k in 0..K {
                let pk = self.weights[[i, k]];
                s += pk * a_fn(y[i] - self.mu[[i, k]], self.sigma[[i, k]]).0;
                for m in 0..K {
                    let (sk, sm) = (self.sigma[[i, k]], self.sigma[[i, m]]);
                    s -= 0.5
                        * pk
                        * self.weights[[i, m]]
                        * a_fn(
                            self.mu[[i, k]] - self.mu[[i, m]],
                            (sk * sk + sm * sm).sqrt(),
                        )
                        .0;
                }
            }
            s
        })
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n = y.len();
        let mut g = Array2::zeros((n, 3 * K - 1));
        for i in 0..n {
            let p: Vec<f64> = (0..K).map(|k| self.weights[[i, k]]).collect();
            let mut c = [0.0f64; K]; // ∂CRPS/∂π_k with π free
            for k in 0..K {
                let (mu_k, s_k) = (self.mu[[i, k]], self.sigma[[i, k]]);
                let (ay, dmy, dsy) = a_fn(y[i] - mu_k, s_k);
                c[k] = ay;
                let mut dmu = -p[k] * dmy;
                // the k = m term: ½·π_k²·∂A(0, √2σ_k)/∂σ_k = π_k²/√π
                let mut dsig = p[k] * dsy - p[k] * p[k] * INV_SQRT_PI;
                for m in 0..K {
                    let s_m = self.sigma[[i, m]];
                    let skm = (s_k * s_k + s_m * s_m).sqrt();
                    let (akm, dm, ds) = a_fn(mu_k - self.mu[[i, m]], skm);
                    c[k] -= p[m] * akm;
                    if m != k {
                        dmu -= p[k] * p[m] * dm;
                        dsig -= p[k] * p[m] * ds * s_k / skm;
                    }
                }
                g[[i, Self::mu_col(k)]] = dmu;
                g[[i, Self::ls_col(k)]] = dsig * s_k;
            }
            let cbar: f64 = (0..K).map(|k| p[k] * c[k]).sum();
            for j in 0..K - 1 {
                g[[i, j]] = p[j] * (c[j] - cbar);
            }
        }
        g
    }

    fn metric(&self) -> Array3<f64> {
        let n = self.weights.nrows();
        let np = 3 * K - 1;
        let mut m = Array3::zeros((n, np, np));
        if crps_diag_only() {
            let d = Scorable::<CRPScore>::diagonal_metric(self);
            for i in 0..n {
                for j in 0..np {
                    m[[i, j, j]] = d[[i, j]];
                }
            }
        } else {
            for i in 0..n {
                m.index_axis_mut(ndarray::Axis(0), i)
                    .assign(&self.crps_full_metric(i));
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{ContinuousCDF, Normal as NormalDist};

    fn one<const K: usize>(p: &[f64]) -> NormalMixture<K> {
        NormalMixture::<K>::from_params(&Array2::from_shape_vec((1, p.len()), p.to_vec()).unwrap())
    }
    const P2: [f64; 5] = [0.3, -0.7, -0.51, 2.5, 0.18];
    const P3: [f64; 8] = [0.4, -0.2, -2.0, -0.4, 0.5, 0.3, 4.0, 0.1];

    fn quad<F: FnMut(f64) -> f64>(lo: f64, hi: f64, k: usize, mut f: F) -> f64 {
        let h = (hi - lo) / k as f64;
        (0..k).map(|j| f(lo + (j as f64 + 0.5) * h)).sum::<f64>() * h
    }

    #[test]
    fn density_integrates_to_one_and_cdf_ppf_roundtrip() {
        let d2 = one::<2>(&P2);
        let d3 = one::<3>(&P3);
        for (name, tot) in [
            (
                "K2",
                quad(-20.0, 25.0, 90_000, |y| d2.pdf(&Array1::from_elem(1, y))[0]),
            ),
            (
                "K3",
                quad(-20.0, 25.0, 90_000, |y| d3.pdf(&Array1::from_elem(1, y))[0]),
            ),
        ] {
            assert!((tot - 1.0).abs() < 1e-8, "{name}: ∫f = {tot}");
        }
        for &q in &[0.01, 0.3, 0.5, 0.77, 0.99] {
            let x = d3.ppf(&Array1::from_elem(1, q))[0];
            assert!((d3.cdf(&Array1::from_elem(1, x))[0] - q).abs() < 1e-9);
        }
    }

    #[test]
    fn k2_layout_matches_legacy_two_component_parameterization() {
        let d = one::<2>(&P2);
        let pi = 1.0 / (1.0 + (-P2[0]).exp());
        assert!((d.weights[[0, 0]] - pi).abs() < 1e-15);
        assert_eq!(d.mu[[0, 0]], P2[1]);
        assert!((d.sigma[[0, 0]] - P2[2].exp()).abs() < 1e-15);
        assert_eq!(d.mu[[0, 1]], P2[3]);
        assert_eq!(NormalMixture2::scale_param_columns(), vec![2, 4]);
        assert_eq!(NormalMixture3::scale_param_columns(), vec![3, 5, 7]);
    }

    fn fd_check<const K: usize, S: crate::scores::Score>(p: &[f64], ys: &[f64])
    where
        NormalMixture<K>: Scorable<S>,
    {
        for &y in ys {
            let ya = Array1::from_elem(1, y);
            let g = Scorable::<S>::d_score(&one::<K>(p), &ya);
            for j in 0..p.len() {
                let eps = 1e-6;
                let (mut pp, mut pm) = (p.to_vec(), p.to_vec());
                pp[j] += eps;
                pm[j] -= eps;
                let fd = (Scorable::<S>::score(&one::<K>(&pp), &ya)[0]
                    - Scorable::<S>::score(&one::<K>(&pm), &ya)[0])
                    / (2.0 * eps);
                assert!(
                    (g[[0, j]] - fd).abs() < 2e-6 * (1.0 + fd.abs()),
                    "K={K} y={y} col {j}: analytic {} vs FD {fd}",
                    g[[0, j]]
                );
            }
        }
    }

    #[test]
    fn log_score_gradients_match_finite_differences() {
        fd_check::<2, LogScore>(&P2, &[-2.0, -0.5, 0.9, 2.4, 5.0]);
        fd_check::<3, LogScore>(&P3, &[-3.0, -0.1, 0.6, 2.2, 4.3]);
    }

    #[test]
    fn crps_gradients_match_finite_differences() {
        fd_check::<2, CRPScore>(&P2, &[-2.0, -0.5, 0.9, 2.4, 5.0]);
        fd_check::<3, CRPScore>(&P3, &[-3.0, -0.1, 0.6, 2.2, 4.3]);
    }

    #[test]
    fn crps_closed_form_matches_integral() {
        let d = one::<3>(&P3);
        for &y in &[-2.5, 0.0, 1.7, 4.9] {
            let ya = Array1::from_elem(1, y);
            let closed = Scorable::<CRPScore>::score(&d, &ya)[0];
            // split at y: the integrand jumps there
            let f = |t: f64| d.cdf(&Array1::from_elem(1, t))[0];
            let num = quad(-40.0, y, 200_000, |t| f(t).powi(2))
                + quad(y, 45.0, 200_000, |t| (f(t) - 1.0).powi(2));
            assert!((closed - num).abs() < 1e-7, "y={y}: {closed} vs {num}");
        }
    }

    /// Diagonal of the CRPS metric equals 2∫(∂F/∂θ_j)² dy (FD of the cdf).
    #[test]
    fn crps_metric_diagonal_matches_quadrature() {
        let p = P3;
        let d = one::<3>(&p);
        let m = Scorable::<CRPScore>::diagonal_metric(&d);
        for j in 0..p.len() {
            let eps = 1e-6;
            let (mut pp, mut pm) = (p.to_vec(), p.to_vec());
            pp[j] += eps;
            pm[j] -= eps;
            let (dp, dm) = (one::<3>(&pp), one::<3>(&pm));
            let val = 2.0
                * quad(-40.0, 45.0, 60_000, |t| {
                    let ta = Array1::from_elem(1, t);
                    ((dp.cdf(&ta)[0] - dm.cdf(&ta)[0]) / (2.0 * eps)).powi(2)
                });
            assert!(
                (m[[0, j]] - val).abs() < 1e-5 * (1.0 + val),
                "col {j}: {} vs {val}",
                m[[0, j]]
            );
        }
    }

    /// Complete-data Fisher dominates the observed Fisher (diagonals, and for
    /// K = 3 the full weight block via the Sherman–Morrison solve agreeing
    /// with a dense solve).
    #[test]
    fn complete_data_fisher_dominates_observed_and_solve_is_exact() {
        for (k, p) in [(2usize, P2.to_vec()), (3, P3.to_vec())] {
            let (diag, e) = if k == 2 {
                let d = one::<2>(&p);
                let mut e = vec![0.0; p.len()];
                quad(-30.0, 35.0, 60_000, |y| {
                    let ya = Array1::from_elem(1, y);
                    let f = d.pdf(&ya)[0];
                    let g = Scorable::<LogScore>::d_score(&d, &ya);
                    for c in 0..p.len() {
                        e[c] += f * g[[0, c]].powi(2);
                    }
                    0.0
                });
                (Scorable::<LogScore>::diagonal_metric(&d), e)
            } else {
                let d = one::<3>(&p);
                let mut e = vec![0.0; p.len()];
                quad(-30.0, 35.0, 60_000, |y| {
                    let ya = Array1::from_elem(1, y);
                    let f = d.pdf(&ya)[0];
                    let g = Scorable::<LogScore>::d_score(&d, &ya);
                    for c in 0..p.len() {
                        e[c] += f * g[[0, c]].powi(2);
                    }
                    0.0
                });
                (Scorable::<LogScore>::diagonal_metric(&d), e)
            };
            let h = 65.0 / 60_000.0;
            for c in 0..p.len() {
                assert!(
                    e[c] * h <= diag[[0, c]] * (1.0 + 1e-6) + 1e-12,
                    "K={k} col {c}"
                );
            }
        }
        // closed-form weight-block solve == generic dense solve on the same
        // (guarded) metric
        let d = one::<3>(&P3);
        let y = Array1::from_elem(1, 0.7);
        let fast = Scorable::<LogScore>::dense_natural_grad(&d, &y, 0.0);
        let mut met = Scorable::<LogScore>::metric(&d);
        let (_, r) = d.row_logpdf_resp(0, 0.7);
        for k in 0..3 {
            let (em, es) = d.guarded_entries(0, 0.7, &r, k);
            met[[0, NormalMixture3::mu_col(k), NormalMixture3::mu_col(k)]] = em;
            met[[0, NormalMixture3::ls_col(k), NormalMixture3::ls_col(k)]] = es;
        }
        let slow = crate::scores::natural_gradient_regularized(
            &Scorable::<LogScore>::d_score(&d, &y),
            &met,
            0.0,
        );
        for c in 0..8 {
            assert!(
                (fast[[0, c]] - slow[[0, c]]).abs() < 1e-9 * (1.0 + slow[[0, c]].abs()),
                "col {c}"
            );
        }
    }

    #[test]
    fn em_init_recovers_separated_mixtures() {
        let sn = NormalDist::new(0.0, 1.0).unwrap();
        let mut ys = Vec::new();
        for i in 0..3000 {
            let z = sn.inverse_cdf((i as f64 + 0.5) / 3000.0);
            ys.push(match i % 10 {
                0..=2 => 0.5 * z,
                3..=6 => 10.0 + z,
                _ => 20.0 + 2.0 * z,
            });
        }
        let p = NormalMixture3::fit(&Array1::from(ys));
        let d = NormalMixture3::from_params(&Array2::from_shape_vec((1, 8), p.to_vec()).unwrap());
        for (k, (w, m)) in [(0.3, 0.0), (0.4, 10.0), (0.3, 20.0)].iter().enumerate() {
            assert!(
                (d.weights[[0, k]] - w).abs() < 0.02,
                "w{k} {}",
                d.weights[[0, k]]
            );
            assert!((d.mu[[0, k]] - m).abs() < 0.15, "μ{k} {}", d.mu[[0, k]]);
        }
    }

    #[test]
    fn mode_is_the_highest_density_point() {
        let d = one::<2>(&[1.5, 0.0, (0.3f64).ln(), 6.0, (2.0f64).ln()]);
        let x = d.mode()[0];
        assert!(
            x.abs() < 0.05,
            "mode {x} should sit on the tall narrow component"
        );
        let fx = d.pdf(&Array1::from_elem(1, x))[0];
        for t in [-1.0, -0.2, 0.2, 1.0, 5.0, 6.0, 7.0] {
            assert!(d.pdf(&Array1::from_elem(1, t))[0] <= fx + 1e-12);
        }
    }

    /// Every entry of the full CRPS metric equals 2∫(∂F/∂θ_a)(∂F/∂θ_b) dy.
    #[test]
    fn crps_full_metric_matches_quadrature() {
        for p in [P2.to_vec(), P3.to_vec()] {
            let np = p.len();
            let cdf_at = |q: &[f64], t: f64| -> f64 {
                if np == 5 {
                    one::<2>(q).cdf(&Array1::from_elem(1, t))[0]
                } else {
                    one::<3>(q).cdf(&Array1::from_elem(1, t))[0]
                }
            };
            let full = if np == 5 {
                one::<2>(&p).crps_full_metric(0)
            } else {
                one::<3>(&p).crps_full_metric(0)
            };
            let eps = 1e-6;
            let (lo, hi, k) = (-40.0, 45.0, 30_000);
            let h = (hi - lo) / k as f64;
            let mut dfs = vec![vec![0.0f64; k]; np];
            for a in 0..np {
                let (mut pp, mut pm) = (p.clone(), p.clone());
                pp[a] += eps;
                pm[a] -= eps;
                for (t_i, slot) in dfs[a].iter_mut().enumerate() {
                    let t = lo + (t_i as f64 + 0.5) * h;
                    *slot = (cdf_at(&pp, t) - cdf_at(&pm, t)) / (2.0 * eps);
                }
            }
            for a in 0..np {
                for b in 0..np {
                    let q: f64 =
                        2.0 * h * dfs[a].iter().zip(&dfs[b]).map(|(u, v)| u * v).sum::<f64>();
                    assert!(
                        (full[[a, b]] - q).abs() < 2e-5 * (1.0 + q.abs()),
                        "np={np} ({a},{b}): closed {} vs quadrature {q}",
                        full[[a, b]]
                    );
                }
            }
        }
    }

    /// The safeguard caps every row's log σ step at GUARD_LOG_SIGMA_STEP,
    /// never lowers the metric below the expected complete-data Fisher, and
    /// leaves rows near a component untouched.
    #[test]
    fn guard_caps_log_sigma_steps_and_is_inactive_near_components() {
        let d = one::<2>(&[0.0, 0.0, (0.7f64).ln(), 2.0, (0.7f64).ln()]);
        let e = Scorable::<LogScore>::diagonal_metric(&d);
        for &y in &[-3.0, -0.4, 0.1, 1.0, 2.2, 5.0, 16.0, 60.0] {
            let ya = Array1::from_elem(1, y);
            let (g, m) = Scorable::<LogScore>::d_score_and_diagonal_metric(&d, &ya);
            for c in 0..5 {
                assert!(
                    m[[0, c]] >= e[[0, c]] - 1e-15,
                    "y={y} col {c}: guarded below expected"
                );
            }
            for k in 0..2 {
                let c = NormalMixture2::ls_col(k);
                assert!(
                    (g[[0, c]] / m[[0, c]]).abs() <= GUARD_LOG_SIGMA_STEP + 1e-9,
                    "y={y}"
                );
                let cm = NormalMixture2::mu_col(k);
                assert_eq!(m[[0, cm]], e[[0, cm]], "μ entries are never guarded");
            }
        }
        // a row sitting on component 0 is unaffected
        let (_, m) =
            Scorable::<LogScore>::d_score_and_diagonal_metric(&d, &Array1::from_elem(1, 0.1));
        assert_eq!(m.row(0), e.row(0));
    }

    /// The CRPS training path caps every row's step (log σ / logits at
    /// GUARD_LOG_SIGMA_STEP, μ_k at GUARD_LOG_SIGMA_STEP·σ_k) and never lowers
    /// the metric below the unguarded diagonal.
    #[test]
    fn crps_guard_caps_steps() {
        // a nearly unused, very narrow component: the unguarded step explodes
        let d = one::<2>(&[6.0, 0.0, (0.01f64).ln(), 3.0, 0.0]);
        let e = Scorable::<CRPScore>::diagonal_metric(&d);
        for &y in &[-2.0, 0.0, 0.3, 3.0, 9.0] {
            let ya = Array1::from_elem(1, y);
            let (g, m) = Scorable::<CRPScore>::d_score_and_diagonal_metric(&d, &ya);
            for c in 0..5 {
                assert!(m[[0, c]] >= e[[0, c]] - 1e-15);
                let cap = if c == 1 || c == 3 {
                    GUARD_LOG_SIGMA_STEP * d.sigma[[0, (c - 1) / 2]]
                } else {
                    GUARD_LOG_SIGMA_STEP
                };
                assert!(
                    (g[[0, c]] / m[[0, c]]).abs() <= cap * (1.0 + 1e-9),
                    "y={y} col {c}"
                );
            }
        }
    }
}
