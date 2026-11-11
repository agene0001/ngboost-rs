//! Point mass at zero plus a continuous head (hurdle / zero-inflated).
//!
//! `Hurdle<D>`: Y = 0 with probability π, otherwise Y ~ D. For a continuous
//! D the hurdle and zero-inflated readings coincide (D puts no mass on 0).
//! Typical targets: intermittent demand, rainfall, insurance claims, costs,
//! scores with a floor — anything with *exact* repeated zeros next to a
//! continuous spread. A continuous head scores such data badly, and a
//! Normal-mixture component collapses onto the atom (σ → 0, NaN), see
//! `docs/known_limitations.md`. Values at another location c can be
//! shifted by −c first.
//!
//! Parameters: [logit π, D's internal parameters…]. The log score is taken
//! w.r.t. counting measure at 0 plus Lebesgue measure elsewhere:
//!   y = 0: −ln π          y ≠ 0: −ln(1 − π) − ln f_D(y)
//! so it is comparable between hurdle heads only; compare against
//! continuous heads by CRPS.
//!
//! Fisher information (expected over Y) is block diagonal:
//!   logit: π(1 − π)        D block: (1 − π) · I_D
//! so the natural gradient is the Bernoulli step for the logit and, on
//! non-zero rows, D's own natural gradient divided by (1 − π) (0 on zero
//! rows) — unbiased for D's natural gradient in expectation over the atom.
//! D's own metric, guards and fused fast paths are reused unchanged.
use crate::dist::{Distribution, DistributionMethods, LogNormal, Normal, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable, natural_gradient_regularized};
use ndarray::{Array1, Array2, Array3, s};
use rand::RngExt;

/// Floor on π and 1 − π inside Fisher entries (as Categorical's
/// natural-gradient probability floor): keeps steps finite when the
/// predicted atom probability saturates.
const P_FLOOR: f64 = 1e-6;

/// Point mass at 0 (probability `pi`) plus the continuous law `inner`.
#[derive(Debug, Clone)]
pub struct Hurdle<D> {
    /// P(Y = 0).
    pub pi: Array1<f64>,
    /// Raw parameter logit π.
    pub logit: Array1<f64>,
    /// Law of Y given Y ≠ 0.
    pub inner: D,
}

/// Zero-inflated Normal (atom inside the support).
pub type HurdleNormal = Hurdle<Normal>;
/// Hurdle LogNormal: zeros plus a positive, right-skewed part.
pub type HurdleLogNormal = Hurdle<LogNormal>;

#[inline]
fn is_atom(y: f64) -> bool {
    y == 0.0
}

/// ln(1 + eˣ) without overflow.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

impl<D: Distribution> Hurdle<D> {
    /// y with the atom rows replaced by a point inside D's support (D's
    /// point prediction), so D's score/gradient are finite there; those rows
    /// are masked out afterwards.
    fn safe_y(&self, y: &Array1<f64>) -> Array1<f64> {
        if !y.iter().any(|&v| is_atom(v)) {
            return y.clone();
        }
        let p = self.inner.predict();
        Array1::from_shape_fn(y.len(), |i| {
            if is_atom(y[i]) {
                if p[i].is_finite() { p[i] } else { 1.0 }
            } else {
                y[i]
            }
        })
    }

    fn logit_fisher(&self, i: usize) -> f64 {
        let p = self.pi[i].clamp(P_FLOOR, 1.0 - P_FLOOR);
        p * (1.0 - p)
    }

    fn cont_weight(&self, i: usize) -> f64 {
        (1.0 - self.pi[i]).max(P_FLOOR)
    }
}

impl<D: Distribution> Distribution for Hurdle<D> {
    fn from_params(params: &Array2<f64>) -> Self {
        let logit = params.column(0).to_owned();
        let pi = logit.mapv(|l| 1.0 / (1.0 + (-l).exp()));
        let inner = D::from_params(&params.slice(s![.., 1..]).to_owned());
        Hurdle { pi, logit, inner }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let n = y.len();
        let pos: Array1<f64> = y.iter().copied().filter(|&v| !is_atom(v)).collect();
        let n0 = (n - pos.len()) as f64;
        // smoothed so an all-zero or zero-free sample starts finite
        let p0 = (n0 + 0.5) / (n as f64 + 1.0);
        let inner = if pos.is_empty() {
            D::fit(&Array1::from_elem(1, 1.0))
        } else {
            D::fit(&pos)
        };
        let mut out = Array1::zeros(1 + inner.len());
        out[0] = (p0 / (1.0 - p0)).ln();
        out.slice_mut(s![1..]).assign(&inner);
        out
    }

    fn validate_targets(y: &Array1<f64>) -> Result<(), &'static str> {
        let pos: Array1<f64> = y.iter().copied().filter(|&v| !is_atom(v)).collect();
        if pos.is_empty() {
            return Ok(());
        }
        D::validate_targets(&pos)
    }

    fn n_params(&self) -> usize {
        1 + self.inner.n_params()
    }

    fn scale_param_index() -> Option<usize> {
        D::scale_param_index().map(|c| c + 1)
    }

    fn scale_param_columns() -> Vec<usize> {
        D::scale_param_columns()
            .into_iter()
            .map(|c| c + 1)
            .collect()
    }

    /// The mean (1 − π)·E_D[Y].
    fn predict(&self) -> Array1<f64> {
        let m = self.inner.predict();
        Array1::from_shape_fn(m.len(), |i| (1.0 - self.pi[i]) * m[i])
    }

    fn params(&self) -> Array2<f64> {
        let inner = self.inner.params();
        let n = inner.nrows();
        let mut p = Array2::zeros((n, 1 + inner.ncols()));
        p.column_mut(0).assign(&self.logit);
        p.slice_mut(s![.., 1..]).assign(&inner);
        p
    }
}

impl<D: Distribution> RegressionDistn for Hurdle<D> {}

impl<D> Scorable<LogScore> for Hurdle<D>
where
    D: Distribution + Scorable<LogScore>,
{
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let ys = self.safe_y(y);
        let inner = Scorable::<LogScore>::score(&self.inner, &ys);
        Array1::from_shape_fn(y.len(), |i| {
            if is_atom(y[i]) {
                softplus(-self.logit[i])
            } else {
                softplus(self.logit[i]) + inner[i]
            }
        })
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let ys = self.safe_y(y);
        let g = Scorable::<LogScore>::d_score(&self.inner, &ys);
        let mut out = Array2::zeros((y.len(), 1 + g.ncols()));
        for i in 0..y.len() {
            let z = is_atom(y[i]);
            out[[i, 0]] = self.pi[i] - if z { 1.0 } else { 0.0 };
            if !z {
                out.slice_mut(s![i, 1..]).assign(&g.row(i));
            }
        }
        out
    }

    fn metric(&self) -> Array3<f64> {
        let m = Scorable::<LogScore>::metric(&self.inner);
        let (n, p) = (m.shape()[0], m.shape()[1]);
        let mut out = Array3::zeros((n, p + 1, p + 1));
        for i in 0..n {
            out[[i, 0, 0]] = self.logit_fisher(i);
            let w = self.cont_weight(i);
            out.slice_mut(s![i, 1.., 1..])
                .assign(&m.slice(s![i, .., ..]).mapv(|v| v * w));
        }
        out
    }

    fn is_diagonal_metric(&self) -> bool {
        Scorable::<LogScore>::is_diagonal_metric(&self.inner)
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        let d = Scorable::<LogScore>::diagonal_metric(&self.inner);
        self.stack_diag(&d)
    }

    fn d_score_and_diagonal_metric(&self, y: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let ys = self.safe_y(y);
        let (g, d) = Scorable::<LogScore>::d_score_and_diagonal_metric(&self.inner, &ys);
        let mut grad = Array2::zeros((y.len(), 1 + g.ncols()));
        for i in 0..y.len() {
            let z = is_atom(y[i]);
            grad[[i, 0]] = self.pi[i] - if z { 1.0 } else { 0.0 };
            if !z {
                grad.slice_mut(s![i, 1..]).assign(&g.row(i));
            }
        }
        (grad, self.stack_diag(&d))
    }

    fn dense_natural_grad(&self, y: &Array1<f64>, reg: f64) -> Array2<f64> {
        if reg > 0.0 {
            // (1 − π)·I_D + reg·I is not a rescaled inner solve; generic path
            return natural_gradient_regularized(
                &Scorable::<LogScore>::d_score(self, y),
                &Scorable::<LogScore>::metric(self),
                reg,
            );
        }
        let ys = self.safe_y(y);
        let ng = Scorable::<LogScore>::dense_natural_grad(&self.inner, &ys, 0.0);
        let mut out = Array2::zeros((y.len(), 1 + ng.ncols()));
        for i in 0..y.len() {
            let z = is_atom(y[i]);
            out[[i, 0]] = (self.pi[i] - if z { 1.0 } else { 0.0 }) / self.logit_fisher(i);
            if !z {
                let w = self.cont_weight(i);
                out.slice_mut(s![i, 1..]).assign(&ng.row(i).mapv(|v| v / w));
            }
        }
        out
    }
}

impl<D> Hurdle<D> {
    fn stack_diag(&self, d: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::zeros((d.nrows(), 1 + d.ncols()));
        for i in 0..d.nrows() {
            let p = self.pi[i].clamp(P_FLOOR, 1.0 - P_FLOOR);
            out[[i, 0]] = p * (1.0 - p);
            let w = (1.0 - self.pi[i]).max(P_FLOOR);
            out.slice_mut(s![i, 1..]).assign(&d.row(i).mapv(|v| v * w));
        }
        out
    }
}

impl<D: DistributionMethods> DistributionMethods for Hurdle<D> {
    fn mean(&self) -> Array1<f64> {
        let m = self.inner.mean();
        Array1::from_shape_fn(m.len(), |i| (1.0 - self.pi[i]) * m[i])
    }

    /// (1 − π)(Var_D + E_D²) − mean².
    fn variance(&self) -> Array1<f64> {
        let (m, v) = (self.inner.mean(), self.inner.variance());
        Array1::from_shape_fn(m.len(), |i| {
            let q = 1.0 - self.pi[i];
            (q * (v[i] + m[i] * m[i]) - (q * m[i]).powi(2)).max(0.0)
        })
    }

    /// Mixed-measure density: π at 0, (1 − π)·f_D(y) elsewhere.
    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let f = self.inner.pdf(y);
        Array1::from_shape_fn(y.len(), |i| {
            if is_atom(y[i]) {
                self.pi[i]
            } else {
                (1.0 - self.pi[i]) * f[i]
            }
        })
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let ys = self.safe_y(y);
        let f = self.inner.logpdf(&ys);
        Array1::from_shape_fn(y.len(), |i| {
            if is_atom(y[i]) {
                -softplus(-self.logit[i])
            } else {
                -softplus(self.logit[i]) + f[i]
            }
        })
    }

    /// (1 − π)·F_D(y) + π·1{y ≥ 0}.
    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        let f = self.inner.cdf(y);
        Array1::from_shape_fn(y.len(), |i| {
            (1.0 - self.pi[i]) * f[i] + if y[i] >= 0.0 { self.pi[i] } else { 0.0 }
        })
    }

    /// Generalised inverse: quantiles inside the atom's probability band
    /// return 0; the rest map to D's quantile of the renormalised level.
    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        let n = q.len();
        let below = self.inner.cdf(&Array1::zeros(n)); // F_D(0)
        let mut u = Array1::from_elem(n, 0.5);
        let mut at_atom = vec![false; n];
        for i in 0..n {
            let (p, c) = (self.pi[i], 1.0 - self.pi[i]);
            let lo = c * below[i]; // P(Y < 0)
            if q[i] < lo {
                u[i] = q[i] / c;
            } else if q[i] <= lo + p {
                at_atom[i] = true;
            } else {
                u[i] = ((q[i] - p) / c).clamp(0.0, 1.0);
            }
        }
        let x = self.inner.ppf(&u);
        Array1::from_shape_fn(n, |i| if at_atom[i] { 0.0 } else { x[i] })
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let mut out = self.inner.sample(n_samples);
        let mut rng = rand::rng();
        for s in 0..n_samples {
            for i in 0..self.pi.len() {
                if rng.random_range(0.0..1.0) < self.pi[i] {
                    out[[s, i]] = 0.0;
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------- CRPS

const INV_SQRT_PI: f64 = 0.5641895835477563;
const INV_SQRT_2PI: f64 = 0.3989422804014327;

#[inline]
fn std_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

#[inline]
fn std_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// y-scale CRPS pieces of a two-parameter continuous head, for the CRPS of
/// `Hurdle<D>` (research/ngboost Cycle 14). For row `i` and a target `y`:
/// C(y) = CRPS_D(y) and ∂C/∂θ; G(y) = ∫₀^y F_D (y > 0) or ∫_y^0 (1 − F_D)
/// (y < 0), 0 at y = 0, and ∂G/∂θ; and D's CRPS metric 2∫∇F∇Fᵀ dt on the
/// y scale. θ are D's raw parameters.
pub trait HurdleCrpsPart: Distribution {
    fn crps_parts(&self, i: usize, y: f64) -> (f64, [f64; 2], f64, [f64; 2]);
    fn crps_metric_y(&self, i: usize) -> [[f64; 2]; 2];
}

impl HurdleCrpsPart for Normal {
    fn crps_parts(&self, i: usize, y: f64) -> (f64, [f64; 2], f64, [f64; 2]) {
        let (mu, s) = (self.loc[i], self.scale[i]);
        let z = (y - mu) / s;
        let (pz, cz) = (std_pdf(z), std_cdf(z));
        let c = s * (z * (2.0 * cz - 1.0) + 2.0 * pz - INV_SQRT_PI);
        let dc = [-(2.0 * cz - 1.0), s * (2.0 * pz - INV_SQRT_PI)];
        // H(t) = (t − μ)Φ(u) + σφ(u) is an antiderivative of F
        let z0 = -mu / s;
        let (p0, c0) = (std_pdf(z0), std_cdf(z0));
        let h = |zz: f64, pp: f64, cc: f64| s * zz * cc + s * pp;
        let g = h(z, pz, cz) - h(z0, p0, c0) - y.min(0.0);
        let dg = [-cz + c0, s * (pz - p0)];
        (c, dc, g, dg)
    }

    fn crps_metric_y(&self, i: usize) -> [[f64; 2]; 2] {
        let s = self.scale[i];
        [[INV_SQRT_PI / s, 0.0], [0.0, 0.5 * s * INV_SQRT_PI]]
    }
}

/// LogNormal on the **y** scale (the crate's `CRPScore` for LogNormal is the
/// CRPS of ln y, which cannot hold an atom at 0). Baran & Lerch (2015):
/// CRPS(y) = y(2Φ(z) − 1) − 2e^{μ+σ²/2}[Φ(z − σ) + Φ(σ/√2) − 1].
impl HurdleCrpsPart for LogNormal {
    fn crps_parts(&self, i: usize, y: f64) -> (f64, [f64; 2], f64, [f64; 2]) {
        let (mu, s) = (self.loc[i], self.scale[i]);
        let e = (mu + 0.5 * s * s).exp();
        let r = std::f64::consts::FRAC_1_SQRT_2 * s;
        let k = std_cdf(r) - 1.0;
        let (yp, z_c, z_p, zs_c) = if y > 0.0 {
            let z = (y.ln() - mu) / s;
            (y, std_cdf(z), std_pdf(z), std_cdf(z - s))
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };
        let c = yp * (2.0 * z_c - 1.0) - 2.0 * e * (zs_c + k);
        let dc_mu = -2.0 * e * (zs_c + k);
        let dc_s = 2.0 * yp * z_p
            - 2.0 * s * e * (zs_c + k)
            - 2.0 * e * std_pdf(r) * std::f64::consts::FRAC_1_SQRT_2;
        let g = yp * z_c - e * zs_c;
        let dg = [-e * zs_c, s * (yp * z_p - s * e * zs_c)];
        (c, [dc_mu, s * dc_s], g, dg)
    }

    fn crps_metric_y(&self, i: usize) -> [[f64; 2]; 2] {
        let (mu, s) = (self.loc[i], self.scale[i]);
        let e = (mu + 0.25 * s * s).exp() * INV_SQRT_PI;
        let cross = e * s * 0.5;
        [[e / s, cross], [cross, s * e * (0.5 + 0.25 * s * s)]]
    }
}

/// CRPS of the hurdle law F_H = (1 − π)F_D + π·1{t ≥ 0}:
///   CRPS_H(y) = (1 − π)² C(y) + π²|y| + 2π(1 − π) G(y)
/// (C, G from [`HurdleCrpsPart`]). Metric 2∫∇F_H∇F_Hᵀ dt in (logit π, θ):
///   ℓℓ: 2π²(1 − π)² C(0),  ℓθ: −π(1 − π)² ∂C(0)/∂θ,  θθ: (1 − π)² M_D,
/// solved per row (3×3). research/ngboost Cycle 14.
impl<D> Scorable<CRPScore> for Hurdle<D>
where
    D: HurdleCrpsPart,
{
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        Array1::from_shape_fn(y.len(), |i| {
            let p = self.pi[i];
            let (c, _, g, _) = self.inner.crps_parts(i, y[i]);
            (1.0 - p).powi(2) * c + p * p * y[i].abs() + 2.0 * p * (1.0 - p) * g
        })
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let mut out = Array2::zeros((y.len(), 3));
        for i in 0..y.len() {
            let p = self.pi[i];
            let q = 1.0 - p;
            let (c, dc, g, dg) = self.inner.crps_parts(i, y[i]);
            let d_pi = -2.0 * q * c + 2.0 * p * y[i].abs() + 2.0 * (1.0 - 2.0 * p) * g;
            out[[i, 0]] = p * q * d_pi;
            for j in 0..2 {
                out[[i, 1 + j]] = q * q * dc[j] + 2.0 * p * q * dg[j];
            }
        }
        out
    }

    fn metric(&self) -> Array3<f64> {
        let n = self.pi.len();
        let mut m = Array3::zeros((n, 3, 3));
        for i in 0..n {
            let p = self.pi[i].clamp(P_FLOOR, 1.0 - P_FLOOR);
            let q = 1.0 - p;
            let (c0, dc0, _, _) = self.inner.crps_parts(i, 0.0);
            let md = self.inner.crps_metric_y(i);
            m[[i, 0, 0]] = 2.0 * p * p * q * q * c0;
            for j in 0..2 {
                let v = -p * q * q * dc0[j];
                m[[i, 0, 1 + j]] = v;
                m[[i, 1 + j, 0]] = v;
                for k in 0..2 {
                    m[[i, 1 + j, 1 + k]] = q * q * md[j][k];
                }
            }
        }
        m
    }

    fn dense_natural_grad(&self, y: &Array1<f64>, reg: f64) -> Array2<f64> {
        natural_gradient_regularized(
            &Scorable::<CRPScore>::d_score(self, y),
            &Scorable::<CRPScore>::metric(self),
            reg,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn hn() -> HurdleNormal {
        // rows: π = 0.3, 0.7; Normal(1.5, 0.8), Normal(−0.4, 2.0)
        let l = |p: f64| (p / (1.0 - p)).ln();
        HurdleNormal::from_params(&array![
            [l(0.3), 1.5, 0.8f64.ln()],
            [l(0.7), -0.4, 2.0f64.ln()]
        ])
    }

    #[test]
    fn score_matches_mixed_density() {
        let d = hn();
        let y = array![0.0, 1.1];
        let s = Scorable::<LogScore>::score(&d, &y);
        assert!((s[0] - (-(0.3f64).ln())).abs() < 1e-12);
        let f =
            (-0.5 * ((1.1f64 + 0.4) / 2.0).powi(2)).exp() / (2.0 * std::f64::consts::TAU.sqrt());
        assert!((s[1] - (-(0.3 * f).ln())).abs() < 1e-12);
        assert!((d.logpdf(&y) + &s).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let d = hn();
        for y in [array![0.0, 0.0], array![2.0, -1.3], array![0.0, 3.1]] {
            let g = Scorable::<LogScore>::d_score(&d, &y);
            let p = d.params();
            for j in 0..3 {
                let h = 1e-6;
                let (mut a, mut b) = (p.clone(), p.clone());
                a.column_mut(j).mapv_inplace(|v| v + h);
                b.column_mut(j).mapv_inplace(|v| v - h);
                let fd = (Scorable::<LogScore>::score(&HurdleNormal::from_params(&a), &y)
                    - Scorable::<LogScore>::score(&HurdleNormal::from_params(&b), &y))
                    / (2.0 * h);
                for i in 0..2 {
                    assert!((fd[i] - g[[i, j]]).abs() < 1e-6, "y={y} j={j} i={i}");
                }
            }
        }
    }

    /// Fisher = E[∇∇ᵀ] over the mixed law (atom + Gauss–Hermite-free check
    /// by exhaustive quadrature on a fine grid).
    #[test]
    fn metric_is_expected_outer_product() {
        let d = hn();
        let m = Scorable::<LogScore>::metric(&d);
        for i in 0..2 {
            let one = HurdleNormal::from_params(&d.params().slice(s![i..i + 1, ..]).to_owned());
            let mut acc = Array2::<f64>::zeros((3, 3));
            // atom
            let g0 = Scorable::<LogScore>::d_score(&one, &array![0.0]);
            acc += &(g0.t().dot(&g0) * one.pi[0]);
            // continuous part: midpoint rule on ±12σ
            let (mu, sd) = (one.inner.loc[0], one.inner.scale[0]);
            let k = 40_000;
            let h = 24.0 * sd / k as f64;
            for t in 0..k {
                let yv = mu - 12.0 * sd + (t as f64 + 0.5) * h;
                let dens = (1.0 - one.pi[0]) * one.inner.pdf(&array![yv])[0];
                let g = Scorable::<LogScore>::d_score(&one, &array![yv]);
                acc += &(g.t().dot(&g) * dens * h);
            }
            for a in 0..3 {
                for b in 0..3 {
                    assert!(
                        (acc[[a, b]] - m[[i, a, b]]).abs() < 1e-5,
                        "row {i} ({a},{b}): {} vs {}",
                        acc[[a, b]],
                        m[[i, a, b]]
                    );
                }
            }
        }
    }

    #[test]
    fn natural_gradient_paths_agree() {
        let d = hn();
        let y = array![0.0, 0.9];
        let dense = Scorable::<LogScore>::dense_natural_grad(&d, &y, 0.0);
        let diag = Scorable::<LogScore>::diagonal_natural_grad(&d, &y, 0.0);
        let generic = natural_gradient_regularized(
            &Scorable::<LogScore>::d_score(&d, &y),
            &Scorable::<LogScore>::metric(&d),
            0.0,
        );
        for (a, b) in dense.iter().zip(generic.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
        for (a, b) in diag.iter().zip(generic.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn cdf_ppf_with_atom() {
        let d = hn();
        let c = d.cdf(&array![0.0, 0.0]);
        // row 0: P(Y ≤ 0) = 0.3 + 0.7·Φ(−1.5/0.8)
        let below = {
            use statrs::distribution::{ContinuousCDF, Normal as SN};
            0.7 * SN::new(0.0, 1.0).unwrap().cdf(-1.5 / 0.8)
        };
        assert!((c[0] - (0.3 + below)).abs() < 1e-12);
        // quantile inside the atom band → 0; above it → continuous
        let q = d.ppf(&array![below + 0.15, 0.95]);
        assert_eq!(q[0], 0.0);
        assert!(q[1] > 0.0);
        let back = d.cdf(&array![1.0, q[1]]);
        assert!((back[1] - 0.95).abs() < 1e-9);
        // fit: smoothed atom share, continuous part from the non-zeros
        let init = HurdleLogNormal::fit(&array![0.0, 0.0, 1.0, std::f64::consts::E]);
        assert!((init[0] - (2.5f64 / 2.5).ln()).abs() < 1e-12);
        assert!((init[1] - 0.5).abs() < 1e-12);
        assert!(HurdleLogNormal::validate_targets(&array![0.0, 2.0]).is_ok());
        assert!(HurdleLogNormal::validate_targets(&array![0.0, -2.0]).is_err());
    }

    /// CRPS of the hurdle = ∫(F_H − 1{t ≥ y})², its gradient = finite
    /// differences, its metric = 2∫∇F_H∇F_Hᵀ (all by quadrature, split at
    /// the atom and at y), for the Normal and LogNormal continuous parts.
    fn crps_checks<D>(p: [f64; 3], ys: &[f64], lo: f64, hi: f64)
    where
        D: HurdleCrpsPart + DistributionMethods,
    {
        let row = ndarray::Array2::from_shape_vec((1, 3), p.to_vec()).unwrap();
        let n = 200_000;
        let grid = |a: f64, b: f64| -> (Array1<f64>, f64) {
            let h = (b - a) / n as f64;
            ((0..n).map(|k| a + (k as f64 + 0.5) * h).collect(), h)
        };
        let wide = |r: &ndarray::Array2<f64>, m: usize| {
            ndarray::Array2::from_shape_fn((m, 3), |(_, j)| r[[0, j]])
        };
        let d = Hurdle::<D>::from_params(&row);
        for &y in ys {
            let mut cuts = vec![lo, 0.0f64.max(lo), y, hi];
            cuts.sort_by(|a, b| a.total_cmp(b));
            let mut num = 0.0;
            for w in cuts.windows(2) {
                if w[1] <= w[0] {
                    continue;
                }
                let (t, h) = grid(w[0], w[1]);
                let f = Hurdle::<D>::from_params(&wide(&row, t.len())).cdf(&t);
                num += t
                    .iter()
                    .zip(f.iter())
                    .map(|(&tt, &ff)| (ff - if tt >= y { 1.0 } else { 0.0 }).powi(2))
                    .sum::<f64>()
                    * h;
            }
            let s = Scorable::<CRPScore>::score(&d, &array![y])[0];
            assert!((s - num).abs() < 1e-5 * num.max(1.0), "y={y}: {s} vs {num}");
            let g = Scorable::<CRPScore>::d_score(&d, &array![y]);
            for j in 0..3 {
                let (mut a, mut b) = (row.clone(), row.clone());
                a[[0, j]] += 1e-6;
                b[[0, j]] -= 1e-6;
                let fd = (Scorable::<CRPScore>::score(&Hurdle::<D>::from_params(&a), &array![y])
                    [0]
                    - Scorable::<CRPScore>::score(&Hurdle::<D>::from_params(&b), &array![y])[0])
                    / 2e-6;
                assert!(
                    (g[[0, j]] - fd).abs() < 1e-6 * fd.abs().max(1.0),
                    "y={y} j={j}: {} vs {fd}",
                    g[[0, j]]
                );
            }
        }
        // metric
        let m = Scorable::<CRPScore>::metric(&d);
        let mut q = [[0.0f64; 3]; 3];
        for w in [[lo, 0.0f64.max(lo)], [0.0f64.max(lo), hi]] {
            if w[1] <= w[0] {
                continue;
            }
            let (t, h) = grid(w[0], w[1]);
            let gr: Vec<Array1<f64>> = (0..3)
                .map(|j| {
                    let (mut a, mut b) = (row.clone(), row.clone());
                    a[[0, j]] += 1e-6;
                    b[[0, j]] -= 1e-6;
                    (Hurdle::<D>::from_params(&wide(&a, t.len())).cdf(&t)
                        - Hurdle::<D>::from_params(&wide(&b, t.len())).cdf(&t))
                        / 2e-6
                })
                .collect();
            for a in 0..3 {
                for b in 0..3 {
                    q[a][b] += 2.0 * (&gr[a] * &gr[b]).sum() * h;
                }
            }
        }
        for a in 0..3 {
            for b in 0..3 {
                let sc = (q[a][a] * q[b][b]).sqrt();
                assert!(
                    (m[[0, a, b]] - q[a][b]).abs() < 1e-5 * sc,
                    "metric ({a},{b}): {} vs {}",
                    m[[0, a, b]],
                    q[a][b]
                );
            }
        }
    }

    #[test]
    fn hurdle_crps_normal_matches_definition() {
        let l = |p: f64| (p / (1.0 - p)).ln();
        crps_checks::<Normal>(
            [l(0.3), 1.5, 0.8f64.ln()],
            &[-1.0, 0.0, 0.7, 3.0],
            -9.0,
            11.0,
        );
        crps_checks::<Normal>([l(0.7), -0.4, 2.0f64.ln()], &[-2.5, 0.0, 1.1], -20.0, 20.0);
    }

    #[test]
    fn hurdle_crps_lognormal_matches_definition() {
        let l = |p: f64| (p / (1.0 - p)).ln();
        crps_checks::<LogNormal>(
            [l(0.4), 0.5, 0.6f64.ln()],
            &[0.0, 0.3, 2.0, 9.0],
            0.0,
            120.0,
        );
        crps_checks::<LogNormal>([l(0.1), -0.3, 0.3f64.ln()], &[0.0, 0.8], 0.0, 30.0);
    }
}
