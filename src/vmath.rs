//! Vectorized transcendental helpers.
//!
//! Dispatch priority for exp over contiguous slices:
//! 1. macOS + `accelerate`: vForce `vvexp` (~2× faster than Apple libm at
//!    n≥500; within 1 ulp, identical edge cases — examples/bench_vforce.rs).
//! 2. `simd-math` feature (any platform): the `wide` crate's `f64x4::exp`
//!    (within 1 ulp in the normal range; exp(x) for x ≲ −745.2 flushes to 0.0
//!    instead of a denormal — examples/bench_wide.rs). OPT-IN because the win
//!    is platform-dependent: ~2.5× SLOWER than libm on Apple Silicon, expected
//!    faster on x86-64/AVX2 (Windows/Linux). Measure with bench_wide first.
//! 3. Otherwise: scalar libm.

use ndarray::Array1;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
mod accel {
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn vvexp(y: *mut f64, x: *const f64, n: *const i32);
    }

    pub fn exp_slice(x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), out.len());
        let n = x.len() as i32;
        unsafe { vvexp(out.as_mut_ptr(), x.as_ptr(), &n) }
    }
}

#[cfg(feature = "simd-math")]
mod wide_math {
    use wide::f64x4;

    /// SIMD exp over a slice via 4-lane chunks, scalar tail.
    pub fn exp_slice(x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), out.len());
        let chunks = x.len() / 4 * 4;
        for i in (0..chunks).step_by(4) {
            let v = f64x4::from([x[i], x[i + 1], x[i + 2], x[i + 3]]);
            let e: [f64; 4] = v.exp().into();
            out[i..i + 4].copy_from_slice(&e);
        }
        for i in chunks..x.len() {
            out[i] = x[i].exp();
        }
    }
}

/// Run the fastest available vectorized exp on a contiguous slice.
/// Returns false when no vectorized backend is compiled in (caller should
/// fall back to scalar libm).
#[inline]
#[allow(unused_variables)]
fn exp_slice_fast(x: &[f64], out: &mut [f64]) -> bool {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        accel::exp_slice(x, out);
        return true;
    }
    #[cfg(all(
        feature = "simd-math",
        not(all(feature = "accelerate", target_os = "macos"))
    ))]
    {
        wide_math::exp_slice(x, out);
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Elementwise exp of a contiguous array.
pub(crate) fn exp_array(x: &Array1<f64>) -> Array1<f64> {
    if let Some(xs) = x.as_slice() {
        let mut out = vec![0.0; xs.len()];
        if exp_slice_fast(xs, &mut out) {
            return Array1::from_vec(out);
        }
    }
    x.mapv(f64::exp)
}

/// Elementwise exp of a (possibly strided) view, e.g. a column of a row-major
/// params matrix. Gathers to a contiguous buffer first when needed — the copy
/// is still cheaper than scalar exp.
pub(crate) fn exp_column(x: &ndarray::ArrayView1<f64>) -> Array1<f64> {
    let mut out = vec![0.0; x.len()];
    let handled = if let Some(s) = x.as_slice() {
        exp_slice_fast(s, &mut out)
    } else {
        let xs: Vec<f64> = x.iter().copied().collect();
        exp_slice_fast(&xs, &mut out)
    };
    if handled {
        return Array1::from_vec(out);
    }
    x.mapv(f64::exp)
}

/// Standard-normal sampler using the polar Box-Muller (Marsaglia) method —
/// rejection form, no trig calls. Deviates are generated in pairs; the spare
/// is cached per sampler instance, so create one sampler per `sample()` call
/// (not globally) and reuse it across all draws in that call.
///
/// Replaces per-draw statrs `inverse_cdf` (an inverse-erf evaluation per
/// element) in the user-invoked `sample()` paths. These paths are already
/// nondeterministic (thread RNG), so the changed RNG stream is fine;
/// distributional correctness is verified by the moment tests below.
pub(crate) struct StdNormalSampler {
    spare: Option<f64>,
}

impl StdNormalSampler {
    pub(crate) fn new() -> Self {
        StdNormalSampler { spare: None }
    }

    /// Draws one standard-normal deviate.
    #[inline]
    pub(crate) fn next<R: rand::RngExt + ?Sized>(&mut self, rng: &mut R) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u: f64 = 2.0 * rng.random::<f64>() - 1.0;
            let v: f64 = 2.0 * rng.random::<f64>() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let f = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * f);
                return u * f;
            }
        }
    }
}

/// Standard Student-t deviate via Bailey's polar method (Bailey 1994) — the
/// Box-Muller analog for the t distribution: draw (u, v) uniform in the unit
/// disk, s = u² + v², return u·√(ν(s^{−2/ν} − 1)/s). Rejection form, no trig,
/// no root-finding.
///
/// Replaces per-draw statrs `inverse_cdf` in the t `sample()` paths: besides
/// costing ~µs–100µs per call, its `inv_beta_reg` Newton iteration is
/// UNBOUNDED and has been observed to hang outright on pathological df (the
/// same failure class removed from the StudentT LogScore metric).
#[inline]
pub(crate) fn sample_std_t<R: rand::RngExt + ?Sized>(rng: &mut R, df: f64) -> f64 {
    loop {
        let u: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let v: f64 = 2.0 * rng.random::<f64>() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (df * (s.powf(-2.0 / df) - 1.0) / s).sqrt();
        }
    }
}

/// Rate-1 gamma deviate via Marsaglia & Tsang (2000), "A simple method for
/// generating gamma variables". For α ≥ 1: with d = α − 1/3, c = 1/√(9d),
/// draw x ~ N(0,1) (reusing the polar Box-Muller sampler), set v = (1+cx)³,
/// reject v ≤ 0, then accept with the squeeze u < 1 − 0.0331x⁴ or the exact
/// test ln u < x²/2 + d(1 − v + ln v); the deviate is d·v. Acceptance is
/// > 95% for all α ≥ 1. For α < 1: boost via Stuart's theorem — draw
/// g ~ Gamma(α+1, 1) and return g·U^{1/α} (U = 0 guarded).
///
/// Replaces per-draw statrs `inverse_cdf` in the gamma `sample()` path:
/// each inversion costs ~15–30 incomplete-gamma evaluations (doubling +
/// bisection + Newton), ~100× this method. The path is already
/// nondeterministic (thread RNG), so the changed RNG stream is fine;
/// distributional correctness is verified by the moment tests below and in
/// `dist::gamma`. Returns a shape-α, rate-1 deviate — scale by 1/rate for
/// Gamma(α, rate).
#[inline]
pub(crate) fn sample_gamma_rate1<R: rand::RngExt + ?Sized>(
    rng: &mut R,
    normal: &mut StdNormalSampler,
    alpha: f64,
) -> f64 {
    // α < 1: sample Gamma(α+1, 1) and apply the U^{1/α} boost at the end.
    let (a, boost) = if alpha < 1.0 {
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE); // guard U = 0
        (alpha + 1.0, u.powf(1.0 / alpha))
    } else {
        (alpha, 1.0)
    };

    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal.next(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u: f64 = rng.random();
        let x2 = x * x;
        // Cheap squeeze first, exact log test only on the rare miss.
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return boost * d * v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_within_1ulp_of_libm(xs: &Array1<f64>, fast: &Array1<f64>) {
        for (&x, &f) in xs.iter().zip(fast.iter()) {
            let s = x.exp();
            if s == f {
                continue;
            }
            assert!(
                s.is_finite() && f.is_finite(),
                "mismatch at x={x}: {f} vs {s}"
            );
            let ulp = (s.to_bits() as i64).abs_diff(f.to_bits() as i64);
            assert!(ulp <= 1, "exp({x}): {ulp} ulp from libm");
        }
    }

    fn test_points() -> Array1<f64> {
        Array1::from_vec(vec![
            -745.0, -709.0, -10.0, -1.0, -1e-12, 0.0, 1e-12, 0.5, 1.0, 10.0, 709.0, 745.0,
            f64::NEG_INFINITY, f64::INFINITY,
        ])
    }

    #[test]
    fn exp_array_matches_libm_within_1ulp() {
        let xs = test_points();
        assert_within_1ulp_of_libm(&xs, &exp_array(&xs));
    }

    /// Moment check for the polar Box-Muller sampler: mean ≈ 0, var ≈ 1,
    /// E[|Z|] ≈ √(2/π), and E[Z⁴] ≈ 3 (kurtosis catches shape errors that
    /// mean/var alone would miss). n = 200_000 → std of the mean ≈ 0.0022.
    #[test]
    fn std_normal_sampler_moments() {
        let mut rng = rand::rng();
        let mut sampler = StdNormalSampler::new();
        let n = 200_000usize;
        let (mut sum, mut sum2, mut sum_abs, mut sum4) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for _ in 0..n {
            let z = sampler.next(&mut rng);
            assert!(z.is_finite());
            sum += z;
            sum2 += z * z;
            sum_abs += z.abs();
            sum4 += z * z * z * z;
        }
        let nf = n as f64;
        let mean = sum / nf;
        let var = sum2 / nf - mean * mean;
        let mean_abs = sum_abs / nf;
        let m4 = sum4 / nf;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
        let expect_abs = (2.0 / std::f64::consts::PI).sqrt();
        assert!((mean_abs - expect_abs).abs() < 0.02, "E|Z| {mean_abs}");
        assert!((m4 - 3.0).abs() < 0.15, "E[Z^4] {m4}");
    }

    /// Moment check for Bailey's polar t sampler: mean ≈ 0 and
    /// var ≈ ν/(ν−2) for ν > 2; heavy-df draws stay finite and centered.
    #[test]
    fn std_t_sampler_moments() {
        let mut rng = rand::rng();
        let n = 400_000usize;
        for df in [5.0f64, 30.0] {
            let (mut sum, mut sum2) = (0.0f64, 0.0f64);
            for _ in 0..n {
                let t = sample_std_t(&mut rng, df);
                assert!(t.is_finite());
                sum += t;
                sum2 += t * t;
            }
            let nf = n as f64;
            let mean = sum / nf;
            let var = sum2 / nf - mean * mean;
            let expect_var = df / (df - 2.0);
            assert!(mean.abs() < 0.03, "df {df} mean {mean}");
            assert!(
                (var - expect_var).abs() < 0.12 * expect_var,
                "df {df} var {var} vs {expect_var}"
            );
        }
        // df = 1.5: moments undefined/infinite — check finiteness and that
        // the sample median sits at 0 (the distribution's median).
        let mut vals: Vec<f64> = (0..100_000)
            .map(|_| sample_std_t(&mut rng, 1.5))
            .collect();
        assert!(vals.iter().all(|v| v.is_finite()));
        vals.sort_unstable_by(|a, b| a.total_cmp(b));
        let median = vals[vals.len() / 2];
        assert!(median.abs() < 0.03, "df 1.5 median {median}");
    }

    /// Moment check for the Marsaglia–Tsang gamma sampler (rate-1): mean ≈ α
    /// and var ≈ α, covering both the boosted α < 1 branch and the direct
    /// α ≥ 1 branch. Bounds are ~5σ of the estimators at n = 200_000.
    #[test]
    fn gamma_sampler_moments() {
        let mut rng = rand::rng();
        let mut normal = StdNormalSampler::new();
        let n = 200_000usize;
        for alpha in [0.3f64, 4.0] {
            let (mut sum, mut sum2) = (0.0f64, 0.0f64);
            for _ in 0..n {
                let g = sample_gamma_rate1(&mut rng, &mut normal, alpha);
                assert!(g.is_finite() && g >= 0.0, "alpha {alpha} draw {g}");
                sum += g;
                sum2 += g * g;
            }
            let nf = n as f64;
            let mean = sum / nf;
            let var = sum2 / nf - mean * mean;
            // SE(mean) = √(α/n); SE(var) = α·√((2 + 6/α)/n).
            let se_mean = (alpha / nf).sqrt();
            let se_var = alpha * ((2.0 + 6.0 / alpha) / nf).sqrt();
            assert!((mean - alpha).abs() < 5.0 * se_mean, "alpha {alpha} mean {mean}");
            assert!((var - alpha).abs() < 5.0 * se_var, "alpha {alpha} var {var}");
        }
    }

    /// Exercise the `wide` backend directly even on macOS, where the dispatch
    /// in exp_array would otherwise always pick vForce.
    #[cfg(feature = "simd-math")]
    #[test]
    fn wide_exp_matches_libm_within_1ulp() {
        let xs = test_points();
        let v = xs.as_slice().unwrap();
        let mut out = vec![0.0; v.len()];
        wide_math::exp_slice(v, &mut out);
        assert_within_1ulp_of_libm(&xs, &Array1::from_vec(out));
    }
}
