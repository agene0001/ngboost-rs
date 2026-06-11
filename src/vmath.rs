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
