//! Vectorized transcendental helpers.
//!
//! On macOS with the `accelerate` feature, exp over contiguous slices goes
//! through vForce's `vvexp` (~2× faster than scalar libm at n≥500). vForce is
//! within 1 ulp of libm `exp` with identical edge-case behavior (overflow at
//! ±745, denormals, ±inf, NaN) — measured in examples/bench_vforce.rs.
//! Everywhere else this falls back to scalar libm.

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

/// Elementwise exp of a contiguous array.
pub(crate) fn exp_array(x: &Array1<f64>) -> Array1<f64> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if let Some(xs) = x.as_slice() {
        let mut out = vec![0.0; xs.len()];
        accel::exp_slice(xs, &mut out);
        return Array1::from_vec(out);
    }
    x.mapv(f64::exp)
}

/// Elementwise exp of a (possibly strided) view, e.g. a column of a row-major
/// params matrix. Gathers to a contiguous buffer first when needed — the copy
/// is still cheaper than scalar exp.
pub(crate) fn exp_column(x: &ndarray::ArrayView1<f64>) -> Array1<f64> {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    {
        let mut out = vec![0.0; x.len()];
        if let Some(s) = x.as_slice() {
            accel::exp_slice(s, &mut out);
        } else {
            let xs: Vec<f64> = x.iter().copied().collect();
            accel::exp_slice(&xs, &mut out);
        }
        return Array1::from_vec(out);
    }
    #[allow(unreachable_code)]
    x.mapv(f64::exp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_array_matches_libm_within_1ulp() {
        let xs = Array1::from_vec(vec![
            -745.0, -709.0, -10.0, -1.0, -1e-12, 0.0, 1e-12, 0.5, 1.0, 10.0, 709.0, 745.0,
            f64::NEG_INFINITY, f64::INFINITY,
        ]);
        let fast = exp_array(&xs);
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
}
