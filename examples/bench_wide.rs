//! Accuracy + speed harness for the `wide` crate's SIMD exp/ln vs scalar libm.
//! Portable counterpart to examples/bench_vforce.rs (which is macOS-only):
//! run this BEFORE wiring wide into any hot path, especially on Windows/Linux.
//!
//! On macOS:   cargo run --release --features accelerate,simd-math --example bench_wide
//! On Windows: cargo run --release --features intel-mkl,simd-math --example bench_wide
//! Enable the `simd-math` feature for real builds ONLY if this reports >1x.
//!
//! IMPORTANT (x86-64): set `RUSTFLAGS="-C target-cpu=native"` first, for this
//! benchmark AND for any real build using simd-math. The default target is
//! SSE2-only, where f64x4 lowers to 2x128-bit halves and exp measures
//! 0.6-0.8x (slower than libm); with AVX2 it measures 1.7-1.9x.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;
use wide::f64x4;

fn wexp(x: &[f64], y: &mut [f64]) {
    let chunks = x.len() / 4 * 4;
    for i in (0..chunks).step_by(4) {
        let v = f64x4::from([x[i], x[i + 1], x[i + 2], x[i + 3]]);
        let e: [f64; 4] = v.exp().into();
        y[i..i + 4].copy_from_slice(&e);
    }
    for i in chunks..x.len() {
        y[i] = x[i].exp();
    }
}

fn wln(x: &[f64], y: &mut [f64]) {
    let chunks = x.len() / 4 * 4;
    for i in (0..chunks).step_by(4) {
        let v = f64x4::from([x[i], x[i + 1], x[i + 2], x[i + 3]]);
        let e: [f64; 4] = v.ln().into();
        y[i..i + 4].copy_from_slice(&e);
    }
    for i in chunks..x.len() {
        y[i] = x[i].ln();
    }
}

fn ulp_diff(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let (ia, ib) = (a.to_bits() as i64, b.to_bits() as i64);
    let ma = if ia < 0 { i64::MIN - ia } else { ia };
    let mb = if ib < 0 { i64::MIN - ib } else { ib };
    ma.abs_diff(mb)
}

fn report_accuracy(name: &str, xs: &[f64], fast: &[f64], scalar: &[f64]) {
    let mut max_ulp = 0u64;
    let mut max_rel = 0.0f64;
    let mut n_diff = 0usize;
    let mut worst_x = 0.0;
    for ((&x, &v), &s) in xs.iter().zip(fast).zip(scalar) {
        let u = ulp_diff(v, s);
        if u > 0 {
            n_diff += 1;
        }
        if u > max_ulp {
            max_ulp = u;
            worst_x = x;
        }
        if s != 0.0 && s.is_finite() {
            max_rel = max_rel.max(((v - s) / s).abs());
        }
    }
    println!(
        "{name}: {} of {} values differ | max {} ulp (at x={worst_x:.6e}) | max rel err {max_rel:.3e}",
        n_diff,
        xs.len(),
        max_ulp
    );
}

fn main() {
    let mut rng = StdRng::seed_from_u64(123);
    let n = 1_000_000;

    let exp_in: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let log_in: Vec<f64> = (0..n)
        .map(|_| 10f64.powf(rng.random_range(-12.0..12.0)))
        .collect();

    let mut w_out = vec![0.0; n];
    let scalar_exp: Vec<f64> = exp_in.iter().map(|&x| x.exp()).collect();
    wexp(&exp_in, &mut w_out);
    report_accuracy("wide exp vs libm", &exp_in, &w_out, &scalar_exp);

    let scalar_log: Vec<f64> = log_in.iter().map(|&x| x.ln()).collect();
    wln(&log_in, &mut w_out);
    report_accuracy("wide ln  vs libm", &log_in, &w_out, &scalar_log);

    // Edge cases
    let edges = vec![
        0.0, -0.0, 1.0, -1.0, f64::MIN_POSITIVE, 1e-300, 745.0, -745.0, 709.0, -709.0,
        f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
    ];
    let mut ve = vec![0.0; edges.len()];
    wexp(&edges, &mut ve);
    println!("\nexp edge cases (x -> wide | libm):");
    for (&x, &v) in edges.iter().zip(&ve) {
        let ok = if v == x.exp() || (v.is_nan() && x.exp().is_nan()) { "" } else { "  <-- MISMATCH" };
        println!("  {x:>10e} -> {v:e} | {:e}{ok}", x.exp());
    }
    let pos_edges = vec![0.0, f64::MIN_POSITIVE, 1e-308, 1.0, f64::INFINITY, f64::NAN, -1.0];
    let mut vl = vec![0.0; pos_edges.len()];
    wln(&pos_edges, &mut vl);
    println!("ln edge cases (x -> wide | libm):");
    for (&x, &v) in pos_edges.iter().zip(&vl) {
        let ok = if v == x.ln() || (v.is_nan() && x.ln().is_nan()) { "" } else { "  <-- MISMATCH" };
        println!("  {x:>10e} -> {v:e} | {:e}{ok}", x.ln());
    }

    // Speed at hot-path sizes
    println!("\nspeed:");
    for size in [500usize, 2000, 5000, 20000] {
        let xs = &exp_in[..size];
        let mut out = vec![0.0; size];
        let reps = 20000000 / size.max(1);

        let t = Instant::now();
        for _ in 0..reps {
            for (o, &x) in out.iter_mut().zip(xs) {
                *o = x.exp();
            }
            std::hint::black_box(&out);
        }
        let scalar_ns = t.elapsed().as_nanos() as f64 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            wexp(xs, &mut out);
            std::hint::black_box(&out);
        }
        let w_ns = t.elapsed().as_nanos() as f64 / reps as f64;

        let ls = &log_in[..size];
        let t = Instant::now();
        for _ in 0..reps {
            for (o, &x) in out.iter_mut().zip(ls) {
                *o = x.ln();
            }
            std::hint::black_box(&out);
        }
        let scalar_ln_ns = t.elapsed().as_nanos() as f64 / reps as f64;

        let t = Instant::now();
        for _ in 0..reps {
            wln(ls, &mut out);
            std::hint::black_box(&out);
        }
        let wln_ns = t.elapsed().as_nanos() as f64 / reps as f64;

        println!(
            "  n={size:>5}: exp {scalar_ns:>9.0}ns -> {w_ns:>8.0}ns ({:.1}x) | ln {scalar_ln_ns:>9.0}ns -> {wln_ns:>8.0}ns ({:.1}x)",
            scalar_ns / w_ns,
            scalar_ln_ns / wln_ns
        );
    }
}
