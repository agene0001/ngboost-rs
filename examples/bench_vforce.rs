//! Accuracy + speed harness for Accelerate vForce vvexp/vvlog vs scalar libm.
//! Run BEFORE wiring vForce into any hot path.
//!
//!   cargo run --release --features accelerate --example bench_vforce

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vvexp(y: *mut f64, x: *const f64, n: *const i32);
    fn vvlog(y: *mut f64, x: *const f64, n: *const i32);
}

fn vexp(x: &[f64], y: &mut [f64]) {
    let n = x.len() as i32;
    unsafe { vvexp(y.as_mut_ptr(), x.as_ptr(), &n) }
}

fn vlog(x: &[f64], y: &mut [f64]) {
    let n = x.len() as i32;
    unsafe { vvlog(y.as_mut_ptr(), x.as_ptr(), &n) }
}

fn ulp_diff(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let (ia, ib) = (a.to_bits() as i64, b.to_bits() as i64);
    // Map to monotonic ordering of floats (works for same-sign finite values)
    let ma = if ia < 0 { i64::MIN - ia } else { ia };
    let mb = if ib < 0 { i64::MIN - ib } else { ib };
    ma.abs_diff(mb)
}

fn report_accuracy(name: &str, xs: &[f64], vforce: &[f64], scalar: &[f64]) {
    let mut max_ulp = 0u64;
    let mut max_rel = 0.0f64;
    let mut n_diff = 0usize;
    let mut worst_x = 0.0;
    for ((&x, &v), &s) in xs.iter().zip(vforce).zip(scalar) {
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

    // exp inputs: log-σ parameters live in roughly [-10, 10] during training
    let exp_in: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    // ln inputs: variances/probabilities/scores span many decades
    let log_in: Vec<f64> = (0..n)
        .map(|_| 10f64.powf(rng.random_range(-12.0..12.0)))
        .collect();

    let mut v_out = vec![0.0; n];
    let scalar_exp: Vec<f64> = exp_in.iter().map(|&x| x.exp()).collect();
    vexp(&exp_in, &mut v_out);
    report_accuracy("vvexp vs libm exp", &exp_in, &v_out, &scalar_exp);

    let scalar_log: Vec<f64> = log_in.iter().map(|&x| x.ln()).collect();
    vlog(&log_in, &mut v_out);
    report_accuracy("vvlog vs libm ln ", &log_in, &v_out, &scalar_log);

    // Edge cases
    let edges = vec![
        0.0, -0.0, 1.0, -1.0, f64::MIN_POSITIVE, 1e-300, 745.0, -745.0, 709.0, -709.0,
        f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
    ];
    let mut ve = vec![0.0; edges.len()];
    vexp(&edges, &mut ve);
    println!("\nexp edge cases (x -> vforce | libm):");
    for (&x, &v) in edges.iter().zip(&ve) {
        println!("  {x:>10e} -> {v:e} | {:e}", x.exp());
    }
    let pos_edges = vec![0.0, f64::MIN_POSITIVE, 1e-308, 1.0, f64::INFINITY, f64::NAN, -1.0];
    let mut vl = vec![0.0; pos_edges.len()];
    vlog(&pos_edges, &mut vl);
    println!("ln edge cases (x -> vforce | libm):");
    for (&x, &v) in pos_edges.iter().zip(&vl) {
        println!("  {x:>10e} -> {v:e} | {:e}", x.ln());
    }

    // Speed at hot-path sizes
    println!("\nspeed (median-ish of 7 reps):");
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
            vexp(xs, &mut out);
            std::hint::black_box(&out);
        }
        let v_ns = t.elapsed().as_nanos() as f64 / reps as f64;

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
            vlog(ls, &mut out);
            std::hint::black_box(&out);
        }
        let vln_ns = t.elapsed().as_nanos() as f64 / reps as f64;

        println!(
            "  n={size:>5}: exp {scalar_ns:>9.0}ns -> {v_ns:>8.0}ns ({:.1}x) | ln {scalar_ln_ns:>9.0}ns -> {vln_ns:>8.0}ns ({:.1}x)",
            scalar_ns / v_ns,
            scalar_ln_ns / vln_ns
        );
    }
}
