//! Comprehensive benchmarks for NGBoost-rs performance analysis
//!
//! Run with: cargo bench --features accelerate
//! Or for specific benchmarks: cargo bench --features accelerate -- natural_gradient

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::{Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use ngboost_rs::dist::{Distribution, Normal};
use ngboost_rs::learners::{
    ArenaDecisionTreeLearner, BaseLearner, DecisionTreeLearner, HistogramLearner, RidgeLearner,
    StumpLearner,
};
use ngboost_rs::ngboost::{LineSearchMethod, NGBRegressor};
use ngboost_rs::scores::{LogScore, Scorable, natural_gradient_regularized};

// ============================================================================
// Data Generation Utilities
// ============================================================================

/// Generate synthetic regression data
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array2::random((n_samples, n_features), Uniform::new(0.0, 1.0).unwrap());

    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let x0: f64 = x[[i, 0]];
        let x1: f64 = x[[i, 1 % n_features]];
        let x2: f64 = x[[i, 2 % n_features]];
        y[i] = 2.0 * x0 + 3.0 * x1.powi(2) - 1.5 * x2 + 0.5;
    }

    // Normalize
    let y_mean = y.mean().unwrap();
    let y_std = y.std(0.0).max(0.1);
    let y = y.mapv(|v| (v - y_mean) / y_std);

    (x, y)
}

/// Generate gradient and metric data for natural gradient benchmarks
fn generate_gradient_data(n_obs: usize, n_params: usize) -> (Array2<f64>, Array3<f64>) {
    let grad = Array2::random((n_obs, n_params), Uniform::new(-1.0, 1.0).unwrap());

    // Generate positive definite metric matrices (Fisher information)
    let mut metric = Array3::zeros((n_obs, n_params, n_params));
    for i in 0..n_obs {
        // Create a positive definite matrix: A * A^T + I
        let a = Array2::random((n_params, n_params), Uniform::new(0.0, 1.0).unwrap());
        let aat = a.dot(&a.t());
        for j in 0..n_params {
            for k in 0..n_params {
                metric[[i, j, k]] = aat[[j, k]];
            }
            metric[[i, j, j]] += 1.0; // Add identity for numerical stability
        }
    }

    (grad, metric)
}

// ============================================================================
// Natural Gradient Benchmarks
// ============================================================================

fn bench_natural_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("natural_gradient");

    // Vary number of observations
    for n_obs in [100, 500, 1000, 5000].iter() {
        let (grad, metric) = generate_gradient_data(*n_obs, 2);

        group.throughput(Throughput::Elements(*n_obs as u64));
        group.bench_with_input(
            BenchmarkId::new("n_obs", n_obs),
            &(&grad, &metric),
            |b, (g, m)| b.iter(|| natural_gradient_regularized(black_box(*g), black_box(*m), 0.0)),
        );
    }

    // Vary number of parameters (distribution complexity)
    for n_params in [1, 2, 3, 5].iter() {
        let (grad, metric) = generate_gradient_data(1000, *n_params);

        group.bench_with_input(
            BenchmarkId::new("n_params", n_params),
            &(&grad, &metric),
            |b, (g, m)| b.iter(|| natural_gradient_regularized(black_box(*g), black_box(*m), 0.0)),
        );
    }

    // Test regularization impact
    for reg in [0.0, 1e-4, 1e-2, 0.1].iter() {
        let (grad, metric) = generate_gradient_data(1000, 2);

        group.bench_with_input(
            BenchmarkId::new("tikhonov_reg", format!("{:.0e}", reg)),
            &(&grad, &metric, *reg),
            |b, (g, m, r)| {
                b.iter(|| natural_gradient_regularized(black_box(*g), black_box(*m), *r))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Base Learner Benchmarks
// ============================================================================

fn bench_base_learners(c: &mut Criterion) {
    let mut group = c.benchmark_group("base_learners");

    // Test different dataset sizes
    for n_samples in [500, 1000, 5000, 10000].iter() {
        let (x, y) = generate_regression_data(*n_samples, 10);

        group.throughput(Throughput::Elements(*n_samples as u64));

        // Stump Learner
        group.bench_with_input(
            BenchmarkId::new("stump", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                let learner = StumpLearner;
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );

        // Decision Tree (depth 3)
        group.bench_with_input(
            BenchmarkId::new("tree_d3", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                let learner = DecisionTreeLearner::new(3);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );

        // Histogram Tree (depth 3)
        group.bench_with_input(
            BenchmarkId::new("histogram_d3", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                let learner = HistogramLearner::new(3);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );

        // Ridge Regression
        group.bench_with_input(
            BenchmarkId::new("ridge", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                let learner = RidgeLearner::new(1.0);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );
    }

    // Test different tree depths
    for depth in [1, 2, 3, 4, 5, 6].iter() {
        let (x, y) = generate_regression_data(2000, 10);

        group.bench_with_input(
            BenchmarkId::new("tree_depth", depth),
            &(&x, &y, *depth),
            |b, (x, y, d)| {
                let learner = DecisionTreeLearner::new(*d);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("histogram_depth", depth),
            &(&x, &y, *depth),
            |b, (x, y, d)| {
                let learner = HistogramLearner::new(*d);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );
    }

    // Test different feature counts
    for n_features in [5, 10, 20, 50, 100].iter() {
        let (x, y) = generate_regression_data(2000, *n_features);

        group.bench_with_input(
            BenchmarkId::new("tree_features", n_features),
            &(&x, &y),
            |b, (x, y)| {
                let learner = DecisionTreeLearner::new(3);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("histogram_features", n_features),
            &(&x, &y),
            |b, (x, y)| {
                let learner = HistogramLearner::new(3);
                b.iter(|| learner.fit(black_box(*x), black_box(*y)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Distribution Benchmarks
// ============================================================================

fn bench_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributions");

    for n_obs in [100, 1000, 5000, 10000].iter() {
        // Create Normal distribution parameters
        let params = Array2::random((*n_obs, 2), Uniform::new(-1.0, 1.0).unwrap());
        let dist = Normal::from_params(&params);
        let y = Array1::random(*n_obs, Uniform::new(-2.0, 2.0).unwrap());

        group.throughput(Throughput::Elements(*n_obs as u64));

        // Score computation
        group.bench_with_input(
            BenchmarkId::new("normal_score", n_obs),
            &(&dist, &y),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );

        // Gradient computation
        group.bench_with_input(
            BenchmarkId::new("normal_d_score", n_obs),
            &(&dist, &y),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );

        // Metric (Fisher Information) computation
        group.bench_with_input(BenchmarkId::new("normal_metric", n_obs), &dist, |b, d| {
            b.iter(|| Scorable::<LogScore>::metric(black_box(d)))
        });

        // Full gradient (natural gradient)
        group.bench_with_input(
            BenchmarkId::new("normal_natural_grad", n_obs),
            &(&dist, &y),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // Standard gradient (no natural gradient)
        group.bench_with_input(
            BenchmarkId::new("normal_standard_grad", n_obs),
            &(&dist, &y),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), false)),
        );
    }

    group.finish();
}

// ============================================================================
// Weibull and TFixedDf Distribution Benchmarks
// ============================================================================

fn bench_weibull_tfixeddf(c: &mut Criterion) {
    use ngboost_rs::dist::studentt::TFixedDf;
    use ngboost_rs::dist::weibull::Weibull;

    let mut group = c.benchmark_group("dist_vectorized");

    for n_obs in [100, 1000, 5000].iter() {
        // Weibull benchmarks
        let params_w = Array2::random((*n_obs, 2), Uniform::new(-0.5, 0.5).unwrap());
        let dist_w = Weibull::from_params(&params_w);
        let y_w = Array1::random(*n_obs, Uniform::new(0.1, 5.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("weibull_score", n_obs),
            &(&dist_w, &y_w),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("weibull_d_score", n_obs),
            &(&dist_w, &y_w),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("weibull_metric", n_obs),
            &dist_w,
            |b, d| b.iter(|| Scorable::<LogScore>::metric(black_box(d))),
        );
        group.bench_with_input(
            BenchmarkId::new("weibull_grad", n_obs),
            &(&dist_w, &y_w),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // TFixedDf benchmarks
        let params_t = Array2::random((*n_obs, 2), Uniform::new(-0.5, 0.5).unwrap());
        let dist_t = TFixedDf::from_params(&params_t);
        let y_t = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("tfixeddf_score", n_obs),
            &(&dist_t, &y_t),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tfixeddf_d_score", n_obs),
            &(&dist_t, &y_t),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tfixeddf_metric", n_obs),
            &dist_t,
            |b, d| b.iter(|| Scorable::<LogScore>::metric(black_box(d))),
        );
        group.bench_with_input(
            BenchmarkId::new("tfixeddf_grad", n_obs),
            &(&dist_t, &y_t),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );
    }

    group.finish();
}

// ============================================================================
// Full Training Loop Benchmarks
// ============================================================================

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    // Scaling with number of samples
    for n_samples in [200, 500, 1000, 2000].iter() {
        let (x, y) = generate_regression_data(*n_samples, 10);

        group.throughput(Throughput::Elements(*n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut ngb = NGBRegressor::new(50, 0.1);
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    // Scaling with number of features
    for n_features in [5, 10, 20, 50].iter() {
        let (x, y) = generate_regression_data(1000, *n_features);

        group.bench_with_input(
            BenchmarkId::new("features", n_features),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut ngb = NGBRegressor::new(50, 0.1);
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    // Scaling with number of estimators
    for n_estimators in [10, 25, 50, 100, 200].iter() {
        let (x, y) = generate_regression_data(500, 10);

        group.bench_with_input(
            BenchmarkId::new("estimators", n_estimators),
            &(&x, &y, *n_estimators),
            |b, (x, y, n)| {
                b.iter(|| {
                    let mut ngb = NGBRegressor::new(*n, 0.1);
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Line Search Benchmarks
// ============================================================================

fn bench_line_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_search");
    group.sample_size(10);

    let (x, y) = generate_regression_data(1000, 10);

    // Binary line search (default)
    group.bench_with_input(
        BenchmarkId::new("method", "binary"),
        &(&x, &y),
        |b, (x, y)| {
            b.iter(|| {
                let mut ngb = NGBRegressor::with_options(
                    50, 0.1, true, 1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
                );
                // Binary is the default
                ngb.fit(black_box(*x), black_box(*y)).unwrap()
            })
        },
    );

    // Golden section line search
    group.bench_with_input(
        BenchmarkId::new("method", "golden_section"),
        &(&x, &y),
        |b, (x, y)| {
            b.iter(|| {
                let mut ngb = NGBRegressor::with_options(
                    50, 0.1, true, 1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
                );
                let mut params = ngb.get_params();
                params.line_search_method = LineSearchMethod::GoldenSection { max_iters: 20 };
                ngb.set_params(params);
                ngb.fit(black_box(*x), black_box(*y)).unwrap()
            })
        },
    );

    group.finish();
}

// ============================================================================
// Prediction Benchmarks
// ============================================================================

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");

    // Train a model first
    let (x_train, y_train) = generate_regression_data(1000, 10);
    let mut ngb = NGBRegressor::new(100, 0.1);
    ngb.fit(&x_train, &y_train).unwrap();

    for n_samples in [100, 500, 1000, 5000, 10000].iter() {
        let (x_test, _) = generate_regression_data(*n_samples, 10);

        group.throughput(Throughput::Elements(*n_samples as u64));

        // Point prediction
        group.bench_with_input(
            BenchmarkId::new("predict", n_samples),
            &(&ngb, &x_test),
            |b, (model, x)| b.iter(|| model.predict(black_box(*x))),
        );

        // Distribution prediction
        group.bench_with_input(
            BenchmarkId::new("pred_dist", n_samples),
            &(&ngb, &x_test),
            |b, (model, x)| b.iter(|| model.pred_dist(black_box(*x))),
        );
    }

    group.finish();
}

// ============================================================================
// Minibatch vs Full Batch Benchmarks
// ============================================================================

fn bench_minibatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("minibatch");
    group.sample_size(10);

    let (x, y) = generate_regression_data(2000, 10);

    for frac in [0.1, 0.25, 0.5, 0.75, 1.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("fraction", format!("{:.2}", frac)),
            &(&x, &y, *frac),
            |b, (x, y, f)| {
                b.iter(|| {
                    let mut ngb = NGBRegressor::with_options(
                        50, 0.1, true, *f,  // minibatch_frac
                        1.0, // col_sample
                        false, 100.0, 1e-4, None, 0.1, false,
                    );
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Natural vs Standard Gradient Benchmarks
// ============================================================================

fn bench_gradient_type(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_type");
    group.sample_size(10);

    let (x, y) = generate_regression_data(1000, 10);

    // With natural gradient
    group.bench_with_input(
        BenchmarkId::new("type", "natural"),
        &(&x, &y),
        |b, (x, y)| {
            b.iter(|| {
                let mut ngb = NGBRegressor::with_options(
                    50, 0.1, true, // natural_gradient = true
                    1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
                );
                ngb.fit(black_box(*x), black_box(*y)).unwrap()
            })
        },
    );

    // Without natural gradient
    group.bench_with_input(
        BenchmarkId::new("type", "standard"),
        &(&x, &y),
        |b, (x, y)| {
            b.iter(|| {
                let mut ngb = NGBRegressor::with_options(
                    50, 0.1, false, // natural_gradient = false
                    1.0, 1.0, false, 100.0, 1e-4, None, 0.1, false,
                );
                ngb.fit(black_box(*x), black_box(*y)).unwrap()
            })
        },
    );

    group.finish();
}

// ============================================================================
// Array Construction Benchmarks (to_2d_array optimization)
// ============================================================================

use ndarray::ShapeBuilder;

/// Row-major construction: scattered writes when building from columns
fn to_2d_array_row_major(cols: Vec<Array1<f64>>) -> Array2<f64> {
    if cols.is_empty() {
        return Array2::zeros((0, 0));
    }
    let nrows = cols[0].len();
    let ncols = cols.len();
    let mut result = Array2::zeros((nrows, ncols)); // row-major
    for (j, col) in cols.iter().enumerate() {
        result.column_mut(j).assign(col); // non-contiguous writes
    }
    result
}

/// Column-major construction: contiguous writes, then reinterpret
fn to_2d_array_col_major(cols: Vec<Array1<f64>>) -> Array2<f64> {
    if cols.is_empty() {
        return Array2::zeros((0, 0));
    }
    let nrows = cols[0].len();
    let ncols = cols.len();
    let mut data = Vec::with_capacity(nrows * ncols);
    for col in cols {
        data.extend(col.into_iter());
    }
    Array2::from_shape_vec((nrows, ncols).f(), data).expect("Shape mismatch")
}

fn bench_to_2d_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_2d_array");

    // Test various sizes
    for (nrows, ncols) in [
        (100, 2),
        (1000, 2),
        (5000, 2),
        (10000, 2),
        (1000, 5),
        (1000, 10),
    ]
    .iter()
    {
        let label = format!("{}x{}", nrows, ncols);

        // Row-major construction
        group.bench_with_input(
            BenchmarkId::new("row_major", &label),
            &(*nrows, *ncols),
            |b, (nr, nc)| {
                b.iter_batched(
                    || {
                        (0..*nc)
                            .map(|_| Array1::random(*nr, Uniform::new(0.0, 1.0).unwrap()))
                            .collect::<Vec<_>>()
                    },
                    |cols| to_2d_array_row_major(black_box(cols)),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        // Column-major construction
        group.bench_with_input(
            BenchmarkId::new("col_major", &label),
            &(*nrows, *ncols),
            |b, (nr, nc)| {
                b.iter_batched(
                    || {
                        (0..*nc)
                            .map(|_| Array1::random(*nr, Uniform::new(0.0, 1.0).unwrap()))
                            .collect::<Vec<_>>()
                    },
                    |cols| to_2d_array_col_major(black_box(cols)),
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark downstream operations on row-major vs column-major arrays
fn bench_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_ops_after_construction");

    for nrows in [1000, 5000].iter() {
        let ncols = 2;
        let label = format!("{}x{}", nrows, ncols);

        // Create both array types
        let cols_for_row: Vec<Array1<f64>> = (0..ncols)
            .map(|_| Array1::random(*nrows, Uniform::new(0.0, 1.0).unwrap()))
            .collect();
        let cols_for_col: Vec<Array1<f64>> = cols_for_row.clone();

        let row_major_arr = to_2d_array_row_major(cols_for_row);
        let col_major_arr = to_2d_array_col_major(cols_for_col);

        let other = Array2::random((*nrows, ncols), Uniform::new(0.0, 1.0).unwrap());

        // Element-wise subtraction (simulates params -= predictions)
        group.bench_with_input(
            BenchmarkId::new("subtract_row_major", &label),
            &(&row_major_arr, &other),
            |b, (arr, other)| {
                b.iter(|| {
                    let mut result = (*arr).clone();
                    result -= black_box(*other);
                    result
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("subtract_col_major", &label),
            &(&col_major_arr, &other),
            |b, (arr, other)| {
                b.iter(|| {
                    let mut result = (*arr).clone();
                    result -= black_box(*other);
                    result
                })
            },
        );

        // Zip iteration (simulates line search compute_next_params)
        group.bench_with_input(
            BenchmarkId::new("zip_row_major", &label),
            &(&row_major_arr, &other),
            |b, (arr, other)| {
                b.iter(|| {
                    let mut result = Array2::zeros(arr.raw_dim());
                    ndarray::Zip::from(&mut result)
                        .and(*arr)
                        .and(*other)
                        .for_each(|r, &a, &o| {
                            *r = a - 0.5 * o;
                        });
                    result
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("zip_col_major", &label),
            &(&col_major_arr, &other),
            |b, (arr, other)| {
                b.iter(|| {
                    let mut result = Array2::zeros(arr.raw_dim());
                    ndarray::Zip::from(&mut result)
                        .and(*arr)
                        .and(*other)
                        .for_each(|r, &a, &o| {
                            *r = a - 0.5 * o;
                        });
                    result
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Distribution Method Benchmarks (Zip vs Scalar Loop)
// ============================================================================

use statrs::distribution::{Continuous, ContinuousCDF, LogNormal as LogNormalDist};

/// Original scalar loop implementation for PDF
fn lognormal_pdf_scalar(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(y.len());
    for i in 0..y.len() {
        if let Ok(d) = LogNormalDist::new(loc[i], scale[i]) {
            result[i] = d.pdf(y[i]);
        }
    }
    result
}

/// Zip-based implementation for PDF
fn lognormal_pdf_zip(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(y.len());
    ndarray::Zip::from(&mut result)
        .and(y)
        .and(loc)
        .and(scale)
        .for_each(|r, &y_i, &loc_i, &scale_i| {
            if let Ok(d) = LogNormalDist::new(loc_i, scale_i) {
                *r = d.pdf(y_i);
            }
        });
    result
}

/// Original scalar loop implementation for CDF
fn lognormal_cdf_scalar(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(y.len());
    for i in 0..y.len() {
        if let Ok(d) = LogNormalDist::new(loc[i], scale[i]) {
            result[i] = d.cdf(y[i]);
        }
    }
    result
}

/// Zip-based implementation for CDF
fn lognormal_cdf_zip(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(y.len());
    ndarray::Zip::from(&mut result)
        .and(y)
        .and(loc)
        .and(scale)
        .for_each(|r, &y_i, &loc_i, &scale_i| {
            if let Ok(d) = LogNormalDist::new(loc_i, scale_i) {
                *r = d.cdf(y_i);
            }
        });
    result
}

// Constants for fast erf approximation (Abramowitz & Stegun formula 7.1.26)
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;
const ERF_P: f64 = 0.3275911;
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Fast error function approximation with ~1e-7 accuracy
#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + ERF_P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let y =
        1.0 - (ERF_A1 * t + ERF_A2 * t2 + ERF_A3 * t3 + ERF_A4 * t4 + ERF_A5 * t5) * (-x * x).exp();
    sign * y
}

/// Inlined CDF formula - no library call overhead, uses fast erf approximation
fn lognormal_cdf_inline(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let mut result = Array1::zeros(y.len());
    ndarray::Zip::from(&mut result)
        .and(y)
        .and(loc)
        .and(scale)
        .for_each(|r, &y_i, &mu, &sigma| {
            if y_i > 0.0 {
                let z = (y_i.ln() - mu) / sigma * INV_SQRT_2;
                *r = 0.5 * (1.0 + erf_approx(z));
            }
        });
    result
}

/// Inlined formula - no library call overhead, potentially SIMD-friendly
fn lognormal_pdf_inline(loc: &Array1<f64>, scale: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    let mut result = Array1::zeros(y.len());
    ndarray::Zip::from(&mut result)
        .and(y)
        .and(loc)
        .and(scale)
        .for_each(|r, &y_i, &mu, &sigma| {
            if y_i > 0.0 {
                let ln_y = y_i.ln();
                let z = (ln_y - mu) / sigma;
                *r = inv_sqrt_2pi / (y_i * sigma) * (-0.5 * z * z).exp();
            }
        });
    result
}

fn bench_distribution_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_methods");

    for n_obs in [100, 1000, 5000, 10000].iter() {
        let loc = Array1::random(*n_obs, Uniform::new(0.0, 2.0).unwrap());
        let scale = Array1::random(*n_obs, Uniform::new(0.1, 1.0).unwrap());
        let y = Array1::random(*n_obs, Uniform::new(0.1, 10.0).unwrap());

        // PDF benchmarks
        group.bench_with_input(
            BenchmarkId::new("pdf_scalar", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_pdf_scalar(black_box(loc), black_box(scale), black_box(y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pdf_zip", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_pdf_zip(black_box(loc), black_box(scale), black_box(y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pdf_inline", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_pdf_inline(black_box(loc), black_box(scale), black_box(y)))
            },
        );

        // CDF benchmarks
        group.bench_with_input(
            BenchmarkId::new("cdf_scalar", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_cdf_scalar(black_box(loc), black_box(scale), black_box(y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cdf_zip", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_cdf_zip(black_box(loc), black_box(scale), black_box(y)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cdf_inline", n_obs),
            &(&loc, &scale, &y),
            |b, (loc, scale, y)| {
                b.iter(|| lognormal_cdf_inline(black_box(loc), black_box(scale), black_box(y)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Optimization Target Benchmarks (NormalFixed*, Cauchy*, TFixedDfFixedVar, LogNormal CRP)
// ============================================================================

fn bench_optimization_targets(c: &mut Criterion) {
    use ngboost_rs::dist::cauchy::{Cauchy, CauchyFixedVar};
    use ngboost_rs::dist::halfnormal::HalfNormal;
    use ngboost_rs::dist::lognormal::LogNormal;
    use ngboost_rs::dist::normal::{NormalFixedMean, NormalFixedVar};
    use ngboost_rs::dist::studentt::{StudentT, TFixedDf, TFixedDfFixedVar};
    use ngboost_rs::scores::CRPScore;

    let mut group = c.benchmark_group("opt_targets");

    for n_obs in [100, 1000, 5000].iter() {
        // --- NormalFixedVar ---
        let params_nfv = {
            let mut p = Array2::zeros((*n_obs, 1));
            for i in 0..*n_obs {
                p[[i, 0]] = (i as f64 * 0.01).sin();
            }
            p
        };
        let dist_nfv = NormalFixedVar::from_params(&params_nfv);
        let y_nfv = Array1::random(*n_obs, Uniform::new(-2.0, 2.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("nfv_log_score", n_obs),
            &(&dist_nfv, &y_nfv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfv_log_d_score", n_obs),
            &(&dist_nfv, &y_nfv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfv_log_metric", n_obs),
            &dist_nfv,
            |b, d| b.iter(|| Scorable::<LogScore>::metric(black_box(d))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfv_crp_score", n_obs),
            &(&dist_nfv, &y_nfv),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfv_crp_d_score", n_obs),
            &(&dist_nfv, &y_nfv),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfv_log_grad", n_obs),
            &(&dist_nfv, &y_nfv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- NormalFixedMean ---
        let params_nfm = {
            let mut p = Array2::zeros((*n_obs, 1));
            for i in 0..*n_obs {
                p[[i, 0]] = 0.5 + (i as f64 * 0.01).sin() * 0.3;
            }
            p
        };
        let dist_nfm = NormalFixedMean::from_params(&params_nfm);
        let y_nfm = Array1::random(*n_obs, Uniform::new(-2.0, 2.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("nfm_log_score", n_obs),
            &(&dist_nfm, &y_nfm),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfm_log_d_score", n_obs),
            &(&dist_nfm, &y_nfm),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfm_log_metric", n_obs),
            &dist_nfm,
            |b, d| b.iter(|| Scorable::<LogScore>::metric(black_box(d))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfm_crp_score", n_obs),
            &(&dist_nfm, &y_nfm),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfm_crp_d_score", n_obs),
            &(&dist_nfm, &y_nfm),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("nfm_log_grad", n_obs),
            &(&dist_nfm, &y_nfm),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- Cauchy ---
        let params_c = Array2::random((*n_obs, 2), Uniform::new(-0.5, 0.5).unwrap());
        let dist_c = Cauchy::from_params(&params_c);
        let y_c = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("cauchy_log_score", n_obs),
            &(&dist_c, &y_c),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("cauchy_log_grad", n_obs),
            &(&dist_c, &y_c),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- CauchyFixedVar ---
        let params_cfv = {
            let mut p = Array2::zeros((*n_obs, 1));
            for i in 0..*n_obs {
                p[[i, 0]] = (i as f64 * 0.01).sin();
            }
            p
        };
        let dist_cfv = CauchyFixedVar::from_params(&params_cfv);
        let y_cfv = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("cfv_log_score", n_obs),
            &(&dist_cfv, &y_cfv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("cfv_log_grad", n_obs),
            &(&dist_cfv, &y_cfv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- TFixedDfFixedVar ---
        let params_tffv = {
            let mut p = Array2::zeros((*n_obs, 1));
            for i in 0..*n_obs {
                p[[i, 0]] = (i as f64 * 0.01).sin();
            }
            p
        };
        let dist_tffv = TFixedDfFixedVar::from_params(&params_tffv);
        let y_tffv = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("tffv_log_score", n_obs),
            &(&dist_tffv, &y_tffv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tffv_log_d_score", n_obs),
            &(&dist_tffv, &y_tffv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tffv_log_grad", n_obs),
            &(&dist_tffv, &y_tffv),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- LogNormal CRPScore ---
        let params_ln = Array2::random((*n_obs, 2), Uniform::new(-0.5, 0.5).unwrap());
        let dist_ln = LogNormal::from_params(&params_ln);
        let y_ln = Array1::random(*n_obs, Uniform::new(0.1, 5.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("ln_crp_score", n_obs),
            &(&dist_ln, &y_ln),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("ln_crp_d_score", n_obs),
            &(&dist_ln, &y_ln),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("ln_crp_grad", n_obs),
            &(&dist_ln, &y_ln),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- HalfNormal CRPScore ---
        let params_hn = {
            let mut p = Array2::zeros((*n_obs, 1));
            for i in 0..*n_obs {
                p[[i, 0]] = 0.5 + (i as f64 * 0.01).sin() * 0.3;
            }
            p
        };
        let dist_hn = HalfNormal::from_params(&params_hn);
        let y_hn = Array1::random(*n_obs, Uniform::new(0.1, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("hn_crp_score", n_obs),
            &(&dist_hn, &y_hn),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("hn_crp_d_score", n_obs),
            &(&dist_hn, &y_hn),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("hn_crp_grad", n_obs),
            &(&dist_hn, &y_hn),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- StudentT LogScore ---
        let params_st = {
            let mut p = Array2::zeros((*n_obs, 3));
            for i in 0..*n_obs {
                p[[i, 0]] = (i as f64 * 0.01).sin();
                p[[i, 1]] = 0.3 + (i as f64 * 0.02).cos() * 0.2;
                p[[i, 2]] = 3.0_f64.ln(); // df ~ 3
            }
            p
        };
        let dist_st = StudentT::from_params(&params_st);
        let y_st = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("st_log_score", n_obs),
            &(&dist_st, &y_st),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("st_log_d_score", n_obs),
            &(&dist_st, &y_st),
            |b, (d, y)| b.iter(|| Scorable::<LogScore>::d_score(black_box(*d), black_box(*y))),
        );

        // --- TFixedDf CRPScore ---
        let params_tfd = {
            let mut p = Array2::zeros((*n_obs, 2));
            for i in 0..*n_obs {
                p[[i, 0]] = (i as f64 * 0.01).sin();
                p[[i, 1]] = 0.3 + (i as f64 * 0.02).cos() * 0.2;
            }
            p
        };
        let dist_tfd = TFixedDf::from_params(&params_tfd);
        let y_tfd = Array1::random(*n_obs, Uniform::new(-3.0, 3.0).unwrap());

        group.bench_with_input(
            BenchmarkId::new("tfd_crp_score", n_obs),
            &(&dist_tfd, &y_tfd),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tfd_crp_d_score", n_obs),
            &(&dist_tfd, &y_tfd),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tfd_crp_grad", n_obs),
            &(&dist_tfd, &y_tfd),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::grad(black_box(*d), black_box(*y), true)),
        );

        // --- TFixedDfFixedVar CRPScore ---
        group.bench_with_input(
            BenchmarkId::new("tffv_crp_score", n_obs),
            &(&dist_tffv, &y_tffv),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::score(black_box(*d), black_box(*y))),
        );
        group.bench_with_input(
            BenchmarkId::new("tffv_crp_d_score", n_obs),
            &(&dist_tffv, &y_tffv),
            |b, (d, y)| b.iter(|| Scorable::<CRPScore>::d_score(black_box(*d), black_box(*y))),
        );
    }

    group.finish();
}

// ============================================================================
// Arena vs Legacy Tree Benchmark (full training loop comparison)
// ============================================================================

fn bench_arena_vs_legacy(c: &mut Criterion) {
    use ngboost_rs::ngboost::NGBoost;
    use ngboost_rs::scores::LogScore;

    let mut group = c.benchmark_group("arena_vs_legacy");
    group.sample_size(10);

    for n_samples in [500, 2000].iter() {
        let (x, y) = generate_regression_data(*n_samples, 10);

        // Legacy DecisionTreeLearner (current default)
        group.bench_with_input(
            BenchmarkId::new("legacy_tree", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let learner = DecisionTreeLearner::new(3);
                    let mut ngb =
                        NGBoost::<Normal, LogScore, DecisionTreeLearner>::new(50, 0.1, learner);
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );

        // ArenaDecisionTreeLearner (flat vec storage)
        group.bench_with_input(
            BenchmarkId::new("arena_tree", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let learner = ArenaDecisionTreeLearner::new(3);
                    let mut ngb = NGBoost::<Normal, LogScore, ArenaDecisionTreeLearner>::new(
                        50, 0.1, learner,
                    );
                    ngb.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    benches,
    bench_natural_gradient,
    bench_base_learners,
    bench_distributions,
    bench_weibull_tfixeddf,
    bench_training,
    bench_line_search,
    bench_prediction,
    bench_minibatch,
    bench_gradient_type,
    bench_to_2d_array,
    bench_array_operations,
    bench_distribution_methods,
    bench_optimization_targets,
    bench_arena_vs_legacy,
);

criterion_main!(benches);
