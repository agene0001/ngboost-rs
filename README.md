# ngboost-rs

A Rust implementation of [NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/) (Natural Gradient Boosting for Probabilistic Prediction).

NGBoost is a modular boosting algorithm that allows you to obtain full probability distributions for your predictions, not just point estimates. This enables uncertainty quantification, prediction intervals, and probabilistic forecasting.

## Features

- **Probabilistic Predictions**: Get full probability distributions, not just point estimates
- **Multiple Distributions**: Support for Normal, Poisson, Gamma, Exponential, Laplace, Weibull, and more
- **Classification Support**: Bernoulli and multi-class Categorical distributions
- **Flexible Scoring Rules**: LogScore and CRPScore implementations
- **Natural Gradient Boosting**: Uses the natural gradient for efficient optimization on probability distribution manifolds
- **Generic Design**: Easily extensible with custom distributions and base learners
- **Fast histogram trees by default**: `NGBRegressor`/`NGBClassifier` use a cached
  255-bin histogram learner — 2.5–2.7× faster end-to-end than exact trees at
  n ≥ 2000 with held-out accuracy at parity or better across regression,
  classification, survival, outlier, and discrete-feature benchmarks
  (`tests/accuracy_parity.rs`). For exact sklearn-equivalent trees (bit-level
  parity with Python ngboost), use `NGBExactRegressor` with `default_tree_learner()`.

## Installation

**⚠️ Important: No BLAS backend is enabled by default.**

This library relies on a BLAS/LAPACK backend for matrix operations. To ensure cross-platform compatibility (e.g., macOS vs Windows), no backend is selected by default. You **must** explicitly enable one of the features below in your `Cargo.toml`, otherwise the build fails (on Windows/Linux with a `compile_error!` telling you so; on macOS with unresolved LAPACK symbols such as `dgesv_` / `dgesdd_` at link time).

Two rules that avoid almost every build problem:

1. **Enable exactly one backend feature, and enable it on `ngboost-rs`** (`accelerate`, `intel-mkl` or `openblas`). The feature forwards the right backend selection to `ndarray-linalg`/`lax` and links the library for you.
2. **Do not add `ndarray-linalg`, `blas-src`, `openblas-src` or `intel-mkl-src` with a backend feature to your own `Cargo.toml`.** Cargo unifies features across the whole dependency graph, so a second backend selection anywhere (e.g. `ndarray-linalg = { features = ["intel-mkl-static"] }` next to `ngboost-rs` with `openblas`) ends in ``error[E0259]: the name `_src` is defined multiple times`` inside `lax`. If your crate needs `ndarray-linalg` for its own linear algebra, depend on it **without** any backend feature and let `ngboost-rs` pick the backend. If you really do link a BLAS yourself, enable `ngboost-rs`'s `external-blas` feature to silence the check.

### Platform-Specific BLAS Backend

Choose the configuration that matches your operating system and hardware.

#### macOS (Accelerate - Recommended)
Uses Apple's native Accelerate framework. No additional setup required.
```toml
[dependencies]
ngboost-rs = { version = "1.1", features = ["accelerate"] }
```

#### Linux (OpenBLAS)
Install OpenBLAS via your package manager:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel

# Arch
sudo pacman -S openblas
```

Then in `Cargo.toml`:
```toml
[dependencies]
ngboost-rs = { version = "1.1", features = ["openblas"] }
```

#### Windows

**Option 1: Intel MKL (Recommended)**

Intel MKL is the easiest option for Windows. It downloads pre-built binaries automatically and works on both Intel and AMD processors.
```toml
[dependencies]
ngboost-rs = { version = "1.1", features = ["intel-mkl"] }
```

That's it! No additional setup required.

> **Note for AMD users**: Intel MKL works fine on AMD processors. Intel removed artificial performance limitations years ago, so performance is good on modern AMD CPUs.

---

**Option 2: OpenBLAS via vcpkg**

If you prefer OpenBLAS, you'll need to install it via vcpkg. This requires more setup but avoids the Intel dependency.

##### Step 1: Install vcpkg

If you don't have vcpkg installed:
```powershell
# Clone vcpkg to a permanent location (e.g., C:\vcpkg)
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Integrate with your system
.\vcpkg integrate install
```

##### Step 2: Install LAPACK (includes BLAS)
```powershell
vcpkg install lapack-reference:x64-windows
```

> **Note**: This may take 20-40 minutes to build as it compiles Fortran code from source.

##### Step 3: Set Environment Variables
```powershell
# Set permanently for your user account
[System.Environment]::SetEnvironmentVariable("OPENBLAS_PATH", "C:\vcpkg\installed\x64-windows", "User")
[System.Environment]::SetEnvironmentVariable("OPENBLAS_LIB_DIR", "C:\vcpkg\installed\x64-windows\lib", "User")
```

Restart your terminal after setting these.

##### Step 4: Configure Cargo (Optional but Recommended)

Create a `.cargo/config.toml` file in your project root:
```toml
[target.x86_64-pc-windows-msvc]
rustflags = [
    "-L", "C:\\vcpkg\\installed\\x64-windows\\lib",
]
```

##### Step 5: Add to Cargo.toml
```toml
[dependencies]
ngboost-rs = { version = "1.1", features = ["openblas"] }
```

##### Troubleshooting OpenBLAS on Windows

If you get linker errors like `unresolved external symbol cblas_dgemm` or `unresolved external symbol sgetrf_`:

1. **Verify installation**:
```powershell
   dir C:\vcpkg\installed\x64-windows\lib\*.lib
```
You should see `blas.lib` and `lapack.lib` (or similar).

2. **Check environment variables**:
```powershell
   echo $env:OPENBLAS_PATH
   echo $env:OPENBLAS_LIB_DIR
```

3. **Clean rebuild**:
```powershell
   cargo clean
   cargo build
```

### Troubleshooting (all platforms)

| Symptom | Cause | Fix |
|---|---|---|
| `error: ngboost-rs: no BLAS/LAPACK backend selected` | No backend feature enabled (Windows/Linux). | Add `features = ["intel-mkl"]` or `["openblas"]` to the `ngboost-rs` dependency. |
| Linker errors: `unresolved external symbol dgesv_` / `undefined reference to dgesdd_` / `cblas_dgemm` | No backend feature enabled (macOS), or the backend library is not where the linker looks. | macOS: add `features = ["accelerate"]`. Linux/Windows OpenBLAS: see the vcpkg / package-manager steps above. |
| ``error[E0259]: the name `_src` is defined multiple times`` in `lax-0.18.x/src/lib.rs` | **Two** backends enabled on `lax` at once (e.g. `ngboost-rs` with `openblas` *plus* a direct `ndarray-linalg` dependency with `intel-mkl-static`, or `features = ["intel-mkl", "openblas"]`). | Keep one backend. Run `cargo tree -e features -i lax` to see which dependency enables the second one and remove it. |
| `error: ngboost-rs: more than one BLAS backend feature is enabled` | Two of `accelerate` / `intel-mkl` / `openblas` are on (possibly from different crates in a workspace). | Same as above: keep one. |

> **Why did adding `ndarray-linalg = { version = "0.17", features = ["intel-mkl-static"] }` "work"?** Because `0.17` is a *different* major version from the `0.18` that `ngboost-rs` uses, so Cargo built a second, separate copy of `lax` that happened to link MKL into the binary. It satisfies the linker by accident, doubles your compile time, and breaks as soon as both sides use the same version. Use `features = ["intel-mkl"]` on `ngboost-rs` instead.

---

## Quick Start

### Regression Example
```rust
use ndarray::{Array1, Array2};
use ngboost_rs::dist::Normal;
use ngboost_rs::learners::StumpLearner;
use ngboost_rs::ngboost::NGBoost;
use ngboost_rs::scores::LogScore;

fn main() {
    // Your training data
    let x_train: Array2<f64> = /* your features */;
    let y_train: Array1<f64> = /* your targets */;

    // Create and train the model
    let mut model: NGBoost<Normal, LogScore, StumpLearner> = 
        NGBoost::new(100, 0.1, StumpLearner);
    
    model.fit(&x_train, &y_train).expect("Failed to fit");

    // Make point predictions
    let predictions = model.predict(&x_test);

    // Get full predicted distributions (with uncertainty)
    let pred_dist = model.pred_dist(&x_test);
    println!("Predicted mean: {:?}", pred_dist.loc);
    println!("Predicted std:  {:?}", pred_dist.scale);
}
```

### Classification Example
```rust
use ngboost_rs::dist::Bernoulli;
use ngboost_rs::dist::ClassificationDistn;

// Binary classification
let mut model: NGBoost<Bernoulli, LogScore, StumpLearner> = 
    NGBoost::new(50, 0.1, StumpLearner);

model.fit(&x_train, &y_train).expect("Failed to fit");

// Get class predictions
let predictions = model.predict(&x_test);

// Get class probabilities
let pred_dist = model.pred_dist(&x_test);
let probabilities = pred_dist.class_probs();  // Shape: (n_samples, n_classes)
```

## Available Distributions

### Regression Distributions

| Distribution | Parameters | Use Case |
|-------------|------------|----------|
| `Normal` | loc, scale | General continuous data |
| `NormalFixedVar` | loc | When variance is known/fixed |
| `NormalFixedMean` | scale | When mean is known/fixed |
| `LogNormal` | loc, scale | Positive, right-skewed data |
| `Exponential` | scale | Waiting times, survival |
| `Gamma` | shape, rate | Positive continuous data |
| `Poisson` | rate | Count data (Fisher-weighted / Newton leaves by default since 1.2.0, see `docs/fisher_weight_power.md`) |
| `NegativeBinomial` | mean, dispersion | Over-dispersed count data (Var = μ + μ²/r; LogScore, CRPScore) |
| `NormalMixture2`, `NormalMixture3`, `NormalMixture<K>` | weights, K × (loc, scale) | Conditionally multimodal targets (`NGBMixtureRegressor` = K = 2; `NGBAutoMixtureRegressor` picks K ∈ 1..4 (cold and warm-grown candidates) by validation NLL; see `docs/normal_mixture.md`) |
| `Hurdle<D>`, `HurdleNormal`, `HurdleLogNormal` | logit P(Y=0), D's parameters | Exact zeros plus a continuous part: intermittent demand, rainfall, claims, costs (see `docs/hurdle.md`) |
| `Laplace` | loc, scale | Heavy-tailed data |
| `Weibull` | shape, scale | Survival analysis |
| `HalfNormal` | scale | Positive data near zero |
| `StudentT` | loc, scale, df | Heavy tails, robust |
| `TFixedDf` | loc, scale | T with fixed df=3 |
| `Cauchy` | loc, scale | Very heavy tails |

### Classification Distributions

| Distribution | Parameters | Use Case |
|-------------|------------|----------|
| `Bernoulli` | 1 logit | Binary classification |
| `Categorical<K>` | K-1 logits | K-class classification |
| `Categorical3` | 2 logits | 3-class classification |
| `Categorical10` | 9 logits | 10-class (e.g., digits) |

### Multivariate Distributions

| Distribution | Parameters | Use Case |
|-------------|------------|----------|
| `MultivariateNormal<P>` | P*(P+3)/2 | Multi-output regression |

## Scoring Rules

NGBoost supports different scoring rules for training:

- **LogScore**: Negative log-likelihood (default, most common)
- **CRPScore**: Continuous Ranked Probability Score (proper scoring rule).
  Since 1.2.0 every head's CRPS metric is the true 2∫∇F∇Fᵀ dt. Upstream's
  metrics for Normal, LogNormal, NormalFixedMean, HalfNormal, Gamma, Weibull,
  the t family and Poisson had the wrong units, so steps scaled with 1/σ
  (`docs/crps_metrics.md`).
```rust
use ngboost_rs::scores::{LogScore, CRPScore};

// Using LogScore (default)
let model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(100, 0.1, StumpLearner);

// Using CRPScore
let model: NGBoost<Normal, CRPScore, StumpLearner> = NGBoost::new(100, 0.1, StumpLearner);
```

## Scale recalibration (default on since 1.2.0)

Early-stopped NGBoost is systematically overconfident out of sample (90 %
intervals cover ≈ 85 %). Since 1.2.0, whenever a validation set is present
(`fit_with_validation`, or early stopping with `validation_fraction`), the
model fits `log σ ← a + b·log σ` on it at the end of `fit` and folds the map
into the stored model — lower held-out NLL on every benchmark dataset, μ
untouched, saved-model format unchanged. `model.scale_calibration` shows
`(column, a, b)`; set `auto_calibrate_scale = false` for the pre-1.2 /
upstream behaviour. Details and evidence: [`docs/scale_calibration.md`](docs/scale_calibration.md).

`NegativeBinomial` gets the same map on its dispersion, `log r ← a + b·log r`,
but only with **≥ 300 validation rows**. In-sample r is too large (intervals
too narrow) for the same reason σ is too small. The map gave −0.002…−0.026 NLL
on 5 count DGPs and 3 count time series (7–10/10 seed wins each), and was
never worse. With 100 rows it was a coin flip where r is weakly identified
(R59–R63).

**Time series** (`examples/ts_bench.rs`, lag features, chronological test
block; research Cycle 16):

- **Validate on the most recent block**, not a random subset, with
  `fit_with_validation` or `validation_split = ValidationSplit::Chronological`.
  Across 7 synthetic (head, series) pairs, a random validation split was
  worse in 5 (+0.01…+0.14 NLL, 1–4/10 seed wins) and about even in 2. It
  was also worse on co2 (0/5 origins). On the two small real series (elnino,
  sunspots; 30–70 validation rows) it scored better, but noisily.
- **Keep the recalibration on.** Fitted on the recent block it absorbs
  variance drift. Turning it off cost +0.04 / +0.24 NLL on synthetic Normal
  series (stationary / drifting) and +0.02 / +0.67 on co2 / elnino.
- **Refit on everything before forecasting.**
  `model.refit_full(&x_all, &y_all, None, None)` refits on train +
  validation (in time order) for the kept number of rounds and carries the
  recalibration over. Poisson: −0.01 / −0.05 NLL on stationary / drifting
  series (7–8/10), −0.63 on sunspots. NegativeBinomial: −0.002…−0.028.
  Normal (joint fit): neutral. Staged fits cannot be replayed, so fit with
  `staged_fit = Some(false)` if you want to refit.

Fisher-weighted base learners (`fisher_weight_power`): **on by default for
Poisson, NegativeBinomial and Categorical (binary and multi-class)** (the unweighted natural-gradient leaf under-predicts totals when
rates are heterogeneous, and the Bernoulli one under-predicts the positive
rate by 6–15 % on imbalanced data; β = 1 is the XGBoost/LogitBoost
second-order tree),
opt-in and dataset-dependent for Normal — numbers and an evaluation protocol
in [`docs/fisher_weight_power.md`](docs/fisher_weight_power.md). If your
counts are over-dispersed (√mean((y−λ̂)²/λ̂) ≫ 1 on a holdout), use the
`NegativeBinomial` head instead of Poisson. Measured negative results and
open problems: [`docs/known_limitations.md`](docs/known_limitations.md).

## Count targets: Poisson or NegativeBinomial?

`CountModel::fit_auto` fits a Poisson model and tests its validation
predictions for over-dispersion (Dean–Lawless score statistic, z > 1.645).
If the test fires, it switches to a NegativeBinomial head with the same
settings:

```rust
use ngboost_rs::CountModel;
let mut template = NGBoost::<Poisson, LogScore, _>::new(500, 0.05, default_base_learner());
template.early_stopping_rounds = Some(50);
let (model, z) = CountModel::fit_auto(&template, &x_train, &y_train, &x_val, &y_val)?;
let (lo, hi) = model.interval(&x_test, 0.1);   // 90 % interval
```

Measured (`research/ngboost`, R64):
- **Over-dispersed counts** (2 NB time series, 2 iid NB sets, sunspots): it
  picked NB every time, for −0.07…−1.9 NLL. 90 % coverage went from
  0.53–0.81 to 0.87–0.91.
- **Poisson series with a drifting level:** NB on 10/10, −0.09 NLL,
  coverage 0.82 → 0.88.
- **True Poisson data:** where λ is easy it stayed Poisson (0/10 and 2/10
  switches, cost ≤ +0.002). Where λ̂ is hard to estimate it switched on
  6–9/10 seeds, and NB won there (−0.006, −0.014).

## Staged fit (default since 1.2.0 for Normal, LogNormal, StudentT)

The joint fit under-trains the location. Its early stopping fires when the
scale, fitted to shrinking in-sample residuals, turns overconfident — long
before the location has converged.

With early stopping and validation data, `fit_with_validation` (and `fit`
with `validation_fraction`) now fits in stages:

1. the location first;
2. then the other parameters;
3. then the location again, weighted by its Fisher entry (feasible GLS);
4. then the rest again.

The result is an ordinary model (same format and prediction).

Measured against the joint fit:

- **Normal:** −0.02…−0.15 NLL on 7 of 10 datasets, neutral on 3 (10 seeds).
- **LogNormal:** −0.01…−0.12 (9–10/10).
- **StudentT:** −0.06…−0.46 on all 8 datasets (79/80 seed wins, 10 seeds).
  The joint fit put ν ≤ 1 (undefined mean and CRPS) or close to it on some
  rows in 15 of 80 runs, the staged fit in 1. The staged fit's 90 %
  intervals are a little narrower (coverage 0.86–0.92).

Cost ≈ 1.3–3.7× the tree fits. Opt out with `staged_fit = Some(false)`.
The default stages only when the scale recalibration will also run
(`auto_calibrate_scale` on and ≥ 40 validation rows). The staged scale is
fitted to the residuals of a longer-trained location and is too narrow
until recalibrated: without it the staged fit lost +0.04…+0.21 NLL on 4 of
5 datasets at n = 3000, and up to +14 at n = 700.
Laplace and NegativeBinomial (mean, then dispersion) can opt in, but both
are mixed. NegativeBinomial's CRPS improves (−0.10 on over-dispersed data),
but NLL only matches the dispersion recalibration alone, it costs 2–3×, and
it lost badly on a 300-row real series. On time series, the staged Normal
fit was NLL-neutral against joint (CRPS mostly better) and lost on the
small elnino series (5/5 origins). The Normal LogScore path also
caps outlier log-σ steps (|z| > 7.1 only). Evidence:
[`docs/normal_head.md`](docs/normal_head.md).

## Uncertainty Quantification

One of the key advantages of NGBoost is uncertainty estimation:
```rust
let pred_dist = model.pred_dist(&x_test);

// For Normal distribution
for i in 0..n_samples {
    let mean = pred_dist.loc[i];
    let std = pred_dist.scale[i];
    
    // 95% confidence interval
    let ci_lower = mean - 1.96 * std;
    let ci_upper = mean + 1.96 * std;
    
    println!("Prediction: {:.2} [{:.2}, {:.2}]", mean, ci_lower, ci_upper);
}
```

## Examples

Run the examples to see NGBoost in action:
```bash
# Basic regression
cargo run --example regression --features accelerate  # macOS
cargo run --example regression --features intel-mkl   # Windows

# Binary classification
cargo run --example classification --features intel-mkl

# Comparing different distributions
cargo run --example distributions --features intel-mkl

# Uncertainty quantification
cargo run --example uncertainty --features intel-mkl
```

## API Reference

### NGBoost
```rust
impl<D, S, B> NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    /// Create a new NGBoost model
    /// - n_estimators: Number of boosting iterations
    /// - learning_rate: Step size for updates (typically 0.01-0.1)
    /// - base_learner: The base learner to use
    pub fn new(n_estimators: usize, learning_rate: f64, base_learner: B) -> Self;

    /// Fit the model to training data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str>;

    /// Make point predictions
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Get predicted probability distributions
    pub fn pred_dist(&self, x: &Array2<f64>) -> D;
}
```

### Distribution Trait
```rust
pub trait Distribution: Sized + Clone + Debug {
    /// Create distribution from parameters
    fn from_params(params: &Array2<f64>) -> Self;
    
    /// Fit initial parameters from data
    fn fit(y: &Array1<f64>) -> Array1<f64>;
    
    /// Number of distribution parameters
    fn n_params(&self) -> usize;
    
    /// Point prediction (e.g., mean)
    fn predict(&self) -> Array1<f64>;
}
```

## Performance Tips

1. **Learning Rate**: Start with 0.1 and decrease if overfitting
2. **Number of Estimators**: More is usually better, but watch for overfitting
3. **Distribution Choice**: Match the distribution to your data characteristics
4. **Natural Gradient**: Enabled by default, provides faster convergence
5. **Release Mode**: Always use `cargo build --release` for production - significantly faster

### Building for Performance

Always compile in release mode for best performance:
```bash
cargo build --release
cargo run --release --example regression --features intel-mkl
```

The release profile includes:
- Full optimizations (`opt-level = 3`)
- Link-time optimization (`lto = "fat"`)

Debug builds are intentionally slower but compile faster during development.

## Comparison with Python NGBoost

This Rust implementation started as a port of the [Python NGBoost library](https://github.com/stanfordmlgroup/ngboost). Since 1.2.0 it also carries measured improvements as defaults (see `docs/`), each with an opt-out.

### Reproducing upstream (1.1.2) behaviour

Upgrading from 1.1.2: see [`CHANGELOG.md`](CHANGELOG.md) for every changed
default, its opt-out, and the one source-level change (`NGBoostParams`
struct literals).

Set all of these to get fits that are bit-identical to 1.1.2, the last
release whose defaults matched upstream NGBoost:

```rust
model.auto_calibrate_scale = false;     // validation-set scale recalibration
model.fisher_weight_power = Some(0.0);  // Fisher-weighted trees (Poisson, NegativeBinomial, Categorical)
model.refit_intercept = Some(false);    // per-round intercept refit (Categorical K > 2)
model.staged_fit = Some(false);         // staged fit (Normal, LogNormal, StudentT)
```

and, process-wide (read once at first use):

```bash
NGBOOST_NORMAL_GUARD=0          # Normal log-σ step guard
NGBOOST_CRPS_METRIC_LEGACY=1    # upstream CRPS metrics (wrong units; see docs/crps_metrics.md)
```

`tests/test_upstream_parity.rs` checks this on 26 head/score combinations
against hashes recorded from a 1.1.2 build (with the defaults, 20 of the 26
differ). The two env switches exist for A/B comparison; the old CRPS metrics
are a bug, not an alternative.

| Feature | Python | Rust |
|---------|--------|------|
| Core Algorithm | ✅ | ✅ |
| Natural Gradient | ✅ | ✅ |
| LogScore | ✅ | ✅ |
| CRPScore | ✅ | ✅ (every univariate head except Cauchy; see `docs/crps_metrics.md`) |
| Regression Distributions | 16 | 16 |
| Classification | ✅ | ✅ |
| Survival/Censoring | ✅ | Not yet |
| Scikit-learn Integration | ✅ | N/A |

### Distribution Parity

All Python distributions have been ported:
- Normal, NormalFixedVar, NormalFixedMean
- LogNormal, Exponential, Gamma, Poisson
- Laplace, Weibull, HalfNormal
- StudentT, TFixedDf, TFixedDfFixedVar
- Cauchy, CauchyFixedVar
- Bernoulli, Categorical (k-class)
- MultivariateNormal

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://arxiv.org/abs/1910.03225)
- [Stanford ML Group - NGBoost](https://stanfordmlgroup.github.io/projects/ngboost/)
- [Original Python Implementation](https://github.com/stanfordmlgroup/ngboost)

## Acknowledgments

This is a Rust port of the excellent [NGBoost Python library](https://github.com/stanfordmlgroup/ngboost) developed by the Stanford ML Group.