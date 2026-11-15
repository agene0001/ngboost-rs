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
| `Gamma` | shape, rate | Positive continuous data (upstream parameterization) |
| `GammaMean` | mean, shape | Positive continuous data; the same family in Fisher-orthogonal coordinates — prefer it to `Gamma` (−0.03…−0.32 held-out NLL on six datasets; staged fit by default; LogScore, CRPScore) |
| `Poisson` | rate | Count data (Fisher-weighted / Newton leaves by default since 1.2.0, see `docs/fisher_weight_power.md`) |
| `NegativeBinomial` | mean, dispersion | Over-dispersed count data (Var = μ + μ²/r; LogScore, CRPScore) |
| `NormalMixture2`, `NormalMixture3`, `NormalMixture<K>` | weights, K × (loc, scale) | Conditionally multimodal targets (`NGBMixtureRegressor` = K = 2; `NGBAutoMixtureRegressor` picks K ∈ 1..4 (cold and warm-grown candidates) by validation NLL; see `docs/normal_mixture.md`) |
| `Hurdle<D>`, `HurdleNormal`, `HurdleLogNormal` | logit P(Y=0), D's parameters | Exact zeros plus a continuous part: intermittent demand, rainfall, claims, costs (see `docs/hurdle.md`) |
| `Laplace` | loc, scale | Heavy-tailed data |
| `Weibull` | shape, scale | Survival analysis |
| `HalfNormal` | scale | Positive data near zero |
| `StudentT` | loc, scale, df | Heavy tails, robust |
| `StudentTFiniteVar` | loc, scale, df > 2.1 | Heavy tails with a guaranteed finite mean and variance (`StudentT` can predict df ≤ 2 or ≤ 1); same held-out NLL on average |
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

On real low-count data (1.3.0 research, Cycle 26: 14 019 football matches
from six leagues, and three public count sets; held-out NLL):

| data | variance / mean | NegativeBinomial − Poisson | `fit_auto` chose NegativeBinomial |
|---|---|---|---|
| football goals, home and away (mean 1.5 and 1.2) | 1.07–1.10 | **+0.001 … +0.007** (Poisson better in all eight comparisons) | 13 of 232 fits |
| doctor visits (mean 0.3) | 2.1 | −0.024 | 24 of 30 |
| articles per biochemist (mean 1.7) | 2.2 | −0.076 | 9 of 10 |
| physician visits (mean 5.8) | 7.9 | −1.50 | 30 of 30 |

So for a goals-type target start from Poisson, or `fit_auto`, which lost
nothing measurable against the better head on any of these. On the
football data nothing about the *fit* moved the score by more than 0.007
— the leaf size (−0.000 … −0.006 for the 1.3.0 default against leaf 1),
the staged NegativeBinomial fit with or without cross-fitting, or a
30-trial tuner search (−0.0001) — while the features were worth 0.05–0.07
over a constant rate and market odds another 0.014.

Read the football rows for what they are: one sport, with basic features
(market odds, recent form, Elo) and no feature engineering, so the target
carries little signal. They show that the 1.3.0 defaults do no harm
there; they do not show that the fit options stop mattering once the
features are strong. On the synthetic sets where the rate varies strongly
the same options were worth 0.03–0.07 (leaf size, tuner search,
cross-fitted staged NegativeBinomial), and no default in this crate was
chosen from the football data.

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

`GammaMean` and `StudentTFiniteVar` (1.3.0) are staged by default as well.

### Cross-fitted residuals (opt-in, 1.3.0)

The scale stages above are fitted to the residuals of a location that saw
the same rows, so those residuals are too small; the validation-set
recalibration repairs the level afterwards. Setting

```rust
model.staged_crossfit_folds = 5;
```

gives the scale stages out-of-fold location predictions for the training
rows instead (5 extra location fits per location stage; 2.4–6.6× the fit
time of the default staged fit, median 5.6×). Normal head, six
datasets × 10 seeds, held-out NLL against the default:

- 2100 training rows: −0.02…−0.09 on four datasets, ±0.01 on two;
- 490 training rows: −0.03…−0.09 on five datasets;
- only 30 validation rows (too few to recalibrate, so the default falls
  back to the joint fit): −0.08…−0.30 on five datasets.

It loses on a heavy-tailed target on small data (protein: +0.07 at 490
training rows, +0.09 with 30 validation rows). Row by row, 5 of 1410 test
rows carry that loss: targets in the far mode of a two-humped
distribution, which a sharper σ punishes under a Gaussian tail. With
`StudentTFiniteVar` the loss is gone (+0.004) — on such data change the
head rather than rely on cross-fitting with `Normal`.

**Why it is not the default.** It was tested as one (the shipped default
model, all five staged heads, 19 head × dataset pairs, 100 to 20 000
rows, two independent sets of 10 seeds; rule fixed before the runs):

| | held-out NLL, on − off | fits lower | RMSE | fit time |
|---|---|---|---|---|
| under 400 rows (fewer than 40 validation rows) | −0.02 … −0.09 on average, single datasets +0.1 … +0.3 | 65–83 % | −2 … −7 % | about 10× (the default is a short joint fit there) |
| 400–6000 rows, 5 folds | −0.045 (se 0.007) | 1543 of 1900 | −1.1 % | 2.4–6.6×, median 5.6× |
| 400–6000 rows, 3 folds | −0.043 (se 0.007) | 1474 of 1900 | −0.9 % | 0.9–5.0×, median 3.2× |
| 20 000 rows, 5 folds | −0.018 | 15 of 20 | −0.7 % | about 3× |

So switching it on is worth it when calibration matters more than fit
time and you have at least a few hundred rows: 18 of 19 head × dataset
pairs improved, and three folds buy nearly all of it. It gains nothing
for the Normal head on a heavy-tailed target (protein: +0.005 NLL, RMSE
+0.5 %), and always-on failed the no-harm rule on small data.

Other staged heads, three or four datasets × 10 seeds each: `GammaMean`
−0.005…−0.084 NLL with means unaffected; `StudentTFiniteVar` −0.008…−0.048 NLL, means
unaffected or slightly better; `LogNormal` −0.005…−0.124 NLL, means
unaffected or slightly better.

### LogNormal scale ceiling (1.3.0)

`LogNormal` predictions clamp σ at four times the model's baseline σ (its
stored scale constant: after a staged fit the rms residual scale of
ln y). The predictive mean exp(μ + σ²/2) is exponential in σ², and on a
target that is not log-normal in its tails (kin8nm, bounded above) a few
rows fitted with a very large σ made it explode: RMSE up to 0.57 with the
default fit and up to 10²⁵ with cross-fitting. With the ceiling both are
0.15 and NLL is 0.001–0.003 better; on true log-normal data, a
gamma-distributed target and a real cost target no row reaches it and
results are unchanged to the last digit. Only predictions are clamped —
the fit and the stored model are the same — so
`LogNormal::from_params(&model.pred_param_unbounded(&x))` gives the
unclamped distribution, and `Normal` fitted on ln y is the same model
without a ceiling. A region with less than 1/16 of the rows and a true σ
above four times the baseline would be under-dispersed; none was seen.

`HurdleLogNormal` has no such ceiling (its joint fit has no baseline to
measure against); instead its `predict` returns (1 − π)·E[min(Y, c)] with
c the largest target seen in training, stored with the model as
`target_max`. That bounds every point prediction by the largest target;
on three benchmark datasets it changes RMSE by −0.8 %, −0.6 % and +0.1 %.
`pred_dist(x).mean()` is the plain mean. Model files written before 1.3.0
still load (`load_model`, or `SerializedNGBoost::from_bincode` for your
own bincode blobs) and keep the plain mean.

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

and build histogram learners with the pre-1.3 leaf size,
`HistogramLearner::with_params(depth, 255, 1, 2)` (the default
`min_samples_leaf` is 20 since 1.3.0; see below). `default_tree_learner()`
and `arena_tree_learner()` still use scikit-learn's leaf 1; for
`DecisionTreeLearner::new(depth)` write
`DecisionTreeLearner::with_params(depth, 2, 1)`. A `LogNormal` model's
pre-1.3 predictions are `pred_param_unbounded`.

and, process-wide (read once at first use):

```bash
NGBOOST_NORMAL_GUARD=0          # Normal log-σ step guard
NGBOOST_CRPS_METRIC_LEGACY=1    # upstream CRPS metrics (wrong units; see docs/crps_metrics.md)
```

`tests/test_upstream_parity.rs` checks this on 26 head/score combinations
against hashes recorded from a 1.1.2 build (with the defaults, 20 of the 26
differ). The two env switches exist for A/B comparison; the old CRPS metrics
are a bug, not an alternative.

## Tree shape and hyperparameter search (1.3.0)

`HistogramLearner::new(depth)` uses `min_samples_leaf = 20` (1 before
1.3.0). Against leaf 1 at depth 3 it lowered held-out NLL by 0.014–0.062
on six datasets at 2100 training rows and by more at 210–490 rows, and was
never significantly worse in 50 dataset × head cases (ten head families,
including multiclass, hurdle and mixture heads; survival and
sample-weighted fits were not measured).

On small data the default shrinks: a tree fitted on n rows uses
`min(20, n / 10)`, so nothing changes from 200 rows up. Against a fixed 20
that lowered RMSE by 5.0 % at 70 training rows, 1.7 % at 105 and 1.2 % at
140 (six datasets × 20 seeds, pooled; one dataset gains 4–25 %, none is
significantly worse). An explicit leaf size
(`HistogramLearner::with_params`) is used exactly as given. On count and
classification heads the shrinking leaf is a trade rather than a gain: a
strong nonlinear signal gains a lot (Poisson: −0.25…−0.60 NLL at 70–105
rows), a weak one loses a little (+0.03…+0.05); for small, weak-signal
count data pass leaf 20 explicitly.

`DecisionTreeLearner::new(depth)` and `ArenaDecisionTreeLearner::new(depth)`
use the same default (they used leaf 1 before 1.3.0). Against leaf 1 on
six datasets: −0.01…−0.17 held-out NLL at 210–2100 training rows for the
exact learner, −0.00…−0.08 at 2100 rows for the arena learner.
`default_tree_learner()`, `arena_tree_learner()` and `default_sklearn()`
are the scikit-learn-compatible learners and keep leaf 1.

Tuning tree shape is worth more than that default: tuned configurations
beat the defaults by 0.06–0.27 held-out NLL on five of six datasets, with
`min_samples_leaf` the strongest knob. `hyper_opt` ships the measured
space and a matching builder:

```rust
use ngboost_rs::hyper_opt::{
    default_histogram_model, default_search_space, hyper_opt_with_config, HyperOptConfig,
};

let cfg = HyperOptConfig { n_trials: 40, n_folds: 5, ..Default::default() };
let result = hyper_opt_with_config::<Normal, LogScore, HistogramLearner, _>(
    &x, &y, &default_search_space(), cfg, default_histogram_model::<Normal, LogScore>,
)?;
let mut model = default_histogram_model::<Normal, LogScore>(&result.best_params);
model.fit(&x, &y)?;
```

The space is learning rate 0.02–0.2 (log), depth 2–8, `min_samples_leaf`
5–200 (log), `max_bins` 8–255 (log) and `col_sample` 0.6–1. `n_estimators` is not searched: with
early stopping it is only a cap, and a low sampled cap truncates low
learning rates. The builder sets the cap to `100 / learning_rate` and the
patience to `2.5 / learning_rate` (50–200 rounds); fold fits use the same
patience rule when a custom builder sets none. The search-space study used
the Normal head on datasets of up to 12 000 rows
(`research/automl-hyperparameter/`).

Run end to end (30 trials, 5 folds, 2400 training rows, five datasets × 8
seeds, the winner refitted and scored on untouched rows), held-out NLL:

| compared with | this search | lower on |
|---|---|---|
| the 1.2.0 example (learning rate, depth, `n_estimators` 200–1500) | −0.093 (se 0.015) | 37 of 40 |
| no tuning (`default_histogram_model` with no parameters) | −0.077 (se 0.011) | 36 of 40 |

The 1.2.0 example was no better than not tuning (+0.016, lower on 22 of
40): on two datasets its sampled round cap and leaf 1 cost more than the
search gained. About 40 % of the difference to it comes from the two
changed defaults (leaf 20, the patience rule) and the rest from the
space.

On other heads the margins are about half (LogNormal, Poisson,
NegativeBinomial; five datasets × 8 seeds): −0.045 against the 1.2.0
example (lower on 31 of 40) and −0.037 against no tuning (34 of 40). It
was never worse than not tuning, and gained nothing on a Poisson target
whose rate barely varies. Classification heads were not measured.

Those comparisons used the space without `max_bins`. Searching it too, at
the same 30 trials (eight dataset-head pairs × 8 seeds), lowered held-out
NLL by a further 0.009 (lower on 41 of 64 searches): 0.03–0.05 on kin8nm,
where the winning models use 18–19 bins instead of 255, and nothing beyond
noise on the other six (one was 0.02 worse, 1.5 standard errors). Remove
the key from the map to search without it.

For heads with a staged fit (Normal, LogNormal, StudentT,
StudentTFiniteVar, GammaMean), `default_search_space_with_crossfit()`
also lets the search turn on cross-fitted residuals. At the same 30
trials (seven dataset-head pairs × 8 seeds) that lowered held-out NLL by
0.043 (lower on 42 of 56 searches, −0.16 … +0.004 by dataset), and the
winning model used cross-fitting in 52 of 56. Point predictions did not
improve, and the search takes several times longer, which is why this is
a separate function and not the default.

Choosing the *head* by comparing two searches' CV scores is not
recommended: at the same total budget it was 0.007 worse than searching
one head, it helped only where the noise was truly heavy-tailed, and on
one dataset it picked the worse head every time. Choose the head from
what you know about the target.

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