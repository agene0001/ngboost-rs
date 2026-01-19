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

## Installation

**⚠️ Important: No BLAS backend is enabled by default.**

This library relies on a BLAS/LAPACK backend for matrix operations. To ensure cross-platform compatibility (e.g., macOS vs Windows), no backend is selected by default. You **must** explicitly enable one of the features below in your `Cargo.toml`, otherwise the project will fail to link.

### Platform-Specific BLAS Backend

Choose the configuration that matches your operating system and hardware.

#### macOS (Accelerate - Recommended)
Uses Apple's native Accelerate framework.
```toml
[dependencies]
ngboost-rs = { version = "0.1", features = ["accelerate"] }
ndarray = "0.15"
```

#### Linux (OpenBLAS)
First, install OpenBLAS via your package manager:
```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora
sudo dnf install openblas-devel

# Arch
sudo pacman -S openblas
```

Then in `Cargo.toml`:
```toml
[dependencies]
ngboost-rs = { version = "0.1", features = ["openblas"] }
ndarray = "0.15"
```

#### Windows

On Windows, you have two choices depending on your CPU and setup preference:

**Option 1: OpenBLAS (Recommended for AMD processors)**
Use this if you have an AMD processor (Ryzen/Threadripper) or if you do not wish to install the Intel OneAPI Toolkit.
```toml
[dependencies]
ngboost-rs = { version = "0.1", features = ["openblas"] }
ndarray = "0.15"
```

**Option 2: Intel MKL (Recommended for Intel processors)**
This provides the best performance on Intel CPUs but requires external setup.
*   **Prerequisite:** You must install the [Intel OneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) (specifically the Math Kernel Library).
*   **Setup:** You may need to run `setvars.bat` or configure environment variables before building.
```toml
[dependencies]
ngboost-rs = { version = "0.1", features = ["intel-mkl"] }
ndarray = "0.15"
```

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
| `Poisson` | rate | Count data |
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
- **CRPScore**: Continuous Ranked Probability Score (proper scoring rule)

```rust
use ngboost_rs::scores::{LogScore, CRPScore};

// Using LogScore (default)
let model: NGBoost<Normal, LogScore, StumpLearner> = NGBoost::new(100, 0.1, StumpLearner);

// Using CRPScore
let model: NGBoost<Normal, CRPScore, StumpLearner> = NGBoost::new(100, 0.1, StumpLearner);
```

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
cargo run --example regression

# Binary classification
cargo run --example classification

# Comparing different distributions
cargo run --example distributions

# Uncertainty quantification
cargo run --example uncertainty
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
5. **Release Mode**: Always use `cargo build --release` for production - it's significantly faster

### Building for Performance

For best performance, always compile in release mode:

```bash
cargo build --release
cargo run --release --example regression --features accelerate
```

The release profile includes:
- Full optimizations (`opt-level = 3`)
- Link-time optimization (`lto = "fat"`)

Debug builds are intentionally slower but compile faster during development.

## Comparison with Python NGBoost

This Rust implementation aims to be compatible with the [Python NGBoost library](https://github.com/stanfordmlgroup/ngboost):

| Feature | Python | Rust |
|---------|--------|------|
| Core Algorithm | Yes | Yes |
| Natural Gradient | Yes | Yes |
| LogScore | Yes | Yes |
| CRPScore | Yes | Yes (Normal, Laplace) |
| Regression Distributions | 16 | 16 |
| Classification | Yes | Yes |
| Survival/Censoring | Yes | Not yet |
| Scikit-learn Integration | Yes | N/A |

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
