// Link the selected BLAS/LAPACK backend. The `extern crate` reference is
// required for the backend's build-script link flags to be applied to targets
// (like the lib's own unit tests) that don't otherwise reference it.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "openblas")]
extern crate openblas_src;

// Fail fast with one actionable line instead of a wall of unresolved LAPACK
// symbols (`dgesv_`, `dgesdd_`, ...) at link time. macOS is exempt because this
// crate's own tests/examples get Accelerate from a target-specific
// dev-dependency without any feature; downstream macOS users still need
// `features = ["accelerate"]`. `external-blas` is the escape hatch for users
// who link a BLAS/LAPACK themselves (e.g. via `blas-src`/`ndarray-linalg`).
#[cfg(not(any(
    feature = "accelerate",
    feature = "intel-mkl",
    feature = "openblas",
    feature = "external-blas",
    target_os = "macos",
)))]
compile_error!(
    "ngboost-rs: no BLAS/LAPACK backend selected. Enable exactly one of the \
     `accelerate` (macOS), `intel-mkl` (Windows/Linux) or `openblas` features \
     on ngboost-rs itself -- do NOT add `ndarray-linalg` with a backend feature \
     to your own Cargo.toml -- or enable `external-blas` if you link one \
     yourself. See https://github.com/agene0001/ngboost-rs#installation"
);

// Two backends at once either link duplicate symbols or, for intel-mkl +
// openblas, fail inside `lax` with `error[E0259]: the name `_src` is defined
// multiple times` (lax has one `extern crate <backend>_src as _src;` per
// backend feature, and Cargo unifies features across the whole graph).
#[cfg(any(
    all(feature = "accelerate", feature = "intel-mkl"),
    all(feature = "accelerate", feature = "openblas"),
    all(feature = "intel-mkl", feature = "openblas"),
))]
compile_error!(
    "ngboost-rs: more than one BLAS backend feature is enabled (accelerate / \
     intel-mkl / openblas). Enable exactly one; run \
     `cargo tree -e features -i lax` to see which dependency enables the other."
);

pub mod api;
pub mod auto_mixture;
pub mod count_auto;
pub mod dist;
pub mod evaluation;
pub mod hyper_opt;
pub mod learners;
pub mod ngboost;
pub mod scores;
pub mod survival;
pub(crate) mod vmath;

// Re-export commonly used types at crate root
pub use auto_mixture::{AutoMixtureModel, GrowStrategy, KSearch, NGBAutoMixtureRegressor};
pub use count_auto::{CountModel, DISPERSION_Z_THRESHOLD, dispersion_z};
pub use ngboost::{
    EvalsResult, LearningRateSchedule, LineSearchMethod, NGBClassifier, NGBExactRegressor,
    NGBHistRegressor, NGBMixtureRegressor, NGBMultiClassifier, NGBMultiClassifier3,
    NGBMultiClassifier4, NGBMultiClassifier5, NGBMultiClassifier10, NGBRegressor, NGBoost,
    NGBoostParams, ValidationSplit,
};
pub use scores::{CRPScore, LogScore, Scorable, Score, natural_gradient_regularized};

// Re-export distribution traits
pub use dist::{ClassificationDistn, Distribution, DistributionMethods, RegressionDistn};

// Re-export evaluation functions
pub use evaluation::{
    CalibrationCurveData, CalibrationResult, PITHistogramData, brier_score, calculate_calib_error,
    calibration_curve_data, calibration_regression, calibration_time_to_event, concordance_index,
    concordance_index_uncensored_only, log_loss, mean_absolute_error, mean_squared_error,
    pit_histogram, root_mean_squared_error,
};
