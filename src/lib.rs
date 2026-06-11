// Link the selected BLAS/LAPACK backend. The `extern crate` reference is
// required for the backend's build-script link flags to be applied to targets
// (like the lib's own unit tests) that don't otherwise reference it.
#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "intel-mkl")]
extern crate intel_mkl_src;
#[cfg(feature = "openblas")]
extern crate openblas_src;

pub mod api;
pub mod dist;
pub mod evaluation;
pub mod hyper_opt;
pub mod learners;
pub mod ngboost;
pub mod scores;
pub mod survival;
pub(crate) mod vmath;

// Re-export commonly used types at crate root
pub use ngboost::{
    EvalsResult, LearningRateSchedule, LineSearchMethod, NGBClassifier, NGBExactRegressor,
    NGBHistRegressor, NGBMultiClassifier, NGBMultiClassifier10, NGBMultiClassifier3,
    NGBMultiClassifier4, NGBMultiClassifier5, NGBRegressor, NGBoost, NGBoostParams,
};
pub use scores::{natural_gradient_regularized, CRPScore, LogScore, Scorable, Score};

// Re-export distribution traits
pub use dist::{ClassificationDistn, Distribution, DistributionMethods, RegressionDistn};

// Re-export evaluation functions
pub use evaluation::{
    brier_score, calculate_calib_error, calibration_curve_data, calibration_regression,
    calibration_time_to_event, concordance_index, concordance_index_uncensored_only, log_loss,
    mean_absolute_error, mean_squared_error, pit_histogram, root_mean_squared_error,
    CalibrationCurveData, CalibrationResult, PITHistogramData,
};
