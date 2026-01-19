pub mod api;
pub mod dist;
pub mod evaluation;
pub mod learners;
pub mod ngboost;
pub mod scores;
pub mod survival;

// Re-export commonly used types at crate root
pub use ngboost::{
    EvalsResult, LearningRateSchedule, LineSearchMethod, NGBClassifier, NGBMultiClassifier,
    NGBMultiClassifier10, NGBMultiClassifier3, NGBMultiClassifier4, NGBMultiClassifier5,
    NGBRegressor, NGBoost, NGBoostParams,
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
