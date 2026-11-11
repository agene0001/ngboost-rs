pub mod categorical;
pub mod cauchy;
pub mod exponential;
pub mod gamma;
pub mod halfnormal;
pub mod hurdle;
pub mod laplace;
pub mod lognormal;
pub mod multivariate_normal;
pub mod negative_binomial;
pub mod normal;
pub mod normal_mixture;
pub mod poisson;
pub mod studentt;
pub mod weibull;

// Re-export all distributions for convenience
pub use categorical::{
    Bernoulli, Categorical, Categorical3, Categorical4, Categorical5, Categorical10,
};
pub use cauchy::{Cauchy, CauchyFixedVar};
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use halfnormal::HalfNormal;
pub use hurdle::{Hurdle, HurdleCrpsPart, HurdleLogNormal, HurdleNormal};
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use multivariate_normal::{
    MultivariateNormal, MultivariateNormal2, MultivariateNormal3, MultivariateNormal4,
};
pub use negative_binomial::NegativeBinomial;
pub use normal::{Normal, NormalFixedMean, NormalFixedVar};
pub use normal_mixture::{NormalMixture, NormalMixture2, NormalMixture3};
pub use poisson::Poisson;
pub use studentt::{StudentT, TFixedDf, TFixedDfFixedVar};
pub use weibull::Weibull;

use crate::scores::{Scorable, Score};
use ndarray::{Array1, Array2};
use std::fmt::Debug;

/// A trait for probability distributions.
pub trait Distribution: Sized + Clone + Debug {
    /// Creates a new distribution from a set of parameters.
    fn from_params(params: &Array2<f64>) -> Self;

    /// Fits the distribution to the data `y` and returns the initial parameters.
    fn fit(y: &Array1<f64>) -> Array1<f64>;

    /// Validate that targets lie inside the distribution's support before
    /// fitting. The default accepts anything (non-finite values are already
    /// rejected by the training entry points); distributions with restricted
    /// support override this so out-of-support targets fail loudly instead
    /// of silently corrupting training (2026-07-23 audit: negative class
    /// labels trained as class 0 via `as usize` saturation, labels >= K
    /// panicked mid-fit, LogNormal y = 0 NaN'd training immediately).
    fn validate_targets(_y: &Array1<f64>) -> Result<(), &'static str> {
        Ok(())
    }

    /// Returns the number of parameters for this distribution.
    fn n_params(&self) -> usize;

    /// Index of the internal parameter column that is a **log-scale** — an
    /// affine map on it rescales the predictive spread without moving the
    /// location — or `None`. Drives the default post-hoc scale recalibration
    /// at the end of `NGBoost::fit` (`NGBoost::auto_calibrate_scale`). Only
    /// loc/scale families return `Some`: Normal, Laplace, LogNormal, Cauchy,
    /// StudentT, TFixedDf (column 1) and NormalFixedMean (column 0).
    fn scale_param_index() -> Option<usize> {
        None
    }

    /// All log-scale columns, recalibrated jointly with one shared (a, b) by
    /// the default end-of-fit recalibration. Defaults to
    /// [`Self::scale_param_index`]; `NormalMixture2` returns both component
    /// scales.
    fn scale_param_columns() -> Vec<usize> {
        Self::scale_param_index().into_iter().collect()
    }

    /// Fewest validation rows for the default end-of-fit recalibration to
    /// run at all (from this many it fits the shift; from 300 also the
    /// slope). 40 for the loc/scale heads (R9 sweep); NegativeBinomial
    /// returns 300 because a shared dispersion shift fitted on 100 rows was
    /// a coin flip where r is weakly identified (research Cycle 16, R63).
    fn auto_calibration_min_rows() -> usize {
        40
    }

    /// Default exponent β for Fisher-weighted base-learner fits when the model
    /// does not set `fisher_weight_power` / `fisher_weighted_learner`
    /// (`None` = upstream unweighted fit). Poisson, NegativeBinomial and
    /// Categorical (incl. Bernoulli) return `Some(1.0)` (since 1.2.0): its unweighted natural-gradient leaf, mean(1 − y/λ), is
    /// a mean of ratios that under-predicts totals when rates are
    /// heterogeneous, while the weight-λ leaf Σ(λ−y)/Σλ is the Newton step
    /// (identical to XGBoost's `count:poisson` second-order trees); measured
    /// never worse and up to −0.10 NLL / −7.5 % RMSE (`docs/fisher_weight_power.md`).
    /// Override per model with `fisher_weight_power = Some(0.0)`.
    fn default_fisher_weight_power() -> Option<f64> {
        None
    }

    /// Index of the **location** column for the staged fit
    /// (`NGBoost::fit_staged`): location first, then the remaining columns,
    /// then the location again weighted by its Fisher entry (feasible GLS),
    /// then the rest. `None` = no staged fit. Normal, Laplace, LogNormal,
    /// StudentT, TFixedDf and Cauchy return `Some(0)`.
    fn location_column() -> Option<usize> {
        None
    }

    /// Whether `fit_with_validation` uses the staged fit by default (LogScore,
    /// early stopping and validation data present). True for Normal,
    /// LogNormal and StudentT (since 1.2.0; research/ngboost Cycles 13–14;
    /// Laplace was mixed and stays joint).
    fn default_staged_fit() -> bool {
        false
    }

    /// Default for `NGBoost::refit_intercept` (per-round unshrunk intercept
    /// step) when the model leaves it `None`. True only for Categorical K > 2.
    fn default_refit_intercept() -> bool {
        false
    }

    /// Returns a point prediction.
    fn predict(&self) -> Array1<f64>;

    /// Returns the parameters of the distribution.
    fn params(&self) -> Array2<f64>;

    /// Calculates the gradient of the score with respect to the distribution's parameters.
    fn grad<S: Score>(&self, y: &Array1<f64>, _score: S, natural: bool) -> Array2<f64>
    where
        Self: Scorable<S>,
    {
        Scorable::grad(self, y, natural)
    }

    fn total_score<S: Score>(&self, y: &Array1<f64>, _score: S) -> f64
    where
        Self: Scorable<S>,
    {
        Scorable::total_score(self, y, None)
    }
}

/// A sub-trait for distributions used in regression.
pub trait RegressionDistn: Distribution {}

/// A sub-trait for distributions used in classification.
pub trait ClassificationDistn: Distribution {
    fn class_probs(&self) -> Array2<f64>;
}

/// A trait providing common distribution helper methods.
///
/// This trait provides scipy-like methods for distributions:
/// - `mean()`, `std()`, `var()` - moments
/// - `pdf()`, `logpdf()` - probability density functions
/// - `cdf()` - cumulative distribution function
/// - `ppf()` - percent point function (inverse CDF / quantile function)
/// - `sample()` - random sampling
/// - `interval()` - confidence intervals
pub trait DistributionMethods: Distribution {
    /// Returns the mean of the distribution for each observation.
    fn mean(&self) -> Array1<f64>;

    /// Returns the variance of the distribution for each observation.
    fn variance(&self) -> Array1<f64>;

    /// Returns the standard deviation of the distribution for each observation.
    fn std(&self) -> Array1<f64> {
        self.variance().mapv(f64::sqrt)
    }

    /// Evaluates the probability density function at point y for each observation.
    fn pdf(&self, y: &Array1<f64>) -> Array1<f64>;

    /// Evaluates the log probability density function at point y for each observation.
    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.pdf(y).mapv(|p| p.ln())
    }

    /// Evaluates the cumulative distribution function at point y for each observation.
    fn cdf(&self, y: &Array1<f64>) -> Array1<f64>;

    /// Evaluates the percent point function (inverse CDF / quantile function).
    /// Returns the value y such that P(Y <= y) = q.
    fn ppf(&self, q: &Array1<f64>) -> Array1<f64>;

    /// Generates random samples from the distribution.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to generate per observation
    ///
    /// # Returns
    /// Array of shape (n_samples, n_observations)
    fn sample(&self, n_samples: usize) -> Array2<f64>;

    /// Returns the confidence interval for each observation.
    ///
    /// # Arguments
    /// * `alpha` - Significance level (e.g., 0.05 for 95% CI)
    ///
    /// # Returns
    /// Tuple of (lower bounds, upper bounds)
    fn interval(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        let lower_q = Array1::from_elem(self.mean().len(), alpha / 2.0);
        let upper_q = Array1::from_elem(self.mean().len(), 1.0 - alpha / 2.0);
        (self.ppf(&lower_q), self.ppf(&upper_q))
    }

    /// Returns the survival function (1 - CDF) at point y for each observation.
    fn sf(&self, y: &Array1<f64>) -> Array1<f64> {
        1.0 - self.cdf(y)
    }

    /// Returns the median of the distribution for each observation.
    fn median(&self) -> Array1<f64> {
        let q = Array1::from_elem(self.mean().len(), 0.5);
        self.ppf(&q)
    }

    /// Returns the mode of the distribution for each observation (if well-defined).
    /// Default implementation returns the mean; override for distributions where mode != mean.
    fn mode(&self) -> Array1<f64> {
        self.mean()
    }
}

/// A trait for multivariate distributions with additional methods.
pub trait MultivariateDistributionMethods: Distribution {
    /// Returns the mean vector of the distribution for each observation.
    /// Shape: (n_observations, n_dimensions)
    fn mean(&self) -> Array2<f64>;

    /// Returns the covariance matrix of the distribution for each observation.
    /// Shape: (n_observations, n_dimensions, n_dimensions)
    fn covariance(&self) -> ndarray::Array3<f64>;

    /// Generates random samples from the multivariate distribution.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to generate per observation
    ///
    /// # Returns
    /// Array of shape (n_samples, n_observations, n_dimensions)
    fn sample(&self, n_samples: usize) -> ndarray::Array3<f64>;
}
