pub mod categorical;
pub mod cauchy;
pub mod exponential;
pub mod gamma;
pub mod halfnormal;
pub mod laplace;
pub mod lognormal;
pub mod multivariate_normal;
pub mod normal;
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
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use multivariate_normal::{
    MultivariateNormal, MultivariateNormal2, MultivariateNormal3, MultivariateNormal4,
};
pub use normal::{Normal, NormalFixedMean, NormalFixedVar};
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

    /// Returns the number of parameters for this distribution.
    fn n_params(&self) -> usize;

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
