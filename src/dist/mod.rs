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
    Bernoulli, Categorical, Categorical10, Categorical3, Categorical4, Categorical5,
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
    fn params(&self) -> &Array2<f64>;

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
