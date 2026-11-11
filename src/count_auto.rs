//! Automatic Poisson → NegativeBinomial switch for count targets (research
//! Cycle 17).
//!
//! [`CountModel::fit_auto`] fits the Poisson head, tests its validation
//! predictions for over-dispersion with the Dean–Lawless score statistic
//!
//! &nbsp;&nbsp;&nbsp;&nbsp;z = Σ[(y − λ̂)² − y] / √(2 Σ λ̂²),
//!
//! (Dean & Lawless 1989, J. Amer. Statist. Assoc. 84; ≈ N(0, 1) under a
//! Poisson with known λ), and switches to a `NegativeBinomial` head with the
//! same settings when z > 1.645. λ̂ is predicted out of sample, so z also
//! flags predictive over-dispersion from estimation error in λ̂, which the
//! NegativeBinomial absorbs as well.
//!
//! Measured over 4 count time series, 5 iid count DGPs (10 seeds each) and
//! sunspots (5 origins), against always-Poisson (`research/ngboost`, R64):
//! - Over-dispersed data (2 NB series, 2 iid NB DGPs, sunspots): picked NB
//!   on every seed. NLL −0.07…−1.9 (all seeds), 90 % coverage 0.53–0.81 →
//!   0.87–0.91. A Poisson series with level drift: NB on 10/10, −0.092
//!   (9/10), coverage 0.82 → 0.88.
//! - True Poisson data: NB on 0/10 and 2/10 seeds where λ is easy to
//!   estimate (cost 0.000 and +0.002), and on 6/10 and 9/10 where it is
//!   not (−0.006 and −0.014: NB wins there).
//! - Choosing by validation NLL instead was a coin flip on true-Poisson
//!   data (+0.0025): NB's dispersion is recalibrated on the same rows.
//!
//! Validation data is required: the test needs out-of-sample predictions.
//! For time series pass the most recent block.
use crate::dist::{DistributionMethods, NegativeBinomial, Poisson};
use crate::learners::BaseLearner;
use crate::ngboost::NGBoost;
use crate::scores::LogScore;
use ndarray::{Array1, Array2};

/// One-sided 5 % critical value of the dispersion test (`fit_auto`).
pub const DISPERSION_Z_THRESHOLD: f64 = 1.645;

/// Dean–Lawless over-dispersion statistic of counts `y` against Poisson
/// means `lambda`: Σ[(y − λ)² − y] / √(2 Σ λ²). Large positive values mean
/// the counts vary more than a Poisson allows.
pub fn dispersion_z(y: &Array1<f64>, lambda: &Array1<f64>) -> f64 {
    assert_eq!(y.len(), lambda.len(), "dispersion_z: length mismatch");
    let num: f64 = y
        .iter()
        .zip(lambda.iter())
        .map(|(&yi, &li)| (yi - li).powi(2) - yi)
        .sum();
    let den = (2.0 * lambda.iter().map(|l| l * l).sum::<f64>()).sqrt();
    if den > 0.0 { num / den } else { 0.0 }
}

/// A count model whose head was chosen by [`CountModel::fit_auto`].
pub enum CountModel<B: BaseLearner + Clone> {
    Poisson(NGBoost<Poisson, LogScore, B>),
    NegativeBinomial(NGBoost<NegativeBinomial, LogScore, B>),
}

macro_rules! dispatch {
    ($model:expr, $m:ident => $body:expr) => {
        match $model {
            CountModel::Poisson($m) => $body,
            CountModel::NegativeBinomial($m) => $body,
        }
    };
}

impl<B: BaseLearner + Clone> CountModel<B> {
    /// Fit `template` (an unfitted Poisson model whose settings are used for
    /// both heads) on `x, y` with validation rows `x_val, y_val`; switch to
    /// a NegativeBinomial head with the same settings when the validation
    /// dispersion statistic exceeds [`DISPERSION_Z_THRESHOLD`]. Returns the
    /// model and the statistic.
    pub fn fit_auto(
        template: &NGBoost<Poisson, LogScore, B>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<(Self, f64), &'static str> {
        Self::fit_auto_with_threshold(template, x, y, x_val, y_val, DISPERSION_Z_THRESHOLD)
    }

    /// [`Self::fit_auto`] with a custom threshold (e.g. 2.326 for a 1 %
    /// test; `f64::INFINITY` = always Poisson, `f64::NEG_INFINITY` = always
    /// NegativeBinomial).
    pub fn fit_auto_with_threshold(
        template: &NGBoost<Poisson, LogScore, B>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
        threshold: f64,
    ) -> Result<(Self, f64), &'static str> {
        if y_val.is_empty() {
            return Err("fit_auto: validation rows are required");
        }
        let mut pois = template.unfitted_like::<Poisson>();
        pois.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
        let z = dispersion_z(y_val, &pois.predict(x_val));
        if z > threshold {
            let mut nb = template.unfitted_like::<NegativeBinomial>();
            nb.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
            Ok((CountModel::NegativeBinomial(nb), z))
        } else {
            Ok((CountModel::Poisson(pois), z))
        }
    }

    pub fn is_negative_binomial(&self) -> bool {
        matches!(self, CountModel::NegativeBinomial(_))
    }

    /// Predicted mean count.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        dispatch!(self, m => m.predict(x))
    }

    /// Mean negative log-likelihood of `y`.
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        dispatch!(self, m => m.score(x, y))
    }

    /// Predictive variance.
    pub fn variance(&self, x: &Array2<f64>) -> Array1<f64> {
        dispatch!(self, m => m.pred_dist(x).variance())
    }

    /// P(Y = y) per row.
    pub fn pmf(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        dispatch!(self, m => m.pred_dist(x).pdf(y))
    }

    /// P(Y ≤ y) per row.
    pub fn cdf(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        dispatch!(self, m => m.pred_dist(x).cdf(y))
    }

    /// Quantile at level `q` per row.
    pub fn ppf(&self, x: &Array2<f64>, q: f64) -> Array1<f64> {
        dispatch!(self, m => m.pred_dist(x).ppf(&Array1::from_elem(x.nrows(), q)))
    }

    /// Central (1 − alpha) prediction interval per row.
    pub fn interval(&self, x: &Array2<f64>, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        dispatch!(self, m => m.pred_dist(x).interval(alpha))
    }

    /// [`NGBoost::refit_full`] on the chosen head.
    pub fn refit_full(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
        rounds: Option<u32>,
    ) -> Result<Self, &'static str> {
        Ok(match self {
            CountModel::Poisson(m) => {
                CountModel::Poisson(m.refit_full(x, y, sample_weight, rounds)?)
            }
            CountModel::NegativeBinomial(m) => {
                CountModel::NegativeBinomial(m.refit_full(x, y, sample_weight, rounds)?)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispersion_z_is_zero_mean_under_poisson_moments() {
        // exact Poisson moments: (y−λ)² − y has expectation 0
        let lam = Array1::from(vec![2.0, 5.0, 9.0]);
        let y = Array1::from(vec![2.0, 5.0, 9.0]);
        // (0 − 2) + (0 − 5) + (0 − 9) = −16 over √(2·110)
        let z = dispersion_z(&y, &lam);
        assert!((z - (-16.0 / (220.0f64).sqrt())).abs() < 1e-12);
        let wide = Array1::from(vec![0.0, 15.0, 30.0]);
        assert!(dispersion_z(&wide, &lam) > 5.0);
    }
}
