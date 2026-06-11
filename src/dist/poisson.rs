use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{Array1, Array2, Array3, Zip, array};
use rand::prelude::*;
use statrs::distribution::{Discrete, DiscreteCDF, Poisson as PoissonDist};

/// The Poisson distribution.
#[derive(Debug, Clone)]
pub struct Poisson {
    pub rate: Array1<f64>,
}

impl Distribution for Poisson {
    fn from_params(params: &Array2<f64>) -> Self {
        let rate = crate::vmath::exp_column(&params.column(0));
        Poisson { rate }
    }

    fn fit(y: &Array1<f64>) -> Array1<f64> {
        let mean = y.mean().unwrap_or(1.0).max(1e-6);
        array![mean.ln()]
    }

    fn n_params(&self) -> usize {
        1
    }

    fn predict(&self) -> Array1<f64> {
        self.rate.clone()
    }

    fn params(&self) -> Array2<f64> {
        let n = self.rate.len();
        let mut p = Array2::zeros((n, 1));
        p.column_mut(0).assign(&self.rate.mapv(f64::ln));
        p
    }
}

impl RegressionDistn for Poisson {}

impl DistributionMethods for Poisson {
    fn mean(&self) -> Array1<f64> {
        // Mean of Poisson is rate (lambda)
        self.rate.clone()
    }

    fn variance(&self) -> Array1<f64> {
        // Variance of Poisson is also rate (lambda)
        self.rate.clone()
    }

    fn std(&self) -> Array1<f64> {
        self.rate.mapv(f64::sqrt)
    }

    fn pdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // PMF for Poisson (discrete, so we call it pdf for interface consistency)
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let y_int = y[i].round() as u64;
            if y[i] >= 0.0 {
                if let Ok(d) = PoissonDist::new(self.rate[i]) {
                    result[i] = d.pmf(y_int);
                }
            }
        }
        result
    }

    fn logpdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // Log PMF for Poisson
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            let y_int = y[i].round() as u64;
            if y[i] >= 0.0 {
                if let Ok(d) = PoissonDist::new(self.rate[i]) {
                    result[i] = d.ln_pmf(y_int);
                }
            } else {
                result[i] = f64::NEG_INFINITY;
            }
        }
        result
    }

    fn cdf(&self, y: &Array1<f64>) -> Array1<f64> {
        // CDF for Poisson: P(X <= y)
        let mut result = Array1::zeros(y.len());
        for i in 0..y.len() {
            if y[i] < 0.0 {
                result[i] = 0.0;
            } else {
                let y_floor = y[i].floor() as u64;
                if let Ok(d) = PoissonDist::new(self.rate[i]) {
                    result[i] = d.cdf(y_floor);
                }
            }
        }
        result
    }

    fn ppf(&self, q: &Array1<f64>) -> Array1<f64> {
        // Inverse CDF (quantile function) for Poisson
        // Returns the smallest integer k such that CDF(k) >= q
        let mut result = Array1::zeros(q.len());
        for i in 0..q.len() {
            let q_clamped = q[i].clamp(0.0, 1.0 - 1e-15);
            if let Ok(d) = PoissonDist::new(self.rate[i]) {
                // Use inverse_cdf which returns the smallest k where P(X <= k) >= q
                result[i] = d.inverse_cdf(q_clamped) as f64;
            }
        }
        result
    }

    fn sample(&self, n_samples: usize) -> Array2<f64> {
        let n_obs = self.rate.len();
        let mut samples = Array2::zeros((n_samples, n_obs));
        let mut rng = rand::rng();

        for i in 0..n_obs {
            let rate = self.rate[i];
            for s in 0..n_samples {
                // Knuth's algorithm, run in chunks of rate <= 500 so e^{-rate}
                // never underflows to 0 (which would cap draws near ~700).
                // Exact since Poisson(a + b) = Poisson(a) + Poisson(b).
                let mut remaining = rate;
                let mut total = 0u64;
                while remaining > 0.0 {
                    let chunk = remaining.min(500.0);
                    remaining -= chunk;

                    let l = (-chunk).exp();
                    let mut k = 0u64;
                    let mut p = 1.0_f64;
                    loop {
                        k += 1;
                        let u: f64 = rng.random();
                        p *= u;
                        if p <= l {
                            break;
                        }
                    }
                    total += k - 1;
                }
                samples[[s, i]] = total as f64;
            }
        }
        samples
    }

    fn median(&self) -> Array1<f64> {
        // Median of Poisson is approximately rate - ln(2) + 1/(3*rate) for large rate
        // For exact, use ppf(0.5)
        let q = Array1::from_elem(self.rate.len(), 0.5);
        self.ppf(&q)
    }

    fn mode(&self) -> Array1<f64> {
        // Mode of Poisson is floor(rate)
        // For integer rate, both floor(rate) and floor(rate)-1 are modes
        self.rate.mapv(|r| r.floor())
    }
}

impl Poisson {
    /// Returns the probability mass function (PMF) at y.
    /// This is an alias for pdf() but with a more appropriate name for discrete distributions.
    pub fn pmf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.pdf(y)
    }

    /// Returns the log probability mass function at y.
    pub fn ln_pmf(&self, y: &Array1<f64>) -> Array1<f64> {
        self.logpdf(y)
    }
}

impl Scorable<LogScore> for Poisson {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // Fisher Information for Poisson: rate (1 param)
        let n_obs = self.rate.len();
        let mut diag = Array2::zeros((n_obs, 1));
        Zip::from(diag.column_mut(0))
            .and(&self.rate)
            .for_each(|d, &rate| {
                *d = rate;
            });
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // -log_pmf = -(y * ln(rate) - rate - ln_gamma(y + 1))
        let mut scores = Array1::zeros(y.len());
        Zip::from(&mut scores)
            .and(y)
            .and(&self.rate)
            .for_each(|s, &y_i, &rate| {
                if rate > 0.0 {
                    let log_pmf =
                        y_i * rate.ln() - rate - statrs::function::gamma::ln_gamma(y_i + 1.0);
                    *s = -log_pmf;
                } else {
                    *s = f64::MAX;
                }
            });
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        // d/d(log(rate)) of -log_pmf
        // log_pmf = -rate + y*log(rate) - log(y!)
        // d(-log_pmf)/d(log(rate)) = rate - y
        let grad = &self.rate - y;

        d_params.column_mut(0).assign(&grad);
        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Fisher Information for Poisson: rate
        let n_obs = self.rate.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        for i in 0..n_obs {
            fi[[i, 0, 0]] = self.rate[i];
        }

        fi
    }
}

/// Build Poisson pmf and cdf tables for k = 0..=k_max using the multiplicative
/// recurrence pmf(k+1) = pmf(k)·λ/(k+1) — no special-function calls per entry.
/// Falls back to statrs pmf when e^{-λ} underflows (λ ≳ 708).
fn poisson_tables(lambda: f64, k_max: i64) -> (Vec<f64>, Vec<f64>) {
    let len = (k_max + 1).max(1) as usize;
    let mut pmf = vec![0.0; len];
    let mut cdf = vec![0.0; len];

    let p0 = (-lambda).exp();
    if p0 > 0.0 {
        let mut p = p0;
        let mut f = 0.0;
        for (k, (pk, fk)) in pmf.iter_mut().zip(cdf.iter_mut()).enumerate() {
            if k > 0 {
                p *= lambda / k as f64;
            }
            f += p;
            *pk = p;
            *fk = f.min(1.0);
        }
    } else if let Ok(d) = PoissonDist::new(lambda) {
        let mut f = 0.0;
        for (k, (pk, fk)) in pmf.iter_mut().zip(cdf.iter_mut()).enumerate() {
            let p = d.pmf(k as u64);
            f += p;
            *pk = p;
            *fk = f.min(1.0);
        }
    }
    (pmf, cdf)
}

/// CRPS metric value for a single rate λ: M = Σ_y g(y)² · P(Y=y), where
/// g(y) = λ · dCRPS/dλ = 2λ·(Σ_{k≥y} pmf(k) − Σ_k F(k)·pmf(k)).
///
/// Uses prefix/suffix sums over a single pmf/cdf table — O(k_max) total,
/// replacing the previous O(k_max²) nested loop with per-element cdf calls.
fn poisson_crps_metric_value(lambda: f64) -> f64 {
    let std_dev = lambda.sqrt();
    let y_max = ((lambda + 8.0 * std_dev).ceil() as i64).max(5);
    // Largest inner truncation over all y: max(λ+6σ, y_max+10)
    let k_total = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_max + 10);

    let (pmf, cdf) = poisson_tables(lambda, k_total);
    let len = pmf.len();

    // fp[k] = Σ_{j≤k} F(j)·pmf(j)  (prefix);  suf[k] = Σ_{j≥k} pmf(j)  (suffix)
    let mut fp = vec![0.0; len];
    let mut acc = 0.0;
    for k in 0..len {
        acc += cdf[k] * pmf[k];
        fp[k] = acc;
    }
    let mut suf = vec![0.0; len + 1];
    for k in (0..len).rev() {
        suf[k] = suf[k + 1] + pmf[k];
    }

    let mut metric_val = 0.0;
    for y_int in 0..=y_max {
        let y_us = y_int as usize;
        let pmf_y = pmf[y_us];
        if pmf_y < 1e-300 {
            continue;
        }

        // Same per-y truncation as the original nested loop
        let inner_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_int + 10) as usize;
        let inner_max = inner_max.min(len - 1);

        // d_crps = 2·(Σ_{y≤k≤inner_max} pmf(k) − Σ_{k≤inner_max} F(k)·pmf(k))
        let tail = suf[y_us] - suf[inner_max + 1];
        let d_crps = 2.0 * (tail - fp[inner_max]);
        let g = lambda * d_crps; // d/d(log λ) = λ · d/dλ

        metric_val += g * g * pmf_y;
    }
    metric_val
}

impl Scorable<CRPScore> for Poisson {
    fn is_diagonal_metric(&self) -> bool {
        true
    }

    fn diagonal_metric(&self) -> Array2<f64> {
        // CRPScore metric for Poisson: M = Σ_y g(y)² · P(Y=y) (1 param)
        let n_obs = self.rate.len();
        let mut diag = Array2::zeros((n_obs, 1));
        for i in 0..n_obs {
            diag[[i, 0]] = poisson_crps_metric_value(self.rate[i]);
        }
        diag
    }

    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // Discrete CRPS: CRPS = Σ_{k=0}^{∞} (F(k) - 1{y ≤ k})²
        // (identity for integer-valued distributions; verified vs Monte Carlo).
        // Truncated where the probability mass becomes negligible.
        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let lambda = self.rate[i];
            let y_i = y[i].round() as i64; // Round to nearest integer

            // Cover most of the probability mass: mean + 6σ (>99.99%)
            let std_dev = lambda.sqrt();
            let k_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_i + 10);
            let (_, cdf) = poisson_tables(lambda, k_max);

            let mut crps = 0.0;
            for (k, &f_k) in cdf.iter().enumerate() {
                let indicator = if y_i <= k as i64 { 1.0 } else { 0.0 };
                let diff = f_k - indicator;
                crps += diff * diff;
            }
            scores[i] = crps;
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of discrete CRPS w.r.t. log(rate):
        // dF(k)/dλ = -pmf(k), so
        // dCRPS/d(log λ) = λ · Σ_k 2(F(k) - 1{y≤k})·(-pmf(k))
        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let lambda = self.rate[i];
            let y_i = y[i].round() as i64;

            let std_dev = lambda.sqrt();
            let k_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_i + 10);
            let (pmf, cdf) = poisson_tables(lambda, k_max);

            let mut d_crps = 0.0;
            for k in 0..pmf.len() {
                let indicator = if y_i <= k as i64 { 1.0 } else { 0.0 };
                let diff = cdf[k] - indicator;
                d_crps += 2.0 * diff * (-pmf[k]);
            }
            d_params[[i, 0]] = lambda * d_crps;
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Deterministic CRPScore metric: M = Σ_y g(y)² · P(Y=y)
        let n_obs = self.rate.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));
        for i in 0..n_obs {
            fi[[i, 0, 0]] = poisson_crps_metric_value(self.rate[i]);
        }
        fi
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_poisson_distribution_methods() {
        let params = Array2::from_shape_vec((2, 1), vec![1.0_f64.ln(), 5.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        // Test mean: rate
        let mean = dist.mean();
        assert_relative_eq!(mean[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 5.0, epsilon = 1e-10);

        // Test variance: rate
        let var = dist.variance();
        assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(var[1], 5.0, epsilon = 1e-10);

        // Test mode: floor(rate) - but for integer rates, mode = rate
        // Due to floating point, exp(ln(5)) ≈ 5 but floor might give 4
        let mode = dist.mode();
        assert_relative_eq!(mode[0], 1.0, epsilon = 1e-10);
        // Mode for Poisson(5) is 4 or 5 (both valid), we use floor which may give 4
        assert!(mode[1] == 4.0 || mode[1] == 5.0);
    }

    #[test]
    fn test_poisson_cdf_ppf() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        // CDF at mean (5) should be around 0.616 for Poisson(5)
        let y = Array1::from_vec(vec![5.0]);
        let cdf = dist.cdf(&y);
        assert!(cdf[0] > 0.5 && cdf[0] < 0.7);

        // PPF at 0.5 should be around median
        let q = Array1::from_vec(vec![0.5]);
        let ppf = dist.ppf(&q);
        assert!(ppf[0] >= 4.0 && ppf[0] <= 5.0);
    }

    #[test]
    fn test_poisson_pmf() {
        let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        // PMF at 2 for Poisson(2): exp(-2) * 2^2 / 2! = exp(-2) * 2 ≈ 0.271
        let y = Array1::from_vec(vec![2.0]);
        let pmf = dist.pmf(&y);
        let expected = (-2.0_f64).exp() * 4.0 / 2.0;
        assert_relative_eq!(pmf[0], expected, epsilon = 1e-10);

        // PMF at 0 for Poisson(2): exp(-2) ≈ 0.135
        let y = Array1::from_vec(vec![0.0]);
        let pmf = dist.pmf(&y);
        assert_relative_eq!(pmf[0], (-2.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_sample() {
        let params = Array2::from_shape_vec((1, 1), vec![3.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let samples = dist.sample(1000);
        assert_eq!(samples.shape(), &[1000, 1]);

        // All samples should be non-negative integers
        assert!(samples.iter().all(|&x| x >= 0.0 && x.fract() == 0.0));

        // Check that sample mean is close to rate = 3
        let sample_mean: f64 = samples.column(0).iter().sum::<f64>() / samples.nrows() as f64;
        assert!((sample_mean - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_poisson_sample_large_rate() {
        // rate = 2000 > 708: e^{-rate} underflows to 0, so unchunked Knuth
        // sampling would cap draws near ~700 instead of centering at 2000
        let params = Array2::from_shape_vec((1, 1), vec![2000.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let samples = dist.sample(500);
        let sample_mean: f64 = samples.column(0).iter().sum::<f64>() / samples.nrows() as f64;
        // std of the mean is sqrt(2000/500) = 2, so +-25 is > 12 sigma
        assert!(
            (sample_mean - 2000.0).abs() < 25.0,
            "sample mean {} far from rate 2000",
            sample_mean
        );
    }

    #[test]
    fn test_poisson_fit() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let params = Poisson::fit(&y);
        assert_eq!(params.len(), 1);
        // Mean of y is 3.0, so log(rate) should be log(3)
        assert_relative_eq!(params[0], 3.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_logscore() {
        let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let y = Array1::from_vec(vec![2.0]);
        let score = Scorable::<LogScore>::score(&dist, &y);

        // Score should be finite and positive
        assert!(score[0].is_finite());
        assert!(score[0] > 0.0);

        // Score should equal -ln(pmf(2))
        let expected = -dist.ln_pmf(&y)[0];
        assert_relative_eq!(score[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![2.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let y = Array1::from_vec(vec![2.0]);
        let d_score = Scorable::<LogScore>::d_score(&dist, &y);

        // d(-log_pmf)/d(log(rate)) = rate - y = 2 - 2 = 0
        assert_relative_eq!(d_score[[0, 0]], 0.0, epsilon = 1e-10);

        // For y = 1, d_score = 2 - 1 = 1
        let y = Array1::from_vec(vec![1.0]);
        let d_score = Scorable::<LogScore>::d_score(&dist, &y);
        assert_relative_eq!(d_score[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_metric() {
        let params = Array2::from_shape_vec((1, 1), vec![3.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let metric = Scorable::<LogScore>::metric(&dist);
        // Fisher Information for Poisson is rate
        assert_relative_eq!(metric[[0, 0, 0]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_median() {
        let params = Array2::from_shape_vec((1, 1), vec![10.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        // Median for Poisson(10) is around 10
        let median = dist.median();
        assert!(median[0] >= 9.0 && median[0] <= 10.0);
    }

    #[test]
    fn test_poisson_interval() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let (lower, upper) = dist.interval(0.1);
        // 90% interval for Poisson(5) should roughly contain most probability
        assert!(lower[0] >= 0.0);
        assert!(upper[0] > lower[0]);
        assert!(lower[0] <= 5.0);
        assert!(upper[0] >= 5.0);
    }

    #[test]
    fn test_poisson_cdf_bounds() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        // CDF at very large value should be close to 1
        let y = Array1::from_vec(vec![100.0]);
        let cdf = dist.cdf(&y);
        assert!(cdf[0] > 0.999);

        // CDF at negative value should be 0
        let y_neg = Array1::from_vec(vec![-1.0]);
        let cdf_neg = dist.cdf(&y_neg);
        assert_relative_eq!(cdf_neg[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_crpscore() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap(); // rate = 5
        let dist = Poisson::from_params(&params);

        let y = Array1::from_vec(vec![5.0]);
        let score = Scorable::<CRPScore>::score(&dist, &y);

        // CRPS should be finite and non-negative
        assert!(score[0].is_finite());
        assert!(score[0] >= 0.0);

        // CRPS at the mean should be relatively small
        assert!(score[0] < 1.0);
    }

    #[test]
    fn test_poisson_crpscore_extreme() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap(); // rate = 5
        let dist = Poisson::from_params(&params);

        // CRPS at extreme values should be larger
        let y_low = Array1::from_vec(vec![0.0]);
        let y_high = Array1::from_vec(vec![15.0]);
        let y_mean = Array1::from_vec(vec![5.0]);

        let score_low = Scorable::<CRPScore>::score(&dist, &y_low);
        let score_high = Scorable::<CRPScore>::score(&dist, &y_high);
        let score_mean = Scorable::<CRPScore>::score(&dist, &y_mean);

        // Scores at extremes should be larger than at the mean
        assert!(score_low[0] > score_mean[0]);
        assert!(score_high[0] > score_mean[0]);
    }

    #[test]
    fn test_poisson_crpscore_d_score() {
        let params = Array2::from_shape_vec((1, 1), vec![3.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let y = Array1::from_vec(vec![3.0]);
        let d_score = Scorable::<CRPScore>::d_score(&dist, &y);

        // Gradient should be finite
        assert!(d_score[[0, 0]].is_finite());
    }

    #[test]
    fn test_poisson_crpscore_metric() {
        let params = Array2::from_shape_vec((1, 1), vec![5.0_f64.ln()]).unwrap();
        let dist = Poisson::from_params(&params);

        let metric = Scorable::<CRPScore>::metric(&dist);

        // Metric should be positive
        assert!(metric[[0, 0, 0]] > 0.0);
    }

    // ========================================================================
    // Equivalence tests: fast table/prefix-sum CRPS vs the original
    // O(k_max²) statrs-based computation
    // ========================================================================

    /// Original nested-loop metric using statrs cdf/pmf directly (reference).
    fn reference_crps_metric(lambda: f64) -> f64 {
        let d = PoissonDist::new(lambda).unwrap();
        let std_dev = lambda.sqrt();
        let k_max = ((lambda + 8.0 * std_dev).ceil() as i64).max(5);
        let mut metric_val = 0.0;
        for y_int in 0..=k_max {
            let pmf_y = d.pmf(y_int as u64);
            if pmf_y < 1e-300 {
                continue;
            }
            let mut d_crps = 0.0;
            let inner_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_int + 10);
            for k in 0..=inner_max {
                let f_k = d.cdf(k as u64);
                let indicator = if y_int <= k { 1.0 } else { 0.0 };
                d_crps += 2.0 * (f_k - indicator) * (-d.pmf(k as u64));
            }
            let g = lambda * d_crps;
            metric_val += g * g * pmf_y;
        }
        metric_val
    }

    #[test]
    fn test_poisson_crps_metric_matches_reference() {
        for lambda in [0.3, 1.0, 3.0, 7.5, 25.0, 80.0] {
            let fast = poisson_crps_metric_value(lambda);
            let reference = reference_crps_metric(lambda);
            assert_relative_eq!(fast, reference, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_poisson_crps_score_dscore_match_reference() {
        for lambda in [0.5, 2.0, 9.0, 40.0] {
            let params = Array2::from_shape_vec((1, 1), vec![lambda_f64_ln(lambda)]).unwrap();
            let dist = Poisson::from_params(&params);
            let d = PoissonDist::new(lambda).unwrap();
            let std_dev = lambda.sqrt();

            for y_v in [0.0, 1.0, (lambda * 0.7).round(), (lambda * 1.5).round()] {
                let y = Array1::from_vec(vec![y_v]);
                let y_i = y_v.round() as i64;
                let k_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_i + 10);

                // Reference score and d_score with statrs cdf/pmf
                let mut ref_score = 0.0;
                let mut ref_dcrps = 0.0;
                for k in 0..=k_max {
                    let f_k = d.cdf(k as u64);
                    let ind = if y_i <= k { 1.0 } else { 0.0 };
                    ref_score += (f_k - ind) * (f_k - ind);
                    ref_dcrps += 2.0 * (f_k - ind) * (-d.pmf(k as u64));
                }
                let ref_d = lambda * ref_dcrps;

                let score = Scorable::<CRPScore>::score(&dist, &y);
                let d_score = Scorable::<CRPScore>::d_score(&dist, &y);
                assert_relative_eq!(score[0], ref_score, max_relative = 1e-8);
                assert_relative_eq!(d_score[[0, 0]], ref_d, max_relative = 1e-8, epsilon = 1e-12);
            }
        }
    }

    fn lambda_f64_ln(v: f64) -> f64 {
        v.ln()
    }
}
