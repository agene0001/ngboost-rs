use crate::dist::{Distribution, DistributionMethods, RegressionDistn};
use crate::scores::{CRPScore, LogScore, Scorable};
use ndarray::{array, Array1, Array2, Array3};
use rand::prelude::*;
use statrs::distribution::{Discrete, DiscreteCDF, Poisson as PoissonDist};

/// The Poisson distribution.
#[derive(Debug, Clone)]
pub struct Poisson {
    pub rate: Array1<f64>,
    _params: Array2<f64>,
}

impl Distribution for Poisson {
    fn from_params(params: &Array2<f64>) -> Self {
        let rate = params.column(0).mapv(f64::exp);
        Poisson {
            rate,
            _params: params.clone(),
        }
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

    fn params(&self) -> &Array2<f64> {
        &self._params
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
                // Use Knuth's algorithm for Poisson sampling
                // This is more reliable than inverse CDF for the statrs implementation
                let l = (-rate).exp();
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
                samples[[s, i]] = (k - 1) as f64;
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
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut scores = Array1::zeros(y.len());
        for (i, &y_i) in y.iter().enumerate() {
            if let Ok(d) = PoissonDist::new(self.rate[i]) {
                scores[i] = -d.ln_pmf(y_i.round() as u64);
            } else {
                scores[i] = f64::MAX;
            }
        }
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

impl Scorable<CRPScore> for Poisson {
    fn score(&self, y: &Array1<f64>) -> Array1<f64> {
        // CRPS for Poisson distribution (discrete CRPS)
        // For discrete distributions:
        // CRPS = E|X - y| - 0.5 * E|X - X'|
        // where X, X' are independent draws from the Poisson distribution
        //
        // For Poisson with rate λ, there's a closed-form expression:
        // CRPS = y * (2*F(y) - 1) - λ * (2*F(y-1) - 1) + 2*λ*f(floor(y)) - λ*exp(-2λ)*I_0(2λ)
        // where F is CDF, f is PMF, and I_0 is modified Bessel function
        //
        // Simpler approximation using the identity:
        // CRPS = 2 * Σ_{k=0}^{∞} (F(k) - 1{y ≤ k}) * (F(k) - 1{y ≤ k-1})
        //      = Σ_{k=0}^{∞} (F(k) - 1{y ≤ k})²
        //
        // We compute this by summing over k values with significant probability mass

        let mut scores = Array1::zeros(y.len());

        for i in 0..y.len() {
            let lambda = self.rate[i];
            let y_i = y[i].round() as i64; // Round to nearest integer

            if let Ok(d) = PoissonDist::new(lambda) {
                // Compute CRPS using the discrete formula
                // CRPS = Σ_{k=0}^{∞} (F(k) - 1{y ≤ k})²
                // We truncate the sum when probability mass becomes negligible

                let mut crps = 0.0;

                // Determine range to sum over (cover most of the probability mass)
                // Use mean ± 6*std for Poisson (covers >99.99% of mass)
                let std_dev = lambda.sqrt();
                let k_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_i + 10);

                for k in 0..=k_max {
                    let f_k = d.cdf(k as u64);
                    let indicator = if y_i <= k { 1.0 } else { 0.0 };
                    let diff = f_k - indicator;
                    crps += diff * diff;
                }

                scores[i] = crps;
            } else {
                scores[i] = f64::MAX;
            }
        }
        scores
    }

    fn d_score(&self, y: &Array1<f64>) -> Array2<f64> {
        // Gradient of discrete CRPS w.r.t. log(rate)
        // Using numerical differentiation or analytical form
        //
        // d(CRPS)/d(log(λ)) = λ * d(CRPS)/d(λ)
        //
        // For Poisson CRPS, the gradient involves:
        // d(F(k))/d(λ) = -f(k) + f(k-1) for k >= 1, and -f(0) for k=0
        // where f is the PMF

        let n_obs = y.len();
        let mut d_params = Array2::zeros((n_obs, 1));

        for i in 0..n_obs {
            let lambda = self.rate[i];
            let y_i = y[i].round() as i64;

            if let Ok(d) = PoissonDist::new(lambda) {
                let mut d_crps = 0.0;

                let std_dev = lambda.sqrt();
                let k_max = ((lambda + 6.0 * std_dev).ceil() as i64).max(y_i + 10);

                for k in 0..=k_max {
                    let f_k = d.cdf(k as u64);
                    let indicator = if y_i <= k { 1.0 } else { 0.0 };
                    let diff = f_k - indicator;

                    // d(F(k))/d(λ) for Poisson:
                    // F(k) = Γ(k+1, λ) / k! = Q(k+1, λ) (regularized incomplete gamma)
                    // d(F(k))/d(λ) = -f(k) where f(k) is the PMF
                    let pmf_k = d.pmf(k as u64);
                    let df_dlambda = -pmf_k;

                    d_crps += 2.0 * diff * df_dlambda;
                }

                // Convert to d/d(log(λ))
                d_params[[i, 0]] = lambda * d_crps;
            }
        }

        d_params
    }

    fn metric(&self) -> Array3<f64> {
        // Metric for discrete CRPS
        // Using an approximation based on the Poisson variance structure
        let n_obs = self.rate.len();
        let mut fi = Array3::zeros((n_obs, 1, 1));

        // For Poisson CRPS, the metric scales with rate
        // Approximation: metric ≈ 2 * sqrt(rate) / sqrt(pi)
        for i in 0..n_obs {
            let lambda = self.rate[i];
            fi[[i, 0, 0]] = 2.0 * lambda.sqrt() / std::f64::consts::PI.sqrt();
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
}
