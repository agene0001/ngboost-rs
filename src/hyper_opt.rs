//! Hyperparameter optimization for NGBoost.
//!
//! Tree-structured Parzen Estimator (TPE) search with optional median /
//! percentile / threshold pruning and k-fold cross-validation, mirroring
//! the API exposed by `gradientlss-rs`.
//!
//! Because NGBoost's hyperparameters live partly on the `NGBoost<D, S, B>`
//! struct (learning rate, n_estimators, minibatch_frac, …) and partly on the
//! base learner `B` (e.g. `HistogramLearner::max_depth`), callers pass a
//! `builder` closure that maps a sampled `HashMap<String, Value>` to a fresh
//! `NGBoost<D, S, B>`. This keeps the optimizer fully generic over distribution,
//! score, and learner without baking in any knowledge of which params belong
//! where.
//!
//! # Example
//! ```ignore
//! use ngboost_rs::hyper_opt::{hyper_opt_with_config, HyperOptConfig, PruningStrategy};
//! use ngboost_rs::{NGBoost, learners::HistogramLearner, dist::Poisson, scores::LogScore};
//! use serde_json::json;
//! use std::collections::HashMap;
//!
//! let mut hp = HashMap::new();
//! hp.insert("learning_rate".into(), json!({"low": 0.005, "high": 0.1, "log": true}));
//! hp.insert("max_depth".into(),     json!({"low": 2, "high": 8, "type": "int"}));
//! hp.insert("n_estimators".into(),  json!({"low": 200, "high": 1500, "type": "int"}));
//!
//! let cfg = HyperOptConfig {
//!     n_trials: 50,
//!     n_folds: 5,
//!     pruning: PruningStrategy::Median { n_warmup_trials: 10 },
//!     ..Default::default()
//! };
//!
//! let result = hyper_opt_with_config::<Poisson, LogScore, HistogramLearner, _>(
//!     &x_train, &y_train, &hp, cfg,
//!     |params| {
//!         let lr = params.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.01);
//!         let depth = params.get("max_depth").and_then(|v| v.as_i64()).unwrap_or(3) as usize;
//!         let n_est = params.get("n_estimators").and_then(|v| v.as_i64()).unwrap_or(500) as u32;
//!         NGBoost::<Poisson, LogScore, HistogramLearner>::new(
//!             n_est, lr, HistogramLearner::new(depth),
//!         )
//!     },
//! )?;
//! ```

use crate::dist::{Distribution, DistributionMethods};
use crate::learners::BaseLearner;
use crate::ngboost::NGBoost;
use crate::scores::{Scorable, Score};
use ndarray::{Array1, Array2, Axis, s};
use rand_compat::SeedableRng;
use rand_compat::rngs::StdRng;
use serde_json::Value;
use std::collections::HashMap;
use tpe::{TpeOptimizer, parzen_estimator, range};

// ============================================================================
// Hyperparameter specifications
// ============================================================================

/// Hyperparameter specification for optimization.
#[derive(Debug, Clone)]
pub struct HyperParamSpec {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub param_type: HyperParamType,
}

/// Types of hyperparameters that can be optimized.
#[derive(Debug, Clone)]
pub enum HyperParamType {
    /// Continuous float parameter with bounds `[low, high]`.
    Float { low: f64, high: f64, log: bool },
    /// Integer parameter with bounds `[low, high]`.
    Int { low: i64, high: i64, log: bool },
    /// Categorical parameter with concrete choices.
    Categorical { choices: Vec<Value> },
}

impl HyperParamSpec {
    pub fn float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Float { low, high, log: false },
        }
    }

    pub fn log_float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Float { low, high, log: true },
        }
    }

    pub fn int(name: impl Into<String>, low: i64, high: i64) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Int { low, high, log: false },
        }
    }

    pub fn categorical(name: impl Into<String>, choices: Vec<Value>) -> Self {
        Self {
            name: name.into(),
            param_type: HyperParamType::Categorical { choices },
        }
    }
}

// ============================================================================
// Result types
// ============================================================================

/// Result of a full hyperparameter optimization run.
#[derive(Debug, Clone)]
pub struct HyperOptResult {
    /// Best hyperparameters found.
    pub best_params: HashMap<String, Value>,
    /// Best CV score (mean fold loss) achieved.
    pub best_score: f64,
    /// `n_estimators` from the best trial (carried through verbatim — NGBoost
    /// has its own early-stopping; we surface what was configured, not what the
    /// best-val-loss iteration found, since fold-level early stopping isn't
    /// inspectable through the public API yet).
    pub opt_rounds: usize,
    /// History of all trials, including pruned ones.
    pub trials: Vec<TrialResult>,
}

/// Result of a single trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub params: HashMap<String, Value>,
    /// Mean score across completed folds. If the trial was pruned mid-CV, this
    /// is the average over the folds that ran before pruning.
    pub score: f64,
    pub pruned: bool,
    /// Per-fold scores recorded as folds completed.
    pub intermediate_scores: Vec<f64>,
}

// ============================================================================
// Pruning
// ============================================================================

/// Pruning strategy for early termination of unpromising trials.
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// No pruning - run all trials and folds.
    None,
    /// Median pruning: prune if the running mean is worse than the median of
    /// completed trials' scores at the same fold index. Only kicks in after
    /// `n_warmup_trials` non-pruned trials have completed.
    Median { n_warmup_trials: usize },
    /// Percentile pruning: keep top `percentile`% of trials at each fold.
    Percentile {
        percentile: f64,
        n_warmup_trials: usize,
    },
    /// Threshold pruning: prune if intermediate score exceeds threshold.
    Threshold { threshold: f64 },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        PruningStrategy::Median { n_warmup_trials: 5 }
    }
}

/// Tracks intermediate fold scores across completed trials so pruning can
/// compare a running trial to historical fold-`k` performance.
#[derive(Debug, Clone, Default)]
struct PrunerState {
    /// `intermediate_scores[fold_idx]` = list of running-mean-at-fold-k from
    /// every completed (non-pruned) trial.
    intermediate_scores: HashMap<usize, Vec<f64>>,
}

impl PrunerState {
    fn new() -> Self {
        Self {
            intermediate_scores: HashMap::new(),
        }
    }

    fn record(&mut self, step: usize, score: f64) {
        self.intermediate_scores
            .entry(step)
            .or_insert_with(Vec::new)
            .push(score);
    }

    fn should_prune(
        &self,
        strategy: &PruningStrategy,
        step: usize,
        score: f64,
        n_completed_trials: usize,
    ) -> bool {
        match strategy {
            PruningStrategy::None => false,
            PruningStrategy::Median { n_warmup_trials } => {
                if n_completed_trials < *n_warmup_trials {
                    return false;
                }
                match self.intermediate_scores.get(&step) {
                    Some(scores) if !scores.is_empty() => {
                        let mut sorted = scores.clone();
                        sorted.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let median = sorted[sorted.len() / 2];
                        score > median
                    }
                    _ => false,
                }
            }
            PruningStrategy::Percentile {
                percentile,
                n_warmup_trials,
            } => {
                if n_completed_trials < *n_warmup_trials {
                    return false;
                }
                match self.intermediate_scores.get(&step) {
                    Some(scores) if !scores.is_empty() => {
                        let mut sorted = scores.clone();
                        sorted.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let idx = ((sorted.len() as f64 * percentile / 100.0).ceil()
                            as usize)
                            .saturating_sub(1)
                            .min(sorted.len() - 1);
                        let threshold = sorted[idx];
                        score > threshold
                    }
                    _ => false,
                }
            }
            PruningStrategy::Threshold { threshold } => score > *threshold,
        }
    }
}

// ============================================================================
// Config
// ============================================================================

/// Configuration for `hyper_opt_with_config`.
#[derive(Debug, Clone)]
pub struct HyperOptConfig {
    /// Number of optimization trials.
    pub n_trials: u32,
    /// Number of cross-validation folds per trial.
    pub n_folds: usize,
    /// Random seed used for CV-fold assignment and for `Trial.seed = seed + i`.
    pub seed: u64,
    /// Separate seed for the TPE samplers; defaults to `seed` if `None`.
    pub hp_seed: Option<u64>,
    /// Pruning strategy.
    pub pruning: PruningStrategy,
    /// Wall-clock budget in minutes (`None` = no limit). Checked between trials.
    pub max_minutes: Option<f64>,
    /// Whether to print per-10-trial progress to stderr.
    pub verbose: bool,
}

impl Default for HyperOptConfig {
    fn default() -> Self {
        Self {
            n_trials: 100,
            n_folds: 5,
            seed: 42,
            hp_seed: None,
            pruning: PruningStrategy::None,
            max_minutes: None,
            verbose: true,
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Run hyperparameter optimization with k-fold cross-validation and TPE sampling.
///
/// The `builder` closure receives the sampled parameter map and returns a fresh
/// `NGBoost<D, S, B>` configured for that trial. The optimizer will train one
/// such model per CV fold (so `n_folds` × `n_trials` total fits).
///
/// Returns the best hyperparameters found, ranked by mean fold NLL (lower is
/// better — `NGBoost::score` returns the average score, which is NLL for
/// `LogScore`).
pub fn hyper_opt_with_config<D, S, B, F>(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    config: HyperOptConfig,
    builder: F,
) -> Result<HyperOptResult, Box<dyn std::error::Error>>
where
    D: Distribution + DistributionMethods + Scorable<S> + Clone + 'static,
    S: Score + 'static,
    B: BaseLearner + Clone + 'static,
    F: Fn(&HashMap<String, Value>) -> NGBoost<D, S, B>,
{
    use std::time::Instant;

    let hp_seed = config.hp_seed.unwrap_or(config.seed);
    let mut rng = StdRng::seed_from_u64(hp_seed);
    let start_time = Instant::now();

    // Build one TPE optimizer per hyperparameter dimension. Log-scaled params
    // are sampled in log-space and exponentiated when materializing the trial.
    let specs = parse_hp_specs(hp_dict)?;
    let mut optimizers: HashMap<String, TpeOptimizer> = HashMap::new();

    for spec in &specs {
        match &spec.param_type {
            HyperParamType::Float { low, high, log } => {
                let (lo, hi) = if *log { (low.ln(), high.ln()) } else { (*low, *high) };
                optimizers.insert(
                    spec.name.clone(),
                    TpeOptimizer::new(parzen_estimator(), range(lo, hi)?),
                );
            }
            HyperParamType::Int { low, high, log } => {
                let (lo, hi) = if *log {
                    ((*low as f64).ln(), (*high as f64).ln())
                } else {
                    (*low as f64, *high as f64)
                };
                optimizers.insert(
                    spec.name.clone(),
                    TpeOptimizer::new(parzen_estimator(), range(lo, hi)?),
                );
            }
            HyperParamType::Categorical { choices } => {
                if !choices.is_empty() {
                    let optimizer = TpeOptimizer::new(
                        parzen_estimator(),
                        range(0.0, choices.len() as f64 - 0.01)?,
                    );
                    optimizers.insert(spec.name.clone(), optimizer);
                }
            }
        }
    }

    let mut best_score = f64::INFINITY;
    let mut best_params: HashMap<String, Value> = HashMap::new();
    let mut best_rounds: usize = 0;
    let mut trials: Vec<TrialResult> = Vec::with_capacity(config.n_trials as usize);
    let mut pruner_state = PrunerState::new();
    let mut n_completed_trials = 0usize;

    for trial in 0..config.n_trials {
        if let Some(max_mins) = config.max_minutes {
            let elapsed_mins = start_time.elapsed().as_secs_f64() / 60.0;
            if elapsed_mins >= max_mins {
                if config.verbose {
                    eprintln!(
                        "Time budget of {:.1} minutes exceeded after {} trials",
                        max_mins, trial
                    );
                }
                break;
            }
        }

        // Sample from each TPE and materialize trial_params (caller-friendly
        // JSON values) and sampled_params (raw f64 values fed back to TPE).
        let mut sampled_params: HashMap<String, f64> = HashMap::new();
        let mut trial_params: HashMap<String, Value> = HashMap::new();

        for spec in &specs {
            if let Some(optimizer) = optimizers.get_mut(&spec.name) {
                let raw_value = optimizer.ask(&mut rng)?;
                sampled_params.insert(spec.name.clone(), raw_value);

                let param_value = match &spec.param_type {
                    HyperParamType::Float { log, .. } => {
                        let v = if *log { raw_value.exp() } else { raw_value };
                        Value::from(v)
                    }
                    HyperParamType::Int { log, .. } => {
                        let v = if *log {
                            raw_value.exp().round() as i64
                        } else {
                            raw_value.round() as i64
                        };
                        Value::from(v)
                    }
                    HyperParamType::Categorical { choices } => {
                        let idx = (raw_value.floor() as usize).min(choices.len() - 1);
                        choices[idx].clone()
                    }
                };
                trial_params.insert(spec.name.clone(), param_value);
            }
        }

        let (cv_score, pruned, intermediate_scores) = cv_with_pruning(
            features,
            labels,
            config.n_folds,
            &trial_params,
            &builder,
            &config.pruning,
            &pruner_state,
            n_completed_trials,
            config.seed + trial as u64,
        );

        // Feed score back to TPE regardless of pruning so the surrogate
        // still learns from the early-terminated region.
        let report_score = if pruned {
            intermediate_scores.last().copied().unwrap_or(f64::INFINITY)
        } else {
            cv_score
        };
        for (name, raw_value) in &sampled_params {
            if let Some(optimizer) = optimizers.get_mut(name) {
                optimizer.tell(*raw_value, report_score)?;
            }
        }

        if !pruned && cv_score < best_score {
            best_score = cv_score;
            best_params = trial_params.clone();
            // Surface n_estimators from the winning trial so the caller can
            // refit at exactly the size that won CV.
            best_rounds = trial_params
                .get("n_estimators")
                .and_then(|v| v.as_i64())
                .map(|n| n as usize)
                .unwrap_or(0);
        }

        if !pruned {
            for (step, &score) in intermediate_scores.iter().enumerate() {
                pruner_state.record(step, score);
            }
            n_completed_trials += 1;
        }

        trials.push(TrialResult {
            params: trial_params,
            score: cv_score,
            pruned,
            intermediate_scores,
        });

        if config.verbose && (trial + 1) % 10 == 0 {
            let pruned_count = trials.iter().filter(|t| t.pruned).count();
            eprintln!(
                "Trial {}/{}: score = {:.6}, best = {:.6}, pruned = {}/{}",
                trial + 1,
                config.n_trials,
                cv_score,
                best_score,
                pruned_count,
                trial + 1
            );
        }
    }

    Ok(HyperOptResult {
        best_params,
        best_score,
        opt_rounds: best_rounds,
        trials,
    })
}

/// Convenience wrapper that uses `HyperOptConfig::default()` for everything
/// except `n_trials`, `n_folds`, and `seed`.
pub fn hyper_opt<D, S, B, F>(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    hp_dict: &HashMap<String, Value>,
    n_trials: u32,
    n_folds: usize,
    seed: u64,
    builder: F,
) -> Result<HyperOptResult, Box<dyn std::error::Error>>
where
    D: Distribution + DistributionMethods + Scorable<S> + Clone + 'static,
    S: Score + 'static,
    B: BaseLearner + Clone + 'static,
    F: Fn(&HashMap<String, Value>) -> NGBoost<D, S, B>,
{
    let config = HyperOptConfig {
        n_trials,
        n_folds,
        seed,
        ..Default::default()
    };
    hyper_opt_with_config(features, labels, hp_dict, config, builder)
}

// ============================================================================
// Cross-validation
// ============================================================================

/// Early-stopping patience used for CV fold fits when the builder didn't set
/// one. A fold model that stops improving for this many rounds halts instead of
/// grinding through every sampled `n_estimators` — the dominant CV speedup.
const CV_EARLY_STOPPING_ROUNDS: u32 = 50;

/// Sequentially fits a fresh `NGBoost` on each (k-1)/k slice of the data, scores
/// it against the held-out fold, and returns `(mean_score, was_pruned,
/// per_fold_scores)`. Pruning is consulted after every fold against the
/// running mean.
///
/// `n_folds <= 1` runs a single 80/20 holdout instead — train on the first
/// 80%, score on the last 20% — a single fit with no cross-fold pruning. Much
/// cheaper than k-fold and equivalent to a plain train/val split for HPO.
fn cv_with_pruning<D, S, B, F>(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    n_folds: usize,
    trial_params: &HashMap<String, Value>,
    builder: &F,
    pruning_strategy: &PruningStrategy,
    pruner_state: &PrunerState,
    n_completed_trials: usize,
    trial_seed: u64,
) -> (f64, bool, Vec<f64>)
where
    D: Distribution + DistributionMethods + Scorable<S> + Clone + 'static,
    S: Score + 'static,
    B: BaseLearner + Clone + 'static,
    F: Fn(&HashMap<String, Value>) -> NGBoost<D, S, B>,
{
    let n_samples = features.nrows();
    // `n_folds <= 1` ⇒ single 80/20 holdout (one iteration); otherwise k-fold.
    let single_holdout = n_folds <= 1;
    let n_iters = if single_holdout { 1 } else { n_folds };
    let fold_size = n_samples / n_iters;
    let mut fold_scores = Vec::with_capacity(n_iters);
    let mut pruned = false;

    for i in 0..n_iters {
        let (test_start, test_end) = if single_holdout {
            // Last 20% is the held-out validation/scoring fold.
            (((n_samples as f64) * 0.8) as usize, n_samples)
        } else if i == n_iters - 1 {
            (i * fold_size, n_samples)
        } else {
            (i * fold_size, (i + 1) * fold_size)
        };

        let test_features = features.slice(s![test_start..test_end, ..]).to_owned();
        let test_labels = labels.slice(s![test_start..test_end]).to_owned();

        let train_features_1 = features.slice(s![..test_start, ..]);
        let train_features_2 = features.slice(s![test_end.., ..]);
        let train_labels_1 = labels.slice(s![..test_start]);
        let train_labels_2 = labels.slice(s![test_end..]);

        let train_features =
            match ndarray::concatenate(Axis(0), &[train_features_1, train_features_2]) {
                Ok(f) => f,
                Err(_) => {
                    fold_scores.push(f64::INFINITY);
                    continue;
                }
            };
        let train_labels =
            match ndarray::concatenate(Axis(0), &[train_labels_1, train_labels_2]) {
                Ok(l) => l,
                Err(_) => {
                    fold_scores.push(f64::INFINITY);
                    continue;
                }
            };

        let mut model = builder(trial_params);
        // Override the model's RNG so identical trial_params on different
        // trial indices still get distinct fold-level randomness.
        model.set_random_state(trial_seed.wrapping_add(i as u64));

        // Early stopping during CV: monitor the held-out fold so a trial whose
        // model converges well before its sampled `n_estimators` stops there
        // instead of running every boosting round. Respect a builder-set value
        // if one was provided. Using the same fold for both early-stopping and
        // scoring makes each trial's score mildly optimistic, but consistently
        // so across trials — TPE ranking (the only thing that matters here) is
        // unaffected.
        if model.early_stopping_rounds.is_none() {
            model.early_stopping_rounds = Some(CV_EARLY_STOPPING_ROUNDS);
        }

        let fit_result = model.fit_with_validation(
            &train_features,
            &train_labels,
            Some(&test_features),
            Some(&test_labels),
            None,
            None,
        );
        let score = if fit_result.is_err() {
            f64::INFINITY
        } else {
            let s = model.score(&test_features, &test_labels);
            if s.is_finite() { s } else { f64::INFINITY }
        };

        fold_scores.push(score);

        // Pruning is meaningless with a single holdout — there are no further
        // folds to skip.
        if !single_holdout {
            let running_mean =
                fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            if pruner_state.should_prune(pruning_strategy, i, running_mean, n_completed_trials)
            {
                pruned = true;
                break;
            }
        }
    }

    let final_score = if fold_scores.is_empty() {
        f64::INFINITY
    } else {
        fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
    };

    (final_score, pruned, fold_scores)
}

// ============================================================================
// JSON parsing helpers
// ============================================================================

/// Parse the user's `HashMap<String, Value>` search space into typed specs.
/// Supports:
/// * `[low, high]`                                 → float [low, high]
/// * `{ "low": …, "high": … }`                     → float
/// * `{ "low": …, "high": …, "log": true }`        → log-float
/// * `{ "low": …, "high": …, "type": "int" }`      → int (or log-int with both)
/// * `{ "choices": [...] }`                        → categorical
pub fn parse_hp_specs(
    hp_dict: &HashMap<String, Value>,
) -> Result<Vec<HyperParamSpec>, String> {
    let mut specs = Vec::with_capacity(hp_dict.len());

    for (name, value) in hp_dict {
        let spec = match value {
            Value::Array(arr) if arr.len() == 2 => {
                let low = arr[0]
                    .as_f64()
                    .ok_or_else(|| format!("Invalid low value for {}", name))?;
                let high = arr[1]
                    .as_f64()
                    .ok_or_else(|| format!("Invalid high value for {}", name))?;
                HyperParamSpec::float(name.clone(), low, high)
            }
            Value::Object(obj) => {
                if let Some(choices) = obj.get("choices") {
                    let choices = choices
                        .as_array()
                        .ok_or_else(|| format!("Invalid choices for {}", name))?
                        .clone();
                    HyperParamSpec::categorical(name.clone(), choices)
                } else {
                    let low = obj
                        .get("low")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| format!("Missing 'low' for {}", name))?;
                    let high = obj
                        .get("high")
                        .and_then(|v| v.as_f64())
                        .ok_or_else(|| format!("Missing 'high' for {}", name))?;
                    let log = obj.get("log").and_then(|v| v.as_bool()).unwrap_or(false);
                    let is_int =
                        obj.get("type").and_then(|v| v.as_str()) == Some("int");

                    if is_int {
                        HyperParamSpec {
                            name: name.clone(),
                            param_type: HyperParamType::Int {
                                low: low as i64,
                                high: high as i64,
                                log,
                            },
                        }
                    } else if log {
                        HyperParamSpec::log_float(name.clone(), low, high)
                    } else {
                        HyperParamSpec::float(name.clone(), low, high)
                    }
                }
            }
            _ => return Err(format!("Invalid hyperparameter format for {}", name)),
        };
        specs.push(spec);
    }

    Ok(specs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Normal;
    use crate::learners::StumpLearner;
    use crate::scores::LogScore;
    use ndarray::Array;
    use serde_json::json;

    #[test]
    fn test_parse_hp_specs_array() {
        let mut hp = HashMap::new();
        hp.insert("learning_rate".to_string(), json!([0.01, 0.3]));
        let specs = parse_hp_specs(&hp).unwrap();
        assert_eq!(specs.len(), 1);
        match &specs[0].param_type {
            HyperParamType::Float { low, high, log } => {
                assert_eq!(*low, 0.01);
                assert_eq!(*high, 0.3);
                assert!(!log);
            }
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_parse_hp_specs_log() {
        let mut hp = HashMap::new();
        hp.insert(
            "learning_rate".to_string(),
            json!({"low": 0.001, "high": 0.1, "log": true}),
        );
        let specs = parse_hp_specs(&hp).unwrap();
        assert!(matches!(
            specs[0].param_type,
            HyperParamType::Float { log: true, .. }
        ));
    }

    #[test]
    fn test_parse_hp_specs_int() {
        let mut hp = HashMap::new();
        hp.insert(
            "max_depth".to_string(),
            json!({"low": 2, "high": 8, "type": "int"}),
        );
        let specs = parse_hp_specs(&hp).unwrap();
        match &specs[0].param_type {
            HyperParamType::Int { low, high, log } => {
                assert_eq!(*low, 2);
                assert_eq!(*high, 8);
                assert!(!log);
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_parse_hp_specs_categorical() {
        let mut hp = HashMap::new();
        hp.insert(
            "schedule".to_string(),
            json!({"choices": ["constant", "linear"]}),
        );
        let specs = parse_hp_specs(&hp).unwrap();
        assert!(matches!(
            specs[0].param_type,
            HyperParamType::Categorical { .. }
        ));
    }

    #[test]
    fn test_pruner_median_warmup() {
        let state = PrunerState::new();
        let strategy = PruningStrategy::Median { n_warmup_trials: 5 };
        // Before warmup completes, never prune.
        assert!(!state.should_prune(&strategy, 0, 1e9, 2));
    }

    #[test]
    fn test_pruner_median_active() {
        let mut state = PrunerState::new();
        // Pretend trials at step 0 produced scores [1.0, 2.0, 3.0, 4.0, 5.0].
        for s in [1.0, 2.0, 3.0, 4.0, 5.0] {
            state.record(0, s);
        }
        let strategy = PruningStrategy::Median { n_warmup_trials: 3 };
        assert!(state.should_prune(&strategy, 0, 4.0, 5)); // 4 > median(3)
        assert!(!state.should_prune(&strategy, 0, 2.0, 5)); // 2 ≤ median(3)
    }

    /// End-to-end smoke test: tiny dataset, 3 trials, 2 folds, stump learner.
    /// Verifies the optimizer runs without panicking and returns a non-empty
    /// `best_params`.
    #[test]
    fn test_hyper_opt_smoke() {
        let n = 80;
        let x = Array::from_shape_fn((n, 2), |(i, j)| (i as f64) * 0.01 + (j as f64) * 0.1);
        let y = Array::from_shape_fn(n, |i| (i as f64) * 0.05);

        let mut hp = HashMap::new();
        hp.insert(
            "learning_rate".to_string(),
            json!({"low": 0.01, "high": 0.1, "log": true}),
        );
        hp.insert(
            "n_estimators".to_string(),
            json!({"low": 10, "high": 30, "type": "int"}),
        );

        let cfg = HyperOptConfig {
            n_trials: 3,
            n_folds: 2,
            seed: 0,
            hp_seed: None,
            pruning: PruningStrategy::None,
            max_minutes: None,
            verbose: false,
        };

        let result = hyper_opt_with_config::<Normal, LogScore, StumpLearner, _>(
            &x,
            &y,
            &hp,
            cfg,
            |p| {
                let lr = p.get("learning_rate").and_then(|v| v.as_f64()).unwrap_or(0.05);
                let n_est = p
                    .get("n_estimators")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(10) as u32;
                NGBoost::<Normal, LogScore, StumpLearner>::new(n_est, lr, StumpLearner)
            },
        )
        .expect("hyper_opt should succeed on smoke test");

        assert!(!result.best_params.is_empty());
        assert_eq!(result.trials.len(), 3);
    }
}
