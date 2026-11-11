//! `NGBAutoMixtureRegressor`: choose the number of Normal-mixture components
//! by validation NLL (research/ngboost Cycle 11).
//!
//! Fits a single-Normal head (K = 1) and `NormalMixture<K>` heads for each
//! candidate K on the same training and validation data — each with early
//! stopping and the default scale recalibration on that validation set — and
//! keeps the head with the lowest validation NLL. Measured over 7 datasets ×
//! 10 seeds (K ∈ {1, 2, 3, 4}): within noise of the best fixed K everywhere
//! (worst +0.006 ± 0.006 NLL), better than every fixed K on unimodal data
//! (−0.007 … −0.044), and within 0.001–0.015 of a test-set oracle. The same
//! validation set drives early stopping, recalibration and selection, so its
//! NLL is mildly optimistic; use a test set for reporting.
//!
//! `search` adds warm-grown candidates (each K grown from the K − 1 fit;
//! default `Both` = cold and warm families selected jointly) and `greedy`
//! stops each family at the first K that does not improve validation NLL
//! (research/ngboost Cycle 12).
//!
//! The training rule is a type parameter: `NGBAutoMixtureRegressor` (=
//! `<LogScore>`, `new`) or `NGBAutoMixtureRegressor<CRPScore>`
//! (`new_crps`). Candidates are compared by the validation value of the
//! training rule (NLL or CRPS).
use crate::dist::{DistributionMethods, Normal, NormalMixture};
use crate::learners::{HistTreeNode, HistogramLearner, TrainedBaseLearner, TrainedHistogramTree};
use crate::ngboost::{LearningRateSchedule, LineSearchMethod, NGBoost};
use crate::scores::{CRPScore, LogScore, Scorable, Score};
use ndarray::{Array1, Array2};

/// The selected head.
pub enum AutoMixtureModel<S: Score = LogScore>
where
    Normal: Scorable<S>,
    NormalMixture<2>: Scorable<S>,
    NormalMixture<3>: Scorable<S>,
    NormalMixture<4>: Scorable<S>,
{
    Normal(NGBoost<Normal, S, HistogramLearner>),
    K2(NGBoost<NormalMixture<2>, S, HistogramLearner>),
    K3(NGBoost<NormalMixture<3>, S, HistogramLearner>),
    K4(NGBoost<NormalMixture<4>, S, HistogramLearner>),
}

macro_rules! dispatch {
    ($model:expr, $m:ident => $body:expr) => {
        match $model {
            AutoMixtureModel::Normal($m) => $body,
            AutoMixtureModel::K2($m) => $body,
            AutoMixtureModel::K3($m) => $body,
            AutoMixtureModel::K4($m) => $body,
        }
    };
}

/// Normal-mixture regressor that picks its number of components (1–4) by
/// validation NLL. See the module docs for the evidence.
pub struct NGBAutoMixtureRegressor<S: Score = LogScore>
where
    Normal: Scorable<S>,
    NormalMixture<2>: Scorable<S>,
    NormalMixture<3>: Scorable<S>,
    NormalMixture<4>: Scorable<S>,
{
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub base_learner: HistogramLearner,
    /// Early stopping on the validation set for every candidate (default 50).
    pub early_stopping_rounds: Option<u32>,
    /// Candidate component counts, each in 1..=4 (default `[1, 2, 3, 4]`).
    pub candidates: Vec<usize>,
    pub random_state: Option<u64>,
    /// Candidate families (default `Both`; see [`KSearch`]). `Warm` and
    /// `Both` require `candidates` to be 1..=max.
    pub search: KSearch,
    /// How warm candidates add the new component (default `Split`).
    pub grow_strategy: GrowStrategy,
    /// `Split` growth: the two halves start at μ ∓ `split_delta`·σ̄ with σ ×
    /// `split_shrink` (defaults 0.8, 0.6). `split_delta = 0, split_shrink =
    /// 1` gives two identical halves — the source model reparametrised,
    /// whose continued boosting equals continuing the source (a control).
    pub split_delta: f64,
    pub split_shrink: f64,
    /// Stop increasing K once a candidate does not improve validation NLL
    /// over the previous one (candidates are tried in ascending order).
    pub greedy: bool,
    /// (K, warm?, validation score, boosting rounds trained) for every
    /// candidate fitted by the last `fit`, in fitting order. For warm
    /// candidates the rounds are those added on top of K − 1.
    pub tried: Vec<(usize, bool, f64, usize)>,
    model: Option<AutoMixtureModel<S>>,
    selected_warm: bool,
}

/// Which candidate families `NGBAutoMixtureRegressor::fit` tries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KSearch {
    /// Every K fitted from scratch (EM-initialised mixtures).
    Cold,
    /// K = 1 from scratch, each larger K grown from the previous fit
    /// (`grow_strategy`). ≈ 46 % of the cold cost, but a different model: it
    /// keeps the Normal head's conditional mean — better on unimodal targets,
    /// much worse on multimodal ones (research/ngboost R42).
    Warm,
    /// Both families (K = 1 shared), selected jointly by validation NLL.
    Both,
}

/// How a K-component fit is grown from the (K−1)-component one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowStrategy {
    /// Split the source's reference (last) component into two copies at
    /// μ ∓ `SPLIT_DELTA`·σ̄ with σ × `SPLIT_SHRINK`, each taking half its
    /// weight (split-and-merge EM: Ueda, Nakano, Ghahramani & Hinton, Neural
    /// Computation 2000). For 1 → 2 this splits the single Normal.
    Split,
    /// Add a new component (weight `GROW_WEIGHT`) at the training rows the
    /// source fits worst (greedy mixture learning: Verbeek et al. 2003).
    Misfit,
}

const SPLIT_DELTA: f64 = 0.8;
const SPLIT_SHRINK: f64 = 0.6;

/// Weight given to a new component when growing K − 1 → K.
const GROW_WEIGHT: f64 = 0.05;
/// Fraction of worst-fitted training rows that place the new component.
const GROW_MISFIT_FRAC: f64 = 0.10;

/// Fitted-ensemble pieces of a source model (any head) for `grow`.
struct Parts<'a> {
    init: &'a Array1<f64>,
    models: &'a [Vec<Box<dyn TrainedBaseLearner>>],
    scalings: &'a [f64],
    lrs: &'a [f64],
    cols: &'a [Vec<usize>],
}

fn clone_learner(l: &dyn TrainedBaseLearner) -> Result<Box<dyn TrainedBaseLearner>, &'static str> {
    Ok(l.to_serializable()
        .ok_or("warm start needs serializable base learners")?
        .to_trait_object())
}

impl NGBAutoMixtureRegressor<LogScore> {
    /// LogScore-trained candidates, selected by validation NLL.
    pub fn new(n_estimators: u32, learning_rate: f64) -> Self {
        Self::with_score(n_estimators, learning_rate)
    }
}

impl NGBAutoMixtureRegressor<CRPScore> {
    /// CRPScore-trained candidates, selected by validation CRPS.
    pub fn new_crps(n_estimators: u32, learning_rate: f64) -> Self {
        Self::with_score(n_estimators, learning_rate)
    }
}

impl<S: Score> NGBAutoMixtureRegressor<S>
where
    Normal: Scorable<S>,
    NormalMixture<2>: Scorable<S>,
    NormalMixture<3>: Scorable<S>,
    NormalMixture<4>: Scorable<S>,
{
    fn with_score(n_estimators: u32, learning_rate: f64) -> Self {
        NGBAutoMixtureRegressor {
            n_estimators,
            learning_rate,
            base_learner: HistogramLearner::new(3),
            early_stopping_rounds: Some(50),
            candidates: vec![1, 2, 3, 4],
            random_state: None,
            search: KSearch::Both,
            grow_strategy: GrowStrategy::Split,
            split_delta: SPLIT_DELTA,
            split_shrink: SPLIT_SHRINK,
            greedy: false,
            tried: Vec::new(),
            model: None,
            selected_warm: false,
        }
    }

    fn build<D>(&self) -> NGBoost<D, S, HistogramLearner>
    where
        D: crate::dist::Distribution + Scorable<S> + Clone,
    {
        NGBoost::with_full_options(
            self.n_estimators,
            self.learning_rate,
            self.base_learner.clone(),
            true,
            1.0,
            1.0,
            false,
            1.0,
            1e-4,
            self.early_stopping_rounds,
            0.1,
            LearningRateSchedule::Constant,
            0.0,
            LineSearchMethod::Binary,
            self.random_state,
        )
    }

    /// Build a K1-component mixture from the (K1−1)-component source and
    /// continue boosting it. Every column of the grown model is either a
    /// round-for-round copy of a source column's trees plus a constant shift,
    /// or zero trees plus a constant, so no new model format is needed; see
    /// [`GrowStrategy`] for where the new component starts. The new
    /// component is the softmax reference of the grown model.
    #[allow(clippy::too_many_arguments)]
    fn grow<const K1: usize>(
        &self,
        src: Parts,
        k_src: usize,
        src_dist: (&Array1<f64>, &Array1<f64>),
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<NGBoost<NormalMixture<K1>, S, HistogramLearner>, &'static str>
    where
        NormalMixture<K1>: Scorable<S>,
    {
        assert_eq!(K1, k_src + 1);
        let (src_logpdf, src_ref_sigma) = src_dist;
        let n_new = 3 * K1 - 1;
        let reference = k_src - 1; // source reference component
        // source columns of component k (μ, log σ)
        let src_mu = |k: usize| if k_src == 1 { 0 } else { (k_src - 1) + 2 * k };
        // per-new-column (source column, constant shift); None = zero trees
        let mut map: Vec<(Option<usize>, f64)> = vec![(None, 0.0); n_new];
        match self.grow_strategy {
            GrowStrategy::Split => {
                let ln2 = std::f64::consts::LN_2;
                let sbar = src_ref_sigma.mean().unwrap_or(1.0);
                for j in 0..K1 - 1 {
                    // other components: l'_j = l_j + ln 2; the split half that
                    // keeps the reference's index: l' = 0 (equal to the new half)
                    map[j] = if j + 1 < k_src {
                        (Some(j), ln2)
                    } else {
                        (None, 0.0)
                    };
                }
                for k in 0..k_src {
                    let (dmu, dls) = if k == reference {
                        (-self.split_delta * sbar, self.split_shrink.ln())
                    } else {
                        (0.0, 0.0)
                    };
                    map[(K1 - 1) + 2 * k] = (Some(src_mu(k)), dmu);
                    map[(K1 - 1) + 2 * k + 1] = (Some(src_mu(k) + 1), dls);
                }
                map[(K1 - 1) + 2 * k_src] = (Some(src_mu(reference)), self.split_delta * sbar);
                map[(K1 - 1) + 2 * k_src + 1] =
                    (Some(src_mu(reference) + 1), self.split_shrink.ln());
            }
            GrowStrategy::Misfit => {
                let c = ((1.0 - GROW_WEIGHT) / GROW_WEIGHT).ln();
                for j in 0..K1 - 1 {
                    map[j] = if j + 1 < k_src {
                        (Some(j), c)
                    } else {
                        (None, c)
                    };
                }
                for k in 0..k_src {
                    map[(K1 - 1) + 2 * k] = (Some(src_mu(k)), 0.0);
                    map[(K1 - 1) + 2 * k + 1] = (Some(src_mu(k) + 1), 0.0);
                }
                let mut order: Vec<usize> = (0..y.len()).collect();
                order.sort_by(|&a, &b| src_logpdf[a].total_cmp(&src_logpdf[b]));
                let m = ((y.len() as f64 * GROW_MISFIT_FRAC) as usize)
                    .max(20)
                    .min(y.len());
                let worst: Vec<f64> = order[..m].iter().map(|&i| y[i]).collect();
                let wm = worst.iter().sum::<f64>() / m as f64;
                let ymean = y.mean().unwrap_or(0.0);
                let ysd =
                    (y.iter().map(|v| (v - ymean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
                let wsd = (worst.iter().map(|v| (v - wm).powi(2)).sum::<f64>() / m as f64)
                    .sqrt()
                    .max(1e-3 * ysd)
                    .max(1e-12);
                map[(K1 - 1) + 2 * k_src] = (None, wm);
                map[(K1 - 1) + 2 * k_src + 1] = (None, wsd.ln());
            }
        }
        let mut init = Array1::zeros(n_new);
        for (col, &(sc, shift)) in map.iter().enumerate() {
            init[col] = sc.map_or(0.0, |c| src.init[c]) + shift;
        }
        let n_features = x.ncols();
        let mut models: Vec<Vec<Box<dyn TrainedBaseLearner>>> =
            Vec::with_capacity(src.models.len());
        for round in src.models {
            let mut out: Vec<Box<dyn TrainedBaseLearner>> = Vec::with_capacity(n_new);
            for &(sc, _) in &map {
                out.push(match sc {
                    Some(c) => clone_learner(round[c].as_ref())?,
                    None => Box::new(TrainedHistogramTree::new(
                        HistTreeNode::Leaf { value: 0.0 },
                        n_features,
                    )),
                });
            }
            models.push(out);
        }
        let mut model = self.build::<NormalMixture<K1>>();
        model.init_params = Some(init);
        model.base_models = models;
        model.scalings = src.scalings.to_vec();
        model.effective_learning_rates = src.lrs.to_vec();
        model.col_idxs = src.cols.to_vec();
        model.partial_fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
        Ok(model)
    }

    fn fit_cold(
        &self,
        k: usize,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<AutoMixtureModel<S>, &'static str> {
        Ok(match k {
            1 => {
                let mut m = self.build::<Normal>();
                m.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
                AutoMixtureModel::Normal(m)
            }
            2 => {
                let mut m = self.build::<NormalMixture<2>>();
                m.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
                AutoMixtureModel::K2(m)
            }
            3 => {
                let mut m = self.build::<NormalMixture<3>>();
                m.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
                AutoMixtureModel::K3(m)
            }
            _ => {
                let mut m = self.build::<NormalMixture<4>>();
                m.fit_with_validation(x, y, Some(x_val), Some(y_val), None, None)?;
                AutoMixtureModel::K4(m)
            }
        })
    }

    /// Grow a `k`-component mixture from a fitted (k − 1)-component head
    /// (`grow_strategy`) and continue boosting it with early stopping on
    /// (x_val, y_val). `prev` is usually a candidate from a previous `fit`.
    pub fn grow_from(
        &self,
        k: usize,
        prev: &AutoMixtureModel<S>,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<AutoMixtureModel<S>, &'static str> {
        let lp = dispatch!(prev, m => m.pred_dist(x).logpdf(y));
        let ref_sigma: Array1<f64> = match prev {
            AutoMixtureModel::Normal(m) => m.pred_dist(x).scale,
            AutoMixtureModel::K2(m) => m.pred_dist(x).sigma.column(1).to_owned(),
            AutoMixtureModel::K3(m) => m.pred_dist(x).sigma.column(2).to_owned(),
            AutoMixtureModel::K4(m) => m.pred_dist(x).sigma.column(3).to_owned(),
        };
        Ok(dispatch!(prev, m => {
            let parts = Parts {
                init: m.init_params.as_ref().unwrap(),
                models: &m.base_models,
                scalings: &m.scalings,
                lrs: &m.effective_learning_rates,
                cols: &m.col_idxs,
            };
            match k {
                2 => AutoMixtureModel::K2(self.grow::<2>(parts, 1, (&lp, &ref_sigma), x, y, x_val, y_val)?),
                3 => AutoMixtureModel::K3(self.grow::<3>(parts, 2, (&lp, &ref_sigma), x, y, x_val, y_val)?),
                _ => AutoMixtureModel::K4(self.grow::<4>(parts, 3, (&lp, &ref_sigma), x, y, x_val, y_val)?),
            }
        }))
    }

    /// Fit candidates on (x, y), early-stop and recalibrate each on
    /// (x_val, y_val), keep the lowest validation NLL. Validation data is
    /// required (it is the selection signal); for time series pass a
    /// chronological block. `search` chooses the candidate families (cold,
    /// warm-grown, or both — K = 1 is shared); with `greedy`, each family
    /// stops at the first K that does not improve its validation NLL.
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<(), &'static str> {
        if self.candidates.is_empty() || self.candidates.iter().any(|&k| !(1..=4).contains(&k)) {
            return Err("candidates must be a non-empty subset of 1..=4");
        }
        let mut cands = self.candidates.clone();
        cands.sort_unstable();
        cands.dedup();
        let warm = matches!(self.search, KSearch::Warm | KSearch::Both);
        if warm && cands != (1..=cands[cands.len() - 1]).collect::<Vec<_>>() {
            return Err("warm search needs candidates 1..=max (each K is grown from K − 1)");
        }
        self.tried.clear();
        let mut fitted: Vec<AutoMixtureModel<S>> = Vec::new();
        let val = |m: &AutoMixtureModel<S>| dispatch!(m, mm => Scorable::<S>::total_score(&mm.pred_dist(x_val), y_val, None));
        let rounds = |m: &AutoMixtureModel<S>| dispatch!(m, mm => mm.base_models.len());
        // cold family (K = 1 is shared with the warm family)
        let mut prev_val = f64::INFINITY;
        let mut k1_index = None;
        for &k in &cands {
            if k > 1 && self.search == KSearch::Warm {
                break;
            }
            let model = self.fit_cold(k, x, y, x_val, y_val)?;
            let v = val(&model);
            self.tried.push((k, false, v, rounds(&model)));
            if k == 1 {
                k1_index = Some(fitted.len());
            }
            fitted.push(model);
            if self.greedy && !(v < prev_val) {
                break;
            }
            prev_val = v;
        }
        // warm family, grown from the cold K = 1 fit
        if warm && let Some(mut prev_idx) = k1_index {
            let mut prev_val = self.tried[prev_idx].2;
            for &k in cands.iter().filter(|&&k| k > 1) {
                let src_rounds = rounds(&fitted[prev_idx]);
                let model = self.grow_from(k, &fitted[prev_idx], x, y, x_val, y_val)?;
                let v = val(&model);
                self.tried.push((k, true, v, rounds(&model) - src_rounds));
                fitted.push(model);
                prev_idx = fitted.len() - 1;
                if self.greedy && !(v < prev_val) {
                    break;
                }
                prev_val = v;
            }
        }
        let idx = (0..fitted.len())
            .filter(|&i| self.tried[i].2.is_finite())
            .min_by(|&a, &b| self.tried[a].2.total_cmp(&self.tried[b].2))
            .ok_or("no candidate produced a finite validation NLL")?;
        self.model = Some(fitted.swap_remove(idx));
        self.selected_warm = self.tried[idx].1;
        Ok(())
    }

    /// Whether the selected head is a warm-grown one (false for K = 1).
    pub fn selected_is_warm(&self) -> bool {
        self.selected_warm
    }

    fn model(&self) -> &AutoMixtureModel<S> {
        self.model
            .as_ref()
            .expect("NGBAutoMixtureRegressor: call fit() first")
    }

    /// The selected head, for anything not exposed here.
    pub fn selected(&self) -> Option<&AutoMixtureModel<S>> {
        self.model.as_ref()
    }

    /// Number of components of the selected head (1 = single Normal).
    pub fn n_components(&self) -> Option<usize> {
        self.model.as_ref().map(|m| match m {
            AutoMixtureModel::Normal(_) => 1,
            AutoMixtureModel::K2(_) => 2,
            AutoMixtureModel::K3(_) => 3,
            AutoMixtureModel::K4(_) => 4,
        })
    }

    /// Predictive mean (the RMSE-optimal point forecast).
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        dispatch!(self.model(), m => m.pred_dist(x).mean())
    }

    /// Predictive median (the MAE-optimal point forecast).
    pub fn predict_median(&self, x: &Array2<f64>) -> Array1<f64> {
        dispatch!(self.model(), m => m.pred_dist(x).median())
    }

    /// Mean negative log-likelihood on (x, y) (whatever the training rule).
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        dispatch!(self.model(), m => Scorable::<LogScore>::total_score(&m.pred_dist(x), y, None))
    }

    /// Mean CRPS on (x, y) (closed form for every candidate head).
    pub fn crps(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        dispatch!(self.model(), m => Scorable::<CRPScore>::total_score(&m.pred_dist(x), y, None))
    }

    pub fn logpdf(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        dispatch!(self.model(), m => m.pred_dist(x).logpdf(y))
    }

    pub fn cdf(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        dispatch!(self.model(), m => m.pred_dist(x).cdf(y))
    }

    pub fn ppf(&self, x: &Array2<f64>, q: &Array1<f64>) -> Array1<f64> {
        dispatch!(self.model(), m => m.pred_dist(x).ppf(q))
    }

    /// Central (1 − alpha) prediction interval per row.
    pub fn interval(&self, x: &Array2<f64>, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        dispatch!(self.model(), m => m.pred_dist(x).interval(alpha))
    }
}
