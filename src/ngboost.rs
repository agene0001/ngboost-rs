use crate::dist::normal::Normal;
use crate::dist::Distribution;
use crate::learners::{default_tree_learner, BaseLearner, StumpLearner, TrainedBaseLearner};
use crate::scores::{LogScore, Scorable, Score};
use ndarray::{Array1, Array2};
use std::marker::PhantomData;

pub struct NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    // Hyperparameters
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub natural_gradient: bool,

    // Base learner
    base_learner: B,

    // State
    pub base_models: Vec<Vec<Box<dyn TrainedBaseLearner>>>,
    pub scalings: Vec<f64>,
    pub init_params: Option<Array1<f64>>,

    // Generics
    _dist: PhantomData<D>,
    _score: PhantomData<S>,
}

impl<D, S, B> NGBoost<D, S, B>
where
    D: Distribution + Scorable<S> + Clone,
    S: Score,
    B: BaseLearner + Clone,
{
    pub fn new(n_estimators: u32, learning_rate: f64, base_learner: B) -> Self {
        NGBoost {
            n_estimators,
            learning_rate,
            natural_gradient: true,
            base_learner,
            base_models: Vec::new(),
            scalings: Vec::new(),
            init_params: None,
            _dist: PhantomData,
            _score: PhantomData,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.init_params = Some(D::fit(y));
        let n_params = self.init_params.as_ref().unwrap().len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(self.init_params.as_ref().unwrap()));

        let score_marker = S::default();

        for _ in 0..self.n_estimators {
            let dist = D::from_params(&params);
            let grads = Scorable::grad(&dist, y, self.natural_gradient);

            let mut fitted_learners: Vec<Box<dyn TrainedBaseLearner>> = Vec::new();
            let mut predictions_cols: Vec<Array1<f64>> = Vec::new();

            for j in 0..n_params {
                let grad_j = grads.column(j).to_owned();
                let learner = self.base_learner.clone();
                let fitted = learner.fit(x, &grad_j)?;
                predictions_cols.push(fitted.predict(x));
                fitted_learners.push(fitted);
            }

            let predictions = to_2d_array(predictions_cols);

            let scale = self.line_search(&predictions, &params, y, score_marker);
            self.scalings.push(scale);
            self.base_models.push(fitted_learners);

            params -= &(self.learning_rate * scale * &predictions);
        }

        Ok(())
    }

    fn get_params(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_params = self.init_params.as_ref().unwrap().len();
        let mut params = Array2::from_elem((x.nrows(), n_params), 0.0);
        params
            .outer_iter_mut()
            .for_each(|mut row| row.assign(self.init_params.as_ref().unwrap()));

        for (i, learners) in self.base_models.iter().enumerate() {
            let scale = self.scalings[i];

            let predictions_cols: Vec<Array1<f64>> =
                learners.iter().map(|learner| learner.predict(x)).collect();

            let predictions = to_2d_array(predictions_cols);

            params -= &(self.learning_rate * scale * &predictions);
        }
        params
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> D {
        let params = self.get_params(x);
        D::from_params(&params)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.pred_dist(x).predict()
    }

    fn line_search(
        &self,
        resids: &Array2<f64>,
        start: &Array2<f64>,
        y: &Array1<f64>,
        _score: S,
    ) -> f64 {
        let mut scale = 1.0;
        let initial_score = Scorable::total_score(&D::from_params(start), y, None);

        // Scale up
        loop {
            let next_params = start - &(resids * scale);
            let score = Scorable::total_score(&D::from_params(&next_params), y, None);
            if score >= initial_score || !score.is_finite() || scale > 256.0 {
                break;
            }
            scale *= 2.0;
        }

        // Scale down
        loop {
            let next_params = start - &(resids * scale);
            let score = Scorable::total_score(&D::from_params(&next_params), y, None);
            if score < initial_score && score.is_finite() {
                break;
            }
            if scale < 1e-4 {
                return 0.0;
            }
            scale *= 0.5;
        }

        scale
    }
}

fn to_2d_array(cols: Vec<Array1<f64>>) -> Array2<f64> {
    if cols.is_empty() {
        return Array2::zeros((0, 0));
    }
    let nrows = cols[0].len();
    let ncols = cols.len();
    let mut arr = Array2::zeros((nrows, ncols));
    for (j, col) in cols.iter().enumerate() {
        arr.column_mut(j).assign(col);
    }
    arr
}

// High-level API
pub struct NGBRegressor {
    model: NGBoost<Normal, LogScore, StumpLearner>,
}

impl NGBRegressor {
    pub fn new(n_estimators: u32, learning_rate: f64) -> Self {
        Self {
            model: NGBoost::new(n_estimators, learning_rate, default_tree_learner()),
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), &'static str> {
        self.model.fit(x, y)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        self.model.predict(x)
    }

    pub fn pred_dist(&self, x: &Array2<f64>) -> Normal {
        self.model.pred_dist(x)
    }
}
