# O7 — Leaf default of the exact tree learners

Cycle 19, 2026-11-14.

## Question

O2 changed `HistogramLearner`'s default `min_samples_leaf` from 1 to 20 on
evidence gathered with that learner only. `DecisionTreeLearner` and
`ArenaDecisionTreeLearner` kept leaf 1, so `HistogramLearner::new(3)` and
`DecisionTreeLearner::new(3)` disagreed. Does the exact split finder want
the same leaf size? The mechanism in O2 (location trees chasing single
rows) does not depend on how splits are found, so the prediction was yes.

## Test

`c18_bench --mode treeleaf`: the same model, data and scoring as O2's
sweep (`--mode leaf --fit default`: Normal, default staged fit, depth 3,
learning rate 0.05, round cap 100/lr, patience 2.5/lr), with the learner
swapped. Six datasets × 10 seeds. `treeleaf.py` prints the tables.

Leaf 20 minus leaf 1, held-out NLL (se), seeds with lower NLL:

| | het_a | het_b | homo | tnoise | kin8nm | protein | mean |
|---|---|---|---|---|---|---|---|
| exact, 210 rows | −0.103 (0.029) 9 | −0.142 (0.047) 8 | −0.056 (0.020) 8 | −0.139 (0.066) 9 | −0.010 (0.016) 7 | −0.043 (0.018) 9 | −0.082 |
| exact, 490 rows | −0.150 (0.023) 10 | −0.121 (0.029) 9 | −0.107 (0.025) 9 | −0.166 (0.029) 10 | −0.019 (0.013) 7 | −0.027 (0.015) 8 | −0.098 |
| exact, 2100 rows | −0.062 (0.013) 9 | −0.022 (0.008) 8 | −0.066 (0.022) 9 | −0.091 (0.019) 10 | −0.025 (0.008) 9 | −0.013 (0.005) 7 | −0.047 |
| arena, 2100 rows | −0.078 (0.022) 10 | −0.037 (0.008) 8 | −0.016 (0.013) 7 | −0.066 (0.025) 8 | −0.032 (0.007) 10 | −0.001 (0.007) 3 | −0.038 |

RMSE moves the same way (exact: −0.003…−0.25, never significantly up;
arena: protein +0.021, 0.7 se). The effect is as large as for the
histogram learner (O2: −0.014…−0.062 at 2100 rows) or larger.

## Decision (approved by the user, implemented 2026-11-14)

`DecisionTreeLearner::new(depth)` and `ArenaDecisionTreeLearner::new(depth)`
use `HistogramLearner::DEFAULT_MIN_SAMPLES_LEAF` with the adaptive rule of
O6, the same as `HistogramLearner::new`.

Not changed, on purpose: `default_sklearn()` of both learners and the
helpers built on it, `default_tree_learner()` and `arena_tree_learner()`.
Their documented contract is scikit-learn's defaults (Python NGBoost's
base learner), which the upstream-parity and accuracy-parity tests rely
on; they keep leaf 1 and say so. `with_params` is explicit as before.

## Limits

Normal head, six datasets, depth 3. The arena learner was measured at one
size. Small-data behaviour (below 200 rows) was measured for the histogram
learner only (O6) and is assumed to carry over.
