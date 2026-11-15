# AutoML / hyperparameter research log

Tuner research for `src/hyper_opt.rs`. Infrastructure and the search
space: `search_space.md`. Results tables: `study_comparisons.md`. Commands
and file formats: `reports/README.md`. Learning-rule work lives in
`research/optimization-algorithm/`; the two are kept apart on purpose —
a gain from *tuning* is not a gain from a better *tuner*.

## 2026-11-13 — Cycle 18 (first tuner cycle)

Branch `research/cycle18` on 70b314b (1.2.0). `src/hyper_opt.rs` was not
modified; the samplers were re-implemented in `examples/c18_bench.rs`
with the same `tpe` crate calls, parameter order and transforms, so the
library tuner is the reference and nothing about it changed.

### Starting point

No tuner benchmark existed: no random-search control, no trial log, and
every one of Cycles 0–17's 67 report sets used depth 3, `min_samples_leaf`
1 and learning rate 0.05 or 0.01. The consumer-side notes in
`hyper_opt.rs` ("best found in the first 10–20 trials, the rest improves
by < 0.11 %") could mean a flat surface, a collapsed sampler, or a search
space that left out the knobs that matter.

### Hypotheses, stated before the runs, and what happened

| | Hypothesis | Verdict |
|---|---|---|
| A1 | Tree shape, never varied, is worth more than the sampler | **Confirmed.** Tuned beats default on test by 0.06–0.27 NLL on five of six datasets; `min_samples_leaf` is the main lever |
| A2 | Scoring a trial on the fold it early-stops and recalibrates on distorts selection | **Rejected.** Optimistic by 0.02–0.06 but ranks better than an untouched fold; selection is equal or better on five of six |
| A3 | Low round caps and a fixed patience of 50 bias the search against low learning rates | **Confirmed.** 0.01–0.04 NLL at lr 0.01 from patience alone; 0.05–0.3 with a cap of 500–1500 |
| A4 | TPE without a random startup phase collapses early; random search would do as well | **Rejected.** TPE beats random by 0.02–0.05 at 40 trials on the real objective; startup makes no reliable difference |
| A5 | 5-fold CV makes the Fisher-weight exponent β selectable where one holdout did not (Cycle 3) | **Confirmed but moot.** CV picks β = 0 on protein 10/10; under 1.2.0 the staged fit dominates the joint β variants anyway |

### Bottleneck

The search space, not the sampler. The documented example searches
learning rate, depth and `n_estimators`. It omits the strongest knob
(`min_samples_leaf`), includes one that should not be searched
(`n_estimators`, which early stopping turns into a cap that truncates low
learning rates), and runs every fold with a patience that does not scale
with the learning rate.

### Existing sampler behaviour

Independent per-parameter TPE, γ = 0.1, 24 candidates, no startup,
Parzen prior at the range midpoint. On a 5-dimensional space this is
enough: by 20 trials it is ahead of random search on every dataset, on
both the real objective (8 sampler seeds) and the surrogate (200). Two
changes were tried and neither earns its place:

- 10 or 20 random startup trials: indistinguishable from none by trial 30.
- γ = 0.25: worse everywhere, more duplicate proposals.

Not tried: a joint (multivariate) TPE. The measured interaction that
would motivate it is learning rate × round budget, and that is better
removed from the space (A3) than modelled.

### Decisions

Nothing in `src/hyper_opt.rs` was changed. Recommendations, in order of
expected effect on out-of-sample accuracy:

1. **Search `min_samples_leaf`** (log scale, ≈ 5–200) and `max_depth`
   (2–8). These two carry most of the gain. Leaf size alone, at the
   default shape, is worth −0.01…−0.14 (`optimization-algorithm` O2).
2. **Do not search `n_estimators`.** Fix a cap that scales with the
   learning rate (≈ 60–100/lr) and let early stopping choose.
3. **Scale the fold patience with the learning rate** (≈ 2.5/lr, floor
   50) instead of the fixed `CV_EARLY_STOPPING_ROUNDS = 50`. Today a
   caller can do this through the builder; the library default cannot.
   With 2 and 3 in place the learning rate is nearly free between 0.005
   and 0.05 and can be fixed near 0.02–0.05.
4. `col_sample` only above ≈ 0.6; `minibatch_frac` need not be searched.
5. Keep TPE as it is. Keep scoring on the early-stopping fold.
6. 20–40 trials is where the curve flattens on these problems.

**Update, same day — approved by the user and implemented** (see
"Implemented" at the end of this file). The list below is kept as it was
written before the approval.

Candidates for a library change, **not made** (each needs approval):

- a fold-patience rule that depends on the learning rate (changes
  `hyper_opt` results);
- a documented default search space, replacing the module example;
- the tuned defaults themselves — see O2 for `min_samples_leaf`.

### Limits of this evidence

- One head (Normal), LogScore, six datasets (four synthetic with ten
  features, kin8nm, protein at 12 000 rows), n ≤ 12 000. The leaf-size
  effect was checked on six more heads in O2; the rest of this log was
  not.
- Trials were scored on one 10 % holdout, not the 5-fold CV `hyper_opt`
  defaults to. The reuse mechanism per fold is the same; fold averaging
  reduces noise for both scores.
- Sampler comparison on the real objective: 3 datasets, 8 sampler seeds,
  one data seed, 40 trials. The surrogate adds seeds, not independence.
- Data seeds: 3 for the pool and the learning-rate grid, 10 for the
  selection study.
- The round budget rule used throughout (cap 100/lr, patience 2.5/lr) is
  itself a choice; it was not tuned.
- Pruning was not studied: with parallel folds, the default, it saves no
  compute (a trial is only "pruned" after all its folds have run).

### Literature

- Bergstra, Bardenet, Bengio, Kégl, "Algorithms for Hyper-Parameter
  Optimization", NeurIPS 2011 — TPE, including the random startup phase
  and γ.
  https://lacuna.tiptreesystems.com/work/algorithms-for-hyper-parameter-optimization/version/art_c144df8b3e2847169d18565155596c93
- Bergstra & Bengio 2012 (random search as the control), Eggensperger et
  al. 2015 (surrogate benchmarks for sampler comparison) and Watanabe
  2023 (roles of TPE's components) — cited from memory, not checked this
  cycle; the last was searched for in Lacuna and not found.

### Next questions

1. Does the picture hold for the count heads and for `hyper_opt`'s
   5-fold objective? (Poisson / NegativeBinomial pool at smaller budget.)
2. A patience rule in `cv_with_pruning` — measure through the library on
   the consumer's search space before proposing it.
3. Depth is the only knob whose direction is data-dependent: is there a
   cheap diagnostic (validation RMSE of the location stage at depth 3 vs
   6) that predicts it, so that depth need not be searched?
4. Top-k or fold-model ensembling of the search's own trials — the usual
   next gain once selection is working; changes the model class, so it
   belongs in a separate proposal.
5. Whether het_b-like targets (noise scale tied to the mean's dominant
   feature), where the reused score flattered the leaf-1 default, are
   common enough to justify an optional nested score.

## 2026-11-13 — Implemented (after approval)

The user approved the two library candidates above. In `src/hyper_opt.rs`:

| Change | What | Evidence |
|---|---|---|
| Fold patience | `cv_early_stopping_rounds(lr) = clamp(round(2.5/lr), 50, 200)` replaces the fixed 50 when the builder sets no `early_stopping_rounds`. Learning rates ≥ 0.05 are unchanged | A3 (`reports/A3_lrcap_summary.txt`) |
| Round cap helper | `default_round_cap(lr) = min(6000, ceil(100/lr))` | A3; the cap used by every A1/A4 trial |
| Default space | `default_search_space()`: learning rate 0.02–0.2 (log), depth 2–8, `min_samples_leaf` 5–200 (log int), `col_sample` 0.6–1.0. No `n_estimators`, no `minibatch_frac` | A1 (`reports/A1_pool_summary.txt`), decisions 1–4 |
| Builder | `default_histogram_model::<D, S>(params)`: reads the space, sets the cap and the patience | — |
| Module example | now the default space; the old example searched `n_estimators` 200–1500 | A3 |

Differences between the shipped space and the one the study sampled
(`search_space.md`): `min_samples_leaf` starts at 5 instead of 1 (the
selected median was ≈ 50 and leaf 1 never won), `col_sample` starts at 0.6
instead of 0.4 (lower only hurt), and `minibatch_frac` is dropped (no
effect). The shipped space is therefore a sub-box of the measured one; it
was **not** re-run as a search of its own.

Not changed: the sampler (TPE, γ = 0.1, no startup), fold reuse for early
stopping and scoring, pruning. Tests: `hyper_opt::tests::{fold_patience_…,
default_space_…, default_builder_…, default_space_search_runs_end_to_end}`.

Still open: the space on count heads and under the 5-fold objective; a
search over the shipped sub-box as such.

## 2026-11-14 — Cycle 19: the shipped search, end to end (A6)

Closes the two points left open above for the Normal head: the shipped
sub-box searched as such, and under `hyper_opt`'s 5-fold objective. The
user asked for this benchmark; nothing in `src/hyper_opt.rs` was tuned on
it.

**Question.** Does `default_search_space` + `default_histogram_model`,
run through `hyper_opt_with_config` exactly as documented, give a better
deployed model than the 1.2.0 module example at the same trial budget —
and than not tuning?

**Design** (`c18_bench --mode hpo`, `reports/A6_hpo_<dataset>.csv`,
`hpo_summary.py`). Five datasets (het_a, het_b, tnoise, kin8nm, protein),
3000 rows each, 8 data seeds. Training pool = the 80 % that is not test
(2400 rows); the tuner sees only the pool. 30 trials, 5 contiguous folds,
no pruning, the library's TPE, seed = data seed. Each arm's winner is
refitted by the arm's own builder on the whole pool (early stopping on
the library's 10 % holdout) and scored once on the test block.

| arm | space | leaf | fold patience | round cap |
|---|---|---|---|---|
| `untuned` | — (`default_histogram_model` with no parameters) | 20 | 50 | 2000 |
| `old` | 1.2.0 example: lr 0.005–0.1 log, depth 2–8, `n_estimators` 200–1500 | 1 | 50 | sampled |
| `oldspace` | the same space on the 1.3.0 defaults | 20 | 2.5/lr | sampled |
| `new` | `default_search_space` | searched 5–200 | 2.5/lr | 100/lr |

**Result**, held-out NLL, Δ (se), searches with lower NLL of 8:

| | new − old | new − untuned | old − untuned | oldspace − old |
|---|---|---|---|---|
| het_a | −0.184 (0.024) 8 | −0.117 (0.017) 8 | +0.067 (0.027) 1 | −0.064 (0.029) 7 |
| het_b | −0.011 (0.005) 6 | −0.046 (0.012) 8 | −0.035 (0.012) 8 | −0.007 (0.005) 6 |
| kin8nm | −0.069 (0.010) 8 | −0.144 (0.012) 8 | −0.075 (0.012) 8 | −0.047 (0.013) 7 |
| protein | −0.015 (0.005) 7 | −0.015 (0.008) 6 | +0.000 (0.008) 5 | −0.007 (0.005) 5 |
| tnoise | −0.187 (0.032) 8 | −0.064 (0.032) 6 | +0.123 (0.028) 0 | −0.072 (0.023) 7 |
| pooled, of 40 | −0.093 (0.015) 37 | −0.077 (0.011) 36 | +0.016 (0.014) 22 | −0.040 (0.009) 32 |

RMSE: new − old −0.051 (0.010), lower on 31 of 40; new − untuned −0.047
(0.008), 36 of 40.

**Reading.**

- The shipped search is better than the old recipe on every dataset and
  better than not tuning on every dataset; the margins are large on three
  (0.06–0.19) and small on het_b and protein (0.011–0.015).
- The 1.2.0 recipe was *worse than not tuning* on het_a and tnoise
  (+0.067, +0.123; 1 and 0 of 8 searches better). Its winners there have a
  median round cap near 1000–1150 at learning rates of 0.07 with leaf 1:
  the search optimises the wrong knobs while the default it competes
  with already has leaf 20. This is A1's and A3's finding seen from the
  user's side.
- Attribution: the same old space on the 1.3.0 defaults (`oldspace`)
  recovers −0.040 of the −0.093, so about two fifths come from the leaf
  default and the patience rule and three fifths from searching leaf size
  and dropping `n_estimators`.
- Winning parameters (medians): leaf 122, 126 and 106 on het_a, tnoise
  and kin8nm; 16 and 21 on het_b and protein; depth 2–4 on the synthetic
  sets, 6–8 on kin8nm and protein — as in A1.

**Cost.** Mean search time, new / old: 77/64 s, 38/53 s, 125/69 s, 65/71
s, 66/68 s (het_a, het_b, kin8nm, protein, tnoise). Equal trials, not
equal compute: on kin8nm the new space's deeper trees and longer fits cost
1.8×. All five searches ran concurrently on a loaded laptop, so these are
indications only.

**Failures.** None: no search or refit returned an error or a non-finite
score (the error files are empty).

**Guards.** The test block was scored once per arm and never used to
choose anything. Both searches had the same trials, folds, data and
seeds; neither arm was tuned afterwards. The gain is a *search-space and
defaults* gain, not a sampler gain — the sampler is the same.

**Limits.** One head, LogScore, one size (2400 training rows), 30 trials.
The `old` arm reconstructs 1.2.0 behaviour inside the 1.3.0 library (leaf
1, patience 50 through the builder) and refits with early stopping, which
the 1.2.0 example did not spell out. Count heads are still unmeasured.

**Decision.** Keep the shipped default space and builder as they are.

### Next questions

1. The same comparison on a count head (Poisson / NegativeBinomial) and
   on LogNormal, where leaf size acted through the same location trees.
2. het_b and protein gain little from any search here: is the default
   already near the optimum, or is 30 trials too few for their flatter
   surfaces? (A1's pool can answer this without new fits.)
3. With `min_samples_leaf` adaptive below 200 rows (optimization O6),
   should the space's lower bound of 5 follow the data size too?

## 2026-11-14 — Cycle 20: the shipped search on three more heads (A7)

A6's first open question. Same design as A6 (`c18_bench --mode hpo --head
<head>`; 30 trials, 5 folds, 2400 training rows, 8 seeds, the winner
refitted on the pool and scored on the untouched test block; arms
`untuned`, `old`, `oldspace`, `new` as in A6), with the head swapped:
LogNormal on lognorm_het and kin8nm, Poisson on pois_nl and pois_low,
NegativeBinomial on nb_over (`reports/A7_hpo_<head>_<dataset>.csv`,
`reports/A7_hpo_summary.txt`). Nothing in `src/hyper_opt.rs` was changed
for or after these runs.

Held-out NLL, Δ (se), searches lower of 8:

| head, dataset | new − old | new − untuned | old − untuned |
|---|---|---|---|
| LogNormal, kin8nm | −0.082 (0.020) 7 | −0.072 (0.018) 7 | +0.010 (0.012) 4 |
| LogNormal, lognorm_het | −0.075 (0.013) 8 | −0.047 (0.014) 8 | +0.028 (0.013) 2 |
| Poisson, pois_nl | −0.056 (0.016) 7 | −0.034 (0.009) 7 | +0.022 (0.012) 2 |
| Poisson, pois_low | +0.001 (0.002) 3 | −0.002 (0.002) 4 | −0.004 (0.001) 6 |
| NegativeBinomial, nb_over | −0.015 (0.007) 6 | −0.030 (0.007) 8 | −0.015 (0.003) 8 |
| pooled, of 40 | −0.045 (0.008) 31 | −0.037 (0.006) 34 | +0.008 (0.005) 22 |

`oldspace − old` is −0.011 (0.008), lower on 32 of 40.

**Reading.**

- The shipped search is better than the 1.2.0 recipe on four of five
  datasets and tied on the fifth (pois_low, a low, weakly varying rate:
  every arm is within 0.006 of every other — there is nothing to tune).
  It is never worse than not tuning.
- The margins are about half of the Normal head's (−0.045 against −0.093
  pooled): the count heads have one parameter per row to fit, and the
  leaf default already takes part of the gain.
- The 1.2.0 recipe is again no better than not tuning on average (+0.008)
  and worse on three of five datasets.
- Winning leaf sizes (medians): 83–120 for LogNormal, 95–195 for the
  count heads — as for Normal, well above the default of 20 at this size.
- By O12 (optimization log) a single-dataset difference below 0.02 is
  noise: the NegativeBinomial and pois_low rows say "no difference" for
  new − old; the LogNormal and pois_nl rows and the pooled row are
  effects.

**Cost.** Mean search time new / old: 74/75 s, 47/58 s (LogNormal),
33/34 s, 11/17 s (Poisson), 42/112 s (NegativeBinomial; the old space's
low learning rates run long there). Loaded machine, indications only.

**Failures.** None.

**Decision.** Keep the shipped default space for these heads too. The
documentation now says it was measured on four heads and where it does
not help.

### Next questions

1. Classification heads (Bernoulli, Categorical) — still unmeasured.
2. Whether `min_samples_leaf` should have a higher default at 2000+ rows:
   the tuned median is 80–200 on eight of ten datasets across A6 and A7.
   That is a default change and needs its own sweep.

## 2026-11-14 — Cycle 23: `max_bins` in the default search space? (A8)

Prompted by optimization O14: with the library's binning, held-out NLL
moved by 0.06–0.08 between 255, 32 and 8 bins, in opposite directions on
kin8nm and protein. `max_bins` was fixed at 255 in every earlier study
and is not in the shipped space. The user asked for it to be added, with
its own comparison.

**Candidate.** `max_bins`: integer, log scale, 8–255 (the range O14
covered; 255 is the learner's maximum). Fixed before any run.

**Design.** As A6/A7 (`c18_bench --mode hpo`): 30 trials, 5 folds, 2400
training rows, 8 seeds, winner refitted on the pool, test block scored
once. Arms: `new` — the four keys shipped so far (learning rate, depth,
leaf, `col_sample`); `newbins` — the same plus `max_bins`. Equal trials,
so the fifth dimension has to pay for the dilution it causes. Normal on
het_a, het_b, tnoise, kin8nm, protein; LogNormal on lognorm_het and
kin8nm; Poisson on pois_nl.

**Decision rule, written before the results.** Add the key to
`default_search_space` if, pooled over the eight dataset-head pairs,
`newbins − new` is ≤ 0 with more than half of the 64 searches lower, and
no pair is worse by more than 0.02 at more than 2 standard errors (0.02
being the single-dataset noise floor of optimization O12). Otherwise
`default_histogram_model` keeps reading the key — so a caller can search
it — but the shipped space stays as it is.

**Result** (`reports/A8_bins_<head>_<dataset>.csv`,
`reports/A8_bins_summary.txt`), held-out NLL, `newbins − new` (se),
searches lower of 8, and the median winning bin count:

| head, dataset | NLL | lower | RMSE | bins |
|---|---|---|---|---|
| Normal, kin8nm | −0.026 (0.014) | 7 | −0.005 (0.002) | 18 |
| LogNormal, kin8nm | −0.051 (0.013) | 8 | −0.005 (0.002) | 19 |
| Normal, protein | −0.010 (0.008) | 6 | −0.016 (0.025) | 68 |
| Normal, tnoise | −0.009 (0.019) | 4 | −0.001 (0.005) | 60 |
| LogNormal, lognorm_het | −0.007 (0.007) | 6 | −0.031 (0.047) | 128 |
| Normal, het_a | +0.001 (0.007) | 4 | −0.004 (0.005) | 150 |
| Poisson, pois_nl | +0.006 (0.006) | 3 | +0.011 (0.029) | 113 |
| Normal, het_b | +0.024 (0.016) | 3 | +0.002 (0.007) | 175 |
| pooled, of 64 | −0.009 (0.005) | 41 | −0.006 (0.007), 36 | |

**Against the rule.** Pooled ≤ 0: yes. More than half lower: 41 of 64.
No pair worse by more than 0.02 at more than 2 se: het_b is +0.024 at
1.5 se — inside the rule, and the one result that argues the other way.
The rule is met.

**Reading.** The gain is kin8nm under both heads, where the search
settles on 18–19 bins; that agrees with O14, which found 8–32 bins better
than 255 there by 0.06–0.08 with everything else at its default. On the
other six pairs the search keeps 60–175 bins and the difference is noise.
The cost of the extra dimension at 30 trials is visible only as het_b's
+0.024 and is not significant. Search time did not rise (fewer bins make
trees cheaper): 63 against 96 s on LogNormal/kin8nm.

**Decision.** `max_bins` (8–255, log int) is in `default_search_space`
from 1.3.0; `default_histogram_model` reads it. The untuned default stays
at 255. Check: the harness arm `shipped` (the library's own space) gives
the same rows as `newbins`.

**Failures.** None.

### Next questions

1. A default for `max_bins` without a search: is there a cheap signal
   (target smoothness, validation gain of the first trees at 32 against
   255 bins) that says when to coarsen?
2. Does 40–50 trials remove het_b's small loss?
3. Classification heads, still unmeasured in A6–A8.

## 2026-11-14 — Cycle 24: cross-fitting and the head as search choices (A9, A10)

The two untested levers named after A8. The user asked for both to be
tested the same way. Designs and decision rules below were written
before any run.

### A9 — `staged_crossfit_folds` as a two-way choice

**Candidate.** The shipped five-key space plus `staged_crossfit_folds`
∈ {0, 5} (categorical). `default_histogram_model` reads the key.

**Design.** As A6–A8: 30 trials, 5 folds, 2400 training rows, 8 seeds,
winner refitted, test scored once. Baseline: the shipped space — the
`newbins` rows of A8, which the harness arm `shipped` reproduces digit
for digit. Staged heads only, since the option does nothing elsewhere:
Normal on het_a, het_b, tnoise, kin8nm, protein; LogNormal on
lognorm_het, kin8nm.

**Decision rule.** The key will not go into `default_search_space`
whatever the result: it acts only on staged heads and multiplies the fit
time (2.6–4.8×), and the default space is head-agnostic. The question is
whether to *recommend* it as an extra key for staged heads: yes if the
pooled difference is ≤ −0.01 with more than half of the 56 searches
lower and no pair worse by more than 0.02 at more than 2 standard
errors.

### A10 — choosing the head by the search's own CV score

**Candidate.** `hyper_opt` is generic over one head, so a head cannot be
a key of the space. The procedure a user can run today: one search per
candidate head, keep the head whose best CV score (mean held-out log
score, comparable across heads for the same target) is lower.

**Design.** `c18_bench --mode hposel`: one 30-trial search of the
shipped space per head; the winner after 15 and after 30 trials is
refitted and scored. Pairs: Normal against StudentTFiniteVar on het_a,
het_b, tnoise, kin8nm, protein; LogNormal against GammaMean on
lognorm_het and kin8nm. 8 seeds. Compared, per dataset and seed:

- `single` — the first head alone at 30 trials (what a user does today);
- `select 2×15` — both heads at 15 trials, head chosen by CV score: the
  same total trial count;
- `select 2×30` — both at 30 trials: twice the budget;
- `oracle` — the better head on test, as a bound, never used to choose.

**Decision rule.** Recommend head selection in the documentation if
`select 2×15` is not worse than `single` pooled (≤ 0, more than half of
the searches lower or equal) — i.e. splitting a fixed budget over two
heads costs nothing on average — and say on which targets it pays. No
library change in this cycle.

### A9 — result

`reports/A9_xfit_<head>_<dataset>.csv`, `reports/A9_xfit_summary.txt`.
Held-out NLL, with the key − shipped space (se), searches lower of 8, and
how many winners use cross-fitting:

| head, dataset | NLL | lower | winner cross-fits |
|---|---|---|---|
| LogNormal, kin8nm | −0.159 (0.018) | 8 | 8 |
| Normal, tnoise | −0.064 (0.025) | 7 | 7 |
| Normal, kin8nm | −0.039 (0.016) | 6 | 8 |
| Normal, het_a | −0.030 (0.013) | 7 | 8 |
| Normal, protein | −0.012 (0.014) | 6 | 8 |
| LogNormal, lognorm_het | +0.001 (0.008) | 4 | 7 |
| Normal, het_b | +0.004 (0.010) | 4 | 6 |
| pooled, of 56 | −0.043 (0.009) | 42 | 52 |

RMSE: no gain — LogNormal/kin8nm −0.005 (8 of 8), protein and
lognorm_het about 1 % worse (1.6 se each), the rest within noise.

**Against the rule.** Pooled ≤ −0.01: yes (−0.043). More than half
lower: 42 of 56. No pair worse by more than 0.02 at more than 2 se: the
worst is +0.004. Met.

**Reading.** The search chooses cross-fitting almost always (52 of 56)
and is right to: this is the O1 effect (−0.037 at the default shape)
surviving tuning. Protein, where cross-fitting lost on 490 rows, does not
lose at 2400 rows inside a search. It is a calibration gain — the log
score improves, point predictions do not. Search time was 105–451 s
against 30–63 s for the shipped space on a machine running fourteen
searches at once: several times longer, in line with the 2.6–4.8× of a
single cross-fitted fit.

**Decision.** As stated beforehand, not in `default_search_space`. New:
`hyper_opt::default_search_space_with_crossfit()` (the default space plus
the key), documented as the recommended space for staged heads when the
search time is affordable; `default_histogram_model` reads the key.

Since the search picks it in 52 of 56 cases, a simpler reading is that
staged heads want cross-fitting *on* at this size, searched or not. That
is a default change for the fit itself and belongs to the user.

### A10 — result

`reports/A10_headsel_<group>_<dataset>.csv`,
`reports/A10_headsel_summary.txt`. Held-out NLL against `single` (the
first head, 30 trials):

| pair, dataset | second head alone | select 2×15 | select 2×30 | second head picked (2×30) |
|---|---|---|---|---|
| Normal / StudentTFiniteVar, tnoise | −0.083 | −0.066 (0.057) | −0.075 (0.025) | 7 of 8 |
| …, kin8nm | −0.005 | +0.006 | −0.013 (0.009) | 2 |
| …, protein | +0.003 | +0.004 | +0.003 | 8 |
| …, het_a | +0.051 | +0.018 | 0.000 | 0 |
| …, het_b | +0.058 | +0.028 | 0.000 | 1 |
| LogNormal / GammaMean, lognorm_het | +0.109 | +0.013 | +0.018 | 1 |
| …, kin8nm | +0.037 | **+0.047 (0.010)** | **+0.037 (0.012)** | 8 |
| pooled, of 56 | +0.024 | +0.007 (0.010), lower 15, equal 7 | −0.004 (0.007), lower 13, equal 29 | |

The oracle (the better head on test) is −0.017 pooled.

**Against the rule.** `select 2×15` pooled ≤ 0 with more than half lower
or equal: +0.007, 22 of 56. Not met.

**Reading.**

- Splitting a 30-trial budget over two heads costs more than the choice
  gains: 15 trials find a worse configuration (het_a, het_b +0.02–0.03
  even when the right head is picked).
- With twice the budget the choice is right where the gap is large
  (tnoise, t-distributed noise: −0.075) and harmless on the Normal-family
  datasets, but on kin8nm it preferred GammaMean to LogNormal in 8 of 8
  searches and was 0.037 worse on test. The fold score is computed on
  models stopped and recalibrated on the scoring fold; it ranks
  configurations of one head well (A2) and is evidently not a safe
  yardstick across heads whose calibration behaves differently.
- The ceiling is low anyway: an oracle gains 0.017.

**Decision.** No recommendation and no API for head selection. Guidance:
choose the head from the target (heavy-tailed noise → StudentTFiniteVar;
positive and skewed → LogNormal or GammaMean), and treat a CV comparison
of heads as informative only when the gap is large.

**Failures.** None in either study.

### Next questions

1. Cross-fitting on by default for staged heads (a fit default, the
   user's call): A9 and O1 agree on about −0.04 NLL at 2000+ rows; the
   known cost is 2.6–4.8× fit time and a loss on small heavy-tailed data
   with the Normal head (O11).
2. Why the fold score misranks GammaMean against LogNormal on kin8nm —
   the LogNormal scale ceiling acts at prediction time inside folds too,
   so that is not it; the recalibration of the two heads differs.
3. Classification heads, still unmeasured in A6–A10.

