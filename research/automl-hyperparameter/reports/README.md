# Cycle 18 tuner reports — reproducible commands

Build and conventions as in `research/optimization-algorithm/reports/README.md`
(`C18` = the `c18_bench` example, run from the repository root; `DS6` =
`het_a,het_b,homo,tnoise,kin8nm,protein`; protein subsampled to 12 000
rows; 70/10/20 split A / V / test per data seed).

The search space, budget rule (round cap min(6000, 100/lr), patience
clamp(2.5/lr, 50, 200)) and the description of `hyper_opt` are in
`../search_space.md`. All fits use the Normal head, LogScore,
`HistogramLearner` and the 1.2.0 defaults (staged fit + recalibration)
unless a `fit` column says otherwise. The model seed equals the data seed,
so each (dataset, seed, configuration) is a deterministic objective.

| File | What | Command |
|---|---|---|
| `A1_pool.csv` | the trial log of the sensitivity / tuned-vs-default / reuse-vs-nested study: 120 configurations × 6 datasets × 3 data seeds, two fits each. Configurations 0 and 1 are the reference defaults (lr 0.05 and 0.02, depth 3, leaf 1, no subsampling); 2–119 are uniform draws in sampling space, i.e. a random search | `C18 --mode pool --datasets DS6 --seeds 3 --n-cfg 120` |
| `A1_pool_summary.txt` | | `python3 -I ../pool_summary.py A1_pool.csv` |
| `A3_lrcap.csv` | learning rate × (round cap, patience) policy, default shape, staged and joint fits, 3 seeds, at most 6000 rows | `C18 --mode lrcap --datasets het_a,het_b,kin8nm,protein --seeds 3 --max-rows 6000`; `python3 -I ../study_summary.py lrcap A3_lrcap.csv` |
| `A4_search_real.csv` | sampler comparison on the real objective (data seed 0): random, TPE as in `hyper_opt`, TPE after 10 random trials; 8 sampler seeds × 40 trials | `C18 --mode search --datasets het_a,kin8nm,tnoise --samplers random,tpe,tpe10 --hp-seeds 8 --trials 40`; `python3 -I ../study_summary.py search A4_search_real.csv` |
| `A4_search_surrogate_summary.txt` | the same comparison plus `tpe20` and `tpe10g25` (γ = 0.25) on a k-NN surrogate of each dataset's seed-0 pool rows; 200 sampler seeds × 80 trials. The 43 MB trial file is not kept; it is deterministic | `C18 --mode search --objective surr:research/automl-hyperparameter/reports/A1_pool.csv --datasets DS6 --samplers random,tpe,tpe10,tpe20,tpe10g25 --hp-seeds 200 --trials 80 > /tmp/s.csv`; `python3 -I ../study_summary.py search /tmp/s.csv` |
| `A5_select.csv`, `A5_select_summary.txt` | choosing among joint (β = 0, 0.5, 1) and staged fits by one 10 % holdout vs 5-fold CV, default shape, 10 seeds | `C18 --mode select --datasets DS6 --seeds 10`; `python3 -I ../study_summary.py select A5_select.csv` |

## Columns

`A1_pool.csv`:
- `val_reuse`: fit on A with V as the validation set, scored on V — what a
  `hyper_opt` fold does (early stopping, the staged fit's stage stops, the
  recalibration and the score all on the same rows).
- `val_nested`: fit on A minus its last |V| rows, with those rows as the
  validation set, scored on the untouched V.
- `test_*`, `cov90`, `calib`: held-out test metrics of the `val_reuse`
  model. `test2_nll`: test NLL of the `val_nested` model.
- `n_iter`: rounds kept. `secs`: wall time under heavy machine load — not
  a timing.

`A4_search_*.csv`: one row per trial; `score` = `val_reuse` of the trial,
`best_score` / `test_at_best` = running incumbent and its test NLL.
`dup%` in the summary: share of trials whose score exactly repeats an
earlier trial of the same study (duplicate evaluations; integer
parameters only).

`A6_hpo_<dataset>.csv` (Cycle 19; one file per dataset, het_a, het_b,
tnoise, kin8nm, protein): `C18 --mode hpo --datasets <dataset> --seeds 8
--max-rows 3000 --trials 30` (arms `untuned,old,oldspace,new`; 5 folds).
Columns: `cv_score` = the winner's mean fold score, `opt_rounds` =
`HyperOptResult::opt_rounds`, `lr … cs` = the winning parameters (NaN =
not in that arm's space), `n_est` = the refit's round cap, `n_iter` = the
refit's rounds, `test_*` = the refitted winner on the test block.
`A6_hpo_summary.txt` = `python3 -I research/automl-hyperparameter/hpo_summary.py`
followed by the same with `--ref untuned`.

`A7_hpo_<head>_<dataset>.csv` (Cycle 20): `C18 --mode hpo --head <head>
--datasets <dataset> --seeds 8 --max-rows 3000 --trials 30` for lognormal
on lognorm_het and kin8nm, poisson on pois_nl and pois_low, negbin on
nb_over. Columns as `A6`; `test_crps`, `cov90` and `calib` are NaN (only
the log score and the RMSE of the mean are computed for these heads).
`A7_hpo_summary.txt` = `hpo_summary.py --files 'A7_hpo_*.csv'` followed by
the same with `--ref untuned`.

`A8_bins_<head>_<dataset>.csv` (Cycle 23): `C18 --mode hpo --head <head>
--datasets <dataset> --seeds 8 --max-rows 3000 --trials 30 --arms
new,newbins` for normal on het_a, het_b, tnoise, kin8nm, protein;
lognormal on lognorm_het, kin8nm; poisson on pois_nl. `new` is the
four-key space of A6/A7 (spelled out in the harness), `newbins` adds
`max_bins`, `shipped` is whatever `default_search_space` returns (equal to
`newbins` since 1.3.0). Extra column `max_bins`. `A8_bins_summary.txt` =
`hpo_summary.py --files 'A8_bins_*.csv' --ref new`.

`A9_xfit_<head>_<dataset>.csv` (Cycle 24): `C18 --mode hpo --head <head>
--datasets <dataset> --seeds 8 --max-rows 3000 --trials 30 --arms xfit`
for normal on het_a, het_b, tnoise, kin8nm, protein and lognormal on
lognorm_het, kin8nm. Extra column `xfit` (the winner's
`staged_crossfit_folds`). Baseline: the `newbins` rows of the `A8` files
(= the shipped space). `A9_xfit_summary.txt` = `hpo_summary.py --files
'A9_xfit_*.csv,A8_bins_normal_*.csv,A8_bins_lognormal_*.csv' --ref
newbins`.

`A10_headsel_<group>_<dataset>.csv` (Cycle 24): `C18 --mode hposel --heads
normal,studenttfv --datasets <dataset> --seeds 8 --max-rows 3000 --trials
30` (`real`: het_a, het_b, tnoise, kin8nm, protein) and `--heads
lognormal,gammamean` (`positive`: lognorm_het, kin8nm). One row per head
and budget: the best of the first 15 trials and of all 30, each refitted.
`A10_headsel_summary.txt` = `headsel_summary.py`.

`A5_select.csv`: `holdout` = score on V of the model fitted on A; `cv5` =
mean fold score of 5 contiguous folds over A ∪ V, each fold used as that
fit's validation set (as `hyper_opt` does); `test_nll` = the A-fitted
model on test.

Failures: no fit failed or returned a non-finite score in any file.
Status is therefore implicit; a failed fit would appear as NaN (pool) or
+inf (search).

Test data: the 20 % test block was used for evaluation only. No
configuration, sampler setting or decision in this cycle was chosen on
test numbers; the "oracle" columns are reported as bounds, not used.
