# Cycle 18 reports — reproducible commands

Build (as for `research/ngboost/`):

```
export CARGO_TARGET_DIR=/Volumes/Build/ngboost-rs \
  CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  CARGO_PROFILE_RELEASE_STRIP=false
cargo build --release --features accelerate --example c18_bench --example research_bench
```

`C18` = `…/release/examples/c18_bench`, `RB` = `…/release/examples/research_bench`,
run from the repository root. `DS6` = `het_a,het_b,homo,tnoise,kin8nm,protein`.
Defaults: lr 0.05, depth 3, round cap 2000, patience 50, protein
subsampled to 12 000 rows, 70/10/20 split, 10 seeds. Branch
`research/cycle18` on top of 70b314b.

Summaries: `python3 -I research/c18_pair.py FILE… --by <cols> --ref <label> --metrics <cols>`
(paired on dataset and seed).

| File | Command |
|---|---|
| `O2_leaf_sweep.csv` | `C18 --mode leaf --datasets DS6 --seeds 10` (fit modes default, joint; leaves 1, 5, 10, 20, 50, 100, 200). Pair: `--by fit,msl --ref default/1 --metrics test_nll,test_rmse,cov90 --where fit=default` |
| `O2_leaf_sweep_n700.csv` | `C18 --mode leaf --datasets DS6 --seeds 10 --max-rows 700 --leaves 1,5,10,20,50 --fit default` |
| `O2_leafsplit.csv` | `C18 --mode leafsplit --datasets DS6 --seeds 10 --big 50` (variants `mu<leaf>_sg<leaf>`; pair `--ref mu1_sg1`) |
| `O2_heads_<head>_ml{1,20}.csv` | `RB --dist <head> --variants <v> --datasets <d> --seeds 10 --min-leaf {1,20}`: poisson `base` on pois_lin, pois_nl, pois_low, nb_over; negbin `auto` on nb_over, nb_low, pois_nl; bernoulli `base` on sprint_cvd, flchain_death, support_hospdead; laplace `auto` on het_a, tnoise, kin8nm, support_logcost; lognormal `auto` on lognorm_het, kin8nm, support_cost; gammamean `auto+staged` on gamma_het, lognorm_het, kin8nm |
| `O1_oof_n.csv` | `C18 --mode oof --datasets DS6 --seeds 10` (variants lib_staged, lib_staged_nocal, ts_in, ts_in_cal, ts_oof, ts_oof_cal; K = 5). Pair `--ref lib_staged --metrics nll,cov90,calib,rmse` |
| `O1_oof_n700.csv`, `O1_oof_n700_default.csv` | the same with `--max-rows 700`; `--variants lib_default` for the second |
| `O1_oof_val30.csv`, `O1_oof_val30_default.csv` | `--max-rows 3000 --val-rows 30`; `--variants lib_default` for the second. NB: `lib_staged` forces the staged fit; with 30 validation rows the library default is `lib_default` (joint, uncalibrated) — pair against that |
| `O1_oof_leaf20.csv` | `C18 --mode oof --datasets DS6 --seeds 10 --leaf 20 --variants lib_staged,ts_in_cal,ts_oof,ts_oof_cal` |
| `O3_nu<f>_<part>.csv` | `NGBOOST_STUDENTT_NU_MIN=<f> RB --dist studentt --variants auto+joint,auto+staged --datasets <part> --seeds 10 --max-rows 12000`, f ∈ {0, 1, 2}; parts `het_a,het_b,homo,tnoise` / `lognorm_het,kin8nm,support_logcost` / `protein`. f ∈ {1.1, 2.1}: `--variants auto+staged` only. All 80 fits per floor and variant completed |
| `O4_gamma_<part>.csv` | `RB --dist gamma --variants base --datasets <part> --seeds 10 --max-rows 12000`; parts `gamma_homo,gamma_het` / `gamma_shared,lognorm_het` / `kin8nm,support_cost` |
| `O4_gammamean_<part>.csv` | `RB --dist gammamean --variants base,fwp1,auto,auto+fwp1,auto+staged`, same parts |

Column notes. `c18_bench` oof/leafsplit rows: `calib` = sqrt(mean err² /
mean σ̂²) on test (> 1 overconfident); `extra` = in-sample / cross-fitted
mean squared residual of the stage-3 location (the share of residual
variance the in-sample fit hides). `research_bench` rows are as in
`research/ngboost/reports/README.md`.

New synthetic targets in `research_bench::synth`: `gamma_homo` (Gamma,
shape 4), `gamma_het` (shape 1 + exp(0.5 + 0.7·x₄)), `gamma_shared` (shape
1 + exp(0.5 + 0.7·x₀)), all with mean exp(0.5 + 0.3·μ_a(x)).

Tests added: `tests/test_c18_nu_floor.rs` (own process: sets the floor to
2), unit tests in `src/dist/gamma_mean.rs`.

Machine note: every run of this cycle shared a 12-core laptop on battery
with up to 15 other bench processes (load average 130–240), so the `secs`
columns are not timings.

## Added after approval (same day, working tree on top of 70b314b)

`C18` here is the rebuilt `c18_bench` (variants `lib_xfit`,
`lib_xfit_nocal` = the library's `staged_crossfit_folds = 5`, with and
without recalibration); `RB` the rebuilt `research_bench` (`--dist
studenttfv`, variant token `xfit`). `--leaf` / `--min-leaf` are explicit
in every command, so the files do not depend on the library default.

| File | Command |
|---|---|
| `O2_leaf_sweep_n150.csv`, `O2_leaf_sweep_n300.csv` | `C18 --mode leaf --datasets DS6 --seeds 10 --max-rows {150,300} --leaves 1,5,10,20 --fit default` |
| `O1_oof_n700_leaf20.csv` | `C18 --mode oof --datasets DS6 --seeds 10 --max-rows 700 --leaf 20 --variants lib_default,ts_in_cal,ts_oof,ts_oof_cal` |
| `O1_lib_xfit_leaf20.csv`, `O1_lib_xfit_leaf1.csv` | `C18 --mode oof --datasets DS6 --seeds 10 [--leaf 20] --variants lib_staged,lib_xfit,lib_xfit_nocal` |
| `O1_lib_xfit_n700_leaf20.csv` | `C18 --mode oof --datasets DS6 --seeds 10 --max-rows 700 --leaf 20 --variants lib_default,lib_xfit` |
| `O1_lib_xfit_val30_leaf20.csv` | `C18 --mode oof --datasets DS6 --seeds 10 --max-rows 3000 --val-rows 30 --leaf 20 --variants lib_default,lib_xfit` |
| `O1_lib_xfit_lognormal.csv` | `RB --dist lognormal --variants auto,auto+xfit --datasets lognorm_het,kin8nm,support_cost --seeds 10 --min-leaf 20` |
| `O1_lib_xfit_gammamean.csv` | `RB --dist gammamean --variants auto,auto+xfit --datasets gamma_het,lognorm_het,kin8nm --seeds 10 --min-leaf 20` |
| `O1_lib_xfit_studenttfv.csv` | `RB --dist studenttfv --variants auto,auto+xfit --datasets het_a,tnoise,kin8nm,protein --seeds 10 --min-leaf 20` |
| `O3_fv_<part>.csv` | `RB --dist studenttfv --variants auto+staged --datasets <part> --seeds 10 --max-rows 12000` with the shipped `StudentTFiniteVar`; parts as for `O3_nu*`. Identical to `O3_nu2.1_<part>.csv` in every metric column |
| `O3_fvulp_<part>.csv` | the same with an arithmetically equivalent earlier version of the head (c = e/(2.1+e), θ₂⁰ = ln(exp(ln 3) − 2.1)); kept as a measurement of rounding sensitivity. That version is not in the tree |

Summaries: `python3 -I research/c18_pair.py <file> --by variant --ref
<lib_staged|lib_default|auto> --metrics nll,rmse,cov90`. The O3 files of
different runs share the variant label, so they are paired by a small
ad-hoc script on (dataset, seed); the table is in
`proposals/O3_studentt_nu_floor.md`.

| File | Command |
|---|---|
| `O2_heads_cat3_ml{1,20}.csv` | `RB --dist cat3 --variants base --datasets cat3_nl,cat3_imb --seeds 10 --n-synth 6000 --min-leaf {1,20}` |
| `O2_heads_cat5_ml{1,20}.csv` | `RB --dist cat5 --variants base --datasets cat5_nl --seeds 10 --n-synth 6000 --min-leaf {1,20}` |
| `O2_heads_cat8_ml{1,20}.csv` | `RB --dist cat8 --variants base --datasets support_dzgroup --seeds 10 --min-leaf {1,20}` |
| `O2_heads_hurdle_ml{1,20}.csv` | `RB --dist zero --variants hnormal,hlognormal --datasets zi_lognorm,zi_normal,support_adlsc --seeds 10 --min-leaf {1,20}` |
| `O2_heads_mix2_ml{1,20}.csv` | `RB --dist mix2 --variants auto --datasets bimodal,trimodal,het_a --seeds 10 --min-leaf {1,20}` |

Paired summary: `python3 -I research/optimization-algorithm/leaf_pair.py cat3 cat5 cat8 hurdle mix2`.

## Cycle 19 (2026-11-14)

Same build and conventions. `C18` gained `--mode treeleaf` and `--mode
hpo`; `RB` gained the `NGB_PARAM_DIAG=1` stderr diagnostic (per-column
parameter ranges, the three test rows with the largest squared error, and
one `CAPDIAG` line per fit and k: test NLL and RMSE with the scale column
clamped at reference + ln k; `NGB_CAP_REF=init` takes the model's stored
scale constant as the reference instead of the marginal fit).

| File | Command |
|---|---|
| `O5_capdiag_marginal.csv` | `NGB_PARAM_DIAG=1 RB --dist lognormal --variants auto,auto+xfit --datasets kin8nm,lognorm_het,support_cost --seeds 10 --min-leaf 20 2>&1 >/dev/null \| rg '^CAPDIAG' \| sd '^CAPDIAG ' ''` (header added by hand). The diagnostic reads `pred_param_unbounded`, so it emulates on the model's unclamped sum |
| `O5_capdiag_init.csv` | the same with `NGB_CAP_REF=init`, variants `auto,auto+xfit,joint+auto`, datasets `kin8nm,lognorm_het,gamma_het,support_cost` |
| `O5_ceiling_lognormal.csv` | `RB --dist lognormal --variants auto,auto+xfit,joint+auto --datasets kin8nm,lognorm_het,gamma_het,support_cost --seeds 10 --min-leaf 20` on the shipped code. The uncapped counterpart is `O1_lib_xfit_lognormal.csv`, written before the ceiling existed: the same command on the current tree gives the capped numbers, and the uncapped ones are the `k=inf` rows of the `O5_capdiag` files |
| `O6_leaf_small_n{100,150,200,300}.csv` | `C18 --mode leaf --datasets DS6 --seeds 20 --max-rows {100,150,200,300} --leaves 1,3,5,7,10,14,20 --fit default` |
| `O7_treeleaf_exact_n{300,700,3000}.csv` | `C18 --mode treeleaf --learner exact --datasets DS6 --seeds 10 --max-rows {300,700,3000} --leaves 1,20` |
| `O7_treeleaf_arena_n3000.csv` | the same with `--learner arena --max-rows 3000` |
| `O8_exs_normal.csv` | with `proposals/O8_exact_scale_step.patch` applied: `RB --dist normal --variants auto,auto+exs,auto+xfit,auto+xfit+exs --datasets DS6 --seeds 10 --min-leaf 20 --max-rows 3000` |
| `O8_exs_lognormal_kin8nm.csv` | with the patch applied and before the ceiling existed: `RB --dist lognormal --variants auto,auto+exs,auto+xfit,auto+xfit+exs --datasets kin8nm --seeds 10 --min-leaf 20` |

Summaries: `python3 -I research/optimization-algorithm/cap_summary.py
<O5_capdiag file>`, `leaf_small.py`, `treeleaf.py` (no arguments), and
`c18_pair.py … --by variant --ref auto --metrics nll,rmse,cov90` for the
`RB` files.

The `secs` columns of this cycle are not usable: up to seven benchmark
processes shared the machine.

Test data: the rule of O6 and the learners' default of O7 were fixed
before their sweeps. The ceiling factor of O5 was read off how many rows
reach the ceiling on four datasets (training rows give the same answer as
test rows); it is flat in 3…4 where it matters.

## Cycle 20 (2026-11-14)

`C18` gained `--mode survmean`, `--mode rows`, `--mode noise`, `--weights
tri` for `--mode leaf`, the datasets `lognorm_het`, `pois_*` and `nb_*`
(draw for draw as in `RB`) and `--head` for `--mode hpo`. `RB` gained the
dataset `zi_kin8nm` (kin8nm with a seeded random fifth of the targets set
to 0).

| File | Command |
|---|---|
| `O9_gamma_leaf20.csv` | `RB --dist gamma --variants base --datasets gamma_homo,gamma_het,gamma_shared,lognorm_het,kin8nm,support_cost --seeds 10 --max-rows 12000 --min-leaf 20` (leaf-1 counterpart: `O4_gamma_*.csv`). The diagnosis of single fits: add `NGB_PARAM_DIAG=1`, `--seed0 6 --seeds 1` (gamma_shared) or `--seed0 0 --seeds 1` (kin8nm), `--min-leaf 1` |
| `O9_survival_lognormal_kin8nm.csv` | `C18 --mode survmean --datasets kin8nm --seeds 10` |
| `O9_hurdle_zi_kin8nm.csv` | `RB --dist zero --variants hlognormal,hnormal --datasets zi_kin8nm --seeds 10 --min-leaf 20` |
| `O10_small_<head>_n<N>_ml<leaf>.csv` | `RB --dist <head> --variants <v> --datasets <d> --seeds 20 --n-synth <N> --min-leaf <leaf>`, N ∈ {100, 150, 200}, leaf ∈ {the rule's 7, 10, 14; 20}: poisson `base` on pois_lin,pois_nl,pois_low; negbin `auto` on nb_over,nb_low; bernoulli `base` on bern_lin,bern_nl,bern_imb; cat3 `base` on cat3_nl |
| `O10_weighted_leaf_n{100,150,200}.csv` | `C18 --mode leaf --datasets DS6 --seeds 20 --max-rows <N> --leaves <rule>,20 --fit default --weights tri` |
| `O10_weighted_leaf_n3000.csv` | `C18 --mode leaf --datasets DS6 --seeds 10 --max-rows 3000 --leaves 1,20 --fit default --weights tri` |
| `O11_rows_protein_n700.csv` | `C18 --mode rows --datasets protein --seeds 10 --max-rows 700 --fit default,xfit --leaf 20` |
| `O11_{normal,studenttfv}_protein_n700.csv` | `RB --dist {normal,studenttfv} --variants auto,auto+xfit --datasets protein --seeds 10 --max-rows 700 --min-leaf 20` |
| `O12_noise_normal.csv` | `C18 --mode noise --datasets DS6 --seeds 10 --reps 4 --fit default,xfit --leaf 20 --max-rows 3000` |

Summaries (no arguments, from the repository root): `leaf_small_heads.py`,
`rows_protein.py`, `noise_summary.py`; the weighted-leaf files pair like
`leaf_small.py` (leaf column `msl`); `c18_pair.py` for the `RB` files.

## Cycle 21 (2026-11-14)

`RB`'s hurdle path (`--dist zero`) now scores the RMSE of `model.predict`
instead of `pred_dist(x).mean()`; they differ only for `HurdleLogNormal`.

| File | Command |
|---|---|
| `O13_hurdle_capped.csv` | `RB --dist zero --variants hlognormal,hnormal --datasets zi_lognorm,zi_kin8nm,support_adlsc --seeds 10 --min-leaf 20`. Plain-mean counterparts, written before the change: `O2_heads_hurdle_ml20.csv` (zi_lognorm, support_adlsc) and `O9_hurdle_zi_kin8nm.csv`; the same commands now give the shipped prediction |

## Cycle 22 (2026-11-14)

`C18` gained `--mode binning` and `--mode mixinit`.

| File | Command |
|---|---|
| `O14_binning.csv` | `C18 --mode binning --datasets het_a,kin8nm,protein --seeds 10 --bins 255,32,8 --max-rows 6000` (variants `lib,q,km`) |
| `O14_binning_exp.csv` | the same with `--xtransform exp` (every feature → exp(1.5·x)); protein's rows are NaN for `lib` because its features overflow under that map |
| `O15_mixinit.csv` | `C18 --mode mixinit --seeds 10` |
| `O15_mix2_single_start.csv` | `RB --dist mix2 --variants auto --datasets bimodal,trimodal,het_a,protein --seeds 10 --min-leaf 20` |
| `O15_mix2_multi_start.csv` | with `proposals/O15_mixture_multi_start.patch` applied: the same command under `NGBOOST_MIX_STARTS=1` |

Summaries: `binning_summary.py` (no arguments). The dependency audit:
`cargo tree --features accelerate --prefix none -e normal,build,dev`,
then `rg -l -i 'k-?means|k_means|kmeans|lloyd|centroid|clustering'` over
those crates' directories in `~/.cargo/registry/src/`.


## Cycle 25 (2026-11-14)

`C18` gained `--mode xdef`: the model of
`hyper_opt::default_histogram_model` with no keys, fitted by `fit(x, y)`
on a pool of `--sizes` rows and scored on 2000 rows outside the pool;
arms `off`, `k5`, `k3` set `staged_crossfit_folds`. `HEADS` below stands
for five runs, one per head with its datasets: `normal`
(het_a,het_b,homo,tnoise,kin8nm,protein), `studentt` and `studenttfv`
(het_a,tnoise,kin8nm,protein), `lognormal` (lognorm_het,kin8nm),
`gammamean` (gamma_het,lognorm_het,kin8nm).

| File | Command |
|---|---|
| `O16_xdef_<head>.csv` | `C18 --mode xdef --heads <head> --datasets … --sizes 100,200,400,800,1600,3200,6000 --seed0 100 --seeds 10 --arms off,k5,k3` for each of `HEADS` |
| `O16_xdef2_<head>.csv` | the same with `--sizes 200,300,400,800,1600,3200,6000 --seed0 200` |
| `O16_xdef_tri_{normal,lognormal}.csv` | `C18 --mode xdef --heads normal --datasets het_a,kin8nm,protein` / `--heads lognormal --datasets kin8nm`, `--sizes 800,3200 --seed0 100 --seeds 10 --arms off,k5 --weights tri` |
| `O16_xdef_chrono_{normal,lognormal}.csv` | the same with `--split chrono` instead of `--weights tri` |
| `O16_xdef_large_{normal,studenttfv}.csv` | `--heads normal --datasets het_a,tnoise,protein` / `--heads studenttfv --datasets protein`, `--sizes 20000 --seed0 100 --seeds 5 --arms off,k5,k3` |
| `O16_xdef_serial_{normal,lognormal,studentt}.csv` | `--sizes 800,6000 --seed0 100 --seeds 3 --arms off,k3,k5 --serial 1` on normal (het_a,kin8nm,protein), lognormal (kin8nm), studentt (het_a); run with nothing else on the machine |
| `O16_xdef_stage1_*.txt`, `O16_xdef_stage2_*.txt`, `O16_xdef_both_V40_*.txt`, `O16_xdef_{tri,chrono}_summary.txt`, `O16_xdef_large_k{5,3}.txt` | `xdef_summary.py [--arm k5\|k3] [--rule U\|S\|G --min-size 400] [--cells 1] FILES` |

The `secs` column of every file except the `serial` ones was measured
with twelve fits running at once; use it for ratios only.
`NGB_XDEF_DIAG=1` prints, per fit, the six test rows with the largest
error and their parameters (used for the one LogNormal outlier).

## Cycle 26 (2026-11-14)

`C18` gained `--mode counts`; `load_split` (and so `--mode hpo`) reads the
prepared real count files. The raw data are not in the repository:

- football: `https://www.football-data.co.uk/mmz4281/<season>/<league>.csv`
  for leagues E0 E1 D1 I1 SP1 F1 and seasons 1920 2021 2122 2223 2324
  2425, saved as `<league>_<season>.csv`;
- `https://vincentarelbundock.github.io/Rdatasets/csv/pscl/bioChemists.csv`,
  `…/AER/DoctorVisits.csv`, `…/AER/NMES1988.csv`;
- prepared with `python3 -I prep_counts.py <raw football dir> <raw
  rdatasets dir> ../data_counts/prepared` (the bench's default
  `--data-dir`, relative to the repository root).

`ARMS` = `pois,pois_l1,nb,nb_l1,nb_st,nb_x5,nb_x3`.

| File | Command |
|---|---|
| `O17_counts_chrono.csv` | `C18 --mode counts --design chrono --datasets fbo_home,fbo_away,fbf_home,fbf_away --leagues all,0,1,2,3,4,5 --arms ARMS` |
| `O17_counts_synth.csv` | `C18 --mode counts --datasets pois_goals,nb_goals,nb_over,nb_low,pois_nl --sizes 800,3200 --seed0 300 --seeds 10 --arms ARMS` |
| `O17_counts_real.csv` | `C18 --mode counts --datasets doctor,biochem,nmes --sizes 800,3200,10000 --seed0 300 --seeds 10 --arms ARMS` |
| `O17_counts_football.csv` | `C18 --mode counts --datasets fbo_home,fbo_away,fbf_home,fbf_away --sizes 800,3200,10000 --seed0 300 --seeds 10 --arms ARMS` |
| `O17_counts_auto_{chrono,synth,real,football}.csv` | the four commands above with `--arms auto` (the `staged` column is 1 where `fit_auto` chose NegativeBinomial) |
| `O17_hpo_poisson_<dataset>.csv` | `C18 --mode hpo --head poisson --datasets <dataset> --seed0 300 --seeds 8 --arms untuned,shipped --max-rows 12000` |
| `O17_summary_<arm>_vs_<ref>.txt` | `counts_summary.py --arm <arm> --ref <ref> [--rule 1] FILES` (the `auto` summaries read the eight `counts` files together) |
| `O17_hpo_summary.txt` | `counts_hpo_summary.py reports/O17_hpo_poisson_*.csv` |

Timings were taken with many fits running at once: ratios only.
