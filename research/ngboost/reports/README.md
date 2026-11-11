# Reproducible benchmark runs (2026-11-09)

All runs: `examples/research_bench.rs`, built with
```
export CARGO_TARGET_DIR=/Volumes/Build/ngboost-rs \
       CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
       CARGO_PROFILE_RELEASE_STRIP=false
cargo build --release --features accelerate --example research_bench
B=/Volumes/Build/ngboost-rs/release/examples/research_bench
```
Protocol: per (dataset, seed) shuffle with StdRng(seed), split 70/10/20
train/val/test, `NGBoost<Normal, LogScore, HistogramLearner(depth 3)>`,
natural gradient, binary line search, early stopping on the val split
(patience below), test metrics at the truncated best iteration. Identical
hyperparameters for every variant within a run. Datasets: `het_a`, `het_b`,
`homo`, `tnoise` (synthetic, DGPs in the harness; n = `--n-synth`),
`kin8nm` (UCI, 8192 rows), `protein` (UCI, first `--max-rows` of a seeded
shuffle of 45 730 rows). Data files: `../ngboost/data/uci/`.

Variant tokens: `base`; `fw` (β=1 Fisher-weighted learner); `fwp<β>`
(`fisher_weight_power`); `calib` (library moment recalibration);
`mle`/`affine` (harness post-hoc, val-fitted); `libaffine`
(`NGBoost::calibrate_scale_affine`); `auto` (the 1.2.0 default path —
`base` and every other token force `auto_calibrate_scale = false`); `fwsel`
(β ∈ {0, 0.5, 1} by val NLL); tokens combine with `+`; `icpt`/`noicpt` set `refit_intercept`; `cap<κ>` sets `fisher_weight_cap`. `--val-rows N` caps
the validation split; `--dist normal|normalg|laplace|studentt|lognormal|poisson|negbin|negbincrps|bernoulli|cat3|cat4|cat5|cat8|cat10|mix2|mix2crps|mix3|mix3crps|mix4|automix` picks the head
(generic path: NLL/RMSE/coverage only, tokens base/auto/libaffine/fw/fwp;
count heads: `pois_*`/`nb_*` DGPs, randomized-PIT coverage, `calib` = dispersion
index, `extra` = Σy/Σμ̂, tokens base/fw/fwp/ratio/rate/oracle; `mean_sigma`
holds the geometric-mean r̂ for negbin).

| run | command (after `$B`) |
|---|---|
| R1 | `--variants base,fwp0.25,fwp0.5,fwp0.75,fw,calib,fwp0.5+calib --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R2 | `--variants base,fwp0.5,fw,calib --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100 --max-rows 12000` |
| R3 | `--variants base,fwp0.5,fw,calib --datasets het_a,het_b,homo,tnoise --seeds 5 --seed0 200 --lr 0.05 --n-est 2000 --patience 50 --n-synth 12000` |
| R4 | `--variants base,calib,mle,affine,fwp0.5+affine,fw+affine --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R5 | `--variants base,calib,mle,affine --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100 --max-rows 12000` |
| R6 | `--variants base,calib,mle,affine --datasets het_a,het_b,homo,tnoise --seeds 5 --seed0 200 --lr 0.05 --n-est 2000 --patience 50 --n-synth 12000` |
| R7 | `--variants base,affine,libaffine --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R8 | `--variants base,fwsel,libaffine,fwsel+libaffine --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R9_valrows_N (N = 20,40,80,160) | `--variants base,mle,affine,auto --datasets het_b,kin8nm,tnoise,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000 --val-rows N` |
| R9_valrows_N (N = 400,800) | `--variants base,mle,affine --datasets het_b,kin8nm,tnoise,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000 --val-rows N` |
| R10_laplace / R10_studentt | `--dist laplace` (or `studentt`) `--variants base,auto --datasets het_a,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R11 | `--variants base,libaffine,auto --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R12 | `--dist poisson --variants base,fwp0.25,fwp0.5,fw,ratio,rate,fw+rate,oracle --datasets pois_lin,pois_nl,pois_low,nb_over,nb_low --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 3000` |
| R13 | `--dist poisson --variants base,fwp0.5,fw,rate,oracle --datasets pois_lin,pois_nl,pois_low,nb_over,nb_low --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100 --n-synth 3000` |
| R14 | `--dist poisson --variants base,fwp0.5,fw,rate,oracle --datasets pois_lin,pois_nl,pois_low,nb_over,nb_low --seeds 5 --seed0 200 --lr 0.05 --n-est 2000 --patience 50 --n-synth 12000` |
| R15 | `--dist negbin --variants base,fwp0.5,fw,oracle --datasets nb_over,nb_low,pois_nl,pois_low,pois_lin --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 3000` |
| R16 | `--dist negbin --variants base,fw,oracle --datasets nb_over,nb_low,pois_nl,pois_low --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100 --n-synth 3000` |
| R17 | `--variants base --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` (extra = in-sample z-rms, extra2 = h̄) |
| R18 | `--variants base,fw,fw+cap3,fw+cap10,fw+cap30,fwp0.5,fwp0.5+cap10 --datasets het_a,het_b,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R19a/b/c | `--dist lognormal --variants base,fwp0.5,fw --datasets lognorm_het,kin8nm --seeds 10 …`; `--dist lognormal … --datasets protein --max-rows 12000 --y-shift 0.1`; `--dist normal --variants base,fwp0.5,fw,fw+cap10 --datasets lognorm_het --seeds 10 …` |
| R20/R20b | `--dist bernoulli --variants base,fwp0.5,fw --datasets bern_lin,bern_nl,bern_imb --seeds 10 --lr 0.05 …`; `--variants base,fw --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100` |
| R21/R21b | `--dist poisson --variants base,icpt,fwp0,fwp0+icpt,oracle --datasets pois_lin,pois_nl,pois_low,nb_over,nb_low --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 3000`; `--variants base,icpt,oracle --datasets pois_lin,pois_nl,pois_low,nb_over --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100` (extra2 = train Σy/Σλ̂) |
| R22 | `--dist negbin --variants base,icpt,oracle --datasets nb_over,nb_low,pois_nl,pois_low --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 3000` |
| R23 | `--dist bernoulli --variants base,icpt,fw,fw+icpt --datasets bern_lin,bern_nl,bern_imb --seeds 10 --lr 0.05 --n-est 2000 --patience 50` (extra2 = train mean p̂ / mean y) |
| R24 | `--variants base,icpt --datasets het_a,het_b,homo,tnoise,kin8nm,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` |
| R25 | `cargo run --release --features accelerate --example protein_probe` → `R25_protein_probe.txt` |
| R26/R26b | `--dist bernoulli --variants base,icpt,fw,fw+icpt --datasets bern_lin,bern_nl,bern_imb,bern_imb6,bern_rare --seeds 10 --seed0 300 --lr 0.05 --n-est 2000 --patience 50 --n-synth 12000`; `--variants base,icpt,fw --seeds 5 --seed0 400 --lr 0.01 --n-est 6000 --patience 100 --n-synth 12000` |
| R27/R27b | `--dist bernoulli --variants base,fwp0,fwp0+icpt --datasets sprint_cvd,flchain_death,support_hospdead --seeds 10 --lr 0.05 --n-est 2000 --patience 50`; `--variants base,fwp0 --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100` |
| R28/R28b | `--dist cat3 --variants base,fwp0.5,fwp1,icpt,fwp1+icpt --datasets cat3_nl,cat3_imb --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 6000`; `--dist cat5 --variants base,fwp0.5,fwp1,icpt --datasets cat5_nl …` |
| R29 | Normal: `--variants base,auto --datasets bimodal,protein,kin8nm,het_a,homo,tnoise --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000`; mixture: same with `--dist mix2 --variants base`; R29b: `bimodal,protein --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100` |
| R30/R30b | `--dist mix2 --variants base,auto` on the R29 grid; lr 0.01 `--variants auto` |
| R31/R31b | `--dist cat4 --variants base,fwp0.5,fwp1,fwp1+icpt --datasets support_dzclass,flchain_cause --seeds 10 --lr 0.05 --n-est 2000 --patience 50`; `--variants base,fwp1 --seeds 5 --seed0 100 --lr 0.01 --n-est 6000 --patience 100` |
| R32 | `--dist cat4 --variants fwp0+noicpt,base --datasets support_dzclass --seeds 10 --lr 0.05 --n-est 2000 --patience 50` (default path check) |
| R33/R33b | `--dist cat8 --datasets support_dzgroup`, `--dist cat10 --datasets digits`, `--dist cat3 --datasets wine --seeds 20`, `--variants fwp0+noicpt,fwp1+noicpt,base --seeds 10 --lr 0.05 --n-est 2000 --patience 50`; lr 0.01: `--variants fwp0+noicpt,base --seeds 5 --seed0 100 --n-est 6000 --patience 100` |
| R34 | `NGB_POINT_CSV=R34_points.csv` with `--dist mix2 --variants auto` then `--dist normalg --variants auto`, `--datasets protein,bimodal,trimodal,kin8nm,het_a,homo --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` (point file columns: dataset, seed, variant, RMSE mean/median/mode, MAE mean/median/mode) |
| R35 | `--dist mix2crps --variants auto` on the R34 grid; `--dist mix3crps --variants auto --datasets protein,bimodal,trimodal`; `--dist negbin|negbincrps --variants base --datasets nb_over,nb_low,pois_nl,pois_low --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --n-synth 3000` |
| R36 | `NGB_POINT_CSV=R34_points_mix3.csv --dist mix3 --variants auto` on the R34 grid |
| R37 | `cargo run --release --features accelerate --example mixture_sigma_probe` (first version: boosting path of one drifting row) → `R37_sigma_probe.txt` |
| R37b | same probe, 3 seeds, full protein, guard on (default) / `NGBOOST_MIX_GUARD=0` → `R37b_drift_guard_on.txt` / `_off.txt` |
| R38 | `--dist normalg|mix2|mix3|mix4 --variants auto --datasets protein,bimodal,trimodal,kin8nm,het_a,homo,tnoise --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000`, guard on (`*_guard.csv`) and `NGBOOST_MIX_GUARD=0` (`*_noguard.csv`); column `extra` = validation NLL. K selection: `python3 -I ../select_k.py R38_normalg_guard.csv R38_mix2_guard.csv R38_mix3_guard.csv R38_mix4_guard.csv` |
| R39 | `NGBOOST_MIX_CRPS_FULL=1` (was the default at the time) `--dist mix2crps --variants auto` on the R34 grid; `--dist mix3crps --datasets protein,bimodal,trimodal` |
| R40 | `--dist normalg|mix2|mix3|mix4 --variants auto --datasets flchain_futime,support_logdtime,support_logcost --seeds 10 --lr 0.05 --n-est 2000 --patience 50`; K selection via `../select_k.py` |
| R41 | `--dist mix2crps --variants auto` on the R34 grid; `--dist mix3crps --datasets protein,bimodal,trimodal` (CRPS step guard on; off = R35, reproduced bit-for-bit) |
| R42 | `--dist automix --variants cold,cold+greedy,warm,warm+greedy,misfit --datasets protein,bimodal,trimodal,kin8nm,het_a,homo,tnoise,flchain_futime,support_logdtime,support_logcost --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` (code then had `warm_start: bool`) |
| R43 | same grid, `--variants cold,both,both+greedy` (`KSearch`); summary: `python3 -I ../automix_summary.py R43_automix_both.csv` |
| R44 | `--dist normprobe --variants base,cont,matched,sym2,warm2,warm2d,warm2s,l2,trace --datasets kin8nm,het_a,het_b,homo,tnoise,protein --seeds 10 --lr 0.05 --n-est 2000 --patience 50 --max-rows 12000` (why warm mixtures win; `extra` = moment-matched-Normal NLL for mixtures; `trace` columns in the runner doc) |
| R45 | `--dist normprobe --variants base,cales,pls,twostage,twostage+nocal,warm2` on 10 datasets (R44 set + lognorm_het, support_logdtime, flchain_futime, support_logcost), same flags |
| R46 | `--variants base,twostage,twostage+fgls,twostage+polish`, same grid (guard off) |
| R47 | R46 grid `--variants base,twostage,twostage+fgls` with `NGBOOST_NORMAL_GUARD=5` |
| R48 | `--variants base,twostage+fgls` with `NGBOOST_NORMAL_GUARD=25` (= the new default) |
| R49 | `--dist zero --variants normal,hnormal,hlognormal,hmix2,automix --datasets zi_lognorm,zi_normal,support_adlsc,support_totmcst` (guard off; timed out in totmcst → `R49_hurdle_partial.csv`); R49b totmcst (guard off, negatives kept); **R49c** all four datasets without automix, default guard; **R49d** totmcst with the 2 negative costs dropped. Summary: `python3 -I ../zero_summary.py R49c_hurdle_guard.csv` |
| R50 | `--dist automix --variants both,crps+cold,crps+both`, 6 seeds, split in R50a (kin8nm, 6 seeds kept from a killed 10-seed run), R50b (het_a, homo), R50c (tnoise, bimodal), R50d (het_b). Summary: `python3 -I ../crps_search_summary.py R50a_*.csv R50b_*.csv R50c_*.csv R50d_*.csv` |
| R51 | `--dist normalcrps --variants auto` on the R45 grid, legacy (upstream) Normal CRPS metric |
| R52 | same with the corrected CRPS metric |
| R54a | `--dist normprobe --variants base,staged` (base = joint; staged = the generic `fit_staged`) on the R45 grid, default guard |
| R54b | `--dist laplace --variants auto+joint,auto+staged`, R45 grid |
| R54c | `--dist studentt --variants auto+joint,auto+staged`, 6 seeds, in parts a (kin8nm, het_a, het_b, homo partial), b (lognorm_het, protein partial), c (homo, tnoise, support_logcost); merged in `R54c_studentt_staged_combined.csv`. The first attempt crashed on a statrs NaN panic (`invalid/R54c_*`); a and b were cut by the time limit while the laptop slept |
| R54d | `--dist lognormal --variants auto+joint,auto+staged --datasets lognorm_het,kin8nm,support_cost` |
| R55 | CRPS-trained heads, `--variants auto`, run with and without `NGBOOST_CRPS_METRIC_LEGACY=1` (`*_legacy.csv` / `*_fixed.csv`): `poissoncrps` (pois_lin, pois_nl, pois_low, nb_over; 10 seeds), `halfnormalcrps` (lognorm_het, kin8nm, support_cost; 6), `studenttcrps` (tnoise, het_a, kin8nm; 4), `weibullcrps` (lognorm_het, support_cost; 4). Pair with `../pair.py X_legacy.csv auto X_fixed.csv auto` |
| R56 | `--dist zero --variants normal,hnormal,hnormalcrps,hlognormal,hlognormalcrps` on the four zero-inflated targets, 10 seeds |
| R57 | `--dist automix --variants both`, 5 seeds, kin8nm, het_a, homo, bimodal, protein (staged K = 1 candidate); pair with R43 `both` |
| R54e | `c15/R54e_st_*`: R54c's StudentT grid, seeds 6–9 (protein 5–9); merged with R54c into `c15/R54ce_studentt_staged_10seeds.csv` |
| R55b | `c15/R55b_stcrps_*`: `--dist studenttcrps --variants auto`, tnoise, het_a, kin8nm, 10 seeds; legacy (`NGBOOST_CRPS_METRIC_LEGACY=1`) split into `_part0` (tnoise 0–4) and `*_a`/`*_b` seed ranges, merged as `_legacy_all.csv` |
| R58 | `c15/R58_*`: `--dist gammacrps|weibullcrps --variants auto`, legacy and fixed metric, lognorm_het and support_cost (10 seeds), kin8nm (6; legacy 0–3 Gamma / 0–2 Weibull in the main + `_b` files); `--dist gamma|weibull` LogScore reference (`*_logscore.csv`, 10 seeds) |
| R55c / R58c | `c15/R55c_*`: legacy StudentT CRPS kin8nm seeds 6–8, one process per seed (merged into `R55b_stcrps_legacy_all.csv`). `c15/R58c_*`: Gamma legacy kin8nm seeds 4–9, Weibull legacy 3–9, both fixed 6–9 (merged as `R58_*_kin8nm_all.csv`). A first attempt was stopped on battery; the rerun used `../c15c_ordered.txt` with `../c15_run_guarded.sh` |
| parity | `c15/parity_*`: probe source, 1.1.2 hashes, current-default hashes (see `tests/test_upstream_parity.rs`) |
Summaries: `../c15_summary.sh` (run from `research/ngboost`).

`superseded/` holds the first (broad) guard's R38 runs; rejected, not used.
Reproducibility check (Cycle 11): with the guard and full CRPS metric
switched off the code reproduces R34/R35 bit-for-bit (iterations, NLL, CRPS).

`invalid/` holds a first attempt at R34–R36 whose flags were silently
dropped (zsh does not word-split `$VAR`); superseded, not logged.
Mixture heads are evaluated with their exact CRPS; other generic heads by
numerical integration. Real regression targets added in Cycle 12: `flchain_futime`, `support_logdtime`, `support_logcost` (costs > 0). Real data added in Cycle 10: `support_dzgroup`
(8 classes), `digits` / `wine` (exported from the installed scikit-learn to
`../data/`).

Paired comparisons across two report files (e.g. Normal vs mixture on the
same splits): `python3 -I ../pair.py A.csv variantA B.csv variantB`.
Real datasets: `sprint_cvd`, `flchain_death`, `support_hospdead` (binary),
`support_dzclass`, `flchain_cause` (4-class), from `../ngboost/data/surv/`
with time/post-outcome columns removed.

Each `R*.csv` holds the per-(dataset, seed, variant) rows followed by a
`# SUMMARY` block (mean ± sd, paired Δ vs the first variant with standard
error and seed-win counts). `../log_run.sh <id> <decision> <report>`
appends the rows to `../experiments.csv` with timestamp and commit.
Single fit wall time at these sizes: 0.05–0.6 s, so a full run is minutes.
| R59 / R59r | `rb_c16 --dist negbin --variants base,staged,rcal,staged+rcal` on nb_over,nb_low,pois_nl,pois_lin,pois_low, 10 seeds (R59r: uncorrected stage-3 weights) → `c16/` |
| R60 | `ts_c16 --dist {normal,poisson,negbin}` temporal grid (chrono/random/nocal/joint/staged/refit/loccal/rcal/oracle); synthetic 10 seeds, co2/elnino/sunspots 5 rolling origins → `c16/R60_*` |
| R61 | `ts_c16b --dist normal --variants chrono+joint,chrono+joint+carry` (refit with carried calibration) → `c16/R61_*` |
| R62 | staged vs joint: `ts_c16 --n {600,1000,2000}` and `rb_c16 --dist normalg --variants auto+joint,auto+staged --val-rows {60,150}` → `c16/R62_*` |
| R63 | `rb_c16c --dist negbin --variants base,auto,staged+auto` at default (300) and `--val-rows 100` → `c16/R63_*`; all paired summaries in `c16/c16_summary.txt` (`c16_pair.py`) |
| R64 | `ts_c17 --dist countsel` on tsp_stat,tsp_drift,tsnb_stat,tsnb_drift and iid_pois_nl,iid_pois_lin,iid_pois_low,iid_nb_over,iid_nb_low (10 seeds), sunspots (5) → `c17/R64_*` |
| R65 | `ts_c17b --dist normal --datasets elnino,co2 --variants chrono,chrono+joint,chrono+nocal,chrono+joint+nocal --seeds 10 --te-frac 0.05 --va-frac 0.1`; `rb_c16c --dist normalg --variants auto+joint,auto+staged,joint,staged --max-rows 700` → `c17/R65_*` |
| R66 | `rb_c16c --dist normalg --variants joint,staged` (n = 3000) and `auto+joint,auto+staged --val-rows 30` → `c17/R66_*` |
| R67 | `ts_c17c` refit_full kept vs scaled rounds (`chrono,chrono+refit,chrono+refits`; Normal with `+joint`) → `c17/R67_*`; summaries `c17/c17_summary.txt` |
