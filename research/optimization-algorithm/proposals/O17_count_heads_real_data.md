# O17 — Count heads on real low-count data; cross-fitted staged NegativeBinomial

Status: test plan, written 2026-11-14 **before any model run of this
cycle**. Results go below the line "Results". No library default is
changed by this cycle without the user's approval.

## Why

The user's own models use the Poisson, NegativeBinomial and Normal heads
(sports betting). Two gaps for the count heads:

1. Every count benchmark so far is synthetic with mean counts of about
   3–7 (`pois_*`, `nb_*`). Goals-type targets have means of 1–1.5.
2. The staged NegativeBinomial fit is opt-in since Cycle 16
   (known limitation 51): it improved the mean but fitted r to in-sample
   residuals of that better mean, so it was overconfident (nb_over +0.06
   NLL, coverage 0.79); with the recalibration it only matched the
   recalibration alone at 2–3× the cost. Cross-fitted residuals (O1) are
   the remedy for exactly that failure and were never tried on this head.

## Questions

- **Q1 (rule below).** Is the staged fit with cross-fitted residuals
  (`staged_fit = Some(true)`, `staged_crossfit_folds = 5`) better than
  the NegativeBinomial default?
- **Q2 (descriptive).** On real low-count data, does the branch's leaf
  default (20, adaptive) beat the 1.2.0 leaf of 1 for Poisson and
  NegativeBinomial, as it did on synthetic data (O2: −0.043, −0.061, ≈ 0)?
- **Q3 (descriptive).** Poisson against NegativeBinomial on those data.

## Data

Real (prepared by `prep_counts.py`; raw files downloaded 2026-11-14 and
kept outside the repository in `../data_counts/`):

| name | rows | mean | var/mean | zeros | what |
|---|---|---|---|---|---|
| `fbo_home`, `fbo_away` | 14 017 | 1.50, 1.24 | 1.10, 1.07 | 23 %, 30 % | home / away goals per match, six European leagues, seasons 2019/20–2024/25 (football-data.co.uk); features: market odds before the match (not closing) and team form |
| `fbf_home`, `fbf_away` | 14 019 | same | same | same | the same matches with no odds: Elo ratings and form only |
| `doctor` | 5190 | 0.30 | 2.11 | 80 % | doctor visits (AER `DoctorVisits`) |
| `biochem` | 915 | 1.69 | 2.19 | 30 % | articles per biochemist (pscl `bioChemists`) |
| `nmes` | 4406 | 5.77 | 7.91 | 16 % | physician office visits (AER `NMES1988`; the other visit counts of the same survey year are dropped) |

Synthetic, new, at goals-like rates: `pois_goals` (Poisson, mean about
0.3–4.5, median about 1.2, log-linear with one interaction) and
`nb_goals` (the same mean, r = 6).

Every football feature uses only matches before the row's match.

## Design

`c18_bench --mode counts`. Model: `default_histogram_model` with no keys
(learning rate 0.05, depth 3, leaf adaptive, early stopping), fitted by
`fit(x, y)`.

- **iid design** (all datasets): random shuffle per seed, test = 20 % of
  the rows (at most 2000), pool = the rest, capped at `--size`; pools of
  800 and 3200 rows and, for football, 10 000. Seeds 300–309 (never used
  before). The fit holds out 10 % at random.
- **chronological design** (football only): rolling origin by season.
  Test = one season (2021/22 … 2024/25), training = every earlier season;
  `ValidationSplit::Chronological`. Pooled over the six leagues (4 cells
  per dataset) and per league (24 cells per dataset). These fits are
  deterministic, so a cell is one observation.

Arms: `pois` (Poisson default), `pois_l1` (leaf 1), `nb`
(NegativeBinomial default), `nb_l1`, `nb_st` (staged, in-sample), `nb_x5`
and `nb_x3` (staged, cross-fitted).

Metrics: held-out NLL (primary), RMSE of the mean, Σy/Σμ̂ on test, 90 %
interval coverage, rounds, wall time.

## Decision rule for Q1 (fixed before the runs)

`nb_x5` against `nb`, paired on dataset, design, cell, size and seed.
Standard errors are over seeds (iid) or cells (chronological) within a
dataset, and over dataset means for pooled figures. A count NLL moves
less than a Normal one (the whole recalibration default was worth
0.002–0.026), so the thresholds are smaller than in O16.

Recommend cross-fitted staging as the NegativeBinomial default only if:

- **R1 benefit.** Pooled NLL difference ≤ −0.005 and at least 2 standard
  errors below zero over the dataset means, and lower on at least 60 % of
  paired fits — on the real datasets alone as well as on all datasets.
- **R2 no harm.** No dataset worse by more than 0.005 at more than 2
  standard errors, in either design.
- **R3 point predictions.** Pooled relative RMSE change at most +0.5 %;
  no dataset worse by more than 1 % at more than 2 standard errors.
- **R4 robustness.** No fit error or non-finite output that `nb` does not
  also have.

If the rule passes, `nb_x3` replaces `nb_x5` in the recommendation when
its pooled NLL is within 0.002 and it passes R1–R4. If it passes on these
seeds, it is confirmed on seeds 400–409 before being recommended. Fit
time is reported, not gated.

Q2 and Q3 have no rule: the tables are reported as measured.

## Follow-ups added after the main run, before running them

Both descriptive, no rule; added because the main run showed Poisson
ahead of NegativeBinomial on every football row.

- **F1.** Does `CountModel::fit_auto` (Poisson, switched to
  NegativeBinomial when the validation dispersion test fires) choose the
  better head on these data? Arm `auto`, same datasets, designs and seeds;
  its validation rows are the last 10 % of the pool.
- **F2.** Does the shipped tuner search help the Poisson head on football
  goals? `c18_bench --mode hpo --head poisson`, arms `untuned` and
  `shipped`, 30 trials, 5 folds, seeds 300–307, random 70/10/20 split of
  12 000 matches. Expectation from A7's weak-signal Poisson set (a tie):
  little or nothing.

## Results

2814 fits in the main run and 402 `auto` fits, none failed. Files:
`reports/O17_counts_{chrono,synth,real,football}.csv`,
`reports/O17_counts_auto_*.csv`, `reports/O17_hpo_poisson_*.csv`;
summaries `reports/O17_summary_<arm>_vs_<ref>.txt` (`counts_summary.py`)
and `reports/O17_hpo_summary.txt`. A row below is one dataset in one
design (11 real, 5 synthetic).

### Scale of the problem on football goals

Rolling-origin NLL, six leagues pooled, mean of four test seasons:

| | home goals | away goals |
|---|---|---|
| constant-rate Poisson (no features) | 1.571 | 1.445 |
| Poisson head, form and Elo only | 1.513 | 1.399 |
| Poisson head, with market odds | 1.499 | 1.385 |
| largest gap between any two of the seven fit arms, same features | 0.007 | 0.004 |

Features are worth 0.05–0.07 NLL and the odds another 0.014; every
choice about *how* the model is fitted moves the score by a tenth of
that.

### Q1 — cross-fitted staged NegativeBinomial: the rule fails

`nb_x5 − nb`:

| scope | NLL (se over rows) | fits lower | RMSE |
|---|---|---|---|
| real (11 rows) | −0.0021 (0.0012), 1.7 se | 181/302 (60 %) | +0.03 % |
| synthetic (5 rows) | −0.0138 (0.0073) | 66/100 | −0.4 % |
| all (16 rows) | −0.0057 (0.0027), 2.2 se | 247/402 (61 %) | −0.1 % |

**R1 fails on the real data** (needs ≤ −0.005 at 2 se); R2–R4 pass.

- Where it helps is strong over-dispersion or a strongly varying rate:
  nb_over −0.036 (19/20, RMSE −2.9 %), pois_nl −0.027 (15/20).
- Football: home goals −0.003 … −0.006 in all four rows (2.4–2.8 se in
  three); away goals +0.004, +0.001, +0.003, −0.003.
- doctor −0.003, biochem −0.009 (0.010), nmes +0.003; nb_goals +0.001,
  nb_low −0.002, pois_goals −0.005.
- Cost: 4–14× the default NegativeBinomial fit on the real sets, 16–36×
  on nb_over and pois_nl (contended timings).
- `nb_x3` is the same picture (real −0.0020, all −0.0053). The in-sample
  staged fit (`nb_st`) repeats Cycle 16: +0.113 on nb_over, +0.013 on
  nmes (2.9 se), −0.005 … −0.007 on home goals.

No confirmation run was made: the rule did not pass. The staged
NegativeBinomial fit stays opt-in, with or without cross-fitting.

### Q2 — leaf default against leaf 1

| | real, NLL (se) | fits lower | RMSE | synthetic |
|---|---|---|---|---|
| Poisson, default − leaf 1 | −0.0024 (0.0009), 2.7 se | 171/302 (57 %) | −0.2 % | −0.026: nb_over −0.058, pois_nl −0.065, rest ≤ 0.004 |
| NegativeBinomial | −0.0017 (0.0010), 1.6 se | 177/302 (59 %) | −0.2 % | −0.011: pois_nl −0.043, rest ≤ 0.006 |

On the eight football rows the Poisson difference is −0.0001 … −0.0026
and the NegativeBinomial one −0.006 … +0.000: a small gain or nothing,
never a loss. Largest real gains: nmes −0.010, biochem −0.005, doctor
−0.004 (Poisson). One adverse row: NegativeBinomial on biochem, +0.007
(1.8 se, 732 rows). The synthetic evidence (O2) overstated what the leaf
default is worth on low-count data, because its gains came from the two
high-rate, strongly nonlinear sets.

### Q3 — Poisson or NegativeBinomial

`nb − pois`, NLL (se), fits where NegativeBinomial is lower:

| data | var/mean | NLL | lower |
|---|---|---|---|
| football, 8 rows | 1.07–1.10 | **+0.001 … +0.007** (2–3.6 se in six rows) | 4–11 of 28–30 |
| doctor | 2.1 | −0.024 (0.003) | 30/30 |
| biochem | 2.2 | −0.076 (0.013) | 9/10 |
| nmes | 7.9 | −1.50 (0.03) | 30/30 |
| nb_goals (r = 6) | | −0.012 | 18/20 |
| pois_goals | 1 | +0.006 (0.004) | 6/20 |

Goals are close enough to Poisson that the NegativeBinomial head costs
0.001–0.007 (limitation 4: its dispersion converges slowly when the data
are nearly Poisson). Cross-fitted staging removes about half of that
cost on average (`nb_x5 − pois` +0.001 … +0.005) but never beats Poisson
there.

### F1 — the automatic switch picks correctly

`CountModel::fit_auto` chose NegativeBinomial on 13 of 232 football fits,
30/30 on nmes, 9/10 on biochem, 24/30 on doctor, 20/20 on nb_over and
nb_low, 16/20 on nb_goals, 1/20 on pois_goals, and 18/20 on pois_nl (the
known false alarm on a strongly nonlinear true-Poisson rate). Against the
better head of each dataset it loses nothing measurable: football
`auto − pois` −0.002 … +0.003 (none significant), nmes `auto − nb`
−0.002, biochem +0.001, doctor +0.007 (0.005; the six fits it left on
Poisson).

### F2 — tuning does not help on football goals

Poisson head, shipped search (30 trials, 5 folds, 27–33 s) against the
untuned default, 8 seeds: −0.0016, +0.0005, +0.0006, −0.0000 on
fbf_away, fbf_home, fbo_away, fbo_home; pooled −0.0001 (0.0005). The
winners are shallow and heavily regularised (median depth 2–3, leaf
31–111), i.e. close to what the default already is on this signal.

### Reading

1. For a goals-type target the head matters and the fit options do not:
   use Poisson (or `fit_auto`, which chooses it). Nothing tested here —
   leaf size, staging, cross-fitting, tuning — moves the football score
   by more than 0.007, against 0.05–0.07 for the features.
2. The branch's defaults are safe for this use case (never worse) and
   worth little on it (≤ 0.003).
3. NegativeBinomial earns its place only with real over-dispersion
   (var/mean of 2 or more here), where it is worth 0.02–1.5.

Not tested: basketball- or run-scale counts, the Normal head on point
totals or margins, the user's own features, a time-decayed fit.

### Scope of these conclusions (added 2026-11-15, after the user's caution)

The football features are basic — market odds, recent form, Elo — with
no feature engineering. What follows from that:

- **Reading 1 is conditional on signal strength.** "The fit options do
  not matter" was measured on a target where the features explain 0.05–
  0.07 NLL. On the two synthetic sets with a strongly varying rate the
  same options are worth 0.03–0.07 (leaf: −0.058, −0.065; cross-fitted
  staged NegativeBinomial: −0.036, −0.027; tuner on pois_nl in A7:
  −0.056). Stronger features move a real target toward that regime; how
  far was not measured. Within football, the richer feature set (with
  odds) showed no more sensitivity to the fit than the poorer one, which
  is weak evidence that goals stay in the low-signal regime — market
  odds are close to the best single feature there is — but it is one
  sport and that argument is reasoning, not a measurement.
- **Poisson over NegativeBinomial on goals is the robust one.** Better
  features lower the conditional variance, so the residual
  over-dispersion can only shrink.
- **The only decision of this cycle was a rejection**, and it does not
  rest on the football data alone: on the five synthetic rows the
  cross-fitted staged fit is also below two standard errors (−0.0138,
  1.9 se), its gain sits in 2 of 16 rows, and it costs 4–36× the fit
  time. It remains available as an opt-in and is worth trying on a
  strongly over-dispersed or strongly varying count target.
- **No default was accepted on the strength of these data.** The leaf
  default, the tuner space and the cross-fitting rule were decided in
  Cycles 18–25 on other data; Cycle 26 only checked that the defaults do
  no harm here.

**Decision: nothing changed in the library.**
