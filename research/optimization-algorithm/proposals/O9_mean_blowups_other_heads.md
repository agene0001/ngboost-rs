# O9 — Mean blow-ups in the other heads

Cycle 20, 2026-11-14. O5 bounded the `LogNormal` scale. Three other
places compute a mean that is steeply nonlinear in a fitted parameter:
`Gamma` (mean = α/β, a difference of two trees on the log scale),
`HurdleLogNormal`, and the survival wrapper's `LogNormal`. Do they show
the same failure, and what stops it?

## Gamma

Recorded in Cycle 18 (`reports/O4_gamma_*.csv`, leaf 1): test RMSE 114 on
gamma_shared seed 6 (median 2.9), 4.6 on kin8nm seed 0 (median 0.22), 12.2
on lognorm_het (median 5.4), with ordinary NLL.

Diagnosis (`research_bench`, `NGB_PARAM_DIAG=1`): one test row with
log β = −7.1 (gamma_shared) or −5.3 (kin8nm) against a median of −0.16 or
2.5 — a single-row leaf driving the rate towards zero. It is a leaf-size
artefact, not a property of the parameterisation.

With the 1.3.0 leaf default (`reports/O9_gamma_leaf20.csv`, the same six
datasets × 10 seeds):

| | RMSE median / max, leaf 1 | leaf 20 | NLL leaf 1 → 20 |
|---|---|---|---|
| gamma_homo | 2.62 / 3.67 | 2.55 / 2.86 | 1.745 → 1.731 |
| gamma_het | 2.97 / 3.54 | 2.85 / 3.57 | 1.880 → 1.853 |
| gamma_shared | 2.96 / **114** | 2.76 / 3.18 | 1.848 → 1.836 |
| lognorm_het | 5.40 / **12.2** | 5.15 / 6.57 | 2.298 → 2.266 |
| kin8nm | 0.219 / **4.62** | 0.211 / 0.218 | −0.132 → −0.129 |
| support_cost | 3.81·10⁴ / 4.12·10⁴ | 3.79·10⁴ / 4.12·10⁴ | 10.886 → 10.887 |

No blow-up in 60 fits. Nothing was changed in the head; the regression
test `gamma_mean_stays_sane_with_the_default_learner`
(`tests/test_cycle20.rs`) holds the property on a gamma_shared-like
target. `Gamma` built on a leaf-1 learner (`with_params(…, 1, …)`,
`default_tree_learner()`) is still exposed; `GammaMean` is not.

## Survival wrapper, LogNormal

`c18_bench --mode survmean` (`reports/O9_survival_lognormal_kin8nm.csv`):
`NGBSurvival<LogNormal>` on kin8nm as the event time, 10 seeds, with no
censoring and with 30 % of the rows censored. Largest fitted log σ 0.30,
largest mean/median ratio 2.5, RMSE of the mean 0.177 against 0.176 for
the median. No blow-up. The wrapper's fit is joint — it has neither the
staged scale stages nor cross-fitting, which is where `LogNormal` broke.

## HurdleLogNormal

`research_bench --dist zero --variants hlognormal,hnormal --datasets
zi_kin8nm` (kin8nm with a random fifth of the targets set to 0;
`reports/O9_hurdle_zi_kin8nm.csv`), 10 seeds: RMSE 0.328–0.351 against
0.326–0.347 for `HurdleNormal`. No blow-up. Also a joint fit.

One failure was found while writing the regression test, on a synthetic
bounded target with a point mass at its floor (values clamped at 0.04, so
ln y has an atom at −3.2 and a marginal log-scale near 0.9): one of four
`HurdleLogNormal` fits stopped after 80 rounds with log σ = 1.22 on a few
rows and a largest predicted mean of 31 against a largest target of 1.45.
The plain `LogNormal` (staged, cross-fitted, joint) and the survival
wrapper were sane on the same data.

Delegating the scale ceiling to the hurdle head would not have helped:
its reference is the marginal log-scale, the ceiling would sit at
log σ = 1.28, and the bad rows are at 1.22. This is the limit already
stated for O5 — the ceiling bounds σ relative to a baseline, and with a
baseline near 1 a σ within the bound still gives a mean hundreds of times
the median.

## Decision

- `Gamma`: closed by the leaf default; regression test added.
- Survival `LogNormal`: not affected; regression test added.
- `HurdleLogNormal`: not affected on the benchmark case; one failure on a
  constructed floor-atom target, **not fixed** — no bound relative to the
  model's own scale addresses it, and a bound on the mean would need a
  stored target range. Recorded as `docs/known_limitations.md` 66.
  **Superseded the same day:** the user approved the stored number and
  the format change; see `O13_hurdle_capped_prediction.md`.

## Next question

A point prediction that does not depend on the upper tail — the median,
or the mean of the distribution truncated at the largest training target
— for the log-normal family. The second needs one stored number.
