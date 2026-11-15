# O15 — The mixture head's EM start: quantile groups, k-means, restarts (no change)

Cycle 22, 2026-11-14. Second place where a clustering step could sit (see
O14 for the question and the dependency audit).

## Baseline and candidates

`NormalMixture<K>::fit` runs EM on the marginal target, started from K
equal-count groups of the sorted sample (group means and standard
deviations, equal weights).

- Candidate A: start from 1-D k-means groups instead (Lloyd from the
  quantile partition; group sizes as weights). k-means is the usual EM
  initialiser; in one dimension it is exact enough and instant, so e-means
  has nothing to speed up.
- Candidate B: several starts, keep the highest likelihood. As a
  reference, the best of 30 random starts; as a deterministic prototype,
  every split of the sorted sample at deciles
  (`O15_mixture_multi_start.patch`, behind `NGBOOST_MIX_STARTS=1`; not in
  the tree).

## Marginal fits

`c18_bench --mode mixinit` (`reports/O15_mixinit.csv`): eight targets —
four synthetic mixtures (two separated modes; a 5 % component far from
the bulk; a narrow mode inside a wide one; three modes of unequal size)
and the targets of protein, kin8nm, het_a and tnoise — K = 2 and 3, 10
seeds, 2100 rows, scored on fresh rows.

- This file's copy of the EM loop reproduces the library's fit exactly
  (difference 0 in 160 cases).
- **k-means start against quantile start: the same fit.** Mean difference
  in held-out log-likelihood at most 0.001 on any target, largest single
  case 0.007. EM walks to the same optimum from both, including the
  5 % component that an equal-count split does not isolate.
- Best of 30 random starts: equal everywhere except one case — K = 2 on
  the three-mode target, **+0.075** per row on all 10 seeds. Fitting two
  components to three modes has two distinct optima; the quantile start
  and the k-means start both land in the worse one.

## End to end

Does that one case matter after boosting? `research_bench --dist mix2
--variants auto`, leaf 20, 10 seeds, with and without the multi-start
prototype (`reports/O15_mix2_single_start.csv`,
`reports/O15_mix2_multi_start.csv`):

| dataset | fits changed | NLL, multi − single |
|---|---|---|
| bimodal | 0 of 10 | 0 |
| trimodal (three modes, K = 2) | 0 of 10 | 0 |
| protein | 0 of 10 | 0 |
| het_a | 10 of 10 | −0.014 (0.027), 5 of 10 |

The refactor needed for the prototype left the single-start fits
bit-identical to `O2_heads_mix2_ml20.csv` (30 of 30). On the benchmark's
own three-mode target the decile starts found nothing better; on het_a
they found a marginally higher likelihood for a unimodal target and the
end result is noise.

## Decision

No change. A k-means start is equivalent to the current one; several
starts help a marginal fit in one constructed case and nothing end to
end. The start is only the constant the trees begin from — the boosting
moves every component per row afterwards.
