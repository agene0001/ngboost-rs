# O6 — Leaf size that adapts to small data

Cycle 19, 2026-11-14. Follow-up to O2 (`O2_leaf_regularisation.md`), whose
small-data check found the leaf-20 default a wash at about 100 training
rows, with leaf 5–10 better on one dataset.

## Observed limitation

`min_samples_leaf = 20` on n rows allows at most n/20 leaves per tree: 3 on
70 rows, 5 on 105. A depth-3 tree then cannot use its depth, and a target
whose mean depends strongly on one feature (het_b) is under-fitted: RMSE
+13 % at 105 rows and +25 % at 70 against leaf 7–10.

## Baseline and candidate

Baseline: a fixed leaf of 20 (the 1.3.0 default from O2).

Candidate, stated before the sweep below was run (it is what O2's 105-row
table suggested):

    leaf(n) = min(20, max(1, floor(n / 10)))

with n the number of rows the tree is fitted on. From 200 rows up nothing
changes. It keeps the same *relative* resolution on small data that leaf
20 gives at 200 rows (at most ten leaves per tree).

No convergence argument is affected: the rule only changes which
partitions the least-squares tree may choose.

## Test

`c18_bench --mode leaf`, Normal head, default staged/joint fit, depth 3,
learning rate 0.05; six datasets × **20 seeds** (seeds 10–19 had not been
used in O2); total sizes 100, 150, 200 and 300 rows (70 % fitted on: 70,
105, 140, 210); leaves 1, 3, 5, 7, 10, 14, 20. The rule's value at each
size is one of the swept leaves (7, 10, 14, 20), so no interpolation is
needed. `leaf_small.py` produces the tables.

Rule against a fixed 20, 20 seeds per dataset:

| rows fitted on | rule | pooled RMSE | NLL lower on | per dataset |
|---|---|---|---|---|
| 70 | 7 | −5.0 % (se 1.2), 77/120 | 73/120, median −0.032 | het_b −24.5 % RMSE (20/20), tnoise −4.5 %; others within ±1.2 % |
| 105 | 10 | −1.7 % (0.8), 61/120 | 66/120, median −0.007 | het_b −13.2 % (18/20), protein −1.4 % (15/20); kin8nm +2.3 % (1.6) |
| 140 | 14 | −1.2 % (0.5), 74/120 | 68/120, median −0.007 | het_b −4.4 % (14/20); others within ±1.1 % |
| 210 | 20 | identical | — | every smaller leaf lost to 20 (mean NLL +0.013…+0.048) |

Mean NLL differences are dominated by a handful of diverging fits on
protein (standard errors 0.3) and are not reported as evidence; medians
and win counts are. Across the three sizes NLL is lower on 207 of 360
fits (57.5 %).

## Reading

- One dataset (het_b) carries most of the gain, and it is large there.
  The other five are within noise in both directions; none is
  significantly worse (largest adverse: kin8nm RMSE +2.3 %, 1.4 se, at
  105 rows; het_a NLL +0.019, 0.8 se).
- The fresh seeds alone do not confirm an NLL gain at 105 rows (+0.005
  mean over datasets, against −0.034 on the seeds O2 had used). The RMSE
  gain on het_b holds on both halves.
- At 210 rows the ordering is monotone in favour of 20, which is why the
  rule is capped there and not, say, at n/5.

Simpler competing explanation: "any smaller leaf helps on tiny data". Not
quite. Leaf 1 is not better than 20 at 105 rows (NLL +0.018) or at 140
(RMSE +0.014; its mean NLL of −0.024 comes from protein's diverging
fits). No fixed leaf is best at all three sizes — by mean RMSE it is 7 at
70 rows, 14 at 105 and 7 at 140 — and the rule is within 0.013 RMSE of
the best one at each.

## Decision (approved by the user, implemented 2026-11-14)

`HistogramLearner::new` / `default_histogram` set a new public field
`adaptive_leaf = true`; a tree fitted on n rows then uses
`adaptive_min_samples_leaf(min_samples_leaf, n)`. `with_params` sets it to
false, so an explicit leaf size is used exactly as given and the leaf-1
opt-out of the upstream parity test is unchanged. The exact learners get
the same field (O7). `hyper_opt::default_histogram_model` uses the
adaptive default only when no `min_samples_leaf` key is present.

"Rows the tree is fitted on" is what the learner is handed: after the
validation split (10 % by default), a minibatch if `minibatch_frac < 1`,
and four fifths of the rows inside `hyper_opt` folds or cross-fit folds.

## Limits

Normal head only; six datasets; depth 3; learning rate 0.05. Sample
weights, classification and count heads were not measured at these sizes.
The rule was fixed in advance, but the cap of 20 and the divisor of 10
were not searched, so a better rule may exist.
