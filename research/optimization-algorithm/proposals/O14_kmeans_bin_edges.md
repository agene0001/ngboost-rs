# O14 — k-means bin edges for the histogram trees (rejected)

Cycle 22, 2026-11-14. Asked by the user after Meta FAIR's e-means paper
(arXiv 2611.09501): is there a k-means anywhere in this crate or in a
crate it calls that e-means could replace — and do the two places where a
clustering step *could* sit (feature binning, the mixture head's start;
O15) gain from one?

## Dependency audit

`cargo tree -e normal,build,dev` lists 100 crates besides this one. None
of their sources contains a k-means, a Lloyd iteration, a centroid or a
clustering routine (`rg -i 'k-?means|k_means|kmeans|lloyd|centroid|
clustering'` over the registry sources; the only two text hits are
people called Lloyd in a credit line and a release note). The tuner's
`tpe` crate uses Parzen and histogram density estimators. This crate has
none either. There is nothing for e-means to replace.

## Baseline and candidate

Baseline (`compute_bin_edges`): at most 255 edges per feature, taken as
equally spaced order statistics of the distinct training values — an
equal-count (quantile) partition for a continuous feature. Computed once
per fit by one sort per feature.

Candidate: edges from 1-D k-means on the training values (the scheme of
scikit-learn's `KBinsDiscretizer(strategy="kmeans")`): Lloyd's algorithm
started from the quantile partition. It minimises the within-bin variance
of the feature, so it spends bins where a feature is spread out and saves
them where it is dense.

e-means is an *optimiser* for the k-means objective at a scale (billions
of vectors in a thousand dimensions) where a Lloyd pass is expensive. In
one dimension a cluster is a contiguous range of the sorted sample, a
Lloyd pass is k prefix-sum lookups, and it converges in a few dozen passes
on any column this crate will see. So the only real question is whether
the k-means *objective* gives better edges; if it did, plain Lloyd would
deliver it.

One property is known in advance. A tree's splits depend only on the
order of a feature's values, and so do quantile edges: the model is
invariant to any strictly increasing transformation of a feature.
k-means edges depend on distances and are not.

## Test

`c18_bench --mode binning` (`reports/O14_binning.csv`,
`reports/O14_binning_exp.csv`, `binning_summary.py`). The harness
replaces every feature by its bin index under edges learned on the
training rows; with at most 255 codes the library gives each code its own
bin, so both rules go through the same tree builder. `q`, the library's
own rule applied this way, is a control. Normal head, default staged fit,
leaf 20, 6000 rows, 10 seeds, 255 / 32 / 8 bins.

- Control: `q` equals the library's own binning in 90 of 90 fits (score
  and rounds).
- k-means against quantile edges, held-out NLL (se), seeds lower:

| | 255 bins | 32 bins | 8 bins |
|---|---|---|---|
| het_a | +0.003 (0.010) 4 | −0.008 (0.020) 4 | −0.028 (0.011) 8 |
| kin8nm | −0.001 (0.005) 4 | −0.002 (0.008) 6 | +0.014 (0.008) 3 |
| protein | −0.021 (0.015) 7 | +0.062 (0.216) 4 | −0.028 (0.128) 9 |

  Nothing at the default 255 bins; nothing consistent below.
- The same data after x → exp(1.5·x) on every feature (het_a, kin8nm;
  protein's features overflow under this map and are left out):

| | 255 bins | 32 bins | 8 bins |
|---|---|---|---|
| quantile edges | identical fits, 20 of 20 | identical, 20 of 20 | identical, 20 of 20 |
| k-means edges, het_a | −0.007 | −0.021 | **+0.342** |
| k-means edges, kin8nm | −0.010 | −0.001 | **+0.154** |

  (change in mean NLL against the untransformed features). With few bins
  k-means puts them in the stretched tail and starves the dense region:
  RMSE +29 % and +12 %.

## Decision

Rejected. No gain at the default resolution, and a dependence on the
scale of each feature that trees are otherwise free of. With no k-means
objective worth having, e-means has no role here.

## A by-product worth a follow-up

The number of bins itself matters more than how the edges are placed,
and in a data-dependent direction (library binning, mean NLL):

| | 255 bins | 32 bins | 8 bins |
|---|---|---|---|
| kin8nm | −0.527 | −0.584 | −0.603 |
| het_a | 1.314 | 1.307 | 1.434 |
| protein | 2.988 | 3.050 | 3.109 |

Coarser bins act as a regulariser on kin8nm (−0.06…−0.08, far above the
noise of O12) and cost resolution on protein. `max_bins` is not in the
tuner's default search space; this is a candidate for it, not a default
change. **Done the same day** (automl A8): the key is in
`default_search_space` after a search with it beat the same search
without it by 0.009 pooled (41 of 64 searches), all of it on kin8nm.
