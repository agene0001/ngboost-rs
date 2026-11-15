# O10 — The adaptive leaf on other heads and with sample weights

Cycle 20, 2026-11-14. O6 measured the rule min(20, n/10) on the Normal
head, unweighted. This is the same comparison — the rule's leaf (7, 10,
14) against a fixed 20 at 70, 105 and 140 fitted rows, 20 seeds — on
count and classification heads and on weighted Normal fits.

## Count and classification heads

`research_bench`, `--n-synth {100,150,200}`, `--min-leaf {rule, 20}`
(`reports/O10_small_<head>_n<N>_ml<leaf>.csv`, `leaf_small_heads.py`).
Held-out NLL of the rule minus leaf 20 (se), seeds lower of 20:

| head, dataset | 70 rows | 105 rows | 140 rows |
|---|---|---|---|
| Poisson, pois_nl (strong nonlinear rate) | **−0.599** (0.121) 19 | **−0.255** (0.054) 17 | +0.021 (0.076) 13 |
| Poisson, pois_lin | +0.018 (0.034) 10 | +0.017 (0.023) 10 | +0.001 (0.017) 9 |
| Poisson, pois_low (weak signal) | **+0.048** (0.019) 5 | **+0.044** (0.014) 4 | +0.005 (0.012) 9 |
| NegativeBinomial, nb_over | −0.129 (0.104) 13 | +0.118 (0.107) 11 | −0.015 (0.016) 10 |
| NegativeBinomial, nb_low | −0.000 (0.021) 10 | −0.029 (0.026) 9 | +0.015 (0.012) 13 |
| Bernoulli, bern_lin | −0.062 (0.105) 12 | **+0.026** (0.012) 6 | +0.002 (0.015) 7 |
| Bernoulli, bern_nl | −0.018 (0.020) 15 | +0.013 (0.017) 9 | +0.019 (0.029) 9 |
| Bernoulli, bern_imb | −0.025 (0.094) 12 | +0.119 (0.077) 8 | −0.002 (0.004) 12 |
| Categorical K = 3, cat3_nl | −0.016 (0.036) 12 | +0.022 (0.024) 9 | +0.012 (0.015) 7 |

NLL is lower with the rule on 280 of 540 fits.

## Weighted Normal fits

`c18_bench --mode leaf --weights tri` (every training and validation row
weighted 0.25, 1 or 4 at random; test scored unweighted;
`reports/O10_weighted_leaf_n<N>.csv`), six datasets × 20 seeds:

| rows fitted on | pooled RMSE | NLL lower on |
|---|---|---|
| 70 | −5.0 % (1.1); het_b −19.8 % | 57 of 120 |
| 105 | −0.4 % (0.7); het_b −10.3 % | 66 of 120 |
| 140 | −0.3 % (0.5) | 60 of 120 |

And the leaf-20 default itself under weights, 2100 rows, 10 seeds: leaf
20 against leaf 1 is −0.041, −0.027, −0.091, −0.031 NLL on het_a, het_b,
tnoise, kin8nm (8–10 of 10), +0.068 (0.051) on homo and +0.005 (0.011) on
protein; lower on 45 of 60 fits.

## Reading

The rule is not a general improvement on these heads. It is a trade:

- where the target has a strong nonlinear signal that a three- to
  five-leaf tree cannot express, it gains a lot (pois_nl −0.25…−0.60 NLL
  and −8…−15 % RMSE; het_b in O6 and here);
- where the signal is weak or linear, the larger leaf was the better
  regulariser and the rule costs a little (pois_low +0.044…+0.048, about
  3 se; bern_lin +0.026 at 105 rows, 2 se);
- everywhere else it is noise.

The gains are five to ten times the size of the losses, and both vanish
by 140 rows. The leaf-20 default holds under sample weights.

## Decision

Keep the adaptive default. The two significant losses are recorded in
`docs/known_limitations.md` 62, with the opt-out
(`HistogramLearner::with_params(depth, 255, 20, 2)`) for small, weak-signal
count data. No rule was re-tuned on these results.
