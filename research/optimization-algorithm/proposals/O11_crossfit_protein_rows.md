# O11 — Where cross-fitting loses on protein

Cycle 20, 2026-11-14. Open since O1: with 490 training rows the
cross-fitted staged Normal fit is 0.069 NLL worse than the default on
protein (4 of 10 seeds better), while it wins on the other five datasets.

## Row-level comparison

`c18_bench --mode rows --datasets protein --max-rows 700 --fit
default,xfit --leaf 20` writes μ, σ and the score of every test row for
both fits (`reports/O11_rows_protein_n700.csv`, `rows_protein.py`);
1410 test rows over 10 seeds.

- Mean difference +0.069; **median −0.010**; cross-fitting is better on
  744 of 1410 rows.
- The 5 worst rows carry +0.072. Without them the mean is −0.003; without
  the 20 worst it is −0.042.
- Those rows are targets of 14–21 where both fits predict 2–6: the far
  mode of protein's two-humped target, predicted in the near mode. The
  default fit gives them σ = 3–6 (|z| 2–5); the cross-fitted fit, whose
  scale is sharper there (σ = 1.6–3.3), has |z| 5–10. One row alone costs
  40 nats.
- By the default fit's |z|: rows with |z| < 1 gain 0.053 each (−0.034 in
  total), rows with |z| ≥ 2 lose 0.84–5.0 each (+0.096 in total).

So the scale is not worse — it is better on most rows and in the median.
The Gaussian tail turns a handful of wrong-mode rows into the whole
average. That is a misspecified head meeting an honest σ, not a defect of
the residuals; the earlier guess ("tail rows") holds, row by row.

## Check: a head with heavier tails

`research_bench`, protein, 700 rows, 10 seeds, leaf 20
(`reports/O11_normal_protein_n700.csv`,
`reports/O11_studenttfv_protein_n700.csv`):

| head | default NLL | cross-fit − default | seeds lower |
|---|---|---|---|
| Normal | 3.080 | +0.069 (0.031) | 4 of 10 |
| StudentTFiniteVar | 3.039 | +0.004 (0.019) | 5 of 10 |

With a heavy-tailed head the loss is gone (and the head is 0.04 better to
begin with). Cross-fitting does not help there either.

## Decision

Nothing to change in the cross-fit code. Cross-fitting stays opt-in; the
guidance in `docs/known_limitations.md` 58 now says why it loses and what
to use instead: on a small sample whose target has a second mode or heavy
tails, choose `StudentTFiniteVar` (or a mixture head) before turning on
cross-fitting with `Normal`.
