"""O10: the adaptive leaf rule against a fixed 20 on small data, count and
classification heads (`reports/O10_small_<head>_n<N>_ml<leaf>.csv`,
`research_bench`).

  python3 -I research/optimization-algorithm/leaf_small_heads.py

Pairs the rule's leaf (7, 10, 14 at 100, 150, 200 total rows) with leaf 20
on (dataset, seed). Run from the repo root.
"""
import csv, math, os, statistics as st

R = "research/optimization-algorithm/reports/"
RULE = {100: 7, 150: 10, 200: 14}


def load(f):
    d = {}
    if not os.path.exists(f):
        return d
    for r in csv.DictReader(open(f)):
        s = r.get("seed")
        if not (s and s.isdigit()):
            continue
        try:
            d[(r["dataset"], int(s))] = (float(r["nll"]), float(r["rmse"]))
        except (ValueError, TypeError):
            pass
    return d


def stat(v):
    n = len(v)
    m = sum(v) / n
    return m, (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5


tot_w = tot_n = 0
for head in ("poisson", "negbin", "bernoulli", "cat3"):
    for n, rule in RULE.items():
        a = load(R + f"O10_small_{head}_n{n}_ml{rule}.csv")
        b = load(R + f"O10_small_{head}_n{n}_ml20.csv")
        for ds in sorted({k[0] for k in a}):
            seeds = [s for (d, s) in a if d == ds and (ds, s) in b]
            dn = [a[(ds, s)][0] - b[(ds, s)][0] for s in seeds]
            dn = [x for x in dn if math.isfinite(x)]
            dr = [a[(ds, s)][1] / b[(ds, s)][1] - 1 for s in seeds if b[(ds, s)][1] > 0]
            dr = [x for x in dr if math.isfinite(x)]
            if len(dn) < 2:
                continue
            m, se = stat(dn)
            tot_w += sum(x < 0 for x in dn)
            tot_n += len(dn)
            rm = "n/a"
            if len(dr) >= 2:
                mr, ser = stat(dr)
                rm = f"{100 * mr:+.1f} % ({100 * ser:.1f})"
            print(f"{head:9s} n={n} leaf {rule:2d} vs 20  {ds:9s} NLL {m:+.4f} ({se:.4f}) median {st.median(dn):+.4f}"
                  f" wins {sum(x < 0 for x in dn):2d}/{len(dn)} | RMSE {rm}")
print(f"\nNLL lower with the rule on {tot_w} of {tot_n} fits")
