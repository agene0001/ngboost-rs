"""O6: the adaptive leaf rule against a fixed leaf of 20 on small data
(`reports/O6_leaf_small_n<N>.csv`, `c18_bench --mode leaf`).

  python3 -I research/optimization-algorithm/leaf_small.py

For each total size N (70 % of it is fitted on) the rule min(20, n/10) maps
to one of the swept leaf sizes; the script pairs that leaf with leaf 20 on
(dataset, seed). Run from the repo root.
"""
import csv, math, statistics as st

R = "research/optimization-algorithm/reports/"
DS = ["het_a", "het_b", "homo", "tnoise", "kin8nm", "protein"]
GRID = [1, 3, 5, 7, 10, 14, 20]


def load(n):
    d = {}
    for r in csv.DictReader(open(R + f"O6_leaf_small_n{n}.csv")):
        try:
            d[(r["dataset"], int(r["seed"]), int(r["msl"]))] = (
                float(r["test_nll"]), float(r["test_rmse"]), float(r["cov90"]))
        except ValueError:
            pass
    return d


def stat(v):
    n = len(v)
    m = sum(v) / n
    return m, (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5


for n in (100, 150, 200, 300):
    d = load(n)
    n_fit = int(0.7 * n)
    rule = min(20, max(1, n_fit // 10))
    print(f"\ntotal {n} rows, {n_fit} fitted on: rule leaf {rule}; every swept leaf against 20 "
          f"(mean ΔNLL | mean ΔRMSE over the six datasets, 20 seeds)")
    for g in GRID[:-1]:
        mn, mr = [], []
        for ds in DS:
            mn.append(st.mean(d[(ds, s, g)][0] - d[(ds, s, 20)][0] for s in range(20)))
            mr.append(st.mean(d[(ds, s, g)][1] - d[(ds, s, 20)][1] for s in range(20)))
        print(f"   leaf {g:2d}: {st.mean(mn):+.3f} | {st.mean(mr):+.3f}")
    if rule == 20:
        continue
    print(f"  rule (leaf {rule}) against 20, per dataset:")
    rel, alln = [], []
    for ds in DS:
        dn = [d[(ds, s, rule)][0] - d[(ds, s, 20)][0] for s in range(20)]
        dr = [d[(ds, s, rule)][1] / d[(ds, s, 20)][1] - 1 for s in range(20)]
        m, se = stat(dn)
        mr, ser = stat(dr)
        rel += dr
        alln += dn
        print(f"   {ds:8s} NLL {m:+.3f} ({se:.3f}) median {st.median(dn):+.3f} wins {sum(x < 0 for x in dn):2d}/20"
              f" | RMSE {100 * mr:+.1f} % ({100 * ser:.1f}) wins {sum(x < 0 for x in dr):2d}/20")
    m, se = stat(rel)
    print(f"   pooled   RMSE {100 * m:+.1f} % ({100 * se:.1f}), lower on {sum(x < 0 for x in rel)}/120;"
          f" NLL median {st.median(alln):+.4f}, lower on {sum(x < 0 for x in alln)}/120")
    for lo, hi in ((0, 10), (10, 20)):
        ms = [st.mean(d[(ds, s, rule)][0] - d[(ds, s, 20)][0] for s in range(lo, hi)) for ds in DS]
        print(f"   seeds {lo}-{hi - 1}: mean ΔNLL over datasets {st.mean(ms):+.4f}")
