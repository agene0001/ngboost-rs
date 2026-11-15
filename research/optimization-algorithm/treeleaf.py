"""O7: leaf 20 against leaf 1 for the exact tree learners
(`reports/O7_treeleaf_<learner>_n<N>.csv`, `c18_bench --mode treeleaf`).

  python3 -I research/optimization-algorithm/treeleaf.py

Run from the repo root.
"""
import csv, glob, os

R = "research/optimization-algorithm/reports/"
DS = ["het_a", "het_b", "homo", "tnoise", "kin8nm", "protein"]


def stat(v):
    n = len(v)
    m = sum(v) / n
    return m, (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5


for f in sorted(glob.glob(R + "O7_treeleaf_*.csv")):
    d = {}
    for r in csv.DictReader(open(f)):
        d[(r["dataset"], int(r["seed"]), int(r["msl"]))] = (float(r["test_nll"]), float(r["test_rmse"]))
    print(f"\n{os.path.basename(f)}: leaf 20 − leaf 1")
    ms = []
    for ds in DS:
        seeds = [s for s in range(64) if (ds, s, 1) in d and (ds, s, 20) in d]
        dn = [d[(ds, s, 20)][0] - d[(ds, s, 1)][0] for s in seeds]
        dr = [d[(ds, s, 20)][1] - d[(ds, s, 1)][1] for s in seeds]
        m, se = stat(dn)
        mr, ser = stat(dr)
        ms.append(m)
        print(f"  {ds:8s} NLL {m:+.4f} ({se:.4f}) {sum(x < 0 for x in dn)}/{len(dn)} | RMSE {mr:+.4f} ({ser:.4f})")
    print(f"  mean NLL {sum(ms) / len(ms):+.4f}")
