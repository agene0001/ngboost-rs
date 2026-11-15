"""O14: quantile bin edges against 1-D k-means bin edges for the histogram
trees (`reports/O14_binning.csv`, `reports/O14_binning_exp.csv`;
`c18_bench --mode binning`).

  python3 -I research/optimization-algorithm/binning_summary.py

`lib` = the library's binning at `bins` bins; `q` = the same rule applied
by pre-coding the features (a control: it should equal `lib`); `km` =
1-D k-means edges applied the same way. Run from the repo root.
"""
import csv, collections, os

R = "research/optimization-algorithm/reports/"
d = {}
for f in ("O14_binning.csv", "O14_binning_exp.csv"):
    if not os.path.exists(R + f):
        continue
    for r in csv.DictReader(open(R + f)):
        try:
            d[(r["dataset"], int(r["bins"]), r["variant"], int(r["seed"]))] = (
                float(r["test_nll"]), float(r["test_rmse"]), int(r["n_iter"]))
        except ValueError:
            pass


def stat(v):
    n = len(v)
    m = sum(v) / n
    return m, (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5


dss = []
for k in d:
    if k[0] not in dss:
        dss.append(k[0])
for ds in dss:
    for bins in sorted({k[1] for k in d if k[0] == ds}, reverse=True):
        seeds = sorted(k[3] for k in d if k[:3] == (ds, bins, "lib"))
        base = sum(d[(ds, bins, "lib", s)][0] for s in seeds) / len(seeds)
        line = f"{ds:12s} bins {bins:3d}  lib NLL {base:8.4f}"
        for v in ("q", "km"):
            ss = [s for s in seeds if (ds, bins, v, s) in d]
            if len(ss) < 2:
                continue
            dn = [d[(ds, bins, v, s)][0] - d[(ds, bins, "lib", s)][0] for s in ss]
            dr = [d[(ds, bins, v, s)][1] / d[(ds, bins, "lib", s)][1] - 1 for s in ss]
            m, se = stat(dn)
            mr, _ = stat(dr)
            same = sum(d[(ds, bins, v, s)][::2] == d[(ds, bins, "lib", s)][::2] for s in ss)
            line += f" | {v} {m:+.4f} ({se:.4f}) {sum(x < 0 for x in dn)}/{len(dn)} RMSE {100 * mr:+.1f} % identical {same}"
        print(line)
# invariance: the same data after a strictly increasing map of every feature
print("\nmean NLL before / after x -> exp(1.5 x), and fits that are identical (score and rounds)")
for ds in [x for x in dss if not x.endswith("_exp") and x + "_exp" in dss]:
    for bins in sorted({k[1] for k in d if k[0] == ds}, reverse=True):
        for v in ("lib", "km"):
            ss = sorted(k[3] for k in d if k[:3] == (ds, bins, v) and (ds + "_exp", bins, v, k[3]) in d)
            if not ss:
                continue
            a = sum(d[(ds, bins, v, s)][0] for s in ss) / len(ss)
            b = sum(d[(ds + "_exp", bins, v, s)][0] for s in ss) / len(ss)
            same = sum(d[(ds, bins, v, s)][::2] == d[(ds + "_exp", bins, v, s)][::2] for s in ss)
            print(f"  {ds:8s} bins {bins:3d} {v:3s} {a:8.4f} -> {b:8.4f} ({b - a:+.4f})  identical {same}/{len(ss)}")
