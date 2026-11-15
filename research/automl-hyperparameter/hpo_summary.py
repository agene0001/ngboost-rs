"""Summary of the end-to-end tuner comparison (`c18_bench --mode hpo`,
`reports/A6_hpo_<dataset>.csv`, or `reports/A7_hpo_<head>_<dataset>.csv`
with `--files`): held-out test scores of the refitted
winner per arm, paired by (dataset, seed) against a reference arm.

  python3 -I research/automl-hyperparameter/hpo_summary.py [--ref old] [--files 'A7_hpo_poisson_*.csv']

`--files` takes one glob or several separated by commas; files whose names
agree after their two-part prefix are merged (A8's `newbins` rows are the
baseline of A9's `xfit` rows).

Run from the repo root.
"""
import csv, glob, math, sys, collections

ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "old"
pattern = sys.argv[sys.argv.index("--files") + 1] if "--files" in sys.argv else "A6_hpo_*.csv"
R = "research/automl-hyperparameter/reports/"
rows = []
for f in sorted(x for pat in pattern.split(",") for x in glob.glob(R + pat)):
    # label = the file name after its two-part prefix: "het_a" for
    # A6_hpo_het_a.csv, "lognormal_kin8nm" for A7_hpo_lognormal_kin8nm.csv,
    # so the same dataset under two heads stays apart
    label = f.rsplit("/", 1)[-1][:-4].split("_", 2)[2]
    for r in csv.DictReader(open(f)):
        if (r.get("seed") or "").isdigit():
            r["dataset"] = label
            rows.append(r)
d = {(r["dataset"], int(r["seed"]), r["arm"]): r for r in rows}
datasets = sorted({k[0] for k in d})
arms = [a for a in ["untuned", "old", "oldspace", "new", "newbins", "shipped", "xfit"] if any(k[2] == a for k in d)]


def stat(v):
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(v) / n
    return m, (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5


print(f"Δ = arm − {ref} on the test block (negative = better); n = seeds present in both")
for met in ("test_nll", "test_rmse"):
    print(f"\n{met}")
    pooled = collections.defaultdict(list)
    for ds in datasets:
        line = f"  {ds:8s}"
        for arm in arms:
            seeds = [s for s in range(64) if (ds, s, arm) in d and (ds, s, ref) in d]
            if not seeds:
                line += f" | {arm}: n/a"
                continue
            base = sum(float(d[(ds, s, arm)][met]) for s in seeds) / len(seeds)
            if arm == ref:
                line += f" | {arm} {base:.4f} (n={len(seeds)})"
                continue
            dif = [float(d[(ds, s, arm)][met]) - float(d[(ds, s, ref)][met]) for s in seeds]
            dif = [x for x in dif if math.isfinite(x)]
            m, se = stat(dif)
            pooled[arm] += dif
            line += f" | {arm} {m:+.4f} ({se:.4f}) {sum(x < 0 for x in dif)}/{len(dif)}"
        print(line)
    print("  pooled  " + "".join(
        f" | {a} {stat(pooled[a])[0]:+.4f} ({stat(pooled[a])[1]:.4f}) {sum(x < 0 for x in pooled[a])}/{len(pooled[a])}"
        for a in arms if a != ref and pooled[a]))

print("\nsearch seconds (mean), refit rounds (mean), winning parameters (median)")
for ds in datasets:
    for arm in arms:
        rs = [r for r in rows if r["dataset"] == ds and r["arm"] == arm]
        if not rs:
            continue
        med = lambda k: sorted(float(r[k]) for r in rs)[len(rs) // 2]
        mean = lambda k: sum(float(r[k]) for r in rs) / len(rs)
        bins = f" bins {med('max_bins'):.0f}" if rs[0].get("max_bins") not in (None, "", "NaN") else ""
        print(f"  {ds:8s} {arm:9s} secs {mean('search_secs'):7.1f}  rounds {mean('n_iter'):7.0f}  cap {med('n_est'):6.0f}"
              f"  lr {med('lr'):.3f} depth {med('depth'):.0f} leaf {med('msl'):.0f} col {med('cs'):.2f}{bins}")
