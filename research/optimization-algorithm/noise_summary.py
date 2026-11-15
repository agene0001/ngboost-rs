"""O12: how much of a small NLL difference is last-bit noise?
(`reports/O12_noise_normal.csv`, `c18_bench --mode noise`.)

Replicate 0 is the plain fit. Replicates 1…R−1 multiply every training
target by 1 ± 1e-12 (a seeded random sign per row); validation and test
targets are untouched.

  python3 -I research/optimization-algorithm/noise_summary.py

Run from the repo root.
"""
import csv, collections, math, statistics as st

d = {}
for r in csv.DictReader(open("research/optimization-algorithm/reports/O12_noise_normal.csv")):
    try:
        d[(r["dataset"], int(r["seed"]), int(r["rep"]), r["fit"])] = (float(r["test_nll"]), int(r["n_iter"]))
    except ValueError:
        pass
dss = []
for k in d:
    if k[0] not in dss:
        dss.append(k[0])
fits = sorted({k[3] for k in d})
reps = sorted({k[2] for k in d})
seeds = sorted({k[1] for k in d})


def pooled_sd(groups):
    """Pooled within-group standard deviation."""
    ss = sum(sum((x - st.mean(g)) ** 2 for x in g) for g in groups)
    df = sum(len(g) - 1 for g in groups)
    return math.sqrt(ss / df) if df else float("nan")


print(f"{len(seeds)} seeds, replicates {reps}; NLL on the untouched test block\n")
print("per fit: sd of one fit's NLL across replicates (pooled over seeds), largest |change| from replicate 0,")
print("         fits whose round count changed, and the sd of the 10-seed mean across replicates")
allsd = collections.defaultdict(list)
for ds in dss:
    for f in fits:
        groups = [[d[(ds, s, r, f)][0] for r in reps if (ds, s, r, f) in d] for s in seeds]
        groups = [g for g in groups if len(g) > 1]
        big = max(abs(d[(ds, s, r, f)][0] - d[(ds, s, 0, f)][0]) for s in seeds for r in reps[1:] if (ds, s, r, f) in d)
        moved = sum(d[(ds, s, r, f)][1] != d[(ds, s, 0, f)][1] for s in seeds for r in reps[1:] if (ds, s, r, f) in d)
        n_alt = sum(1 for s in seeds for r in reps[1:] if (ds, s, r, f) in d)
        means = [st.mean(d[(ds, s, r, f)][0] for s in seeds) for r in reps]
        sd = pooled_sd(groups)
        allsd[f].append((sd, st.stdev(means)))
        print(f"  {ds:8s} {f:8s} per-fit sd {sd:.4f}  max |Δ| {big:.4f}  rounds changed {moved}/{n_alt}  sd of the mean {st.stdev(means):.4f}")
print()
for f in fits:
    print(f"  {f:8s} over datasets: per-fit sd {st.mean(x[0] for x in allsd[f]):.4f} (max {max(x[0] for x in allsd[f]):.4f}),"
          f" sd of a 10-seed mean {st.mean(x[1] for x in allsd[f]):.4f} (max {max(x[1] for x in allsd[f]):.4f})")

if "default" in fits and "xfit" in fits:
    print("\npaired effect xfit − default (mean over seeds) in each replicate, and its sd across replicates")
    sds = []
    for ds in dss:
        eff = [st.mean(d[(ds, s, r, "xfit")][0] - d[(ds, s, r, "default")][0] for s in seeds) for r in reps]
        sds.append(st.stdev(eff))
        print(f"  {ds:8s} " + " ".join(f"{e:+.4f}" for e in eff) + f"   sd {st.stdev(eff):.4f}  range {max(eff) - min(eff):.4f}")
    print(f"  mean sd of a 10-seed paired effect: {st.mean(sds):.4f} (max {max(sds):.4f})")
    eff_all = [st.mean(st.mean(d[(ds, s, r, 'xfit')][0] - d[(ds, s, r, 'default')][0] for s in seeds) for ds in dss) for r in reps]
    print("  mean over the six datasets: " + " ".join(f"{e:+.4f}" for e in eff_all) + f"   sd {st.stdev(eff_all):.4f}")
