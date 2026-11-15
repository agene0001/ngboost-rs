"""O5: what a ceiling on the LogNormal log-scale would do, emulated on
fitted models (`research_bench` with NGB_PARAM_DIAG=1 writes one CAPDIAG
line per fit and k: the test NLL / RMSE of the mean with the scale column
clamped at reference + ln k).

  python3 -I research/optimization-algorithm/cap_summary.py \
      research/optimization-algorithm/reports/O5_capdiag_init.csv

Δ NLL is against the default staged fit (`auto`) without a ceiling.
"""
import csv, collections, math, sys

rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) == 9 and r[0] != "dataset"]
d = collections.defaultdict(dict)
for ds, seed, var, k, nll, rmse, hit, hit_tr, s0 in rows:
    d[(ds, var, k)][seed] = (float(nll), float(rmse), int(hit), int(hit_tr), float(s0))
ks = ["1.5", "2", "3", "4", "6", "8", "12", "inf"]
dss = []
for r in rows:
    if r[0] not in dss:
        dss.append(r[0])
for ds in dss:
    for var in ("joint+auto", "auto", "auto+xfit"):
        if (ds, var, "inf") not in d:
            continue
        base = d[(ds, "auto", "inf")]
        s0 = [v[4] for v in d[(ds, var, "inf")].values()]
        print(f"{ds} {var}  n={len(s0)}  reference scale {math.exp(sum(s0) / len(s0)):.3f}")
        for k in ks:
            v = d.get((ds, var, k), {})
            seeds = sorted(s for s in v if s in base)
            if len(seeds) < 2:
                continue
            dn = [v[s][0] - base[s][0] for s in seeds]
            m = sum(dn) / len(dn)
            se = (sum((x - m) ** 2 for x in dn) / (len(dn) - 1) / len(dn)) ** 0.5
            rm = sorted(v[s][1] for s in seeds)
            print(
                f"   k={k:<4s} ΔNLL {m:+.4f} ({se:.4f})  RMSE median {rm[len(rm) // 2]:.4g} max {rm[-1]:.4g}"
                f"  rows at the ceiling: test {sum(v[s][2] for s in seeds) / len(seeds):.1f}"
                f" train {sum(v[s][3] for s in seeds) / len(seeds):.1f}"
            )
