"""Summarise an automix report: paired ΔNLL vs the first variant, search cost
(boosting rounds) and chosen-K distribution.  usage: python3 -I automix_summary.py R.csv"""
import csv, math, sys
from collections import defaultdict
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if l.strip() and not l.startswith("#"))]
variants = list(dict.fromkeys(r["variant"] for r in rows))
by = defaultdict(dict)
for r in rows:
    by[(r["dataset"], r["seed"])][r["variant"]] = r
dsets = list(dict.fromkeys(r["dataset"] for r in rows))
base = variants[0]
def se(v):
    m = sum(v) / len(v); return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else 0.0
print(f"{'dataset':<17} {base + ' NLL':>10}" + "".join(f"{'Δ ' + v + ' (se) wins cost':>34}" for v in variants[1:]))
tot = {v: [0.0, 0.0] for v in variants}
for d in dsets:
    ks = [k for k in by if k[0] == d]
    b = sum(float(by[k][base]["nll"]) for k in ks) / len(ks)
    line = f"{d:<17} {b:10.4f}"
    for v in variants[1:]:
        dn = [float(by[k][v]["nll"]) - float(by[k][base]["nll"]) for k in ks]
        wins = sum(1 for x in dn if x < 0)
        cost = sum(float(by[k][v]["extra2"]) for k in ks) / sum(float(by[k][base]["extra2"]) for k in ks)
        line += f"   {sum(dn)/len(dn):+.4f} ({se(dn):.4f}) {wins:>2}/{len(ks)} {cost:5.0%}"
    print(line)
    for v in variants:
        tot[v][0] += sum(float(by[k][v]["extra2"]) for k in ks)
        tot[v][1] += sum(float(by[k][v]["secs"]) for k in ks)
print("\ncost relative to " + base + ": " + ", ".join(f"{v} rounds {tot[v][0]/tot[base][0]:.0%} time {tot[v][1]/tot[base][1]:.0%}" for v in variants))
print("\nchosen K (counts K=1..4) [warm picks]:")
for d in dsets:
    ks = [k for k in by if k[0] == d]
    parts = []
    for v in variants:
        c = [sum(1 for k in ks if int(float(by[k][v]["extra"])) == K) for K in (1, 2, 3, 4)]
        w = sum(1 for k in ks if float(by[k][v]["mean_sigma"] or 0) == 1.0) if by[ks[0]][v]["mean_sigma"] not in ("NaN", "") else 0
        parts.append(f"{v} {c} [{w}]")
    print(f"  {d:<17} " + "  ".join(parts))
