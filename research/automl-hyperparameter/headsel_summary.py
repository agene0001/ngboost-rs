"""A10: choosing the head by the tuner's CV score
(`reports/A10_headsel_<group>_<dataset>.csv`, `c18_bench --mode hposel`).

  python3 -I research/automl-hyperparameter/headsel_summary.py

Per (dataset, seed) there is one row per head and budget (15, 30 trials).
`single` = the first head at 30 trials; `select 2x15` / `select 2x30` =
the head with the lower CV score at that budget; `oracle` = the head with
the lower *test* NLL at 30 trials (a bound, not a procedure). Run from
the repo root.
"""
import csv, glob, collections

R = "research/automl-hyperparameter/reports/"
d = {}
order = {}
for f in sorted(glob.glob(R + "A10_headsel_*.csv")):
    group = f.rsplit("/", 1)[-1].split("_")[2]
    for r in csv.DictReader(open(f)):
        if not (r.get("seed") or "").isdigit():
            continue
        ds = f"{group}:{r['dataset']}"
        order.setdefault(ds, [])
        if r["head"] not in order[ds]:
            order[ds].append(r["head"])
        d[(ds, int(r["seed"]), r["head"], int(r["trials"]))] = (float(r["cv_score"]), float(r["test_nll"]), float(r["test_rmse"]))


def stat(v):
    n = len(v)
    m = sum(v) / n
    se = (sum((x - m) ** 2 for x in v) / (n - 1) / n) ** 0.5 if n > 1 else float("nan")
    return m, se


pool = collections.defaultdict(list)
picks = collections.Counter()
print("held-out NLL; Δ against `single` (the first head, 30 trials); negative = better\n")
for ds, heads in order.items():
    h0, h1 = heads[0], heads[1]
    seeds = sorted(s for (dd, s, h, t) in d if dd == ds and h == h0 and t == 30
                   and all((ds, s, h, b) in d for h in heads for b in (15, 30)))
    if len(seeds) < 2:
        print(f"{ds}: incomplete ({len(seeds)} seeds)")
        continue
    single = [d[(ds, s, h0, 30)][1] for s in seeds]
    other = [d[(ds, s, h1, 30)][1] - d[(ds, s, h0, 30)][1] for s in seeds]
    line = f"{ds:22s} n={len(seeds)} {h0} {stat(single)[0]:8.4f} | {h1} alone {stat(other)[0]:+.4f} ({stat(other)[1]:.4f})"
    for b in (15, 30):
        sel = []
        n1 = 0
        for s in seeds:
            pick = min(heads, key=lambda h: d[(ds, s, h, b)][0])
            n1 += pick == h1
            sel.append(d[(ds, s, pick, b)][1] - d[(ds, s, h0, 30)][1])
        m, se = stat(sel)
        pool[b] += sel
        picks[(ds, b)] = n1
        line += f" | select 2x{b} {m:+.4f} ({se:.4f}) lower {sum(x < 0 for x in sel)} equal {sum(x == 0 for x in sel)}, picked {h1} {n1}/{len(seeds)}"
    orc = [min(d[(ds, s, h, 30)][1] for h in heads) - d[(ds, s, h0, 30)][1] for s in seeds]
    pool["oracle"] += orc
    pool["other"] += other
    line += f" | oracle {stat(orc)[0]:+.4f}"
    print(line)
print()
for k, name in ((15, "select 2x15"), (30, "select 2x30"), ("other", "second head alone"), ("oracle", "oracle")):
    v = pool[k]
    if v:
        m, se = stat(v)
        print(f"pooled {name:18s} {m:+.4f} ({se:.4f}); lower on {sum(x < 0 for x in v)}, equal on {sum(x == 0 for x in v)} of {len(v)}")
