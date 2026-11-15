"""Paired summary for Cycle 18 CSVs (c18_bench and research_bench output).

  python3 -I c18_pair.py FILE [FILE...] --by fit,msl --ref default/1 \
      [--metrics test_nll,test_rmse,cov90] [--where fit=default]

Rows are paired on (dataset, seed). `--by` names the column(s) that make up
the variant label (joined with '/'). For each dataset and each non-reference
variant: mean of the metric, mean difference to the reference ± standard
error, and the number of seeds on which the variant is lower. The first
metric gets the win count. Lines starting with '#' and research_bench's
trailing summary block are skipped.
"""
import csv, math, sys
from collections import defaultdict

args = sys.argv[1:]
files, opt = [], {}
i = 0
while i < len(args):
    if args[i].startswith("--"):
        opt[args[i][2:]] = args[i + 1]
        i += 2
    else:
        files.append(args[i])
        i += 1
by = opt.get("by", "variant").split(",")
metrics = opt.get("metrics", "nll,rmse,cov90").split(",")
where = dict(kv.split("=") for kv in opt["where"].split(",")) if "where" in opt else {}

data = defaultdict(dict)  # (dataset, variant) -> seed -> row
order = []
for f in files:
    header = None
    for line in open(f):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = line.split(",")
        if cells[0] == "dataset":
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        r = dict(zip(header, cells))
        try:
            int(r["seed"])
        except (KeyError, ValueError):
            continue
        if any(r.get(k) != v for k, v in where.items()):
            continue
        v = "/".join(r[c] for c in by)
        if v not in order:
            order.append(v)
        data[(r["dataset"], v)][r["seed"]] = r
ref = opt.get("ref", order[0])
datasets = []
for d, _ in data:
    if d not in datasets:
        datasets.append(d)


def num(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def stats(v):
    v = [x for x in v if math.isfinite(x)]
    if not v:
        return float("nan"), float("nan"), 0
    m = sum(v) / len(v)
    se = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else float("nan")
    return m, se, len(v)


print(f"reference: {ref}   (Δ = variant − reference; wins = seeds with lower {metrics[0]})")
head = f"{'dataset':<16}{'variant':<22}{'n':>3}"
for m in metrics:
    head += f"{m:>11}{'Δ (se)':>19}"
print(head + f"{'wins':>7}{'nonfinite':>10}")
tot = defaultdict(lambda: defaultdict(list))
for d in datasets:
    base = data.get((d, ref), {})
    for v in order:
        rows = data.get((d, v), {})
        seeds = sorted(set(rows) & set(base), key=int) if v != ref else sorted(rows, key=int)
        if not seeds:
            continue
        line = f"{d:<16}{v:<22}{len(seeds):>3}"
        wins = bad = 0
        for k, m in enumerate(metrics):
            a = [num(rows[s][m]) for s in seeds]
            mean_a, _, _ = stats(a)
            if v == ref:
                line += f"{mean_a:>11.4f}{'':>19}"
                if k == 0:
                    bad = sum(1 for x in a if not math.isfinite(x))
                continue
            diff = [num(rows[s][m]) - num(base[s][m]) for s in seeds]
            md, sd, _ = stats(diff)
            line += f"{mean_a:>11.4f}{md:>+10.4f} ({sd:.4f})"
            tot[v][m].append(md)
            if k == 0:
                wins = sum(1 for x in diff if x < 0)
                bad = sum(1 for x in a if not math.isfinite(x))
        line += f"{'' if v == ref else str(wins) + '/' + str(len(seeds)):>7}{bad:>10}"
        print(line)
    print()
print("mean Δ over datasets:")
for v in order:
    if v == ref or v not in tot:
        continue
    print(f"  {v:<22}" + "".join(f"{m}: {sum(x for x in tot[v][m] if math.isfinite(x)) / max(1, sum(1 for x in tot[v][m] if math.isfinite(x))):+.4f}   " for m in metrics))
