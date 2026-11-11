"""Choose the number of mixture components by validation NLL (research/ngboost Cycle 11).

usage: python3 -I select_k.py normal.csv mix2.csv mix3.csv mix4.csv
Each CSV is a research_bench report on identical splits (same seeds); column
`extra` holds the validation NLL of the final model. For every
(dataset, seed) the rule picks the head with the lowest validation NLL and is
scored on test NLL; compared with each fixed head and with a test-set oracle
(reference only — it peeks at the test set).
"""
import csv, math, sys
from collections import defaultdict

names = ["K=1", "K=2", "K=3", "K=4"]
runs = []
for path in sys.argv[1:5]:
    d = {}
    for r in csv.DictReader(l for l in open(path) if l.strip() and not l.startswith("#")):
        d[(r["dataset"], r["seed"])] = (float(r["extra"]), float(r["nll"]))
    runs.append(d)
keys = sorted(set.intersection(*[set(d) for d in runs]))
by = defaultdict(list)
for k in keys:
    by[k[0]].append([runs[h][k] for h in range(4)])

def se(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v)) if len(v) > 1 else 0.0

print(f"{'dataset':<9} " + " ".join(f"{n:>8}" for n in names) + f" {'val-select':>11} {'oracle':>8}  {'picks (K=1..4)':<16} {'Δ select − best fixed':>22}")
for ds, rows in by.items():
    n = len(rows)
    fixed = [sum(r[h][1] for r in rows) / n for h in range(4)]
    pick = [min(range(4), key=lambda h: r[h][0]) for r in rows]
    sel = [r[p][1] for r, p in zip(rows, pick)]
    orc = [min(r[h][1] for h in range(4)) for r in rows]
    best = min(range(4), key=lambda h: fixed[h])
    diff = [s - r[best][1] for s, r in zip(sel, rows)]
    counts = [pick.count(h) for h in range(4)]
    print(f"{ds:<9} " + " ".join(f"{f:8.4f}" for f in fixed)
          + f" {sum(sel)/n:11.4f} {sum(orc)/n:8.4f}  {str(counts):<16} {sum(diff)/n:+9.4f} (se {se(diff):.4f}) vs {names[best]}")
