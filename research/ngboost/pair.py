"""Pair two research_bench CSVs on (dataset, seed): python3 -I pair.py A.csv varA B.csv varB
Prints per-dataset mean ± se of B − A for nll, rmse, cov90, plus wins (B better on NLL)."""
import csv, sys, math
from collections import defaultdict
def load(path, var):
    out = {}
    for r in csv.DictReader(l for l in open(path) if l.strip() and not l.startswith("#")):
        if r["variant"] == var:
            out[(r["dataset"], r["seed"])] = r
    return out
a, b = load(sys.argv[1], sys.argv[2]), load(sys.argv[3], sys.argv[4])
by = defaultdict(list)
for k in sorted(set(a) & set(b)):
    by[k[0]].append((a[k], b[k]))
print(f"{'dataset':<10} {'n':>2} {'NLL A':>8} {'NLL B':>8} {'ΔNLL (B−A)':>18} {'wins':>6} {'ΔCRPS (B−A)':>19} {'ΔRMSE':>8} {'cov90 A→B':>12}")
for ds, rows in by.items():
    d = [float(y["nll"]) - float(x["nll"]) for x, y in rows]
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) if n > 1 else 0.0
    wins = sum(1 for v in d if v < 0)
    na = sum(float(x["nll"]) for x, _ in rows) / n; nb = sum(float(y["nll"]) for _, y in rows) / n
    dr = sum(float(y["rmse"]) - float(x["rmse"]) for x, y in rows) / n
    ca = sum(float(x["cov90"]) for x, _ in rows) / n; cb = sum(float(y["cov90"]) for _, y in rows) / n
    dc = [float(y["crps"]) - float(x["crps"]) for x, y in rows]
    mc = sum(dc) / n
    sc = math.sqrt(sum((v - mc) ** 2 for v in dc) / (n - 1) / n) if n > 1 else 0.0
    print(f"{ds:<10} {n:>2} {na:>8.4f} {nb:>8.4f} {m:>+9.4f} (se {se:.4f}) {wins:>2}/{n:<3} {mc:>+9.4f} (se {sc:.4f}) {dr:>+8.3f} {ca:>5.3f}→{cb:.3f}")
