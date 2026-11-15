"""O11: where does the cross-fitted staged fit lose on protein with 490
training rows? Per-test-row scores of the default and the cross-fitted fit
(`reports/O11_rows_protein_n700.csv`, `c18_bench --mode rows`).

  python3 -I research/optimization-algorithm/rows_protein.py

Run from the repo root.
"""
import csv, collections, math, statistics as st

rows = collections.defaultdict(dict)
for r in csv.DictReader(open("research/optimization-algorithm/reports/O11_rows_protein_n700.csv")):
    try:
        rows[(int(r["seed"]), int(r["row"]))][r["fit"]] = (
            float(r["y"]), float(r["mu"]), float(r["sigma"]), float(r["nll"]), int(r["n_iter"]))
    except ValueError:
        pass
pairs = {k: v for k, v in rows.items() if "default" in v and "xfit" in v}
seeds = sorted({k[0] for k in pairs})
print("per seed: ΔNLL (xfit − default), share of it carried by the 5 worst rows, rounds, median σ ratio")
tot = []
for s in seeds:
    ks = [k for k in pairs if k[0] == s]
    d = [pairs[k]["xfit"][3] - pairs[k]["default"][3] for k in ks]
    n = len(d)
    worst = sorted(d, reverse=True)[:5]
    ratio = st.median(pairs[k]["xfit"][2] / pairs[k]["default"][2] for k in ks)
    rmse = lambda f: math.sqrt(sum((pairs[k][f][0] - pairs[k][f][1]) ** 2 for k in ks) / n)
    print(f"  seed {s}: Δ {sum(d) / n:+.4f}  5 worst rows {sum(worst) / n:+.4f}  rest {(sum(d) - sum(worst)) / n:+.4f}"
          f"  rounds {pairs[ks[0]]['default'][4]}/{pairs[ks[0]]['xfit'][4]}  σ ratio {ratio:.2f}"
          f"  RMSE {rmse('default'):.3f}/{rmse('xfit'):.3f}")
    tot += [(pairs[k]["xfit"][3] - pairs[k]["default"][3], k) for k in ks]
n = len(tot)
allv = [x[0] for x in tot]
print(f"\nall {n} test rows: mean Δ {sum(allv) / n:+.4f}, median Δ {st.median(allv):+.4f}, rows better with xfit {sum(x < 0 for x in allv)}/{n}")
srt = sorted(allv, reverse=True)
for top in (5, 10, 20, 50):
    print(f"  the {top} worst rows carry {sum(srt[:top]) / n:+.4f}; without them the mean is {sum(srt[top:]) / (n - top):+.4f}")

# what do the worst rows look like?
print("\nthe 12 worst rows: seed, y, default (μ, σ, |z|), xfit (μ, σ, |z|), Δ")
for dlt, k in sorted(tot, reverse=True)[:12]:
    a, b = pairs[k]["default"], pairs[k]["xfit"]
    print(f"  {k[0]:2d}  y {a[0]:6.2f} | {a[1]:6.2f} {a[2]:5.2f} {abs(a[0] - a[1]) / a[2]:5.1f} | {b[1]:6.2f} {b[2]:5.2f} {abs(b[0] - b[1]) / b[2]:5.1f} | {dlt:+.2f}")

# binned by the default fit's |z|
print("\nΔ by the default fit's |z| on the row")
bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)]
for lo, hi in bins:
    sel = [d for d, k in tot if lo <= abs(pairs[k]["default"][0] - pairs[k]["default"][1]) / pairs[k]["default"][2] < hi]
    if sel:
        print(f"  |z| in [{lo}, {hi if hi < 1e9 else 'inf'}): {len(sel):4d} rows, mean Δ {sum(sel) / len(sel):+.4f}, contribution {sum(sel) / n:+.4f}")
# binned by the scale ratio
print("\nΔ by σ_xfit / σ_default on the row")
for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 1.25), (1.25, 2), (2, 1e9)]:
    sel = [d for d, k in tot if lo <= pairs[k]["xfit"][2] / pairs[k]["default"][2] < hi]
    if sel:
        print(f"  ratio in [{lo}, {hi if hi < 1e9 else 'inf'}): {len(sel):4d} rows, mean Δ {sum(sel) / len(sel):+.4f}, contribution {sum(sel) / n:+.4f}")
