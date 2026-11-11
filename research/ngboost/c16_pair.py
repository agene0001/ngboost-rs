"""Paired summary of one ts_bench / research_bench CSV against a reference variant.

    python3 -I c16_pair.py FILE.csv REF_VARIANT [VARIANT ...]

Per dataset and variant: mean NLL, mean ± se and median of Δ = variant − ref
(NLL, CRPS) over shared seeds, NLL wins (Δ < 0), mean cov90, calib, extra.
"""
import csv, math, sys
from collections import defaultdict

path, ref = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(l for l in open(path) if l.strip() and not l.startswith("#"))]
data = defaultdict(dict)  # (ds, var) -> seed -> row
order_ds, order_var = [], []
for r in rows:
    ds, v = r["dataset"], r["variant"]
    data[(ds, v)][r["seed"]] = r
    if ds not in order_ds:
        order_ds.append(ds)
    if v not in order_var:
        order_var.append(v)
variants = sys.argv[3:] or [v for v in order_var if v != ref]


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError):
        return float("nan")


def stats(d):
    d = [x for x in d if math.isfinite(x)]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    s = sorted(d)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return m, se, med, n


print(f"# {path} — reference: {ref}")
print(f"{'dataset':<11}{'variant':<22}{'n':>3}{'NLL':>9}{'ΔNLL':>9}{'se':>7}{'med':>8}{'wins':>7}"
      f"{'ΔCRPS':>9}{'se':>7}{'cov90':>7}{'calib':>7}{'extra':>8}{'iters':>6}")
for ds in order_ds:
    base = data.get((ds, ref), {})
    if not base:
        continue
    for v in [ref] + variants:
        cur = data.get((ds, v), {})
        seeds = sorted(set(base) & set(cur), key=int)
        if not seeds:
            continue
        nll = stats([f(cur[s], "nll") for s in seeds])
        dn = [f(cur[s], "nll") - f(base[s], "nll") for s in seeds]
        dc = [f(cur[s], "crps") - f(base[s], "crps") for s in seeds]
        mn, sn, md, n = stats(dn)
        mc, sc, _, _ = stats(dc)
        wins = sum(1 for x in dn if x < 0)
        cov = stats([f(cur[s], "cov90") for s in seeds])[0]
        cal = stats([f(cur[s], "calib") for s in seeds])[0]
        ex = stats([f(cur[s], "extra") for s in seeds])[0]
        it = stats([f(cur[s], "n_iter") for s in seeds])[0]
        tag = "" if v != ref else "  (ref)"
        print(f"{ds:<11}{v:<22}{len(seeds):>3}{nll[0]:>9.4f}{mn:>+9.4f}{sn:>7.4f}{md:>+8.4f}"
              f"{wins:>4}/{len(seeds):<2}{mc:>+9.4f}{sc:>7.4f}{cov:>7.3f}{cal:>7.3f}{ex:>8.3f}{it:>6.0f}{tag}")
