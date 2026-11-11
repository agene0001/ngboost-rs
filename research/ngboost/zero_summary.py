"""Summarise research_bench `--dist zero` CSVs (Cycle 13 hurdle heads).
python3 -I zero_summary.py FILE.csv [FILE2.csv ...]
Per dataset × head: mean ± sd of CRPS (comparable across heads), NLL on the
non-zero test rows (comparable across heads), mixed-measure NLL (hurdle
heads only), Brier of P(Y=0), 90 % coverage; paired ΔCRPS vs `normal`
with se and seed wins. Failed fits (NaN) are counted, not averaged."""
import csv, sys, math
from collections import defaultdict
rows = []
for f in sys.argv[1:]:
    rows += [r for r in csv.DictReader(l for l in open(f) if l.strip() and not l.startswith('#'))]
by = defaultdict(dict)
for r in rows:
    by[(r['dataset'], r['variant'])][r['seed']] = r
def ms(v):
    v = [x for x in v if math.isfinite(x)]
    if not v: return float('nan'), float('nan'), 0
    m = sum(v) / len(v)
    try:
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    except OverflowError:
        sd = float('inf')
    return m, sd, len(v)
def med(v):
    v = sorted(x for x in v if math.isfinite(x))
    return v[len(v) // 2] if v else float('nan')
dsets = sorted({d for d, _ in by}, key=lambda d: [r['dataset'] for r in rows].index(d))
for ds in dsets:
    heads = [v for (d, v) in by if d == ds]
    zero = ms([float(r['extra2']) for r in by[(ds, heads[0])].values()])[0]
    print(f"\n{ds}  (test share of exact zeros {zero:.3f})")
    print(f"  {'head':<11}{'CRPS mean':>12}{'median':>9}{'NLL y≠0':>17}{'NLL mixed':>17}{'Brier P0':>10}{'cov90':>7}  ΔCRPS vs normal (se) wins")
    base = by.get((ds, 'normal'), {})
    for h in heads:
        R = by[(ds, h)]
        c = ms([float(r['crps']) for r in R.values()]); nz = ms([float(r['extra']) for r in R.values()])
        nl = ms([float(r['nll']) for r in R.values()]); br = ms([float(r['calib']) for r in R.values()])
        cv = ms([float(r['cov90']) for r in R.values()])
        fails = sum(1 for r in R.values() if not math.isfinite(float(r['crps'])))
        d = [float(R[s]['crps']) - float(base[s]['crps']) for s in R if s in base
             and math.isfinite(float(R[s]['crps'])) and math.isfinite(float(base[s]['crps']))]
        dm = ms(d); se = dm[1] / math.sqrt(dm[2]) if dm[2] > 1 else float('nan')
        w = sum(1 for x in d if x < 0)
        mixed = f"{nl[0]:>8.4f}±{nl[1]:<7.4f}" if h.startswith('h') else f"{'—':>16}"
        cm = med([float(r['crps']) for r in R.values()])
        print(f"  {h:<11}{c[0]:>12.4g}{cm:>9.4f}{nz[0]:>8.4f}±{nz[1]:<7.4f}{mixed:>17}{br[0]:>10.4f}{cv[0]:>7.3f}"
              + ("" if h == 'normal' else f"  {dm[0]:+.4g} ({se:.3g}) {w}/{len(d)}") + (f"  [{fails} failed]" if fails else ""))
