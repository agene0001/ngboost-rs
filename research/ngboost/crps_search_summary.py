"""Cycle 13 / R50: LogScore Both-search vs CRPS-trained cold and Both search.
python3 -I crps_search_summary.py FILE.csv [...]; paired vs variant `both`."""
import csv, sys, math
from collections import defaultdict
rows = []
for f in sys.argv[1:]:
    rows += [r for r in csv.DictReader(l for l in open(f) if l.strip() and not l.startswith('#'))]
by = defaultdict(dict)
for r in rows:
    by[(r['dataset'], r['variant'])][r['seed']] = r
print(f"{'dataset':<10}{'variant':<11}{'n':>3}{'NLL':>9}{'CRPS':>9}{'RMSE':>8}{'cov90':>7}{'K':>5}{'warm':>5}{'rounds':>8}"
      f"{'ΔNLL (se) wins':>24}{'ΔCRPS (se) wins':>26}")
for ds in dict.fromkeys(r['dataset'] for r in rows):
    base = by[(ds, 'both')]
    for v in ['both', 'crps+cold', 'crps+both']:
        R = by.get((ds, v), {})
        if not R: continue
        n = len(R); f = lambda k: sum(float(r[k]) for r in R.values()) / n
        out = f"{ds:<10}{v:<11}{n:>3}{f('nll'):>9.4f}{f('crps'):>9.4f}{f('rmse'):>8.4f}{f('cov90'):>7.3f}{f('extra'):>5.1f}{f('mean_sigma'):>5.1f}{f('n_iter'):>8.0f}"
        if v != 'both':
            for k in ('nll', 'crps'):
                d = [float(R[s][k]) - float(base[s][k]) for s in R if s in base]
                m = sum(d) / len(d); se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
                out += f"{m:>+10.4f} ({se:.4f}) {sum(1 for x in d if x < 0)}/{len(d)}"
        print(out)
