#!/usr/bin/env python3
"""Tuner follow-up of Cycle 26 (O17, F2): `shipped` search against the
untuned default, Poisson head on football goals.

    python3 -I counts_hpo_summary.py reports/O17_hpo_poisson_*.csv

Stdlib only.
"""
import collections
import csv
import math
import sys


def mean(v):
    return sum(v) / len(v)


def se(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


pooled = []
print(f"{'dataset':<10} {'untuned NLL':>11} {'shipped-untuned':>16} {'se':>7} {'lower':>6} {'dRMSE%':>7} "
      f"{'search s':>8}  winner medians lr/depth/leaf/bins")
for f in sorted(sys.argv[1:]):
    t = collections.defaultdict(dict)
    with open(f) as fh:
        for r in csv.DictReader(fh):
            t[r["seed"]][r["arm"]] = r
    d = [float(a["shipped"]["test_nll"]) - float(a["untuned"]["test_nll"]) for a in t.values()]
    rr = [float(a["shipped"]["test_rmse"]) / float(a["untuned"]["test_rmse"]) - 1 for a in t.values()]

    def med(k):
        return sorted(float(a["shipped"][k]) for a in t.values())[len(t) // 2]

    ds = next(iter(t.values()))["untuned"]["dataset"]
    pooled.append(mean(d))
    print(f"{ds:<10} {mean([float(a['untuned']['test_nll']) for a in t.values()]):>11.4f} {mean(d):>+16.4f} "
          f"{se(d):>7.4f} {sum(x < 0 for x in d):>3}/{len(d)} {100 * mean(rr):>+7.2f} "
          f"{mean([float(a['shipped']['search_secs']) for a in t.values()]):>8.0f}  "
          f"{med('lr'):.3f}/{med('depth'):.0f}/{med('msl'):.0f}/{med('max_bins'):.0f}")
print(f"pooled over {len(pooled)} datasets: {mean(pooled):+.4f} (se {se(pooled):.4f})")
