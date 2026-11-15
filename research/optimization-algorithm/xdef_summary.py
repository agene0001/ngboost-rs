#!/usr/bin/env python3
"""Summary of `c18_bench --mode xdef` (Cycle 25, O16): cross-fitting on
against off for the shipped default model.

    python3 -I xdef_summary.py [--arm k5] [--rule U|S|G] [--min-size N]
                               [--cells 1] FILE...

Rules (see proposals/O16_crossfit_default.md):
  U  every cell
  S  only cells where the `off` arm staged
  G  only cells with size >= --min-size
Paired differences are arm - off on the same head, dataset, size and seed.
Stdlib only.
"""
import csv
import math
import sys
from collections import defaultdict


def mean(v):
    return sum(v) / len(v)


def se(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


def tval(v):
    s = se(v)
    return mean(v) / s if s and s > 0 else float("nan")


def main():
    args = sys.argv[1:]
    opt = {"arm": "k5", "rule": "U", "min-size": "0", "cells": "0"}
    files = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            opt[args[i][2:]] = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1
    arm, rule, min_size = opt["arm"], opt["rule"], int(opt["min-size"])

    rows = defaultdict(dict)
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["head"], r["dataset"], int(r["size"]), int(r["seed"]))
                rows[key][r["arm"]] = r

    # paired records
    recs = []
    fail = defaultdict(int)
    for (head, ds, size, seed), arms in sorted(rows.items()):
        if "off" not in arms or arm not in arms:
            continue
        o, k = arms["off"], arms[arm]
        if o["finite"] != "1":
            fail["off"] += 1
        if k["finite"] != "1":
            fail[arm] += 1
        if o["finite"] != "1" or k["finite"] != "1":
            continue
        staged_off = o["staged"] == "1"
        if rule == "S" and not staged_off:
            continue
        if rule == "G" and size < min_size:
            continue
        recs.append(
            dict(
                head=head,
                ds=ds,
                size=size,
                seed=seed,
                staged_off=staged_off,
                d=float(k["nll"]) - float(o["nll"]),
                r=float(k["rmse"]) / float(o["rmse"]) - 1.0,
                cov_o=float(o["cov90"]),
                cov_k=float(k["cov90"]),
                t_o=float(o["secs"]),
                t_k=float(k["secs"]),
            )
        )
    print(f"arm {arm} - off, rule {rule}" + (f" (size >= {min_size})" if rule == "G" else ""))
    print(f"paired fits: {len(recs)}; non-finite or failed: off {fail['off']}, {arm} {fail[arm]}")
    if not recs:
        return

    def group(keyf):
        g = defaultdict(list)
        for r in recs:
            g[keyf(r)].append(r)
        return g

    # ---- by size
    print("\nBy pool size (mean over head x dataset pairs; se over pair means)")
    print(f"{'size':>6} {'pairs':>5} {'dNLL':>8} {'se':>6} {'lower':>9} {'dRMSE%':>7} {'se':>5} "
          f"{'cov off':>7} {'cov on':>7} {'time x':>6} {'staged off':>10}")
    for size, rs in sorted(group(lambda r: r["size"]).items()):
        pm = defaultdict(list)
        pr = defaultdict(list)
        for r in rs:
            pm[(r["head"], r["ds"])].append(r["d"])
            pr[(r["head"], r["ds"])].append(r["r"])
        dm = [mean(v) for v in pm.values()]
        rm = [mean(v) for v in pr.values()]
        low = sum(r["d"] < 0 for r in rs)
        print(f"{size:>6} {len(dm):>5} {mean(dm):>+8.4f} {se(dm):>6.4f} {low:>4}/{len(rs):<4} "
              f"{100 * mean(rm):>+7.2f} {100 * se(rm):>5.2f} "
              f"{mean([r['cov_o'] for r in rs]):>7.3f} {mean([r['cov_k'] for r in rs]):>7.3f} "
              f"{sum(r['t_k'] for r in rs) / sum(r['t_o'] for r in rs):>6.2f} "
              f"{sum(r['staged_off'] for r in rs):>5}/{len(rs):<4}")

    # ---- by head x dataset (per-seed means across sizes, se over seeds)
    print("\nBy head x dataset (per-seed mean over sizes; se over seeds)")
    print(f"{'head':<11} {'dataset':<12} {'dNLL':>8} {'se':>6} {'t':>5} {'lower':>9} {'dRMSE%':>7} {'se':>5} {'t':>5}")
    pair_means, pair_rm = [], []
    harm_pairs, harm_rmse = [], []
    for (head, ds), rs in sorted(group(lambda r: (r["head"], r["ds"])).items()):
        bs = defaultdict(list)
        br = defaultdict(list)
        for r in rs:
            bs[r["seed"]].append(r["d"])
            br[r["seed"]].append(r["r"])
        dv = [mean(v) for v in bs.values()]
        rv = [mean(v) for v in br.values()]
        low = sum(r["d"] < 0 for r in rs)
        pair_means.append(mean(dv))
        pair_rm.append(mean(rv))
        flag = ""
        if mean(dv) > 0.02 and tval(dv) > 2:
            harm_pairs.append((head, ds))
            flag += " NLL!"
        if mean(rv) > 0.01 and tval(rv) > 2:
            harm_rmse.append((head, ds))
            flag += " RMSE!"
        print(f"{head:<11} {ds:<12} {mean(dv):>+8.4f} {se(dv):>6.4f} {tval(dv):>+5.1f} {low:>4}/{len(rs):<4} "
              f"{100 * mean(rv):>+7.2f} {100 * se(rv):>5.2f} {tval(rv):>+5.1f}{flag}")

    # ---- cells
    harm_cells = []
    cell_rows = []
    for (head, ds, size), rs in sorted(group(lambda r: (r["head"], r["ds"], r["size"])).items()):
        dv = [r["d"] for r in rs]
        rv = [r["r"] for r in rs]
        bad = mean(dv) > 0.05 and tval(dv) > 2
        if bad:
            harm_cells.append((head, ds, size))
        cell_rows.append(
            f"{head:<11} {ds:<12} {size:>6} {mean(dv):>+8.4f} {se(dv):>6.4f} {tval(dv):>+5.1f} "
            f"{sum(d < 0 for d in dv):>2}/{len(dv):<2} {100 * mean(rv):>+7.2f} "
            f"{mean([r['cov_o'] for r in rs]):>6.3f} {mean([r['cov_k'] for r in rs]):>6.3f} "
            f"{sum(r['t_k'] for r in rs) / sum(r['t_o'] for r in rs):>6.2f} "
            f"{'staged' if rs[0]['staged_off'] else 'joint':>6}{'  HARM' if bad else ''}"
        )
    if opt["cells"] == "1":
        print("\nCells")
        print(f"{'head':<11} {'dataset':<12} {'size':>6} {'dNLL':>8} {'se':>6} {'t':>5} {'low':>5} "
              f"{'dRMSE%':>7} {'cov off':>6} {'cov on':>6} {'time x':>6} {'off is':>6}")
        print("\n".join(cell_rows))

    # ---- pooled and the rule
    d_all = [r["d"] for r in recs]
    low = sum(d < 0 for d in d_all)
    print("\nPooled")
    print(f"  dNLL over fits {mean(d_all):+.4f}; over pair means {mean(pair_means):+.4f} "
          f"(se {se(pair_means):.4f}, t {tval(pair_means):+.1f}, {len(pair_means)} pairs)")
    print(f"  lower on {low}/{len(d_all)} fits ({100 * low / len(d_all):.0f} %)")
    print(f"  dRMSE over pair means {100 * mean(pair_rm):+.2f} % (se {100 * se(pair_rm):.2f})")
    print(f"  coverage off {mean([r['cov_o'] for r in recs]):.3f}, on {mean([r['cov_k'] for r in recs]):.3f}")
    print(f"  wall time, sum on / sum off: {sum(r['t_k'] for r in recs) / sum(r['t_o'] for r in recs):.2f}")
    r1 = mean(pair_means) <= -0.02 and tval(pair_means) <= -2 and low / len(d_all) >= 0.6
    r2 = not harm_pairs and not harm_cells
    r3 = mean(pair_rm) <= 0.005 and not harm_rmse
    r4 = fail[arm] <= fail["off"]
    print("\nDecision rule")
    print(f"  R1 benefit           {'pass' if r1 else 'FAIL'}")
    print(f"  R2 no harm           {'pass' if r2 else 'FAIL'}  pairs {harm_pairs} cells {harm_cells}")
    print(f"  R3 point predictions {'pass' if r3 else 'FAIL'}  pairs {harm_rmse}")
    print(f"  R4 robustness        {'pass' if r4 else 'FAIL'}")


if __name__ == "__main__":
    main()
