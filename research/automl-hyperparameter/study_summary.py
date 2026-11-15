"""Summaries for the Cycle 18 tuner experiments (c18_bench output).

  python3 -I study_summary.py search reports/A4_search_real.csv [more.csv]
  python3 -I study_summary.py select reports/A5_select.csv
  python3 -I study_summary.py lrcap  reports/A3_lrcap.csv
"""
import csv, math, sys
from collections import defaultdict


def mean(v):
    v = [x for x in v if math.isfinite(x)]
    return sum(v) / len(v) if v else float("nan")


def se(v):
    v = [x for x in v if math.isfinite(x)]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


def rows(paths):
    for p in paths:
        for r in csv.DictReader(l for l in open(p) if l.strip() and not l.startswith("#")):
            yield r


def search(paths):
    # best-so-far validation score and the test NLL of the incumbent, per
    # (dataset, objective, sampler), averaged over sampler seeds
    d = defaultdict(lambda: defaultdict(dict))
    for r in rows(paths):
        d[(r["dataset"], r["objective"], r["sampler"])][int(r["hp_seed"])][int(r["trial"])] = (
            float(r["best_score"]), float(r["test_at_best"]), float(r["score"]))
    T = [5, 10, 20, 30, 40, 60, 80]
    last = None
    for key in sorted(d):
        ds, obj, sm = key
        if (ds, obj) != last:
            last = (ds, obj)
            print(f"\n== {ds} [{obj} objective]: best validation score after t trials (mean ± se over sampler seeds) | test NLL of incumbent")
            print(f"{'sampler':<10}{'seeds':>6}" + "".join(f"{'t=' + str(t):>17}" for t in T) + f"{'test@end':>18}{'dup%':>6}")
        runs = d[key]
        nt = min(len(v) for v in runs.values())
        line = f"{sm:<10}{len(runs):>6}"
        for t in T:
            if t > nt:
                line += f"{'':>17}"
                continue
            b = [runs[h][t - 1][0] for h in runs]
            line += f"{mean(b):>9.4f} ±{se(b):.4f}"
        te = [runs[h][nt - 1][1] for h in runs]
        # share of trials whose score repeats an earlier trial's exactly
        dup = []
        for h in runs:
            seen, n = set(), 0
            for t in range(nt):
                s = runs[h][t][2]
                n += s in seen
                seen.add(s)
            dup.append(100.0 * n / nt)
        print(line + f"{mean(te):>10.4f} ±{se(te):.4f}{mean(dup):>6.1f}")


def select(paths):
    d = defaultdict(dict)
    for r in rows(paths):
        d[(r["dataset"], r["seed"])][r["cand"]] = (float(r["holdout"]), float(r["cv5"]), float(r["test_nll"]))
    cands = []
    for v in d.values():
        for c in v:
            if c not in cands:
                cands.append(c)
    dss = []
    for k in d:
        if k[0] not in dss:
            dss.append(k[0])
    print("test NLL (mean over seeds). 'by holdout' / 'by cv5': the candidate with the best score on one")
    print("10 % holdout / on 5-fold CV over train+holdout; the model evaluated is the same holdout-fitted one.")
    print(f"{'dataset':<9}{'n':>3}" + "".join(f"{c:>11}" for c in cands) + f"{'by holdout':>12}{'by cv5':>10}{'oracle':>9}"
          f"{'cv5−holdout (se)':>20}{'cv5 wins':>10}   picks by holdout | by cv5")
    for ds in dss:
        ks = [k for k in d if k[0] == ds and len(d[k]) == len(cands)]
        if not ks:
            continue
        per = {c: [d[k][c][2] for k in ks] for c in cands}
        bh, bc, orc, diff = [], [], [], []
        ph, pc = defaultdict(int), defaultdict(int)
        for k in ks:
            ch = min(cands, key=lambda c: d[k][c][0])
            cc = min(cands, key=lambda c: d[k][c][1])
            ph[ch] += 1
            pc[cc] += 1
            bh.append(d[k][ch][2]); bc.append(d[k][cc][2]); orc.append(min(d[k][c][2] for c in cands))
            diff.append(d[k][cc][2] - d[k][ch][2])
        print(f"{ds:<9}{len(ks):>3}" + "".join(f"{mean(per[c]):>11.4f}" for c in cands)
              + f"{mean(bh):>12.4f}{mean(bc):>10.4f}{mean(orc):>9.4f}{mean(diff):>+11.4f} ({se(diff):.4f})"
              + f"{sum(1 for x in diff if x < 0):>5}/{sum(1 for x in diff if x != 0):<4}   "
              + " ".join(f"{c}:{ph[c]}" for c in cands) + " | " + " ".join(f"{c}:{pc[c]}" for c in cands))


def lrcap(paths):
    d = defaultdict(list)
    for r in rows(paths):
        d[(r["dataset"], r["fit"], float(r["lr"]), r["policy"])].append(
            (float(r["test_nll"]), int(r["n_iter"]), int(r["n_est"]), float(r["secs"])))
    pols = ["cap500_p50", "cap1500_p50", "capS_p50", "capS_pS"]
    for ds in sorted({k[0] for k in d}):
        for fit in sorted({k[1] for k in d if k[0] == ds}):
            print(f"\n== {ds} / {fit}: mean test NLL (mean rounds kept; * = some fit reached its cap, joint only)")
            print(f"{'lr':>7}" + "".join(f"{p:>24}" for p in pols))
            for lr in sorted({k[2] for k in d if k[0] == ds and k[1] == fit}, reverse=True):
                line = f"{lr:>7}"
                for p in pols:
                    v = d.get((ds, fit, lr, p))
                    if not v:
                        line += f"{'':>24}"
                        continue
                    hit = fit == "joint" and any(x[1] >= x[2] for x in v)
                    line += f"{mean(x[0] for x in v):>12.4f} ({mean(x[1] for x in v):>6.0f}{'*' if hit else ' '})  "
                print(line)


{"search": search, "select": select, "lrcap": lrcap}[sys.argv[1]](sys.argv[2:])
