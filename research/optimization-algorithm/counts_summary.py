#!/usr/bin/env python3
"""Summary of `c18_bench --mode counts` (Cycle 26, O17).

    python3 -I counts_summary.py --arm nb_x5 --ref nb [--rule 1] FILE...

Paired differences arm - ref on the same dataset, design, cell, seed and
training size. One table row per dataset x design x size (iid) or
dataset x design x pooled/per-league (chronological); pooled figures are
means over dataset rows of a design group. `--rule 1` prints the O17
decision rule (meant for --arm nb_x5/nb_x3 --ref nb). Stdlib only.
"""
import csv
import math
import sys
from collections import defaultdict

REAL = ("fbo_", "fbf_", "doctor", "biochem", "nmes")


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
    opt = {"arm": "nb_x5", "ref": "nb", "rule": "0"}
    files = []
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            opt[args[i][2:]] = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1
    arm, ref = opt["arm"], opt["ref"]
    rows = defaultdict(dict)
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["dataset"] == "dataset":
                    continue
                key = (r["dataset"], r["design"], r["cell"], r["seed"], r["n_train"])
                rows[key][r["arm"]] = r
    recs = []
    fail = defaultdict(int)
    for (ds, design, cell, seed, n), arms in sorted(rows.items()):
        if arm not in arms or ref not in arms:
            continue
        a, b = arms[arm], arms[ref]
        for name, r in ((arm, a), (ref, b)):
            if r["finite"] != "1":
                fail[name] += 1
        if a["finite"] != "1" or b["finite"] != "1":
            continue
        if design == "chrono":
            grp = "pooled leagues" if cell == "all" else "per league"
        else:
            grp = f"n={n}"
        recs.append(
            dict(
                ds=ds, design=design, grp=grp, unit=(cell, seed),
                d=float(a["nll"]) - float(b["nll"]),
                r=float(a["rmse"]) / float(b["rmse"]) - 1.0,
                nll_a=float(a["nll"]), nll_b=float(b["nll"]),
                bias_a=float(a["bias"]), bias_b=float(b["bias"]),
                cov_a=float(a["cov90"]), cov_b=float(b["cov90"]),
                t_a=float(a["secs"]), t_b=float(b["secs"]),
            )
        )
    print(f"{arm} - {ref}; paired fits {len(recs)}; failed or non-finite: {arm} {fail[arm]}, {ref} {fail[ref]}")
    if not recs:
        return
    g = defaultdict(list)
    for r in recs:
        g[(r["design"], r["ds"], r["grp"])].append(r)

    def nkey(k):
        design, ds, grp = k
        n = int(grp[2:]) if grp.startswith("n=") else 0
        return (design, ds, n, grp)

    print(f"\n{'design':<7} {'dataset':<11} {'group':<15} {'n':>3} {'NLL ' + ref:>9} {'dNLL':>8} {'se':>6} {'t':>5} "
          f"{'lower':>7} {'dRMSE%':>7} {'t':>5} {'bias ' + ref:>8} {'bias':>6} {'cov ' + ref:>7} {'cov':>6} {'time x':>6}")
    by_ds = defaultdict(list)  # (design, ds) -> list of row dicts
    for k in sorted(g, key=nkey):
        rs = g[k]
        d = [r["d"] for r in rs]
        rr = [r["r"] for r in rs]
        print(f"{k[0]:<7} {k[1]:<11} {k[2]:<15} {len(rs):>3} {mean([r['nll_b'] for r in rs]):>9.4f} "
              f"{mean(d):>+8.4f} {se(d):>6.4f} {tval(d):>+5.1f} {sum(x < 0 for x in d):>3}/{len(d):<3} "
              f"{100 * mean(rr):>+7.2f} {tval(rr):>+5.1f} {mean([r['bias_b'] for r in rs]):>8.3f} "
              f"{mean([r['bias_a'] for r in rs]):>6.3f} {mean([r['cov_b'] for r in rs]):>7.3f} "
              f"{mean([r['cov_a'] for r in rs]):>6.3f} "
              f"{sum(r['t_a'] for r in rs) / sum(r['t_b'] for r in rs):>6.2f}")
        by_ds[(k[0], k[1])].append((d, rr))

    # dataset-level: all fits of a dataset within a design
    print(f"\n{'design':<7} {'dataset':<11} {'fits':>5} {'dNLL':>8} {'se':>6} {'t':>5} {'lower':>9} {'dRMSE%':>7} {'se':>5} {'t':>5}")
    harm, harm_r = [], []
    ds_means = defaultdict(list)
    ds_rm = defaultdict(list)
    lows = defaultdict(lambda: [0, 0])
    for (design, ds), parts in sorted(by_ds.items()):
        d = [x for p in parts for x in p[0]]
        rr = [x for p in parts for x in p[1]]
        flag = ""
        if mean(d) > 0.005 and tval(d) > 2:
            harm.append((design, ds))
            flag += " NLL!"
        if mean(rr) > 0.01 and tval(rr) > 2:
            harm_r.append((design, ds))
            flag += " RMSE!"
        print(f"{design:<7} {ds:<11} {len(d):>5} {mean(d):>+8.4f} {se(d):>6.4f} {tval(d):>+5.1f} "
              f"{sum(x < 0 for x in d):>4}/{len(d):<4} {100 * mean(rr):>+7.2f} {100 * se(rr):>5.2f} {tval(rr):>+5.1f}{flag}")
        kind = "real" if ds.startswith(REAL) else "synthetic"
        for scope in (kind, "all"):
            ds_means[scope].append(mean(d))
            ds_rm[scope].append(mean(rr))
            lows[scope][0] += sum(x < 0 for x in d)
            lows[scope][1] += len(d)

    print("\nPooled over dataset x design means")
    ok1 = True
    for scope in ("real", "synthetic", "all"):
        v = ds_means[scope]
        if not v:
            continue
        lo, n = lows[scope]
        print(f"  {scope:<9} {len(v):>2} rows: dNLL {mean(v):+.4f} (se {se(v):.4f}, t {tval(v):+.1f}); "
              f"lower on {lo}/{n} fits ({100 * lo / n:.0f} %); dRMSE {100 * mean(ds_rm[scope]):+.2f} % "
              f"(se {100 * se(ds_rm[scope]):.2f})")
        if scope in ("real", "all"):
            ok1 &= mean(v) <= -0.005 and tval(v) <= -2 and lo / n >= 0.6
    if opt["rule"] == "1":
        r3 = mean(ds_rm["all"]) <= 0.005 and not harm_r
        print("\nDecision rule (O17)")
        print(f"  R1 benefit           {'pass' if ok1 else 'FAIL'}")
        print(f"  R2 no harm           {'pass' if not harm else 'FAIL'}  {harm}")
        print(f"  R3 point predictions {'pass' if r3 else 'FAIL'}  {harm_r}")
        print(f"  R4 robustness        {'pass' if fail[arm] <= fail[ref] else 'FAIL'}")


if __name__ == "__main__":
    main()
