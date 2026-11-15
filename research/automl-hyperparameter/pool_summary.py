"""Summarise a c18_bench `--mode pool` CSV.

  python3 -I pool_summary.py reports/A1_pool.csv

Sections: (1) tuned vs default on held-out test, (2) selection by the
reused block vs the untouched block as the trial budget grows, (3) score
optimism and rank agreement with test, (4) per-knob sensitivity.
Configs 0 and 1 are the reference defaults; 2.. are uniform random draws.
"""
import csv, math, random, sys
from collections import defaultdict

rows = [r for r in csv.DictReader(open(sys.argv[1]))]
F = ["lr", "depth", "msl", "mb", "cs", "n_iter", "secs", "val_reuse", "val_nested",
     "test_nll", "test_crps", "test_rmse", "cov90", "calib", "test2_nll"]
by = defaultdict(dict)
for r in rows:
    d = {k: float(r[k]) for k in F}
    if math.isfinite(d["val_reuse"]) and math.isfinite(d["val_nested"]) and math.isfinite(d["test_nll"]):
        by[(r["dataset"], int(r["seed"]))][int(r["cfg"])] = d
datasets = sorted({k[0] for k in by})


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def se(v):
    v = list(v)
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den > 0 else float("nan")


print("== 1. held-out test NLL: defaults vs selected (mean over seeds; Δ vs default lr .05)")
print(f"{'dataset':<9}{'seeds':>6}{'default':>9}{'lr.02':>9}{'median':>9}{'by reuse':>10}{'by nested':>10}{'oracle':>9}"
      f"{'Δ reuse':>17}{'Δ nested':>17}  wins(reuse)")
for ds in datasets:
    ks = [k for k in by if k[0] == ds]
    dflt, lr02, med, br, bn, orc, dr, dn = [], [], [], [], [], [], [], []
    for k in ks:
        p = by[k]
        rnd = [c for c in p if c >= 2]
        t = sorted(p[c]["test_nll"] for c in rnd)
        cr = min(p, key=lambda c: p[c]["val_reuse"])
        cn = min(p, key=lambda c: p[c]["val_nested"])
        dflt.append(p[0]["test_nll"]); lr02.append(p[1]["test_nll"]); med.append(t[len(t) // 2])
        br.append(p[cr]["test_nll"]); bn.append(p[cn]["test_nll"]); orc.append(t[0])
        dr.append(p[cr]["test_nll"] - p[0]["test_nll"]); dn.append(p[cn]["test_nll"] - p[0]["test_nll"])
    print(f"{ds:<9}{len(ks):>6}{mean(dflt):>9.4f}{mean(lr02):>9.4f}{mean(med):>9.4f}{mean(br):>10.4f}{mean(bn):>10.4f}{mean(orc):>9.4f}"
          f"{mean(dr):>+9.4f} ({se(dr):.4f}){mean(dn):>+9.4f} ({se(dn):.4f})  {sum(1 for v in dr if v < 0)}/{len(dr)}")

print("\n== 1b. the selected configurations (by reuse score)")
for k in sorted(by):
    p = by[k]
    c = min(p, key=lambda c: p[c]["val_reuse"])
    d = p[c]
    print(f"{k[0]:<9} seed {k[1]}: cfg {c:>3} lr {d['lr']:.3f} depth {int(d['depth'])} leaf {int(d['msl']):>3} mb {d['mb']:.2f} cs {d['cs']:.2f}"
          f"  rounds {int(d['n_iter']):>5}  test {d['test_nll']:.4f} (default {p[0]['test_nll']:.4f}; RMSE {d['test_rmse']:.4f} vs {p[0]['test_rmse']:.4f}; cov90 {d['cov90']:.3f} vs {p[0]['cov90']:.3f})")

print("\n== 2. expected test NLL of the argmin after k random trials (2000 resamples; Δ vs default)")
random.seed(0)
KS = [5, 10, 20, 40, 80]
print(f"{'dataset':<9}" + "".join(f"{'k=' + str(k) + ' reuse':>13}{'nested':>9}" for k in KS) + f"{'regret(reuse−nested) k=40':>28}")
for ds in datasets:
    ks = [k for k in by if k[0] == ds]
    line = f"{ds:<9}"
    reg = None
    for kk in KS:
        a, b = [], []
        for k in ks:
            p = by[k]
            rnd = [c for c in p if c >= 2]
            for _ in range(2000):
                sub = random.sample(rnd, min(kk, len(rnd)))
                a.append(p[min(sub, key=lambda c: p[c]["val_reuse"])]["test_nll"] - p[0]["test_nll"])
                b.append(p[min(sub, key=lambda c: p[c]["val_nested"])]["test_nll"] - p[0]["test_nll"])
        line += f"{mean(a):>+13.4f}{mean(b):>+9.4f}"
        if kk == 40:
            reg = mean(a) - mean(b)
    print(line + f"{reg:>+28.4f}")

print("\n== 3. optimism (test − score) and rank agreement with test, random configs")
print(f"{'dataset':<9}{'test−reuse':>12}{'test−nested':>13}{'ρ(reuse,test)':>15}{'ρ(nested,test)':>16}{'ρ(test2,test)':>15}"
      f"   ρ(optimism_reuse, knob): lr depth leaf mb cs rounds")
for ds in datasets:
    ks = [k for k in by if k[0] == ds]
    o1, o2, s1, s2, s3 = [], [], [], [], []
    kc = defaultdict(list)
    for k in ks:
        p = by[k]
        rnd = [c for c in p if c >= 2]
        t = [p[c]["test_nll"] for c in rnd]
        vr = [p[c]["val_reuse"] for c in rnd]
        vn = [p[c]["val_nested"] for c in rnd]
        opt = [a - b for a, b in zip(t, vr)]
        o1.append(mean(opt)); o2.append(mean(a - b for a, b in zip(t, vn)))
        s1.append(spearman(vr, t)); s2.append(spearman(vn, t))
        s3.append(spearman([p[c]["test2_nll"] for c in rnd], t))
        for knob in ["lr", "depth", "msl", "mb", "cs", "n_iter"]:
            kc[knob].append(spearman([p[c][knob] for c in rnd], opt))
    print(f"{ds:<9}{mean(o1):>+12.4f}{mean(o2):>+13.4f}{mean(s1):>15.3f}{mean(s2):>16.3f}{mean(s3):>15.3f}   "
          + " ".join(f"{mean(kc[k]):+.2f}" for k in ["lr", "depth", "msl", "mb", "cs", "n_iter"]))

print("\n== 4. sensitivity: Spearman ρ(knob, test NLL) and tercile means of test NLL − per-split median")
for metric in ["test_nll", "test_rmse"]:
    print(f"-- {metric}")
    print(f"{'dataset':<9}" + "".join(f"{k:>24}" for k in ["lr", "depth", "min_leaf", "minibatch", "col_sample"]))
    for ds in datasets:
        ks = [k for k in by if k[0] == ds]
        line = f"{ds:<9}"
        for knob in ["lr", "depth", "msl", "mb", "cs"]:
            rho, terc = [], [[], [], []]
            for k in ks:
                p = by[k]
                rnd = [c for c in p if c >= 2]
                t = [p[c][metric] for c in rnd]
                x = [p[c][knob] for c in rnd]
                rho.append(spearman(x, t))
                med = sorted(t)[len(t) // 2]
                o = sorted(range(len(rnd)), key=lambda i: x[i])
                n3 = len(o) // 3
                for g in range(3):
                    idx = o[g * n3:(g + 1) * n3] if g < 2 else o[2 * n3:]
                    terc[g].append(mean(t[i] - med for i in idx))
            line += f"  {mean(rho):+.2f} [{mean(terc[0]):+.3f} {mean(terc[1]):+.3f} {mean(terc[2]):+.3f}]"
        print(line)
