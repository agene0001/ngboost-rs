"""Leaf 20 against leaf 1 for the O2 head files (the two runs share their
variant labels, so `c18_pair.py` cannot pair them).

  python3 -I research/optimization-algorithm/leaf_pair.py cat3 cat5 hurdle …

pairs `reports/O2_heads_<head>_ml1.csv` with `…_ml20.csv` on (dataset,
variant, seed) and prints the mean difference ± standard error and the
seeds on which leaf 20 is lower, for NLL and RMSE. Run from the repo root.
"""
import sys,math,collections
R='research/optimization-algorithm/reports/'
def load(f):
    d={};hdr=None
    try: fh=open(f)
    except OSError: return d
    for line in fh:
        c=line.strip().split(',')
        if c[0]=='dataset': hdr=c; continue
        if hdr is None or len(c)!=len(hdr) or line.startswith('#'): continue
        r=dict(zip(hdr,c))
        try: int(r['seed'])
        except: continue
        d[(r['dataset'],r['variant'],int(r['seed']))]=r
    return d
def f(x):
    try: return float(x)
    except: return float('nan')
for head in sys.argv[1:]:
    a,b=load(R+f'O2_heads_{head}_ml1.csv'),load(R+f'O2_heads_{head}_ml20.csv')
    groups=collections.OrderedDict()
    for k in b:
        if k in a: groups.setdefault(k[:2],[]).append(k[2])
    for (ds,v),seeds in groups.items():
        out=[]
        for met in ('nll','rmse'):
            dif=[f(b[(ds,v,s)][met])-f(a[(ds,v,s)][met]) for s in seeds]
            dif=[x for x in dif if math.isfinite(x)]
            n=len(dif)
            if n<2: out.append('   n/a'); continue
            m=sum(dif)/n; se=(sum((x-m)**2 for x in dif)/(n-1)/n)**.5
            out.append(f"{met} {m:+.4f} ({se:.4f}) {sum(x<0 for x in dif)}/{n}")
        base=sum(f(a[(ds,v,s)]['nll']) for s in seeds)/len(seeds)
        print(f"{head:8s} {ds:16s} {v:11s} leaf1 nll {base:8.4f} | leaf20-leaf1: {out[0]} | {out[1]}")
