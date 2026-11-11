#!/bin/bash
# Cycle 15 summaries (run from research/ngboost). Pairs on (dataset, seed).
R=reports; C=reports/c15
# R54c+e: StudentT staged vs joint, 10 seeds (seeds 0-5 from R54c, 6-9 + protein 5 from R54e)
python3 -I - <<'PY'
import csv, glob
rows = {}
for f in ["reports/R54c_studentt_staged_combined.csv"] + sorted(glob.glob("reports/c15/R54e_st_*.csv")):
    for r in csv.DictReader(l for l in open(f) if l.strip() and not l.startswith("#")):
        rows[(r["dataset"], r["seed"], r["variant"])] = r
out = "reports/c15/R54ce_studentt_staged_10seeds.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values())).keys()))
    w.writeheader()
    for k in sorted(rows):
        w.writerow(rows[k])
PY
echo "== R54c+e StudentT: staged − joint"
python3 -I pair.py $C/R54ce_studentt_staged_10seeds.csv auto+joint $C/R54ce_studentt_staged_10seeds.csv auto+staged
# legacy run was split: tnoise 0-4 (part0), the rest in *_a / *_b
cat $C/R55b_stcrps_legacy_part0.csv $C/R55b_stcrps_legacy_*_[ab].csv $C/R55c_stcrps_legacy_kin8nm_s*.csv 2>/dev/null > $C/R55b_stcrps_legacy_all.csv
echo "== R55b StudentT CRPS: fixed − legacy"
python3 -I pair.py $C/R55b_stcrps_legacy_all.csv auto $C/R55b_stcrps_fixed.csv auto
# legacy kin8nm was split (seeds 0-1 / 0 in the main run, the rest in *_b)
for h in gammacrps weibullcrps; do
  cat $C/R58_${h}_legacy_kin8nm.csv $C/R58_${h}_legacy_kin8nm_b.csv $C/R58c_${h}_legacy_kin8nm_s*.csv 2>/dev/null > $C/R58_${h}_legacy_kin8nm_all.csv
  cat $C/R58_${h}_fixed_kin8nm.csv $C/R58c_${h}_fixed_kin8nm_c.csv 2>/dev/null > $C/R58_${h}_fixed_kin8nm_all.csv
done
for h in gammacrps weibullcrps; do
  for ds in lognorm_het support_cost kin8nm; do
    echo "== R58 $h $ds: fixed − legacy"
    L=$C/R58_${h}_legacy_$ds.csv; [ $ds = kin8nm ] && L=$C/R58_${h}_legacy_kin8nm_all.csv
    F=$C/R58_${h}_fixed_$ds.csv; [ $ds = kin8nm ] && F=$C/R58_${h}_fixed_kin8nm_all.csv
    python3 -I pair.py $L auto $F auto | tail -n +2
  done
done
for h in gamma weibull; do
  for ds in lognorm_het support_cost kin8nm; do
    echo "== R58 ${h}crps fixed − ${h} LogScore, $ds"
    F=$C/R58_${h}crps_fixed_$ds.csv; [ $ds = kin8nm ] && F=$C/R58_${h}crps_fixed_kin8nm_all.csv
    python3 -I pair.py $C/R58_${h}_logscore.csv auto $F auto | tail -n +2
  done
done
