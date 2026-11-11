#!/usr/bin/env bash
# Usage: research/ngboost/log_run.sh <change_id> <decision> <report.csv>
# Appends every data row of a research_bench report to experiments.csv with
# timestamp + commit metadata. Header of report: dataset,seed,variant,...
set -euo pipefail
change="$1"; decision="$2"; report="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
cd "$(dirname "$0")"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
commit=$(git -C ../.. rev-parse --short HEAD)
rg -v '^(dataset,|#|$)' "$report" | while IFS=, read -r ds seed variant n_iter secs train_nll nll crps rmse cov50 cov90 calib mean_sigma extra extra2; do
  echo "$ts,$commit,$change,$ds,70/10/20,$seed,$variant,$n_iter,$secs,$train_nll,$nll,$crps,$rmse,$cov50,$cov90,$calib,$mean_sigma,$decision" >> experiments.csv
done
echo "logged $(rg -c "$change" experiments.csv) rows for $change"
