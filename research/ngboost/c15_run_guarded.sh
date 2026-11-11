#!/bin/bash
# Like c15_run.sh, but skips the job when on battery below 25 %.
b=$(pmset -g batt | grep -o '[0-9]*%' | tr -d %)
if pmset -g batt | grep -q "Battery Power" && [ "${b:-100}" -lt 25 ]; then
  echo "SKIP $1 battery ${b}% $(date +%H:%M:%S)"; exit 0
fi
exec "$(dirname "$0")/c15_run.sh" "$@"
