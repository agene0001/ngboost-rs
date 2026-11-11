#!/bin/bash
# usage: c15_run.sh NAME LEGACY ARGS...   (run from repo root)
name=$1; legacy=$2; shift 2
out=research/ngboost/reports/c15/$name
NGBOOST_CRPS_METRIC_LEGACY=$legacy /Volumes/Build/ngboost-rs/rb_c15 "$@" > $out.csv 2> $out.err
echo "DONE $name exit=$? $(date +%H:%M:%S)"
