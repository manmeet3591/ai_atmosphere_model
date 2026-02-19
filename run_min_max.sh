#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="norm_daily_mean_per_var_json"
MAX_JOBS=16

PLEV=(50 100 150 200 250 300 400 500 600 700 850 925 1000)
VARS=(v z t q)

running=0

for v in "${VARS[@]}"; do
  for lev in "${PLEV[@]}"; do
    python get_data_var.py --variable "$v" --pressure-level "$lev" --out-dir "$OUT_DIR" &
    ((running+=1))

    # If we have MAX_JOBS in flight, wait for the whole batch
    if (( running >= MAX_JOBS )); then
      wait
      running=0
    fi
  done
done

# wait for any leftovers
wait
