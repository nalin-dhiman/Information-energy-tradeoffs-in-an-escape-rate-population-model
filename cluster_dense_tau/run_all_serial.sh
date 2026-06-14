#!/usr/bin/env bash
set -euo pipefail

# Use this only if you do not have a scheduler. On a cluster, prefer SLURM array.

while read -r tau; do
  [[ -z "$tau" ]] && continue
  python3 cluster_dense_tau/run_dense_tau_single.py \
    --tau_c "$tau" \
    --beta_e 1 \
    --beta_c 0.03 \
    --n_trials 20 \
    --seeds 0,1,2 \
    --restarts 6 \
    --max_steps 8
done < cluster_dense_tau/tau_values_dense.txt

python3 cluster_dense_tau/analyze_dense_tau_scaling.py
