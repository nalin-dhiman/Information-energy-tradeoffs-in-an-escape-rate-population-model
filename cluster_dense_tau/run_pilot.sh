#!/usr/bin/env bash
set -euo pipefail

# Run this first on the cluster login/interactive node or as a short batch job.
# It benchmarks one middle tau value so you can estimate full wall time.

python3 cluster_dense_tau/run_dense_tau_single.py \
  --tau_c 0.1 \
  --beta_e 1 \
  --beta_c 0.03 \
  --n_trials 20 \
  --seeds 0,1,2 \
  --restarts 3 \
  --max_steps 8

