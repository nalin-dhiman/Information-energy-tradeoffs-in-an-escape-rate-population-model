#!/usr/bin/env bash
set -euo pipefail

# Direct-run parallel launcher for systems without SLURM.
# Run from the folder that contains both `code/` and `cluster_dense_tau/`.
#
# Recommended usage:
#   nohup bash cluster_dense_tau/run_parallel_background.sh > logs/dense_tau_master.log 2>&1 &
#   disown

PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
N_TRIALS="${N_TRIALS:-20}"
SEEDS="${SEEDS:-0,1,2}"
RESTARTS="${RESTARTS:-6}"
MAX_STEPS="${MAX_STEPS:-8}"
BETA_E="${BETA_E:-1}"
BETA_C="${BETA_C:-0.03}"

mkdir -p logs

echo "Dense tau run started: $(date)"
echo "PARALLEL_JOBS=${PARALLEL_JOBS}"
echo "N_TRIALS=${N_TRIALS}"
echo "SEEDS=${SEEDS}"
echo "RESTARTS=${RESTARTS}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "BETA_E=${BETA_E}"
echo "BETA_C=${BETA_C}"

cat cluster_dense_tau/tau_values_dense.txt | xargs -I{} -P "${PARALLEL_JOBS}" bash -c '
set -euo pipefail
tau="$1"
tag="${tau//./p}"
echo "Launching tau=${tau} at $(date)"
python3 -u cluster_dense_tau/run_dense_tau_single.py \
  --tau_c "$tau" \
  --beta_e "'"${BETA_E}"'" \
  --beta_c "'"${BETA_C}"'" \
  --n_trials "'"${N_TRIALS}"'" \
  --seeds "'"${SEEDS}"'" \
  --restarts "'"${RESTARTS}"'" \
  --max_steps "'"${MAX_STEPS}"'" \
  > "logs/tau_${tag}.out" 2> "logs/tau_${tag}.err"
echo "Finished tau=${tau} at $(date)"
' _ {}

echo "All tau jobs finished: $(date)"
python3 cluster_dense_tau/analyze_dense_tau_scaling.py > logs/dense_tau_analysis.out 2> logs/dense_tau_analysis.err
echo "Analysis finished: $(date)"
