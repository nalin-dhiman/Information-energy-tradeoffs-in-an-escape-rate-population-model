#!/usr/bin/env bash
#SBATCH --job-name=dense_tau
#SBATCH --array=1-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dense_tau_%A_%a.out
#SBATCH --error=logs/dense_tau_%A_%a.err

set -euo pipefail

mkdir -p logs

# Activate your environment here, for example:
# source ~/.bashrc
# conda activate your_environment
# or:
# source venv/bin/activate

tau="$(sed -n "${SLURM_ARRAY_TASK_ID}p" cluster_dense_tau/tau_values_dense.txt)"

python3 cluster_dense_tau/run_dense_tau_single.py \
  --tau_c "$tau" \
  --beta_e 1 \
  --beta_c 0.03 \
  --n_trials 20 \
  --seeds 0,1,2 \
  --restarts 6 \
  --max_steps 8
