#!/usr/bin/env bash
#SBATCH --job-name=analyze_tau
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analyze_tau_%j.out
#SBATCH --error=logs/analyze_tau_%j.err

set -euo pipefail

mkdir -p logs

# Activate the same Python environment used for the sweep.

python3 cluster_dense_tau/analyze_dense_tau_scaling.py
