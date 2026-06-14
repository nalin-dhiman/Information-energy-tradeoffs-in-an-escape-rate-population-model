# Information-energy tradeoffs in an escape-rate population model

This repository contains code, configuration files, processed data tables, and generated figures for simulations of information-energy tradeoffs in a stochastic escape-rate population model.

The repository is organized as a computational release and contains only code, data tables, configuration files, and generated image assets.

## Contents

- `code/src/`: simulation, model, stimulus, estimator, and I/O utilities.
- `code/configs/`: YAML configuration files for stimulus, model, estimator, and objective settings.
- `code/scripts/`: analysis and plotting scripts.
- `data/results_tables/`: consolidated processed tables used by the analysis scripts.
- `data/run_tables/`: additional processed run tables.
- `figures/`: generated figure files.
- `cluster_dense_tau/`: scripts for the dense stimulus-time-scale sweep on a cluster or workstation.

## Basic Usage

Run scripts from the repository root. For example:

```bash
python3 code/scripts/b7_c_pubfigs.py
python3 code/scripts/b11_plot_numeric_controls.py
```

The dense stimulus-time-scale sweep can be run with:

```bash
bash cluster_dense_tau/run_parallel_background.sh
```

or adapted to SLURM using the scripts in `cluster_dense_tau/`.

## Python Dependencies

The scripts use standard scientific Python packages:

- numpy
- scipy
- pandas
- matplotlib
- pyyaml
- scikit-learn

## Scope

The energy term in these simulations is a firing-rate proxy, and the decoding estimate is an operational linear-decoder quantity. The processed tables and scripts are provided to make the computational workflow inspectable and reusable.
