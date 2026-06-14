# Dense tau_c Scaling Test

This folder is for testing whether the apparent `r* ~ tau_c^-1/2` relation is
robust enough to call a candidate scaling law.

## Critical interpretation

Do not call it a new law just because the power-law fit looks good.
It becomes convincing only if:

- the slope is stable near `-0.5`,
- the confidence interval is reasonably tight,
- leave-one-out slopes do not jump wildly,
- power law beats plausible alternatives by AICc,
- the trend survives decoder/bin/lag sensitivity later.

The strongest safe language is:

> candidate finite-range scaling law in this optimized escape-rate model

## What to upload to the cluster

Upload the repository root, or at minimum:

- `code/`
- `cluster_dense_tau/`

Run commands from the repository root, not from inside `cluster_dense_tau`.

## Python packages needed

The code uses:

- numpy
- pandas
- scipy
- matplotlib
- pyyaml

## Step 1: pilot timing

Run one tau point first:

```bash
bash cluster_dense_tau/run_pilot.sh
```

At the end it prints `elapsed_seconds` and writes:

```text
runs/dense_tau_jobs/tau_0p1_be_1p0_bc_0p03/tables/tau_sweep_result.csv
```

Full 11-point wall time for a SLURM array is approximately the slowest single
tau job, not 11 times the job, because all tau values run in parallel.
For serial running, multiply the pilot time by about 11.

## Step 2: full SLURM array

Edit `slurm_dense_tau_array.sh` to activate your conda/venv environment, then:

```bash
sbatch cluster_dense_tau/slurm_dense_tau_array.sh
```

Check progress:

```bash
squeue -u "$USER"
```

After all array jobs finish, run:

```bash
sbatch cluster_dense_tau/slurm_analyze_dense_tau.sh
```

or directly:

```bash
python3 cluster_dense_tau/analyze_dense_tau_scaling.py
```

## What to send back

Send back this folder:

```text
runs/dense_tau_jobs/
runs/dense_tau_analysis/
logs/
```

If transfer size is an issue, send only:

```text
runs/dense_tau_jobs/*/tables/tau_sweep_result.csv
runs/dense_tau_jobs/*/metadata.json
runs/dense_tau_analysis/tables/
runs/dense_tau_analysis/figures/
logs/
```

## Expected time

Be conservative.

One tau job performs roughly:

```text
6 restarts x (1 initial + up to 8 sweeps x 12 coordinate-neighbor evaluations)
```

Each evaluation uses:

```text
20 trials x 3 seeds = 60 simulations
```

Worst-case per tau is therefore about `35,000` simulations. Some jobs stop
earlier if the coordinate search converges.

Rough practical expectation:

- fast cluster CPU: 2-8 hours per tau job,
- ordinary shared CPU: 8-24 hours per tau job,
- serial run of all 11 tau values: likely days.

Use the pilot timing rather than trusting these estimates.
