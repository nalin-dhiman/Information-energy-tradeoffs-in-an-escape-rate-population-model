# Information-Energy Tradeoffs in an Escape-Rate Population Model

This repository contains the source code, data, and publication-quality results for the analysis of information processing efficiency in a neural population model based on the hazard (escape-rate) framework.

## Repository Structure

- `code/`: Core implementation of the model and analysis pipeline.
    - `src/`: Neural simulator, stimuli generators, and Mutual Information (MI) estimators.
    - `scripts/`: Production scripts for optimization, sweeps, and plotting.
    - `configs/`: YAML-based configuration system defining experiment parameters.
- `results/`:
    - `figures/`: Publication-ready vector PDF figures (Fig 1-7, including Pareto frontiers and scaling laws).
    - `tables/`: Consolidated statistical tables (CSV) including Pareto optima and time-constant sweeps.

## Key Findings

1. **Efficiency Scaling**: Optimal firing rates show power-law scaling with stimulus timescale ($\tau_c$). We find a scaling exponent of $\approx -0.48$, consistent with metabolic efficiency requirements.
2. **Pareto Frontier**: Stage 2 optimization reveals a significantly extended cost-efficiency frontier compared to Stage 1, highlighting the role of quadratic synaptic and adaptation terms.
3. **Robustness**: Results are consistent across both Gaussian and Switching (telegraph) stimulus protocols.

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pyyaml`

To install dependencies:
```bash
pip install -r code/requirements.txt
```
*(Note: A requirements file is included in the code directory).*

### Running Analyses

The pipeline is organized into versioned stages. To reproduce the final results:

1. **Statistical Consolidation**:
   ```bash
   python3 code/scripts/b7_a_final_stats.py
   ```
2. **Scaling Quantification**:
   ```bash
   python3 code/scripts/b7_b_scaling_fit.py
   ```
3. **Figure Generation**:
   ```bash
   python3 code/scripts/b7_c_pubfigs.py
   ```

## Contact
Nalin Dhiman - [Mail](d24008@students.iitmandi.ac.in)
