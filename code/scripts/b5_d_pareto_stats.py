import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import itertools
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.config import load_config
from src.io.paths import create_run_dir
from src.io.runlog import log_run
from scripts.b5_stage2_optimization import evaluate_point_stage2

def run_stats_verification(args):
    base_cfg = load_config(args.config)
    # create_run_dir auto-increments.
    run_dir = create_run_dir(major_version=5) 
    print(f"Starting Phase 21 (Statistical Pareto Redo): {run_dir}")
    
    # helper to resolve includes
    def resolve_includes(cfg, root=Path(__file__).parent.parent):
        for k, v in cfg.items():
            if isinstance(v, str) and v.endswith('.yaml'):
                p = Path(v)
                if not p.exists(): p = root / v
                if p.exists():
                    print(f"Resolving {k}: {p}")
                    with open(p) as f:
                        cfg[k] = __import__('yaml').safe_load(f)
                        resolve_includes(cfg[k], root) # Recurse
            elif isinstance(v, dict):
                resolve_includes(v, root)
    
    resolve_includes(base_cfg)
    
    # Validation Configuration
    N_TRIALS = 20
    SEEDS = [0, 1, 2] # Explicit verification seeds
    
    # Strict Metadata Logging
    # We need to extract these from base_cfg or define them
    dt = base_cfg['simulation']['dt']
    trial_T = base_cfg['simulation']['T']
    N = base_cfg['simulation']['N']
    bandwidth = base_cfg['stimulus'].get('bandwidth', 5.0)
    dt_eff = 1.0 / (2 * bandwidth)
    
    extra_meta = {
        'notes': 'Phase 21 Audit: Statistical Pareto Verification',
        'n_trials': N_TRIALS,
        'seeds': SEEDS,
        'claim': 'Stage 2 Extends Stage 1',
        'input_sources': ['v2_i', 'v5_c'],
        # Required keys for runlog
        'dt': dt,
        'dt_eff': dt_eff, 
        'N': N,
        'trials': N_TRIALS,
        'trial_T': trial_T,
        # Lists for "sweeps" even if not sweeping here, to satisfy log schema
        'tau_list': [0.02], 
        'seed_list': SEEDS,
        'beta_E_list': [], # Populated from data
        'beta_C_list': []
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    # Load Optima
    s1_path = Path("runs/v2_i/tables/opt_best.csv") 
    s2_path = Path("runs/v5_c/tables/stage2_best.csv")
    
    if not s1_path.exists(): s1_path = Path("runs/v2_a/tables/opt_best.csv") # Fallback
    
    df_s1 = pd.read_csv(s1_path)
    df_s2 = pd.read_csv(s2_path)
    
    # Update lists in meta after loading
    extra_meta['beta_E_list'] = sorted(list(set(df_s1['beta_E'].unique()) | set(df_s2['beta_E'].unique())))
    if 'beta_C' in df_s2.columns:
        extra_meta['beta_C_list'] = sorted(df_s2['beta_C'].unique().tolist())
    else:
        extra_meta['beta_C_list'] = [0.0]
        
    # Re-log with full meta (log_run overwrites run.json)
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    print(f"Loaded {len(df_s1)} Stage 1 points and {len(df_s2)} Stage 2 points.")
    
    # Eval Function Wrapper
    def eval_row(row, cfg, stage_label):
        # Extract theta
        theta = {k: row[k] for k in ['theta0', 'thetaV', 'thetaa', 'thetaVV', 'thetaaa', 'thetaVa'] if k in row}
        # Ensure defaults for missing
        for k in ['thetaVV', 'thetaaa', 'thetaVa']:
            if k not in theta: theta[k] = 0.0
            
        res = evaluate_point_stage2(theta, cfg, n_trials=N_TRIALS, seeds=SEEDS, tau=0.02)
        if not res: return None
        
        # Add metadata
        res['beta_E'] = row['beta_E']
        res['beta_C'] = row.get('beta_C', 0.0)
        res['stage'] = stage_label
        res['seed_count'] = len(SEEDS)
        res['trial_count'] = N_TRIALS
        
        # Calculate Objective stats
        L1 = sum(abs(theta[k]) for k in theta)
        res['L1_theta'] = L1
        baseline = 5.0 # Fixed from config/optimization history
        res['J_mean'] = res['I_lower_mean'] - res['beta_E'] * (res['E_mean'] + baseline) - res['beta_C'] * L1
        
        # Error Propagation: J_std approx I_std (assuming E variance is second order or tracked elsewhere)
        res['J_std'] = res['I_lower_std'] 
        
        return res

    results = []
    
    print("Evaluating Stage 1...")
    for i, row in df_s1.iterrows():
        r = eval_row(row, base_cfg, 'Stage 1')
        if r: results.append(r)
        
    print("Evaluating Stage 2...")
    for i, row in df_s2.iterrows():
        r = eval_row(row, base_cfg, 'Stage 2')
        if r: results.append(r)
        
    df_res = pd.DataFrame(results)
    
    # Write Tables
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(tables_dir / "pareto_stats.csv", index=False)
    
    # Compute Union Front
    # Sort by Energy, keep point if I > max(I of all pts with lower E) ? 
    # Or strict definition: A dominates B if E_A <= E_B and I_A >= I_B (with at least one inequality strict)
    # We want the "frontier".
    
    # 1. Sort by Mean Energy
    sorted_res = df_res.sort_values('E_mean')
    union_front = []
    current_max_I = -np.inf
    
    for idx, row in sorted_res.iterrows():
        if row['I_lower_mean'] > current_max_I:
            union_front.append(row)
            current_max_I = row['I_lower_mean']
            
    df_union = pd.DataFrame(union_front)
    df_union.to_csv(tables_dir / "pareto_union_front.csv", index=False)
    print(f"Union Front Computed: {len(df_union)} points.")
    
    # Plotting
    plot_pareto_bars(df_res, df_union, run_dir / "figures")

def plot_pareto_bars(df, df_union, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Stage 1
    s1 = df[df['stage'] == 'Stage 1'].sort_values('E_mean')
    plt.errorbar(s1['E_mean'], s1['I_lower_mean'], 
                 yerr=s1['I_lower_std'], xerr=None, fmt='o-', label='Stage 1 (Baseline)', capsize=3, alpha=0.6, color='gray')
                 
    # Stage 2
    s2 = df[df['stage'] == 'Stage 2']
    for bc in sorted(s2['beta_C'].unique()):
        sub = s2[s2['beta_C'] == bc].sort_values('E_mean')
        plt.errorbar(sub['E_mean'], sub['I_lower_mean'], 
                     yerr=sub['I_lower_std'], fmt='s', label=f'Stage 2 (BetaC={bc})', capsize=3, alpha=0.8)
    
    # Union Front Overlay
    plt.plot(df_union['E_mean'], df_union['I_lower_mean'], 'k--', linewidth=2, label='Union Front', alpha=0.5)

    plt.xlabel('Energy (Hz)')
    plt.ylabel('Information (bits/s)')
    plt.title('Stage 1 vs Stage 2: Cost-Efficiency Extension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / "pareto_extension_errorbars.pdf")
    
    # Zoom View
    plt.xlim(0, 100) # Assuming most interesting dynamics < 100Hz
    plt.ylim(bottom=0)
    plt.title('Stage 1 vs Stage 2: Low Energy Regime')
    plt.savefig(fig_dir / "pareto_extension_zoom.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    args = parser.parse_args()
    run_stats_verification(args)

if __name__ == "__main__":
    main()
