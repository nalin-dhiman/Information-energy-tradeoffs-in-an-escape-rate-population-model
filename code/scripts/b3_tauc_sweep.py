import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.paths import create_run_dir
from src.io.runlog import log_run
from src.io.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--alt_stim', action='store_true', help='Use alternate stimulus (Robustness check)')
    args = parser.parse_args()
    
    # Load config to get N, T etc for metadata
    base_cfg = load_config(args.config)
    
    # Phase 14 Audit Runs
    suffix_str = "_altstim" if args.alt_stim else ""
    run_dir = create_run_dir(major_version=3) # Auto-increments
    print(f"Starting Audit Run ({run_dir.name}{suffix_str}): {run_dir}")
    
    tau_c_vals = [0.02, 0.05, 0.1, 0.2]
    beta_e_vals = [0.0, 1.0, 10.0] 
    
    # Log Aggregate Run
    stim_type = 'gauss_switching' if args.alt_stim else 'gauss_bandlimited' 
    
    extra_meta = {
        'dt': base_cfg['simulation'].get('dt', 0.001),
        'dt_eff': 'variable',
        'N': base_cfg['simulation'].get('N', 200),
        'trials': 20,
        'trial_T': base_cfg['simulation'].get('T', 5.0),
        'tau_list': tau_c_vals,
        'seed_list': [0, 1, 2],
        'beta_E_list': beta_e_vals,
        'beta_C_list': [0.0, 0.01],
        'stimulus_type': stim_type,
        'notes': 'Pipeline Integrity Audit Run'
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    aggregated_results = []
    
    for tau in tau_c_vals:
        print(f"\n=== Processing Tau_c = {tau} ===")
        sub_run_dir = run_dir / f"tau_{tau}"
        
        # Check if result already exists to skip
        res_file = sub_run_dir / "tables" / "opt_best.csv"
        if res_file.exists():
             print(f"Skipping {tau} (already done)")
        else:
            cmd = [
                "python3", "scripts/b1_grid_refine_theta.py",
                "--config", args.config,
                "--beta_e", *[str(b) for b in beta_e_vals],
                "--tau_c", str(tau),
                "--run_dir", str(sub_run_dir)
            ]
            
            print(f"Running optimization: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Error optimizing for tau={tau}")
                continue
            
        # Collect Results
        if res_file.exists():
            df = pd.read_csv(res_file)
            df['tau_c'] = tau
            aggregated_results.append(df)
            
    if not aggregated_results:
        print("No results collected.")
        return
        
    full_df = pd.concat(aggregated_results, ignore_index=True)
    full_df.to_csv(run_dir / "tables" / f"tauc_sweep{suffix_str}.csv", index=False)
    
    # Plotting
    plot_dir = run_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Rate vs Tau_c 
    plt.figure()
    for be in beta_e_vals:
        sub = full_df[np.isclose(full_df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            # Updated: E_mean_Hz
            rate = sub['E_mean_Hz'] + 5.0 
            plt.plot(sub['tau_c'], rate, 'o-', label=f'Beta_E={be}')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Total Energy (Rate + 5Hz)')
    plt.title(f'Rate vs Tau_c {suffix_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"rate_vs_tauc{suffix_str}.pdf")
    
    # 2. Bits/Joule vs Tau_c
    plt.figure()
    for be in beta_e_vals:
        sub = full_df[np.isclose(full_df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            # Updated: I_lower_mean_bits_per_s
            I = sub['I_lower_mean_bits_per_s']
            E = sub['E_mean_Hz'] + 5.0
            bpj = I / E
            plt.plot(sub['tau_c'], bpj, 's-', label=f'Beta_E={be}')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Efficiency (Bits/Joule)')
    plt.title(f'Efficiency vs Tau_c {suffix_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"bpj_vs_tauc{suffix_str}.pdf")
    
    # 3. Theta vs Tau_c
    plt.figure()
    for be in beta_e_vals:
        sub = full_df[np.isclose(full_df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            plt.plot(sub['tau_c'], sub['theta0'], '^-', label=f'Theta0 (be={be})')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Parameter Value')
    plt.title(f'Adaptation vs Tau_c {suffix_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"theta_vs_tauc{suffix_str}.pdf")

if __name__ == "__main__":
    main()
