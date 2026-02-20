import sys
import os
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.paths import create_run_dir
from src.io.config import load_config
from src.io.runlog import log_run

def run_control_sweep(args, mode='lag'):
    # Mode: 'lag' or 'bin'
    
    # 1. Setup Run Dir
    if mode == 'lag':
        run_tag = 'v4_a'
        param_vals = [5, 10, 20]
        param_name = 'lag_taps'
        override_key = '--lag_taps' # We need to support this in b1
        # Actually b1 takes config. We can inject into config via temporary file or CLI override if supported.
        # b1 supports --tau_c. Does it support arbitrary config overrides? No.
        # We should modify b1 to accept json overrides or just use temp configs.
        # Or simpler: modify b1 to accept --overrides "key=val"
        # Or just write temp configs.
    elif mode == 'bin':
        run_tag = 'v4_b' # Auto increment will handle
        param_vals = [0.01, 0.02, 0.05]
        param_name = 'bin_dt'
    elif mode == 'trials':
        run_tag = 'v4_f'
        param_vals = [10, 20, 40]
        param_name = 'trials'
    
    run_dir = create_run_dir(major_version=4, root=None) 
    
    print(f"Starting {mode} sensitivity sweep in {run_dir}")
    
    base_cfg_path = args.config
    base_cfg = load_config(base_cfg_path)
    
    # Log Run
    extra_meta = {
        'control_mode': mode,
        'param_values': param_vals,
        'tau_c': 0.02, 
        'beta_E': [0, 0.3, 1, 3, 10],
        'notes': f'Artifact Control {mode}',
        # Strict Metadata for Runlog
        'dt': base_cfg['simulation'].get('dt', 0.001),
        'dt_eff': 'variable (sweep)',
        'N': base_cfg['simulation'].get('N', 200),
        'trials': param_vals if mode == 'trials' else 20, 
        'trial_T': base_cfg['simulation'].get('T', 5.0),
        'tau_list': [0.02],
        'seed_list': [0, 1, 2],
        'beta_E_list': [0, 0.3, 1, 3, 10], # From b4 logic
        'beta_C_list': [0.0, 0.01] # From b1 logic
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    aggregated_results = []
    
    for val in param_vals:
        print(f"\n=== Testing {param_name} = {val} ===")
        sub_dir = run_dir / f"{param_name}_{val}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp config with override
        temp_cfg = base_cfg.copy()
        
        if mode == 'lag':
            temp_cfg.setdefault('decode', {})['lag_taps'] = int(val)
        elif mode == 'bin':
            # Use 'decode' section to override feature_mode and bin_dt
            # b1 script is updated to look for these in 'decode' section
            temp_cfg.setdefault('decode', {})['bin_dt'] = float(val)
            temp_cfg.setdefault('decode', {})['feature_mode'] = 'spikecount_lags'
        elif mode == 'trials':
            # Override trials in simulation and decode
            val_int = int(val)
            temp_cfg['simulation']['trials'] = val_int # Not standard key? b1 checks this?
            # b1 uses `n_trials=20` argument locally in `main`, but passes it to `evaluate_point`.
            # b1 main hardcodes `trials=20` in `extra_meta`.
            # b1 uses `args.trials`? No. 
            # b1 code: `grid_results = ... (..., 20)`. It hardcodes 20.
            # I NEED TO UPDATE b1 TO ACCEPT TRIALS ARGUMENT.
            pass 
            
        # Write temp config
        import yaml
        temp_cfg_path = sub_dir / "temp_config.yaml"
        with open(temp_cfg_path, 'w') as f:
            yaml.dump(temp_cfg, f)
            
        # Run b1 with this config
        # We need to enforce tau_c=0.02
        cmd = [
            "python3", "scripts/b1_grid_refine_theta.py",
            "--config", str(temp_cfg_path),
            "--beta_e", "0", "0.3", "1", "3", "10",
            "--tau_c", "0.02",
            "--run_dir", str(sub_dir)
        ]
        
        # Pass trials arg if mode is trials
        if mode == 'trials':
            cmd.extend(["--trials", str(val)])
        
        subprocess.run(cmd, check=True)
        
        # Harvest
        res_file = sub_dir / "tables" / "opt_best.csv"
        if res_file.exists():
            df = pd.read_csv(res_file)
            df[param_name] = val
            aggregated_results.append(df)
            
    # Save Aggregate
    full_df = pd.DataFrame() # Fallback
    if aggregated_results:
        full_df = pd.concat(aggregated_results, ignore_index=True)
        full_df.to_csv(run_dir / "tables" / f"{mode}_sensitivity.csv", index=False)
        
        # Plotting - Rate Stability
        plot_results(full_df, run_dir, param_name, mode)
    else:
        print("No results aggregated.")

def plot_results(df, run_dir, param_name, mode):
    plot_dir = run_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    betas = sorted(df['beta_E'].unique())
    
    # 1. Rate shift
    plt.subplot(1, 2, 1)
    for be in betas:
        # Use close comparison for float beta_E
        sub = df[np.isclose(df['beta_E'], be)].sort_values(param_name)
        if not sub.empty:
            plt.plot(sub[param_name], sub['E_mean_Hz'], 'o-', label=f'Beta={be}')
    plt.xlabel(param_name)
    plt.ylabel('Rate (Hz)')
    plt.title(f'{mode}: Rate Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Logic Gate Check
    gate_passed = True
    deviations = []
    
    for be in betas:
        sub = df[np.isclose(df['beta_E'], be)]
        if len(sub) > 1:
            r = sub['E_mean_Hz'].values
            if np.mean(r) < 1e-6:
                pct_change = 0.0
            else:
                pct_change = (np.max(r) - np.min(r)) / np.mean(r)
            deviations.append(pct_change)
            if pct_change > 0.25:
                gate_passed = False
                
    max_dev = max(deviations) if deviations else 0.0
    
    plt.suptitle(f"Gate Passed: {gate_passed} (Max Dev: {max_dev:.1%})")
    plt.savefig(plot_dir / f"{mode}_sensitivity_rate.pdf")
    print(f"Gate {mode}: {gate_passed} (Max deviation {max_dev:.1%})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--mode', choices=['lag', 'bin', 'trials'], required=True)
    args = parser.parse_args()
    
    run_control_sweep(args, args.mode)

if __name__ == "__main__":
    main()
