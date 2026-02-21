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
    
    if mode == 'lag':
        run_tag = 'v4_a'
        param_vals = [5, 10, 20]
        param_name = 'lag_taps'
        override_key = '--lag_taps'
    elif mode == 'bin':
        run_tag = 'v4_b' 
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
    
    extra_meta = {
        'control_mode': mode,
        'param_values': param_vals,
        'tau_c': 0.02, 
        'beta_E': [0, 0.3, 1, 3, 10],
        'notes': f'Artifact Control {mode}',
        'dt': base_cfg['simulation'].get('dt', 0.001),
        'dt_eff': 'variable (sweep)',
        'N': base_cfg['simulation'].get('N', 200),
        'trials': param_vals if mode == 'trials' else 20, 
        'trial_T': base_cfg['simulation'].get('T', 5.0),
        'tau_list': [0.02],
        'seed_list': [0, 1, 2],
        'beta_E_list': [0, 0.3, 1, 3, 10],
        'beta_C_list': [0.0, 0.01] 
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    aggregated_results = []
    
    for val in param_vals:
        print(f"\n=== Testing {param_name} = {val} ===")
        sub_dir = run_dir / f"{param_name}_{val}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        temp_cfg = base_cfg.copy()
        
        if mode == 'lag':
            temp_cfg.setdefault('decode', {})['lag_taps'] = int(val)
        elif mode == 'bin':
            
            temp_cfg.setdefault('decode', {})['bin_dt'] = float(val)
            temp_cfg.setdefault('decode', {})['feature_mode'] = 'spikecount_lags'
        elif mode == 'trials':
            val_int = int(val)
            temp_cfg['simulation']['trials'] = val_int 
            pass 
            
        import yaml
        temp_cfg_path = sub_dir / "temp_config.yaml"
        with open(temp_cfg_path, 'w') as f:
            yaml.dump(temp_cfg, f)
            
      
        cmd = [
            "python3", "scripts/b1_grid_refine_theta.py",
            "--config", str(temp_cfg_path),
            "--beta_e", "0", "0.3", "1", "3", "10",
            "--tau_c", "0.02",
            "--run_dir", str(sub_dir)
        ]
        
        if mode == 'trials':
            cmd.extend(["--trials", str(val)])
        
        subprocess.run(cmd, check=True)
        
        res_file = sub_dir / "tables" / "opt_best.csv"
        if res_file.exists():
            df = pd.read_csv(res_file)
            df[param_name] = val
            aggregated_results.append(df)
            
    full_df = pd.DataFrame() 
    if aggregated_results:
        full_df = pd.concat(aggregated_results, ignore_index=True)
        full_df.to_csv(run_dir / "tables" / f"{mode}_sensitivity.csv", index=False)
        
        plot_results(full_df, run_dir, param_name, mode)
    else:
        print("No results aggregated.")

def plot_results(df, run_dir, param_name, mode):
    plot_dir = run_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    betas = sorted(df['beta_E'].unique())
    
    plt.subplot(1, 2, 1)
    for be in betas:
        sub = df[np.isclose(df['beta_E'], be)].sort_values(param_name)
        if not sub.empty:
            plt.plot(sub[param_name], sub['E_mean_Hz'], 'o-', label=f'Beta={be}')
    plt.xlabel(param_name)
    plt.ylabel('Rate (Hz)')
    plt.title(f'{mode}: Rate Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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
