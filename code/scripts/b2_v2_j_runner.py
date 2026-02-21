import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import load_config
from src.io.paths import create_run_dir
from src.simulate import run_simulation
from src.estimators.mi_lower_decode import estimate_mi_lower_decode



def convolve_spikes(spikes, tau, dt):
    pop_spikes = np.sum(spikes, axis=1)
    N = spikes.shape[1]
    alpha = dt / tau 
    inst_rate = pop_spikes / (N * dt)
   
    from scipy.signal import lfilter
    b = [alpha]
    a = [1, -(1 - alpha)]
    A_smooth = lfilter(b, a, inst_rate)
    return A_smooth

def evaluate_calibration(theta, cfg, n_trials=20, seeds=[0,1,2], tau=0.02):
    
    results_seeds = []
    
    for seed in seeds:
        S_trials = []
        A_trials = []
        E_trials = []
        
        for tr in range(n_trials):
            trial_seed = seed * 1000 + tr 
            current_cfg = json.loads(json.dumps(cfg))
            
            current_cfg['hazard']['theta0'] = float(theta[0])
            current_cfg['hazard']['thetaV'] = float(theta[1])
            current_cfg['hazard']['thetaa'] = float(theta[2])
            current_cfg['simulation']['seed'] = trial_seed
            current_cfg['stimulus']['seed'] = trial_seed
            
            try:
                data = run_simulation(current_cfg)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Simulation Failed: {e}")
                return None
                
            S = data['S']
            spikes = data['spikes']
            dt = data['dt']
            
            A = convolve_spikes(spikes, tau, dt)
            mean_rate = np.mean(A)
            
            S_trials.append(S)
            A_trials.append(A)
            E_trials.append(mean_rate)
            
        lcfg = cfg['estimators']['lower'].copy()
        lcfg['split'] = 'trial'
        lcfg['n_trials'] = n_trials
        lcfg['seed'] = seed
        lcfg['feature_mode'] = 'rate_lags'
        
        try:
            res_l = estimate_mi_lower_decode(S_trials, A_trials, dt, lcfg)
            I_lower = res_l.get('I_lower_bits_per_s', 0.0)
        except:
            I_lower = 0.0
            
        E_mean = np.mean(E_trials)
        results_seeds.append({
            'seed': seed,
            'I_lower': I_lower, 
            'E_rate': E_mean,
            'I_over_E': I_lower / E_mean if E_mean > 0 else 0.0
        })
        
    return results_seeds

def optimize_expanded(run_dir, base_cfg):
   
    import multiprocessing
    from scripts.b1_grid_refine_theta import evaluate_point # reuse if possible?
   
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--prev_run', type=str, default='runs/v2_i')
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)

    if 'estimators' in base_cfg:
        for k,v in base_cfg['estimators'].items():
            if isinstance(v, str):
                p = Path(v)
                if not p.exists(): p = Path(__file__).parent.parent / str(p)
                if p.exists(): 
                     print(f"Loaded estimator {k} from {p}")
                     base_cfg['estimators'][k] = load_config(str(p))
                else:
                     print(f"Warning: Could not find estimator {k} at {p}")
            print(f"Estimator {k} Type: {type(base_cfg['estimators'][k])}")
    if isinstance(base_cfg.get('stimulus'), str):
        p = Path(base_cfg.get('stimulus'))
        if not p.exists(): p = Path(__file__).parent.parent / str(p)
        if p.exists(): 
            print(f"Loaded stimulus config from {p}")
            base_cfg['stimulus'] = load_config(str(p))
        else:
            print(f"Warning: Could not find stimulus config at {p}")

    print(f"Stimulus Config Type: {type(base_cfg.get('stimulus'))}")
    if isinstance(base_cfg.get('stimulus'), str):
        print("Error: Stimulus config is still a string!")
        return

    base_cfg['simulation']['N'] = 200
    base_cfg['simulation']['T'] = 5.0
    base_cfg.setdefault('decode', {})['trial_T'] = 5.0
    
    run_dir = create_run_dir(major_version=2) # v2_j
    print(f"Starting Phase 11 (v2_j): {run_dir}")
    
    print("Step 1: Calibration (re-evaluating v2_i optima)...")
    prev_path = Path(args.prev_run) / "tables" / "opt_best.csv"
    if not prev_path.exists():
        print(f"Error: {prev_path} not found.")
        return
        
    df_prev = pd.read_csv(prev_path)
    
    solutions = df_prev[['theta0', 'thetaV', 'thetaa']].drop_duplicates()
    
    calibration_data = []
    
    for i, row in solutions.iterrows():
        theta = (row['theta0'], row['thetaV'], row['thetaa'])
        print(f"  Calibrating theta={theta}...")
        
      
        
        calib_cfg = base_cfg.copy()
        calib_cfg['simulation']['N'] = 500
        
        seeds_res = evaluate_calibration(theta, calib_cfg, n_trials=20)
        
        if seeds_res:
            res_df = pd.DataFrame(seeds_res)
            
            mean_I = res_df['I_lower'].mean()
            std_I = res_df['I_lower'].std()
            mean_E = res_df['E_rate'].mean()
            std_E = res_df['E_rate'].std()
            mean_IE = res_df['I_over_E'].mean()
            std_IE = res_df['I_over_E'].std()
            
            calibration_data.append({
                'theta0': theta[0], 'thetaV': theta[1], 'thetaa': theta[2],
                'mean_I': mean_I, 'std_I': std_I,
                'mean_E': mean_E, 'std_E': std_E,
                'mean_IE': mean_IE, 'std_IE': std_IE
            })
            
    calib_df = pd.DataFrame(calibration_data)
    calib_df.to_csv(run_dir / "tables" / "energy_calibration.csv", index=False)
    
    plt.figure()
    plt.errorbar(calib_df['mean_E'], calib_df['mean_I'], 
                 xerr=calib_df['std_E'], yerr=calib_df['std_I'], 
                 fmt='o', capsize=5)
    plt.xlabel('Energy (Rate Hz)')
    plt.ylabel('Info Lower (bits/s)')
    plt.title('Energy Calibration (v2_i optima)')
    plt.savefig(run_dir / "figures" / "I_vs_E_calibration.pdf")
    
    print("Step 2: Expanded Optim (BetaE {0, 0.3, 1, 3, 10})...")
   
    
    import subprocess
    cmd = [
        "python3", "scripts/b1_grid_refine_theta.py",
        "--config", args.config,
        "--beta_e", "0.0", "0.3", "1.0", "3.0", "10.0",
        "--run_dir", str(run_dir) 
    ]
   
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
