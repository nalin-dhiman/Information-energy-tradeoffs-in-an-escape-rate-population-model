import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import itertools
from pathlib import Path
from scipy.signal import lfilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import parse_cli_overrides, save_config, load_config
from src.io.paths import create_run_dir
from src.io.runlog import get_run_metadata
from src.simulate import run_simulation
from src.estimators.mi_lower_decode import estimate_mi_lower_decode

def convolve_spikes(spikes, tau, dt):
    
    pop_spikes = np.sum(spikes, axis=1) # (T,)
    N = spikes.shape[1]
    alpha = dt / tau 
    inst_rate = pop_spikes / (N * dt)
    b = [alpha]
    a = [1, -(1 - alpha)]
    A_smooth = lfilter(b, a, inst_rate)
    return A_smooth

def evaluate_point(theta, cfg, n_trials=20, seeds=[0,1,2], tau=0.02, cutoff=50.0):
    
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
                print(f"Simulation Failed at {theta}: {e}")
                return None
                
            S = data['S']
            spikes = data['spikes']
            dt = data['dt']
            
            A = convolve_spikes(spikes, tau, dt)
            mean_rate = np.mean(A)
            
            S_trials.append(S)
            A_trials.append(A)
            E_trials.append(mean_rate) 
            
        lcfg = cfg['estimators']['lower']
        lcfg['split'] = 'trial'
        lcfg['n_trials'] = n_trials
        lcfg['seed'] = seed
        if cutoff: lcfg['bandwidth'] = cutoff
        
        d_params = cfg.get('decode', {})
        if 'ridge_alpha' in d_params: lcfg.setdefault('parameters', {})['alpha'] = float(d_params['ridge_alpha'])
        if 'lag_taps' in d_params: lcfg['lags'] = int(d_params['lag_taps'])

        try:
            res_l = estimate_mi_lower_decode(S_trials, A_trials, dt, lcfg)
            I_lower = res_l.get('I_lower_bits_per_s', 0.0)
        except Exception as e:
            print(f"Estimation Failed at {theta}: {e}")
            I_lower = 0.0
            
        E_mean = np.mean(E_trials)
        results_seeds.append({
            'I_lower': I_lower, 'E_raw': E_mean
        })
        
    vals = pd.DataFrame(results_seeds)
    return {
        'theta': theta,
        'I_mean': vals['I_lower'].mean(),
        'I_std': vals['I_lower'].std(),
        'E_mean': vals['E_raw'].mean(),
        'E_std': vals['E_raw'].std()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)
    
    if isinstance(base_cfg.get('stimulus'), str):
        p = Path(base_cfg.get('stimulus'))
        if not p.exists(): p = Path(__file__).parent.parent / str(p)
        if p.exists(): base_cfg['stimulus'] = load_config(str(p))

    if 'estimators' in base_cfg:
         for k,v in base_cfg['estimators'].items():
             if isinstance(v, str):
                 p = Path(v)
                 if not p.exists(): p = Path(__file__).parent.parent / str(p)
                 if p.exists(): base_cfg['estimators'][k] = load_config(str(p))
                 
    run_dir = create_run_dir(major_version=2)
    print(f"Starting Stage 1 Optimization (Phase 9): {run_dir}")
    
    run_json_path = run_dir / "run.json"
    full_log = {'config': base_cfg, 'args': vars(args)}
    with open(run_json_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)
        
  
    theta0_grid = np.linspace(-6, 2, 10)
    thetaV_grid = np.linspace(-5, 5, 5)
    thetaa_grid = np.linspace(-5, 5, 5)
    
    beta_E_vals = [0.0, 0.1, 0.3]
    beta_C_vals = [0.0, 0.01]
    baseline_rate = base_cfg['objective'].get('baseline_rate', 5.0)
    
    grid_results = []
    
    print(f"Running Grid Search: {len(theta0_grid)}x{len(thetaV_grid)}x{len(thetaa_grid)} = {len(theta0_grid)*len(thetaV_grid)*len(thetaa_grid)} points.")
    
    best_candidates = {} 
    
    for t0, tV, ta in itertools.product(theta0_grid, thetaV_grid, thetaa_grid):
        theta = (t0, tV, ta)
        
        
        base_cfg['simulation']['N'] = 500 
        
        res = evaluate_point(theta, base_cfg, n_trials=20)
        
        if res is None: continue
        
        for bE in beta_E_vals:
            for bC in beta_C_vals:
                E_total = bE * res['E_mean'] + baseline_rate
                C_theta = np.sum(np.abs(theta))
                
                J = res['I_mean'] - E_total - bC * C_theta
                
                row = res.copy()
                del row['theta']
                row.update({
                    'theta0': t0, 'thetaV': tV, 'thetaa': ta,
                    'beta_E': bE, 'beta_C': bC,
                    'J': J, 'E_total': E_total, 'C_theta': C_theta
                })
                grid_results.append(row)
                
                key = (bE, bC)
                if key not in best_candidates or J > best_candidates[key]['J']:
                    best_candidates[key] = row
                    
    df_grid = pd.DataFrame(grid_results)
    df_grid.to_csv(run_dir / "tables" / "opt_grid_results.csv", index=False)
    
    print("Starting Refinement Step...")
  
    
    refined_results = []
    
    for (bE, bC), best_row in best_candidates.items():
        print(f"Refining for bE={bE}, bC={bC} (Current Best J={best_row['J']:.3f})")
        
        current_theta = np.array([best_row['theta0'], best_row['thetaV'], best_row['thetaa']])
        step_sizes = np.array([0.5, 0.5, 0.5])
        
        for i_iter in range(3):
            improved = False
            for dim in range(3):
                for direction in [-1, 1]:
                    test_theta = current_theta.copy()
                    test_theta[dim] += direction * step_sizes[dim]
                    
                    res = evaluate_point(test_theta, base_cfg, n_trials=20)
                    if res is None: continue
                    
                    E_total = bE * res['E_mean'] + baseline_rate
                    C_theta = np.sum(np.abs(test_theta))
                    J = res['I_mean'] - E_total - bC * C_theta
                    
                    if J > best_row['J']:
                        best_row = res.copy()
                        del best_row['theta']
                        best_row.update({
                            'theta0': test_theta[0], 'thetaV': test_theta[1], 'thetaa': test_theta[2],
                            'beta_E': bE, 'beta_C': bC,
                            'J': J, 'E_total': E_total, 'C_theta': C_theta
                        })
                        current_theta = test_theta
                        improved = True
                        
            if not improved:
                step_sizes *= 0.5
                
        refined_results.append(best_row)
        
    df_refined = pd.DataFrame(refined_results)
    df_refined.to_csv(run_dir / "tables" / "opt_best.csv", index=False)
    
    fig_dir = run_dir / "figures"
    
    plt.figure()
    plt.scatter(df_grid['E_total'], df_grid['I_mean'], c='gray', alpha=0.3, s=10, label='Grid')
    
    
    plt.scatter(df_refined['E_total'], df_refined['I_mean'], c='red', s=50, label='Optimized')
    
    plt.xlabel('Energy Proxy (Hz + Offset)')
    plt.ylabel('Information (bits/s)')
    plt.title('Stage 1 Optimization')
    plt.legend()
    plt.savefig(fig_dir / "pareto_light_IE.pdf")
    
    print("Optimization Complete.")

if __name__ == "__main__":
    main()
