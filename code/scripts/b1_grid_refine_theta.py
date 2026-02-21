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

from src.io.config import load_config
from src.io.paths import create_run_dir
from src.io.runlog import log_run
from src.simulate import run_simulation
from src.estimators.mi_lower_decode import estimate_mi_lower_decode
from src.estimators.mi_upper import estimate_mi_upper_gaussian

def convolve_spikes(spikes, tau, dt):
    pop_spikes = np.sum(spikes, axis=1) 
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
        lcfg['feature_mode'] = cfg.get('estimators', {}).get('lower', {}).get('feature_mode', 'rate_lags')
        if 'feature_mode' in cfg.get('decode', {}):
             lcfg['feature_mode'] = cfg['decode']['feature_mode']
        if cutoff: lcfg['bandwidth'] = cutoff
        
        if 'ridge_alpha' in cfg.get('decode', {}):
             lcfg.setdefault('parameters', {})['alpha'] = float(cfg['decode']['ridge_alpha'])
        if 'lag_taps' in cfg.get('decode', {}):
             lcfg['lags'] = int(cfg['decode']['lag_taps'])
        if 'bin_dt' in cfg.get('decode', {}):
             lcfg['bin_dt'] = float(cfg['decode']['bin_dt'])

        try:
            res_l = estimate_mi_lower_decode(S_trials, A_trials, dt, lcfg)
            I_lower = res_l.get('I_lower_bits_per_s', 0.0)
            
            ucfg = cfg['estimators']['upper'].copy()
            ucfg['bandwidth'] = cutoff
            
            S_concat = np.concatenate(S_trials)
            A_concat = np.concatenate(A_trials)
            res_u = estimate_mi_upper_gaussian(S_concat, A_concat, dt, ucfg)
            I_upper = res_u.get('I_upper_surrogate_bits_per_s', 0.0)
            
        except Exception as e:
            I_lower = 0.0
            I_upper = 0.0
            res_l = {}
            
        E_mean = np.mean(E_trials)
        results_seeds.append({
            'I_lower': I_lower, 
            'I_upper_surrogate': I_upper, 
            'E_mean': E_mean,
            'mse_test': res_l.get('mse_test', np.nan),
            'r2_test': res_l.get('r2_test', np.nan),
            'var_S_test': res_l.get('var_S_test', np.nan),
            'dt_eff': res_l.get('dt_eff', np.nan),
            'clipped': res_l.get('clipped', False)
        })
        
    vals = pd.DataFrame(results_seeds)
    return {
        'theta': theta,
        'I_lower_mean_bits_per_s': vals['I_lower'].mean(),
        'I_lower_std_bits_per_s': vals['I_lower'].std(),
        'I_upper_surrogate_mean_bits_per_s': vals['I_upper_surrogate'].mean(),
        'E_mean_Hz': vals['E_mean'].mean(), 
        'mse_test_mean': vals['mse_test'].mean(),
        'r2_test_mean': vals['r2_test'].mean(),
        'var_S_test_mean': vals['var_S_test'].mean(),
        'dt_eff': vals['dt_eff'].mean() 
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--run_dir', type=str, default=None, help='Force specific run directory')
    parser.add_argument('--beta_e', type=float, nargs='+', default=[0.0, 0.1, 0.3], help='List of Beta E values')
    parser.add_argument('--tau_c', type=float, default=None, help='Stimulus correlation time (optional override)')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per point')
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

    if args.tau_c is not None:
         if 'stimulus' in base_cfg: 
              base_cfg['stimulus']['tau_c'] = args.tau_c
         else:
              base_cfg['stimulus'] = {'tau_c': args.tau_c}


    
    base_cfg['simulation']['N'] = 200
    base_cfg.setdefault('decode', {})
    base_cfg['decode']['trial_T'] = 5.0
    base_cfg['simulation']['T'] = 5.0 
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)
    else:
        run_dir = create_run_dir(major_version=2)
        
    print(f"Starting Stage 1 (Phase 10/11/12): {run_dir}")
    print(f"  Params: N={base_cfg['simulation']['N']}, T={base_cfg['simulation']['T']}s, Trials={args.trials}")
    print(f"  Beta E List: {args.beta_e}")
    if args.tau_c: print(f"  Tau C: {args.tau_c}")
    
    stime = base_cfg.get('stimulus', {})
    if isinstance(stime, str): stime = {}
        
    dt = base_cfg['simulation'].get('dt', 0.001)
   
    
    cutoff = stime.get('cutoff_hz', 20.0)
    dt_eff = 1.0 / (2.0 * cutoff)
    
    extra_meta = {
        'dt': dt,
        'dt_eff': dt_eff, 
        'N': base_cfg['simulation']['N'],
        'trials': args.trials,
        'trial_T': base_cfg['simulation']['T'],
        'tau_list': [args.tau_c] if args.tau_c else [],
        'seed_list': [0, 1, 2], 
        
        'beta_E_list': args.beta_e,
        'beta_C_list': [0.0, 0.01]
    }
    
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
        
    theta0_grid = np.linspace(-6, 2, 10)
    thetaV_grid = np.linspace(-5, 5, 5)
    thetaa_grid = np.linspace(-5, 5, 5)
    
    beta_E_vals = args.beta_e
    beta_C_vals = [0.0, 0.01]
    baseline_rate = 5.0
    
    grid_results = []
    
    print(f"Grid: {len(theta0_grid)}x{len(thetaV_grid)}x{len(thetaa_grid)} pts. BetaE={beta_E_vals}, BetaC={beta_C_vals}")


        param_list = list(itertools.product(theta0_grid, thetaV_grid, thetaa_grid))
    
    print(f"Parallel Grid Search with 20 processes...")
    import multiprocessing
    
    point_cache = {}
    
    with multiprocessing.Pool(processes=20) as pool:
        args_list = [(theta, base_cfg, 20) for theta in param_list]
        chunk = max(1, len(args_list) // 40) 
        
        
        
        
        
        for i, res in enumerate(pool.starmap(evaluate_point, args_list, chunksize=chunk)):
            theta = param_list[i]
            if res is not None:
                point_cache[theta] = res
                
                
                
                
                if i % 50 == 0:
                    print(f"  Grid {i}/{len(param_list)} done.")
             

    best_candidates = {} 
    
    for bE in beta_E_vals:
        for bC in beta_C_vals:
            candidates = []
            for theta, res in point_cache.items():
                E = res['E_mean_Hz'] + baseline_rate 
                C = np.sum(np.abs(theta))
                J = res['I_lower_mean_bits_per_s'] - bE * E - bC * C
                candidates.append((J, theta, res))
                
                row = res.copy()
                del row['theta']
                row.update({'theta0': theta[0], 'thetaV': theta[1], 'thetaa': theta[2],
                            'beta_E': bE, 'beta_C': bC, 'J': J, 'C_theta': C})
                grid_results.append(row)
                
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_candidates[(bE, bC)] = candidates[:1] 
            
    pd.DataFrame(grid_results).to_csv(run_dir / "tables" / "opt_grid_results.csv", index=False)
    

    print("Refining (Parallel Neighbors)...")
    refined_results = []
    

    
    
    
    with multiprocessing.Pool(processes=20) as pool:
    
        for (bE, bC), top_list in best_candidates.items():
            if not top_list: continue
            start_node = top_list[0]
            curr_theta = np.array(start_node[1])
            curr_J = start_node[0]
            curr_res = start_node[2]
            
            print(f"  Obj(bE={bE}, bC={bC}): Start J={curr_J:.2f}")
            
            step_sz = np.array([0.5, 0.5, 0.5])
            
            for iter in range(3):
                neighbors = []
                for d in range(3):
                     for sgn in [-1, 1]:
                          t_new = curr_theta.copy()
                          t_new[d] += sgn * step_sz[d]
                          neighbors.append(tuple(t_new))
                
                to_eval = [t for t in neighbors if t not in point_cache]
                
                if to_eval:
                    args = [(t, base_cfg, 20) for t in to_eval]
                    new_res = pool.starmap(evaluate_point, args)
                    for t, r in zip(to_eval, new_res):
                        if r: point_cache[t] = r
                
                improved = False
                for t in neighbors:
                    if t in point_cache:
                        res = point_cache[t]
                        C = np.sum(np.abs(t))
                        J = res['I_lower_mean_bits_per_s'] - bE * (res['E_mean_Hz'] + baseline_rate) - bC * C
                        if J > curr_J:
                            curr_J = J
                            curr_theta = np.array(t)
                            curr_res = res
                            improved = True
                            
                if not improved: step_sz *= 0.5
                
            row = curr_res.copy()
            del row['theta']
            row.update({
                'theta0': curr_theta[0], 'thetaV': curr_theta[1], 'thetaa': curr_theta[2],
                'beta_E': bE, 'beta_C': bC,
                'J': curr_J, 'C_theta': np.sum(np.abs(curr_theta))
            })
            refined_results.append(row)
            print(f"    End J={curr_J:.2f}")

    pd.DataFrame(refined_results).to_csv(run_dir / "tables" / "opt_best.csv", index=False)
    
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    df_grid = pd.DataFrame(grid_results)
    df_opt = pd.DataFrame(refined_results)
    
    plt.figure()
    plt.scatter(df_grid['E_mean_Hz'], df_grid['I_lower_mean_bits_per_s'], c='lightgray', s=10, label='Grid')
    sc = plt.scatter(df_opt['E_mean_Hz'], df_opt['I_lower_mean_bits_per_s'], c=df_opt['beta_C'], cmap='viridis', s=50, label='Refined')
    plt.colorbar(sc, label='Beta_C')
    plt.xlabel('Energy (Rate Hz)')
    plt.ylabel('Info Lower (bits/s)')
    plt.title('Pareto IE')
    plt.savefig(fig_dir / "pareto_IE.pdf")
    
    df_opt.to_csv(run_dir / "tables" / "opt_upper_diagnostic.csv", index=False)

if __name__ == "__main__":
    main()
