import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import itertools
import multiprocessing
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.config import load_config
from src.io.paths import create_run_dir
from src.io.runlog import log_run
from src.simulate import run_simulation
from src.estimators.mi_lower_decode import estimate_mi_lower_decode
from src.estimators.mi_upper import estimate_mi_upper_gaussian
from scipy.signal import lfilter

# Evaluation Function (Similar to b1 but handles expanded theta)
def evaluate_point_stage2(theta_dict, cfg, n_trials=20, seeds=[0,1,2], tau=0.02, cutoff=50.0):
    results_seeds = []
    
    # Extract params
    theta0 = theta_dict.get('theta0', 0.0)
    thetaV = theta_dict.get('thetaV', 0.0)
    thetaa = theta_dict.get('thetaa', 0.0)
    thetaVV = theta_dict.get('thetaVV', 0.0)
    thetaaa = theta_dict.get('thetaaa', 0.0)
    thetaVa = theta_dict.get('thetaVa', 0.0)
    
    for seed in seeds:
        S_trials = []
        A_trials = []
        E_trials = []
        
        for tr in range(n_trials):
            trial_seed = seed * 1000 + tr 
            current_cfg = json.loads(json.dumps(cfg))
            
            # Set params
            h = current_cfg.setdefault('hazard', {})
            h['theta0'] = float(theta0)
            h['thetaV'] = float(thetaV)
            h['thetaa'] = float(thetaa)
            h['thetaVV'] = float(thetaVV)
            h['thetaaa'] = float(thetaaa)
            h['thetaVa'] = float(thetaVa)
            
            current_cfg['simulation']['seed'] = trial_seed
            current_cfg['stimulus']['seed'] = trial_seed
            
            try:
                data = run_simulation(current_cfg)
            except Exception as e:
                return None
                
            S = data['S']
            spikes = data['spikes']
            dt = data['dt']
            
            # Simple Rate Feature for Decoder (Robust)
            # Convolve spikes
            pop_spikes = np.sum(spikes, axis=1)
            N = spikes.shape[1]
            alpha = dt / tau 
            inst_rate = pop_spikes / (N * dt)
            b = [alpha]
            a = [1, -(1 - alpha)]
            A_smooth = lfilter(b, a, inst_rate)
            mean_rate = np.mean(A_smooth)
            
            S_trials.append(S)
            A_trials.append(A_smooth)
            E_trials.append(mean_rate)
            
        # Estimators
        lcfg = cfg['estimators']['lower'].copy()
        lcfg['split'] = 'trial'
        lcfg['n_trials'] = n_trials
        lcfg['seed'] = seed
        # Use config overrides if present
        if 'decode' in cfg:
             if 'feature_mode' in cfg['decode']: lcfg['feature_mode'] = cfg['decode']['feature_mode']
             if 'lag_taps' in cfg['decode']: lcfg['lags'] = int(cfg['decode']['lag_taps'])
        if cutoff: lcfg['bandwidth'] = cutoff
        
        try:
            res_l = estimate_mi_lower_decode(S_trials, A_trials, dt, lcfg)
            I_lower = res_l.get('I_lower_bits_per_s', 0.0)
            
            # Diagnostic Upper
            ucfg = cfg['estimators']['upper'].copy()
            ucfg['bandwidth'] = cutoff
            S_concat = np.concatenate(S_trials)
            A_concat = np.concatenate(A_trials)
            res_u = estimate_mi_upper_gaussian(S_concat, A_concat, dt, ucfg)
            I_upper = res_u.get('I_upper_surrogate_bits_per_s', 0.0)
            
        except Exception:
            I_lower = 0.0
            I_upper = 0.0
            
        E_mean = np.mean(E_trials)
        results_seeds.append({
            'I_lower': I_lower,
            'I_upper_surrogate': I_upper,
            'E_mean': E_mean
        })
        
    vals = pd.DataFrame(results_seeds)
    return {
        'params': theta_dict,
        'I_lower_mean': vals['I_lower'].mean(),
        'I_lower_std': vals['I_lower'].std(),
        'I_upper_mean': vals['I_upper_surrogate'].mean(),
        'I_upper_std': vals['I_upper_surrogate'].std(),
        'E_mean': vals['E_mean'].mean(),
        'E_std': vals['E_mean'].std()
    }

def optimize_stage2(args, base_cfg):
    # Setup
    # base_cfg passed in
    run_dir = create_run_dir(major_version=5)
    print(f"Starting Stage 2 Optimization: {run_dir}")
    if args.tau_c: print(f"  Tau C: {args.tau_c}")
    
    # Metadata
    extra_meta = {
        'algorithm': 'coordinate_descent_random_restart',
        'beta_E_list': args.beta_e,
        'beta_C_list': args.beta_c,
        'restarts': args.restarts,
        'tau_c': args.tau_c, 
        'notes': 'High-Dim Optimization (Quadratic Hazard)',
        # Strict Metadata
        'dt': base_cfg['simulation'].get('dt', 0.001),
        'dt_eff': 0.01, # Approx for tau_c=0.02
        'N': base_cfg['simulation'].get('N', 200),
        'trials': 20, # Hardcoded in evaluate function default
        'trial_T': base_cfg['simulation'].get('T', 5.0),
        'tau_list': [args.tau_c],
        'seed_list': [0, 1, 2]
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    # Constraints / Search Space
    # Param keys: theta0, thetaV, thetaa, thetaVV, thetaaa, thetaVa
    param_keys = ['theta0', 'thetaV', 'thetaa', 'thetaVV', 'thetaaa', 'thetaVa']
    
    # Seeds (Start from Stage 1 optima)
    start_seeds = []
    
    # Try to load Stage 1 results (Prioritize recent successful runs)
    potential_sources = [
        "runs/v2_a/tables/opt_best.csv",
        "runs/v4_b/tables/lag_sensitivity.csv", # Use lag=5 or 10 points
        "runs/v3_d/tau_0.02/tables/opt_grid_results.csv"
    ]
    
    df_s1 = pd.DataFrame()
    for p in potential_sources:
        pp = Path(p)
        if pp.exists():
            print(f"Loading Stage 1 seeds from {pp}")
            df_s1 = pd.read_csv(pp)
            break # Use first found
            
    # Optimization Loop
    final_results = []
    
    import multiprocessing
    pool = multiprocessing.Pool(processes=20) 
    
    baseline_rate = 5.0
    
    for be in args.beta_e:
        for bc in args.beta_c:
            best_J = -np.inf
            best_res = None
            
            # Construct seeds for this beta_E
            current_seeds = []
            
            # 1. Stage 1 Seed (matched beta_E)
            if not df_s1.empty:
                # Find closest beta_E
                # Or exact match
                match = df_s1[np.isclose(df_s1['beta_E'], be)]
                if not match.empty:
                    row = match.iloc[0]
                    seed_dict = {
                        'theta0': row.get('theta0', 0.0),
                        'thetaV': row.get('thetaV', 0.0),
                        'thetaa': row.get('thetaa', 0.0),
                        'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0
                    }
                    current_seeds.append(seed_dict)
            
            # 2. Random/Heuristic Seeds
            current_seeds.append({'theta0': 0.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0})
            current_seeds.append({'theta0': -2.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0})
            
            print(f"DEBUG: Processing BetaE={be}, BetaC={bc}, Seeds={len(current_seeds)}")
            
            print(f"\n=== Optimization BetaE={be}, BetaC={bc} ===")
            
            for restart_idx, seed_params in enumerate(current_seeds[:args.restarts]):
                print(f"  DEBUG: Restart {restart_idx} with params {seed_params}")
                current_p = seed_params.copy()
                # Initial Eval
                try:
                    res = evaluate_point_stage2(current_p, base_cfg)
                except Exception as e:
                    print(f"  CRITICAL ERROR in evaluate: {e}")
                    res = None
                
                if res is None: 
                    print("  DEBUG: Initial eval failed (None)")
                    continue
                
                # J = I - bE*E - bC*|theta|
                L1 = sum(abs(v) for v in current_p.values())
                curr_J = res['I_lower_mean'] - be * (res['E_mean'] + baseline_rate) - bc * L1
                
                print(f"  Restart {restart_idx}: Start J={curr_J:.2f}")
                
                step_sz = 0.5
                patience = 3
                
                for step in range(10): # Max steps
                    # Generate neighbors (Coordinate Descent)
                    neighbors = []
                    # +/- step for each param
                    for k in param_keys:
                        for sgn in [-1, 1]:
                            n_p = current_p.copy()
                            n_p[k] += sgn * step_sz
                            neighbors.append(n_p)
                    
                    # Eval neighbors in parallel
                    # We can use the pool efficiently here
                    chunk = max(1, len(neighbors) // 20)
                    eval_args = [(p, base_cfg) for p in neighbors]
                    neighbor_results = pool.starmap(evaluate_point_stage2, eval_args)
                    
                    # Find best
                    improved = False
                    for p, r in zip(neighbors, neighbor_results):
                        if not r: continue
                        L1_n = sum(abs(v) for v in p.values())
                        J_n = r['I_lower_mean'] - be * (r['E_mean'] + baseline_rate) - bc * L1_n
                        
                        if J_n > curr_J + 1e-4:
                            curr_J = J_n
                            current_p = p
                            improved = True
                            # Greedy? Or Best improvement? Greedy is faster.
                            # Let's take best improvement in batch.
                            # Actually, we iterate all and find max.
                            
                    if improved:
                        print(f"    Step {step}: J -> {curr_J:.2f}")
                    else:
                        # Decay step
                        step_sz *= 0.5
                        if step_sz < 0.05: break
                        
                # End of restart
                if curr_J > best_J:
                    best_J = curr_J
                    best_res = current_p.copy()
                    best_res.update({
                        'I_lower': res['I_lower_mean'], # Using last verified? No, need to re-eval final point?
                        # Using stored metrics from successful neighbor
                        # Actually we didn't store the metrics of the winner in the loop perfectly.
                        # Let's re-eval optimum one last time to be sure.
                    })
            
            # Final Eval of Global Best for this Beta
            if best_res:
                final_metrics = evaluate_point_stage2(best_res, base_cfg)
                if final_metrics:
                    row = best_res.copy()
                    row['beta_E'] = be
                    row['beta_C'] = bc
                    row['J'] = best_J
                    row['I_lower_mean'] = final_metrics['I_lower_mean']
                    row['E_mean'] = final_metrics['E_mean']
                    row['L1_theta'] = sum(abs(best_res[k]) for k in param_keys if k in best_res)
                    final_results.append(row)
                    print(f"  >> Best: J={best_J:.2f}, E={row['E_mean']:.1f}Hz, I={row['I_lower_mean']:.1f}")

    # Save
    df = pd.DataFrame(final_results)
    df.to_csv(run_dir / "tables" / "stage2_best.csv", index=False)
    print("Optimization Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--beta_e', type=float, nargs='+', default=[0.0, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument('--beta_c', type=float, nargs='+', default=[0.01, 0.03, 0.1])
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--tau_c', type=float, default=0.02, help='Stimulus correlation time')
    args = parser.parse_args()
    
    # Apply tau_c override
    base_cfg = load_config(args.config)
    
    # Load subconfigs first (copied from b1)
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
    
    # Pass modified config to optimize (via args logic needing update? No, pass base_cfg directly)
    # optimize_stage2 loads config again? No, it calls load_config(args.config).
    
    # We need to change how optimize_stage2 loads config.
    # Refactor optimize_stage2 to accept the config object or path.
    # Actually, optimize_stage2 calls `load_config(args.config)`.
    # Let's pass the valid config to it.
    
    optimize_stage2(args, base_cfg)



if __name__ == "__main__":
    main()
