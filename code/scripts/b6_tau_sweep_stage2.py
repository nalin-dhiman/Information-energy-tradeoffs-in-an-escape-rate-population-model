import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.config import load_config
from src.io.paths import create_run_dir
from src.io.runlog import log_run
from scripts.b5_stage2_optimization import optimize_stage2

def run_tau_sweep_stage2(args):
    # Determine mode and run directory
    suffix = 'a' # Default v6_a
    notes = 'Stage 2 Tau Sweep (Gaussian)'
    stim_override = None
    
    if args.mode == 'switching':
        suffix = 'b' # v6_b
        notes = 'Stage 2 Tau Sweep (Switching Stimulus)'
        stim_override = 'configs/stimulus/gauss_switching.yaml'
    elif args.mode == 'ablation':
        suffix = 'c' # v6_c
        notes = 'Stage 2 Ablations (BetaC=0 or BetaE=0)'
        
    run_dir = create_run_dir(major_version=6)
    print(f"Starting Phase 18/19/20 ({args.mode}): {run_dir}")
    
    # Load Base Config
    base_cfg = load_config(args.config)
    
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
    
    # Apply Stimulus Override
    if stim_override:
        p = Path(stim_override)
        if not p.exists(): p = Path(__file__).parent.parent / str(p)
        if p.exists(): 
            print(f"Overriding stimulus config with {p}")
            base_cfg['stimulus'] = load_config(str(p))
            
    # Metadata
    tau_list = args.tau_c #[0.02, 0.05, 0.1, 0.2]
    beta_E_list = args.beta_e #[0, 1, 10]
    
    extra_meta = {
        'mode': args.mode,
        'tau_list': tau_list,
        'beta_E_list': beta_E_list,
        'beta_C_list': args.beta_c,
        'notes': notes,
        # Strict Metadata Extraction
        'dt': base_cfg['simulation'].get('dt', 0.0001),
        'trial_T': base_cfg['simulation'].get('duration', 5.0),
        'seed_list': [0, 1, 2], # Used in optimization
        'N': base_cfg['simulation'].get('N', 200),
        'trials': 20,
        'dt_eff': 1.0 / (2 * base_cfg['stimulus'].get('bandwidth', 5.0)) # Calculate from config or default
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
    # Loop
    results = []
    
    # Param keys
    param_keys = ['theta0', 'thetaV', 'thetaa', 'thetaVV', 'thetaaa', 'thetaVa']
    baseline_rate = 5.0
    
    import multiprocessing
    # Use pool for parallel eval of neighbors
    # Reduced processes to 5 to avoid OOM with Switching stimulus
    with multiprocessing.Pool(processes=5) as pool:
    
        for tau in tau_list:
            print(f"\n=== Tau C = {tau} ===")
            # Update config for tau
            current_base_cfg = base_cfg.copy() # Shallow copy
            if 'stimulus' not in current_base_cfg: current_base_cfg['stimulus'] = {}
            # Deep copy stimulus if needed? 
            # Actually load_config returns dicts.
            # To be safe:
            import copy
            current_base_cfg = copy.deepcopy(base_cfg)
            current_base_cfg['stimulus']['tau_c'] = tau
            
            # Optimize for each BetaE
            loop_beta_E = beta_E_list
            loop_beta_C = args.beta_c

            if args.mode == 'ablation':
                # Argument logic: 
                # If we want BetaC=0 ablation: run with normal betaE, force betaC=0
                # If we want BetaE=0 ablation: run with betaE=0, normal betaC
                
                # To support both in one script, we rely on the input args.
                # If user passes --beta_c 0, it's BetaC ablation.
                # If user passes --beta_e 0, it's BetaE ablation.
                # We just use the lists as provided.
                pass

            for be in loop_beta_E:
                for bc in loop_beta_C:
                    if args.mode == 'ablation':
                         print(f"  Ablation Run: BetaE={be}, BetaC={bc}")
                    else:
                         print(f"  Optimizing BetaE={be}, BetaC={bc}...")
                    
                    best_J = -np.inf
                    best_res = None
                    
                    # Seeds (Standard Robust + High Rate Start)
                    seeds = [
                        {'theta0': 0.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0},
                        {'theta0': -2.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0},
                        {'theta0': 2.0, 'thetaV': 6.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0}
                    ]
                    
                    from scripts.b5_stage2_optimization import evaluate_point_stage2
                    
                    for seed_idx, start_p in enumerate(seeds):
                        curr_p = start_p.copy()
                        
                        # Initial Eval
                        try:
                            # Use map for single point to avoid pickling issues? No, direc call is fine.
                            # But evaluate_point_stage2 is robust?
                            res = evaluate_point_stage2(curr_p, current_base_cfg, tau=tau)
                        except Exception as e:
                            print(f"    Seed {seed_idx} Init Failed: {e}")
                            res = None
                        
                        if not res: continue
                        
                        L1 = sum(abs(v) for v in curr_p.values())
                        curr_J = res['I_lower_mean'] - be * (res['E_mean'] + baseline_rate) - bc * L1
                        
                        # Descent
                        step_sz = 0.5
                        for step in range(8): # Limit steps for sweep speed
                            neighbors = []
                            for k in param_keys:
                                for sgn in [-1, 1]:
                                    n_p = curr_p.copy()
                                    n_p[k] += sgn * step_sz
                                    neighbors.append(n_p)
                            
                            # Parallel Eval
                            eval_args = [(p, current_base_cfg, 20, [0,1,2], tau) for p in neighbors]
                            # evaluate_point_stage2 signature: theta_dict, cfg, n_trials=20, seeds=[0,1,2], tau=0.02
                            # Use starmap
                            neighbor_results = pool.starmap(evaluate_point_stage2, eval_args)
                            
                            improved = False
                            for p, r in zip(neighbors, neighbor_results):
                                if not r: continue
                                L1_n = sum(abs(v) for v in p.values())
                                J_n = r['I_lower_mean'] - be * (r['E_mean'] + baseline_rate) - bc * L1_n
                                
                                if J_n > curr_J + 1e-4:
                                    curr_J = J_n
                                    curr_p = p
                                    improved = True
                            
                            if not improved:
                                step_sz *= 0.5
                                if step_sz < 0.1: break
                        
                        # Check Best
                        if curr_J > best_J:
                            best_J = curr_J
                            best_res = curr_p.copy()
                            # Get final metrics - Re-eval to get full stats including stds
                            final_metrics = evaluate_point_stage2(best_res, current_base_cfg, tau=tau)
                            
                            # Extract derived physical units
                            bandwidth = current_base_cfg['stimulus'].get('bandwidth', 50.0)
                            dt_eff = 1.0 / (2 * bandwidth)
                            tau_ms = tau * 1000.0
                            
                            best_res.update({
                                'J': best_J,
                                'I_lower_mean': final_metrics['I_lower_mean'],
                                'I_lower_std': final_metrics['I_lower_std'],
                                'I_upper_mean': final_metrics['I_upper_mean'],
                                'I_upper_std': final_metrics['I_upper_std'],
                                'rate_mean': final_metrics['E_mean'], # Rename for clarity per request
                                'rate_std': final_metrics['E_std'],
                                'E_mean': final_metrics['E_mean'], # Keep for compat
                                'tau_c': tau,
                                'beta_E': be,
                                'beta_C': bc,
                                # Audit Columns
                                'dt_eff': dt_eff,
                                'cutoff': bandwidth,
                                'tau_ms': tau_ms,
                                'seed_count': 3,
                                'trial_count': 20
                            })

                    if best_res:
                        results.append(best_res)
                        print(f"    >> Done: Tau={tau}, bE={be} -> J={best_J:.2f}, Rate={best_res['rate_mean']:.1f}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "tables" / "tau_sweep_results.csv", index=False)
    print("Sweep Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--mode', choices=['gaussian', 'switching', 'ablation'], default='gaussian')
    parser.add_argument('--tau_c', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument('--beta_e', type=float, nargs='+', default=[0, 1, 10])
    parser.add_argument('--beta_c', type=float, nargs='+', default=[0.03]) # Default penalty
    args = parser.parse_args()
    
    # If ablation, override defaults if not provided? 
    if args.mode == 'ablation':
        # If user didn't specify, we might want specific ablation sets.
        # But let's assume arguments control it.
        pass
        
    run_tau_sweep_stage2(args)

if __name__ == "__main__":
    main()
