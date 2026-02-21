
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.config import load_config
from src.io.paths import create_run_dir
from src.io.runlog import log_run

from scripts.b6_tau_sweep_stage2 import run_tau_sweep_stage2
from scripts.b5_stage2_optimization import optimize_stage2



def run_ablation_audit(args):
    run_dir = create_run_dir(major_version=6)
    print(f"Starting Phase 21 (Ablation Audit): {run_dir}")
    
    base_cfg = load_config(args.config)
    
    def resolve_includes_top(cfg, root=Path(__file__).parent.parent):
         for k, v in cfg.items():
            if isinstance(v, str) and v.endswith('.yaml'):
                p = Path(v)
                if not p.exists(): p = root / v
                if p.exists():
                    with open(p) as f:
                        cfg[k] = __import__('yaml').safe_load(f)
                        resolve_includes_top(cfg[k], root)
            elif isinstance(v, dict):
                resolve_includes_top(v, root)
    resolve_includes_top(base_cfg)
    
    dt = base_cfg['simulation']['dt']
    trial_T = base_cfg['simulation']['T']
    N = base_cfg['simulation']['N']
    bandwidth = base_cfg['stimulus'].get('bandwidth', 50.0)
    dt_eff = 1.0 / (2 * bandwidth)

    extra_meta = {
        'notes': 'Phase 21 Audit: Ablations',
        'protocols': ['BetaC=0', 'BetaE=0'],
        'seeds': [0, 1, 2],
        'trials': 20,
        'dt': dt,
        'dt_eff': dt_eff, 
        'N': N,
        'trial_T': trial_T,
        'tau_list': [0.02, 0.1], 
        'seed_list': [0, 1, 2],
        'beta_E_list': [0.0, 1.0, 10.0], 
        'beta_C_list': [0.0, 0.03]
    }
    log_run(run_dir, {'config': base_cfg, 'args': vars(args)}, extra_meta)
    
   
    print("\n=== Running Ablation: BetaC=0 ===")
    
    
    
    tau_list = [0.02, 0.1]
    
    sets = [
        ('ablation_betaC0', {'beta_c': [0.0], 'beta_e': [0.0, 1.0, 10.0]}),
        ('ablation_betaE0', {'beta_c': [0.0, 0.03], 'beta_e': [0.0]})
    ]
    
    import multiprocessing
    pool = multiprocessing.Pool(processes=5)
    
    from scripts.b5_stage2_optimization import evaluate_point_stage2
    
    for label, params in sets:
        print(f"\n--- Protocol: {label} ---")
        results = []
        b_c_list = params['beta_c']
        b_e_list = params['beta_e']
        
        for tau in tau_list:
            print(f"  Tau={tau}")
            import copy
            current_cfg = copy.deepcopy(base_cfg)
            if 'stimulus' not in current_cfg: current_cfg['stimulus'] = {}
            current_cfg['stimulus']['tau_c'] = tau
            
            def resolve_includes(cfg, root=Path(__file__).parent.parent):
                 for k, v in cfg.items():
                    if isinstance(v, str) and v.endswith('.yaml'):
                        p = Path(v)
                        if not p.exists(): p = root / v
                        if p.exists():
                            with open(p) as f:
                                cfg[k] = __import__('yaml').safe_load(f)
                                resolve_includes(cfg[k], root)
                    elif isinstance(v, dict):
                        resolve_includes(v, root)
            resolve_includes(current_cfg)
            
            bandwidth = current_cfg['stimulus'].get('bandwidth', 50.0)
            
            param_keys = ['theta0', 'thetaV', 'thetaa', 'thetaVV', 'thetaaa', 'thetaVa']
            baseline_rate = 5.0
            
            for be in b_e_list:
                for bc in b_c_list:
                    
                    print(f"    Optimizing bE={be}, bC={bc}...")
                    
                    best_J = -np.inf
                    best_res = None
                    
                    seeds = [
                        {'theta0': 0.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0},
                        {'theta0': -2.0, 'thetaV': 5.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0},
                        {'theta0': 2.0, 'thetaV': 6.0, 'thetaa': 0.0, 'thetaVV': 0.0, 'thetaaa': 0.0, 'thetaVa': 0.0}
                    ]
                    
                    for seed_idx, start_p in enumerate(seeds):
                        curr_p = start_p.copy()
                        
                        try:
                            res = evaluate_point_stage2(curr_p, current_cfg, tau=tau)
                        except Exception: res = None
                        
                        if not res: continue
                        
                        L1 = sum(abs(v) for v in curr_p.values())
                        curr_J = res['I_lower_mean'] - be * (res['E_mean'] + baseline_rate) - bc * L1
                        
                        step_sz = 0.5
                        for step in range(8):
                            neighbors = []
                            for k in param_keys:
                                for sgn in [-1, 1]:
                                    n_p = curr_p.copy()
                                    n_p[k] += sgn * step_sz
                                    neighbors.append(n_p)
                                    
                            eval_args = [(p, current_cfg, 20, [0,1,2], tau) for p in neighbors]
                            n_results = pool.starmap(evaluate_point_stage2, eval_args)
                            
                            improved = False
                            for p, r in zip(neighbors, n_results):
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
                        
                        if curr_J > best_J:
                            best_J = curr_J
                            best_res = curr_p.copy()
                            final_metrics = evaluate_point_stage2(best_res, current_cfg, tau=tau)
                            dt_eff = 1.0 / (2*bandwidth)
                            best_res.update({
                                'J': best_J,
                                'I_lower_mean': final_metrics['I_lower_mean'],
                                'I_lower_std': final_metrics['I_lower_std'],
                                'I_upper_mean': final_metrics['I_upper_mean'],
                                'I_upper_std': final_metrics['I_upper_std'],
                                'rate_mean': final_metrics['E_mean'],
                                'rate_std': final_metrics['E_std'],
                                'tau_c': tau,
                                'beta_E': be,
                                'beta_C': bc,
                                'dt_eff': dt_eff,
                                'cutoff': bandwidth,
                                'tau_ms': tau * 1000.0,
                                'trial_count': 20,
                                'seed_count': 3
                            })
                            
                    if best_res:
                        results.append(best_res)
                        print(f"      >> Result: J={best_J:.2f}, Rate={best_res['rate_mean']:.1f}")
        
        df = pd.DataFrame(results)
        df.to_csv(run_dir / "tables" / f"{label}.csv", index=False)
        print(f"Saved {label}.csv")
        
    pool.close()
    pool.join()
    
    
    import subprocess
    for label, _ in sets:
        csv_path = run_dir / "tables" / f"{label}.csv"
        
        print(f"Plotting {label}...")
        subprocess.run(["python3", "scripts/b6_plot_tau.py", str(csv_path)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    args = parser.parse_args()
    run_ablation_audit(args)

if __name__ == "__main__":
    main()
