import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from scipy.signal import lfilter

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import parse_cli_overrides, save_config, load_config
from src.io.paths import create_run_dir
from src.io.runlog import get_run_metadata
from src.simulate import run_simulation
from src.estimators.mi_upper import estimate_mi_upper_gaussian
from src.estimators.mi_lower_decode import estimate_mi_lower_decode

def convolve_spikes(spikes, tau, dt):
    """
    Convolve spikes with exponential kernel to get rate A(t).
    Input: spikes (T, N)
    Output: A_smooth (T,)
    """
    pop_spikes = np.sum(spikes, axis=1) # (T,)
    N = spikes.shape[1]
    alpha = dt / tau 
    inst_rate = pop_spikes / (N * dt)
    b = [alpha]
    a = [1, -(1 - alpha)]
    A_smooth = lfilter(b, a, inst_rate)
    return A_smooth

def main():
    # 1. Config & Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--tau_list', type=str, default="0.01,0.02,0.05")
    parser.add_argument('--seed_list', type=str, default="0,1,2")
    parser.add_argument('--null_control', action='store_true', help="Run shuffled control analysis")
    
    args, unknown = parser.parse_known_args()
    base_cfg = load_config(args.config)
    
    # Resolve sub-configs
    if isinstance(base_cfg.get('stimulus'), str):
        p = Path(base_cfg.get('stimulus'))
        if p.exists(): base_cfg['stimulus'] = load_config(str(p))
    if 'estimators' in base_cfg:
         for k,v in base_cfg['estimators'].items():
             if isinstance(v, str):
                 p = Path(v)
                 if p.exists(): base_cfg['estimators'][k] = load_config(str(p))
                 
    # 2. Setup Run
    run_dir = create_run_dir(major_version=1)
    print(f"Starting Hardened Run (Phase 6): {run_dir}")
    
    # Strict Metadata
    metadata = get_run_metadata()
    full_log = {'metadata': metadata, 'config': base_cfg, 'args': vars(args)}
    run_json_path = run_dir / "run.json"
    with open(run_json_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)
        
    # 3. Parameters
    theta0_vals = np.linspace(-5.0, 1.0, 15)
    tau_vals = [float(x) for x in args.tau_list.split(',')]
    seeds = [int(x) for x in args.seed_list.split(',')]
    
    # Decoder Config
    decode_cfg = base_cfg.get('decode', {})
    n_trials = decode_cfg.get('n_trials', 20)
    is_trial_split = (decode_cfg.get('split') == 'trial')
    if is_trial_split:
        print(f"Trial Mode: {n_trials} trials per condition.")
    else:
        n_trials = 1
        
    results = []
    diagnostics = []
    null_results = []
    estimator_meta = []
    
    default_tau = 0.02
    if default_tau not in tau_vals: tau_vals.append(default_tau)
    tau_vals.sort()
    
    cutoff = base_cfg['stimulus'].get('cutoff_freq', None)
    
    print(f"Sweeping: Seeds={seeds}, Taus={tau_vals}, Theta0 (15 pts)")
    
    for seed in seeds:
        print(f"  Seed {seed}")
        
        for theta0 in theta0_vals:
            # Data Collection (Multi-Trial)
            S_trials = []
            A_tau_trials = {t: [] for t in tau_vals}
            A_pop_raw_trials = [] 
            
            for tr in range(n_trials):
                trial_seed = seed * 1000 + tr 
                current_cfg = json.loads(json.dumps(base_cfg))
                current_cfg['hazard']['theta0'] = float(theta0)
                current_cfg['simulation']['seed'] = trial_seed
                current_cfg['stimulus']['seed'] = trial_seed
                
                data = run_simulation(current_cfg)
                S = data['S']
                spikes = data['spikes']
                dt = data['dt']
                
                S_trials.append(S)
                A_pop_raw_trials.append(data['A'])
                
                for tau in tau_vals:
                    A_tau = convolve_spikes(spikes, tau, dt)
                    A_tau_trials[tau].append(A_tau)
            
            # Energy Calculation
            beta_E = base_cfg['objective'].get('beta_E', 1.0)
            baseline = base_cfg['objective'].get('baseline_rate', 0.0)
            avg_rate_raw = np.mean([np.mean(a) for a in A_pop_raw_trials])
            E_total = beta_E * avg_rate_raw + baseline
            
            # ESTIMATION LOOP (Per Tau)
            for tau in tau_vals:
                avg_rate_tau = np.mean([np.mean(a) for a in A_tau_trials[tau]])
                
                if avg_rate_tau < 1e-9:
                    I_u, I_l = 0.0, 0.0
                else:
                    # Upper Bound (Concat)
                    S_concat = np.concatenate(S_trials)
                    A_concat = np.concatenate(A_tau_trials[tau])
                    ucfg = base_cfg['estimators']['upper']
                    ucfg['parameters']['smooth_tau'] = tau
                    res_u = estimate_mi_upper_gaussian(S_concat, A_concat, dt, ucfg)
                    # I_u = res_u.get('I_upper_bits_per_s', res_u.get('mi_rate'))
                    # Explicit check
                    I_u = res_u.get('I_upper_bits_per_s', 0.0)
                    
                    if not estimator_meta:
                        estimator_meta.append(res_u.get('diagnostics', {}))

                    # Lower Bound
                    lcfg = current_cfg['estimators']['lower']
                    lcfg['split'] = 'trial' if is_trial_split else 'block'
                    lcfg['n_trials'] = n_trials
                    lcfg['seed'] = seed 
                    if cutoff: lcfg['bandwidth'] = cutoff
                    
                    # Propagate decode section params
                    d_params = current_cfg.get('decode', {})
                    if 'ridge_alpha' in d_params:
                         lcfg.setdefault('parameters', {})['alpha'] = float(d_params['ridge_alpha'])
                    if 'lag_taps' in d_params:
                         lfc = int(d_params['lag_taps'])
                         # lags usually a range or int. mi_lower_decode uses 'lags' as int taps
                         lcfg['lags'] = lfc
                    
                    # Pass List of Trials
                    res_l = estimate_mi_lower_decode(S_trials, A_tau_trials[tau], dt, lcfg)
                    I_l = res_l.get('I_lower_bits_per_s', 0.0)
                    
                    if len(estimator_meta) < 2:
                        estimator_meta.append(res_l.get('diagnostics', {}))
                        
                    # Diagnostics required: 
                    # run_id, seed, tau_ms, theta0, mean_rate, var_S_test, mse_test, r2_test, dt_eff, mi_eff_sample, I_lower_bits_per_s, clipped_flag
                    diag_row = {
                        'run_id': run_dir.name, 
                        'seed': seed, 
                        'tau_ms': tau*1000, 
                        'theta0': theta0, 
                        'mean_rate': avg_rate_raw,
                        'var_S_test': res_l.get('var_S_test'),
                        'mse_test': res_l.get('mse_test'),
                        'r2_test': res_l.get('r2_test'),
                        'dt_eff': res_l.get('dt_eff'),
                        'mi_bits_per_eff_sample': res_l.get('mi_eff_sample'),
                        'I_lower_bits_per_s': I_l,
                        'clipped_flag': res_l.get('clipped')
                    }
                    diagnostics.append(diag_row)
                    
                    # NULL CONTROL (If Requested)
                    if args.null_control and tau == default_tau:
                         # Shuffle S relative to A
                         # Just shuffle S lists
                         rng_null = np.random.RandomState(seed + 999)
                         S_shuff = list(S_trials)
                         rng_null.shuffle(S_shuff)
                         
                         res_null = estimate_mi_lower_decode(S_shuff, A_tau_trials[tau], dt, lcfg)
                         I_null = res_null.get('I_lower_bits_per_s', 0.0)
                         null_results.append({
                             'seed': seed, 'theta0': theta0, 'I_null': I_null, 'I_real': I_l
                         })

                bpj_u = I_u / E_total if E_total > 1e-9 else 0.0
                bpj_l = I_l / E_total if E_total > 1e-9 else 0.0
                
                results.append({
                    'seed': seed, 'theta0': theta0, 'tau': tau,
                    'mean_rate': avg_rate_raw, 
                    'E_total': E_total,
                    'I_upper_bits_per_s': I_u, 
                    'I_lower_bits_per_s': I_l,
                    'bpj_upper': bpj_u, 'bpj_lower': bpj_l
                })
                
    # 4. Save Outputs
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "tables" / "sweep_theta0.csv", index=False)
    
    df_diag = pd.DataFrame(diagnostics)
    df_diag.to_csv(run_dir / "tables" / "decoder_diagnostics.csv", index=False)
    
    if args.null_control:
        df_null = pd.DataFrame(null_results)
        df_null.to_csv(run_dir / "tables" / "null_control.csv", index=False)
    
    with open(run_dir / "tables" / "estimator_metadata.json", "w") as f:
        json.dump(estimator_meta, f, indent=2, default=str)
        
    # 5. Plotting
    colors = ['b', 'g', 'r']
    fig_dir = run_dir / "figures"
    
    # Efficiency vs Rate
    plt.figure()
    for i, s in enumerate(seeds):
        sub = df[(df['seed'] == s) & (df['tau'] == default_tau)]
        plt.plot(sub['mean_rate'], sub['bpj_upper'], f'{colors[i]}--', label=f'Upper S{s}')
        plt.plot(sub['mean_rate'], sub['bpj_lower'], f'{colors[i]}-', label=f'Lower S{s}')
    plt.xlabel('Rate (Hz)')
    plt.ylabel('Efficiency (bits/J)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir / "bits_per_joule_vs_rate.pdf")
    
    # Info vs Rate
    plt.figure()
    for i, s in enumerate(seeds):
        sub = df[(df['seed'] == s) & (df['tau'] == default_tau)]
        plt.plot(sub['mean_rate'], sub['I_upper_bits_per_s'], f'{colors[i]}--', label=f'Upper S{s}')
        plt.plot(sub['mean_rate'], sub['I_lower_bits_per_s'], f'{colors[i]}-', label=f'Lower S{s}')
    plt.xlabel('Rate (Hz)')
    plt.ylabel('Information (bits/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_dir / "info_vs_rate.pdf")
    
    # Energy vs Rate
    plt.figure()
    sub = df[(df['seed'] == seeds[0]) & (df['tau'] == default_tau)]
    plt.plot(sub['mean_rate'], sub['E_total'], 'k-')
    plt.xlabel('Rate (Hz)')
    plt.ylabel('Energy (J/s Proxy)')
    plt.grid(True)
    plt.savefig(fig_dir / "energy_vs_rate.pdf")

    # 6. GATES
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    summary_path = log_dir / "summary.txt"
    log = []
    
    def log_print(s):
        print(s)
        log.append(s)
        
    log_print("\n--- GATES ---")
    
    # G1: I_lower not zero
    i_lower_max = df.groupby('seed')['I_lower_bits_per_s'].max()
    if np.sum(i_lower_max > 1e-3) >= 2:
        log_print("PASS G1: I_lower valid.")
    else:
        log_print("FAIL G1: I_lower too small/zero.")
        with open(summary_path, 'w') as f: f.write('\n'.join(log))
        sys.exit(1)
        
    # G2: Interior Optimum (Lower)
    # Check per seed
    pass_g2 = 0
    for s in seeds:
         sub = df[(df['seed'] == s) & (df['tau'] == default_tau)] 
         sub = sub.sort_values('mean_rate')
         bpj = sub['bpj_lower'].values
         # Ignore endpoints (first/last)
         idx = np.argmax(bpj)
         if 0 < idx < len(bpj)-1:
             pass_g2 += 1
    
    if pass_g2 >= 2:
        log_print(f"PASS G2: Interior Optimum ({pass_g2}/{len(seeds)} seeds)")
    else:
        log_print(f"FAIL G2: Boundary Optimum ({pass_g2}/{len(seeds)} seeds)")
        with open(summary_path, 'w') as f: f.write('\n'.join(log))
        sys.exit(1)
        
    # G3: Consistency (Peak Rate)
    pass_g3 = 0
    for s in seeds:
        sub = df[(df['seed'] == s) & (df['tau'] == default_tau)].sort_values('mean_rate')
        rate_u = sub.loc[sub['bpj_upper'].idxmax(), 'mean_rate']
        rate_l = sub.loc[sub['bpj_lower'].idxmax(), 'mean_rate']
        ratio = abs(rate_u - rate_l) / (max(rate_u, rate_l) + 1e-9)
        log_print(f"Seed {s}: Peak U={rate_u:.2f}, L={rate_l:.2f} (Diff={ratio:.2f})")
        if ratio < 0.25: pass_g3 += 1
        
    if pass_g3 >= 2:
        log_print(f"PASS G3: Consistent Peaks.")
    else:
        log_print(f"FAIL G3: Divergent Peaks.")
        with open(summary_path, 'w') as f: f.write('\n'.join(log))
        sys.exit(1)

    # G4: Null Control
    if args.null_control:
        null_avg = df_null['I_null'].mean()
        real_avg = df_null['I_real'].mean()
        log_print(f"Null Control: I_null_avg={null_avg:.3f}, I_real_avg={real_avg:.3f}")
        if null_avg < 0.1 * real_avg or null_avg < 0.1: # 10% threshold or absolute small
            log_print("PASS G4: Null Control.")
        else:
            log_print("FAIL G4: High Null Information.")
            with open(summary_path, 'w') as f: f.write('\n'.join(log))
            sys.exit(1)
            
    log_print("STAGE 0 CLEAN PASS (Phase 6)")
    with open(summary_path, 'w') as f: f.write('\n'.join(log))

if __name__ == "__main__":
    main()
