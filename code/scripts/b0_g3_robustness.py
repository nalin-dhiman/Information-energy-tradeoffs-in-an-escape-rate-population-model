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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import parse_cli_overrides, save_config, load_config
from src.io.paths import create_run_dir
from src.io.runlog import get_run_metadata
from src.simulate import run_simulation
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--seeds', type=str, default="0,1,2")
    
    args, unknown = parser.parse_known_args()
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
    base_cfg['simulation']['N'] = 2000
    base_cfg.setdefault('decode', {})
    base_cfg['decode']['n_trials'] = 20
    base_cfg['decode']['split'] = 'trial'
    
    run_dir = create_run_dir(major_version=1)
    print(f"Starting G3' Robustness Check (Phase 8): {run_dir}")
    
    metadata = get_run_metadata()
    full_log = {'metadata': metadata, 'config': base_cfg, 'args': vars(args)}
    run_json_path = run_dir / "run.json"
    with open(run_json_path, "w") as f:
        json.dump(full_log, f, indent=2, default=str)
        
    theta0_vals = np.linspace(-5.0, 1.0, 15)
    tau = 0.02
    seeds = [int(x) for x in args.seeds.split(',')]
    n_trials = base_cfg['decode']['n_trials']
    
    results = []
    diag_rate = []
    diag_spike = []
    
    print(f"Running Comparison: Rate vs SpikeCount (Tau={tau}s, N=2000, 20 trials)")
    
    for seed in seeds:
         print(f"  Seed {seed}")
         for theta0 in theta0_vals:
              S_trials = []
              Rate_trials = []
              Spike_trials = []
              
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
                   
                   rate_feature = convolve_spikes(spikes, tau, dt)
                   Rate_trials.append(rate_feature)
                   
                   
                   pop_counts = np.sum(spikes, axis=1)
                   Spike_trials.append(pop_counts)
                   
                   if tr == 0:
                       mean_rate = np.mean(rate_feature)
                       E_total = mean_rate 
                   
              lcfg_r = base_cfg['estimators']['lower'].copy()
              lcfg_r['split'] = 'trial'
              lcfg_r['n_trials'] = n_trials
              lcfg_r['seed'] = seed
              lcfg_r['features'] = 'rate'
              if 'bandwidth' in base_cfg['stimulus']:
                   lcfg_r['bandwidth'] = base_cfg['stimulus']['cutoff_freq']
              
              d_params = base_cfg.get('decode', {})
              if 'ridge_alpha' in d_params: lcfg_r.setdefault('parameters', {})['alpha'] = float(d_params['ridge_alpha'])
              if 'lag_taps' in d_params: lcfg_r['lags'] = int(d_params['lag_taps'])

              res_r = estimate_mi_lower_decode(S_trials, Rate_trials, dt, lcfg_r)
              I_rate = res_r.get('I_lower_bits_per_s', 0.0)
              
              diag_rate.append({
                  'seed': seed, 'theta0': theta0, 'mean_rate': mean_rate,
                  'I_lower': I_rate, 'r2': res_r.get('r2_test'), 'features': 'rate'
              })

              lcfg_s = lcfg_r.copy()
              lcfg_s['features'] = 'spikecount'
             
              cutoff = base_cfg['stimulus'].get('cutoff_freq', 50.0)
              lcfg_s['spike_binsize'] = 1.0 / (2.0 * cutoff) 
              
              res_s = estimate_mi_lower_decode(S_trials, Spike_trials, dt, lcfg_s)
              I_spike = res_s.get('I_lower_bits_per_s', 0.0)
              
              diag_spike.append({
                  'seed': seed, 'theta0': theta0, 'mean_rate': mean_rate,
                  'I_lower': I_spike, 'r2': res_s.get('r2_test'), 'features': 'spikecount'
              })
              
              results.append({
                  'seed': seed, 'theta0': theta0, 
                  'mean_rate': mean_rate,
                  'I_lower_rate': I_rate,
                  'I_lower_spike': I_spike,
                  'diff': I_rate - I_spike
              })

    df = pd.DataFrame(results)
    df.to_csv(run_dir / "tables" / "sweep_theta0.csv", index=False)
    
    pd.DataFrame(diag_rate).to_csv(run_dir / "tables" / "decoder_diagnostics_rate.csv", index=False)
    pd.DataFrame(diag_spike).to_csv(run_dir / "tables" / "decoder_diagnostics_spikecount.csv", index=False)

    fig_dir = run_dir / "figures"
    
    plt.figure()
    colors = ['b', 'g', 'r']
    for i, s in enumerate(seeds):
        sub = df[df['seed'] == s].sort_values('mean_rate')
        plt.plot(sub['mean_rate'], sub['I_lower_rate'], f'{colors[i]}o-', label=f'Rate S{s}')
        plt.plot(sub['mean_rate'], sub['I_lower_spike'], f'{colors[i]}x--', label=f'Spike S{s}')
    plt.xlabel('Rate (Hz)')
    plt.ylabel('Information (bits/s)')
    plt.legend()
    plt.title("G3' Check: Rate vs SpikeCount Features")
    plt.grid(True)
    plt.savefig(fig_dir / "info_rate_vs_spike.pdf")
    
    print("\n--- G3' ROBUSTNESS CHECK ---")
    pass_g3p = 0
    for s in seeds:
        sub = df[df['seed'] == s].sort_values('mean_rate')
        
        idx_r = sub['I_lower_rate'].idxmax()
        val_r = sub.loc[idx_r, 'I_lower_rate']
        rate_at_peak_r = sub.loc[idx_r, 'mean_rate']
        
        idx_s = sub['I_lower_spike'].idxmax()
        val_s = sub.loc[idx_s, 'I_lower_spike']
        rate_at_peak_s = sub.loc[idx_s, 'mean_rate']
        
        ratio_val = abs(val_r - val_s) / (max(val_r, val_s) + 1e-9)
        ratio_loc = abs(rate_at_peak_r - rate_at_peak_s) / (max(rate_at_peak_r, rate_at_peak_s) + 1e-9)
        
        print(f"Seed {s}: Peak Rate-Dec={val_r:.2f} (@{rate_at_peak_r:.1f}Hz), Spike-Dec={val_s:.2f} (@{rate_at_peak_s:.1f}Hz)")
        print(f"        Diff Val={ratio_val:.2f}, Diff Loc={ratio_loc:.2f}")
        
        if ratio_loc < 0.25:
            pass_g3p += 1
            
    summary_path = run_dir / "logs" / "summary.txt"
    with open(summary_path, 'w') as f:
        if pass_g3p >= 2:
            msg = "PASS G3': Feature Sets Agree on Peak Location."
            print(msg)
            f.write(msg + "\n")
        else:
            msg = "FAIL G3': Feature Sets Disagree. Proceed only with explicit decoder-dependence claim."
            print(msg)
            f.write(msg + "\n")

if __name__ == "__main__":
    main()
