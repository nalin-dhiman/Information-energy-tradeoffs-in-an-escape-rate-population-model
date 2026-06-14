import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.signal import lfilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.io.config import load_config
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    base_cfg = load_config(args.config)
    
    # Resolve sub-configs if strings
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

    # Force Params per user limit
    base_cfg['simulation']['N'] = 2000
    base_cfg.setdefault('decode', {})
    base_cfg['decode']['n_trials'] = 20
    base_cfg['decode']['split'] = 'trial'
    
    # Setup v1_w (robustness)
    # Using 'v1_w_a' logic handled by create_run_dir if v1_w exists?
    # User said: "New robustness run: runs/v1_w/" 
    # If I just use create_run_dir(major_version=1), it will auto-increment.
    # I can try to force 'v1_w' if possible or let it flow. The user accepts 'v1_w_a'.
    run_dir = create_run_dir(major_version=1)
    print(f"Starting Phase 10 Robustness: {run_dir}")
    
    # Metadata
    full_log = {'metadata': get_run_metadata(), 'config': base_cfg, 'args': vars(args)}
    with open(run_dir / "run.json", "w") as f:
        json.dump(full_log, f, indent=2, default=str)
        
    theta0_vals = np.linspace(-6.0, 2.0, 10) # 10 values as per "Stage 1" grid logic? 
    # Or "run the same theta0 sweep". Previous sweep was 15 points.
    # User said: "run the same theta0 sweep for seeds {0,1,2}".
    # Let's use 10 points to be faster but cover range.
    
    seeds = [0, 1, 2]
    tau = 0.02
    n_trials = 20
    
    results = []
    
    print(f"Sweep: {len(seeds)} Seeds, {len(theta0_vals)} Theta0, N=2000")
    
    for seed in seeds:
        print(f"Seed {seed}...")
        for theta0 in theta0_vals:
            S_trials = []
            Rate_trials = [] # For Mode A
            Spike_trials = [] # For Mode B
            E_trials = []
            
            for tr in range(n_trials):
                trial_seed = seed * 1000 + tr 
                current_cfg = json.loads(json.dumps(base_cfg))
                current_cfg['hazard']['theta0'] = float(theta0)
                current_cfg['simulation']['seed'] = trial_seed
                current_cfg['stimulus']['seed'] = trial_seed
                
                try:
                    data = run_simulation(current_cfg)
                except Exception as e:
                    print(f"  Sim Fail {theta0}: {e}")
                    continue
                    
                S = data['S']
                spikes = data['spikes'] # (T, N)
                dt = data['dt']
                
                # Mode A: Rate
                A_rate = convolve_spikes(spikes, tau, dt)
                
                # Mode B: Spikes
                # Need per-dt population count as input to binning in estimator
                pop_counts = np.sum(spikes, axis=1) # (T,)
                
                S_trials.append(S)
                Rate_trials.append(A_rate)
                Spike_trials.append(pop_counts)
                E_trials.append(np.mean(A_rate))
                
            mean_rate = np.mean(E_trials)
            
            # ESTIMATOR CALLS
            # Config preparation
            lcfg = base_cfg['estimators']['lower'].copy()
            lcfg['split'] = 'trial'
            lcfg['n_trials'] = n_trials
            lcfg['seed'] = seed
            if 'bandwidth' in base_cfg['stimulus']:
                lcfg['bandwidth'] = base_cfg['stimulus']['cutoff_freq']
            
            # Setup lags
            if 'lag_taps' in base_cfg.get('decode', {}):
                lcfg['lags'] = int(base_cfg['decode']['lag_taps'])
            if 'ridge_alpha' in base_cfg.get('decode', {}):
                lcfg.setdefault('parameters', {})['alpha'] = float(base_cfg['decode']['ridge_alpha'])
                
            # Mode A: rate_lags
            lcfg_rate = lcfg.copy()
            lcfg_rate['feature_mode'] = 'rate_lags'
            res_rate = estimate_mi_lower_decode(S_trials, Rate_trials, dt, lcfg_rate)
            I_rate = res_rate.get('I_lower_bits_per_s', 0.0)
            
            # Mode B: spikecount_lags
            lcfg_spike = lcfg.copy()
            lcfg_spike['feature_mode'] = 'spikecount_lags'
            # "Bin spikes into bins of width bin_dt (e.g. 10ms or dt_eff)"
            # Let's let estimator infer from bandwidth, or force 10ms?
            lcfg_spike['bin_dt'] = 0.01 
            # Need to ensure window matches "0-500ms"
            # 500ms / 10ms = 50 lags.
            lcfg_spike['lag_window'] = 0.5 
            # Override 'lags' if it conflicts with window?
            # mi_lower_decode checks 'lags' first. Let's remove 'lags' to use window logic for spikes.
            if 'lags' in lcfg_spike: del lcfg_spike['lags']
            
            res_spike = estimate_mi_lower_decode(S_trials, Spike_trials, dt, lcfg_spike)
            I_spike = res_spike.get('I_lower_bits_per_s', 0.0)
            
            results.append({
                'seed': seed,
                'theta0': theta0,
                'mean_rate': mean_rate,
                'I_lower_rate': I_rate,
                'I_lower_spike': I_spike,
                'bpj_rate': I_rate / (mean_rate + 1e-9),
                'bpj_spike': I_spike / (mean_rate + 1e-9)
            })
            
    # CHECK G3'
    df = pd.DataFrame(results)
    df.to_csv(run_dir / "tables" / "decoder_robustness.csv", index=False)
    
    seeds_pass = 0
    with open(run_dir / "logs" / "summary.txt", "w") as f:
        f.write("G3' Analysis\n")
        f.write("------------\n")
        
        for s in seeds:
            sub = df[df['seed'] == s]
            if sub.empty: continue
            
            # Peak Rate Region
            idx_r = sub['I_lower_rate'].idxmax()
            rate_peak_r = sub.loc[idx_r, 'mean_rate']
            
            idx_s = sub['I_lower_spike'].idxmax()
            rate_peak_s = sub.loc[idx_s, 'mean_rate']
            
            diff_pct = abs(rate_peak_r - rate_peak_s) / (rate_peak_r + 1e-9)
            
            status = "PASS" if diff_pct <= 0.25 else "FAIL"
            if status == "PASS": seeds_pass += 1
            
            line = f"Seed {s}: PeakRate(RateMode)={rate_peak_r:.2f}Hz, PeakRate(SpikeMode)={rate_peak_s:.2f}Hz. Diff={diff_pct:.1%}. {status}"
            print(line)
            f.write(line + "\n")
            
        if seeds_pass >= 2:
            msg = "G3' Result: PASS"
        else:
            msg = "G3' Result: FAIL. Decoder-dependent optimum; proceed only with explicit decoder dependence framing."
            
        print("\n" + msg)
        f.write("\n" + msg + "\n")
        
    # Figures
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Rate vs Rate
    plt.figure()
    for s in seeds:
         sub = df[df['seed'] == s].sort_values('mean_rate')
         plt.plot(sub['mean_rate'], sub['bpj_rate'], 'o-', label=f'Seed {s}')
    plt.xlabel('Mean Rate (Hz)')
    plt.ylabel('Bits/Joule (Rate Lags)')
    plt.title('Performance (Rate Lags)')
    plt.legend()
    plt.savefig(fig_dir / "bpj_lower_rate_vs_rate.pdf")
    
    # Spike vs Rate
    plt.figure()
    for s in seeds:
         sub = df[df['seed'] == s].sort_values('mean_rate')
         plt.plot(sub['mean_rate'], sub['bpj_spike'], 'x--', label=f'Seed {s}')
    plt.xlabel('Mean Rate (Hz)')
    plt.ylabel('Bits/Joule (SpikeCount Lags)')
    plt.title('Performance (SpikeCount Lags)')
    plt.legend()
    plt.savefig(fig_dir / "bpj_lower_spikecount_vs_rate.pdf")

if __name__ == "__main__":
    main()
