import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.paths import create_run_dir, get_latest_run_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_run', type=str, default=None, help='Path to source run dir (e.g., runs/v1_h)')
    args = parser.parse_args()
    
    # 1. Load Source Data
    if args.source_run:
        source_dir = Path(args.source_run)
    else:
        source_dir = get_latest_run_dir(ROOT)

    # If LATEST points to a later-stage run, fall back to the newest run that has the Stage-0 CSV.
    csv_path = source_dir / "tables" / "sweep_theta0.csv"
    if not csv_path.exists() and not args.source_run:
        runs_root = ROOT / "runs"
        if runs_root.exists():
            candidates = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("v")],
                               key=lambda p: p.stat().st_mtime, reverse=True)
            for cand in candidates:
                cp = cand / "tables" / "sweep_theta0.csv"
                if cp.exists():
                    source_dir = cand
                    csv_path = cp
                    break

    print(f"Loading source data from: {source_dir}")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run Stage-0 sweep first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Load Source Config for Baseline
    run_json_path = source_dir / "run.json"
    baseline_rate = 1.0 # Default
    if run_json_path.exists():
        with open(run_json_path) as f:
            meta = json.load(f)
            # config in meta['config'] or meta depending on how it was saved
            cfg = meta.get('config', {})
            baseline_rate = cfg.get('objective', {}).get('baseline_rate', 1.0)
            print(f"Loaded baseline_rate: {baseline_rate} Hz from config")
    else:
        print("Warning: run.json not found, using default baseline=1.0")
    
    # 2. Create Output Run (v2_a...)
    run_dir = create_run_dir(major_version=2, root=ROOT)
    print(f"Starting Analysis Run: {run_dir}")
    
    # Write metadata
    with open(run_dir / "run.json", "w") as f:
         json.dump({'source_run': str(source_dir), 'args': vars(args)}, f)
    
    # 3. Pareto Analysis
    beta_E_vals = [0.0, 0.1, 0.3, 1.0]
    beta_C_vals = [0.0, 0.01, 0.03]
    
    results = []
    
    for bE in beta_E_vals:
        for bC in beta_C_vals:
            # Objective: Maximize I_lower - Cost
            # Cost = bE * Rate + Baseline + bC * |theta|
            # Note: optimization should include baseline? 
            # Baseline is constant shift, doesn't affect argmax if bE is fixed.
            # But "opt_E" should be the *Total Energy*.
            
            # Recalculate Energy for this bE
            energy = bE * df['mean_rate'] + baseline_rate
            
            # Objective
            # L = I_lower - energy - bC * |theta0|
            # L = I_lower - bE*Rate - Baseline - bC*|theta0|
            
            df['objective'] = df['I_lower'] - energy - bC * np.abs(df['theta0'])
            
            best_idx = df['objective'].idxmax()
            best_row = df.loc[best_idx]
            
            # Store Opt E
            opt_E_val = bE * best_row['mean_rate'] + baseline_rate
            
            results.append({
                'beta_E': bE,
                'beta_C': bC,
                'opt_theta0': best_row['theta0'],
                'opt_rate': best_row['mean_rate'],
                'opt_I': best_row['I_lower'],
                'opt_E': opt_E_val
            })
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(run_dir / "tables" / "pareto_light.csv", index=False)
    
    # 4. Plot Pareto
    plt.figure()
    
    # Plot Source Sweep (reference, using default bE=1 from source?)
    # Just plot I vs Rate
    plt.plot(df['mean_rate'], df['I_lower'], 'k--', label='Sweep (bC=0)', alpha=0.3)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Plot Pareto Fronts
    # Each bC defines a curve? Or each bE?
    # Usually we plot E vs I.
    # Varying bE traces out the frontier.
    # Group by bC.
    
    for i, bC in enumerate(beta_C_vals):
        sub = df_res[df_res['beta_C'] == bC].sort_values('opt_E')
        # If bE varies, E varies.
        plt.plot(sub['opt_rate'], sub['opt_I'], marker=markers[i%len(markers)], label=f'beta_C={bC}')
        
    plt.xlabel("Mean Rate (Hz) [Energy Proxy]")
    plt.ylabel("Information (bits/s)")
    plt.title("Pareto Light (Optimization)")
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / "figures" / "pareto_light.pdf")
    plt.close()
    
    print("Pareto analysis complete.")

if __name__ == "__main__":
    main()
