
import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.paths import create_run_dir
from src.io.runlog import log_run

def run_final_stats(args):
    
    
    run_root = Path("runs/v7_a_final_stats")
    run_root.mkdir(parents=True, exist_ok=True)
    
    tables_dir = run_root / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    print(f"Starting Phase 22 (Final Stats): {run_root}")
    
    
    pareto_in = Path("results/tables/pareto_stats.csv")
    tau_in = Path("results/tables/tau_sweep_results.csv")
    
    if not pareto_in.exists():
        print(f"CRITICAL: {pareto_in} not found.")
        sys.exit(1)
        
    if not tau_in.exists():
        print(f"CRITICAL: {tau_in} not found.")
        sys.exit(1)

    print("Consolidating Pareto Stats...")
    df_p = pd.read_csv(pareto_in)
    
    BASELINE_RATE = 5.0
    
    def get_col(df, candidates):
        for c in candidates:
            if c in df.columns: return df[c]
        return None

   
    out_p = pd.DataFrame()
    out_p['beta_E'] = df_p['beta_E']
    out_p['beta_C'] = df_p.get('beta_C', 0.0)
    
    out_p['mean_I_lower_bits_per_s'] = df_p['I_lower_mean']
    out_p['std_I_lower_bits_per_s'] = df_p.get('I_lower_std', 0.0)
    
    out_p['mean_E_total'] = df_p['E_mean']
    out_p['std_E_total'] = df_p.get('E_std', 0.0) 
    
    out_p['mean_rate'] = df_p['E_mean']
    out_p['std_rate'] = df_p.get('E_std', 0.0)
    
   
    out_p['mean_bits_per_J'] = df_p['I_lower_mean'] / (df_p['E_mean'] + BASELINE_RATE)
    
    I = df_p['I_lower_mean']
    Is = df_p.get('I_lower_std', 0.0)
    E = df_p['E_mean'] + BASELINE_RATE
    Es = df_p.get('E_std', 0.0)
    
    out_p['std_bits_per_J'] = out_p['mean_bits_per_J'] * np.sqrt((Is/I)**2 + (Es/E)**2)
    
    out_p['stage'] = df_p.get('stage', 'Stage 2')
    
    out_p.to_csv(tables_dir / "pareto_consolidated.csv", index=False)
    print(f"Saved {len(out_p)} rows to pareto_consolidated.csv")
    
    
    pool = out_p.sort_values('mean_E_total')
    union_front = []
    
    current_max_I = -np.inf
    
    for idx, row in pool.iterrows():
        if row['mean_I_lower_bits_per_s'] > current_max_I:
            union_front.append(row)
            current_max_I = row['mean_I_lower_bits_per_s']
            
    pd.DataFrame(union_front).to_csv(tables_dir / "pareto_union_front.csv", index=False)
    print(f"Saved {len(union_front)} points to pareto_union_front.csv")


    print("Consolidating Tau Sweep...")
    df_t = pd.read_csv(tau_in)
    
    out_t = pd.DataFrame()
    out_t['beta_E'] = df_t['beta_E']
    out_t['tau_c'] = df_t.get('tau_c', df_t.get('tau', 0.0))
    
    if 'rate_mean' in df_t.columns:
        rate_mean = df_t['rate_mean']
        rate_std = df_t.get('rate_std', 0.0)
    else:
        rate_mean = df_t['E_mean']
        rate_std = df_t.get('E_std', 0.0)
        
    out_t['optimal_rate_mean'] = rate_mean
    out_t['optimal_rate_std'] = rate_std
    
    out_t['mean_I_lower'] = df_t['I_lower_mean']
    out_t['std_I_lower'] = df_t.get('I_lower_std', 0.0)
    
    out_t['mean_E_total'] = rate_mean
    out_t['std_E_total'] = rate_std
    
    out_t['mean_bits_per_J'] = df_t['I_lower_mean'] / (rate_mean + BASELINE_RATE)
    
    I = df_t['I_lower_mean']
    Is = df_t.get('I_lower_std', 0.0)
    E = rate_mean + BASELINE_RATE
    Es = rate_std
    out_t['std_bits_per_J'] = out_t['mean_bits_per_J'] * np.sqrt((Is/I)**2 + (Es/E)**2)
    
    out_t.to_csv(tables_dir / "tauc_consolidated.csv", index=False)
    print(f"Saved {len(out_t)} rows to tauc_consolidated.csv")
    
    with open(run_root / "summary.txt", "w") as f:
        f.write("Paper B Final Summary (v7_a)\n")
        f.write("==============================\n")
        f.write(f"Pareto Points: {len(out_p)}\n")
        f.write(f"Union Front Points: {len(union_front)}\n")
        f.write("Status: Tables Consolidated.\n")

if __name__ == "__main__":
    run_final_stats(None)
