import pandas as pd
import numpy as np
import sys

def check_gate(file_path, param_name):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
        
    betas = df['beta_E'].unique()
    failures = []
    
    print(f"--- Check {file_path} ---")
    report_rows = []
    
    for be in betas:
        sub = df[np.isclose(df['beta_E'], be)]
        if len(sub) > 1:
            # 1. Parameter Sweep Stability (Range)
            r = sub['E_mean_Hz'].values
            if np.mean(r) < 1e-6:
                pct_var = 0.0
            else:
                pct_var = (np.max(r) - np.min(r)) / np.mean(r)
            
            # 2. Optimum Location Shift (Argmax)
            # Find argmax I_lower per parameter value, then compare optima?
            # No, these sweeps are "sensitivity at fixed theta*"? 
            # OR "re-optimize at each lag"?
            # Prompt says "Repeat Stage 1 optimization...".
            # So `sub` contains *optimized* results for each lag/bin.
            # We want to see if the *optimized rate* shifts.
            
            # Optimum Shift = max_rate - min_rate / mean_rate (which is pct_var above)
            # AND "objective value at argmax".
            
            j_vals = sub['J'].values # Assuming J is in CSV.
            if 'J' not in sub.columns:
                 # Reconstruct J? Or just check rate?
                 # b1 output `opt_best.csv` has `J`.
                 # b4 output `sensitivity.csv` is concat of `opt_best`.
                 pass
            
            # Check deviation
            rate_status = "PASS" if pct_var <= 0.25 else "FAIL"
            
            report_rows.append({
                'beta_E': be,
                'min_rate': np.min(r),
                'max_rate': np.max(r),
                'pct_var': pct_var,
                'status': rate_status,
                'param': param_name
            })
            
            if rate_status == "FAIL": failures.append((be, pct_var))
            
    return failures, pd.DataFrame(report_rows)

f1, df1 = check_gate('runs/v4_b/tables/lag_sensitivity.csv', 'lag_taps')
f2, df2 = check_gate('runs/v4_d/tables/bin_sensitivity.csv', 'bin_dt')
# Note: v4_e is the run directory for trials mode, output file is trials_sensitivity.csv
f3, df3 = check_gate('runs/v4_e/tables/trials_sensitivity.csv', 'trials') 

full_report = pd.concat([df1, df2, df3])
print("\n=== Robustness Gate Report (v4_e, v4_f) ===")
print(full_report)
full_report.to_csv('runs/v4_b/tables/robustness_gate_report_extended.csv', index=False)
print("\n=== Robustness Gate Report (v4_e) ===")
print(full_report)
full_report.to_csv('runs/v4_b/tables/robustness_gate_report.csv', index=False) 
# Save to common location or newest run?
# Let's save to a dedicated report dir? Or just one of them.


f1 = check_gate('runs/v4_b/tables/lag_sensitivity.csv', 'lag_taps')
f2 = check_gate('runs/v4_d/tables/bin_sensitivity.csv', 'bin_dt')
print(f"Failures: Lag={f1}, Bin={f2}")
