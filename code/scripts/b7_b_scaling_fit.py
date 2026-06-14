
import sys
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

def run_scaling_fit():
    run_root = Path("runs/v7_b_scaling_fit")
    run_root.mkdir(parents=True, exist_ok=True)
    tables_dir = run_root / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    print(f"Starting Phase 22 (Scaling Fit): {run_root}")
    
    # Input
    infile = Path("runs/v7_a_final_stats/tables/tauc_consolidated.csv")
    if not infile.exists():
        print("CRITICAL: Consolidated tauc stats not found.")
        sys.exit(1)
        
    df = pd.read_csv(infile)
    
    results = []
    
    # Iterate BetaE
    betas = sorted(df['beta_E'].unique())
    
    for be in betas:
        sub = df[df['beta_E'] == be].sort_values('tau_c')
        # Need at least 3 points for a decent fit
        if len(sub) < 3:
            print(f"Skipping BetaE={be}: insufficient points ({len(sub)})")
            continue
            
        x = np.log(sub['tau_c']) # Natural log? or log10? Slopes are scaling exponents, usually natural or base10 doesn't matter for exponent value (b), but standard is ln or log10. ln is safer for scipy.
        # Power law r = A * tau^b => log(r) = log(A) + b * log(tau)
        y = np.log(sub['optimal_rate_mean'])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 95% CI for slope
        # t-score for 95% CI with n-2 degrees of freedom
        dof = len(x) - 2
        t_score = stats.t.ppf(0.975, dof)
        ci_lower = slope - t_score * std_err
        ci_upper = slope + t_score * std_err
        
        res = {
            'beta_E': be,
            'slope': slope,
            'intercept': intercept,
            'R2': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pass': (slope < -0.1) and (p_value < 0.05) # "Significantly negative" check logic
        }
        results.append(res)
        print(f"BetaE={be}: Slope={slope:.3f} (CI: [{ci_lower:.3f}, {ci_upper:.3f}]), R2={r_value**2:.3f}")
        
    out = pd.DataFrame(results)
    out.to_csv(tables_dir / "tauc_scaling_fit.csv", index=False)
    print("Saved fit results.")
    
    # Append to summary
    summary_path = Path("runs/v7_a_final_stats/summary.txt")
    if summary_path.exists():
        with open(summary_path, "a") as f:
            f.write("\nScaling Fits (v7_b)\n")
            f.write("-------------------\n")
            for idx, r in out.iterrows():
                f.write(f"BetaE={r['beta_E']}: Slope={r['slope']:.3f} (R2={r['R2']:.3f})\n")

if __name__ == "__main__":
    run_scaling_fit()
