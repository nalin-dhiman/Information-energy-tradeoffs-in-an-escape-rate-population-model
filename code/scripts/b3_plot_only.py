import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--suffix', type=str, default="")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    csv_name = f"tauc_sweep{args.suffix}.csv"
    csv_path = run_dir / "tables" / csv_name
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    plot_dir = run_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    beta_e_vals = sorted(df['beta_E'].unique())
    
    plt.figure()
    for be in beta_e_vals:
        sub = df[np.isclose(df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            rate = sub['E_mean'] + 5.0 
            plt.plot(sub['tau_c'], rate, 'o-', label=f'Beta_E={be}')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Total Energy (Rate + 5Hz)')
    plt.title(f'Rate vs Tau_c {args.suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"rate_vs_tauc{args.suffix}.pdf")
    print(f"Saved {plot_dir / f'rate_vs_tauc{args.suffix}.pdf'}")

    plt.figure()
    for be in beta_e_vals:
        sub = df[np.isclose(df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            I = sub['I_lower_mean']
            E = sub['E_mean'] + 5.0
            bpj = I / E
            plt.plot(sub['tau_c'], bpj, 's-', label=f'Beta_E={be}')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Efficiency (Bits/Joule)')
    plt.title(f'Efficiency vs Tau_c {args.suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"bpj_vs_tauc{args.suffix}.pdf")
    print(f"Saved {plot_dir / f'bpj_vs_tauc{args.suffix}.pdf'}")

    plt.figure()
    for be in beta_e_vals:
        sub = df[np.isclose(df['beta_E'], be)]
        if not sub.empty:
            sub = sub.sort_values('tau_c')
            plt.plot(sub['tau_c'], sub['theta0'], '^-', label=f'Theta0 (be={be})')
            
    plt.xlabel('Tau_c (s)')
    plt.ylabel('Parameter Value')
    plt.title(f'Adaptation vs Tau_c {args.suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"theta_vs_tauc{args.suffix}.pdf")
    print(f"Saved {plot_dir / f'theta_vs_tauc{args.suffix}.pdf'}")

if __name__ == "__main__":
    main()
