import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Path to run directory (e.g. runs/v2_g)')
    args = parser.parse_args()
    
    run_path = Path(args.run_dir)
    tables_dir = run_path / "tables"
    figures_dir = run_path / "figures"
    
    opt_best_path = tables_dir / "opt_best.csv"
    if not opt_best_path.exists():
        print(f"Error: {opt_best_path} not found.")
        return
        
    df = pd.read_csv(opt_best_path)
    
    # Filter for beta_C = 0 or 0.01? 
    # User might want to see effect of Beta_E for fixed Beta_C.
    # Color lines by Beta_C.
    
    beta_c_vals = df['beta_C'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Theta 0
    ax = axes[0]
    for bc in beta_c_vals:
        sub = df[df['beta_C'] == bc].sort_values('beta_E')
        ax.plot(sub['beta_E'], sub['theta0'], 'o-', label=f'beta_C={bc}')
    ax.set_xlabel('Beta_E')
    ax.set_ylabel('Theta 0 (Mean Input)')
    ax.set_title('Theta 0 vs Beta_E')
    ax.legend()
    
    # Theta V
    ax = axes[1]
    for bc in beta_c_vals:
        sub = df[df['beta_C'] == bc].sort_values('beta_E')
        ax.plot(sub['beta_E'], sub['thetaV'], 's-', label=f'beta_C={bc}')
    ax.set_xlabel('Beta_E')
    ax.set_ylabel('Theta V (Threshold)')
    ax.set_title('Theta V vs Beta_E')
    
    # Theta a
    ax = axes[2]
    for bc in beta_c_vals:
        sub = df[df['beta_C'] == bc].sort_values('beta_E')
        ax.plot(sub['beta_E'], sub['thetaa'], '^-', label=f'beta_C={bc}')
    ax.set_xlabel('Beta_E')
    ax.set_ylabel('Theta a (Adaptation)')
    ax.set_title('Theta a vs Beta_E')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "theta_vs_betaE.pdf")
    print(f"Saved {figures_dir / 'theta_vs_betaE.pdf'}")

if __name__ == "__main__":
    main()
