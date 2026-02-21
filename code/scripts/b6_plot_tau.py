import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

def plot_tau_sweep(csv_path):
    df = pd.read_csv(csv_path)
    run_dir = Path(csv_path).parent.parent
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    if 'rate_mean' in df.columns:
        df['E_mean'] = df['rate_mean']
    elif 'E_mean' not in df.columns:
        print("Error: No rate column found (rate_mean or E_mean)")
        return

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='tau_c', y='E_mean', hue='beta_E', style='beta_E', markers=True, palette='viridis')
    plt.xlabel('Tau C (s)')
    plt.ylabel('Optimal Rate (Hz)')
    plt.title('Optimal Rate vs Time Constant')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig(fig_dir / "rate_vs_tauc_betaE.pdf")
    plt.close()
    
  
    df['BPJ'] = df['I_lower_mean'] / (df['E_mean'] + 5.0)
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='tau_c', y='BPJ', hue='beta_E', style='beta_E', markers=True, palette='viridis')
    plt.xlabel('Tau C (s)')
    plt.ylabel('Efficiency (Bits/J)')
    plt.title('Efficiency vs Time Constant')
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / "bpj_vs_tauc_betaE.pdf")
    plt.close()
    
    if 'beta_E' in df.columns:
        betas = df['beta_E'].unique()
        target_be = betas[len(betas)//2] 
        sub = df[df['beta_E'] == target_be]
        
        if not sub.empty:
            plt.figure(figsize=(8, 6))
            if 'theta0' in sub.columns: plt.plot(sub['tau_c'], sub['theta0'], 'o-', label='Theta 0 (Input)')
            if 'thetaV' in sub.columns: plt.plot(sub['tau_c'], sub['thetaV'], 's-', label='Theta V (Thresh)')
            plt.xlabel('Tau C (s)')
            plt.ylabel('Parameter Value')
            plt.title(f'Parameter Adaptation vs Tau (BetaE={target_be})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(fig_dir / "theta_components_vs_tauc.pdf")
            plt.close()

    if 'I_upper_mean' in df.columns:
        plt.figure(figsize=(6, 6))
        
        mx = max(df['I_upper_mean'].max(), df['I_lower_mean'].max())
        plt.plot([0, mx], [0, mx], 'k--', alpha=0.5, label='Identity')
        
        sns.scatterplot(data=df, x='I_upper_mean', y='I_lower_mean', hue='tau_c', palette='coolwarm')
        plt.xlabel('Upper Bound (Gaussian Rate)')
        plt.ylabel('Lower Bound (Decoder)')
        plt.title('Estimator Consistency Audit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fig_dir / "audit_upper_vs_lower.pdf")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    args = parser.parse_args()
    
    plot_tau_sweep(args.csv_file)

if __name__ == "__main__":
    main()
