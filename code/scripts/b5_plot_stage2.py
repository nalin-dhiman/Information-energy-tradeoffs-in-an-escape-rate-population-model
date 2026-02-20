import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

def load_stage1_proxy():
    # Use v4_b lag=5 results as robust Stage 1 proxy
    path = Path("runs/v4_b/tables/lag_sensitivity.csv")
    if not path.exists():
        print(f"Warning: Stage 1 proxy {path} not found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Filter for lag=5 assumption or just take all?
    # v4_b swept lags. Let's take lag_taps=5 subset if available.
    if 'lag_taps' in df.columns:
        df = df[df['lag_taps'] == 5]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage2_file', default='runs/v5_c/tables/stage2_best.csv')
    args = parser.parse_args()
    
    df_s2 = pd.read_csv(args.stage2_file)
    df_s1 = load_stage1_proxy()
    
    fig_dir = Path(args.stage2_file).parent.parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Pareto Overlay
    plt.figure(figsize=(8, 6))
    if not df_s1.empty:
        plt.scatter(df_s1['E_mean_Hz'], df_s1['I_lower_mean_bits_per_s'], 
                    label='Stage 1 (Robust Proxy)', color='gray', alpha=0.5, s=30)
    
    # Stage 2
    # Check columns
    e_col = 'E_mean_Hz' if 'E_mean_Hz' in df_s2.columns else 'E_mean'
    i_col = 'I_lower_mean_bits_per_s' if 'I_lower_mean_bits_per_s' in df_s2.columns else 'I_lower_mean'
    
    sc = plt.scatter(df_s2[e_col], df_s2[i_col], 
                     c=df_s2['beta_C'], cmap='viridis', label='Stage 2 (Quadratic)', s=60, edgecolors='k')
    plt.colorbar(sc, label='Beta_C (L1 Penalty)')
    
    plt.xlabel('Energy Rate (Hz)')
    plt.ylabel('Information Rate (bits/s)')
    plt.title('Stage 1 vs Stage 2 Pareto Front')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_dir / "pareto_stage2.pdf")
    plt.close()
    
    # 2. Sparsity Analysis
    # Plot L1 norm of quadratic terms vs Beta_C
    # Or just count non-zero params?
    # Let's plot L1_theta vs Beta_C
    plt.figure(figsize=(8, 6))
    sns.stripplot(data=df_s2, x='beta_C', y='L1_theta', hue='beta_E', palette='coolwarm')
    plt.xlabel('Beta_C')
    plt.ylabel('L1 Norm of Parameters')
    plt.title('Parameter Sparsity vs Regularization')
    plt.savefig(fig_dir / "sparsity_vs_betaC.pdf")
    plt.close()
    
    print(f"Plots saved to {fig_dir}")

if __name__ == "__main__":
    main()
