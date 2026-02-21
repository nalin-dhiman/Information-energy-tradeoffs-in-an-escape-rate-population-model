
import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "font.family": "serif",
    "figure.figsize": (3.5, 2.8),
    "pdf.fonttype": 42,
    "lines.linewidth": 2,
    "lines.markersize": 4,
    "errorbar.capsize": 2
})

RUN_ROOT = Path("runs/v7_c_pubfigs")
FIG_DIR = RUN_ROOT / "figures"

def setup():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    print(f"Generating Figures in {FIG_DIR}")

def fig1_model_objective():
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.axis('off')
    
    text = (
        r"$\mathbf{Hazard\ Model}$" + "\n" +
        r"$\lambda(t) = \exp(\theta_0 + \mathbf{v}^\top \mathbf{s}(t) + \mathbf{a}^\top \mathbf{h}(t))$" + "\n\n" +
        r"$\mathbf{Dual\ Objective}$" + "\n" +
        r"$\max_{\theta} \mathcal{L}(\theta) = I(\mathbf{s}; \mathbf{r}) - \beta_E E - \beta_C C(\theta)$" + "\n\n" +
        r"$C(\theta) = \|\theta\|_1 + \|\theta_{quad}\|_1$"
    )
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10)
    plt.savefig(FIG_DIR / "fig1_model_objective.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 1 saved.")

def fig2_estimator_validation():
   
    df = pd.read_csv("results/tables/tau_sweep_results.csv")
    
    plt.figure(figsize=(3.5, 3.5))
    mx = max(df['I_upper_mean'].max(), df['I_lower_mean'].max()) * 1.1
    plt.plot([0, mx], [0, mx], 'k--', dashes=(2,2), alpha=0.6, linewidth=1)
    
   
    plt.scatter(df['I_upper_mean'], df['I_lower_mean'], c=df['tau_c'], cmap='viridis', s=20, edgecolor='k', linewidth=0.5)
    plt.colorbar(label=r'$\tau_c$ (s)')
    
    plt.xlabel('Upper Bound (Gaussian) [bits/s]')
    plt.ylabel('Lower Bound (Decoder) [bits/s]')
    plt.title('Estimator Consistency')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_estimator_validation.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 2 saved.")

def fig3_pareto_frontier():
   
    df = pd.read_csv("runs/v7_a_final_stats/tables/pareto_consolidated.csv")
    
    plt.figure()
    
    s1 = df[df['stage'] == 'Stage 1'].sort_values('mean_E_total')
    plt.errorbar(s1['mean_E_total'], s1['mean_I_lower_bits_per_s'],
                 xerr=s1['std_E_total'], yerr=s1['std_I_lower_bits_per_s'],
                 fmt='o-', color='gray', label='Stage 1', alpha=0.7, markersize=3, elinewidth=1)
                 
    s2 = df[df['stage'] == 'Stage 2'].sort_values('mean_E_total')
    
    plt.errorbar(s2['mean_E_total'], s2['mean_I_lower_bits_per_s'],
                 xerr=s2['std_E_total'], yerr=s2['std_I_lower_bits_per_s'],
                 fmt='s', color='#d62728', label='Stage 2', markersize=4, elinewidth=1)
                 
    
    uf = pd.read_csv("runs/v7_a_final_stats/tables/pareto_union_front.csv").sort_values('mean_E_total')
    plt.plot(uf['mean_E_total'], uf['mean_I_lower_bits_per_s'], 'k--', alpha=0.5, label='Frontier')

    plt.xlabel('Energy Rate (Hz)')
    plt.ylabel('Information Rate (bits/s)')
    plt.legend(loc='lower right')
    plt.title('Pareto Frontier Extension')
    plt.savefig(FIG_DIR / "fig3_pareto_frontier.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 3 saved.")

def fig4_tau_transition():
    
    df = pd.read_csv("runs/v7_a_final_stats/tables/tauc_consolidated.csv")
    fits = pd.read_csv("runs/v7_b_scaling_fit/tables/tauc_scaling_fit.csv")
    
    plt.figure()
    
    betas = sorted(df['beta_E'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
    
    for i, be in enumerate(betas):
        sub = df[df['beta_E'] == be].sort_values('tau_c')
        plt.errorbar(sub['tau_c'], sub['optimal_rate_mean'], 
                     yerr=sub['optimal_rate_std'],
                     fmt='o', color=colors[i], label=f'$\\beta_E={be}$')
        
        fit_row = fits[fits['beta_E'] == be]
        if not fit_row.empty:
            slope = fit_row.iloc[0]['slope']
            intercept = fit_row.iloc[0]['intercept']
            x = np.array([sub['tau_c'].min(), sub['tau_c'].max()])
            y = np.exp(intercept + slope * np.log(x))
            plt.plot(x, y, '-', color=colors[i], alpha=0.5)
            
            if be == 1.0:
                 plt.text(x[0]*1.2, y[0]*1.2, f"b={slope:.2f}", color=colors[i], fontsize=8)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\tau_c$ (s)')
    plt.ylabel('Optimal Rate (Hz)')
    plt.legend()
    plt.title('Timescale Adaptation')
    plt.savefig(FIG_DIR / "fig4_tau_transition.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 4 saved.")

def fig5_sparsity_structure():
   
    df = pd.read_csv("results/tables/pareto_stats.csv")
    if 'theta0' not in df.columns:
       
        pass 
        
    
    try:
        df = pd.read_csv("runs/v5_i/tables/opt_best.csv") 
        
        df = pd.read_csv("results/tables/tau_sweep_results.csv")
    except:
        print("Skipping Fig 5 (data missing)")
        return

   
    df = df[df['beta_E'] == 1.0]
    
    grp = df.groupby('tau_c')[['theta0', 'thetaV']].mean().reset_index()
    
    plt.plot(grp['tau_c'], grp['theta0'], 'o-', label=r'$\theta_0$ (Bias)')
    plt.plot(grp['tau_c'], grp['thetaV'], 's-', label=r'$\theta_V$ (Gain)')
    
    plt.xlabel(r'$\tau_c$ (s)')
    plt.ylabel('Parameter Value')
    plt.xscale('log')
    plt.legend()
    plt.title(r'Structure vs $\tau_c$ ($\beta_E=1$)')
    plt.savefig(FIG_DIR / "fig5_sparsity_structure.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 5 saved.")

def fig6_robustness():
   
    g_df = pd.read_csv("runs/v7_a_final_stats/tables/tauc_consolidated.csv")
    
    s_path = Path("runs/v6_i/tables/tau_sweep_results.csv")
    if not s_path.exists():
        print("Warning: Switching data not found for Fig 6.")
        return

    s_df = pd.read_csv(s_path)
    
    if 'rate_mean' not in s_df.columns: s_df['rate_mean'] = s_df['E_mean']
    s_df['BPJ'] = s_df['I_lower_mean'] / (s_df['rate_mean'] + 5.0)
    
    be = 1.0
    g_sub = g_df[g_df['beta_E'] == be]
    s_sub = s_df[s_df['beta_E'] == be] 
    
    plt.figure()
    
    plt.errorbar(g_sub['tau_c'], g_sub['mean_bits_per_J'], 
                 yerr=g_sub['std_bits_per_J'], fmt='o-', label='Gaussian')
                 
    plt.plot(s_sub['tau_c'], s_sub['BPJ'], 's--', label='Switching') 
    plt.xlabel(r'$\tau_c$ (s)')
    plt.ylabel('Efficiency (bits/J)')
    plt.title(r'Robustness ($\beta_E=1$)')
    plt.legend()
    plt.savefig(FIG_DIR / "fig6_robustness.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 6 saved.")

def fig7_ablations():
    root = Path("results/tables")
    bc0 = pd.read_csv(root / "ablation_betaC0.csv")
    be0 = pd.read_csv(root / "ablation_betaE0.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    
    
    param_keys = ['theta0', 'thetaV', 'thetaa', 'thetaVV', 'thetaaa', 'thetaVa']
    
    bc0['BPJ'] = bc0['I_lower_mean'] / (bc0['rate_mean'] + 5.0)
    
    axes[0].plot(bc0['beta_E'], bc0['BPJ'], 'o-', label=r'$\beta_C=0$')
    axes[0].set_xlabel(r'$\beta_E$')
    axes[0].set_ylabel('Efficiency (bits/J)')
    axes[0].set_title('Effect of Sparsity')
    axes[0].legend()
    
   
    axes[1].plot(be0['tau_c'], be0['rate_mean'], 's-', color='orange', label=r'$\beta_E=0$')
    axes[1].set_xlabel(r'$\tau_c$ (s)')
    axes[1].set_ylabel('Rate (Hz)')
    axes[1].set_title('Unconstrained Rate')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_ablations.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 7 saved.")

def main():
    setup()
    fig1_model_objective()
    fig2_estimator_validation()
    fig3_pareto_frontier()
    fig4_tau_transition()
    fig5_sparsity_structure()
    fig6_robustness()
    fig7_ablations()
    
    with open(RUN_ROOT / "summary.txt", "w") as f:
        f.write("Paper B Publication Figures Generated.\n")
        f.write("All strict style requirements applied.\n")

if __name__ == "__main__":
    main()
