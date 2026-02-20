import sys
import os
import argparse
from pathlib import Path

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.b6_tau_sweep_stage2 import run_tau_sweep_stage2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    args = parser.parse_args()
    
    print("=== Running Ablation: BetaC=0 (No Sparsity) ===")
    # Create args object
    class Args:
        config = args.config
        mode = 'ablation'
        tau_c = [0.02, 0.1] # Subset as requested
        beta_e = [1.0] # Typical value
        beta_c = [0.0] # Force 0
    
    run_tau_sweep_stage2(Args())
    
    print("\n=== Running Ablation: BetaE=0 (No Energy Penalty) ===")
    class ArgsE:
        config = args.config
        mode = 'ablation'
        tau_c = [0.02] # Just one point needed to show collapse
        beta_e = [0.0] # Force 0
        beta_c = [0.03] # Normal penalty
        
    run_tau_sweep_stage2(ArgsE())

if __name__ == "__main__":
    main()
