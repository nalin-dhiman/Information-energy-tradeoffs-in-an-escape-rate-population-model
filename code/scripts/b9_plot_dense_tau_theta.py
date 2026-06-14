#!/usr/bin/env python3
"""Plot optimized hazard parameters across stimulus time scale."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "results_tables" / "dense_tau_merged.csv"
OUTPUT = ROOT / "figures" / "theta_components_vs_tauc.pdf"


def main():
    df = pd.read_csv(INPUT).sort_values("tau_c")
    params = [
        ("theta0", r"$\theta_0$"),
        ("thetaV", r"$\theta_V$"),
        ("thetaa", r"$\theta_a$"),
        ("thetaVV", r"$\theta_{VV}$"),
        ("thetaaa", r"$\theta_{aa}$"),
        ("thetaVa", r"$\theta_{Va}$"),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    markers = ["o", "s", "^", "v", "D", "P"]
    for marker, (column, label) in zip(markers, params):
        if column in df:
            ax.plot(
                df["tau_c"],
                df[column],
                marker=marker,
                lw=1.1,
                ms=4,
                label=label,
            )

    ax.axhline(0.0, color="0.35", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel("optimized parameter value")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
