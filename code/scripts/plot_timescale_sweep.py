#!/usr/bin/env python3
"""Plot optimized rate across the dense stimulus-timescale sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "results_tables" / "dense_tau_merged.csv"
FIGURES = ROOT / "figures"


def main():
    data = pd.read_csv(DATA).sort_values("tau_c")
    fig, ax = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)

    ax.errorbar(
        data["tau_c"],
        data["rate_mean"],
        yerr=data["rate_std"],
        color="#1F4E79",
        marker="o",
        markersize=4,
        linewidth=1.25,
        capsize=2,
        label="re-optimized rate",
    )
    ax.axvline(0.02, color="#E15759", linestyle="--", linewidth=1.0)
    ax.axvline(0.10, color="#59A14F", linestyle=":", linewidth=1.2)
    ax.text(0.02, 9.02, r"$\tau_m$", color="#E15759", ha="center", va="bottom")
    ax.text(0.10, 9.02, r"$\tau_a$", color="#3C7D38", ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(0.0085, 0.58)
    ax.set_ylim(1.6, 9.55)
    ax.set_xlabel(r"stimulus correlation time $\tau_c$ (s)")
    ax.set_ylabel("optimized mean rate (Hz)")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "tau_transition.pdf")
    fig.savefig(FIGURES / "tau_transition.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
