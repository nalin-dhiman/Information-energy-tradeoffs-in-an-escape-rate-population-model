#!/usr/bin/env python3
"""Plot optimized rate across the dense stimulus-timescale sweep in milliseconds."""

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
        1000.0 * data["tau_c"],
        data["rate_mean"],
        yerr=data["rate_std"],
        color="#1F4E79",
        marker="o",
        markersize=4,
        linewidth=1.25,
        capsize=2,
        label="re-optimized rate",
    )
    ax.axvline(20.0, color="#E15759", linestyle="--", linewidth=1.0)
    ax.axvline(100.0, color="#59A14F", linestyle=":", linewidth=1.2)
    ax.annotate(
        r"$\tau_m$",
        xy=(20.0, 9.02),
        xytext=(5, 0),
        textcoords="offset points",
        color="#E15759",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$\tau_a$",
        xy=(100.0, 9.02),
        xytext=(5, 0),
        textcoords="offset points",
        color="#3C7D38",
        ha="left",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_xlim(8.5, 580.0)
    ax.set_ylim(1.6, 9.55)
    ax.set_xlabel(r"stimulus correlation timescale $\tau_c$ (ms)")
    ax.set_ylabel("optimized mean rate (Hz)")
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0,
        fontsize=7,
    )
    ax.spines[["top", "right"]].set_visible(False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "tau_transition.pdf")
    fig.savefig(FIGURES / "tau_transition.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
