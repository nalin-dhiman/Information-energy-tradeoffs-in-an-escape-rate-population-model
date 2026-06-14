#!/usr/bin/env python3
"""Plot compact numerical-control summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "data" / "results_tables"
FIG_DIR = ROOT / "figures"


def main() -> None:
    null = pd.read_csv(TABLE_DIR / "control_null_summary.csv")
    spike = pd.read_csv(TABLE_DIR / "control_spike_decoder_robustness_summary.csv")
    finite_n = pd.read_csv(TABLE_DIR / "control_finite_N_diagnostic_summary.csv")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.28])
    ax_null = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_spike = fig.add_subplot(gs[1, :])

    ax = ax_null
    x = list(range(len(null)))
    ax.axhline(0, color="0.25", linewidth=1.0)
    ax.scatter(x, null["I_dec_bits_per_s_mean"], s=34, color="#4C78A8", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in null["tau_c"]])
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel(r"$\widehat I_{\rm dec}$ (bits/s)")
    ax.set_title("Null input")
    ax.set_ylim(-0.01, 0.08)
    ax.set_yticks([0.00, 0.04, 0.08])
    ax.text(0.5, 0.70, "0.00 bits/s", transform=ax.transAxes, ha="center", va="center")
    ax.spines[["top", "right"]].set_visible(False)

    ax = ax_spike
    tau_vals = sorted(spike["tau_c"].unique())
    width = 0.32
    centers = list(range(len(tau_vals)))
    colors = {"rate_lags": "#4C78A8", "spikecount_lags": "#F58518"}
    labels = {"rate_lags": "smoothed-rate lags", "spikecount_lags": "spike-count lags"}
    for i, mode in enumerate(["rate_lags", "spikecount_lags"]):
        sub = spike[spike["feature_mode"] == mode].set_index("tau_c").loc[tau_vals]
        xpos = [c + (i - 0.5) * width for c in centers]
        ax.bar(
            xpos,
            sub["I_dec_bits_per_s_mean"],
            width=width,
            yerr=sub["I_dec_bits_per_s_std"],
            label=labels[mode],
            color=colors[mode],
            capsize=2,
        )
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{t:g}" for t in tau_vals])
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel(r"$\widehat I_{\rm dec}$ (bits/s)")
    ax.set_title("Feature robustness")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    ax.spines[["top", "right"]].set_visible(False)

    ax = ax_n
    n2000 = finite_n.loc[finite_n["N"] == 2000, "I_dec_bits_per_s_mean"].iloc[0]
    ax.errorbar(
        finite_n["N"],
        finite_n["relative_to_N2000_I_dec"],
        yerr=finite_n["I_dec_bits_per_s_std"] / n2000,
        color="#54A24B",
        marker="o",
        linewidth=1.5,
        capsize=2,
    )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks([500, 1000, 2000, 4000])
    ax.set_xlim(300, 4200)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\widehat I_{\rm dec}/\widehat I_{\rm dec}(2000)$")
    ax.set_title("Fixed-theta finite size")
    ax.set_ylim(0.5, 1.35)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    ax.spines[["top", "right"]].set_visible(False)

    for label, ax in zip(["A", "B", "C"], [ax_null, ax_spike, ax_n]):
        ax.text(
            0.02,
            0.96,
            label,
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )

    fig.savefig(FIG_DIR / "numeric_controls.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "numeric_controls.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
