#!/usr/bin/env python3
"""Analyze optimization initialization, baseline cost, and model sparsity."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "results_tables"
RUN_TABLES = ROOT / "data" / "run_tables"
FIGURES = ROOT / "figures"


def summarize_optimizer_restarts():
    detail = pd.read_csv(RESULTS / "optimizer_restart_detail.csv")
    rows = []
    for tau_c, group in detail.groupby("tau_c"):
        warm = group[group["initialization"] == "warm_start"]
        direct = group[group["initialization"] == "direct_full_parameter"]
        warm_matched = warm[warm["budget_group"] == "matched"]
        rows.append(
            {
                "tau_c": tau_c,
                "warm_best_J_all4": warm["final_J"].max(),
                "warm_best_J_matched2": warm_matched["final_J"].max(),
                "direct_best_J_matched2": direct["final_J"].max(),
                "delta_J_all4_minus_direct2": (
                    warm["final_J"].max() - direct["final_J"].max()
                ),
                "delta_J_matched2_minus_direct2": (
                    warm_matched["final_J"].max() - direct["final_J"].max()
                ),
                "warm_final_J_range": warm["final_J"].max()
                - warm["final_J"].min(),
                "direct_final_J_range": direct["final_J"].max()
                - direct["final_J"].min(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("tau_c")
    summary.to_csv(RESULTS / "optimizer_initialization_comparison.csv", index=False)
    return detail, summary


def calculate_baseline_cost_sensitivity():
    gaussian = pd.read_csv(RESULTS / "dense_tau_merged.csv").rename(
        columns={"I_lower_mean": "I_dec", "E_mean": "rate"}
    )
    switching = pd.read_csv(RUN_TABLES / "tau_sweep_switching.csv")
    switching = switching[
        np.isclose(switching["beta_E"], 1.0)
        & np.isclose(switching["beta_C"], 0.03)
    ].rename(columns={"I_lower_mean": "I_dec", "E_mean": "rate"})

    rows = []
    for stimulus, frame in (("gaussian", gaussian), ("switching", switching)):
        frame = frame.sort_values("tau_c")
        reference = frame["I_dec"] / (frame["rate"] + 5.0)
        for baseline_rate in (0.0, 2.5, 5.0, 10.0):
            efficiency = frame["I_dec"] / (frame["rate"] + baseline_rate)
            rank_tau, _ = kendalltau(reference, efficiency)
            for tau_c, value in zip(frame["tau_c"], efficiency):
                rows.append(
                    {
                        "stimulus": stimulus,
                        "tau_c": tau_c,
                        "baseline_rate_Hz": baseline_rate,
                        "efficiency_bits_per_spike_equivalent": value,
                        "kendall_tau_vs_baseline_5_Hz": rank_tau,
                    }
                )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(RESULTS / "baseline_cost_sensitivity.csv", index=False)
    return sensitivity


def summarize_quadratic_selection():
    pareto = pd.read_csv(RUN_TABLES / "stage2_best_with_theta.csv")
    dense = pd.read_csv(RESULTS / "dense_tau_merged.csv")
    rows = []
    for source, frame in (("penalty_grid", pareto), ("dense_tau", dense)):
        for coefficient in ("thetaVV", "thetaaa", "thetaVa"):
            selected = ~np.isclose(frame[coefficient], 0.0)
            rows.append(
                {
                    "source": source,
                    "coefficient": coefficient,
                    "selected_count": int(selected.sum()),
                    "total_count": int(len(frame)),
                    "selected_fraction": float(selected.mean()),
                }
            )

    selection = pd.DataFrame(rows)
    selection.to_csv(RESULTS / "quadratic_parameter_selection.csv", index=False)
    return selection


def plot_diagnostics(comparison, sensitivity, selection):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), constrained_layout=True)

    ax = axes[0]
    gaussian = sensitivity[sensitivity["stimulus"] == "gaussian"]
    colors = {0.0: "#4C78A8", 2.5: "#59A14F", 5.0: "#E45756", 10.0: "#B279A2"}
    for baseline_rate, group in gaussian.groupby("baseline_rate_Hz"):
        ax.plot(
            group["tau_c"],
            group["efficiency_bits_per_spike_equivalent"],
            marker="o",
            markersize=2.8,
            linewidth=1.1,
            color=colors[baseline_rate],
            label=fr"$r_0={baseline_rate:g}$ Hz",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel(r"$\eta$ (bits/spike-equivalent)")
    ax.set_title("Baseline-cost sensitivity")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[1]
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.plot(
        comparison["tau_c"],
        comparison["delta_J_matched2_minus_direct2"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        color="#F28E2B",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel(r"best $J_{\rm warm}-J_{\rm direct}$")
    ax.set_title("Matched restart budget")

    ax = axes[2]
    subset = selection[selection["source"] == "penalty_grid"]
    labels = {
        "thetaVV": r"$\theta_{VV}$",
        "thetaaa": r"$\theta_{AA}$",
        "thetaVa": r"$\theta_{VA}$",
    }
    ax.bar(
        [labels[value] for value in subset["coefficient"]],
        subset["selected_fraction"],
        color=["#76B7B2", "#EDC948", "#E15759"],
        width=0.62,
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction selected")
    ax.set_title("Quadratic-term selection")

    for label, ax in zip("ABC", axes):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "optimization_diagnostics.pdf")
    fig.savefig(FIGURES / "optimization_diagnostics.png", dpi=300)
    plt.close(fig)


def main():
    detail, comparison = summarize_optimizer_restarts()
    sensitivity = calculate_baseline_cost_sensitivity()
    selection = summarize_quadratic_selection()
    plot_diagnostics(comparison, sensitivity, selection)

    matched_wins = int((comparison["delta_J_matched2_minus_direct2"] > 0).sum())
    all_wins = int((comparison["delta_J_all4_minus_direct2"] > 0).sum())
    print(f"Matched 2-vs-2 warm-start wins: {matched_wins}/{len(comparison)}")
    print(f"All archived warm-start wins: {all_wins}/{len(comparison)}")
    print(
        "Median all-start objective advantage: "
        f"{comparison['delta_J_all4_minus_direct2'].median():.3f}"
    )
    print(f"Restart records analyzed: {len(detail)}")


if __name__ == "__main__":
    main()
