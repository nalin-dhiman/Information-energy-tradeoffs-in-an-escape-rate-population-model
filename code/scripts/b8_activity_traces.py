#!/usr/bin/env python3
"""Generate representative optimized response traces."""

import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.io.config import load_config  # noqa: E402
from src.simulate import run_simulation  # noqa: E402


OPT_THETA = {
    "theta0": -3.25,
    "thetaV": 5.25,
    "thetaa": 0.0,
    "thetaVV": -1.0,
    "thetaaa": 0.0,
    "thetaVa": 0.0,
}


def resolved_base_config():
    cfg = load_config(str(ROOT / "configs" / "base.yaml"))
    stim_path = ROOT / cfg["stimulus"]
    cfg["stimulus"] = load_config(str(stim_path))
    cfg["simulation"]["T"] = 4.0
    cfg["simulation"]["N"] = 2000
    cfg["simulation"]["seed"] = 7
    cfg["stimulus"]["seed"] = 7
    cfg["stimulus"]["tau_c"] = 0.1
    cfg["hazard"].update(OPT_THETA)
    return cfg


def simulate_trace(stim_type):
    cfg = resolved_base_config()
    cfg["stimulus"]["type"] = stim_type
    data = run_simulation(copy.deepcopy(cfg))
    dt = data["dt"]
    spikes = data["spikes"]
    pop_spikes = np.sum(spikes, axis=1)
    inst_rate = pop_spikes / (spikes.shape[1] * dt)
    tau_a = cfg["model"]["tau_a"]
    alpha = dt / tau_a
    smooth_rate = lfilter([alpha], [1, -(1 - alpha)], inst_rate)
    t = np.arange(len(data["S"])) * dt
    return t, data["S"], inst_rate, smooth_rate


def main():
    traces = [
        ("Band-limited Gaussian", "gauss_bandlimited"),
        ("Switching variance", "gauss_switching"),
    ]
    fig, axes = plt.subplots(
        len(traces),
        1,
        figsize=(6.8, 4.8),
        sharex=True,
        constrained_layout=True,
    )

    for ax, (title, stim_type) in zip(axes, traces):
        t, stim, inst_rate, smooth_rate = simulate_trace(stim_type)
        mask = (t >= 1.0) & (t <= 3.5)
        tt = t[mask] - t[mask][0]
        stim_z = stim[mask]
        rate = smooth_rate[mask]
        rate_norm = (rate - np.mean(rate)) / (np.std(rate) + 1e-12)

        ax.plot(tt, stim_z, color="#1f77b4", lw=1.0, label="stimulus")
        ax.plot(tt, rate_norm, color="#d62728", lw=1.0, label="population rate")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("z-score")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("time after burn-in (s)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    out = PACKAGE / "figures" / "activity_traces.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
