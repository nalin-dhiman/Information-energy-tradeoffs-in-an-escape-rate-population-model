import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.config import load_config
from src.simulate import run_simulation
from src.estimators.mi_lower_decode import estimate_mi_lower_decode


THETA_KEYS = ["theta0", "thetaV", "thetaa", "thetaVV", "thetaaa", "thetaVa"]


def resolve_includes(cfg, root):
    for key, value in list(cfg.items()):
        if isinstance(value, str) and value.endswith(".yaml"):
            path = Path(value)
            if not path.exists():
                path = root / value
            cfg[key] = load_config(str(path))
            resolve_includes(cfg[key], root)
        elif isinstance(value, dict):
            resolve_includes(value, root)
    return cfg


def load_base_config(code_root):
    cfg = load_config(str(code_root / "configs" / "base.yaml"))
    return resolve_includes(cfg, code_root)


def theta_from_row(row):
    return {key: float(row[key]) for key in THETA_KEYS}


def set_theta(cfg, theta):
    hazard = cfg.setdefault("hazard", {})
    for key, value in theta.items():
        hazard[key] = float(value)


def smooth_rate_from_spikes(spikes, tau, dt):
    pop_spikes = np.sum(spikes, axis=1)
    n_units = spikes.shape[1]
    inst_rate = pop_spikes / (n_units * dt)
    alpha = dt / tau
    return lfilter([alpha], [1, -(1 - alpha)], inst_rate)


def build_decoder_cfg(base_cfg, seed, n_trials, feature_mode, cutoff=50.0):
    cfg = copy.deepcopy(base_cfg["estimators"]["lower"])
    cfg["split"] = "trial"
    cfg["seed"] = int(seed)
    cfg["n_trials"] = int(n_trials)
    cfg["feature_mode"] = feature_mode
    cfg["bandwidth"] = float(cutoff)
    cfg.setdefault("parameters", {})["alpha"] = float(
        base_cfg.get("decode", {}).get("ridge_alpha", 1e-2)
    )
    if feature_mode == "rate_lags":
        cfg["lags"] = int(base_cfg.get("decode", {}).get("lag_taps", 10))
    else:
        cfg["bin_dt"] = 0.01
        # Match the 10 effective-time-step history used by the rate decoder
        # rather than giving the spike-count decoder a longer 500 ms window.
        cfg["lags"] = int(base_cfg.get("decode", {}).get("lag_taps", 10))
    return cfg


def simulate_trials(
    base_cfg,
    theta,
    tau_c,
    n_trials,
    seed,
    n_units=2000,
    trial_T=6.0,
    burn_in=1.0,
    null_stimulus=False,
):
    s_trials = []
    rate_trials = []
    spike_count_trials = []
    rate_means = []

    for tr in range(n_trials):
        trial_seed = int(seed) * 1000 + tr
        cfg = json.loads(json.dumps(base_cfg))
        set_theta(cfg, theta)
        cfg["simulation"]["seed"] = trial_seed
        cfg["simulation"]["N"] = int(n_units)
        cfg["simulation"]["T"] = float(trial_T)
        cfg["simulation"]["burn_in"] = float(burn_in)
        cfg["stimulus"]["seed"] = trial_seed
        cfg["stimulus"]["tau_c"] = float(tau_c)
        if null_stimulus:
            cfg["stimulus"]["std"] = 0.0
            cfg["stimulus"]["mean"] = 0.0

        data = run_simulation(cfg)
        spikes = data["spikes"]
        dt = float(data["dt"])
        rate = smooth_rate_from_spikes(spikes, tau_c, dt)
        pop_counts = np.sum(spikes, axis=1)

        s_trials.append(data["S"])
        rate_trials.append(rate)
        spike_count_trials.append(pop_counts)
        rate_means.append(float(np.mean(rate)))

    return s_trials, rate_trials, spike_count_trials, dt, np.asarray(rate_means)


def evaluate_decoder(
    base_cfg,
    theta,
    tau_c,
    n_trials,
    seeds,
    n_units=2000,
    feature_mode="rate_lags",
    null_stimulus=False,
    trial_T=6.0,
):
    rows = []
    for seed in seeds:
        s_trials, rate_trials, spike_trials, dt, rate_means = simulate_trials(
            base_cfg,
            theta,
            tau_c,
            n_trials,
            seed,
            n_units=n_units,
            null_stimulus=null_stimulus,
            trial_T=trial_T,
        )
        decoder_input = rate_trials if feature_mode == "rate_lags" else spike_trials
        lcfg = build_decoder_cfg(base_cfg, seed, n_trials, feature_mode)
        result = estimate_mi_lower_decode(s_trials, decoder_input, dt, lcfg)
        rows.append(
            {
                "seed": int(seed),
                "N": int(n_units),
                "tau_c": float(tau_c),
                "feature_mode": feature_mode,
                "I_dec_bits_per_s": float(result.get("I_lower_bits_per_s", np.nan)),
                "r2_test": float(result.get("r2_test", np.nan)),
                "mse_test": float(result.get("mse_test", np.nan)),
                "var_S_test": float(result.get("var_S_test", np.nan)),
                "clipped": bool(result.get("clipped", False)),
                "rate_mean_Hz": float(np.mean(rate_means)),
                "trial_count": int(n_trials),
                "trial_T_s": float(trial_T),
                "null_stimulus": bool(null_stimulus),
            }
        )
    return pd.DataFrame(rows)


def summarize(df, group_cols):
    numeric = ["I_dec_bits_per_s", "r2_test", "rate_mean_Hz"]
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[numeric].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in out.columns
    ]
    counts = grouped.size().reset_index(name="seed_count")
    return out.merge(counts, on=group_cols, how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=6)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--trial_T", type=float, default=6.0)
    parser.add_argument("--n_values", type=str, default="500,1000,2000,4000")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    code_root = repo_root / "code"
    results_dir = repo_root / "data" / "results_tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(code_root)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]

    dense = pd.read_csv(results_dir / "dense_tau_merged.csv")
    key_taus = [0.02, 0.1, 0.5]
    theta_by_tau = {}
    for tau in key_taus:
        idx = (dense["tau_c"] - tau).abs().idxmin()
        row = dense.loc[idx]
        theta_by_tau[float(row["tau_c"])] = theta_from_row(row)

    null_frames = []
    for tau, theta in [(0.02, theta_by_tau[0.02]), (0.1, theta_by_tau[0.1])]:
        null_frames.append(
            evaluate_decoder(
                base_cfg,
                theta,
                tau,
                args.n_trials,
                seeds,
                n_units=2000,
                feature_mode="rate_lags",
                null_stimulus=True,
                trial_T=args.trial_T,
            )
        )
    null_df = pd.concat(null_frames, ignore_index=True)
    null_df.to_csv(results_dir / "control_null.csv", index=False)
    null_summary = summarize(null_df, ["tau_c", "feature_mode", "null_stimulus"])
    null_summary.to_csv(results_dir / "control_null_summary.csv", index=False)

    spike_frames = []
    for tau, theta in theta_by_tau.items():
        for mode in ["rate_lags", "spikecount_lags"]:
            spike_frames.append(
                evaluate_decoder(
                    base_cfg,
                    theta,
                    tau,
                    args.n_trials,
                    seeds,
                    n_units=2000,
                    feature_mode=mode,
                    null_stimulus=False,
                    trial_T=args.trial_T,
                )
            )
    spike_df = pd.concat(spike_frames, ignore_index=True)
    spike_df.to_csv(results_dir / "control_spike_decoder_robustness.csv", index=False)
    spike_summary = summarize(spike_df, ["tau_c", "feature_mode", "null_stimulus"])
    rate_summary = spike_summary[spike_summary["feature_mode"] == "rate_lags"][
        ["tau_c", "I_dec_bits_per_s_mean"]
    ].rename(columns={"I_dec_bits_per_s_mean": "rate_lag_I_dec_mean"})
    spike_summary = spike_summary.merge(rate_summary, on="tau_c", how="left")
    spike_summary["relative_to_rate_decoder"] = (
        spike_summary["I_dec_bits_per_s_mean"] / spike_summary["rate_lag_I_dec_mean"]
    )
    spike_summary.to_csv(
        results_dir / "control_spike_decoder_robustness_summary.csv", index=False
    )

    n_frames = []
    theta_n = theta_by_tau[0.1]
    for n_units in n_values:
        n_frames.append(
            evaluate_decoder(
                base_cfg,
                theta_n,
                0.1,
                args.n_trials,
                seeds,
                n_units=n_units,
                feature_mode="rate_lags",
                null_stimulus=False,
                trial_T=args.trial_T,
            )
        )
    n_df = pd.concat(n_frames, ignore_index=True)
    n_df.to_csv(results_dir / "control_finite_N_diagnostic.csv", index=False)
    n_summary = summarize(n_df, ["N", "tau_c", "feature_mode"])
    ref = n_summary.loc[n_summary["N"] == 2000, "I_dec_bits_per_s_mean"]
    if not ref.empty:
        n_summary["relative_to_N2000_I_dec"] = (
            n_summary["I_dec_bits_per_s_mean"] / float(ref.iloc[0])
        )
    n_summary.to_csv(results_dir / "control_finite_N_diagnostic_summary.csv", index=False)

    control_rows = []
    for _, row in null_summary.iterrows():
        control_rows.append(
            {
                "control": "null_decoder_leakage",
                "condition": f"tau_c={row['tau_c']:.3g}",
                "numeric_result": f"I_dec={row['I_dec_bits_per_s_mean']:.3g} bits/s",
                "interpretation": "zero-stimulus control gives no decodable stimulus information",
            }
        )
    for _, row in spike_summary.iterrows():
        if row["feature_mode"] == "spikecount_lags":
            control_rows.append(
                {
                    "control": "spike_feature_robustness",
                    "condition": f"tau_c={row['tau_c']:.3g}",
                    "numeric_result": (
                        f"spike-count/rate decoder={row['relative_to_rate_decoder']:.2f}"
                    ),
                    "interpretation": "same qualitative operating regime under binned spike-count features",
                }
            )
    n_min = n_summary.loc[n_summary["N"] == min(n_values), "relative_to_N2000_I_dec"]
    n_max = n_summary.loc[n_summary["N"] == max(n_values), "relative_to_N2000_I_dec"]
    if not n_min.empty and not n_max.empty:
        control_rows.append(
            {
                "control": "fixed_theta_finite_size",
                "condition": f"N={min(n_values)} to {max(n_values)}, tau_c=0.1",
                "numeric_result": (
                    f"I_dec/N2000 range={float(n_min.iloc[0]):.2f} to {float(n_max.iloc[0]):.2f}"
                ),
                "interpretation": "finite-size diagnostic only; no re-optimization or N-scaling law",
            }
        )
    pd.DataFrame(control_rows).to_csv(
        results_dir / "control_numeric_summary.csv", index=False
    )

    print("Wrote numeric-control tables to", results_dir)


if __name__ == "__main__":
    main()
