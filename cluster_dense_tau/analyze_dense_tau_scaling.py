#!/usr/bin/env python3
"""Merge dense tau jobs and compare scaling models."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def aicc_from_sse(n, k, sse):
    sse = max(float(sse), 1e-300)
    aic = n * np.log(sse / n) + 2 * k
    if n - k - 1 <= 0:
        return np.inf
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def fit_linear_design(x, y, design, k, name):
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    resid = y - pred
    sse = float(np.sum(resid**2))
    return {
        "model": name,
        "space": "log_rate" if "log" in name or name in ["power_law", "exponential"] else "rate",
        "k": k,
        "sse": sse,
        "aicc": aicc_from_sse(len(y), k, sse),
        "params": json.dumps([float(v) for v in beta]),
    }, pred


def saturating_power(tau, c, a, b):
    return c + a * np.power(tau, b)


def merge_results(input_root):
    files = sorted(Path(input_root).glob("tau_*/tables/tau_sweep_result.csv"))
    if not files:
        raise FileNotFoundError(f"No tau_sweep_result.csv files under {input_root}")
    frames = [pd.read_csv(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "trial_count" in df.columns:
        before = len(df)
        df = df[df["trial_count"] >= 20].copy()
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} smoke/incomplete rows with trial_count < 20.")
    if df.empty:
        raise ValueError("No full trial_count>=20 rows found for analysis.")
    df = df.sort_values(["beta_E", "beta_C", "tau_c"]).reset_index(drop=True)
    return df


def analyze_group(sub):
    sub = sub.sort_values("tau_c").copy()
    tau = sub["tau_c"].to_numpy(dtype=float)
    rate = sub["rate_mean"].to_numpy(dtype=float)
    rate_std = sub.get("rate_std", pd.Series(np.zeros(len(sub)))).to_numpy(dtype=float)

    if np.any(rate <= 0):
        raise ValueError("Power-law analysis requires positive rates.")

    log_tau = np.log(tau)
    log_rate = np.log(rate)
    n = len(sub)

    lin = stats.linregress(log_tau, log_rate)
    dof = n - 2
    t_score = stats.t.ppf(0.975, dof) if dof > 0 else np.nan
    ci_lower = lin.slope - t_score * lin.stderr if dof > 0 else np.nan
    ci_upper = lin.slope + t_score * lin.stderr if dof > 0 else np.nan

    fit_rows = []
    pred_rows = []

    row, pred_log = fit_linear_design(
        log_tau,
        log_rate,
        np.column_stack([np.ones(n), log_tau]),
        2,
        "power_law",
    )
    row.update(
        {
            "slope": lin.slope,
            "intercept": lin.intercept,
            "R2": lin.rvalue**2,
            "p_value": lin.pvalue,
            "slope_std_err": lin.stderr,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
    fit_rows.append(row)
    pred_rows.append(("power_law", np.exp(pred_log)))

    row, pred_log = fit_linear_design(
        tau,
        log_rate,
        np.column_stack([np.ones(n), tau]),
        2,
        "exponential",
    )
    fit_rows.append(row)
    pred_rows.append(("exponential", np.exp(pred_log)))

    row, pred_rate = fit_linear_design(
        log_tau,
        rate,
        np.column_stack([np.ones(n), log_tau]),
        2,
        "logarithmic_rate",
    )
    fit_rows.append(row)
    pred_rows.append(("logarithmic_rate", pred_rate))

    row, pred_rate = fit_linear_design(
        tau,
        rate,
        np.column_stack([np.ones(n), tau]),
        2,
        "linear_tau_rate",
    )
    fit_rows.append(row)
    pred_rows.append(("linear_tau_rate", pred_rate))

    if n >= 6:
        try:
            p0 = [max(0.0, float(np.min(rate)) * 0.5), float(np.max(rate)), -0.5]
            popt, _ = curve_fit(
                saturating_power,
                tau,
                rate,
                p0=p0,
                maxfev=20000,
            )
            pred = saturating_power(tau, *popt)
            sse = float(np.sum((rate - pred) ** 2))
            fit_rows.append(
                {
                    "model": "offset_power_law",
                    "space": "rate",
                    "k": 3,
                    "sse": sse,
                    "aicc": aicc_from_sse(n, 3, sse),
                    "params": json.dumps([float(v) for v in popt]),
                }
            )
            pred_rows.append(("offset_power_law", pred))
        except Exception as exc:
            fit_rows.append(
                {
                    "model": "offset_power_law",
                    "space": "rate",
                    "k": 3,
                    "sse": np.nan,
                    "aicc": np.inf,
                    "params": json.dumps({"error": str(exc)}),
                }
            )

    loo = []
    if n >= 5:
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            li = stats.linregress(log_tau[mask], log_rate[mask])
            loo.append(
                {
                    "dropped_tau_c": tau[i],
                    "slope": li.slope,
                    "R2": li.rvalue**2,
                    "p_value": li.pvalue,
                }
            )

    fit_df = pd.DataFrame(fit_rows).sort_values("aicc")
    fit_df["delta_aicc"] = fit_df["aicc"] - fit_df["aicc"].min()

    return fit_df, pd.DataFrame(loo), pred_rows, rate_std


def make_plot(sub, pred_rows, out_pdf):
    sub = sub.sort_values("tau_c")
    tau = sub["tau_c"].to_numpy(dtype=float)
    rate = sub["rate_mean"].to_numpy(dtype=float)
    yerr = sub.get("rate_std", pd.Series(np.zeros(len(sub)))).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    ax.errorbar(tau, rate, yerr=yerr, fmt="o", color="black", capsize=3, label="optimized")
    for name, pred in pred_rows:
        if name in {"power_law", "exponential", "offset_power_law"}:
            ax.plot(tau, pred, lw=1.2, label=name.replace("_", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel("optimized mean rate (Hz)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, default=Path("runs/dense_tau_jobs"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/dense_tau_analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)

    df = merge_results(args.input_root)
    df.to_csv(args.output_dir / "tables" / "dense_tau_merged.csv", index=False)

    all_fit_rows = []
    all_loo_rows = []
    summaries = []

    for (beta_e, beta_c), sub in df.groupby(["beta_E", "beta_C"]):
        fit_df, loo_df, pred_rows, _ = analyze_group(sub)
        fit_df.insert(0, "beta_C", beta_c)
        fit_df.insert(0, "beta_E", beta_e)
        all_fit_rows.append(fit_df)

        if not loo_df.empty:
            loo_df.insert(0, "beta_C", beta_c)
            loo_df.insert(0, "beta_E", beta_e)
            all_loo_rows.append(loo_df)

        tag = f"be_{str(beta_e).replace('.', 'p')}_bc_{str(beta_c).replace('.', 'p')}"
        make_plot(sub, pred_rows, args.output_dir / "figures" / f"dense_tau_scaling_{tag}.pdf")

        best = fit_df.iloc[0]
        power = fit_df[fit_df["model"] == "power_law"].iloc[0]
        summaries.append(
            {
                "beta_E": beta_e,
                "beta_C": beta_c,
                "n_tau": len(sub),
                "best_model": best["model"],
                "power_slope": power.get("slope", np.nan),
                "power_ci_lower": power.get("ci_lower", np.nan),
                "power_ci_upper": power.get("ci_upper", np.nan),
                "power_R2": power.get("R2", np.nan),
                "power_delta_aicc": power["delta_aicc"],
            }
        )

    pd.concat(all_fit_rows, ignore_index=True).to_csv(
        args.output_dir / "tables" / "model_comparison.csv",
        index=False,
    )
    if all_loo_rows:
        pd.concat(all_loo_rows, ignore_index=True).to_csv(
            args.output_dir / "tables" / "leave_one_out_powerlaw.csv",
            index=False,
        )
    pd.DataFrame(summaries).to_csv(args.output_dir / "tables" / "scaling_summary.csv", index=False)

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
