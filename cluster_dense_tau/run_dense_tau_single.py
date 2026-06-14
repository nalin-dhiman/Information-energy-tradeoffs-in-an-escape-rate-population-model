#!/usr/bin/env python3
"""Cluster-safe dense tau_c Stage-2 optimization for one tau value.

This script avoids the automatic run-directory allocator so it is safe to use
inside a SLURM job array. Each job writes to a deterministic folder:

    runs/dense_tau_jobs/tau_<tau>_be_<beta_E>_bc_<beta_C>/
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from scripts.b5_stage2_optimization import evaluate_point_stage2  # noqa: E402
from src.io.config import load_config  # noqa: E402


PARAM_KEYS = ["theta0", "thetaV", "thetaa", "thetaVV", "thetaaa", "thetaVa"]
BASELINE_RATE = 5.0


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


def load_resolved_config(config_path):
    cfg = load_config(str(config_path))
    resolve_includes(cfg, CODE)
    return cfg


def format_float(x):
    return str(float(x)).replace(".", "p").replace("-", "m")


def start_points(tau_c):
    """Return generic starts plus known nondegenerate anchors.

    The dense sweep is meant to test the scaling claim, not to rediscover the
    trivial silent solution. These anchors come from the already audited
    four-point beta_E=1, beta_C=0.03 sweep and are sorted by proximity to tau_c.
    Generic starts are kept as fallbacks so the search can still move away from
    the anchors when the dense grid reveals a better optimum.
    """
    anchors = [
        (
            0.02,
            {
                "theta0": -2.0,
                "thetaV": 2.0,
                "thetaa": 0.0,
                "thetaVV": 0.0,
                "thetaaa": 0.0,
                "thetaVa": 0.0,
            },
        ),
        (
            0.05,
            {
                "theta0": -3.0,
                "thetaV": 5.25,
                "thetaa": 0.0,
                "thetaVV": -1.0,
                "thetaaa": 0.0,
                "thetaVa": 0.0,
            },
        ),
        (
            0.10,
            {
                "theta0": -3.25,
                "thetaV": 5.25,
                "thetaa": 0.0,
                "thetaVV": -1.0,
                "thetaaa": 0.0,
                "thetaVa": 0.0,
            },
        ),
        (
            0.20,
            {
                "theta0": -2.0,
                "thetaV": 2.0,
                "thetaa": -1.0,
                "thetaVV": 0.0,
                "thetaaa": 0.0,
                "thetaVa": 0.0,
            },
        ),
    ]
    ordered_anchors = [
        theta
        for _, theta in sorted(
            anchors,
            key=lambda item: abs(np.log(float(tau_c)) - np.log(item[0])),
        )
    ]
    generic = [
        {
            "theta0": 0.0,
            "thetaV": 5.0,
            "thetaa": 0.0,
            "thetaVV": 0.0,
            "thetaaa": 0.0,
            "thetaVa": 0.0,
        },
        {
            "theta0": -5.5,
            "thetaV": 5.0,
            "thetaa": 0.0,
            "thetaVV": 0.0,
            "thetaaa": 0.0,
            "thetaVa": 0.5,
        },
        {
            "theta0": -2.0,
            "thetaV": 5.0,
            "thetaa": 0.0,
            "thetaVV": 0.0,
            "thetaaa": 0.0,
            "thetaVa": 0.0,
        },
        {
            "theta0": 2.0,
            "thetaV": 6.0,
            "thetaa": 0.0,
            "thetaVV": 0.0,
            "thetaaa": 0.0,
            "thetaVa": 0.0,
        },
    ]
    seen = set()
    starts = []
    for theta in ordered_anchors + generic:
        key = tuple(theta[k] for k in PARAM_KEYS)
        if key not in seen:
            starts.append(theta)
            seen.add(key)
    return starts


def objective(metrics, theta, beta_e, beta_c):
    l1 = sum(abs(theta[k]) for k in PARAM_KEYS)
    return metrics["I_lower_mean"] - beta_e * (metrics["E_mean"] + BASELINE_RATE) - beta_c * l1


def evaluate(theta, cfg, tau_c, n_trials, seeds):
    return evaluate_point_stage2(
        theta,
        cfg,
        n_trials=n_trials,
        seeds=seeds,
        tau=tau_c,
    )


def optimize_one(args):
    cfg = load_resolved_config(args.config)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("stimulus", {})
    cfg["stimulus"]["tau_c"] = float(args.tau_c)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = (
        Path(args.output_root)
        / f"tau_{format_float(args.tau_c)}_be_{format_float(args.beta_e)}_bc_{format_float(args.beta_c)}"
    )
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "tau_c": args.tau_c,
        "beta_E": args.beta_e,
        "beta_C": args.beta_c,
        "n_trials": args.n_trials,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "min_step": args.min_step,
        "initial_step": args.initial_step,
        "baseline_rate": BASELINE_RATE,
        "config": str(args.config),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    best_theta = None
    best_metrics = None
    best_j = -math.inf
    history = []
    t0 = time.time()

    for restart_idx, theta0 in enumerate(start_points(args.tau_c)[: args.restarts]):
        theta = theta0.copy()
        metrics = evaluate(theta, cfg, args.tau_c, args.n_trials, seeds)
        if not metrics:
            continue
        current_j = objective(metrics, theta, args.beta_e, args.beta_c)

        step_size = args.initial_step
        for sweep in range(args.max_steps):
            candidates = []
            for key in PARAM_KEYS:
                for sign in (-1.0, 1.0):
                    trial = theta.copy()
                    trial[key] += sign * step_size
                    candidates.append(trial)

            improved = False
            sweep_best_theta = theta
            sweep_best_metrics = metrics
            sweep_best_j = current_j

            for trial in candidates:
                trial_metrics = evaluate(trial, cfg, args.tau_c, args.n_trials, seeds)
                if not trial_metrics:
                    continue
                trial_j = objective(trial_metrics, trial, args.beta_e, args.beta_c)
                if trial_j > sweep_best_j + args.tol:
                    sweep_best_theta = trial
                    sweep_best_metrics = trial_metrics
                    sweep_best_j = trial_j
                    improved = True

            theta = sweep_best_theta
            metrics = sweep_best_metrics
            current_j = sweep_best_j
            history.append(
                {
                    "restart": restart_idx,
                    "sweep": sweep,
                    "step_size": step_size,
                    "J": current_j,
                    "I_lower_mean": metrics["I_lower_mean"],
                    "I_lower_std": metrics["I_lower_std"],
                    "rate_mean": metrics["E_mean"],
                    "rate_std": metrics["E_std"],
                    **theta,
                }
            )

            if improved:
                print(
                    f"tau={args.tau_c} restart={restart_idx} sweep={sweep} "
                    f"J={current_j:.4f} rate={metrics['E_mean']:.4f}",
                    flush=True,
                )
            else:
                step_size *= 0.5
                if step_size < args.min_step:
                    break

        if current_j > best_j:
            best_theta = theta.copy()
            best_metrics = metrics
            best_j = current_j

    if best_theta is None or best_metrics is None:
        raise RuntimeError("No valid optimization result was produced.")

    bandwidth = cfg["stimulus"].get("bandwidth", cfg["stimulus"].get("cutoff_hz", 50.0))
    row = {
        **best_theta,
        "J": best_j,
        "I_lower_mean": best_metrics["I_lower_mean"],
        "I_lower_std": best_metrics["I_lower_std"],
        "I_upper_mean": best_metrics["I_upper_mean"],
        "I_upper_std": best_metrics["I_upper_std"],
        "rate_mean": best_metrics["E_mean"],
        "rate_std": best_metrics["E_std"],
        "E_mean": best_metrics["E_mean"],
        "tau_c": args.tau_c,
        "beta_E": args.beta_e,
        "beta_C": args.beta_c,
        "dt_eff": 1.0 / (2.0 * bandwidth),
        "cutoff": bandwidth,
        "tau_ms": 1000.0 * args.tau_c,
        "seed_count": len(seeds),
        "trial_count": args.n_trials,
        "elapsed_seconds": time.time() - t0,
    }

    pd.DataFrame([row]).to_csv(tables_dir / "tau_sweep_result.csv", index=False)
    pd.DataFrame(history).to_csv(tables_dir / "optimization_history.csv", index=False)

    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CODE / "configs" / "base.yaml")
    parser.add_argument("--tau_c", type=float, required=True)
    parser.add_argument("--beta_e", type=float, default=1.0)
    parser.add_argument("--beta_c", type=float, default=0.03)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--initial_step", type=float, default=0.5)
    parser.add_argument("--min_step", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "runs" / "dense_tau_jobs",
    )
    args = parser.parse_args()
    optimize_one(args)


if __name__ == "__main__":
    main()
