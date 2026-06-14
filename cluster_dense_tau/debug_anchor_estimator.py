#!/usr/bin/env python3
"""Debug the known tau=0.02 anchor and print estimator diagnostics."""

import argparse
import copy
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from src.estimators.mi_lower_decode import estimate_mi_lower_decode  # noqa: E402
from src.estimators.mi_upper import estimate_mi_upper_gaussian  # noqa: E402
from src.io.config import load_config  # noqa: E402
from src.simulate import run_simulation  # noqa: E402


ANCHORS = {
    0.02: {
        "theta0": -2.0,
        "thetaV": 2.0,
        "thetaa": 0.0,
        "thetaVV": 0.0,
        "thetaaa": 0.0,
        "thetaVa": 0.0,
    },
    0.05: {
        "theta0": -3.0,
        "thetaV": 5.25,
        "thetaa": 0.0,
        "thetaVV": -1.0,
        "thetaaa": 0.0,
        "thetaVa": 0.0,
    },
    0.10: {
        "theta0": -3.25,
        "thetaV": 5.25,
        "thetaa": 0.0,
        "thetaVV": -1.0,
        "thetaaa": 0.0,
        "thetaVa": 0.0,
    },
    0.20: {
        "theta0": -2.0,
        "thetaV": 2.0,
        "thetaa": -1.0,
        "thetaVV": 0.0,
        "thetaaa": 0.0,
        "thetaVa": 0.0,
    },
}


def resolve_includes(cfg, root=CODE):
    for key, value in list(cfg.items()):
        if isinstance(value, str) and value.endswith(".yaml"):
            path = Path(value)
            if not path.exists():
                path = root / value
            cfg[key] = load_config(str(path))
            resolve_includes(cfg[key], root)
        elif isinstance(value, dict):
            resolve_includes(value, root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau_c", type=float, default=0.02)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    cfg = load_config(str(CODE / "configs" / "base.yaml"))
    resolve_includes(cfg)
    cfg["stimulus"]["tau_c"] = args.tau_c

    theta = ANCHORS.get(round(args.tau_c, 2), ANCHORS[0.02]).copy()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print("theta", theta)
    print("seeds", seeds)

    lower_vals = []
    upper_vals = []
    rate_vals = []

    for seed in seeds:
        S_trials = []
        A_trials = []
        E_trials = []

        for tr in range(args.n_trials):
            trial_seed = seed * 1000 + tr
            current_cfg = copy.deepcopy(cfg)
            current_cfg["hazard"].update(theta)
            current_cfg["simulation"]["seed"] = trial_seed
            current_cfg["stimulus"]["seed"] = trial_seed

            data = run_simulation(current_cfg)
            S = data["S"]
            spikes = data["spikes"]
            dt = data["dt"]

            pop_spikes = np.sum(spikes, axis=1)
            inst_rate = pop_spikes / (spikes.shape[1] * dt)
            alpha = dt / args.tau_c
            A_smooth = lfilter([alpha], [1, -(1 - alpha)], inst_rate)

            S_trials.append(S)
            A_trials.append(A_smooth)
            E_trials.append(float(np.mean(A_smooth)))

        lcfg = cfg["estimators"]["lower"].copy()
        lcfg.update(
            {
                "split": "trial",
                "n_trials": args.n_trials,
                "seed": seed,
                "feature_mode": "rate_lags",
                "lags": int(cfg.get("decode", {}).get("lag_taps", 10)),
                "bandwidth": 50.0,
            }
        )
        print("lcfg", lcfg)

        try:
            res_l = estimate_mi_lower_decode(S_trials, A_trials, dt, lcfg)
            print("lower seed", seed, res_l)
            lower_vals.append(res_l.get("I_lower_bits_per_s", np.nan))
        except Exception:
            traceback.print_exc()
            lower_vals.append(np.nan)

        try:
            res_u = estimate_mi_upper_gaussian(
                np.concatenate(S_trials),
                np.concatenate(A_trials),
                dt,
                {**cfg["estimators"]["upper"], "bandwidth": 50.0},
            )
            print("upper seed", seed, res_u)
            upper_vals.append(res_u.get("I_upper_surrogate_bits_per_s", np.nan))
        except Exception:
            traceback.print_exc()
            upper_vals.append(np.nan)

        rate_vals.append(float(np.mean(E_trials)))

    print("SUMMARY")
    print("I_lower_mean", float(np.nanmean(lower_vals)))
    print("I_upper_mean", float(np.nanmean(upper_vals)))
    print("rate_mean", float(np.mean(rate_vals)))


if __name__ == "__main__":
    main()
