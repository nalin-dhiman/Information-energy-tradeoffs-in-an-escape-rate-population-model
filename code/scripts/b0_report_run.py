

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io.paths import get_latest_run_dir


def _most_recent_stage0_run(root: Path) -> Optional[Path]:
    """Find the most recently modified run directory that contains sweep_theta0.csv."""
    runs_root = root / "runs"
    if not runs_root.exists():
        return None

    run_dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in run_dirs:
        if (d / "tables" / "sweep_theta0.csv").exists():
            return d
    return None


def _resolve_run_dir(run_id: Optional[str]) -> Tuple[Path, Path, str]:
    
    note = ""

    if run_id:
        candidate = ROOT / "runs" / run_id
        sweep = candidate / "tables" / "sweep_theta0.csv"
        if not sweep.exists():
            raise FileNotFoundError(f"Run {run_id} does not contain {sweep}.")
        return candidate, sweep, note

    run_dir = get_latest_run_dir(ROOT)
    sweep_csv = run_dir / "tables" / "sweep_theta0.csv"

    if sweep_csv.exists():
        return run_dir, sweep_csv, note

    fallback = _most_recent_stage0_run(ROOT)
    if fallback is None:
        raise FileNotFoundError(
            f"LATEST points to {run_dir}, but no run contains a Stage-0 sweep table. "
            "Run scripts/b0_sanity_sweep_theta0.py first."
        )

    note = f"(Fallback) LATEST points to {run_dir.name}; using most recent Stage-0 run {fallback.name}."
    return fallback, fallback / "tables" / "sweep_theta0.csv", note


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tau_ms" not in df.columns and "tau" in df.columns:
        df["tau_ms"] = 1000.0 * df["tau"]

    if "bits_per_joule_upper" not in df.columns and "bpj_upper" in df.columns:
        df["bits_per_joule_upper"] = df["bpj_upper"]
    if "bits_per_joule_lower" not in df.columns and "bpj_lower" in df.columns:
        df["bits_per_joule_lower"] = df["bpj_lower"]

    if "bits_per_joule_upper" not in df.columns and {"I_upper", "mean_rate"}.issubset(df.columns):
        df["bits_per_joule_upper"] = df["I_upper"] / (df["mean_rate"] + 1e-12)
    if "bits_per_joule_lower" not in df.columns and {"I_lower", "mean_rate"}.issubset(df.columns):
        df["bits_per_joule_lower"] = df["I_lower"] / (df["mean_rate"] + 1e-12)

    return df


def _gates(df: pd.DataFrame) -> list[str]:
    failures = []

    needed = ["theta0", "tau_ms", "mean_rate", "I_upper", "bits_per_joule_upper"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        failures.append(f"Missing required columns: {missing}")
        return failures

    if df[needed].isna().any().any():
        failures.append("NaNs present in required columns")

    if (df["mean_rate"] < 0).any():
        failures.append("Negative mean_rate present")

    if (df["I_upper"] < 0).any():
        failures.append("Negative I_upper present")

    eff = df["bits_per_joule_upper"].to_numpy()
    if np.allclose(eff, eff[0]):
        failures.append("bits_per_joule_upper is flat (no optimum detected)")
    else:
        peak_idx = int(np.nanargmax(eff))
        if peak_idx in (0, len(eff) - 1):
            failures.append("Efficiency peak occurs at boundary of sweep (likely insufficient sweep range)")

        edge_max = max(eff[0], eff[-1])
        if edge_max > 0 and np.nanmax(eff) < 1.05 * edge_max:
            failures.append("Efficiency peak not clearly above edges (weak optimum)")

    try:
        corr = np.corrcoef(df["mean_rate"].to_numpy(), df["I_upper"].to_numpy())[0, 1]
        if np.isnan(corr) or corr < 0.3:
            failures.append(f"Weak/negative correlation between mean_rate and I_upper (corr={corr:.2f})")
    except Exception:
        failures.append("Failed to compute correlation(mean_rate, I_upper)")

    return failures


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None, help="Explicit run id (e.g., v1_n).")
    args = ap.parse_args()

    run_dir, sweep_csv, note = _resolve_run_dir(args.run_id)

    df = pd.read_csv(sweep_csv)
    df = _normalize_columns(df)

    failures = _gates(df)

    peak_row = df.loc[df["bits_per_joule_upper"].idxmax()]

    lines = []
    lines.append("=== Paper B / Stage-0 Report ===")
    lines.append(f"Run dir: {run_dir}")
    if note:
        lines.append(note)
    lines.append(f"Sweep table: {sweep_csv}")
    lines.append("")

    lines.append("Peak efficiency (upper bound):")
    lines.append(
        f"  theta0={peak_row['theta0']:.3f}, tau_ms={peak_row['tau_ms']:.1f}, mean_rate={peak_row['mean_rate']:.3f} Hz, "
        f"I_upper={peak_row['I_upper']:.3f} bits/s, bits/spike={peak_row['bits_per_joule_upper']:.3f}"
    )
    lines.append("")

    if failures:
        lines.append("GATES: FAIL")
        for f in failures:
            lines.append(f"  - {f}")
    else:
        lines.append("GATES: PASS")

    report = "\n".join(lines) + "\n"

    out_path = run_dir / "logs" / "report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(report)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
