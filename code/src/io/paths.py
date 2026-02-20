import string
from pathlib import Path
from typing import Optional, List


def get_next_run_id(runs_dir: Path, major_version: int) -> str:
    """Return the next available run ID for a given major version.

    Run IDs are versioned directories under <root>/runs.

    Format:
        v{major} -> v{major}_a -> v{major}_b -> ... -> v{major}_z

    The first run for a major version uses no suffix (e.g. v1).
    Subsequent runs increment a single-letter suffix.
    """
    base_id = f"v{major_version}"
    if not (runs_dir / base_id).exists():
        return base_id

    for suffix in string.ascii_lowercase:
        run_id = f"{base_id}_{suffix}"
        if not (runs_dir / run_id).exists():
            return run_id

    raise RuntimeError(
        f"Too many runs for major version {major_version}. "
        "Please increment the major version."
    )


def create_run_dir(major_version: int, root: Optional[Path] = None) -> Path:
    """Create a new run directory and update <root>/runs/LATEST.

    IMPORTANT: LATEST is written as a *portable run_id* (e.g. 'v2_b'),
    not an absolute path. Older packages may contain absolute paths; the
    reader function keeps backward compatibility.
    """
    root = Path(root) if root is not None else Path.cwd()
    runs_root = root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    run_id = get_next_run_id(runs_root, major_version)
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Standard subfolders
    for sub in ("figures", "tables", "logs"):
        (run_dir / sub).mkdir(exist_ok=True)

    # Update LATEST pointer (portable)
    (runs_root / "LATEST").write_text(run_id + "\n")

    return run_dir


def _list_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    dirs = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    # Stable fallback: newest by mtime
    return sorted(dirs, key=lambda p: p.stat().st_mtime)


def get_latest_run_dir(root: Optional[Path] = None) -> Path:
    """Resolve the latest run directory.

    Resolution order:
      1) If <root>/runs/LATEST exists:
         - If it contains a run_id like 'v2_b', return <root>/runs/<run_id> if it exists.
         - If it contains an absolute path (older packages) and it exists, return it.
      2) Otherwise, fall back to the most recently modified run directory under <root>/runs.

    Raises FileNotFoundError if no run directories exist.
    """
    root = Path(root) if root is not None else Path.cwd()
    runs_root = root / "runs"

    latest_file = runs_root / "LATEST"
    if latest_file.exists():
        token = latest_file.read_text().strip()
        if token:
            # Preferred portable token: run_id
            candidate = runs_root / token
            if candidate.exists():
                return candidate

            # Backward compatibility: absolute path written by older code
            abs_candidate = Path(token)
            if abs_candidate.is_absolute() and abs_candidate.exists():
                return abs_candidate

    run_dirs = _list_run_dirs(runs_root)
    if run_dirs:
        return run_dirs[-1]

    raise FileNotFoundError(
        f"No run directories found under {runs_root}. "
        "Run a sweep first (e.g., scripts/b0_sanity_sweep_theta0.py)."
    )
